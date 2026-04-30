import os, glob, random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

DOC_DIR = "wikitext_docs"
OUT_TRAIN = "wikitext_train_sequences.parquet"
OUT_VAL   = "wikitext_val_sequences.parquet"

CHUNK_TOK_LEN = 256

SEQ_LEN = 8

STRIDE  = 2 # <-- was 2

VAL_DOC_FRAC = 0.30
SEED = 42

MAX_TRAIN_WINDOWS_PER_DOC = 200   # try 50–200; smaller => more docs, less repetition
MAX_VAL_WINDOWS_PER_DOC   = 50    # smaller => val uses more docs

WRITE_CHUNK_ROWS = 4096


def list_shards() -> List[str]:
    shards = sorted(glob.glob(os.path.join(DOC_DIR, "shard_*.parquet")))
    if not shards:
        raise FileNotFoundError(f"No {DOC_DIR}/shard_*.parquet found")
    return shards


def get_embed_dim_from_first_shard(shards: List[str]) -> int:
    pf = pq.ParquetFile(shards[0])
    t = pf.schema_arrow.field("concept_vector").type
    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return 768
    raise ValueError(f"Unexpected concept_vector type: {t}")


def scan_doc_ids(shards: List[str]) -> List[int]:
    doc_ids = set()
    for path in tqdm(shards, desc="Scan shards for doc_ids"):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=["doc_id"]).column("doc_id")
            doc_ids.update(col.to_numpy(zero_copy_only=False).astype(np.int64))
    return sorted(int(x) for x in doc_ids)


def scan_doc_lengths(shards: List[str]) -> Dict[int, int]:
    lengths: Dict[int, int] = {}
    cur_doc = None
    cur_len = 0

    for path in tqdm(shards, desc="Reading shards (doc lengths)"):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            doc_col = pf.read_row_group(rg, columns=["doc_id"]).column("doc_id") \
                        .to_numpy(zero_copy_only=False).astype(np.int64)

            for d in doc_col:
                d = int(d)
                if cur_doc is None:
                    cur_doc = d
                    cur_len = 1
                elif d == cur_doc:
                    cur_len += 1
                else:
                    lengths[cur_doc] = lengths.get(cur_doc, 0) + cur_len
                    cur_doc = d
                    cur_len = 1

    if cur_doc is not None:
        lengths[cur_doc] = lengths.get(cur_doc, 0) + cur_len

    return lengths


def build_windows_for_docs(
    doc_lengths: Dict[int, int],
    allowed_docs: set,
    stride: int,
    max_windows_per_doc: Optional[int],
) -> List[Tuple[int, int]]:
    """
    Build (doc_id, start) windows where each window uses (SEQ_LEN+1) concept vectors:
      x = positions [start .. start+SEQ_LEN-1]
      y = positions [start+1 .. start+SEQ_LEN]
    """
    windows: List[Tuple[int, int]] = []
    need = SEQ_LEN + 1

    # Deterministic order by doc_id helps reproducibility
    for doc_id in sorted(doc_lengths.keys()):
        if doc_id not in allowed_docs:
            continue
        L = int(doc_lengths[doc_id])
        if L < need:
            continue

        per_doc = 0
        for start in range(0, L - need + 1, stride):
            windows.append((doc_id, start))
            per_doc += 1
            if max_windows_per_doc is not None and per_doc >= int(max_windows_per_doc):
                break

    return windows


def load_docs_needed(shards: List[str], needed: set, embed_dim: int) -> Dict[int, np.ndarray]:
    out: Dict[int, List[np.ndarray]] = {}
    cur_doc = None
    cur_vecs: List[np.ndarray] = []

    def flush(doc_id, vec_list):
        if doc_id is None:
            return
        if doc_id in needed and vec_list:
            out[doc_id] = np.concatenate(vec_list, axis=0)

    for path in tqdm(shards, desc="Reading shards (vectors)"):
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["doc_id", "concept_vector"])
            doc_col = tbl.column("doc_id").to_numpy(zero_copy_only=False).astype(np.int64)
            vec_col = tbl.column("concept_vector")

            if isinstance(vec_col, pa.ChunkedArray):
                vec_col = vec_col.combine_chunks()

            if pa.types.is_fixed_size_list(vec_col.type):
                flat = vec_col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
                vec_np = flat.reshape((-1, embed_dim))
            else:
                vec_py = vec_col.to_pylist()
                vec_np = np.asarray(vec_py, dtype=np.float32).reshape((-1, embed_dim))

            for d, v in zip(doc_col, vec_np):
                d = int(d)
                if cur_doc is None:
                    cur_doc = d
                    cur_vecs = [v.reshape(1, -1)]
                elif d == cur_doc:
                    cur_vecs.append(v.reshape(1, -1))
                else:
                    flush(cur_doc, cur_vecs)
                    cur_doc = d
                    cur_vecs = [v.reshape(1, -1)]

    flush(cur_doc, cur_vecs)
    return out


def write_sequences(
    out_path: str,
    windows: List[Tuple[int, int]],
    docs: Dict[int, np.ndarray],
    embed_dim: int,
) -> int:
    if os.path.exists(out_path):
        os.remove(out_path)

    flat_len = SEQ_LEN * embed_dim
    writer = None
    n = 0

    i = 0
    while i < len(windows):
        j = min(i + WRITE_CHUNK_ROWS, len(windows))
        chunk = windows[i:j]

        x_flat = np.empty((len(chunk), flat_len), dtype=np.float32)
        y_flat = np.empty((len(chunk), flat_len), dtype=np.float32)

        for k, (doc_id, start) in enumerate(chunk):
            vecs = docs[doc_id]  # [L, D]
            w = vecs[start:start + SEQ_LEN + 1]  # [SEQ_LEN+1, D]
            x_flat[k] = w[:-1].reshape(-1)
            y_flat[k] = w[1:].reshape(-1)

        ax = pa.FixedSizeListArray.from_arrays(pa.array(x_flat.reshape(-1), type=pa.float32()), flat_len)
        ay = pa.FixedSizeListArray.from_arrays(pa.array(y_flat.reshape(-1), type=pa.float32()), flat_len)
        table = pa.Table.from_arrays([ax, ay], names=["x", "y"])

        if writer is None:
            writer = pq.ParquetWriter(
                out_path,
                table.schema,
                compression="zstd",
                use_dictionary=False,
                write_statistics=False,
            )

        writer.write_table(table)
        n += table.num_rows
        i = j

    if writer is not None:
        writer.close()
    return n


def build_sequences():
    shards = list_shards()
    embed_dim = get_embed_dim_from_first_shard(shards)
    print("[info] embed_dim =", embed_dim)

    print("Scanning doc_ids for train/val split...")
    doc_ids = scan_doc_ids(shards)
    rng = random.Random(SEED)
    rng.shuffle(doc_ids)

    n_val_docs = max(1, int(len(doc_ids) * VAL_DOC_FRAC))
    val_set = set(doc_ids[:n_val_docs])
    train_set = set(doc_ids[n_val_docs:])

    print(f"Docs total: {len(doc_ids)} | train docs: {len(train_set)} | val docs: {len(val_set)}")
    print(f"CHUNK_TOK_LEN={CHUNK_TOK_LEN} | SEQ_LEN={SEQ_LEN} | STRIDE={STRIDE}")
    print(f"MAX_TRAIN_WINDOWS_PER_DOC={MAX_TRAIN_WINDOWS_PER_DOC} | MAX_VAL_WINDOWS_PER_DOC={MAX_VAL_WINDOWS_PER_DOC}")

    print("Pass 1: scanning doc lengths...")
    doc_lengths = scan_doc_lengths(shards)

    need = SEQ_LEN + 1
    n_long_train = sum(1 for d, L in doc_lengths.items() if d in train_set and L >= need)
    n_long_val   = sum(1 for d, L in doc_lengths.items() if d in val_set and L >= need)
    Ls = np.array(list(doc_lengths.values()), dtype=np.int64)
    print(f"[sanity] need L>={need} concepts per doc for 1 window")
    print(f"[sanity] doc_len concepts: min={int(Ls.min())} p50={int(np.median(Ls))} p90={int(np.quantile(Ls,0.9))} max={int(Ls.max())}")
    print(f"[sanity] train docs with L>={need}: {n_long_train} | val docs with L>={need}: {n_long_val}")

    print("Pass 2: building window indexes (NO budget cap in builder)...")
    train_windows = build_windows_for_docs(
        doc_lengths, train_set, stride=STRIDE, max_windows_per_doc=MAX_TRAIN_WINDOWS_PER_DOC
    )
    val_windows = build_windows_for_docs(
        doc_lengths, val_set, stride=STRIDE, max_windows_per_doc=MAX_VAL_WINDOWS_PER_DOC
    )

    train_docs_needed = set(d for d, _ in train_windows)
    val_docs_needed   = set(d for d, _ in val_windows)

    print(f"[train] windows={len(train_windows):,} docs_needed={len(train_docs_needed):,}")
    print(f"[val]   windows={len(val_windows):,} docs_needed={len(val_docs_needed):,}")

    print("Loading train docs (only needed ones)...")
    train_docs = load_docs_needed(shards, train_docs_needed, embed_dim)

    print("Writing train sequences...")
    n_train = write_sequences(OUT_TRAIN, train_windows, train_docs, embed_dim)
    print(f"[train] wrote {n_train:,} sequences -> {OUT_TRAIN}")

    print("Loading val docs (only needed ones)...")
    val_docs = load_docs_needed(shards, val_docs_needed, embed_dim)

    print("Writing val sequences...")
    n_val = write_sequences(OUT_VAL, val_windows, val_docs, embed_dim)
    print(f"[val] wrote {n_val:,} sequences -> {OUT_VAL}")

    print("Done.")


if __name__ == "__main__":
    build_sequences()
