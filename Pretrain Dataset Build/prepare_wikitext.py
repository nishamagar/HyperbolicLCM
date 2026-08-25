import os
from typing import List

import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch import amp

MODEL_NAME = "microsoft/deberta-v3-small"
DOC_OUT_DIR = "wikitext_docs"

CHUNK_TOK_LEN = 256
BATCH_SIZE = 64

TOKEN_BUDGET = 70_000_000

SHARD_MAX_ROWS = 200_000
SHARD_PREFIX = "shard"
ROW_GROUP_SIZE = 50_000

DOC_CHUNKS = 128  
MAX_PENDING_CHUNKS_IN_RAM = 4096  


def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def load_encoder(device: torch.device):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    enc = AutoModel.from_pretrained(MODEL_NAME).to(device)
    enc.eval()
    return tok, enc


@torch.inference_mode()
def embed_chunk_texts(chunk_texts: List[str], tokenizer, encoder, device: torch.device):
    inputs = tokenizer(
        chunk_texts,
        padding=True,
        truncation=True,
        max_length=CHUNK_TOK_LEN,
        return_tensors="pt",
    )
    token_counts = inputs["attention_mask"].sum(dim=1).cpu().to(torch.int32)
    inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

    use_amp = (device.type == "cuda")
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    with amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
        out = encoder(**inputs).last_hidden_state[:, 0, :]  # CLS [N,D]

    return out.detach().cpu().to(torch.float32), token_counts


class ShardWriter:
    def __init__(self, out_dir: str, prefix: str, embed_dim: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.embed_dim = int(embed_dim)

        self.shard_id = 0
        self.writer = None
        self.writer_path = None
        self.rows_in_shard = 0

        self.buf_doc = []
        self.buf_pos = []
        self.buf_vec = []

    def _cur_path(self):
        os.makedirs(self.out_dir, exist_ok=True)
        return os.path.join(self.out_dir, f"{self.prefix}_{self.shard_id:06d}.parquet")

    def _flush_buf_to_file(self):
        if not self.buf_doc:
            return

        doc = np.concatenate(self.buf_doc, axis=0)
        pos = np.concatenate(self.buf_pos, axis=0)
        vec = np.concatenate(self.buf_vec, axis=0).astype(np.float32, copy=False)  # [N,D]

        arr_doc = pa.array(doc, type=pa.int64())
        arr_pos = pa.array(pos, type=pa.int64())

        flat = pa.array(vec.reshape(-1), type=pa.float32())
        arr_vec = pa.FixedSizeListArray.from_arrays(flat, self.embed_dim)

        table = pa.Table.from_arrays(
            [arr_doc, arr_pos, arr_vec],
            names=["doc_id", "position", "concept_vector"],
        )

        if self.writer is None:
            self.writer_path = self._cur_path()
            self.writer = pq.ParquetWriter(
                self.writer_path,
                table.schema,
                compression="zstd",
                use_dictionary=False,
                write_statistics=False,
            )

        self.writer.write_table(table, row_group_size=ROW_GROUP_SIZE)
        self.rows_in_shard += table.num_rows

        self.buf_doc.clear()
        self.buf_pos.clear()
        self.buf_vec.clear()

    def add_doc_batch(self, doc_id: int, start_pos: int, embs_cpu: torch.Tensor):
        n = int(embs_cpu.size(0))
        if n <= 0:
            return
        vec = embs_cpu.numpy()
        doc = np.full((n,), int(doc_id), dtype=np.int64)
        pos = np.arange(start_pos, start_pos + n, dtype=np.int64)

        self.buf_doc.append(doc)
        self.buf_pos.append(pos)
        self.buf_vec.append(vec)

    def finish_doc_boundary(self):
        self._flush_buf_to_file()
        if self.rows_in_shard >= SHARD_MAX_ROWS:
            self.close_shard()

    def close_shard(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.writer_path = None
            self.shard_id += 1
        self.rows_in_shard = 0

    def close_all(self):
        self._flush_buf_to_file()
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def preprocess_wikitext():
    device = pick_device()
    print("[device]", device, "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))

    print("Loading dataset: wikitext-103-raw-v1 (train)")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

    print(f"Loading encoder+tokenizer: {MODEL_NAME} on {device} ...")
    tok, enc = load_encoder(device)
    embed_dim = int(enc.config.hidden_size)
    print("[encoder] hidden_size =", embed_dim)
    print("[pack] chunk_tokens =", CHUNK_TOK_LEN)
    print("[pack] DOC_CHUNKS =", DOC_CHUNKS, "(smaller => more docs)")

    writer = ShardWriter(DOC_OUT_DIR, SHARD_PREFIX, embed_dim)

    total_tokens = 0
    reached_budget = False
    doc_id = 0

    cur_ids: List[int] = []
    cur_chunks: List[str] = []

    def flush_doc():
        nonlocal doc_id, cur_ids, cur_chunks, total_tokens, reached_budget
        if reached_budget:
            cur_ids = []
            cur_chunks = []
            return
        if len(cur_chunks) < 1:
            cur_ids = []
            cur_chunks = []
            return

        pos = 0
        for i in range(0, len(cur_chunks), BATCH_SIZE):
            if reached_budget:
                break

            batch_texts = cur_chunks[i:i + BATCH_SIZE]
            embs_cpu, counts_cpu = embed_chunk_texts(batch_texts, tok, enc, device)

            counts = counts_cpu.numpy().astype(np.int64, copy=False)
            keep = len(counts)

            if total_tokens + int(counts.sum()) > TOKEN_BUDGET:
                running = 0
                keep = 0
                for c in counts:
                    c = int(c)
                    if total_tokens + running + c > TOKEN_BUDGET:
                        break
                    running += c
                    keep += 1
                reached_budget = True

            if keep <= 0:
                break

            embs_keep = embs_cpu[:keep]
            counts_keep = counts[:keep]
            total_tokens += int(counts_keep.sum())

            writer.add_doc_batch(doc_id, pos, embs_keep)
            pos += keep

        if pos >= 1:
            doc_id += 1

        cur_ids = []
        cur_chunks = []
        writer.finish_doc_boundary()

    def add_text(text: str):
        nonlocal cur_ids, cur_chunks
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if not ids:
            return
        cur_ids.extend(ids)

        while len(cur_ids) >= CHUNK_TOK_LEN:
            chunk_ids = cur_ids[:CHUNK_TOK_LEN]
            cur_ids = cur_ids[CHUNK_TOK_LEN:]
            chunk_text = tok.decode(chunk_ids, clean_up_tokenization_spaces=True)
            cur_chunks.append(chunk_text)

    print("Token packing -> chunks -> embeddings -> shard parquet...")

    pbar = tqdm(ds, desc="Embedding WikiText-103", mininterval=1.0)
    for ex in pbar:
        if reached_budget:
            break

        text = ex["text"].strip()
        if not text:
            continue

        add_text(text)

        if len(cur_chunks) >= DOC_CHUNKS:
            flush_doc()

        if len(cur_chunks) >= MAX_PENDING_CHUNKS_IN_RAM:
            flush_doc()

        pbar.set_postfix(tokens=total_tokens, docs=doc_id, shard=writer.shard_id, chunks=len(cur_chunks))

    if not reached_budget and len(cur_chunks) > 0:
        flush_doc()

    writer.close_all()
    print(f"Done. Wrote shards into: {DOC_OUT_DIR}/")
    print(f"Total tokens included (post-tokenizer): {total_tokens:,} / {TOKEN_BUDGET:,}")
    print(f"Final doc_id={doc_id}, shard_id={writer.shard_id}")


if __name__ == "__main__":
    preprocess_wikitext()
