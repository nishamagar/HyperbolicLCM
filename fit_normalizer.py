# # fit_normalizer.py
# import os
# import math
# import torch
# from tqdm import tqdm

# from sequence_dataset import SequenceParquetDataset


# def fit_normalizer(
#     sequences_path: str = "wikitext_train_sequences.parquet",
#     out_path: str = "normalizer.pt",
#     max_sequences: int = 20000,    # sequences to scan
#     batch_sequences: int = 128,    # how many sequences to process per step
#     debug_schema: bool = True,
# ):
#     """
#     Fits per-dimension mean and std over concept vectors in x.

#     - Streams over parquet via SequenceParquetDataset (RAM-safe)
#     - Accumulates in float64 for stability
#     - Saves mu, sigma, and simple diagnostics
#     """

#     if not os.path.exists(sequences_path):
#         raise FileNotFoundError(f"Missing {sequences_path} (run build_wikitext_seq.py first)")

#     ds = SequenceParquetDataset(sequences_path, debug=debug_schema)
#     total_seqs = len(ds)
#     n = min(total_seqs, int(max_sequences))
#     if n <= 0:
#         raise RuntimeError("Dataset empty or max_sequences=0")

#     print(f"[fit_normalizer] dataset size = {total_seqs:,} sequences")
#     print(f"[fit_normalizer] scanning first {n:,} sequences")
#     print(f"[fit_normalizer] batch_sequences={batch_sequences}")

#     sum1 = None  # float64
#     sum2 = None  # float64
#     count = 0

#     # sample norms for diagnostics (collect a small sample only)
#     sample_norms = []

#     # stream in mini-batches (still single-threaded, but far less Python overhead)
#     for start in tqdm(range(0, n, batch_sequences), desc="Streaming mean/std"):
#         end = min(start + batch_sequences, n)

#         xs = []
#         for i in range(start, end):
#             x, _ = ds[i]            # x: [T, D]
#             xs.append(x)

#         x_batch = torch.stack(xs, dim=0).float()   # [B, T, D]
#         xb = x_batch.reshape(-1, x_batch.shape[-1]).double()  # [B*T, D]

#         if sum1 is None:
#             D = xb.shape[1]
#             sum1 = torch.zeros(D, dtype=torch.float64)
#             sum2 = torch.zeros(D, dtype=torch.float64)

#         sum1 += xb.sum(dim=0)
#         sum2 += (xb * xb).sum(dim=0)
#         count += xb.shape[0]

#         # store a small diagnostic sample
#         if len(sample_norms) < 2000:
#             # take up to 256 norms from this batch
#             take = min(256, xb.shape[0])
#             sample_norms.append(xb[:take].float().norm(dim=1))

#     mu = (sum1 / count).float()
#     var = (sum2 / count - mu.double() * mu.double()).clamp_min(1e-12)
#     sigma = var.sqrt().float().clamp_min(1e-6)

#     # diagnostics
#     if sample_norms:
#         samp = torch.cat(sample_norms, dim=0)
#         normed = ((samp.unsqueeze(1) - mu[:1]) / sigma[:1]).norm(dim=1) if samp.dim() == 1 else samp
#         # Better: compute normalized norms using a small tensor sample from real vectors:
#         # We'll redo properly from one tiny pass:
#         pass

#     # Proper normalized-norm diagnostics on a tiny sample batch
#     diag_n = min(512, n)
#     diag_xs = []
#     for i in range(diag_n):
#         x, _ = ds[i]
#         diag_xs.append(x)
#     diag = torch.stack(diag_xs, dim=0).float().reshape(-1, diag_xs[0].shape[-1])
#     normed_diag = ((diag - mu) / sigma).norm(dim=1)
#     avg_norm = normed_diag.mean().item()
#     p95_norm = normed_diag.quantile(0.95).item()

#     print(f"[fit_normalizer] vectors counted: {count:,}")
#     print(f"[fit_normalizer] avg normalized vector norm ≈ {avg_norm:.3f}")
#     print(f"[fit_normalizer] p95 normalized vector norm ≈ {p95_norm:.3f}")

#     torch.save(
#         {
#             "mu": mu.cpu(),
#             "sigma": sigma.cpu(),
#             "avg_norm": avg_norm,
#             "p95_norm": p95_norm,
#             "count": int(count),
#             "scanned_sequences": int(n),
#         },
#         out_path,
#     )
#     print(f"[fit_normalizer] saved -> {out_path}")


# if __name__ == "__main__":
#     fit_normalizer()


# fit_normalizer.py (FAST for FixedSizeList sequences; pyarrow-old safe)
import os
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def _as_fixed_list_array(col) -> pa.Array:
    """
    col can be:
      - pa.ChunkedArray (common)
      - pa.Array / pa.FixedSizeListArray (also possible)
    Returns a *single* pa.Array with contiguous storage if possible.
    """
    if isinstance(col, pa.ChunkedArray):
        # Combine chunks into one Array
        return col.combine_chunks()
    return col


def fit_normalizer(
    sequences_path: str = "wikitext_train_sequences.parquet",
    out_path: str = "normalizer.pt",
    seq_len: int = 8,
    embed_dim: int = 768,
    max_sequences: int = 20000,
):
    if not os.path.exists(sequences_path):
        raise FileNotFoundError(sequences_path)

    pf = pq.ParquetFile(sequences_path)
    total_rows = int(pf.metadata.num_rows)
    n = min(int(max_sequences), total_rows)
    if n <= 0:
        raise RuntimeError("empty dataset")

    flat_len = int(seq_len) * int(embed_dim)
    print(f"[fit_normalizer] rows={total_rows:,} using={n:,} flat_len={flat_len}")

    sum1 = torch.zeros(embed_dim, dtype=torch.float64)
    sum2 = torch.zeros(embed_dim, dtype=torch.float64)
    count = 0

    seen_rows = 0
    last_x_for_diag = None  # small diag sample

    for rg in tqdm(range(pf.num_row_groups), desc="Row-groups"):
        if seen_rows >= n:
            break

        tbl = pf.read_row_group(rg, columns=["x"])
        rows = int(tbl.num_rows)
        take = min(rows, n - seen_rows)
        if take <= 0:
            continue

        col = tbl.column("x")
        arr = _as_fixed_list_array(col)  # FixedSizeListArray most of the time

        # Sanity: must be FixedSizeList
        t = arr.type
        if not pa.types.is_fixed_size_list(t):
            raise TypeError(f"Expected FixedSizeList for 'x', got: {t}")

        # arr.values is the underlying float32 array of length rows * flat_len
        flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

        # IMPORTANT: only use first 'take' rows from this row group
        flat = flat[: take * flat_len]

        # reshape to [take, seq_len, embed_dim]
        x = torch.from_numpy(flat.reshape((take, seq_len, embed_dim)))  # float32 CPU
        xb = x.reshape(-1, embed_dim).double()  # [take*seq_len, D]

        sum1 += xb.sum(dim=0)
        sum2 += (xb * xb).sum(dim=0)
        count += xb.size(0)
        seen_rows += take

        last_x_for_diag = x  # keep last chunk for diagnostics

    mu = (sum1 / count).float()
    var = (sum2 / count - mu.double() * mu.double()).clamp_min(1e-12)
    sigma = var.sqrt().float().clamp_min(1e-6)

    # quick diagnostics (optional)
    if last_x_for_diag is not None:
        with torch.no_grad():
            xb = last_x_for_diag.reshape(-1, embed_dim)
            norms = ((xb - mu) / sigma).norm(dim=1)
            avg_norm = float(norms.mean().item())
            p95_norm = float(norms.quantile(0.95).item())
    else:
        avg_norm, p95_norm = float("nan"), float("nan")

    torch.save(
        {"mu": mu.cpu(), "sigma": sigma.cpu(), "count": int(count),
         "avg_norm": avg_norm, "p95_norm": p95_norm},
        out_path
    )
    print(f"[fit_normalizer] vectors={count:,} avg_norm={avg_norm:.3f} p95_norm={p95_norm:.3f}")
    print(f"[fit_normalizer] saved -> {out_path}")


if __name__ == "__main__":
    fit_normalizer()



