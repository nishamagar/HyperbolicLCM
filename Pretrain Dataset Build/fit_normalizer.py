import os
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def _as_fixed_list_array(col) -> pa.Array:
    if isinstance(col, pa.ChunkedArray):
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
        arr = _as_fixed_list_array(col)  

        t = arr.type
        if not pa.types.is_fixed_size_list(t):
            raise TypeError(f"Expected FixedSizeList for 'x', got: {t}")

        flat = arr.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

        flat = flat[: take * flat_len]

        x = torch.from_numpy(flat.reshape((take, seq_len, embed_dim)))  
        xb = x.reshape(-1, embed_dim).double()  

        sum1 += xb.sum(dim=0)
        sum2 += (xb * xb).sum(dim=0)
        count += xb.size(0)
        seen_rows += take

        last_x_for_diag = x 

    mu = (sum1 / count).float()
    var = (sum2 / count - mu.double() * mu.double()).clamp_min(1e-12)
    sigma = var.sqrt().float().clamp_min(1e-6)

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



