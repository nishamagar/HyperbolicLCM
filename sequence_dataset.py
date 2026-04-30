import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class SequenceParquetDataset(Dataset):
    """
    Expects parquet schema:
      x: FixedSizeList(float32)[SEQ_LEN*D]
      y: FixedSizeList(float32)[SEQ_LEN*D]

    Returns:
      x: torch.float32 [SEQ_LEN, D]
      y: torch.float32 [SEQ_LEN, D]
    """

    def __init__(
        self,
        parquet_path: str,
        seq_len: int,
        embed_dim: int,
        max_rows=None,
        columns=("x", "y"),
        debug_schema: bool = False,
    ):
        self.parquet_path = parquet_path
        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)
        self.flat_len = self.seq_len * self.embed_dim
        self.columns = list(columns)

        self.pf = pq.ParquetFile(parquet_path)

        schema = self.pf.schema_arrow
        if debug_schema:
            print("[SequenceParquetDataset] schema:", schema)

        # ---- schema checks ----
        names = set(schema.names)
        for c in self.columns:
            if c not in names:
                raise ValueError(f"Column '{c}' not found. Available: {schema.names}")

        for c in self.columns:
            t = schema.field(c).type
            if not pa.types.is_fixed_size_list(t):
                raise ValueError(f"Column '{c}' must be FixedSizeList, got: {t}")
            if int(t.list_size) != self.flat_len:
                raise ValueError(
                    f"Column '{c}' list_size={int(t.list_size)} but expected {self.flat_len} "
                    f"(seq_len={self.seq_len} * embed_dim={self.embed_dim})."
                )

        total_rows = int(self.pf.metadata.num_rows)
        self._len = min(int(max_rows), total_rows) if max_rows is not None else total_rows

        # Row-group offsets
        self.rg_counts = [int(self.pf.metadata.row_group(i).num_rows) for i in range(self.pf.num_row_groups)]
        self.rg_offsets = []
        s = 0
        for n in self.rg_counts:
            self.rg_offsets.append(s)
            s += n

        # Cache one row-group
        self._cache_rg = None
        self._cache_x = None  # np.float32 [N, flat_len]
        self._cache_y = None  # np.float32 [N, flat_len]

    def __len__(self):
        return self._len

    def _find_rg(self, idx: int):
        lo, hi = 0, len(self.rg_offsets) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.rg_offsets[mid]
            end = start + self.rg_counts[mid]
            if start <= idx < end:
                return mid, idx - start
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def _fixed_list_to_2d(self, col, nrows: int) -> np.ndarray:
        # ChunkedArray -> Array
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()

        # FixedSizeListArray values are flat float array length nrows*flat_len
        flat = col.values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

        # Make contiguous explicitly (safe for torch.from_numpy zero-copy)
        out = np.ascontiguousarray(flat.reshape((nrows, self.flat_len)), dtype=np.float32)
        return out

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        rg, inner = self._find_rg(idx)

        if self._cache_rg != rg:
            tbl = self.pf.read_row_group(rg, columns=self.columns)
            nrows = int(tbl.num_rows)
            self._cache_x = self._fixed_list_to_2d(tbl.column("x"), nrows)
            self._cache_y = self._fixed_list_to_2d(tbl.column("y"), nrows)
            self._cache_rg = rg

        x = self._cache_x[inner].reshape((self.seq_len, self.embed_dim))
        y = self._cache_y[inner].reshape((self.seq_len, self.embed_dim))
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
