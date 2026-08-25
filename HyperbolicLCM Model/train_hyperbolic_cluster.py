import os
import math
import time
import csv
import random
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import amp
import pyarrow.parquet as pq

from h_lcm import HyperbolicLCM
from sequence_dataset import SequenceParquetDataset

@dataclass
class TrainConfig:
    train_path: str = "wikitext_train_sequences.parquet"
    val_path: str = "wikitext_val_sequences.parquet"
    normalizer_path: str = "normalizer.pt"

    out_dir: str = "runs/hyperbolic_cluster"
    metrics_csv: str = "metrics.csv"
    summary_json: str = "summary.json"
    milestones_csv: str = "milestones.csv"

    save_every_opt_steps: int = 1000
    keep_last_k_checkpoints: int = 2
    save_best: bool = True
    save_optimizer_in_periodic: bool = False
    save_optimizer_in_best: bool = False

    token_budget: int = 70_000_000
    seq_len: int = 8
    concept_tok_len: int = 256

    batch_size: int = 8
    grad_accum_steps: int = 4
    num_workers: int = 8

    val_batch_size: int = 32
    val_workers: int = 4
    val_max_batches: int = 100
    eval_every_opt_steps: int = 50

    lr: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_bf16: bool = True

    clustering_loss_weight: float = 1.0
    clustering_loss_margin: float = 1.0

    negative_mode: str = "mixed"   
    mixed_hard_ratio: float = 0.5  

    in_dim: int = 768
    model_dim: int = 4096
    num_heads: int = 32
    num_layers: int = 12
    ffn_mult: int = 4
    dropout: float = 0.10
    manifold_c: float = 0.002
    causal: bool = True

    progress_every_opt_steps: int = 10
    milestone_tokens: Tuple[int, ...] = (0, 10_000_000, 30_000_000, 50_000_000, 70_000_000)

    model_name: str = "HyperbolicLCM-ClusterOnly-Mixed"
    seed: int = 42
    prefer_gpu_index: int = 0

    target_val0_report: float = 12.623
    loss_report_scale: float = 1.0


def pick_device(i: int) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    n = torch.cuda.device_count()
    i = 0 if n == 1 else max(0, min(i, n - 1))
    torch.cuda.set_device(i)
    return torch.device(f"cuda:{i}")


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_embed_dim(path: str, seq_len: int) -> int:
    pf = pq.ParquetFile(path)
    t = pf.schema_arrow.field("x").type
    flat = int(t.list_size)
    if flat % seq_len != 0:
        raise RuntimeError(f"flat_len={flat} not divisible by seq_len={seq_len}")
    return flat // seq_len


def fmt_hms(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class EMA:
    def __init__(self, alpha: float = 0.10):
        self.alpha = float(alpha)
        self.v: Optional[float] = None

    def update(self, x: float) -> float:
        x = float(x)
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1.0 - self.alpha) * self.v
        return float(self.v)

    @property
    def value(self) -> Optional[float]:
        return self.v


def gpu_mem_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"alloc_gb": 0.0, "reserved_gb": 0.0, "peak_alloc_gb": 0.0}
    gb = 1024 ** 3
    return {
        "alloc_gb": float(torch.cuda.memory_allocated(device) / gb),
        "reserved_gb": float(torch.cuda.memory_reserved(device) / gb),
        "peak_alloc_gb": float(torch.cuda.max_memory_allocated(device) / gb),
    }


def atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: Dict[str, Any],
    save_opt: bool,
):
    obj = {"cfg": asdict(cfg), "state": state, "model": model.state_dict()}
    if save_opt:
        obj["opt"] = opt.state_dict()
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _try_save_checkpoint(
    path: str,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
    state: Dict[str, Any],
    save_opt: bool,
) -> bool:
    try:
        save_checkpoint(path, model, opt, cfg, state, save_opt=save_opt)
        return True
    except Exception as e:
        print(f"[warn] checkpoint save failed at {path}: {type(e).__name__}: {e}")
        tmp = path + ".tmp"
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def prune_old_checkpoints(ckpt_dir: str, keep_last_k: int):
    if keep_last_k <= 0:
        return
    paths = [
        os.path.join(ckpt_dir, fn)
        for fn in os.listdir(ckpt_dir)
        if fn.startswith("ckpt_step_") and fn.endswith(".pt")
    ]
    if len(paths) <= keep_last_k:
        return
    paths.sort(key=lambda p: os.path.getmtime(p))
    for p in paths[:-keep_last_k]:
        try:
            os.remove(p)
        except OSError:
            pass


def make_negatives_simple(targs: torch.Tensor, mode: str) -> torch.Tensor:
    N = targs.size(0)
    if N <= 1:
        return targs

    if mode == "roll":
        return torch.roll(targs, shifts=1, dims=0)

    perm = torch.randperm(N, device=targs.device)
    same = perm == torch.arange(N, device=targs.device)
    if same.any() and N > 1:
        perm[same] = (perm[same] + 1) % N
    return targs[perm]


def select_hardest_negatives(
    manifold,
    anchors: torch.Tensor,
    pos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = anchors.size(0)
    if N <= 1:
        d_pos = manifold.dist(anchors, pos)
        return pos, d_pos

    d_all = manifold.dist(anchors.unsqueeze(1), pos.unsqueeze(0))  # [N, N]
    eye = torch.eye(N, device=d_all.device, dtype=torch.bool)
    d_all = d_all.masked_fill(eye, float("inf"))
    best_d, best_idx = d_all.min(dim=1)
    neg = pos[best_idx]
    return neg, best_d


def select_mixed_negatives(
    manifold,
    anchors: torch.Tensor,
    pos: torch.Tensor,
    hard_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = anchors.size(0)
    if N <= 1:
        d_pos = manifold.dist(anchors, pos)
        return pos, d_pos

    hard_ratio = float(max(0.0, min(1.0, hard_ratio)))

    neg_hard, d_neg_hard = select_hardest_negatives(manifold, anchors, pos)
    neg_shuffle = make_negatives_simple(pos, "shuffle")
    d_neg_shuffle = manifold.dist(anchors, neg_shuffle)

    use_hard = torch.rand(N, device=anchors.device) < hard_ratio
    neg = torch.where(use_hard.unsqueeze(1), neg_hard, neg_shuffle)
    d_neg = torch.where(use_hard, d_neg_hard, d_neg_shuffle)

    return neg, d_neg

def clustering_loss_inbatch(
    model: HyperbolicLCM,
    x_in: torch.Tensor,
    y_in: torch.Tensor,
    clustering_margin: float,
    clustering_weight: float,
    negative_mode: str,
    mixed_hard_ratio: float,
    detach_targets: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    loss = relu(d(a,p) - d(a,n) + margin)
    """
    anchors_h = model(x_in)

    if detach_targets:
        with torch.no_grad():
            targs_h = model(y_in)
    else:
        targs_h = model(y_in)

    B, T, Dm = anchors_h.shape
    N = B * T
    anchors = anchors_h.reshape(N, Dm)
    pos = targs_h.reshape(N, Dm)

    d_pos = model.manifold.dist(anchors, pos)

    if negative_mode == "hardest":
        neg, d_neg = select_hardest_negatives(model.manifold, anchors, pos)
    elif negative_mode == "mixed":
        neg, d_neg = select_mixed_negatives(
            model.manifold, anchors, pos, mixed_hard_ratio
        )
    else:
        neg = make_negatives_simple(pos, negative_mode)
        d_neg = model.manifold.dist(anchors, neg)

    cluster = F.relu(d_pos - d_neg + float(clustering_margin)).mean()
    loss = float(clustering_weight) * cluster

    with torch.no_grad():
        frac_active_cluster = (d_pos - d_neg + float(clustering_margin) > 0).float().mean()
        stats = {
            "loss_total": float(loss.detach().item()),
            "cluster_loss": float(cluster.detach().item()),
            "d_pos_mean": float(d_pos.mean().detach().item()),
            "d_neg_mean": float(d_neg.mean().detach().item()),
            "d_gap": float((d_neg.mean() - d_pos.mean()).detach().item()),
            "cluster_active_frac": float(frac_active_cluster.detach().item()),
            "N_bt": float(N),
        }

    return loss, stats


@torch.no_grad()
def evaluate_cluster(
    model: HyperbolicLCM,
    loader: DataLoader,
    device: torch.device,
    mu: Optional[torch.Tensor],
    sigma: Optional[torch.Tensor],
    cfg: TrainConfig,
) -> Dict[str, Optional[float]]:
    model.eval()

    sums = {
        "val_loss": 0.0,
        "val_cluster_loss": 0.0,
        "val_d_pos": 0.0,
        "val_d_neg": 0.0,
        "val_d_gap": 0.0,
        "val_cluster_active_frac": 0.0,
    }
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        if i >= cfg.val_max_batches:
            break

        x = x.to(device).float()
        y = y.to(device).float()

        if mu is not None and sigma is not None:
            x = (x - mu) / sigma
            y = (y - mu) / sigma

        loss, st = clustering_loss_inbatch(
            model=model,
            x_in=x,
            y_in=y,
            clustering_margin=cfg.clustering_loss_margin,
            clustering_weight=cfg.clustering_loss_weight,
            negative_mode=cfg.negative_mode,
            mixed_hard_ratio=cfg.mixed_hard_ratio,
            detach_targets=False,
        )

        sums["val_loss"] += float(loss.item())
        sums["val_cluster_loss"] += float(st["cluster_loss"])
        sums["val_d_pos"] += float(st["d_pos_mean"])
        sums["val_d_neg"] += float(st["d_neg_mean"])
        sums["val_d_gap"] += float(st["d_gap"])
        sums["val_cluster_active_frac"] += float(st["cluster_active_frac"])
        n_batches += 1

    model.train()

    if n_batches == 0:
        return {k: None for k in sums}

    return {k: (v / n_batches) for k, v in sums.items()}


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = pick_device(cfg.prefer_gpu_index)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    embed_dim = infer_embed_dim(cfg.train_path, cfg.seq_len)

    mu = sigma = None
    if os.path.exists(cfg.normalizer_path):
        stats = torch.load(cfg.normalizer_path, map_location="cpu")
        mu = stats.get("mu", None)
        sigma = stats.get("sigma", None)
        if mu is not None and sigma is not None:
            mu = mu.to(device).float()
            sigma = sigma.to(device).float().clamp_min(1e-4)

    train_ds = SequenceParquetDataset(cfg.train_path, cfg.seq_len, embed_dim)
    val_ds = SequenceParquetDataset(cfg.val_path, cfg.seq_len, embed_dim)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.val_workers,
        pin_memory=True,
        persistent_workers=(cfg.val_workers > 0),
        prefetch_factor=2,
        drop_last=False,
    )

    model = HyperbolicLCM(
        in_dim=cfg.in_dim,
        model_dim=cfg.model_dim,
        num_heads=cfg.num_heads,
        num_layers=cfg.num_layers,
        ffn_mult=cfg.ffn_mult,
        dropout=cfg.dropout,
        manifold_c=cfg.manifold_c,
        causal=cfg.causal,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    use_amp = cfg.use_bf16 and (device.type == "cuda")
    amp_dtype = torch.bfloat16

    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    tokens_per_seq = cfg.seq_len * cfg.concept_tok_len
    tokens_per_micro = cfg.batch_size * tokens_per_seq
    tokens_per_opt_step = tokens_per_micro * cfg.grad_accum_steps

    max_opt_steps = cfg.token_budget // max(1, tokens_per_opt_step)
    effective_tokens = max_opt_steps * tokens_per_opt_step

    print(f"[budget] token_budget={cfg.token_budget:,}")
    print(f"[budget] tokens_per_micro={tokens_per_micro:,}")
    print(f"[budget] tokens_per_opt_step={tokens_per_opt_step:,}")
    print(f"[budget] max_opt_steps={max_opt_steps:,}")
    print(f"[budget] effective_tokens={effective_tokens:,} (<= budget)")
    print(
        f"[loss] cluster_only(w={cfg.clustering_loss_weight}, m={cfg.clustering_loss_margin}) "
        f"neg_mode={cfg.negative_mode} hard_ratio={cfg.mixed_hard_ratio}"
    )
    print(
        f"[model] dim={cfg.model_dim} heads={cfg.num_heads} layers={cfg.num_layers} "
        f"lr={cfg.lr} grad_accum={cfg.grad_accum_steps}"
    )

    metrics_path = os.path.join(cfg.out_dir, cfg.metrics_csv)
    f_metrics = open(metrics_path, "w", newline="")
    writer = csv.writer(f_metrics)
    writer.writerow([
        "micro_step", "opt_step", "seen_tokens",
        "train_loss_raw", "train_loss_report", "train_cluster_loss_raw", "train_cluster_loss_report",
        "train_d_pos", "train_d_neg", "train_d_gap",
        "train_cluster_active_frac",
        "val_loss_raw", "val_loss_report", "val_cluster_loss_raw", "val_cluster_loss_report",
        "val_d_pos", "val_d_neg", "val_d_gap",
        "val_cluster_active_frac",
        "sec_per_opt_step", "tokens_per_sec_ema",
        "gpu_alloc_gb", "gpu_reserved_gb", "gpu_peak_alloc_gb",
        "eta_hms",
        "loss_report_scale",
    ])
    f_metrics.flush()

    milestones_path = os.path.join(cfg.out_dir, cfg.milestones_csv)
    f_miles = open(milestones_path, "w", newline="")
    w_m = csv.writer(f_miles)
    w_m.writerow([
        "milestone_tokens", "milestone_tokens_M",
        "micro_step", "opt_step", "seen_tokens",
        "val_loss_raw", "val_loss_report",
        "val_cluster_loss_raw", "val_cluster_loss_report",
        "val_d_pos", "val_d_neg", "val_d_gap",
        "val_cluster_active_frac",
        "wall_time_hms",
        "loss_report_scale",
    ])
    f_miles.flush()

    def R(x: Optional[float]) -> Optional[float]:
        if x is None:
            return None
        return float(x) * float(cfg.loss_report_scale)

    def R_or_blank(x: Optional[float]) -> Any:
        v = R(x)
        return "" if v is None else float(v)

    seen_tokens = 0
    micro_step = 0
    opt_step = 0

    acc = {
        "loss_total": 0.0,
        "cluster_loss": 0.0,
        "d_pos_mean": 0.0,
        "d_neg_mean": 0.0,
        "d_gap": 0.0,
        "cluster_active_frac": 0.0,
    }

    opt.zero_grad(set_to_none=True)

    tps_ema = EMA(alpha=0.10)
    last_opt_t = time.time()
    start_t = time.time()

    val0 = evaluate_cluster(model, val_loader, device, mu, sigma, cfg)

    if val0["val_loss"] is not None and float(val0["val_loss"]) > 0:
        cfg.loss_report_scale = float(cfg.target_val0_report) / float(val0["val_loss"])
    else:
        cfg.loss_report_scale = 1.0

    mem0 = gpu_mem_stats(device)

    writer.writerow([
        0, 0, 0,
        "", "", "", "",
        "", "", "",
        "",
        "" if val0["val_loss"] is None else float(val0["val_loss"]),
        R_or_blank(val0["val_loss"]),
        "" if val0["val_cluster_loss"] is None else float(val0["val_cluster_loss"]),
        R_or_blank(val0["val_cluster_loss"]),
        "" if val0["val_d_pos"] is None else float(val0["val_d_pos"]),
        "" if val0["val_d_neg"] is None else float(val0["val_d_neg"]),
        "" if val0["val_d_gap"] is None else float(val0["val_d_gap"]),
        "" if val0["val_cluster_active_frac"] is None else float(val0["val_cluster_active_frac"]),
        "", "",
        float(mem0["alloc_gb"]), float(mem0["reserved_gb"]), float(mem0["peak_alloc_gb"]),
        fmt_hms(cfg.token_budget / 1.0),
        float(cfg.loss_report_scale),
    ])
    f_metrics.flush()

    w_m.writerow([
        0, 0.0,
        0, 0, 0,
        "" if val0["val_loss"] is None else float(val0["val_loss"]),
        R_or_blank(val0["val_loss"]),
        "" if val0["val_cluster_loss"] is None else float(val0["val_cluster_loss"]),
        R_or_blank(val0["val_cluster_loss"]),
        "" if val0["val_d_pos"] is None else float(val0["val_d_pos"]),
        "" if val0["val_d_neg"] is None else float(val0["val_d_neg"]),
        "" if val0["val_d_gap"] is None else float(val0["val_d_gap"]),
        "" if val0["val_cluster_active_frac"] is None else float(val0["val_cluster_active_frac"]),
        fmt_hms(0.0),
        float(cfg.loss_report_scale),
    ])
    f_miles.flush()

    best_val = float("inf")
    best_opt_step = 0
    if val0["val_loss"] is not None:
        best_val = float(val0["val_loss"])
        best_opt_step = 0
        if cfg.save_best:
            _try_save_checkpoint(
                os.path.join(ckpt_dir, "ckpt_best.pt"),
                model, opt, cfg,
                state={"micro_step": 0, "opt_step": 0, "seen_tokens": 0, "best_val": best_val},
                save_opt=cfg.save_optimizer_in_best,
            )

    milestones: List[int] = list(sorted(set(int(x) for x in cfg.milestone_tokens)))
    next_m_idx = 0
    while next_m_idx < len(milestones) and milestones[next_m_idx] <= 0:
        next_m_idx += 1

    pbar = tqdm(train_loader, desc="Training (Cluster-Only Mixed)", mininterval=1.0)
    last_logged_val = val0

    try:
        for x, y in pbar:
            if opt_step >= max_opt_steps:
                break
            if seen_tokens + tokens_per_micro > cfg.token_budget:
                break

            micro_step += 1

            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            if mu is not None and sigma is not None:
                x = (x - mu) / sigma
                y = (y - mu) / sigma

            with amp.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                loss, st = clustering_loss_inbatch(
                    model=model,
                    x_in=x,
                    y_in=y,
                    clustering_margin=cfg.clustering_loss_margin,
                    clustering_weight=cfg.clustering_loss_weight,
                    negative_mode=cfg.negative_mode,
                    mixed_hard_ratio=cfg.mixed_hard_ratio,
                    detach_targets=False,
                )

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue

            (loss / cfg.grad_accum_steps).backward()

            for k in acc:
                acc[k] += float(st.get(k, 0.0))

            seen_tokens += tokens_per_micro

            if next_m_idx < len(milestones) and seen_tokens >= milestones[next_m_idx]:
                wall = time.time() - start_t
                v = evaluate_cluster(model, val_loader, device, mu, sigma, cfg)
                w_m.writerow([
                    int(milestones[next_m_idx]),
                    float(milestones[next_m_idx] / 1e6),
                    int(micro_step),
                    int(opt_step),
                    int(seen_tokens),
                    "" if v["val_loss"] is None else float(v["val_loss"]),
                    R_or_blank(v["val_loss"]),
                    "" if v["val_cluster_loss"] is None else float(v["val_cluster_loss"]),
                    R_or_blank(v["val_cluster_loss"]),
                    "" if v["val_d_pos"] is None else float(v["val_d_pos"]),
                    "" if v["val_d_neg"] is None else float(v["val_d_neg"]),
                    "" if v["val_d_gap"] is None else float(v["val_d_gap"]),
                    "" if v["val_cluster_active_frac"] is None else float(v["val_cluster_active_frac"]),
                    fmt_hms(wall),
                    float(cfg.loss_report_scale),
                ])
                f_miles.flush()

                while next_m_idx < len(milestones) and seen_tokens >= milestones[next_m_idx]:
                    next_m_idx += 1

            if (micro_step % cfg.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                train_loss = acc["loss_total"] / cfg.grad_accum_steps
                train_cluster = acc["cluster_loss"] / cfg.grad_accum_steps
                train_dpos = acc["d_pos_mean"] / cfg.grad_accum_steps
                train_dneg = acc["d_neg_mean"] / cfg.grad_accum_steps
                train_dgap = acc["d_gap"] / cfg.grad_accum_steps
                train_active_c = acc["cluster_active_frac"] / cfg.grad_accum_steps

                for k in acc:
                    acc[k] = 0.0

                now = time.time()
                sec_per_opt = now - last_opt_t
                last_opt_t = now

                tps = tokens_per_opt_step / max(sec_per_opt, 1e-9)
                tps_ema_val = tps_ema.update(tps)

                remaining_tokens = max(cfg.token_budget - seen_tokens, 0)
                eta_sec = remaining_tokens / max(tps_ema_val, 1e-9)

                mem = gpu_mem_stats(device)

                v = {k: None for k in [
                    "val_loss", "val_cluster_loss",
                    "val_d_pos", "val_d_neg", "val_d_gap",
                    "val_cluster_active_frac"
                ]}
                if cfg.eval_every_opt_steps > 0 and (opt_step % cfg.eval_every_opt_steps) == 0:
                    v = evaluate_cluster(model, val_loader, device, mu, sigma, cfg)
                    last_logged_val = v

                    if cfg.save_best and (v["val_loss"] is not None) and (float(v["val_loss"]) < best_val):
                        best_val = float(v["val_loss"])
                        best_opt_step = opt_step
                        _try_save_checkpoint(
                            os.path.join(ckpt_dir, "ckpt_best.pt"),
                            model, opt, cfg,
                            state={
                                "micro_step": int(micro_step),
                                "opt_step": int(opt_step),
                                "seen_tokens": int(seen_tokens),
                                "best_val": float(best_val),
                                "time_wall_sec": float(now - start_t),
                                "loss_report_scale": float(cfg.loss_report_scale),
                            },
                            save_opt=cfg.save_optimizer_in_best,
                        )

                writer.writerow([
                    int(micro_step), int(opt_step), int(seen_tokens),
                    float(train_loss), float(train_loss * cfg.loss_report_scale),
                    float(train_cluster), float(train_cluster * cfg.loss_report_scale),
                    float(train_dpos), float(train_dneg), float(train_dgap),
                    float(train_active_c),
                    "" if v["val_loss"] is None else float(v["val_loss"]),
                    R_or_blank(v["val_loss"]),
                    "" if v["val_cluster_loss"] is None else float(v["val_cluster_loss"]),
                    R_or_blank(v["val_cluster_loss"]),
                    "" if v["val_d_pos"] is None else float(v["val_d_pos"]),
                    "" if v["val_d_neg"] is None else float(v["val_d_neg"]),
                    "" if v["val_d_gap"] is None else float(v["val_d_gap"]),
                    "" if v["val_cluster_active_frac"] is None else float(v["val_cluster_active_frac"]),
                    float(sec_per_opt), float(tps_ema_val),
                    float(mem["alloc_gb"]), float(mem["reserved_gb"]), float(mem["peak_alloc_gb"]),
                    fmt_hms(eta_sec),
                    float(cfg.loss_report_scale),
                ])
                f_metrics.flush()

                if cfg.save_every_opt_steps > 0 and (opt_step % cfg.save_every_opt_steps) == 0:
                    ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{opt_step:06d}.pt")
                    _try_save_checkpoint(
                        ckpt_path, model, opt, cfg,
                        state={
                            "micro_step": int(micro_step),
                            "opt_step": int(opt_step),
                            "seen_tokens": int(seen_tokens),
                            "train_loss_raw": float(train_loss),
                            "train_cluster_loss_raw": float(train_cluster),
                            "train_d_pos": float(train_dpos),
                            "train_d_neg": float(train_dneg),
                            "train_d_gap": float(train_dgap),
                            "train_cluster_active_frac": float(train_active_c),
                            "last_val_raw": {k: (None if v[k] is None else float(v[k])) for k in v},
                            "tokens_per_sec_ema": float(tps_ema_val),
                            "time_wall_sec": float(now - start_t),
                            "best_val_raw": float(best_val) if math.isfinite(best_val) else None,
                            "best_opt_step": int(best_opt_step),
                            "loss_report_scale": float(cfg.loss_report_scale),
                        },
                        save_opt=cfg.save_optimizer_in_periodic,
                    )
                    prune_old_checkpoints(ckpt_dir, cfg.keep_last_k_checkpoints)

                if cfg.progress_every_opt_steps > 0 and (opt_step % cfg.progress_every_opt_steps) == 0:
                    print(
                        f"[progress] opt_step={opt_step}/{max_opt_steps} "
                        f"tokens={seen_tokens/1e6:.2f}M/{cfg.token_budget/1e6:.2f}M "
                        f"tps_ema={tps_ema_val:,.0f} ETA={fmt_hms(eta_sec)} "
                        f"loss_raw={train_loss:.4f} loss_rep={train_loss*cfg.loss_report_scale:.3f} "
                        f"cluster_raw={train_cluster:.4f} "
                        f"dpos={train_dpos:.3f} dneg={train_dneg:.3f} dgap={train_dgap:.3f} "
                        f"active_c={train_active_c:.2f} "
                        f"GPU alloc={mem['alloc_gb']:.2f}GiB res={mem['reserved_gb']:.2f}GiB peak={mem['peak_alloc_gb']:.2f}GiB"
                    )

                val_loss = None if last_logged_val is None else last_logged_val.get("val_loss", None)
                val_str = "" if val_loss is None else f"{float(val_loss)*cfg.loss_report_scale:.3f}"

                pbar.set_postfix(
                    train=f"{train_loss*cfg.loss_report_scale:.3f}",
                    val=val_str,
                    dgap=f"{train_dgap:.3f}",
                    tps=f"{tps_ema_val:,.0f}",
                    eta=fmt_hms(eta_sec),
                    opt=f"{opt_step}/{max_opt_steps}",
                )

    finally:
        f_metrics.close()
        f_miles.close()

    val_final = evaluate_cluster(model, val_loader, device, mu, sigma, cfg)

    end_t = time.time()
    wall_sec = end_t - start_t
    wall_hours = wall_sec / 3600.0

    tput = tps_ema.value if tps_ema.value is not None else 0.0
    mem_final = gpu_mem_stats(device)

    tokens_m = seen_tokens / 1e6
    if (val0["val_loss"] is not None) and (val_final["val_loss"] is not None):
        delta_val_raw = float(val0["val_loss"] - val_final["val_loss"])
        delta_per_10m_raw = delta_val_raw / max(tokens_m / 10.0, 1e-12)
    else:
        delta_val_raw = None
        delta_per_10m_raw = None

    summary = {
        "model_name": cfg.model_name,
        "tokens_seen": int(seen_tokens),
        "tokens_seen_m": float(tokens_m),
        "opt_steps": int(opt_step),
        "max_opt_steps_by_budget": int(max_opt_steps),
        "tokens_per_opt_step": int(tokens_per_opt_step),
        "effective_tokens_target": int(effective_tokens),
        "wall_time_sec": float(wall_sec),
        "wall_time_hms": fmt_hms(wall_sec),
        "train_time_hours": float(wall_hours),
        "throughput_tok_per_sec_ema": float(tput),
        "gpu": mem_final,
        "loss_cfg": {
            "cluster_weight": float(cfg.clustering_loss_weight),
            "cluster_margin": float(cfg.clustering_loss_margin),
            "negative_mode": cfg.negative_mode,
            "mixed_hard_ratio": float(cfg.mixed_hard_ratio),
            "lr": float(cfg.lr),
            "grad_accum_steps": int(cfg.grad_accum_steps),
            "batch_size": int(cfg.batch_size),
            "model_dim": int(cfg.model_dim),
            "num_heads": int(cfg.num_heads),
            "num_layers": int(cfg.num_layers),
        },
        "reporting": {
            "target_val0_report": float(cfg.target_val0_report),
            "loss_report_scale": float(cfg.loss_report_scale),
        },
        "val_raw": {
            "val0": {k: (None if val0[k] is None else float(val0[k])) for k in val0},
            "val_final": {k: (None if val_final[k] is None else float(val_final[k])) for k in val_final},
            "delta_val_loss_0_to_final_raw": delta_val_raw,
            "delta_val_loss_per_10m_tokens_raw": delta_per_10m_raw,
            "best_val_loss_seen_raw": None if not math.isfinite(best_val) else float(best_val),
            "best_val_opt_step": int(best_opt_step),
        },
        "val_reported": {
            "val0_report": None if val0["val_loss"] is None else float(val0["val_loss"]) * float(cfg.loss_report_scale),
            "val_final_report": None if val_final["val_loss"] is None else float(val_final["val_loss"]) * float(cfg.loss_report_scale),
        },
    }
    atomic_write_json(os.path.join(cfg.out_dir, cfg.summary_json), summary)

    print("\n=== TRAINING COMPLETE (Cluster-Only Mixed) ===")
    print(f"tokens_seen: {seen_tokens:,} ({tokens_m:.2f}M) / {cfg.token_budget:,}")
    print(f"opt_steps:   {opt_step} / max_opt_steps={max_opt_steps}")
    print(f"wall_time:   {fmt_hms(wall_sec)}  ({wall_hours:.2f} h)")
    print(f"tput_ema:    {float(tput):,.0f} tok/s")
    print(f"GPU alloc/res/peak: {mem_final['alloc_gb']:.2f}/{mem_final['reserved_gb']:.2f}/{mem_final['peak_alloc_gb']:.2f} GiB")
    print(f"loss_rpt_scale: {cfg.loss_report_scale:.6f} (target val@0 ~ {cfg.target_val0_report})")
    print(f"val@0 raw:      {val0}")
    print(f"val@final raw:  {val_final}")
    if val0["val_loss"] is not None and val_final["val_loss"] is not None:
        print(f"val@0 rpt:      {float(val0['val_loss']) * cfg.loss_report_scale:.3f}")
        print(f"val@final rpt:  {float(val_final['val_loss']) * cfg.loss_report_scale:.3f}")
    print(f"best@eval (raw):  val_loss={None if not math.isfinite(best_val) else best_val} at opt_step={best_opt_step}")

    print(f"\nSaved artifacts:")
    print(f" - metrics:    {os.path.join(cfg.out_dir, cfg.metrics_csv)}")
    print(f" - milestones: {os.path.join(cfg.out_dir, cfg.milestones_csv)}")
    print(f" - summary:    {os.path.join(cfg.out_dir, cfg.summary_json)}")
    print(f" - checkpoints:{ckpt_dir}")
    if cfg.save_best:
        print(f" - best ckpt:  {os.path.join(ckpt_dir, 'ckpt_best.pt')}")


if __name__ == "__main__":
    main()
