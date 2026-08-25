from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch

from h_lcm import HyperbolicLCM
from hierarchy_evaluation import (
    align_embeddings,
    evaluate_geometry,
    load_embedding_file,
    load_taxonomy_csv,
)


def _load_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["cfg"]
    model = HyperbolicLCM(
        in_dim=int(cfg["in_dim"]),
        model_dim=int(cfg["model_dim"]),
        num_heads=int(cfg["num_heads"]),
        num_layers=int(cfg["num_layers"]),
        ffn_mult=int(cfg.get("ffn_mult", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        manifold_c=float(cfg["manifold_c"]),
        causal=bool(cfg.get("causal", True)),
        input_scale=float(cfg.get("input_scale", 0.05)),
        input_max_norm=float(cfg.get("input_max_norm", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, cfg


def _load_normalizer(path: Optional[str], device: torch.device):
    if path is None:
        return None, None
    obj = torch.load(path, map_location=device, weights_only=False)
    return obj["mu"].float().to(device), obj["sigma"].float().to(device).clamp_min(1e-4)


@torch.inference_mode()
def _encode_hyperbolic(
    model: HyperbolicLCM,
    inputs: torch.Tensor,
    device: torch.device,
    mu: Optional[torch.Tensor],
    sigma: Optional[torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size].to(device).float()
        if mu is not None:
            batch = (batch - mu) / sigma
        point = model(batch[:, None, :])[:, 0, :]
        outputs.append(point.cpu())
    return torch.cat(outputs, dim=0)


def _json_safe(obj):
    if isinstance(obj, float) and not torch.isfinite(torch.tensor(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    return obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taxonomy", required=True, help="CSV: concept_id,parent_id,split")
    p.add_argument(
        "--input-embeddings",
        required=True,
        help="Aligned DeBERTa vectors (.pt); also used as the default Euclidean baseline",
    )
    p.add_argument("--checkpoint", required=True, help="HyperbolicLCM checkpoint")
    p.add_argument(
        "--euclidean-embeddings",
        help="Optional learned Euclidean baseline vectors; defaults to input embeddings",
    )
    p.add_argument("--normalizer", help="normalizer.pt used during H-LCM training")
    p.add_argument("--output", default="hierarchy_metrics.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    taxonomy = load_taxonomy_csv(args.taxonomy)
    input_map = load_embedding_file(args.input_embeddings)
    input_matrix = align_embeddings(taxonomy, input_map)

    model, cfg = _load_checkpoint(args.checkpoint, device)
    mu, sigma = _load_normalizer(args.normalizer, device)
    hyperbolic = _encode_hyperbolic(
        model, input_matrix, device, mu, sigma, args.batch_size
    )

    euclidean_map = (
        load_embedding_file(args.euclidean_embeddings)
        if args.euclidean_embeddings else input_map
    )
    euclidean = align_embeddings(taxonomy, euclidean_map)
    results: Dict[str, object] = {
        "metadata": {
            "taxonomy": str(Path(args.taxonomy)),
            "n_concepts": len(taxonomy.concept_ids),
            "n_calibration": sum(taxonomy.split[x] == "calibration"
                                 for x in taxonomy.concept_ids),
            "n_test": sum(taxonomy.split[x] == "test"
                          for x in taxonomy.concept_ids),
            "branch_purity_k": args.k,
            "euclidean_source": (
                args.euclidean_embeddings or args.input_embeddings
            ),
        },
        "hyperbolic_lcm": evaluate_geometry(
            taxonomy,
            hyperbolic,
            geometry="poincare",
            curvature=float(cfg["manifold_c"]),
            k=args.k,
        ),
        "euclidean_baseline": evaluate_geometry(
            taxonomy, euclidean, geometry="euclidean", k=args.k
        ),
    }
    results = _json_safe(results)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
