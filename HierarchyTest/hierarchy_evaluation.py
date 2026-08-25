from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class Taxonomy:
    concept_ids: Tuple[str, ...]
    parent: Mapping[str, Optional[str]]
    split: Mapping[str, str]
    adjacency: Mapping[str, Tuple[str, ...]]


def load_taxonomy_csv(path: str) -> Taxonomy:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    required = {"concept_id", "parent_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")

    ids = [str(r["concept_id"]).strip() for r in rows]
    if any(not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("concept_id values must be non-empty and unique")

    id_set = set(ids)
    parent: Dict[str, Optional[str]] = {}
    split: Dict[str, str] = {}

    for row in rows:
        node = str(row["concept_id"]).strip()
        p = str(row["parent_id"]).strip() or None

        if p is not None and p not in id_set:
            raise ValueError(f"Unknown parent_id {p!r} for concept {node!r}")
        if p == node:
            raise ValueError(f"Concept {node!r} cannot be its own parent")

        s = str(row["split"]).strip().lower()
        if s not in {"calibration", "test"}:
            raise ValueError(f"Split for {node!r} must be calibration or test")

        parent[node] = p
        split[node] = s

    roots = [x for x in ids if parent[x] is None]
    if len(roots) != 1:
        raise ValueError(
            f"Taxonomy must be one tree with exactly one root; found {roots}"
        )

    root = roots[0]

    children: Dict[str, List[str]] = defaultdict(list)
    adjacency_lists: Dict[str, List[str]] = {x: [] for x in ids}

    for node, p in parent.items():
        if p is not None:
            children[p].append(node)
            adjacency_lists[p].append(node)
            adjacency_lists[node].append(p)

    # Validate that every node is reachable from the single root.
    visited = {root}
    queue = deque([root])

    while queue:
        node = queue.popleft()
        for child in children[node]:
            if child in visited:
                raise ValueError("Taxonomy contains a cycle")
            visited.add(child)
            queue.append(child)

    if len(visited) != len(ids):
        missing = sorted(id_set - visited)
        raise ValueError(
            f"Taxonomy is disconnected or cyclic; unreachable nodes: {missing}"
        )

    return Taxonomy(
        concept_ids=tuple(ids),
        parent=parent,
        split=split,
        adjacency={k: tuple(v) for k, v in adjacency_lists.items()},
    )


def load_embedding_file(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, Mapping) and "concept_ids" in obj and "embeddings" in obj:
        ids = [str(x) for x in obj["concept_ids"]]
        values = torch.as_tensor(obj["embeddings"]).float()

        if values.ndim != 2 or values.shape[0] != len(ids):
            raise ValueError("embeddings must have shape [len(concept_ids), D]")

        return {node: values[i] for i, node in enumerate(ids)}

    if isinstance(obj, Mapping):
        result = {
            str(k): torch.as_tensor(v).float().squeeze()
            for k, v in obj.items()
        }
        if result and all(v.ndim == 1 for v in result.values()):
            return result

    raise ValueError(
        f"Unsupported embedding file {path}; expected concept_ids+embeddings "
        "or id->vector mapping"
    )


def align_embeddings(
    taxonomy: Taxonomy,
    embeddings: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    missing = [x for x in taxonomy.concept_ids if x not in embeddings]
    if missing:
        raise ValueError(
            f"Missing embeddings for {len(missing)} concepts: {missing[:10]}"
        )

    matrix = torch.stack(
        [embeddings[x].float() for x in taxonomy.concept_ids]
    )

    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("Aligned embeddings must be a finite [N, D] tensor")

    return matrix

# Spearman correlation
def _rankdata(values: torch.Tensor) -> torch.Tensor:
    x = values.detach().double().flatten()
    order = torch.argsort(x, stable=True)
    sorted_x = x[order]
    ranks = torch.empty_like(x)

    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1

        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end

    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() != y.numel() or x.numel() < 2:
        return float("nan")

    rx = _rankdata(x)
    ry = _rankdata(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()

    denom = rx.norm() * ry.norm()

    if denom <= 0:
        return float("nan")

    return float((rx @ ry / denom).item())

def pairwise_tree_distances(taxonomy: Taxonomy) -> torch.Tensor:
    n = len(taxonomy.concept_ids)
    index = {x: i for i, x in enumerate(taxonomy.concept_ids)}
    distances = torch.empty((n, n), dtype=torch.float64)

    for source in taxonomy.concept_ids:
        d = {source: 0}
        queue = deque([source])

        while queue:
            node = queue.popleft()
            for nxt in taxonomy.adjacency[node]:
                if nxt not in d:
                    d[nxt] = d[node] + 1
                    queue.append(nxt)

        i = index[source]
        for target, hops in d.items():
            distances[i, index[target]] = float(hops)

    return distances


def pairwise_euclidean(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x.double(), x.double(), p=2)


def pairwise_poincare(x: torch.Tensor, curvature: float) -> torch.Tensor:
    if curvature <= 0:
        raise ValueError("curvature must be positive")

    z = x.double()
    c = float(curvature)

    max_norm = (1.0 - 1e-7) / math.sqrt(c)
    norms = z.norm(dim=-1, keepdim=True).clamp_min(1e-15)
    z = z * (max_norm / norms).clamp_max(1.0)

    diff2 = (z[:, None, :] - z[None, :, :]).square().sum(-1)
    x2 = z.square().sum(-1)

    denom = (
        (1.0 - c * x2)[:, None]
        * (1.0 - c * x2)[None, :]
    )

    arg = 1.0 + 2.0 * c * diff2 / denom.clamp_min(1e-15)

    return torch.acosh(arg.clamp_min(1.0)) / math.sqrt(c)


def _pair_values(
    matrix: torch.Tensor,
    indices: Sequence[int],
) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long)

    if len(idx) < 2:
        return torch.empty(0, dtype=matrix.dtype)

    sub = matrix[idx][:, idx]
    mask = torch.triu(
        torch.ones_like(sub, dtype=torch.bool),
        diagonal=1,
    )

    return sub[mask]


def tree_metric_preservation(
    tree_distances: torch.Tensor,
    latent_distances: torch.Tensor,
    calibration_indices: Sequence[int],
    test_indices: Sequence[int],
    eps: float = 1e-12,
) -> Dict[str, float]:
    tree_cal = _pair_values(tree_distances, calibration_indices)
    latent_cal = _pair_values(latent_distances, calibration_indices)

    tree_test = _pair_values(tree_distances, test_indices)
    latent_test = _pair_values(latent_distances, test_indices)

    if tree_cal.numel() == 0 or tree_test.numel() == 0:
        raise ValueError("Need at least two calibration and two test concepts")

    scale = (
        (tree_cal @ latent_cal)
        / latent_cal.square().sum().clamp_min(eps)
    )

    predicted_tree_distance = scale * latent_test
    residual = tree_test - predicted_tree_distance

    heldout_mean_relative_distortion = (
        residual.abs() / tree_test.clamp_min(eps)
    ).mean()

    heldout_spearman = spearman_correlation(
        tree_test,
        latent_test,
    )

    return {
        "heldout_mean_relative_distortion": float(
            heldout_mean_relative_distortion
        ),
        "heldout_spearman": heldout_spearman,
    }


def local_structural_recovery(
    taxonomy: Taxonomy,
    distances: torch.Tensor,
    query_indices: Sequence[int],
    k: int = 5,
) -> Dict[str, float]:
    if k < 1:
        raise ValueError("k must be >= 1")

    ids = taxonomy.concept_ids
    index = {x: i for i, x in enumerate(ids)}

    parent_rr = 0.0
    parent_n = 0

    sibling_hits = 0
    sibling_n = 0

    for qi in query_indices:
        node = ids[qi]

        order = torch.argsort(
            distances[qi],
            stable=True,
        ).tolist()

        order = [j for j in order if j != qi]
        topk = order[: min(k, len(order))]

        parent_id = taxonomy.parent[node]

        if parent_id is None:
            continue

        parent_n += 1
        parent_idx = index[parent_id]
        parent_rank = order.index(parent_idx) + 1
        parent_rr += 1.0 / parent_rank

        # Sibling Recall@k
        sibling_indices = {
            index[x]
            for x in ids
            if x != node and taxonomy.parent[x] == parent_id
        }

        if sibling_indices:
            sibling_n += 1
            sibling_hits += int(
                any(j in sibling_indices for j in topk)
            )

    return {
        "parent_mean_reciprocal_rank": (
            parent_rr / parent_n
            if parent_n
            else float("nan")
        ),
        f"sibling_recall_at_{k}": (
            sibling_hits / sibling_n
            if sibling_n
            else float("nan")
        ),
    }


# Main evaluation function
def evaluate_geometry(
    taxonomy: Taxonomy,
    embeddings: torch.Tensor,
    geometry: str,
    curvature: Optional[float] = None,
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Evaluate exactly four hierarchy metrics.

    Returns:
        {
            "tree_metric_preservation": {
                "heldout_mean_relative_distortion": ...,
                "heldout_spearman": ...,
            },
            "local_structural_recovery": {
                "parent_mean_reciprocal_rank": ...,
                "sibling_recall_at_5": ...,
            },
        }
    """
    calibration_indices = [
        i
        for i, x in enumerate(taxonomy.concept_ids)
        if taxonomy.split[x] == "calibration"
    ]

    test_indices = [
        i
        for i, x in enumerate(taxonomy.concept_ids)
        if taxonomy.split[x] == "test"
    ]

    tree_distances = pairwise_tree_distances(taxonomy)

    if geometry == "poincare":
        if curvature is None:
            raise ValueError("Poincaré evaluation requires curvature")

        latent_distances = pairwise_poincare(
            embeddings,
            curvature,
        )

    elif geometry == "euclidean":
        latent_distances = pairwise_euclidean(embeddings)

    else:
        raise ValueError(
            "geometry must be 'poincare' or 'euclidean'"
        )

    return {
        "tree_metric_preservation": tree_metric_preservation(
            tree_distances,
            latent_distances,
            calibration_indices,
            test_indices,
        ),
        "local_structural_recovery": local_structural_recovery(
            taxonomy,
            latent_distances,
            test_indices,
            k=k,
        ),
    }
