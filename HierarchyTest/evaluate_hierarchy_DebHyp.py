from pathlib import Path
import math
import os
import sys
import subprocess

import pandas as pd
import torch

# PATHS
PROJECT_DIR = Path(
    "/home/user/twovolume/Nisha/hlcm/latent"
).resolve()

HLCM_ROOT = Path(
    "/home/user/twovolume/Nisha/hlcm"
).resolve()

TAXONOMY_CSV = (
    PROJECT_DIR / "wordnet_animal_taxonomy.csv"
)

CHECKPOINT = Path(
    "/home/user/twovolume/Nisha/hlcm/"
    "runs/hyperbolic_cluster/checkpoints/ckpt_best.pt"
).resolve()

NORMALIZER = Path(
    "/home/user/twovolume/Nisha/hlcm/normalizer.pt"
).resolve()

INPUT_EMBEDDINGS = (
    PROJECT_DIR / "wordnet_animal_input_embeddings.pt"
)

OUTPUT_TABLE = (
    PROJECT_DIR / "wordnet_animal_four_metrics.csv"
)

os.chdir(PROJECT_DIR)

print("Project directory:", PROJECT_DIR)
print("Taxonomy:", TAXONOMY_CSV)
print("Checkpoint:", CHECKPOINT)
print("Normalizer:", NORMALIZER)
print("Input embeddings:", INPUT_EMBEDDINGS)


model_candidates = sorted(
    HLCM_ROOT.rglob("h_lcm.py")
)

if not model_candidates:
    raise FileNotFoundError(
        "Could not find h_lcm1_Copy1.py"
    )

MODEL_CODE_DIR = model_candidates[0].parent

attention_file = (
    MODEL_CODE_DIR / "hyperbolic_attention.py"
)

if not attention_file.exists():
    raise FileNotFoundError(
        f"Attention file not found beside model: {attention_file}"
    )

if str(MODEL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_CODE_DIR))

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

required_paths = {
    "Project directory": PROJECT_DIR,
    "Taxonomy CSV": TAXONOMY_CSV,
    "Checkpoint": CHECKPOINT,
    "Normalizer": NORMALIZER,
    "Hierarchy evaluator": PROJECT_DIR / "hierarchy_evaluation.py",
    "Evaluation runner": PROJECT_DIR / "evaluate_hierarchy.py",
    "Embedding generator": PROJECT_DIR / "prepare_taxonomy_embeddings.py",
}

missing = []

for label, path in required_paths.items():
    if not path.exists():
        missing.append((label, path))

if missing:
    raise FileNotFoundError(
        "Some required files are missing:\n"
        + "\n".join(
            f"{label}: {path}"
            for label, path in missing
        )
    )

print("Model code directory:", MODEL_CODE_DIR)

from hierarchy_evaluation import (
    load_taxonomy_csv,
    load_embedding_file,
    align_embeddings,
    pairwise_tree_distances,
    pairwise_euclidean,
    pairwise_poincare,
    spearman_correlation,
)

taxonomy = load_taxonomy_csv(
    str(TAXONOMY_CSV)
)

calibration_indices = [
    i
    for i, concept_id in enumerate(taxonomy.concept_ids)
    if taxonomy.split[concept_id] == "calibration"
]

test_indices = [
    i
    for i, concept_id in enumerate(taxonomy.concept_ids)
    if taxonomy.split[concept_id] == "test"
]

print("Total concepts:", len(taxonomy.concept_ids))
print("Calibration concepts:", len(calibration_indices))
print("Test concepts:", len(test_indices))

if not INPUT_EMBEDDINGS.exists():

    command = [
        sys.executable,
        str(
            PROJECT_DIR
            / "prepare_taxonomy_embeddings.py"
        ),
        "--taxonomy",
        str(TAXONOMY_CSV),
        "--output",
        str(INPUT_EMBEDDINGS),
        "--model",
        "microsoft/deberta-v3-small",
        "--batch-size",
        "64",
        "--max-length",
        "256",
        "--device",
        "cuda",
    ]

    process = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        text=True,
        capture_output=True,
    )

    print(process.stdout)

    if process.returncode != 0:
        print(process.stderr)
        raise RuntimeError(
            "DeBERTa embedding generation failed"
        )

input_embedding_map = load_embedding_file(
    str(INPUT_EMBEDDINGS)
)

input_matrix = align_embeddings(
    taxonomy,
    input_embedding_map,
)

print("Aligned DeBERTa embedding shape:", input_matrix.shape)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable in this Jupyter session."
    )

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)

print("DEVICE:", DEVICE)
print("GPU:", torch.cuda.get_device_name(DEVICE))

def _pair_values(matrix, indices):
    idx = torch.as_tensor(
        indices,
        dtype=torch.long,
    )

    if len(idx) < 2:
        return torch.empty(
            0,
            dtype=matrix.dtype,
        )

    sub = matrix[idx][:, idx]

    mask = torch.triu(
        torch.ones_like(
            sub,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    return sub[mask]


def calculate_tree_metrics(
    tree_distances,
    latent_distances,
    calibration_indices,
    test_indices,
    eps=1e-12,
):

    tree_cal = _pair_values(
        tree_distances,
        calibration_indices,
    )

    latent_cal = _pair_values(
        latent_distances,
        calibration_indices,
    )

    tree_test = _pair_values(
        tree_distances,
        test_indices,
    )

    latent_test = _pair_values(
        latent_distances,
        test_indices,
    )

    if tree_cal.numel() == 0 or tree_test.numel() == 0:
        raise ValueError(
            "Need at least two calibration and two test concepts"
        )

    scale = (
        (tree_cal @ latent_cal)
        / latent_cal.square().sum().clamp_min(eps)
    )

    predicted = (
        scale * latent_test
    )

    residual = (
        tree_test - predicted
    )

    heldout_mean_relative_distortion = (
        residual.abs()
        / tree_test.clamp_min(eps)
    ).mean()

    heldout_spearman = spearman_correlation(
        tree_test,
        latent_test,
    )

    return {
        "heldout_mean_relative_distortion":
            float(heldout_mean_relative_distortion),

        "heldout_spearman":
            float(heldout_spearman),
    }


def calculate_local_metrics(
    taxonomy,
    distances,
    query_indices,
):

    ids = taxonomy.concept_ids

    index = {
        concept_id: i
        for i, concept_id in enumerate(ids)
    }

    parent_rr_sum = 0.0
    parent_query_count = 0

    sibling_hits = 0
    sibling_query_count = 0

    for query_index in query_indices:

        node = ids[query_index]

        order = torch.argsort(
            distances[query_index],
            stable=True,
        ).tolist()

        # Remove query concept itself.
        order = [
            j
            for j in order
            if j != query_index
        ]

        top5 = order[:5]

        parent_id = taxonomy.parent[node]

        if parent_id is None:
            continue

        # Parent Mean Reciprocal Rank
        parent_query_count += 1

        parent_index = index[parent_id]

        parent_rank = (
            order.index(parent_index) + 1
        )

        parent_rr_sum += (
            1.0 / parent_rank
        )

        # Sibling Recall@5
        sibling_indices = {
            index[other_id]
            for other_id in ids
            if (
                other_id != node
                and taxonomy.parent[other_id] == parent_id
            )
        }

        if sibling_indices:

            sibling_query_count += 1

            found_sibling = any(
                candidate in sibling_indices
                for candidate in top5
            )

            sibling_hits += int(
                found_sibling
            )

    parent_mrr = (
        parent_rr_sum / parent_query_count
        if parent_query_count
        else float("nan")
    )

    sibling_recall_at_5 = (
        sibling_hits / sibling_query_count
        if sibling_query_count
        else float("nan")
    )

    return {
        "parent_mean_reciprocal_rank":
            parent_mrr,

        "sibling_recall_at_5":
            sibling_recall_at_5,
    }


def evaluate_four_metrics(
    taxonomy,
    embeddings,
    geometry,
    calibration_indices,
    test_indices,
    curvature=None,
):

    tree_distances = pairwise_tree_distances(
        taxonomy
    )

    if geometry == "poincare":

        if curvature is None:
            raise ValueError(
                "Poincare evaluation requires curvature"
            )

        latent_distances = pairwise_poincare(
            embeddings,
            curvature,
        )

    elif geometry == "euclidean":

        latent_distances = pairwise_euclidean(
            embeddings
        )

    else:
        raise ValueError(
            "geometry must be 'poincare' or 'euclidean'"
        )

    tree_results = calculate_tree_metrics(
        tree_distances=tree_distances,
        latent_distances=latent_distances,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
    )

    local_results = calculate_local_metrics(
        taxonomy=taxonomy,
        distances=latent_distances,
        query_indices=test_indices,
    )

    return {
        **tree_results,
        **local_results,
    }

from evaluate_hierarchy import (
    _load_checkpoint,
    _load_normalizer,
    _encode_hyperbolic,
)

model, model_config = _load_checkpoint(
    str(CHECKPOINT),
    DEVICE,
)

normalizer_mean, normalizer_std = _load_normalizer(
    str(NORMALIZER),
    DEVICE,
)

HLCM_BATCH_SIZE = 32

torch.cuda.empty_cache()

with torch.inference_mode():

    hyperbolic_embeddings = _encode_hyperbolic(
        model=model,
        inputs=input_matrix,
        device=DEVICE,
        mu=normalizer_mean,
        sigma=normalizer_std,
        batch_size=HLCM_BATCH_SIZE,
    )

curvature = float(
    model_config["manifold_c"]
)

ball_radius = (
    1.0 / math.sqrt(curvature)
)

embedding_norms = (
    hyperbolic_embeddings.norm(dim=-1)
)

if not torch.isfinite(
    hyperbolic_embeddings
).all():
    raise ValueError(
        "Hyperbolic embeddings contain NaN or Inf"
    )

if not (
    embedding_norms < ball_radius
).all():
    raise ValueError(
        "Some embeddings are outside the Poincare ball"
    )

print(
    "Hyperbolic embedding shape:",
    hyperbolic_embeddings.shape,
)

print(
    "Curvature:",
    curvature,
)

del model
del normalizer_mean
del normalizer_std

torch.cuda.empty_cache()

hyperbolic_results = evaluate_four_metrics(
    taxonomy=taxonomy,
    embeddings=hyperbolic_embeddings,
    geometry="poincare",
    curvature=curvature,
    calibration_indices=calibration_indices,
    test_indices=test_indices,
)

print("Hyperbolic LCM:")
for metric, value in hyperbolic_results.items():
    print(f"{metric}: {value:.4f}")

# CALCULATE RAW DeBERTa BASELINE METRICS
euclidean_embeddings = input_matrix

euclidean_results = evaluate_four_metrics(
    taxonomy=taxonomy,
    embeddings=euclidean_embeddings,
    geometry="euclidean",
    calibration_indices=calibration_indices,
    test_indices=test_indices,
)

print("Raw DeBERTa baseline:")
for metric, value in euclidean_results.items():
    print(f"{metric}: {value:.4f}")

comparison_table = pd.DataFrame(
    {
        "Metric": [
            "heldout_mean_relative_distortion",
            "heldout_spearman",
            "parent_mean_reciprocal_rank",
            "sibling_recall_at_5",
        ],

        "Raw DeBERTa baseline": [
            euclidean_results[
                "heldout_mean_relative_distortion"
            ],
            euclidean_results[
                "heldout_spearman"
            ],
            euclidean_results[
                "parent_mean_reciprocal_rank"
            ],
            euclidean_results[
                "sibling_recall_at_5"
            ],
        ],

        "Hyperbolic LCM": [
            hyperbolic_results[
                "heldout_mean_relative_distortion"
            ],
            hyperbolic_results[
                "heldout_spearman"
            ],
            hyperbolic_results[
                "parent_mean_reciprocal_rank"
            ],
            hyperbolic_results[
                "sibling_recall_at_5"
            ],
        ],
    }
)

comparison_table.to_csv(
    OUTPUT_TABLE,
    index=False,
)

display(
    comparison_table.style
    .format(
        {
            "Raw DeBERTa baseline": "{:.4f}",
            "Hyperbolic LCM": "{:.4f}",
        }
    )
    .hide(axis="index")
)

print(
    "\nSaved:",
    OUTPUT_TABLE,
)
