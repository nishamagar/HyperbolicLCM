from pathlib import Path
import gc
import math
import os
import sys

import pandas as pd
import torch


PROJECT_DIR = Path('/home/user/twovolume/Nisha/hlcm/latent').resolve()
HYPERBOLIC_CODE_DIR = Path('/home/user/twovolume/Nisha/hlcm').resolve()
EUCLIDEAN_CODE_DIR = Path('/home/user/twovolume/Nisha/LCM').resolve()

TAXONOMY_CSV = PROJECT_DIR / 'wordnet_animal_taxonomy.csv'
INPUT_EMBEDDINGS = PROJECT_DIR / 'wordnet_animal_input_embeddings.pt'

HYPERBOLIC_CHECKPOINT = (
    HYPERBOLIC_CODE_DIR / 'runs/hyperbolic_cluster/checkpoints/ckpt_best.pt'
)
HYPERBOLIC_NORMALIZER = HYPERBOLIC_CODE_DIR / 'normalizer.pt'

EUCLIDEAN_CHECKPOINT = (
    EUCLIDEAN_CODE_DIR / 'runs/base_lcm_wt2/checkpoints/ckpt_best.pt'
)
EUCLIDEAN_NORMALIZER = EUCLIDEAN_CODE_DIR / 'normalizer.pt'

OUTPUT_TABLE = PROJECT_DIR / 'wordnet_animal_four_metrics_euc_hyp.csv'

os.chdir(PROJECT_DIR)

for code_dir in (PROJECT_DIR, HYPERBOLIC_CODE_DIR, EUCLIDEAN_CODE_DIR):
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

print('Project:', PROJECT_DIR)
print('Python:', sys.executable)
print('PyTorch:', torch.__version__)

# CHECK REQUIRED FILES
required = {
    'taxonomy': TAXONOMY_CSV,
    'concept embeddings': INPUT_EMBEDDINGS,
    'hyperbolic checkpoint': HYPERBOLIC_CHECKPOINT,
    'hyperbolic normalizer': HYPERBOLIC_NORMALIZER,
    'Euclidean checkpoint': EUCLIDEAN_CHECKPOINT,
    'Euclidean normalizer': EUCLIDEAN_NORMALIZER,
    'hierarchy evaluator': PROJECT_DIR / 'hierarchy_evaluation.py',
    'hyperbolic evaluation helpers': PROJECT_DIR / 'evaluate_hierarchy.py',
    'Euclidean model source': EUCLIDEAN_CODE_DIR / 'b_lcm.py',
}

missing = []

for label, path in required.items():
    print(f'{label}: {path} -> {path.exists()}')
    if not path.exists():
        missing.append(f'{label}: {path}')

if missing:
    raise FileNotFoundError('\n'.join(missing))

if not torch.cuda.is_available():
    raise RuntimeError(
        'CUDA is unavailable; launch this kernel inside the GPU allocation.'
    )

# If CUDA_VISIBLE_DEVICES=1, physical GPU 1 appears as logical cuda:0.
DEVICE = torch.device('cuda:0')
torch.cuda.set_device(0)

print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('Logical device:', DEVICE)
print('GPU:', torch.cuda.get_device_name(0))


from hierarchy_evaluation import (
    align_embeddings,
    load_embedding_file,
    load_taxonomy_csv,
    pairwise_euclidean,
    pairwise_poincare,
    pairwise_tree_distances,
    spearman_correlation,
)

taxonomy = load_taxonomy_csv(str(TAXONOMY_CSV))

input_embedding_map = load_embedding_file(
    str(INPUT_EMBEDDINGS)
)

input_matrix = align_embeddings(
    taxonomy,
    input_embedding_map,
).float()

calibration_indices = [
    i
    for i, concept_id in enumerate(taxonomy.concept_ids)
    if taxonomy.split[concept_id] == 'calibration'
]

test_indices = [
    i
    for i, concept_id in enumerate(taxonomy.concept_ids)
    if taxonomy.split[concept_id] == 'test'
]

if len(calibration_indices) < 2:
    raise ValueError('Need at least two calibration concepts.')

if len(test_indices) < 2:
    raise ValueError('Need at least two test concepts.')

print('Input shape:', tuple(input_matrix.shape))
print('Concepts:', len(taxonomy.concept_ids))
print('Calibration:', len(calibration_indices))
print('Test:', len(test_indices))


def pair_values(matrix, indices):
    idx = torch.as_tensor(indices, dtype=torch.long)

    if len(idx) < 2:
        return torch.empty(0, dtype=matrix.dtype)

    sub = matrix[idx][:, idx]

    mask = torch.triu(
        torch.ones_like(sub, dtype=torch.bool),
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

    tree_cal = pair_values(
        tree_distances,
        calibration_indices,
    )

    latent_cal = pair_values(
        latent_distances,
        calibration_indices,
    )

    tree_test = pair_values(
        tree_distances,
        test_indices,
    )

    latent_test = pair_values(
        latent_distances,
        test_indices,
    )

    if tree_cal.numel() == 0 or tree_test.numel() == 0:
        raise ValueError(
            'Need at least two calibration and two test concepts.'
        )

    scale = (
        (tree_cal @ latent_cal)
        / latent_cal.square().sum().clamp_min(eps)
    )

    predicted_tree_distance = scale * latent_test

    residual = tree_test - predicted_tree_distance

    heldout_mean_relative_distortion = (
        residual.abs()
        / tree_test.clamp_min(eps)
    ).mean()

    heldout_spearman = spearman_correlation(
        tree_test,
        latent_test,
    )

    return {
        'heldout_mean_relative_distortion':
            float(heldout_mean_relative_distortion),

        'heldout_spearman':
            float(heldout_spearman),
    }


def calculate_local_metrics(
    taxonomy,
    distances,
    query_indices,
    k=5,
):
  

    if k != 5:
        raise ValueError(
            'This evaluation is configured specifically for sibling_recall_at_5.'
        )

    ids = taxonomy.concept_ids
    index = {
        concept_id: i
        for i, concept_id in enumerate(ids)
    }

    parent_rr_sum = 0.0
    parent_query_count = 0

    sibling_hit_count = 0
    sibling_query_count = 0

    for query_index in query_indices:

        node = ids[query_index]

        # Sort all concepts by distance from the query.
        order = torch.argsort(
            distances[query_index],
            stable=True,
        ).tolist()

        # Remove the query concept itself.
        order = [
            j for j in order
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

        # Evaluate sibling recall 
        if sibling_indices:

            sibling_query_count += 1

            sibling_found = any(
                candidate in sibling_indices
                for candidate in top5
            )

            sibling_hit_count += int(
                sibling_found
            )

    parent_mrr = (
        parent_rr_sum / parent_query_count
        if parent_query_count
        else float('nan')
    )

    sibling_recall_at_5 = (
        sibling_hit_count / sibling_query_count
        if sibling_query_count
        else float('nan')
    )

    return {
        'parent_mean_reciprocal_rank':
            parent_mrr,

        'sibling_recall_at_5':
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

    # Ground-truth shortest-path distances in taxonomy tree.
    tree_distances = pairwise_tree_distances(
        taxonomy
    )

    # Latent-space pairwise distances.
    if geometry == 'poincare':

        if curvature is None:
            raise ValueError(
                'Poincare evaluation requires curvature.'
            )

        latent_distances = pairwise_poincare(
            embeddings,
            curvature,
        )

    elif geometry == 'euclidean':

        latent_distances = pairwise_euclidean(
            embeddings
        )

    else:
        raise ValueError(
            "geometry must be 'poincare' or 'euclidean'"
        )

    tree_metrics = calculate_tree_metrics(
        tree_distances=tree_distances,
        latent_distances=latent_distances,
        calibration_indices=calibration_indices,
        test_indices=test_indices,
    )

    local_metrics = calculate_local_metrics(
        taxonomy=taxonomy,
        distances=latent_distances,
        query_indices=test_indices,
        k=5,
    )

    return {
        **tree_metrics,
        **local_metrics,
    }

# HYPERBOLIC LCM
from evaluate_hierarchy import (
    _encode_hyperbolic,
    _load_checkpoint,
    _load_normalizer,
)

hyperbolic_model, hyperbolic_config = _load_checkpoint(
    str(HYPERBOLIC_CHECKPOINT),
    DEVICE,
)

hyperbolic_mu, hyperbolic_sigma = _load_normalizer(
    str(HYPERBOLIC_NORMALIZER),
    DEVICE,
)

with torch.inference_mode():

    hyperbolic_embeddings = _encode_hyperbolic(
        model=hyperbolic_model,
        inputs=input_matrix,
        device=DEVICE,
        mu=hyperbolic_mu,
        sigma=hyperbolic_sigma,
        batch_size=32,
    ).float()

curvature = float(
    hyperbolic_config['manifold_c']
)

ball_radius = (
    1.0 / math.sqrt(curvature)
)

assert torch.isfinite(
    hyperbolic_embeddings
).all()

assert (
    hyperbolic_embeddings.norm(dim=-1)
    < ball_radius
).all()

hyperbolic_results = evaluate_four_metrics(
    taxonomy=taxonomy,
    embeddings=hyperbolic_embeddings,
    geometry='poincare',
    curvature=curvature,
    calibration_indices=calibration_indices,
    test_indices=test_indices,
)

print(
    'Hyperbolic latent shape:',
    tuple(hyperbolic_embeddings.shape),
)
print(
    'Hyperbolic curvature:',
    curvature,
)


# Free GPU memory before loading Euclidean LCM.
del hyperbolic_model
del hyperbolic_mu
del hyperbolic_sigma

gc.collect()
torch.cuda.empty_cache()

# EUCLIDEAN LCM
import importlib
import b_lcm

importlib.reload(b_lcm)

from b_lcm import BaseLCM


EUCLIDEAN_ARCH = {
    'embed_dim': 768,
    'model_dim': 256,
    'max_seq_len': 8,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.15,
    'post_dropout': 0.0,
}


euclidean_ckpt = torch.load(
    EUCLIDEAN_CHECKPOINT,
    map_location='cpu',
    weights_only=False,
)

state = euclidean_ckpt['model']


assert tuple(
    state['frontend.pre_linear.weight'].shape
) == (256, 768)

assert tuple(
    state['frontend.pos_emb.weight'].shape
) == (8, 256)

layer_ids = sorted({
    int(key.split('.')[3])
    for key in state
    if key.startswith('lcm.encoder.layers.')
})

assert layer_ids == [0, 1, 2, 3], layer_ids


euclidean_model = BaseLCM(
    **EUCLIDEAN_ARCH
)

euclidean_model.load_state_dict(
    state,
    strict=True,
)

euclidean_model.to(
    DEVICE
).eval()


# Verify checkpoint normalizer.
external_norm = torch.load(
    EUCLIDEAN_NORMALIZER,
    map_location='cpu',
    weights_only=False,
)

assert torch.allclose(
    state['normalizer.mu'].float(),
    external_norm['mu'].float(),
)

assert torch.allclose(
    state['normalizer.sigma'].float(),
    external_norm['sigma'].float(),
)

assert torch.allclose(
    state['normalizer.mu'].float(),
    state['frontend.normalizer.mu'].float(),
)

assert torch.allclose(
    state['normalizer.sigma'].float(),
    state['frontend.normalizer.sigma'].float(),
)


@torch.inference_mode()
def encode_euclidean_latents(
    model,
    inputs,
    device,
    batch_size=256,
):
    outputs = []

    model.eval()

    for start in range(
        0,
        len(inputs),
        batch_size,
    ):

        batch = inputs[
            start:start + batch_size
        ].to(
            device
        ).float()[:, None, :]

        frontend, padding_mask = model.frontend(
            batch,
            padding_mask=None,
        )

        hidden = model.lcm(
            frontend,
            padding_mask,
        )

        outputs.append(
            hidden[:, 0, :]
            .float()
            .cpu()
        )

    return torch.cat(
        outputs,
        dim=0,
    )


euclidean_embeddings = encode_euclidean_latents(
    model=euclidean_model,
    inputs=input_matrix,
    device=DEVICE,
    batch_size=256,
)

assert euclidean_embeddings.shape == (
    len(taxonomy.concept_ids),
    256,
)

assert torch.isfinite(
    euclidean_embeddings
).all()


euclidean_results = evaluate_four_metrics(
    taxonomy=taxonomy,
    embeddings=euclidean_embeddings,
    geometry='euclidean',
    calibration_indices=calibration_indices,
    test_indices=test_indices,
)

print(
    'Euclidean latent shape:',
    tuple(euclidean_embeddings.shape),
)


# Cleanup.
del euclidean_model
del euclidean_ckpt
del state
del external_norm

gc.collect()
torch.cuda.empty_cache()


comparison_table = pd.DataFrame(
    {
        'Metric': [
            'heldout_mean_relative_distortion',
            'heldout_spearman',
            'parent_mean_reciprocal_rank',
            'sibling_recall_at_5',
        ],

        'Euclidean LCM': [
            euclidean_results[
                'heldout_mean_relative_distortion'
            ],
            euclidean_results[
                'heldout_spearman'
            ],
            euclidean_results[
                'parent_mean_reciprocal_rank'
            ],
            euclidean_results[
                'sibling_recall_at_5'
            ],
        ],

        'Hyperbolic LCM': [
            hyperbolic_results[
                'heldout_mean_relative_distortion'
            ],
            hyperbolic_results[
                'heldout_spearman'
            ],
            hyperbolic_results[
                'parent_mean_reciprocal_rank'
            ],
            hyperbolic_results[
                'sibling_recall_at_5'
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
            'Euclidean LCM': '{:.4f}',
            'Hyperbolic LCM': '{:.4f}',
        }
    )
    .hide(axis='index')
)

print(
    '\nSaved table:',
    OUTPUT_TABLE,
)
