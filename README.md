# H-LCM: Hyperbolic Large Concept Models

This repository hosts the official implementation of **H-LCM (Hyperbolic Large Concept Model)**, introduced in:

> **Hyperbolic Large Concept Models: Geometry-Aware Hierarchical Reasoning Beyond Euclidean Space**

H-LCM extends Large Concept Models from Euclidean space to the **Poincaré ball manifold** to better capture hierarchical concept relationships. It combines **tangent-space attention**, **Möbius residual updates**, and a **margin-based hyperbolic clustering objective**. 

## Core Architecture

H-LCM performs concept-level reasoning in the **Poincaré ball** to better represent hierarchical relationships.

Euclidean concept embeddings are first projected into hyperbolic space. The model then applies **tangent-space multi-head attention** and a **hyperbolic feed-forward network**, while **Möbius residual updates** preserve the manifold geometry.

Training uses a **margin-based hyperbolic clustering objective** that pulls related concepts closer and pushes unrelated concepts farther apart.

The model operates in the **Poincaré ball** and uses Geoopt-based hyperbolic operations such as logarithmic/exponential maps, Möbius addition, projection, and manifold distance. 

## Main Configuration

```python
SEED          = 42
TOKEN_BUDGET  = 70_000_000
MODEL_DIM     = 4096
NUM_LAYERS    = 12
CURVATURE     = 0.002
LEARNING_RATE = 5e-5
MANIFOLD      = "PoincareBall"
```

Pretraining uses approximately **70M WikiText tokens** on an **NVIDIA RTX A6000 49GB GPU**. 

## Datasets

H-LCM is evaluated on: ARC-Easy, ARC-Challenge, OpenBookQA, CommonSenseQA, MMLU, BoolQ, MultiRC, MAWPS and GSM8K

## Key Results

### Supervised Fine-Tuning

| Dataset       | Euclidean LCM |     H-LCM |
| ------------- | ------------: | --------: |
| ARC-Challenge |         26.09 | **27.82** |
| OpenBookQA    |         22.20 | **32.20** |
| CommonSenseQA |         22.03 | **29.24** |
| MMLU          |         27.17 | **28.15** |

### Mathematical Reasoning

| Dataset | Euclidean LCM |     H-LCM |
| ------- | ------------: | --------: |
| MAWPS   |         10.99 | **35.77** |
| GSM8K   |         11.90 | **19.56** |

H-LCM achieves particularly strong results on structured mathematical reasoning tasks. 

## Training Efficiency

| Model         |       Throughput |
| ------------- | ---------------: |
| HELM          |        461 tok/s |
| Euclidean LCM |      8,969 tok/s |
| **H-LCM**     | **13,422 tok/s** |
