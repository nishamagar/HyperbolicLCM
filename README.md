# HyperbolicLCM
Implementation of Hyperbolic LCM (H-LCM), a geometry-aware concept-level reasoning model using hyperbolic embeddings (Poincaré ball), tangent-space attention, and Möbius updates to better capture hierarchical structure and improve multi-step reasoning tasks.

# Hyperbolic Large Concept Models (H-LCM)
This repository provides an implementation of Hyperbolic Large Concept Models (H-LCM) — a geometry-aware framework for concept-level reasoning that operates in hyperbolic space instead of traditional Euclidean embeddings.
Modern language models struggle with hierarchical reasoning, long-range dependencies, and structured abstraction. H-LCM addresses these limitations by projecting concept representations into the Poincaré ball, a hyperbolic space that naturally captures tree-like and hierarchical relationships.

# Key Idea
Instead of reasoning over tokens or flat embeddings, H-LCM:
- Transforms text into concept-level representations
- Maps them into hyperbolic space
- Performs reasoning using geometry-aware transformer layers
  
# Core Components
- Hyperbolic Embeddings (Poincaré Ball): Efficient representation of hierarchical structures
- Tangent-Space Attention: Stable transformer attention computed in Euclidean tangent space
- Möbius Residual Updates: Geometry-preserving updates on the manifold
- Hyperbolic Clustering Objective: Encourages structured semantic organization
These components allow H-LCM to reason over concepts in a space aligned with their inherent structure.

# Why Hyperbolic?
Euclidean embeddings flatten hierarchical relationships, leading to distortion and inefficiency. Hyperbolic space, with its exponential volume growth, provides a better inductive bias for representing complex concept hierarchies and multi-step reasoning.

# Results
H-LCM demonstrates:
- Strong performance on structured reasoning benchmarks
- Significant gains on multi-step and mathematical reasoning tasks
- Competitive efficiency despite operating in non-Euclidean space
