"""commap: the residual-stream communication map (v0).

v0 scope (design doc S12, D5): attention heads + interface-matrix nodes (embedding,
positional, unembedding) for GPT-2 small. ~6e4 edges, full statistics pipeline:

    weights.py   load TransformerLens-processed weights; convention guards (D6)
    nodes.py     per-head decompositions: SVD of W_QK, SVD+eig of W_OV
    edges.py     sigma-weighted composition statistics, all classes (D1)
    nulls.py     empirical per-stratum nulls; isotropic baseline (D2)
    fdr.py       Benjamini-Hochberg per edge family (D4)
    partial.py   partialled map: project out global directions (D3)
    recovery.py  pre-registered recovery acceptance test (D5)

Convention (core-research-explainer.tex S2): COLUMN vectors; W_QK := W_Q^T W_K
acts as the bilinear form x_i^T W_QK x_j; W_OV := W_O W_V acts as
Delta x = W_OV x. TransformerLens stores row-convention factors; weights.py
does the conversion once, with assertions, so every other module lives purely
in the memo's convention.
"""

__version__ = "0.1.0"

D_MODEL = 768
D_HEAD = 64
N_LAYERS = 12
N_HEADS = 12
