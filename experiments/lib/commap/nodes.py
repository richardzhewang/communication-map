"""Per-head node decompositions (core-research-explainer.tex S3), float64 on
CPU (D6).

For each head (l, h):
  - SVD of F = W_QK:  F = sum_k sigma_k u_k v_k^T
        u_k = q-side reader directions, v_k = k-side reader directions.
  - SVD of M = W_OV:  M = sum_k s_k a_k b_k^T
        a_k = writer directions (Col(M)), b_k = OV-reader directions.
  - Eigendecomposition of M (copying detection): eigenvalues lambda in C;
        copying score = sum_k Re(lambda_k) / sum_k |lambda_k|  in [-1, 1]
        (Elhage et al. 2021: near +1 => copying head; near -1 => anti-copying).

The edge statistics (edges.py) do NOT need these decompositions (the Gram
trick bypasses them); nodes.py exists to (a) characterize heads (copying
spectra, effective ranks), (b) name the bands of significant edges later
(top singular pairs of the cross matrices).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import N_HEADS, N_LAYERS
from .weights import Weights


def effective_rank(s: np.ndarray) -> float:
    """Entropy-based effective rank: exp(H(p)), p_k = s_k^2 / sum s^2.

    A truncation-free summary of how many channels a head really uses
    (reported descriptively; the D1 statistic needs no truncation).
    """
    p = s**2 / np.sum(s**2)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def copying_score(M: np.ndarray) -> float:
    """sum Re(lambda) / sum |lambda| of the [d, d] operator M (rank <= 64)."""
    lam = np.linalg.eigvals(M)  # complex [d]; ~ d - 64 are numerically zero
    denom = np.sum(np.abs(lam))
    return float(np.sum(lam.real) / denom) if denom > 0 else 0.0


def decompose_heads(w: Weights) -> pd.DataFrame:
    """One row per head with spectral summaries. [144 rows]"""
    rows = []
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            F = w.F(l, h)   # [d, d]
            M = w.M(l, h)   # [d, d]
            sF = np.linalg.svd(F, compute_uv=False)[:64]  # [64]
            sM = np.linalg.svd(M, compute_uv=False)[:64]  # [64]
            rows.append(
                dict(
                    layer=l,
                    head=h,
                    node=f"L{l}H{h}",
                    qk_sigma1=sF[0],
                    qk_frob=float(np.linalg.norm(sF)),
                    qk_eff_rank=effective_rank(sF),
                    ov_s1=sM[0],
                    ov_frob=float(np.linalg.norm(sM)),
                    ov_eff_rank=effective_rank(sM),
                    copying=copying_score(M),
                )
            )
    return pd.DataFrame(rows)
