"""The partialled map (decision D3): project global/mechanical directions out
of every reader and writer, recompute all edges.

Finance framing: the raw map is a correlation network; this is the
partial-correlation network after removing the common factors. Topology
claims (hubs, heavy tails, small-world) must survive partialling.

v0 estimator of the global directions (simple, documented, replaceable):
eigenvectors of the trace-normalized pooled second-moment of ALL reader and
writer Grams. Each Gram is normalized to unit trace so no single component
dominates; the top-k eigendirections of the pool are the directions
'everything' reads/writes -- mechanical/rogue/positional candidates
(Timkey & van Schijndel 2021). k is chosen by inspection of the eigenvalue
spectrum plus the Ahn-Horenstein eigenvalue-ratio estimate (both reported);
full Bai-Ng / Onatski factor counting is design-doc S10.1 work, TODO v1.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .edges import interface_grams, head_grams
from .weights import Weights


def pooled_second_moment(w: Weights) -> np.ndarray:
    """Trace-normalized sum of all reader/writer Grams. [d, d] float64."""
    hg = head_grams(w)
    bg = interface_grams(w)
    d = hg["H_ov"].shape[-1]
    S = np.zeros((d, d))
    for stack in (hg["G_K"], hg["G_Q"], hg["G_V"], hg["H_ov"]):
        tr = np.einsum("nii->n", stack)                       # [144]
        S += np.einsum("nij->ij", stack / tr[:, None, None])  # unit-trace each
    for name in ("H_emb", "H_pos", "G_unemb"):
        Gb = bg[name].astype(np.float64)
        S += Gb / np.trace(Gb)
    return S


def ahn_horenstein(eigvals: np.ndarray, kmax: int = 20) -> int:
    """Eigenvalue-ratio estimate of the number of common factors:
    argmax_{1<=k<=kmax} eigvals_k / eigvals_{k+1} (eigvals descending)."""
    ratios = eigvals[:kmax] / eigvals[1 : kmax + 1]
    return int(np.argmax(ratios) + 1)


def global_directions(w: Weights, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k eigendirections of the pooled second moment.

    Returns (Vk [d, k], eigvals [d] descending)."""
    S = pooled_second_moment(w)
    vals, vecs = np.linalg.eigh(S)          # ascending
    vals, vecs = vals[::-1], vecs[:, ::-1]  # descending
    return vecs[:, :k], vals


def partialled_weights(w: Weights, k: int = 5) -> tuple[Weights, np.ndarray]:
    """Project the top-k global directions out of every factor matrix.

    F = Q K^T -> (P Q)(P K)^T = P F P, and likewise M -> P M P: projecting the
    factors projects both the reading and writing side of every component.
    Interface matrices are projected on their d-dimensional side.

    Returns (projected Weights, eigvals of the pooled second moment)."""
    Vk, vals = global_directions(w, k)       # [d, k], [d]
    P = np.eye(w.Q.shape[2]) - Vk @ Vk.T     # [d, d]
    proj = lambda X: np.einsum("de,...ef->...df", P, X)  # left-multiply last-2 dims
    w2 = replace(
        w,
        Q=proj(w.Q), K=proj(w.K), V=proj(w.V), O=proj(w.O),
        W_E=P @ w.W_E, W_pos=P @ w.W_pos, W_U=w.W_U @ P,
    )
    return w2, vals
