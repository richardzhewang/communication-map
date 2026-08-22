"""Edge statistics: the sigma-weighted composition norm (decision D1).

For a writer with operator W (column conv, writes W x into the stream) and a
reader with matrix R (reads through R x), the edge statistic is

    C = ||R W||_F / (||R||_F ||W||_F)   in [0, 1]

(core-research-explainer.tex S3.5: ||R W||_F^2 = sum_{k,l} sigma_k^2 s_l^2
cos^2(angle) -- the projector statistic with singular values kept; no
truncation choice).

Computed via the Gram trick (never form the [d,d] product per pair):

    ||R W||_F^2 = tr(W^T R^T R W) = tr( G H ),   G := R^T R,  H := W W^T,
    tr(G H) = sum_ij G_ij H_ij   (both symmetric [d, d])

so all pairs in a class are one flat matmul: [n_readers, d^2] @ [d^2, n_writers].

Edge classes in v0 (writer -> reader; sublayer-causal masks, D6):

    head_head_K    OV of (l_w, h_w) -> key slot of (l_r, h_r):    R = F,   need l_w < l_r
    head_head_Q    OV -> query slot:                              R = F^T, need l_w < l_r
    head_head_V    OV -> OV input of later head:                  R = M,   need l_w < l_r
    emb_head_{K,Q,V}, pos_head_{K,Q,V}   interface-matrix writers (always earlier)
    head_unembed   OV -> unembedding readout:                     R = W_U (always later)

Reader Grams via low-rank factors (F = Q K^T, M = O V^T, all factors [d, 64]):
    G_K = F^T F = K (Q^T Q) K^T        G_Q = F F^T = Q (K^T K) Q^T
    G_V = M^T M = V (O^T O) V^T        H   = M M^T = O (V^T V) O^T
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import N_HEADS, N_LAYERS
from .weights import Weights

# ---------------------------------------------------------------- gram helpers


def _sandwich(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """outer [d, r], inner [d, r] -> outer (inner^T inner) outer^T  [d, d]."""
    small = inner.T @ inner            # [r, r]
    return outer @ small @ outer.T     # [d, d]


def head_grams(w: Weights) -> dict[str, np.ndarray]:
    """All per-head Gram matrices, stacked [L*H, d, d] float32 (memory: D1
    statistics are O(1) ratios of ~6e5-term sums; float32 is ample -- the
    float64 requirement (D6) applies to the *decompositions* in nodes.py)."""
    L, H = N_LAYERS, N_HEADS
    G_K, G_Q, G_V, H_ov = [], [], [], []
    for l in range(L):
        for h in range(H):
            Q, K = w.Q[l, h], w.K[l, h]   # [d, 64]
            V, O = w.V[l, h], w.O[l, h]   # [d, 64]
            G_K.append(_sandwich(K, Q))   # F^T F
            G_Q.append(_sandwich(Q, K))   # F F^T
            G_V.append(_sandwich(V, O))   # M^T M
            H_ov.append(_sandwich(O, V))  # M M^T
    stack = lambda ms: np.stack(ms).astype(np.float32)  # [144, d, d]
    return {"G_K": stack(G_K), "G_Q": stack(G_Q), "G_V": stack(G_V), "H_ov": stack(H_ov)}


def interface_grams(w: Weights) -> dict[str, np.ndarray]:
    """Writer Grams for EMB/POS, reader Gram for UNEMB. [d, d] float32 each."""
    return {
        "H_emb": (w.W_E @ w.W_E.T).astype(np.float32),      # [d, d]
        "H_pos": (w.W_pos @ w.W_pos.T).astype(np.float32),  # [d, d]
        "G_unemb": (w.W_U.T @ w.W_U).astype(np.float32),    # [d, d]
    }


# ---------------------------------------------------------------- pair stats


def _small_grams(w: Weights) -> dict[str, np.ndarray]:
    """Per-head [64, 64] Grams of the factor matrices, float64. [144, 64, 64]"""
    fold = lambda X: X.reshape(-1, *X.shape[2:])                    # [144, d, 64]
    g = lambda X: np.matmul(fold(X).transpose(0, 2, 1), fold(X))    # X^T X, BLAS
    return {"GQ": g(w.Q), "GK": g(w.K), "GV": g(w.V), "GO": g(w.O)}


def _pair_stats(G: np.ndarray, Hm: np.ndarray) -> np.ndarray:
    """G [n_r, d, d], Hm [n_w, d, d] -> normalized C [n_r, n_w] in [0, 1]."""
    n_r, d, _ = G.shape
    n_w = Hm.shape[0]
    tr_G = np.einsum("rii->r", G)                     # [n_r] = ||R||_F^2
    tr_H = np.einsum("wii->w", Hm)                    # [n_w] = ||W||_F^2
    raw = G.reshape(n_r, d * d) @ Hm.reshape(n_w, d * d).T  # [n_r, n_w] = ||RW||_F^2
    np.maximum(raw, 0.0, out=raw)                     # clip fp round-off
    # float64 out: downstream z/p columns must hold float64 (pandas 2.x
    # refuses lossy float64 -> float32 overwrites in the interface-null pass)
    return np.sqrt(raw.astype(np.float64) /
                   (tr_G.astype(np.float64)[:, None] * tr_H.astype(np.float64)[None, :]))


def _head_labels() -> tuple[list[str], np.ndarray]:
    labels = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]
    layers = np.repeat(np.arange(N_LAYERS), N_HEADS)  # [144]
    return labels, layers


def compute_edges(w: Weights) -> pd.DataFrame:
    """All v0 edge statistics. Returns one row per candidate edge:
    columns [cls, writer, l_w, reader, l_r, stat]."""
    hg = head_grams(w)
    bg = interface_grams(w)
    labels, layers = _head_labels()
    frames: list[pd.DataFrame] = []

    def head_head(cls: str, G: np.ndarray) -> None:
        C = _pair_stats(G, hg["H_ov"])                # [n_r=144, n_w=144]
        r_idx, w_idx = np.meshgrid(np.arange(144), np.arange(144), indexing="ij")
        mask = layers[w_idx] < layers[r_idx]          # strict: parallel heads don't compose
        frames.append(
            pd.DataFrame(
                {
                    "cls": cls,
                    "writer": np.array(labels)[w_idx[mask]],
                    "l_w": layers[w_idx[mask]],
                    "reader": np.array(labels)[r_idx[mask]],
                    "l_r": layers[r_idx[mask]],
                    "stat": C[mask],
                }
            )
        )

    def interface_writer(cls: str, G: np.ndarray, Hb: np.ndarray, wname: str) -> None:
        C = _pair_stats(G, Hb[None])                  # [144, 1]
        frames.append(
            pd.DataFrame(
                {
                    "cls": cls, "writer": wname, "l_w": -1,
                    "reader": labels, "l_r": layers, "stat": C[:, 0],
                }
            )
        )

    head_head("head_head_K", hg["G_K"])
    head_head("head_head_Q", hg["G_Q"])
    head_head("head_head_V", hg["G_V"])
    for slot, G in [("K", hg["G_K"]), ("Q", hg["G_Q"]), ("V", hg["G_V"])]:
        interface_writer(f"emb_head_{slot}", G, bg["H_emb"], "EMB")
        interface_writer(f"pos_head_{slot}", G, bg["H_pos"], "POS")

    C = _pair_stats(bg["G_unemb"][None], hg["H_ov"])  # [1, 144]
    frames.append(
        pd.DataFrame(
            {
                "cls": "head_unembed", "writer": labels, "l_w": layers,
                "reader": "UNEMB", "l_r": N_LAYERS, "stat": C[0, :],
            }
        )
    )
    return pd.concat(frames, ignore_index=True)
