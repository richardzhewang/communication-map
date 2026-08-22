"""Neuron (MLP) nodes for v1: loading + the mixed dense edge classes.

Each neuron m in layer l is a rank-1 reader-writer pair (design doc S3.2):
pre-activation a_m = w_in . x (reader), contribution Delta x = GELU(a_m) w_out
(writer). TransformerLens (row conv): W_in [L, d, d_mlp] (pre = x @ W_in),
W_out [L, d_mlp, d] (out = act @ W_out). Column-conv vectors are therefore
W_in[l][:, m] (reader) and W_out[l][m, :] (writer); both stacks are [L, d_mlp, d]
with ROWS as vectors. fold_ln folds ln2's gain into W_in and
center_writing_weights centers the W_out writers -- the same D6 processing as
the head factors. Biases are ignored (potential-connectivity geometry only).

v1 edge classes involving neurons (writer -> reader; sublayer-causal masks,
D6ii: within layer l attention writes BEFORE the MLP reads, and the MLP writes
after everything in its layer):

    head_neuron          OV of (l_w, h) -> w_in of (l_r, m)   l_w <= l_r  (== legal!)
    neuron_head_{K,Q,V}  w_out -> F / F^T / M of later head   l_w <  l_r
    emb_neuron, pos_neuron   interface-matrix writers -> w_in         always
    neuron_unembed       w_out -> W_U                         always
    neuron_neuron        w_out -> w_in  (stream.py, tiled)    l_w <  l_r

All statistics are the D1 sigma-weighted norm C = ||R W||_F/(||R||_F ||W||_F),
which collapses when one side is rank-1: head->neuron C = ||M^T w^||_2/||M||_F,
neuron->head_K C = ||F w^||_2/||F||_F, neuron->neuron C = |cos(w_out, w_in)|.
(sigma_1 == Frobenius for every rank-1-sided class, so the A2 companion
statistic is meaningful only for the head-head families and is not computed
here.) All quadratic forms are evaluated through the [64, 64] factor Grams --
never a [d, d] product per pair.

Nulls (v0.1 lessons applied):
  head_neuron / neuron_head_*  bulk empirical null per span stratum (sparse
                               signal assumed; CHECK density on first run --
                               the R4 artifact showed what dense signal does
                               to a bulk null).
  emb/pos_neuron, neuron_unembed  rotation null (A1). Rank-1 readers/writers
      make this EXACT and cheap: for unit w and Haar Q, the null statistic
      C^2 = w^T Q Hb Q^T w = sum_i lambda_i u_i^2 with u uniform on S^{d-1}
      depends only on the eigenvalues of Hb -- one shared null distribution
      per interface matrix, sampled directly on the sphere (no Q draws, no
      conjugations). Moment-fit z as in A1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import D_MODEL, N_HEADS, N_LAYERS
from .edges import _small_grams, interface_grams
from .weights import Weights

D_MLP = 4 * D_MODEL  # 3072


@dataclass
class NeuronWeights:
    """Column-convention neuron vectors, float64, rows = vectors."""

    W_in_r: np.ndarray   # [L, d_mlp, d]  reader vectors (rows of W_in^T)
    W_out_w: np.ndarray  # [L, d_mlp, d]  writer vectors (rows of W_out)


def load_neurons(model) -> NeuronWeights:
    """Extract neuron read/write stacks from a processed HookedTransformer
    (weights.load_model). Cheap views -> float64 copies."""
    import torch

    with torch.no_grad():
        W_in = model.W_in.double().numpy()    # [L, d, d_mlp] (row conv)
        W_out = model.W_out.double().numpy()  # [L, d_mlp, d]
    nw = NeuronWeights(W_in_r=W_in.transpose(0, 2, 1).copy(), W_out_w=W_out.copy())
    assert nw.W_in_r.shape == (N_LAYERS, D_MLP, D_MODEL), nw.W_in_r.shape
    assert nw.W_out_w.shape == (N_LAYERS, D_MLP, D_MODEL), nw.W_out_w.shape
    return nw


def partial_neurons(nw: NeuronWeights, Vk: np.ndarray) -> NeuronWeights:
    """D3 partialling for neuron vectors: project the top-k global directions
    (partial.global_directions) out of every reader and writer row.
    Normalization happens inside the statistics, so no renorm here."""
    proj = lambda X: X - (X @ Vk) @ Vk.T
    return NeuronWeights(W_in_r=proj(nw.W_in_r), W_out_w=proj(nw.W_out_w))


# ---------------------------------------------------------------- helpers


def _unit_rows(X: np.ndarray) -> np.ndarray:
    """Unit-normalize the last axis; zero rows stay zero."""
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(n, 1e-30)


def neuron_labels(l: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add("L", l.astype(str)), np.char.add("N", m.astype(str)))


# ------------------------------------------------- mixed dense edge classes
#
# Each class is computed as a [144, N_tot] (or [N_tot]) stat array, masked by
# the sublayer-causal rule, nulled, BH'd within the family, and REDUCED TO
# SURVIVORS immediately (design S8: keep only survivors; full tables of 1e7
# rows are written only on request).


def _quad_stat(Wv: np.ndarray, inner: np.ndarray, small: np.ndarray, tr: float) -> np.ndarray:
    """C = sqrt(w^ ^T [inner small inner^T] w^ / tr) for unit rows Wv [N, d]:
    B = Wv @ inner [N, r]; C^2 = rowsum((B @ small) * B) / tr."""
    B = Wv @ inner
    return np.sqrt(np.clip(np.einsum("nr,nr->n", B @ small, B), 0.0, None) / tr)


def mixed_head_neuron_stats(w: Weights, nw: NeuronWeights) -> dict[str, np.ndarray]:
    """Stat matrices [144 heads, L*d_mlp neurons] for head_neuron and the three
    neuron_head slots. Masks are applied later (cheap boolean arrays)."""
    sg = _small_grams(w)  # GQ, GK, GV, GO  [144, 64, 64] float64
    Win = _unit_rows(nw.W_in_r).reshape(-1, D_MODEL)    # [N_tot, d]
    Wout = _unit_rows(nw.W_out_w).reshape(-1, D_MODEL)  # [N_tot, d]
    n_heads = N_LAYERS * N_HEADS
    O = w.O.reshape(n_heads, D_MODEL, -1)
    K = w.K.reshape(n_heads, D_MODEL, -1)
    Q = w.Q.reshape(n_heads, D_MODEL, -1)
    V = w.V.reshape(n_heads, D_MODEL, -1)
    tr_H = np.einsum("hij,hij->h", sg["GV"], sg["GO"])   # tr(M M^T) per head
    tr_GK = np.einsum("hij,hij->h", sg["GQ"], sg["GK"])  # tr(F^T F) = tr(F F^T)
    out = {k: np.empty((n_heads, Win.shape[0])) for k in
           ("head_neuron", "neuron_head_K", "neuron_head_Q", "neuron_head_V")}
    for h in range(n_heads):
        # head -> neuron: C^2 = w_in^T M M^T w_in / tr = w^T O GV O^T w / tr
        out["head_neuron"][h] = _quad_stat(Win, O[h], sg["GV"][h], tr_H[h])
        # neuron -> head, K slot: C^2 = w^T F^T F w / tr = w^T K GQ K^T w / tr
        out["neuron_head_K"][h] = _quad_stat(Wout, K[h], sg["GQ"][h], tr_GK[h])
        # Q slot: reader F^T -> C^2 = w^T F F^T w / tr = w^T Q GK Q^T w / tr
        out["neuron_head_Q"][h] = _quad_stat(Wout, Q[h], sg["GK"][h], tr_GK[h])
        # V slot: reader M -> C^2 = w^T M^T M w / tr = w^T V GO V^T w / tr
        out["neuron_head_V"][h] = _quad_stat(Wout, V[h], sg["GO"][h], tr_H[h])
    return out


def interface_neuron_stats(w: Weights, nw: NeuronWeights) -> dict[str, np.ndarray]:
    """[N_tot] stats: emb_neuron, pos_neuron (readers w_in), neuron_unembed
    (writers w_out)."""
    bg = interface_grams(w)
    Win = _unit_rows(nw.W_in_r).reshape(-1, D_MODEL)
    Wout = _unit_rows(nw.W_out_w).reshape(-1, D_MODEL)

    def quad(Wv, Gb):
        Gb = Gb.astype(np.float64)
        return np.sqrt(np.clip(np.einsum("nd,nd->n", Wv @ Gb, Wv), 0, None) / np.trace(Gb))

    return {
        "emb_neuron": quad(Win, bg["H_emb"]),
        "pos_neuron": quad(Win, bg["H_pos"]),
        "neuron_unembed": quad(Wout, bg["G_unemb"]),
    }


def sphere_null_moments(Gb: np.ndarray, n_samp: int = 100_000, seed: int = 0
                        ) -> tuple[float, float]:
    """EXACT rotation null for rank-1-vs-Gram classes, sampled directly:
    C^2 = sum_i lambda_i u_i^2, u uniform on S^{d-1} -- reader-independent.
    Returns (mean, sd) of C for the A1-style moment-fit z."""
    lam = np.clip(np.linalg.eigvalsh(Gb.astype(np.float64)), 0, None)  # [d]
    rng = np.random.default_rng(seed)
    g2 = rng.standard_normal((n_samp, lam.size)) ** 2
    c = np.sqrt((g2 @ lam) / (g2.sum(axis=1) * lam.sum()))
    return float(c.mean()), float(c.std())


def mixed_class_tables(
    w: Weights, nw: NeuronWeights, q_level: float = 0.05,
    n_samp_rot: int = 100_000, seed: int = 0, keep_full: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """All mixed classes end-to-end: stats -> nulls -> per-family BH ->
    survivor tables. Returns ({cls: survivors_df}, family_summary_df).

    Survivor rows: [cls, writer, l_w, reader, l_r, stat, z, p, q]. If
    keep_full, the full table is returned per class instead (memory: ~1e7
    rows total -- fine on 64 GB, avoid on laptops).
    """
    from .fdr import bh_qvalues

    hh = mixed_head_neuron_stats(w, nw)
    bb = interface_neuron_stats(w, nw)
    bg = interface_grams(w)
    head_layer = np.repeat(np.arange(N_LAYERS), N_HEADS)          # [144]
    head_lab = np.array([f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)])
    neu_layer = np.repeat(np.arange(N_LAYERS), D_MLP)             # [N_tot]
    neu_idx = np.tile(np.arange(D_MLP), N_LAYERS)
    neu_lab = neuron_labels(neu_layer, neu_idx)

    out: dict[str, pd.DataFrame] = {}
    fam_rows = []

    def finish(cls, stat, z, p, wl, wn, rl, rn, span):
        q = bh_qvalues(p)
        sig = q <= q_level
        fam_rows.append(dict(cls=cls, n_edges=len(stat), n_sig=int(sig.sum()),
                             max_stat=float(stat.max()), max_z=float(z.max())))
        keep = np.ones_like(sig) if keep_full else sig
        out[cls] = pd.DataFrame({
            "cls": cls, "writer": wn[keep], "l_w": wl[keep],
            "reader": rn[keep], "l_r": rl[keep], "stat": stat[keep],
            "z": z[keep], "p": p[keep], "q": q[keep], "sig": sig[keep],
        }).sort_values("q").reset_index(drop=True)

    # ---- head <-> neuron classes: bulk null per span stratum
    for cls, S in hh.items():
        to_neuron = cls == "head_neuron"
        if to_neuron:  # writer head, reader neuron: l_w <= l_r (D6ii)
            mask = head_layer[:, None] <= neu_layer[None, :]
            wl, rl = head_layer[:, None], neu_layer[None, :]
            wn, rn = head_lab[:, None], neu_lab[None, :]
        else:          # writer neuron (axis 1!), reader head: l_w < l_r
            mask = neu_layer[None, :] < head_layer[:, None]
            wl, rl = neu_layer[None, :], head_layer[:, None]
            wn, rn = neu_lab[None, :], head_lab[:, None]
        wl, rl = np.broadcast_to(wl, S.shape)[mask], np.broadcast_to(rl, S.shape)[mask]
        wn, rn = np.broadcast_to(wn, S.shape)[mask], np.broadcast_to(rn, S.shape)[mask]
        stat, span = S[mask], (rl - wl)
        med = np.zeros_like(stat)
        sd = np.zeros_like(stat)
        for s in np.unique(span):  # bulk fit per span (Efron central matching)
            i = span == s
            m = np.median(stat[i])
            med[i] = m
            sd[i] = np.maximum(1.4826 * np.median(np.abs(stat[i] - m)), 1e-12)
        z = (stat - med) / sd
        p = sps.norm.sf(z)  # one-sided upper: stat in [0, 1]
        finish(cls, stat, z, p, wl, wn, rl, rn, span)

    # ---- interface-matrix classes: exact spherical rotation null (shared, A1)
    rot_cfg = {
        "emb_neuron": ("EMB", bg["H_emb"], neu_layer, neu_lab, True),
        "pos_neuron": ("POS", bg["H_pos"], neu_layer, neu_lab, True),
        "neuron_unembed": ("UNEMB", bg["G_unemb"], neu_layer, neu_lab, False),
    }
    for i, (cls, (bname, Gb, nl, nn, neuron_is_reader)) in enumerate(rot_cfg.items()):
        stat = bb[cls]
        mean, sd = sphere_null_moments(Gb, n_samp_rot, seed + i)
        z = (stat - mean) / max(sd, 1e-12)
        p = sps.norm.sf(z)
        if neuron_is_reader:
            wl, wn = np.full(len(stat), -1), np.full(len(stat), bname, dtype=object)
            rl, rn = nl, nn
        else:
            wl, wn = nl, nn
            rl, rn = np.full(len(stat), N_LAYERS), np.full(len(stat), bname, dtype=object)
        finish(cls, stat, z, p, wl, np.asarray(wn), rl, np.asarray(rn), None)

    fam = pd.DataFrame(fam_rows)
    fam["null_kind"] = np.where(fam["cls"].isin(list(rot_cfg)), "rotation", "bulk")
    return out, fam
