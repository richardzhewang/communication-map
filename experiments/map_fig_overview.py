"""The full communication map at layer resolution (overview figure).

Groups a model's components into 2L+3 nodes: the embedding (E), the
positional matrix (P), the heads of each layer (H0..H{L-1}), the neurons
of each layer (M0..M{L-1}), and the unembedding (U). Every one of the 18
connection classes is scored against its exact rotation null
distribution with the closed-form z of the census (mean 1/d, Weingarten
variance), the three head channels are pooled, and each writer-group x
reader-group block reports the median z and the share of pairs with
|z| >= 2. Causally ineligible blocks are left blank.

Usage: uv run python experiments/map_fig_overview.py [MODEL] [--out DIR]
Writes {out}/fig_map_layer_overview.png (net coupling per block),
{out}/fig_map_layer_overview_medianz.png, and map_layer_overview.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import map_build as mb  # noqa: E402

DEV = "cuda"


def inv_pair(g1, g2):
    """tr(g1 g2) and tr((g1 g2)^2) per head, float64 numpy [Nh]."""
    ab = np.matmul(g1, g2)
    return np.einsum("hii->h", ab), np.einsum("hij,hji->h", ab, ab)


def var_T(d, trG, trG2, trH, trH2):
    """Weingarten variance of T = tr(G Q H Q^T) under Haar Q."""
    return 2.0 / ((d - 1) * (d + 2)) * (trG2 - trG ** 2 / d) * (trH2 - trH ** 2 / d)


def z_of(C2, d, trG, trG2, trH, trH2):
    """Closed-form census z for C^2, given both sides' Gram invariants."""
    sd = np.sqrt(var_T(d, trG, trG2, trH, trH2)) / (trG * trH)
    return (C2 - 1.0 / d) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="gpt2")
    ap.add_argument("--out", default=str(REPO / "results" / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = mb.load_tl(args.model)
    W = mb.extract(model)
    del model
    torch.cuda.empty_cache()
    L, Hh, d, dh, dm = W["L"], W["H"], W["d"], W["dh"], W["d_mlp"]
    Nh, Nt = L * Hh, L * dm
    h_layer = np.repeat(np.arange(L), Hh)
    n_layer = np.repeat(np.arange(L), dm)
    parallel = args.model.startswith("pythia")

    # ---- per-head Gram invariants (float64) ---------------------------
    fold = lambda k: np.asarray(W[k], dtype=np.float64).reshape(Nh, d, dh)
    Qf, Kf, Vf, Of = fold("Q"), fold("K"), fold("V"), fold("O")
    g = lambda X: np.matmul(X.transpose(0, 2, 1), X)
    gQ, gK, gV, gO = g(Qf), g(Kf), g(Vf), g(Of)
    trQK, trQK2 = inv_pair(gQ, gK)
    trOV, trOV2 = inv_pair(gO, gV)
    one = np.ones(1)

    # ---- node layout: E, P, H0, M0, H1, M1, ..., U ----------------------
    nodes = ["E"] + (["P"] if W["learned_pos"] else [])
    for l in range(L):
        nodes += [f"H{l}", f"M{l}"]
    nodes += ["U"]
    N = len(nodes)
    idx = {n: i for i, n in enumerate(nodes)}
    med = np.full((N, N), np.nan)
    share = np.full((N, N), np.nan)
    above = np.full((N, N), np.nan)
    below = np.full((N, N), np.nan)
    count = np.zeros((N, N), dtype=np.int64)
    pools: dict = {}   # (writer node, reader node) -> list of z arrays

    def add(wn, rn, z):
        pools.setdefault((wn, rn), []).append(np.asarray(z, dtype=np.float64))

    # ---- head -> head (three channels pooled) -------------------------
    rows = []
    hh = mb.head_head(W, rows)
    for r in rows:
        lab = lambda s: np.array([int(x[1:x.index("H")]) * Hh + int(x[x.index("H") + 1:]) for x in s])
        ri, wi = lab(r["reader"]), lab(r["writer"])
        trG, trG2 = (trOV, trOV2) if r["cls"].endswith("V") else (trQK, trQK2)
        z = z_of(r["stat"] ** 2, d, trG[ri], trG2[ri], trOV[wi], trOV2[wi])
        for lw in range(L):
            for lr in range(lw + 1, L):
                m = (h_layer[wi] == lw) & (h_layer[ri] == lr)
                add(f"H{lw}", f"H{lr}", z[m])
    print("head-head done", flush=True)

    # ---- mixed classes --------------------------------------------------
    Rhat = mb._unit_rows_t(W["Win"].reshape(Nt, d)).to(DEV)
    What = mb._unit_rows_t(W["Wout"].reshape(Nt, d)).to(DEV)
    O = hh["O"]                                           # [Nh, d, dh] GPU
    Kd = W["K"].reshape(Nh, d, dh).to(DEV)
    Qd = W["Q"].reshape(Nh, d, dh).to(DEV)
    Vd = W["V"].reshape(Nh, d, dh).to(DEV)
    gVt = torch.from_numpy(gV).float().to(DEV)
    gQt = torch.from_numpy(gQ).float().to(DEV)
    gKt = torch.from_numpy(gK).float().to(DEV)
    gOt = torch.from_numpy(gO).float().to(DEV)

    def quad(vecs, proj, small, den):
        """C^2 for every (head, neuron): x = proj_h^T v_m, x^T small_h x / den_h."""
        out_ = np.empty((Nh, Nt), dtype=np.float64)
        blk = 8
        for i in range(0, Nh, blk):
            P = proj[i:i + blk]
            X = torch.matmul(vecs.unsqueeze(0), P)           # [b, Nt, dh]
            Y = torch.matmul(X, small[i:i + blk])
            qf = torch.einsum("bnd,bnd->bn", Y, X).double().cpu().numpy()
            out_[i:i + blk] = qf / den[i:i + blk, None]
            del P, X, Y
        return out_

    # head -> neuron: writer head (H_ov), reader neuron (rank one)
    C2 = quad(Rhat, O, gVt, trOV)
    z = (C2 - 1.0 / d) / (np.sqrt(var_T(d, one, one, trOV, trOV2)) / trOV)[:, None]
    for lw in range(L):
        for lr in range(lw + (1 if parallel else 0), L):
            add(f"H{lw}", f"M{lr}", z[h_layer == lw][:, n_layer == lr].ravel())
    print("head-neuron done", flush=True)
    # neuron -> head: writer neuron (rank one), reader head (K/Q/V)
    for proj, small, trG, trG2 in [(Kd, gQt, trQK, trQK2), (Qd, gKt, trQK, trQK2),
                                   (Vd, gOt, trOV, trOV2)]:
        C2 = quad(What, proj, small, trG)
        z = (C2 - 1.0 / d) / (np.sqrt(var_T(d, trG, trG2, one, one)) / trG)[:, None]
        for lw in range(L):
            for lr in range(lw + 1, L):
                add(f"M{lw}", f"H{lr}", z[h_layer == lr][:, n_layer == lw].ravel())
    print("neuron-head done", flush=True)

    # ---- interface matrices ---------------------------------------------
    ifs = {"E": W["H_emb"].double().numpy()}
    if W["learned_pos"]:
        ifs["P"] = W["H_pos"].double().numpy()
    GU = W["G_unemb"].double().numpy()
    inv1 = lambda M: (float(np.trace(M)), float(np.einsum("ij,ji->", M, M)))
    trGU, trGU2 = inv1(GU)
    GUt = torch.from_numpy(GU).float().to(DEV)
    for name, Hm in ifs.items():
        trH, trH2 = inv1(Hm)
        Ht = torch.from_numpy(Hm).float().to(DEV)
        # -> heads, three channels
        for proj, small, trG, trG2 in [(Kd, gQt, trQK, trQK2), (Qd, gKt, trQK, trQK2),
                                       (Vd, gOt, trOV, trOV2)]:
            S = torch.matmul(torch.matmul(proj.transpose(1, 2), Ht), proj)  # [Nh, dh, dh]
            num = torch.einsum("hij,hji->h", small.double(), S.double()).cpu().numpy()
            C2 = num / (trG * trH)
            z = z_of(C2, d, trG, trG2, trH, trH2)
            for lr in range(L):
                add(name, f"H{lr}", z[h_layer == lr])
        # -> neurons
        qf = torch.einsum("nd,nd->n", Rhat.double() @ Ht.double(), Rhat.double()).cpu().numpy()
        C2 = qf / trH
        z = z_of(C2, d, 1.0, 1.0, trH, trH2)
        for lr in range(L):
            add(name, f"M{lr}", z[n_layer == lr])
        del Ht
    # heads -> unembedding (reader G_U, writer H_ov)
    S = torch.matmul(torch.matmul(O.transpose(1, 2), GUt), O)
    num = torch.einsum("hij,hji->h", gVt.double(), S.double()).cpu().numpy()
    z = z_of(num / (trGU * trOV), d, trGU, trGU2, trOV, trOV2)
    for lw in range(L):
        add(f"H{lw}", "U", z[h_layer == lw])
    # neurons -> unembedding
    qf = torch.einsum("nd,nd->n", What.double() @ GUt.double(), What.double()).cpu().numpy()
    z = z_of(qf / trGU, d, trGU, trGU2, 1.0, 1.0)
    for lw in range(L):
        add(f"M{lw}", "U", z[n_layer == lw])
    print("interfaces done", flush=True)

    # ---- neuron -> neuron (tiles; z is monotone in cos^2) ---------------
    sd_nn = np.sqrt(2.0 * (d - 1) / (d ** 2 * (d + 2)))
    Wout = mb._unit_rows_t(W["Wout"]).to(DEV)
    Win = mb._unit_rows_t(W["Win"]).to(DEV)
    for lw in range(L):
        for lr in range(lw + 1, L):
            c2 = (Wout[lw] @ Win[lr].T).double() ** 2
            zmed = (float(c2.median()) - 1.0 / d) / sd_nn
            zz = (c2 - 1.0 / d) / sd_nn
            i, j = idx[f"M{lw}"], idx[f"M{lr}"]
            above[i, j] = float((zz >= 2).double().mean())
            below[i, j] = float((zz <= -2).double().mean())
            med[i, j], share[i, j], count[i, j] = zmed, above[i, j] + below[i, j], c2.numel()
            del c2, zz
    print("neuron-neuron done", flush=True)

    # ---- aggregate ------------------------------------------------------
    for (wn, rn), zs in pools.items():
        z = np.concatenate(zs)
        i, j = idx[wn], idx[rn]
        med[i, j] = float(np.median(z))
        above[i, j] = float((z >= 2).mean())
        below[i, j] = float((z <= -2).mean())
        share[i, j] = above[i, j] + below[i, j]
        count[i, j] = int(z.size)
    json.dump(dict(model=args.model, nodes=nodes,
                   median_z=np.where(np.isnan(med), None, med).tolist(),
                   share_abs_z_ge_2=np.where(np.isnan(share), None, share).tolist(),
                   share_above_2=np.where(np.isnan(above), None, above).tolist(),
                   share_below_2=np.where(np.isnan(below), None, below).tolist(),
                   count=count.tolist()),
              open(out / "map_layer_overview.json", "w"))

    # ---- plot -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    def grid(ax):
        ax.set_xticks(range(N)); ax.set_xticklabels(nodes, fontsize=6.5, rotation=90)
        ax.set_yticks(range(N)); ax.set_yticklabels(nodes, fontsize=6.5)
        off = N - 1 - 2 * L
        for k in range(L + 1):
            p = off + 2 * k - 0.5
            ax.axhline(p, color="k", lw=0.3, alpha=0.5)
            ax.axvline(p, color="k", lw=0.3, alpha=0.5)
        ax.set_facecolor("#e4e4e4")
        ax.set_xlabel("reader", fontsize=10); ax.set_ylabel("writer", fontsize=10)

    # main figure: net coupling (share super-coupled minus share avoidant)
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    im = ax.imshow((above - below) * 100, cmap="RdBu_r", vmin=-100, vmax=100)
    grid(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("share super-coupled ($z \\geq 2$) minus share avoidant ($z \\leq -2$), %",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_map_layer_overview.png", dpi=200)
    plt.close(fig)
    # companion: median z per block
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    vmax = 50.0
    im = ax.imshow(med, cmap="RdBu_r",
                   norm=SymLogNorm(linthresh=2.0, linscale=0.8, vmin=-vmax, vmax=vmax, base=10))
    grid(ax)
    ticks = [-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50]
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, ticks=ticks)
    cb.ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    cb.set_label("median $z$ against the rotation null", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_map_layer_overview_medianz.png", dpi=200)
    plt.close(fig)
    print(f"saved {out / 'fig_map_layer_overview.png'}")
    print("median z range", np.nanmin(med), np.nanmax(med))


if __name__ == "__main__":
    main()
