"""Synthetic validation of the v1 machinery (no model, no GPU, seconds).

    uv run python experiments/test_pipeline_synthetic.py

T1  streaming histogram-BH == dense BH on the same binned p-values, exactly
    (survivor sets identical), on a small random neuron field with planted
    positive AND negative aligned pairs; planted pairs must be recovered.
T2  mixed-class factor-Gram statistics == naive dense ||R W||_F computation
    (1e-10), for head_neuron, neuron_head_{K,Q,V}, and interface-matrix classes.
T3  sphere-sampled rotation null: E[C^2] == tr(Gb)/d exactly (closed form);
    sampled mean must match within Monte Carlo error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "lib"))

from commap.fdr import bh_qvalues
from commap.neurons import NeuronWeights, _unit_rows, sphere_null_moments
from commap.stream import stream_neuron_neuron

rng = np.random.default_rng(0)
FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAIL
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    FAIL += not ok


# ------------------------------------------------------------------ T1
print("T1: streaming BH == dense BH, planted edges recovered")
L, N, d = 4, 300, 48
W_out = rng.standard_normal((L, N, d))
W_in = rng.standard_normal((L, N, d))
# plant: strong positive alignment (0,5) -> (2,7); strong negative (1,3) -> (3,9)
W_in[2, 7] = W_out[0, 5] + 0.1 * rng.standard_normal(d)
W_in[3, 9] = -W_out[1, 3] + 0.1 * rng.standard_normal(d)
nw = NeuronWeights(W_in_r=W_in, W_out_w=W_out)

res = stream_neuron_neuron(nw, q_level=0.05, n_bins=2001, device="cpu",
                           two_sided=True, verbose=False)
surv = res["survivors"]

# dense reference: same binned p assignment, ordinary BH over the full vector
Wo, Wi = _unit_rows(W_out), _unit_rows(W_in)
edges_b = res["edges"]
fits = {int(r.span): (r.med, r.sd) for r in res["span_fits"].itertuples()}
dense_keys, dense_p = [], []
for lw in range(L):
    for lr in range(lw + 1, L):
        c = np.clip(Wo[lw] @ Wi[lr].T, -1, 1)
        med, sd = fits[lr - lw]
        bins = np.clip(np.digitize(c, edges_b) - 1, 0, res["n_bins"] - 1)
        lo, hi = edges_b[bins], edges_b[bins + 1]
        z_lo, z_hi = (lo - med) / sd, (hi - med) / sd
        z_in = np.minimum(np.abs(z_lo), np.abs(z_hi))
        z_in[(z_lo < 0) & (z_hi > 0)] = 0.0
        p = 2.0 * sps.norm.sf(z_in)
        iw, ir = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
        dense_keys.append(np.stack([np.full(c.size, lw), iw.ravel(),
                                    np.full(c.size, lr), ir.ravel()], axis=1))
        dense_p.append(p.ravel())
dense_keys = np.concatenate(dense_keys)
dense_p = np.concatenate(dense_p)
dense_sig = bh_qvalues(dense_p) <= 0.05
dense_set = {tuple(k) for k in dense_keys[dense_sig]}
stream_set = {(r.l_w, r.m_w, r.l_r, r.m_r) for r in surv.itertuples()}

check("survivor sets identical", stream_set == dense_set,
      f"stream={len(stream_set)} dense={len(dense_set)} "
      f"symdiff={len(stream_set ^ dense_set)}")
check("planted positive edge recovered", (0, 5, 2, 7) in stream_set)
check("planted negative edge recovered (two-sided)", (1, 3, 3, 9) in stream_set)
check("m_total == masked pair count", res["m_total"] == 6 * N * N)

# ------------------------------------------------------------------ T2
print("T2: factor-Gram mixed statistics == naive dense")
from commap import D_HEAD, D_MODEL, N_HEADS, N_LAYERS
from commap.edges import interface_grams
from commap.neurons import interface_neuron_stats, mixed_head_neuron_stats
from commap.weights import Weights

Lh, H, dm, dh = N_LAYERS, N_HEADS, D_MODEL, D_HEAD
wts = Weights(
    Q=rng.standard_normal((Lh, H, dm, dh)) / dm**0.5,
    K=rng.standard_normal((Lh, H, dm, dh)) / dm**0.5,
    V=rng.standard_normal((Lh, H, dm, dh)) / dm**0.5,
    O=rng.standard_normal((Lh, H, dm, dh)) / dm**0.5,
    W_E=rng.standard_normal((dm, 500)), W_pos=rng.standard_normal((dm, 64)),
    W_U=rng.standard_normal((500, dm)),
)
n_small = 5  # a few neurons per layer suffice for the identity check
nw2 = NeuronWeights(W_in_r=rng.standard_normal((Lh, n_small, dm)),
                    W_out_w=rng.standard_normal((Lh, n_small, dm)))
stats = mixed_head_neuron_stats(wts, nw2)
Win_u = _unit_rows(nw2.W_in_r).reshape(-1, dm)
Wout_u = _unit_rows(nw2.W_out_w).reshape(-1, dm)

def naive(cls, h_idx, n_idx):
    l, hh = divmod(h_idx, H)
    F = wts.F(l, hh)
    M = wts.M(l, hh)
    if cls == "head_neuron":
        return np.linalg.norm(M.T @ Win_u[n_idx]) / np.linalg.norm(M)
    R = {"neuron_head_K": F, "neuron_head_Q": F.T, "neuron_head_V": M}[cls]
    return np.linalg.norm(R @ Wout_u[n_idx]) / np.linalg.norm(R)

ok, worst = True, 0.0
for cls in stats:
    for h_idx, n_idx in [(0, 0), (77, 31), (143, 59)]:
        err = abs(stats[cls][h_idx, n_idx] - naive(cls, h_idx, n_idx))
        worst = max(worst, err)
        ok &= err < 1e-10
check("head<->neuron identities", ok, f"max err {worst:.2e}")

bstats = interface_neuron_stats(wts, nw2)
bg = interface_grams(wts)
err = abs(bstats["emb_neuron"][7]
          - np.linalg.norm(wts.W_E.T @ Win_u[7]) / np.linalg.norm(wts.W_E))
err = max(err, abs(bstats["neuron_unembed"][12]
          - np.linalg.norm(wts.W_U @ Wout_u[12]) / np.linalg.norm(wts.W_U)))
check("interface-matrix identities", err < 1e-10, f"max err {err:.2e}")

# ------------------------------------------------------------------ T3
print("T3: sphere-sampled rotation null vs closed form")
Gb = bg["H_pos"].astype(np.float64)
mean, sd = sphere_null_moments(Gb, n_samp=200_000, seed=1)
lam = np.linalg.eigvalsh(Gb)
ex2 = np.trace(Gb) / dm / np.trace(Gb)          # E[C^2] = (tr/d)/tr = 1/d
mc2 = mean**2 + sd**2                           # E[C^2] from the sample
check("E[C^2] == 1/d", abs(mc2 - ex2) / ex2 < 0.01,
      f"mc {mc2:.6f} vs exact {ex2:.6f}")

print(f"\n{'ALL PASS' if FAIL == 0 else f'{FAIL} FAILURES'}")
sys.exit(1 if FAIL else 0)
