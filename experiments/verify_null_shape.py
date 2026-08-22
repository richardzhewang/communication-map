"""Empirical shape of a head pair's rotation null distribution.

The census (Table 2) standardizes each head pair's C^2 by the exact
closed-form moments of its rotation null distribution and reports the
shares beyond +/-2 SD; nothing there assumes a shape for that
distribution. This check samples one representative pair's null
directly, 4,000 Haar rotations of the writer's operator scored by
T = tr(G Q H Q^T) and standardized by the closed-form moments,
and records the shape: for rank-64 operators the null is a
near-symmetric bump with genuine mass in BOTH tails, unlike the
rank-one Beta law of the neuron classes (Figure 3), whose z floor
of about -0.7 makes z <= -2 unreachable. The sampled floor here
(the z of a perfectly orthogonal pair) sits tens of SDs below the
bump, so the below-chance regime of Table 2 is real territory.

Writes the appendix figure and a JSON of shape statistics.

Run: uv run python experiments/verify_null_shape.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from commap.weights import load_gpt2  # noqa: E402

N_ROT = 4000
SEED = 0
WRITER = (4, 7)    # L4H7, OV operator
READER = (8, 10)   # L8H10, QK operator (KQ-composition route)
OUT_JSON = "results/verification/null_shape_gpt2.json"
OUT_FIG = "results/figures/fig_null_z_hist.png"

w = load_gpt2()
d = w.Q.shape[2]
wl, wh = WRITER
rl, rh = READER
R = w.Q[rl, rh] @ w.K[rl, rh].T      # reader operator W_QK, rank <= d_head
W_ov = w.O[wl, wh] @ w.V[wl, wh].T   # writer operator W_OV, rank <= d_head
G = R.T @ R                          # reader Gram, d x d
H = W_ov @ W_ov.T                    # writer Gram, d x d

# closed-form rotation-null moments of T = tr(G Q H Q^T), as in map_build
mu = np.trace(G) * np.trace(H) / d
var = (2.0 / ((d - 1) * (d + 2))
       * (np.trace(G @ G) - np.trace(G) ** 2 / d)
       * (np.trace(H @ H) - np.trace(H) ** 2 / d))
sd = np.sqrt(var)

rng = np.random.default_rng(SEED)
z = np.empty(N_ROT)
for i in range(N_ROT):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    z[i] = (np.trace(G @ Q @ H @ Q.T) - mu) / sd

z_obs = (np.trace(G @ H) - mu) / sd    # the trained (unrotated) pair
z_floor = -mu / sd                     # a perfectly orthogonal pair

report = {
    "model": "gpt2", "writer": f"L{wl}H{wh}", "reader": f"L{rl}H{rh}",
    "route": "head_head_K/Q (W_OV vs W_QK)", "n_rot": N_ROT, "seed": SEED,
    "sample_mean_z": float(z.mean()), "sample_sd_z": float(z.std()),
    "skew": float(stats.skew(z)), "excess_kurtosis": float(stats.kurtosis(z)),
    "frac_below_-2": float((z <= -2).mean()),
    "frac_above_+2": float((z >= 2).mean()),
    "gaussian_tail": 0.02275,
    "min_z": float(z.min()), "max_z": float(z.max()),
    "z_trained": float(z_obs), "z_floor_orthogonal": float(z_floor),
}
Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
Path(OUT_JSON).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))

fig, ax = plt.subplots(figsize=(8.6, 3.9))
ax.hist(z, bins=60, density=True, color="#4C72B0", edgecolor="white",
        linewidth=0.4,
        label=f"sampled null distribution ({N_ROT:,} Haar rotations)")
xx = np.linspace(-4.5, 5.4, 400)
ax.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), color="#C44E52",
        lw=2, label="standard normal")
for v in (-2, 2):
    ax.axvline(v, color="0.4", ls=":", lw=1.4)
ax.text(-1.95, 0.395, "$z=-2$", fontsize=9, color="0.3", ha="right")
ax.text(2.05, 0.30, "$z=+2$", fontsize=9, color="0.3")
ax.annotate(f"trained pair: $z={z_obs:.1f}$",
            xy=(5.3, 0.012), xytext=(3.05, 0.135), fontsize=9,
            color="0.15",
            arrowprops=dict(arrowstyle="->", color="0.15", lw=1.1))
ax.text(-4.4, 0.30,
        f"orthogonal-pair floor\n$z={z_floor:.0f}$, far off-scale",
        fontsize=9, color="0.3")
ax.set_xlim(-4.6, 5.4)
ax.set_ylim(0, 0.42)
ax.set_xlabel(r"$z=(T-\mathbb{E}[T])/\mathrm{SD}[T]$ under Haar rotation "
              r"of the writer (closed-form moments)")
ax.set_ylabel("density")
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
print(f"saved {OUT_FIG}")
