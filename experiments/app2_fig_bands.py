"""Paper figure fig_bands.png: eigenvalue spectrum of the pooled second
moment, k=5 marked. Replaces the old three-panel version (its shape-taxonomy
panels were retired 2026-08-11).

Run: uv run --with matplotlib python experiments/s3c_fig_bands.py
"""
import sys

sys.path.insert(0, "experiments/lib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from commap.partial import global_directions
from commap.weights import load_gpt2

_, vals = global_directions(load_gpt2(), 10)

fig, ax = plt.subplots(figsize=(4.6, 3.2))
ax.plot(range(10), vals[:10], "o-", color="#4C72B0", lw=1.6, ms=6)
ax.set_xlabel("band # (0-indexed)")
ax.set_ylabel("eigenvalue")
ax.grid(color="0.92", lw=0.6)
ax.set_axisbelow(True)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig("results/figures/fig_bands.png", dpi=200, bbox_inches="tight")
print("saved results/figures/fig_bands.png")
