"""Figure 2 of the paper: (left) head-to-head adjacency under C, layer 0 at
the lower left; (right) Louvain communities of the head graph as a
layer x head grid, read from the archived seed-0 assignment
(results/app1/head_communities_seed0.json).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

e = pd.read_csv("results/app1/gpt2/edges_heads.csv.gz")
hh = e[e.cls.str.startswith("head_head") & e.sig].copy()
idx = lambda n: int(n[1:n.index("H")]) * 12 + int(n[n.index("H") + 1:])
hh["wi"] = hh.writer.map(idx)
hh["ri"] = hh.reader.map(idx)

A = np.zeros((144, 144))
for _, r in hh.iterrows():
    A[r.wi, r.ri] += 1
n_edges = len(hh)

assign = json.load(open(
    "results/app1/head_communities_seed0.json"))
memb = np.full(144, -1)
for h, ci in assign.items():
    memb[idx(h)] = ci
n_comm = memb.max() + 1
sizes = [(memb == ci).sum() for ci in range(n_comm)]

IND = [idx(x) for x in ["L5H1", "L5H5", "L6H9", "L7H2", "L7H10"]]
NM = [idx(x) for x in ["L9H6", "L9H9", "L10H0"]]
SI = [idx(x) for x in ["L7H3", "L7H9", "L8H6", "L8H10"]]
ind_ci = int(np.bincount([memb[h] for h in IND]).argmax())

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9),
                         gridspec_kw={"width_ratios": [1.12, 1.0]})

# ---- left: adjacency ----------------------------------------------------
ax = axes[0]
cmap = ListedColormap(["#ffffff", "#DD8452", "#8172B3", "#3A2D59"])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
im = ax.imshow(A.T, origin="lower", cmap=cmap, norm=norm,
               interpolation="nearest", aspect="auto")
ax.set_xlabel("writer head (L0H0 $\\rightarrow$ L11H11)")
ax.set_ylabel("reader head (L0H0 $\\rightarrow$ L11H11)")
for l in range(1, 12):
    ax.axhline(12 * l - 0.5, color="0.85", lw=0.4)
    ax.axvline(12 * l - 0.5, color="0.85", lw=0.4)
t = np.arange(0, 144, 24)
ax.set_xticks(t); ax.set_xticklabels([f"L{i//12}" for i in t])
ax.set_yticks(t); ax.set_yticklabels([f"L{i//12}" for i in t])
ax.set_title(f"significant head$\\to$head channels under $C$ (n={n_edges})")
cb = fig.colorbar(im, ax=ax, fraction=0.046, ticks=[0, 1, 2, 3])
cb.set_label("channels significant")

# ---- right: community grid ---------------------------------------------
ax = axes[1]
rest = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3", "#937860"]
colors, ri = {}, 0
for ci in range(n_comm):
    if ci == ind_ci:
        colors[ci] = "#55A868"
    else:
        colors[ci] = rest[ri]; ri += 1
grid = np.full((12, 12), -1)
for h in range(144):
    grid[h // 12, h % 12] = memb[h]
gcmap = ListedColormap(["#ffffff"] + [colors[ci] for ci in range(n_comm)])
ax.imshow(grid + 1, origin="lower", cmap=gcmap, vmin=0, vmax=n_comm,
          interpolation="nearest")
for k in range(13):
    ax.axhline(k - 0.5, color="w", lw=1.0)
    ax.axvline(k - 0.5, color="w", lw=1.0)
for hs, mk, sz in [(IND, "*", 190), (NM, "o", 62), (SI, "s", 54)]:
    for h in hs:
        ax.scatter(h % 12, h // 12, marker=mk, s=sz, facecolor="none",
                   edgecolor="k", linewidths=1.4, zorder=3)
ax.set_xlabel("head index within layer")
ax.set_ylabel("layer")
ax.set_xticks(range(0, 12, 2))
ax.set_yticks(range(0, 12, 2))
ax.set_title("Louvain communities are vertical")
handles = [Patch(color=colors[ci], label=f"community {ci} (n={sizes[ci]})")
           for ci in range(n_comm)]
handles += [
    plt.scatter([], [], marker="*", s=150, facecolor="none", edgecolor="k",
                label="induction"),
    plt.scatter([], [], marker="o", s=50, facecolor="none", edgecolor="k",
                label="name-mover"),
    plt.scatter([], [], marker="s", s=46, facecolor="none", edgecolor="k",
                label="S-inhibition"),
]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
          fontsize=8, frameon=False)

fig.tight_layout()
fig.savefig("results/figures/fig_map_overview.png", dpi=200, bbox_inches="tight")
print("edges", n_edges, "| communities", n_comm, "| sizes", sizes,
      "| induction in comm", ind_ci, ":",
      sum(memb[h] == ind_ci for h in IND), "/5")
