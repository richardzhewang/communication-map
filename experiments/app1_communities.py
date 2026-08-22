"""Louvain communities of the head-to-head graph, as a layer x head grid.

Graph: undirected over 144 heads; pair linked if any of its K/Q/V channels
is FDR-significant under C (raw map); weight = number of significant
channels; giant connected component. Louvain seed 0 (networkx).

NOTE: recomputed 2026-08-11. The original 2026-08-08 partition was not
archived in full (only its induction community C_ind); this run uses the
same documented construction and is close but not bit-identical. The
assignment vector is saved this time.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

e = pd.read_csv("results/app1/gpt2/edges_heads.csv.gz")
hh = e[e.cls.str.startswith("head_head") & e.sig].copy()
idx = lambda n: int(n[1:n.index("H")]) * 12 + int(n[n.index("H") + 1:])
lab = lambda i: f"L{i//12}H{i%12}"

G = nx.Graph()
G.add_nodes_from(range(144))
for (w, r), c in hh.groupby([hh.writer.map(idx), hh.reader.map(idx)]).size().items():
    G.add_edge(w, r, w=G[w][r]["w"] + c if G.has_edge(w, r) else c)
gcc = G.subgraph(max(nx.connected_components(G), key=len))
comms = sorted(nx.community.louvain_communities(gcc, weight="w", seed=0),
               key=len, reverse=True)
Q = nx.community.modularity(gcc, comms, weight="w")

C_ind = {idx(h) for h in
         json.load(open("results/app1/demoB_v2.json"))["C_ind"]}
# order communities: put the C_ind-matching one at a fixed color (green)
overlap = [len(set(c) & C_ind) for c in comms]
ind_ci = int(np.argmax(overlap))

memb = np.full(144, -1)
for ci, c in enumerate(comms):
    for h in c:
        memb[h] = ci

json.dump({lab(h): int(memb[h]) for h in range(144) if memb[h] >= 0},
          open("results/app1/head_communities_seed0.json", "w"),
          indent=0)

# palette: green reserved for the induction community, old-figure hues for rest
rest = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3", "#937860"]
colors = {}
ri = 0
for ci in range(len(comms)):
    if ci == ind_ci:
        colors[ci] = "#55A868"
    else:
        colors[ci] = rest[ri]; ri += 1

grid = np.full((12, 12), -1)
for h in range(144):
    grid[h // 12, h % 12] = memb[h]
cmap = ListedColormap(["#ffffff"] + [colors[ci] for ci in range(len(comms))])

fig, ax = plt.subplots(figsize=(7.2, 6.0))
ax.imshow(grid + 1, origin="lower", cmap=cmap, vmin=0,
          vmax=len(comms), interpolation="nearest")
for k in range(13):
    ax.axhline(k - 0.5, color="w", lw=1.2)
    ax.axvline(k - 0.5, color="w", lw=1.2)

IND = [idx(x) for x in ["L5H1", "L5H5", "L6H9", "L7H2", "L7H10"]]
NM = [idx(x) for x in ["L9H6", "L9H9", "L10H0"]]
SI = [idx(x) for x in ["L7H3", "L7H9", "L8H6", "L8H10"]]
for hs, mk, sz in [(IND, "*", 210), (NM, "o", 70), (SI, "s", 60)]:
    for h in hs:
        ax.scatter(h % 12, h // 12, marker=mk, s=sz, facecolor="none",
                   edgecolor="k", linewidths=1.5, zorder=3)

ax.set_xlabel("head index within layer")
ax.set_ylabel("layer")
ax.set_xticks(range(12))
ax.set_yticks(range(12))
ax.set_title(f"Louvain communities of the head graph "
             f"(seed 0 recomputed, $Q={Q:.3f}$)")
handles = [Patch(color=colors[ci],
                 label=f"community {ci} (n={len(comms[ci])})"
                 + ("  [induction]" if ci == ind_ci else ""))
           for ci in range(len(comms))]
handles += [plt.scatter([], [], marker="*", s=160, facecolor="none",
                        edgecolor="k", label="induction head"),
            plt.scatter([], [], marker="o", s=55, facecolor="none",
                        edgecolor="k", label="name-mover"),
            plt.scatter([], [], marker="s", s=50, facecolor="none",
                        edgecolor="k", label="S-inhibition")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
          fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("results/figures/communities_grid.png", dpi=200,
            bbox_inches="tight")
print("Q", round(Q, 3), "| sizes", [len(c) for c in comms],
      "| C_ind overlap", overlap[ind_ci], "/", len(C_ind),
      "| induction heads in comm", ind_ci, ":",
      sum(memb[h] == ind_ci for h in IND), "/5")
