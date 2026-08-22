"""Decompose the Demo-2 community knockout for GPT-2 small: is the effect
carried by the five behaviorally identified induction heads alone, or does
the rest of their community carry induction capacity too (backup pool)?

Ablation sets, all mean-ablated at hook_z on the same corpora as
run_community_universality.py:
  A. the top-5 behavioral induction heads alone;
  B. the induction community minus those 5 heads;
  C. the whole community;
  plus 5 layer-matched random control sets each for |A| and |B|.

Reuses the map/Louvain/knockout machinery of run_community_universality.py.

Usage:
    uv run --with datasets --with networkx python experiments/s4a_cs1_decomposition.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "cu", Path(__file__).parent / "lib" / "cs1_pipeline.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)

import networkx as nx

model = cu.load("gpt2")
cfg = model.cfg
ind, pile = cu.corpora(model, "cuda")
df = cu.head_map(model)
surv = cu.survivors(df)
print(f"{len(surv)} survivors", flush=True)

G = nx.Graph()
for r in surv.itertuples():
    if G.has_edge(r.writer, r.reader):
        G[r.writer][r.reader]["w"] += 1
    else:
        G.add_edge(r.writer, r.reader, w=1)
gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
comms = sorted(nx.community.louvain_communities(gcc, weight="w", seed=0),
               key=len, reverse=True)

ind_heads, _ = cu.induction_heads(model, ind)
hits = [len(set(ind_heads) & c) for c in comms]
ci = int(np.argmax(hits))
C_ind = sorted(comms[ci])
A = sorted(set(ind_heads))
B = sorted(set(C_ind) - set(ind_heads))
print(f"induction heads {A} | community n={len(C_ind)} holds {hits[ci]}/5 "
      f"| minus-5 n={len(B)}", flush=True)

means = {"ind": cu.z_means(model, ind), "pile": cu.z_means(model, pile)}
clean = cu.knockout_metrics(model, [], means, ind, pile)
print(f"clean gap {clean['gap']:.3f} nats, pile {clean['pile']:.3f}", flush=True)

lay = lambda n: int(n[1:n.index("H")])
rng = np.random.default_rng(1)

def layer_matched_controls(target, n_sets=5):
    per_layer = {}
    for nm in target:
        per_layer[lay(nm)] = per_layer.get(lay(nm), 0) + 1
    sets = []
    for _ in range(n_sets):
        s = []
        for l, cnt in per_layer.items():
            pool = [f"L{l}H{h}" for h in range(cfg.n_heads)
                    if f"L{l}H{h}" not in target]
            s += list(rng.choice(pool, min(cnt, len(pool)), replace=False))
        sets.append(s)
    return sets

gr = lambda r: 1 - r["gap"] / clean["gap"]
nr = lambda r: r["pile"] - clean["pile"]

out = {"clean_gap": clean["gap"], "clean_pile": clean["pile"],
       "induction_heads": A, "community": C_ind}
for key, hs in [("A_five_heads", A), ("B_comm_minus_five", B),
                ("C_whole_comm", C_ind)]:
    r = cu.knockout_metrics(model, hs, means, ind, pile)
    ctrl = [cu.knockout_metrics(model, s, means, ind, pile)
            for s in layer_matched_controls(hs)]
    out[key] = dict(
        n=len(hs), gap_destroyed=gr(r), pile_rise=nr(r),
        ctrl_gap_destroyed_median=float(np.median([gr(c) for c in ctrl])),
        ctrl_gap_destroyed_max=float(max(gr(c) for c in ctrl)),
        ctrl_pile_rise_median=float(np.median([nr(c) for c in ctrl])))
    o = out[key]
    print(f"{key} (n={len(hs)}): gap destroyed {o['gap_destroyed']:.1%} "
          f"(ctrl med {o['ctrl_gap_destroyed_median']:.1%}, "
          f"max {o['ctrl_gap_destroyed_max']:.1%}) | pile rise "
          f"{o['pile_rise']:.3f} (ctrl med {o['ctrl_pile_rise_median']:.3f})",
          flush=True)

path = Path("results/app1/decomposition_gpt2.json")
path.write_text(json.dumps(out, indent=2))
print("saved", path)


# ---- refinement: separate the known previous-token head L4H11 from B ----
if __name__ != "__refine_guard__":
    PT = "L4H11"
    assert PT in B, f"{PT} not in community remainder"
    for key, hs in [("D_L4H11_alone", [PT]),
                    ("E_B_minus_L4H11", sorted(set(B) - {PT}))]:
        r = cu.knockout_metrics(model, hs, means, ind, pile)
        ctrl = [cu.knockout_metrics(model, s, means, ind, pile)
                for s in layer_matched_controls(hs)]
        out[key] = dict(
            n=len(hs), gap_destroyed=gr(r), pile_rise=nr(r),
            ctrl_gap_destroyed_median=float(np.median([gr(c) for c in ctrl])),
            ctrl_gap_destroyed_max=float(max(gr(c) for c in ctrl)),
            ctrl_pile_rise_median=float(np.median([nr(c) for c in ctrl])))
        o = out[key]
        print(f"{key} (n={len(hs)}): gap destroyed {o['gap_destroyed']:.1%} "
              f"(ctrl med {o['ctrl_gap_destroyed_median']:.1%}, "
              f"max {o['ctrl_gap_destroyed_max']:.1%}) | pile rise "
              f"{o['pile_rise']:.3f} (ctrl med {o['ctrl_pile_rise_median']:.3f})",
              flush=True)
    path.write_text(json.dumps(out, indent=2))
    print("saved", path)
