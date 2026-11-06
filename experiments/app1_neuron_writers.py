"""Neuron-level use of the map: which MLP neurons write into the induction
heads, and what happens when those neurons are ablated.

Part A (descriptive). For each behavioral induction head r, the K-, Q-, and
V-composition coupling of every causally eligible neuron writer m
(l_m < l_r) is computed from weights exactly as in the map's neuron->head
classes (Appendix B). The script reports the top writers per head and
channel, the layer distribution of the top neurons by combined coupling,
and the same in-coupling summary for every other head of the induction
heads' layers, so induction heads can be compared with their neighbours.

Part B (intervention). The top-N neurons by combined coupling into the
induction heads (N = 100, 300, 1000) are mean-ablated (activation replaced
by its corpus mean, the neuron analog of the head ablation), each against
five random neuron sets matched in size and per-layer counts, with the
induction gain and the Pile loss as readouts (same corpora and protocol as
the head ablations).

Usage: python experiments/app1_neuron_writers.py MODEL   (gpt2, pythia-160m)
Writes results/app1/neuron_writers_{MODEL}.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "cu", REPO / "experiments" / "lib" / "cs1_pipeline.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)
N_SEQ, BLOCK = cu.N_SEQ, cu.BLOCK
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
TOPN = [100, 300, 1000]
N_CTRL = 5

model = cu.load(MODEL)
cfg = model.cfg
L, H, d, dh, dm = cfg.n_layers, cfg.n_heads, cfg.d_model, cfg.d_head, cfg.d_mlp
ind, pile = cu.corpora(model, "cuda")
ind_heads, _ = cu.induction_heads(model, ind)
parse = lambda s: (int(s[1:s.index("H")]), int(s[s.index("H") + 1:]))
print(f"[{MODEL}] induction heads {ind_heads}", flush=True)

# ---------------------------------------------------------------- Part A
with torch.no_grad():
    Wout = model.W_out.detach().double()                    # [L, dm, d]
    What = Wout / Wout.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    What = What.reshape(L * dm, d)                           # unit write rows
    dev = What.device
    n_layer = torch.arange(L, device=dev).repeat_interleave(dm)  # layer of neuron
    WQ, WK, WV = model.W_Q.detach().double(), model.W_K.detach().double(), model.W_V.detach().double()
    WO = model.W_O.detach().double().transpose(-1, -2)       # [L, H, d, dh]


def coupling_into(lr, h):
    """C^2 of every neuron (rows) into head (lr, h) for K, Q, V; ineligible
    neurons (l_m >= lr) get NaN."""
    q, k, v, o = WQ[lr, h], WK[lr, h], WV[lr, h], WO[lr, h]
    gQ, gK, gV, gO = q.T @ q, k.T @ k, v.T @ v, o.T @ o
    trQK, trOV = torch.trace(gQ @ gK), torch.trace(gO @ gV)
    yK, yQ, yV = What @ k, What @ q, What @ v                # [N, dh]
    cK = ((yK @ gQ) * yK).sum(1) / trQK
    cQ = ((yQ @ gK) * yQ).sum(1) / trQK
    cV = ((yV @ gO) * yV).sum(1) / trOV
    elig = n_layer < lr
    out = torch.stack([cK, cQ, cV], 1)
    out[~elig] = float("nan")
    return out                                                # [N, 3]


lab = lambda i: f"L{int(n_layer[i])}N{int(i) % dm}"
desc = {"induction_heads": ind_heads, "chance_c2": 1.0 / d, "per_head": {}}
combined = torch.zeros(L * dm, dtype=torch.float64, device=dev)
for hd in ind_heads:
    lr, h = parse(hd)
    c = coupling_into(lr, h)
    tot = torch.nan_to_num(c, nan=0.0).sum(1)
    combined += tot
    entry = {}
    for j, ch in enumerate(["K", "Q", "V"]):
        col = torch.nan_to_num(c[:, j], nan=-1.0)
        top = torch.argsort(col, descending=True)[:10]
        entry[ch] = [(lab(i), round(float(col[i]), 5)) for i in top]
        entry[f"{ch}_max_over_chance"] = round(float(col.max()) * d, 1)
    top200 = torch.argsort(tot, descending=True)[:200]
    entry["top200_layer_hist"] = {int(l): int((n_layer[top200] == l).sum())
                                  for l in range(lr)}
    desc["per_head"][hd] = entry
    print(f"  {hd}: max C2/chance K {entry['K_max_over_chance']} Q "
          f"{entry['Q_max_over_chance']} V {entry['V_max_over_chance']} | "
          f"top-200 layers {entry['top200_layer_hist']}", flush=True)

# comparison: in-coupling summary for every head in the induction heads' layers
layers = sorted({parse(hd)[0] for hd in ind_heads})
comp = {}
for lr in layers:
    for h in range(H):
        c = torch.nan_to_num(coupling_into(lr, h), nan=0.0).sum(1)
        top = torch.sort(c, descending=True).values[:200]
        comp[f"L{lr}H{h}"] = dict(top200_mean_c2=float(top.mean()),
                                 max_c2=float(c.max()),
                                 is_induction=f"L{lr}H{h}" in ind_heads)
ranks = {}
for lr in layers:
    heads = [f"L{lr}H{h}" for h in range(H)]
    order = sorted(heads, key=lambda k: -comp[k]["top200_mean_c2"])
    for hd in heads:
        if comp[hd]["is_induction"]:
            ranks[hd] = order.index(hd) + 1
desc["heads_same_layers"] = comp
desc["induction_head_rank_in_layer_by_top200_mean"] = ranks
print(f"  rank of induction heads among the {H} heads of their layer, by "
      f"mean top-200 neuron in-coupling: {ranks}", flush=True)

# ---------------------------------------------------------------- Part B
order = torch.argsort(combined, descending=True)
top_sets = {N: [int(i) for i in order[:N]] for N in TOPN}
desc["top_neurons_combined"] = [(lab(i), round(float(combined[i]), 5))
                                for i in order[:30]]


def neuron_means(toks):
    names = {f"blocks.{l}.mlp.hook_post" for l in range(L)}
    acc = {}
    with torch.no_grad():
        for i in range(0, len(toks), 8):
            _, c = model.run_with_cache(toks[i:i + 8], return_type=None,
                                        names_filter=lambda n: n in names)
            for l in range(L):
                acc[l] = acc.get(l, 0) + c[f"blocks.{l}.mlp.hook_post"] \
                    .mean(dim=(0, 1)) * (min(8, len(toks) - i) / len(toks))
    return acc


means = {"ind": neuron_means(ind), "pile": neuron_means(pile)}


def ablate_metrics(neurons):
    by_layer = {}
    for i in neurons:
        by_layer.setdefault(int(n_layer[i]), []).append(int(i) % dm)

    def hooks(mkey):
        def make(l):
            idx = torch.tensor(by_layer[l], device="cuda")

            def hook(a, hook):
                a[:, :, idx] = means[mkey][l][idx]
                return a
            return hook
        return [(f"blocks.{l}.mlp.hook_post", make(l)) for l in by_layer]

    with torch.no_grad(), model.hooks(fwd_hooks=hooks("ind") if neurons else []):
        n1 = n2 = 0.0
        for i in range(0, N_SEQ, 8):
            lp = F.log_softmax(model(ind[i:i + 8]).float(), -1)
            nl = -lp[:, :-1].gather(-1, ind[i:i + 8, 1:][..., None]).squeeze(-1)
            n1 += float(nl[:, :BLOCK - 1].mean()) / (N_SEQ // 8)
            n2 += float(nl[:, BLOCK:].mean()) / (N_SEQ // 8)
    with torch.no_grad(), model.hooks(fwd_hooks=hooks("pile") if neurons else []):
        tot = 0.0
        for i in range(0, N_SEQ, 8):
            lp = F.log_softmax(model(pile[i:i + 8]).float(), -1)
            tot += float((-lp[:, :-1].gather(-1, pile[i:i + 8, 1:][..., None])
                          .squeeze(-1)).mean()) / (N_SEQ // 8)
    return dict(gap=n1 - n2, pile=tot, loss_first=n1, loss_second=n2)


clean = ablate_metrics([])
gr = lambda r: 1 - r["gap"] / clean["gap"]
rng = np.random.default_rng(0)
abl = {"clean": clean}
for N, sel in top_sets.items():
    r = ablate_metrics(sel)
    per_layer = {}
    for i in sel:
        per_layer[int(n_layer[i])] = per_layer.get(int(n_layer[i]), 0) + 1
    sel_set = set(sel)
    ctrl = []
    for _ in range(N_CTRL):
        s = []
        for l, cnt in per_layer.items():
            pool = [l * dm + m for m in range(dm) if l * dm + m not in sel_set]
            s += [int(x) for x in rng.choice(pool, cnt, replace=False)]
        ctrl.append(ablate_metrics(s))
    abl[f"top{N}"] = dict(
        n=N, per_layer=per_layer, gap_destroyed=gr(r),
        pile_rise=r["pile"] - clean["pile"],
        loss_first=r["loss_first"], loss_second=r["loss_second"],
        ctrl_gap_destroyed_median=float(np.median([gr(c) for c in ctrl])),
        ctrl_gap_destroyed_max=float(max(gr(c) for c in ctrl)),
        ctrl_pile_rise_median=float(np.median([c["pile"] - clean["pile"]
                                               for c in ctrl])))
    o = abl[f"top{N}"]
    print(f"  top-{N} neurons: gain destroyed {o['gap_destroyed']:.1%} "
          f"(ctrl median {o['ctrl_gap_destroyed_median']:.1%}, max "
          f"{o['ctrl_gap_destroyed_max']:.1%}) | pile +{o['pile_rise']:.3f} "
          f"(ctrl +{o['ctrl_pile_rise_median']:.3f}) | layers {per_layer}",
          flush=True)

out = dict(model=MODEL, descriptive=desc, ablation=abl)
path = REPO / "results" / "app1" / f"neuron_writers_{MODEL}.json"
path.write_text(json.dumps(out, indent=1))
print("saved", path)
