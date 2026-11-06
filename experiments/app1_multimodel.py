"""Application 1: the community ablation replicated across every mapped model.

Unified pipeline, one code path per model:
  1. communities from the model's OWN selected edges
     (results/app1/{model}/edges_heads.csv.gz, Louvain seed 0);
  2. behavioral induction heads from induction-offset attention
     (hook-accumulated, no activation cache, so 6.9B fits if the
     hardware allows the forwards at all);
  3. mean-ablation knockouts: clean, the five behavioral heads, the
     community minus those heads, the full community, and five random
     control sets matched in size and per-layer head counts.

Readouts: fraction of the clean induction gain destroyed and Pile
NLL increase. Results append incrementally to
results/app1/community_replication.json, one entry per model, so a
larger model failing on hardware limits cannot lose earlier rows; a
failure is recorded as {"error": ...} for honest reporting.

Usage: uv run python experiments/app1_multimodel.py [MODEL ...]
       (default: all seven mapped models, 6.9B last)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "lib"))
import cs1_pipeline as cs1  # noqa: E402  (load, corpora, knockout_metrics)

OUT = REPO / "results" / "app1" / "community_replication.json"
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt-neo-125m",
          "pythia-160m", "pythia-2.8b", "pythia-6.9b"]


def head_scores(model, ind):
    """Induction-offset and previous-token attention per head, one hooked
    pass, no activation cache."""
    cfg = model.cfg
    off = cs1.BLOCK - 1
    s_ind = torch.zeros(cfg.n_layers, cfg.n_heads)
    n_batches = 0

    def make(l):
        def hook(pat, hook):
            d = pat.diagonal(offset=-off, dim1=-2, dim2=-1)
            s_ind[l] += d[:, :, off:].mean(dim=(0, 2)).float().cpu()
            return pat
        return hook

    hooks = [(f"blocks.{l}.attn.hook_pattern", make(l))
             for l in range(cfg.n_layers)]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        for i in range(0, cs1.N_SEQ, 4):
            model(ind[i:i + 4], return_type=None)
            n_batches += 1
    return (s_ind / n_batches).numpy()


def z_means(model, toks):
    """Mean attention-value summary per head, hook-accumulated."""
    cfg = model.cfg
    acc = {l: torch.zeros(cfg.n_heads, cfg.d_head, device=toks.device)
           for l in range(cfg.n_layers)}
    n_batches = 0

    def make(l):
        def hook(z, hook):
            acc[l] += z.mean(dim=(0, 1)).detach()
            return z
        return hook

    hooks = [(f"blocks.{l}.attn.hook_z", make(l))
             for l in range(cfg.n_layers)]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        for i in range(0, len(toks), 4):
            model(toks[i:i + 4], return_type=None)
            n_batches += 1
    return {l: (a / n_batches) for l, a in acc.items()}


def communities_from_edges(model_name, H, n_heads_total):
    df = pd.read_csv(REPO / "results" / "app1" / model_name
                     / "edges_heads.csv.gz")
    df = df[df["sig"]]
    idx = lambda s: (s.str.extract(r"L(\d+)H(\d+)").astype(int)
                     .apply(lambda r: r[0] * H + r[1], axis=1).values)
    wi, ri = idx(df["writer"]), idx(df["reader"])
    G = nx.Graph()
    G.add_nodes_from(range(n_heads_total))
    for w, r in zip(wi, ri):
        if G.has_edge(w, r):
            G[w][r]["w"] += 1
        else:
            G.add_edge(w, r, w=1)
    gcc = G.subgraph(max(nx.connected_components(G), key=len))
    return nx.community.louvain_communities(gcc, weight="w", seed=0)


def run_model(name):
    model = cs1.load(name)
    cfg = model.cfg
    H, L = cfg.n_heads, cfg.n_layers
    lab = lambda i: f"L{i // H}H{i % H}"
    ind, pile = cs1.corpora(model, "cuda")

    scores = head_scores(model, ind)
    flat = np.argsort(-scores.ravel())[:5]
    IND = sorted(int(i) for i in flat)
    print(f"  [{name}] induction heads: {[lab(h) for h in IND]}", flush=True)

    comms = communities_from_edges(name, H, L * H)
    hits = [len(set(IND) & set(c)) for c in comms]
    ci = int(np.argmax(hits))
    comm = sorted(comms[ci])
    holds = hits[ci]
    print(f"  [{name}] {len(comms)} communities; C_ind n={len(comm)} "
          f"holds {holds}/5", flush=True)

    means = {"ind": z_means(model, ind), "pile": z_means(model, pile)}
    clean = cs1.knockout_metrics(model, [], means, ind, pile)
    frac = lambda res: (clean["gap"] - res["gap"]) / clean["gap"]

    five = cs1.knockout_metrics(model, [lab(h) for h in IND],
                                means, ind, pile)
    rem_set = [h for h in comm if h not in IND]
    rem = cs1.knockout_metrics(model, [lab(h) for h in rem_set],
                               means, ind, pile)
    full = cs1.knockout_metrics(model, [lab(h) for h in comm],
                                means, ind, pile)
    # Restricted comparison: only the behavioral induction heads that lie
    # inside the community, and the community minus exactly those heads.
    comm_in = set(comm)
    IND_in = [h for h in IND if h in comm_in]
    five_in = cs1.knockout_metrics(model, [lab(h) for h in IND_in],
                                   means, ind, pile)
    rem_in = cs1.knockout_metrics(model, [lab(h) for h in comm
                                          if h not in set(IND_in)],
                                  means, ind, pile)

    # Layer-matched controls: exact size and per-layer head counts. Where a
    # layer holds more community heads than other heads, the shortfall is
    # filled from the community's own heads in that layer (overlap recorded).
    rng = np.random.default_rng(0)
    csets, overlaps = cs1.matched_controls([lab(h) for h in comm], H, 5, rng)
    ctrl = [frac(cs1.knockout_metrics(model, s, means, ind, pile))
            for s in csets]
    # Size-matched controls: the same number of heads drawn from all
    # non-community heads with no layer constraint, hence never overlapping.
    rng2 = np.random.default_rng(100)
    comm_set = set(comm)
    others = [lab(h) for h in range(L * H) if h not in comm_set]
    ctrl_size = [frac(cs1.knockout_metrics(
        model, list(rng2.choice(others, len(comm), replace=False)),
        means, ind, pile)) for _ in range(5)]

    row = dict(model=name, clean_gap=float(clean["gap"]),
               n_comm=int(len(comm)), holds=int(holds),
               induction_heads=[lab(h) for h in IND],
               five=float(frac(five)), remainder=float(frac(rem)),
               full=float(frac(full)),
               ctrl_median=float(np.median(ctrl)),
               ctrl_max=float(np.max(ctrl)),
               ctrl_overlap=[int(o) for o in overlaps],
               ctrl_size_median=float(np.median(ctrl_size)),
               ctrl_size_max=float(np.max(ctrl_size)),
               pile_dnll=float(full["pile"] - clean["pile"]),
               n_ind_in=int(len(IND_in)),
               five_in=float(frac(five_in)),
               remainder_in=float(frac(rem_in)),
               losses={k: dict(first=float(v["loss_first"]),
                               second=float(v["loss_second"]))
                       for k, v in [("clean", clean), ("five", five),
                                    ("remainder", rem), ("full", full),
                                    ("five_in", five_in),
                                    ("remainder_in", rem_in)]})
    print(f"  [{name}] clean {row['clean_gap']:.2f} | five "
          f"{row['five']:.1%} | remainder {row['remainder']:.1%} | full "
          f"{row['full']:.1%} | ctrl med {row['ctrl_median']:.1%} "
          f"(overlap {row['ctrl_overlap']}) | size-matched ctrl med "
          f"{row['ctrl_size_median']:.1%} | "
          f"pile +{row['pile_dnll']:.2f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return row


def main():
    models = sys.argv[1:] or MODELS
    results = json.load(open(OUT)) if OUT.exists() else {}
    for name in models:
        print(f"== {name}", flush=True)
        try:
            results[name] = run_model(name)
        except (torch.cuda.OutOfMemoryError, RuntimeError,
                MemoryError) as e:
            results[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(f"  [{name}] FAILED: {results[name]['error']}",
                  flush=True)
            torch.cuda.empty_cache()
        json.dump(results, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
