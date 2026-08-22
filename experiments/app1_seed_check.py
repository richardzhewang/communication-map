"""Seed-robustness check for the Demo-2 decomposition: recompute the
A/B/C/D/E gap-destruction numbers on a fresh induction corpus (random-block
generator seed 1 instead of 0; community and head identities are
weight-derived and unchanged). Prints seed-1 numbers for comparison with
decomposition_gpt2.json.

Usage:
    uv run --with datasets --with networkx python experiments/s4b_cs1_seed_check.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "cu", Path(__file__).parent / "lib" / "cs1_pipeline.py")
cu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cu)

import networkx as nx

SEED = 1

model = cu.load("gpt2")
ind0, pile = cu.corpora(model, "cuda")

# fresh repeated-block corpus, generator seed SEED (same construction)
tok = model.tokenizer
bos = tok.bos_token_id or tok.eos_token_id or 0
nv = min(model.cfg.d_vocab, len(tok))
special = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
vocab = torch.tensor([t for t in range(nv) if t not in special])
g = torch.Generator().manual_seed(SEED)
blk = vocab[torch.randint(len(vocab), (cu.N_SEQ, cu.BLOCK), generator=g)]
ind = torch.cat([torch.full((cu.N_SEQ, 1), bos),
                 blk.repeat(1, 2)[:, :2 * cu.BLOCK - 1]], 1).to("cuda")

df = cu.head_map(model)
surv = cu.survivors(df)
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
C_ind = sorted(comms[int(np.argmax(hits))])
A = sorted(set(ind_heads))
B = sorted(set(C_ind) - set(A))
print(f"seed-{SEED} induction heads {A} | community n={len(C_ind)} "
      f"holds {max(hits)}/5", flush=True)

means = {"ind": cu.z_means(model, ind), "pile": cu.z_means(model, pile)}
clean = cu.knockout_metrics(model, [], means, ind, pile)
print(f"clean gap {clean['gap']:.3f} nats", flush=True)

PT = "L4H11"
conds = [("A_five_heads", A), ("D_L4H11_alone", [PT]),
         ("E_B_minus_L4H11", sorted(set(B) - {PT})),
         ("B_comm_minus_five", B), ("C_whole_comm", C_ind)]
out = {"seed": SEED, "clean_gap": clean["gap"]}
for key, hs in conds:
    r = cu.knockout_metrics(model, hs, means, ind, pile)
    out[key] = dict(n=len(hs), gap_destroyed=1 - r["gap"] / clean["gap"])
    print(f"{key} (n={len(hs)}): gap destroyed "
          f"{out[key]['gap_destroyed']:.1%}", flush=True)

path = Path(f"results/app1/decomposition_gpt2_seed{SEED}.json")
path.write_text(json.dumps(out, indent=2))
print("saved", path)
