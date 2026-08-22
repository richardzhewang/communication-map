"""Community-organ universality: head-head map -> Louvain -> knockout,
generalized to arbitrary TL models (self-contained reimplementation; the
frozen commap package is untouched).

Per model:
  1. head-head composition map, all K/Q/V slots, sigma-weighted statistic
     C = |R M|_F / (|R|_F |M|_F) via d_head-space Grams (D1), writer layer <
     reader layer;
  2. empirical bulk null per (slot, layer-span) stratum (robust
     center/scale: median + IQR/1.349 — simplified Efron central matching),
     one-sided upper p, BH per slot family at q = 0.05;
  3. Louvain (seed 0) on the survivor multigraph GCC;
  4. induction heads identified behaviorally (attention to offset -(T-1) on
     repeated blocks); C_ind = community with the plurality of the top-5;
     dissociation control = largest community containing none of them;
  5. mean-ablate hook_z of each set; readouts: induction gap + pile NLL;
     5 layer-matched random same-size control sets.

Validation anchor: gpt2 must substantially recover the known 26-head
induction community from the original pipeline (Jaccard reported).

Usage:
    uv run --with datasets --with networkx python experiments/lib/cs1_pipeline.py
"""

from __future__ import annotations

import gc
import json
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

MODELS = ["gpt2", "gpt2-medium", "pythia-160m", "gpt-neo-125M"]
N_SEQ = 32
BLOCK = 128
Q_LEVEL = 0.05
OUT = Path("results/app1")
KNOWN_GPT2_C_IND = {"L0H1","L0H3","L0H4","L0H5","L1H0","L1H1","L3H7","L3H10",
                    "L4H3","L4H7","L4H11","L5H0","L5H1","L5H6","L5H8","L6H9",
                    "L7H2","L7H10","L7H11","L8H1","L8H3","L9H1","L9H4","L9H6",
                    "L9H9","L10H1"}


def load(name, device="cuda"):
    from transformer_lens import HookedTransformer
    kw = dict(fold_ln=True, center_writing_weights=True, center_unembed=True,
              device=device)
    if name.startswith("pythia"):
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{name}")
        if not hasattr(hf, "embed_out"):
            hf.embed_out = getattr(hf, "lm_head", None) or hf.get_output_embeddings()
        return HookedTransformer.from_pretrained(name, hf_model=hf, **kw).eval()
    return HookedTransformer.from_pretrained(name, **kw).eval()


# ------------------------------------------------ 1) head-head composition map

def head_map(model):
    """DataFrame: writer, reader, slot, span, C."""
    import pandas as pd
    cfg = model.cfg
    L, H = cfg.n_layers, cfg.n_heads
    with torch.no_grad():
        Q = model.W_Q.double()                     # [L,H,d,dh]
        K = model.W_K.double()
        V = model.W_V.double()
        O = model.W_O.double().transpose(-1, -2)   # [L,H,d,dh] column conv
        G = {"Q": torch.einsum("lhDe,lhDf->lhef", Q, Q),
             "K": torch.einsum("lhDe,lhDf->lhef", K, K),
             "V": torch.einsum("lhDe,lhDf->lhef", V, V),
             "O": torch.einsum("lhDe,lhDf->lhef", O, O)}
        nrm_R = {"K": torch.einsum("lhef,lhef->lh", G["Q"], G["K"]),
                 "Q": torch.einsum("lhef,lhef->lh", G["Q"], G["K"]),
                 "V": torch.einsum("lhef,lhef->lh", G["O"], G["V"])}
        nrm_M = torch.einsum("lhef,lhef->lh", G["O"], G["V"])
        right = {"K": K, "Q": Q, "V": V}
        left = {"K": "Q", "Q": "K", "V": "O"}
        rows = []
        for lB in range(1, L):
            for lA in range(lB):
                for slot in ("K", "Q", "V"):
                    X = torch.einsum("hDe,gDf->hgef", right[slot][lB], O[lA])
                    GL = G[left[slot]][lB]                     # [H,e,e]
                    GA = G["V"][lA]                            # [H,f,f]
                    Y = torch.einsum("hea,hgab->hgeb", GL, X)
                    Z = torch.einsum("hgeb,gbf->hgef", Y, GA)
                    num = torch.einsum("hgef,hgef->hg", X, Z)  # |R M|_F^2
                    C = torch.sqrt(num / (nrm_R[slot][lB][:, None]
                                          * nrm_M[lA][None, :]))
                    Cc = C.cpu().numpy()
                    for hB in range(H):
                        for hA in range(H):
                            rows.append((f"L{lA}H{hA}", f"L{lB}H{hB}", slot,
                                         lB - lA, Cc[hB, hA]))
    return pd.DataFrame(rows, columns=["writer", "reader", "slot", "span", "C"])


def bh_qvalues(p):
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def survivors(df):
    from scipy.stats import norm
    df = df.copy()
    z = np.empty(len(df))
    for (slot, span), g in df.groupby(["slot", "span"]):
        x = g.C.values
        med = np.median(x)
        scale = max((np.quantile(x, .75) - np.quantile(x, .25)) / 1.349, 1e-12)
        z[g.index] = (x - med) / scale
    df["z"] = z
    df["p"] = norm.sf(df.z)                       # one-sided upper (C >= 0)
    q = np.empty(len(df))
    for _, g in df.groupby("slot"):
        q[g.index] = bh_qvalues(g.p.values)
    df["q"] = q
    return df[df.q <= Q_LEVEL]


# ------------------------------------------------ 2) corpora + behavioral IDs

def corpora(model, device):
    tok = model.tokenizer
    bos = tok.bos_token_id
    if bos is None:
        bos = tok.eos_token_id if tok.eos_token_id is not None else 0
    nv = min(model.cfg.d_vocab, len(tok))
    special = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
    vocab = torch.tensor([t for t in range(nv) if t not in special])
    g = torch.Generator().manual_seed(0)
    blk = vocab[torch.randint(len(vocab), (N_SEQ, BLOCK), generator=g)]
    ind = torch.cat([torch.full((N_SEQ, 1), bos),
                     blk.repeat(1, 2)[:, :2 * BLOCK - 1]], 1).to(device)
    from datasets import load_dataset
    ds = load_dataset("NeelNanda/pile-10k", split="train")
    rows = []
    for doc in ds:
        t = tok(doc["text"])["input_ids"]
        if len(t) >= 255:
            rows.append(t[:255])
        if len(rows) == N_SEQ:
            break
    pile = torch.cat([torch.full((N_SEQ, 1), bos),
                      torch.tensor(rows)], 1).to(device)
    return ind, pile


def induction_heads(model, ind, topk=5):
    """Attention from second-half i to i-(BLOCK-1): the induction offset."""
    cfg = model.cfg
    names = {f"blocks.{l}.attn.hook_pattern" for l in range(cfg.n_layers)}
    score = {}
    off = BLOCK - 1
    with torch.no_grad():
        for i in range(0, N_SEQ, 8):
            _, c = model.run_with_cache(ind[i:i+8], return_type=None,
                                        names_filter=lambda n: n in names)
            for l in range(cfg.n_layers):
                pat = c[f"blocks.{l}.attn.hook_pattern"]     # [B,H,i,j]
                d = pat.diagonal(offset=-off, dim1=2, dim2=3)  # [B,H,pos-off]
                s = d[:, :, off:].mean(dim=(0, 2))             # i >= 2*off
                for h in range(cfg.n_heads):
                    score[(l, h)] = score.get((l, h), 0) + float(s[h]) / (N_SEQ // 8)
    top = sorted(score, key=score.get, reverse=True)[:topk]
    return [f"L{l}H{h}" for l, h in top], {f"L{l}H{h}": v for (l, h), v in score.items()}


# ------------------------------------------------ 3) knockout machinery

def knockout_metrics(model, head_set, means, ind, pile):
    lay = lambda n: int(n[1:n.index("H")])
    hd = lambda n: int(n[n.index("H")+1:])
    by_layer = {}
    for nm in head_set:
        by_layer.setdefault(lay(nm), []).append(hd(nm))

    def hooks(mkey):
        def make(l):
            hs = torch.tensor(by_layer[l]).to(ind.device)
            def hook(z, hook):
                z[:, :, hs] = means[mkey][l][hs]
                return z
            return hook
        return [(f"blocks.{l}.attn.hook_z", make(l)) for l in by_layer]

    with torch.no_grad(), model.hooks(fwd_hooks=hooks("ind") if head_set else []):
        n1 = n2 = 0.0
        for i in range(0, N_SEQ, 8):
            lp = F.log_softmax(model(ind[i:i+8]).float(), -1)
            nl = -lp[:, :-1].gather(-1, ind[i:i+8, 1:][..., None]).squeeze(-1)
            n1 += float(nl[:, :BLOCK-1].mean()) / (N_SEQ // 8)
            n2 += float(nl[:, BLOCK:].mean()) / (N_SEQ // 8)
    with torch.no_grad(), model.hooks(fwd_hooks=hooks("pile") if head_set else []):
        tot = 0.0
        for i in range(0, N_SEQ, 8):
            lp = F.log_softmax(model(pile[i:i+8]).float(), -1)
            tot += float((-lp[:, :-1].gather(-1, pile[i:i+8, 1:][..., None])
                          .squeeze(-1)).mean()) / (N_SEQ // 8)
    return dict(gap=n1 - n2, pile=tot)


def z_means(model, toks):
    cfg = model.cfg
    names = {f"blocks.{l}.attn.hook_z" for l in range(cfg.n_layers)}
    acc = {}
    with torch.no_grad():
        for i in range(0, len(toks), 8):
            _, c = model.run_with_cache(toks[i:i+8], return_type=None,
                                        names_filter=lambda n: n in names)
            for l in range(cfg.n_layers):
                acc[l] = acc.get(l, 0) + c[f"blocks.{l}.attn.hook_z"].mean(dim=(0,1)) \
                         * (min(8, len(toks)-i) / len(toks))
    return acc


# ------------------------------------------------ per-model driver

def run_model(name):
    model = load(name)
    cfg = model.cfg
    ind, pile = corpora(model, "cuda")
    print(f"  [{name}] map ...", flush=True)
    df = head_map(model)
    surv = survivors(df)
    print(f"  [{name}] {len(df)} candidates, {len(surv)} survivors "
          f"({len(surv)/len(df):.1%})", flush=True)

    G = nx.Graph()
    for r in surv.itertuples():
        if G.has_edge(r.writer, r.reader):
            G[r.writer][r.reader]["w"] += 1
        else:
            G.add_edge(r.writer, r.reader, w=1)
    gcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    comms = sorted(nx.community.louvain_communities(gcc, weight="w", seed=0),
                   key=len, reverse=True)
    Qmod = nx.community.modularity(gcc, comms, weight="w")

    ind_heads, _ = induction_heads(model, ind)
    hits = [len(set(ind_heads) & c) for c in comms]
    ci = int(np.argmax(hits))
    C_ind = sorted(comms[ci])
    no_ind = [c for c in comms if not (set(ind_heads) & c)]
    C_ctrl = sorted(max(no_ind, key=len)) if no_ind else []
    print(f"  [{name}] {len(comms)} comms, Q={Qmod:.3f}; induction heads "
          f"{ind_heads}; C_ind n={len(C_ind)} holds {hits[ci]}/5; "
          f"dissoc comm n={len(C_ctrl)}", flush=True)

    means = {"ind": z_means(model, ind), "pile": z_means(model, pile)}
    clean = knockout_metrics(model, [], means, ind, pile)
    r_ind = knockout_metrics(model, C_ind, means, ind, pile)
    r_dis = (knockout_metrics(model, C_ctrl, means, ind, pile)
             if C_ctrl else None)

    lay = lambda n: int(n[1:n.index("H")])
    rng = np.random.default_rng(1)
    per_layer = {}
    for nm in C_ind:
        per_layer[lay(nm)] = per_layer.get(lay(nm), 0) + 1
    ctrls = []
    for _ in range(5):
        s = []
        for l, cnt in per_layer.items():
            pool = [f"L{l}H{h}" for h in range(cfg.n_heads)
                    if f"L{l}H{h}" not in C_ind]
            s += list(rng.choice(pool, min(cnt, len(pool)), replace=False))
        ctrls.append(knockout_metrics(model, s, means, ind, pile))

    gr = lambda r: 1 - r["gap"] / clean["gap"]
    nr = lambda r: r["pile"] - clean["pile"]
    res = dict(
        model=name, n_heads_total=cfg.n_layers * cfg.n_heads,
        n_candidates=len(df), n_survivors=len(surv),
        modularity=Qmod, n_comms=len(comms),
        induction_heads=ind_heads, C_ind=C_ind,
        c_ind_holds=int(hits[ci]),
        gap_clean=clean["gap"],
        gap_reduction=dict(C_ind=gr(r_ind),
                           dissoc=(gr(r_dis) if r_dis else None),
                           ctrl_median=float(np.median([gr(r) for r in ctrls])),
                           ctrl_max=float(max(gr(r) for r in ctrls))),
        pile_rise=dict(C_ind=nr(r_ind),
                       dissoc=(nr(r_dis) if r_dis else None),
                       ctrl_median=float(np.median([nr(r) for r in ctrls]))),
        dissoc_n=len(C_ctrl),
    )
    if name == "gpt2":
        jac = len(set(C_ind) & KNOWN_GPT2_C_IND) / len(set(C_ind) | KNOWN_GPT2_C_IND)
        res["anchor_jaccard_vs_original_pipeline"] = jac
        print(f"  [gpt2] anchor Jaccard vs original C_ind: {jac:.2f}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in MODELS:
        print(f"===== {name} =====", flush=True)
        try:
            results[name] = run_model(name)
            r = results[name]
            print(f"  gap reduction: C_ind {r['gap_reduction']['C_ind']:.3f} "
                  f"| dissoc {r['gap_reduction']['dissoc']} "
                  f"| ctrl med {r['gap_reduction']['ctrl_median']:.3f}", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
            results[name] = dict(model=name, error=traceback.format_exc(limit=2))
    (OUT / "community_universality_4models.json").write_text(json.dumps(results, indent=2))
    print("saved", OUT / "community_universality_4models.json")


if __name__ == "__main__":
    main()
