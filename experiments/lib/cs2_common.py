"""Shared utilities for the Case-study-2 pilot (pooled bands and
corpora), extracted from the retired seven-model battery runner (now in
parked/subspace-battery-2026-08-11/) so the release package has no
dependency on parked code.
"""
import numpy as np
import torch

N_SEQ = 32

def rank_bands(shares):
    """THE band-selection rule (paper Section 5): rank the pooled
    eigendirections by position specificity, the ratio of positional
    coupling to token coupling. shares[b] = (pos multiple, tok multiple).
    The deleted pair is the top 2 of this ranking."""
    return sorted(shares,
                  key=lambda b: -shares[b][0] / max(shares[b][1], 1e-12))

def pooled_bands(model):
    """Top-10 eigendirections of the pooled second moment (D3 recipe)."""
    cfg = model.cfg
    with torch.no_grad():
        S = torch.zeros(cfg.d_model, cfg.d_model, dtype=torch.float64,
                        device=model.W_Q.device)
        for l in range(cfg.n_layers):
            for W in (model.W_Q[l], model.W_K[l], model.W_V[l]):
                G = torch.einsum("hDe,hFe->hDF", W.double(), W.double())
                S += (G / torch.einsum("hii->h", G)[:, None, None]).sum(0)
            WO = model.W_O[l].double()
            G = torch.einsum("hdD,hdF->hDF", WO, WO)
            S += (G / torch.einsum("hii->h", G)[:, None, None]).sum(0)
        WE = model.W_E.double()
        Gp = WE.T @ WE
        S += Gp / torch.trace(Gp)
        if has_learned_pos(model):
            Wp = model.W_pos.double()
            Gp = Wp.T @ Wp
            S += Gp / torch.trace(Gp)
        WU = model.W_U.double()
        Gp = WU @ WU.T
        S += Gp / torch.trace(Gp)
    vecs = torch.linalg.eigh(S.cpu())[1].flip(1).numpy()
    return vecs[:, :10]

def has_learned_pos(model):
    return model.cfg.positional_embedding_type in ("standard", "shortformer") \
        and getattr(model, "pos_embed", None) is not None

def corpora(model, device):
    tok = model.tokenizer
    bos = tok.bos_token_id
    if bos is None:
        bos = tok.eos_token_id if tok.eos_token_id is not None else 0
    nv = min(model.cfg.d_vocab, len(tok))
    special = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
    vocab = torch.tensor([t for t in range(nv) if t not in special])
    g = torch.Generator().manual_seed(0)
    rt = torch.cat([torch.full((N_SEQ, 1), bos),
                    vocab[torch.randint(len(vocab), (N_SEQ, 127), generator=g)]],
                   1).to(device)
    blk = vocab[torch.randint(len(vocab), (N_SEQ, 128), generator=g)]
    ind = torch.cat([torch.full((N_SEQ, 1), bos),
                     blk.repeat(1, 2)[:, :255]], 1).to(device)
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
    return rt, ind, pile

