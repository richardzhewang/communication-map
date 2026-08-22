"""Subspace overlaps behind the validity columns of the deletion table
(paper Section 5): principal cosines between the rule's plane and the
activation-PCA / outlier-coordinate planes, per model, plus the two
baselines against each other. 1 = identical direction, 0 = orthogonal.

Usage: .venv/bin/python experiments/s7a_overlaps.py
Writes results/app2/overlaps.json.
"""
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/home/richard/CS_Projects/communication-map")
sys.path.insert(0, str(REPO / "experiments"))
spec = importlib.util.spec_from_file_location(
    "ru", REPO / "experiments" / "lib" / "cs2_common.py")
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)
torch.set_grad_enabled(False)
N_SEQ = ru.N_SEQ


def overlap(A, B):
    """A [d,ka], B [d,kb] orthonormal columns -> principal cosines list."""
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return [round(float(x), 4) for x in s]


def shares_and_selection(vecs, Wpos, WE, d):
    tp, te = np.linalg.norm(Wpos) ** 2, np.linalg.norm(WE) ** 2
    sh = {}
    for b in range(10):
        v = vecs[:, b]
        sh[b] = (np.linalg.norm(Wpos @ v) ** 2 / tp * d,
                 np.linalg.norm(WE @ v) ** 2 / te * d)
    return sorted(ru.rank_bands(sh)[:2])


def coord_plane(dims, d):
    V = np.zeros((d, len(dims)))
    for j, dd in enumerate(dims):
        V[dd, j] = 1.0
    return V


def tl_model(name):
    from transformer_lens import HookedTransformer
    kw = dict(fold_ln=True, center_writing_weights=True,
              center_unembed=True, device="cuda", dtype=torch.float32)
    if name.startswith("pythia"):
        from transformers import AutoModelForCausalLM
        hf = AutoModelForCausalLM.from_pretrained(
            f"EleutherAI/{name}", torch_dtype=torch.float32)
        hf.embed_out = getattr(hf, "lm_head", None) \
            or hf.get_output_embeddings()
        m = HookedTransformer.from_pretrained(name, hf_model=hf, **kw).eval()
        del hf
        return m
    return HookedTransformer.from_pretrained(name, **kw).eval()


def measure_tl(name):
    model = tl_model(name)
    cfg = model.cfg
    d, L = cfg.d_model, cfg.n_layers
    rt, ind, pile = ru.corpora(model, "cuda")
    vecs = ru.pooled_bands(model)
    nm = f"blocks.{L//2}.hook_resid_pre"

    def mid(toks):
        outs = []
        for i in range(0, N_SEQ, 8):
            _, c = model.run_with_cache(toks[i:i+8], return_type=None,
                                        names_filter=lambda n: n == nm)
            outs.append(c[nm].float().cpu())
        return torch.cat(outs)

    if ru.has_learned_pos(model):
        Wpos = model.W_pos.detach().double().cpu().numpy()
    else:
        Wpos = mid(rt).mean(0).double().numpy()
        Wpos = Wpos - Wpos.mean(0)
    WE = model.W_E.detach().double().cpu().numpy()
    pair = shares_and_selection(vecs, Wpos, WE, d)
    A = mid(pile).reshape(-1, d).double().numpy()
    rms = np.sqrt((A ** 2).mean(0))
    out2 = np.argsort(rms)[::-1][:2].tolist()
    A = A - A.mean(0)
    apca = np.linalg.eigh(A.T @ A)[1][:, ::-1][:, :2]
    V_rule = vecs[:, pair]
    res = dict(model=name, d=d, pair=pair, outlier=out2,
               ov_actpca=overlap(V_rule, apca),
               ov_outlier=overlap(V_rule, coord_plane(out2, d)),
               ov_pca_outlier=overlap(apca, coord_plane(out2, d)),
               chance=2.0 / d)
    for p in model.parameters():
        p.data = torch.empty(0)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def measure_69b():
    from map_build import stream_extract
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "pythia-6.9b"
    W = stream_extract(name)
    d, L = W["d"], W["L"]
    S = torch.zeros(d, d, dtype=torch.float64, device="cuda")
    for key in ("Q", "K", "V", "O"):
        for l in range(L):
            X = W[key][l].to("cuda")
            G = torch.matmul(X, X.transpose(1, 2))
            tr = torch.einsum("hii->h", G)
            S += (G / tr[:, None, None]).sum(0).double()
            del X, G
    He = W["H_emb"].double().cuda()
    S += He / torch.einsum("ii->", He)
    Gu = W["G_unemb"].double().cuda()
    S += Gu / torch.einsum("ii->", Gu)
    del He, Gu
    vecs = torch.linalg.eigh(S.cpu())[1].flip(1).numpy()[:, :10]
    del S, W
    gc.collect()
    torch.cuda.empty_cache()

    tok = AutoTokenizer.from_pretrained(f"EleutherAI/{name}")
    model = AutoModelForCausalLM.from_pretrained(
        f"EleutherAI/{name}", torch_dtype=torch.float32,
        device_map={"": 0}).eval()
    layers = model.gpt_neox.layers
    MIDL = model.config.num_hidden_layers // 2
    bos = tok.bos_token_id or tok.eos_token_id or 0
    nv = min(model.config.vocab_size, len(tok))
    sp = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
    vocab = torch.tensor([t for t in range(nv) if t not in sp])
    g = torch.Generator().manual_seed(0)
    rt = torch.cat([torch.full((N_SEQ, 1), bos),
                    vocab[torch.randint(len(vocab), (N_SEQ, 127),
                                        generator=g)]], 1).cuda()
    blk = vocab[torch.randint(len(vocab), (N_SEQ, 128), generator=g)]
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
                      torch.tensor(rows)], 1).cuda()
    cap = {}

    def catch(mod, a, k):
        cap["h"] = a[0].detach()
        return None

    def mid(toks):
        h = layers[MIDL].register_forward_pre_hook(catch, with_kwargs=True)
        outs = []
        for i in range(0, N_SEQ, 4):
            model(toks[i:i+4])
            outs.append(cap["h"].float().cpu())
        h.remove()
        return torch.cat(outs)

    prof = mid(rt).mean(0).double().numpy()
    prof = prof - prof.mean(0)
    WE = model.gpt_neox.embed_in.weight.detach().float().cpu() \
        .double().numpy()
    pair = shares_and_selection(vecs, prof, WE, d)
    A = mid(pile).reshape(-1, d).double().numpy()
    rms = np.sqrt((A ** 2).mean(0))
    out2 = np.argsort(rms)[::-1][:2].tolist()
    A = A - A.mean(0)
    apca = np.linalg.eigh(A.T @ A)[1][:, ::-1][:, :2]
    V_rule = vecs[:, pair]
    res = dict(model=name, d=d, pair=pair, outlier=out2,
               ov_actpca=overlap(V_rule, apca),
               ov_outlier=overlap(V_rule, coord_plane(out2, d)),
               ov_pca_outlier=overlap(apca, coord_plane(out2, d)),
               chance=2.0 / d)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


results = []
for m in ("gpt2", "gpt2-medium", "gpt2-large", "pythia-160m",
          "pythia-2.8b"):
    results.append(measure_tl(m))
    r = results[-1]
    print(f"{r['model']:14s} rule{r['pair']} out{r['outlier']} | "
          f"cos(rule,actPCA) {r['ov_actpca']} | "
          f"cos(rule,outlier) {r['ov_outlier']} | "
          f"cos(actPCA,outlier) {r['ov_pca_outlier']}", flush=True)
results.append(measure_69b())
r = results[-1]
print(f"{r['model']:14s} rule{r['pair']} out{r['outlier']} | "
      f"cos(rule,actPCA) {r['ov_actpca']} | "
      f"cos(rule,outlier) {r['ov_outlier']} | "
      f"cos(actPCA,outlier) {r['ov_pca_outlier']}", flush=True)
Path(REPO / "results" / "app2" /
     "overlaps.json").write_text(json.dumps(results, indent=1))
