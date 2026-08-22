"""CS2 dose curve for big rotary models, HF-native forwards.

Why not TransformerLens: the fp32 processing copy of a 6.9B model does not
fit this box's RAM next to the HF copy. The TL-processed model is
forward-equivalent to the raw HF model (LN folding is a reparameterization,
write-centering is absorbed by LN's mean-subtraction, unembed centering
cancels in log-softmax), so the readouts run on the HF model loaded
shard-by-shard straight to GPU, and the pooled bands come from
stream_extract's validated processed weights. fp32 forwards are
required: see experiments/s7d_precision_check.py.

Anchor: run with pythia-160m first and compare to the TL fp32 readouts
(clean gain 18.36, top-2-eigenvalue deletion destroys 45.8%).

Usage: .venv/bin/python experiments/s5b_cs2_scale.py MODEL [--batch 4]
Writes results/app2/{MODEL}/cs2_dose.json.
"""
import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from map_build import stream_extract  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cs2_common", REPO / "experiments" / "lib" / "cs2_common.py")
cs2c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs2c)

torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--batch", type=int, default=4)
args = ap.parse_args()
MODEL, B = args.model, args.batch
N_SEQ = 32
out_dir = REPO / "results" / "app2" / MODEL
out_dir.mkdir(parents=True, exist_ok=True)

# ---- bands from the validated streamed weights --------------------------
W = stream_extract(MODEL)
assert not W["learned_pos"], "this runner is for rotary models"
d, L, Hh = W["d"], W["L"], W["H"]
S = torch.zeros(d, d, dtype=torch.float64, device="cuda")
for key in ("Q", "K", "V", "O"):
    for l in range(L):
        X = W[key][l].to("cuda")                     # [H, d, dh]
        G = torch.matmul(X, X.transpose(1, 2))       # [H, d, d] fp32
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
print("bands computed", flush=True)

# ---- HF model straight to GPU -------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_name = f"EleutherAI/{MODEL}" if MODEL.startswith("pythia") else MODEL
tok = AutoTokenizer.from_pretrained(hf_name)
model = AutoModelForCausalLM.from_pretrained(
    hf_name, torch_dtype=torch.float32, device_map={"": 0}).eval()
layers = model.gpt_neox.layers
final_ln = model.gpt_neox.final_layer_norm
print(f"model on gpu: {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

# ---- corpora (cs2_common recipe, HF tokenizer) --------------------------
bos = tok.bos_token_id if tok.bos_token_id is not None \
    else (tok.eos_token_id or 0)
nv = min(model.config.vocab_size, len(tok))
special = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
vocab = torch.tensor([t for t in range(nv) if t not in special])
g = torch.Generator().manual_seed(0)
rt = torch.cat([torch.full((N_SEQ, 1), bos),
                vocab[torch.randint(len(vocab), (N_SEQ, 127), generator=g)]],
               1).cuda()
blk = vocab[torch.randint(len(vocab), (N_SEQ, 128), generator=g)]
ind = torch.cat([torch.full((N_SEQ, 1), bos),
                 blk.repeat(1, 2)[:, :255]], 1).cuda()
from datasets import load_dataset
ds = load_dataset("NeelNanda/pile-10k", split="train")
rows = []
for doc in ds:
    t = tok(doc["text"])["input_ids"]
    if len(t) >= 255:
        rows.append(t[:255])
    if len(rows) == N_SEQ:
        break
pile = torch.cat([torch.full((N_SEQ, 1), bos), torch.tensor(rows)], 1).cuda()

# ---- residual-stream hooks ----------------------------------------------
PROJ = {"V": None}

def pre_hook(mod, hook_args, hook_kwargs):
    V = PROJ["V"]
    if V is None:
        return None
    h = hook_args[0]
    h = h - (h @ V) @ V.T
    return (h, *hook_args[1:]), hook_kwargs

handles = [ly.register_forward_pre_hook(pre_hook, with_kwargs=True)
           for ly in layers]
handles.append(final_ln.register_forward_pre_hook(pre_hook,
                                                  with_kwargs=True))

def mid_layer_states(toks):
    """Mean-free position-profile input: capture layer-MIDL input."""
    cap = {}
    def catch(mod, a, k):
        cap["h"] = a[0].detach()
        return None
    h = layers[MIDL].register_forward_pre_hook(catch, with_kwargs=True)
    outs = []
    for i in range(0, N_SEQ, B):
        model(rt[i:i+B] if toks is rt else toks[i:i+B])
        outs.append(cap["h"].float().cpu())
    h.remove()
    return torch.cat(outs)

MIDL = model.config.num_hidden_layers // 2

def readout(V):
    PROJ["V"] = V
    n1 = n2 = tot = 0.0
    nb = N_SEQ // B
    for i in range(0, N_SEQ, B):
        lp = F.log_softmax(model(ind[i:i+B]).logits.float(), -1)
        nl = -lp[:, :-1].gather(-1, ind[i:i+B, 1:][..., None]).squeeze(-1)
        n1 += float(nl[:, :127].mean()) / nb
        n2 += float(nl[:, 128:].mean()) / nb
        del lp, nl
        lp = F.log_softmax(model(pile[i:i+B]).logits.float(), -1)
        tot += float((-lp[:, :-1].gather(-1, pile[i:i+B, 1:][..., None])
                      .squeeze(-1)).mean()) / nb
        del lp
    PROJ["V"] = None
    return dict(gain=n1 - n2, pile=tot)

# ---- shares, conditions --------------------------------------------------
prof = mid_layer_states(rt).mean(0).double().numpy()     # [pos, d]
prof = prof - prof.mean(0)
tp = np.linalg.norm(prof) ** 2
WE_np = model.gpt_neox.embed_in.weight.detach().float().cpu().double().numpy()
te = np.linalg.norm(WE_np) ** 2
shares = {}
for b in range(10):
    v = vecs[:, b]
    shares[b] = (float(np.linalg.norm(prof @ v) ** 2 / tp * d),
                 float(np.linalg.norm(WE_np @ v) ** 2 / te * d))
print("band (pos x, tok x):", {b: (round(p, 1), round(t, 1))
                               for b, (p, t) in shares.items()}, flush=True)
rule_rank = cs2c.rank_bands(shares)
energy_pair = sorted(sorted(shares, key=lambda b: -shares[b][0])[:2])
print("rule ranking (position specificity):", rule_rank,
      "| pos-energy top-2:", energy_pair, flush=True)

def band_sub(pair):
    return torch.tensor(vecs[:, sorted(pair)].copy(),
                        dtype=torch.float32).cuda()

def orth(M):
    k = M.shape[1]
    return torch.tensor(np.linalg.qr(M)[0][:, :k].copy(),
                        dtype=torch.float32).cuda()

acts = mid_layer_states(pile).reshape(-1, d).double().numpy()
rms = np.sqrt((acts ** 2).mean(0))
outlier2 = np.argsort(rms)[::-1][:2].tolist()
acts = acts - acts.mean(0)
apca = np.linalg.eigh(acts.T @ acts)[1][:, ::-1][:, :2]
del acts

def coord_sub(dims):
    V = torch.zeros(d, len(dims))
    for j, dd in enumerate(dims):
        V[dd, j] = 1.0
    return V.cuda()

conds = {}
for k in (2, 3, 4):
    conds[f"rule_top{k}"] = band_sub(rule_rank[:k])
conds.update({
    f"bands_{energy_pair[0]}_{energy_pair[1]}_posenergy":
        band_sub(energy_pair),
    f"act_pca_L{MIDL}_top2": orth(apca),
    f"outlier_coords_{outlier2[0]}_{outlier2[1]}": coord_sub(outlier2),
})
rng = np.random.default_rng(0)
for s in range(5):
    conds[f"rand{s}_2dim"] = orth(rng.standard_normal((d, 2)))

out = {"model": MODEL, "batch": B, "runner": "hf-native",
       "band_shares": {str(b): shares[b] for b in shares},
       "rule_ranking": rule_rank, "outlier_coords": outlier2}
clean = readout(None)
out["clean"] = clean
print(f"clean gain {clean['gain']:.2f} | clean pile {clean['pile']:.3f}",
      flush=True)
for k, V in conds.items():
    r = readout(V)
    out[k] = dict(gain=r["gain"], pile=r["pile"],
                  gain_destroyed=1 - r["gain"] / clean["gain"],
                  dnll=r["pile"] - clean["pile"], n_dims=int(V.shape[1]))
    o = out[k]
    print(f"{k:40s} dims {o['n_dims']} | destroyed "
          f"{o['gain_destroyed']:7.1%} | dNLL {o['dnll']:+6.2f}", flush=True)

for h in handles:
    h.remove()
(out_dir / "cs2_dose.json").write_text(json.dumps(out, indent=2))
print(f"saved {out_dir / 'cs2_dose.json'}")
