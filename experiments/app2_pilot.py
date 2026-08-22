"""Case-study-2 redesign pilot. argv[1] = model name (default gpt2),
argv[2] = corpus seed for the repeated-block draw (default 0). Projects 2-dim subspaces out
of the residual stream at every layer and measures induction gain + Pile
NLL, for:
  the rule pair    (top-2 by position specificity, the ratio of
                    positional to token coupling; cs2_common.rank_bands),
  map bands {3,5}   (previous selection, GPT-2 only),
  map bands {0,1}   (top-2 by eigenvalue),
  the positional-energy top-2 (baseline: ranking without the token
                    denominator),
  activation PCA top-2 (mid-layer resid_pre on Pile, centered),
  outlier coordinates (top-2 stream dims by activation RMS, and top-1 alone),
  and 5 random 2-dim subspaces.

Usage: uv run --with datasets python experiments/s5_cs2_pilot.py
"""
import importlib.util
import json
import sys
from pathlib import Path

MODEL = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() \
    else "gpt2"
SEED = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 0

import numpy as np
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "ru", Path(__file__).parent / "lib" / "cs2_common.py")
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)

from transformer_lens import HookedTransformer

N_SEQ = ru.N_SEQ
_kw = dict(fold_ln=True, center_writing_weights=True, center_unembed=True,
           device="cuda")
if MODEL.startswith("pythia"):
    from transformers import AutoModelForCausalLM
    _hf = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{MODEL}")
    if not hasattr(_hf, "embed_out"):
        _hf.embed_out = getattr(_hf, "lm_head", None) \
            or _hf.get_output_embeddings()
    model = HookedTransformer.from_pretrained(MODEL, hf_model=_hf,
                                              **_kw).eval()
else:
    model = HookedTransformer.from_pretrained(MODEL, **_kw).eval()
LEARNED_POS = ru.has_learned_pos(model)
cfg = model.cfg
rt, ind, pile = ru.corpora(model, "cuda")
if SEED != 0:
    tok = model.tokenizer
    bos = tok.bos_token_id or tok.eos_token_id or 0
    nv = min(cfg.d_vocab, len(tok))
    sp = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
    vocab = torch.tensor([t for t in range(nv) if t not in sp])
    g = torch.Generator().manual_seed(SEED)
    blk = vocab[torch.randint(len(vocab), (N_SEQ, 128), generator=g)]
    ind = torch.cat([torch.full((N_SEQ, 1), bos),
                     blk.repeat(1, 2)[:, :255]], 1).cuda()
vecs = ru.pooled_bands(model)                       # [768, 10]

# ---- candidate subspaces ------------------------------------------------
# For rotary models (no W_pos) the positional-share analog uses the
# position-profile matrix: mean mid-layer stream state at each position on
# random-token inputs, centered across positions. Fixed rule, no tuning.
MIDL0 = model.cfg.n_layers // 2
if LEARNED_POS:
    Wpos = model.W_pos.detach().double().cpu().numpy()  # [n_ctx, d]
else:
    _nm = f"blocks.{MIDL0}.hook_resid_pre"
    with torch.no_grad():
        _, _c = model.run_with_cache(rt, return_type=None,
                                     names_filter=lambda n: n == _nm)
    Wpos = _c[_nm].mean(0).double().cpu().numpy()       # [pos, d]
    Wpos = Wpos - Wpos.mean(0)
tp = np.linalg.norm(Wpos) ** 2
WE = model.W_E.detach().double().cpu().numpy()
te = np.linalg.norm(WE) ** 2
shares = {}
for b in range(10):
    v = vecs[:, b]
    shares[b] = (np.linalg.norm(Wpos @ v) ** 2 / tp * cfg.d_model,
                 np.linalg.norm(WE @ v) ** 2 / te * cfg.d_model)
print("band (pos x, tok x):", {b: (round(p, 1), round(t, 1))
                               for b, (p, t) in shares.items()})
energy_pair = sorted(sorted(shares, key=lambda b: -shares[b][0])[:2])
spec_pair = sorted(ru.rank_bands(shares)[:2])
print("pos-energy top-2:", energy_pair,
      "| rule (position-specificity) top-2:", spec_pair)

def band_sub(pair):
    return torch.tensor(vecs[:, pair].copy(), dtype=torch.float32).cuda()

def orth(M):
    return torch.tensor(np.linalg.qr(M)[0][:, :2].copy(),
                        dtype=torch.float32).cuda()


MIDL = cfg.n_layers // 2
name6 = f"blocks.{MIDL}.hook_resid_pre"
acts = []
with torch.no_grad():
    for i in range(0, N_SEQ, 8):
        _, c = model.run_with_cache(pile[i:i+8], return_type=None,
                                    names_filter=lambda n: n == name6)
        acts.append(c[name6].reshape(-1, cfg.d_model).float().cpu())
A = torch.cat(acts).double().numpy()
rms = np.sqrt((A ** 2).mean(0))
outlier2 = np.argsort(rms)[::-1][:2].tolist()
A = A - A.mean(0)
apca = np.linalg.eigh(A.T @ A)[1][:, ::-1][:, :2]

def coord_sub(dims):
    V = torch.zeros(cfg.d_model, len(dims))
    for j, dd in enumerate(dims):
        V[dd, j] = 1.0
    return V.cuda()

conds = {
    "clean": None,
}
if MODEL == "gpt2":
    conds["bands_3_5"] = band_sub([3, 5])
conds.update({
    f"bands_{energy_pair[0]}_{energy_pair[1]}_energy": band_sub(energy_pair),
    f"bands_{spec_pair[0]}_{spec_pair[1]}_specific": band_sub(spec_pair),
    f"act_pca_L{MIDL}": orth(apca),
})
if energy_pair != [0, 1] and spec_pair != [0, 1]:
    conds["bands_0_1_topeig"] = band_sub([0, 1])
conds[f"outlier_coords_{outlier2[0]}_{outlier2[1]}"] = coord_sub(outlier2)
conds[f"outlier_coord_{outlier2[0]}"] = coord_sub(outlier2[:1])
rng = np.random.default_rng(0)
for s in range(5):
    conds[f"rand{s}"] = orth(rng.standard_normal((cfg.d_model, 2)))

hook_names = [f"blocks.{l}.hook_resid_pre" for l in range(cfg.n_layers)] \
    + [f"blocks.{cfg.n_layers-1}.hook_resid_post"]

def hooks_for(v):
    if v is None:
        return []
    V = v
    def hook(x, hook):
        return x - (x @ V) @ V.T
    return [(nm, hook) for nm in hook_names]

def readout(v):
    with torch.no_grad(), model.hooks(fwd_hooks=hooks_for(v)):
        n1 = n2 = tot = 0.0
        for i in range(0, N_SEQ, 8):
            lp = F.log_softmax(model(ind[i:i+8]).float(), -1)
            nl = -lp[:, :-1].gather(-1, ind[i:i+8, 1:][..., None]).squeeze(-1)
            n1 += float(nl[:, :127].mean()) / (N_SEQ // 8)
            n2 += float(nl[:, 128:].mean()) / (N_SEQ // 8)
            lp = F.log_softmax(model(pile[i:i+8]).float(), -1)
            tot += float((-lp[:, :-1].gather(-1, pile[i:i+8, 1:][..., None])
                          .squeeze(-1)).mean()) / (N_SEQ // 8)
    return dict(gain=n1 - n2, pile=tot)

rule_mass = float((vecs[outlier2, :][:, spec_pair] ** 2).sum() / 2)
print(f"outlier coords {outlier2} | rule-pair squared mass on them "
      f"{rule_mass:.3f}")
out = {"band_shares": {str(b): shares[b] for b in shares},
       "outlier_coords": outlier2,
       "rule_pair_mass_on_outlier_coords": rule_mass}
clean = readout(None)
out["clean"] = clean
print(f"clean gain {clean['gain']:.2f}")
for k, v in conds.items():
    if k == "clean":
        continue
    r = readout(v)
    out[k] = dict(gain=r["gain"], pile=r["pile"],
                  gain_destroyed=1 - r["gain"] / clean["gain"],
                  dnll=r["pile"] - clean["pile"])
    o = out[k]
    print(f"{k:28s} gain destroyed {o['gain_destroyed']:7.1%} | "
          f"dNLL {o['dnll']:+6.2f}")

Path(f"results/app2/cs2_pilot_{MODEL}_seed{SEED}.json").write_text(
    json.dumps(out, indent=2))
print(f"saved results/app2/cs2_pilot_{MODEL}_seed{SEED}.json")
