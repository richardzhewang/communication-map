"""Precision check behind the fp32 requirement (paper Appendix D): does
half-precision inference move the CS2 readouts (clean induction gain,
pile NLL, one deletion condition) relative to fp32? Runs pythia-160m
both ways with the s5 machinery. Result: bf16 forwards collapse the
clean gain 18.36 -> 10.72 nats (fp16: 16.54); processing-then-casting
is identical to loading in bf16, so the forward pass itself is the
cause. All scale CS2 readouts therefore run fp32."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ru", REPO / "experiments" / "lib" / "cs2_common.py")
ru = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ru)

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

N_SEQ = ru.N_SEQ
MODEL = "pythia-160m"

def build(dtype):
    hf = AutoModelForCausalLM.from_pretrained(f"EleutherAI/{MODEL}")
    if not hasattr(hf, "embed_out"):
        hf.embed_out = getattr(hf, "lm_head", None) or hf.get_output_embeddings()
    m = HookedTransformer.from_pretrained(
        MODEL, hf_model=hf, fold_ln=True, center_writing_weights=True,
        center_unembed=True, device="cuda", dtype=dtype).eval()
    return m

def readout(model, ind, pile, V=None):
    hook_names = [f"blocks.{l}.hook_resid_pre" for l in range(model.cfg.n_layers)] \
        + [f"blocks.{model.cfg.n_layers-1}.hook_resid_post"]
    if V is None:
        hooks = []
    else:
        Vd = V.to(model.cfg.dtype)
        def hook(x, hook):
            return x - (x @ Vd) @ Vd.T
        hooks = [(nm, hook) for nm in hook_names]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        n1 = n2 = tot = 0.0
        for i in range(0, N_SEQ, 8):
            lp = F.log_softmax(model(ind[i:i+8]).float(), -1)
            nl = -lp[:, :-1].gather(-1, ind[i:i+8, 1:][..., None]).squeeze(-1)
            n1 += float(nl[:, :127].mean()) / (N_SEQ // 8)
            n2 += float(nl[:, 128:].mean()) / (N_SEQ // 8)
            lp = F.log_softmax(model(pile[i:i+8]).float(), -1)
            tot += float((-lp[:, :-1].gather(-1, pile[i:i+8, 1:][..., None])
                          .squeeze(-1)).mean()) / (N_SEQ // 8)
    return n1 - n2, tot

results = {}
V_fixed = None
for name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
    model = build(dtype)
    rt, ind, pile = ru.corpora(model, "cuda")
    if V_fixed is None:
        # one fixed deletion subspace: top-2 pooled bands (built in fp32 pass,
        # reused verbatim in bf16 so only the FORWARD precision differs)
        vecs = ru.pooled_bands(model)
        V_fixed = torch.tensor(np.linalg.qr(vecs[:, :2])[0][:, :2].copy(),
                               dtype=torch.float32).cuda()
    g0, p0 = readout(model, ind, pile)
    g1, p1 = readout(model, ind, pile, V_fixed)
    results[name] = dict(clean_gain=g0, clean_pile=p0,
                         del_gain=g1, del_pile=p1,
                         destroyed=1 - g1 / g0, dnll=p1 - p0)
    print(f"[{name}] clean gain {g0:.4f}  pile {p0:.4f} | "
          f"del gain {g1:.4f}  destroyed {1-g1/g0:.4%}  dNLL {p1-p0:+.4f}",
          flush=True)
    del model
    torch.cuda.empty_cache()

a, b = results["fp32"], results["bf16"]
print("\ndeltas (bf16 - fp32):")
for k in a:
    print(f"  {k:12s} {b[k]-a[k]:+.5f}")
