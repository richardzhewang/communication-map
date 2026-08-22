"""Robustness of the position profile to its sample size (paper App. D).

Rebuilds the rotary position profile of pythia-160m from n = 4..128
random-token sequences (three fresh seeds per size, plus the frozen
seed 0 at n = 32) and applies the frozen selection rule to each. Result:
every run selects the identical pair, the selected bands' positional
shares stay at 125x chance or more, and no unselected band exceeds 28x,
so the frozen n = 32 carries a large margin.

Usage: .venv/bin/python experiments/s7c_profile_samplesize.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from map_build import stream_extract  # noqa: E402

torch.set_grad_enabled(False)
MODEL = "pythia-160m"

# ---- bands from the validated streamed weights --------------------------
W = stream_extract(MODEL)
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
vecs = torch.linalg.eigh(S.cpu())[1].flip(1).numpy()[:, :10]
del S, W, He, Gu
print("bands computed", flush=True)

# ---- HF model, cs2 corpus recipe ----------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(f"EleutherAI/{MODEL}")
model = AutoModelForCausalLM.from_pretrained(
    f"EleutherAI/{MODEL}", torch_dtype=torch.float32,
    device_map={"": 0}).eval()
layers = model.gpt_neox.layers
MIDL = model.config.num_hidden_layers // 2

bos = tok.bos_token_id if tok.bos_token_id is not None \
    else (tok.eos_token_id or 0)
nv = min(model.config.vocab_size, len(tok))
special = {i for i in (tok.bos_token_id, tok.eos_token_id) if i is not None}
vocab = torch.tensor([t for t in range(nv) if t not in special])

WE = model.gpt_neox.embed_in.weight.detach().float().cpu().double().numpy()
te = np.linalg.norm(WE) ** 2
tau = [float(np.linalg.norm(WE @ vecs[:, b]) ** 2 / te * d)
       for b in range(10)]
print("tau_hat:", [round(t, 2) for t in tau], flush=True)


def profile(n, seed):
    g = torch.Generator().manual_seed(seed)
    rt = torch.cat(
        [torch.full((n, 1), bos),
         vocab[torch.randint(len(vocab), (n, 127), generator=g)]],
        1).cuda()
    cap = {}

    def catch(mod, a, k):
        cap["h"] = a[0].detach()
        return None

    h = layers[MIDL].register_forward_pre_hook(catch, with_kwargs=True)
    outs = []
    B = 8
    for i in range(0, n, B):
        model(rt[i:i + B])
        outs.append(cap["h"].float().cpu())
    h.remove()
    prof = torch.cat(outs).mean(0).double().numpy()
    return prof - prof.mean(0)


gated = [b for b in range(10) if tau[b] < 2.0]
pairs = set()
for n in (4, 8, 16, 32, 64, 128):
    for seed in (100, 200, 300) if n != 32 else (0, 100, 200, 300):
        prof = profile(n, seed)
        tp = np.linalg.norm(prof) ** 2
        p = [float(np.linalg.norm(prof @ vecs[:, b]) ** 2 / tp * d)
             for b in range(10)]
        pair = tuple(sorted(sorted(gated, key=lambda b: -p[b])[:2]))
        pairs.add(pair)
        print(f"n={n:>3} seed={seed:>3}  pair={list(pair)}  "
              f"p_hat={[round(x, 1) for x in p]}", flush=True)

print("all runs select the identical pair:", len(pairs) == 1, sorted(pairs))
