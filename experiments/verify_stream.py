"""Validate stream_extract against the TL-loaded path on a small model.

Compares every extracted stack elementwise; both paths read the same fp16
checkpoint bits, so differences should be fp32 op-order noise only.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from scale_map import extract, load_tl, stream_extract  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "pythia-160m"
model = load_tl(name)
W_tl = extract(model)
for p in model.parameters():
    p.data = torch.empty(0)
del model
torch.cuda.empty_cache()
W_st = stream_extract(name)

ok = True
for k in ("Q", "K", "V", "O", "Win", "Wout", "H_emb", "G_unemb"):
    a, b = W_tl[k], W_st[k]
    if a.shape != b.shape:
        print(f"{k}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        ok = False
        continue
    scale = a.abs().max().item()
    diff = (a - b).abs().max().item()
    rel = diff / max(scale, 1e-12)
    flag = "OK " if rel < 1e-4 else "FAIL"
    if rel >= 1e-4:
        ok = False
    print(f"{k:8s} max|a| {scale:10.4f}  max|diff| {diff:.3e}  "
          f"rel {rel:.3e}  {flag}")
print("VALIDATED" if ok else "MISMATCH", name)
