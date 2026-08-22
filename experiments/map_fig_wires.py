"""Main-text figure for Finding 4: the read-write |cosine| distribution
over all 622,854,144 neuron->neuron candidate pairs (l_w < l_r), computed
with the pipeline's weight conventions (fold_ln, centered writers), against
the exact chance law for random directions in d=768
(|cos| density 2(1-c^2)^((d-3)/2)/B(1/2,(d-1)/2), Beta(1/2,(d-1)/2) in c^2).

Run: uv run --with matplotlib python experiments/s3b_fig_nn_cosine_hist.py
"""
import sys

sys.path.insert(0, "experiments/lib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import betaln

from commap.neurons import _unit_rows, load_neurons
from commap.weights import load_model

D = 768
BINS = 500

model = load_model("gpt2")
nw = load_neurons(model)
dev = "cuda" if torch.cuda.is_available() else "cpu"
W = torch.tensor(_unit_rows(nw.W_out_w), dtype=torch.float32, device=dev)
R = torch.tensor(_unit_rows(nw.W_in_r), dtype=torch.float32, device=dev)

hist = np.zeros(BINS)
mx, n_tot = 0.0, 0
for lw in range(W.shape[0]):
    for lr in range(lw + 1, W.shape[0]):
        c = (W[lw] @ R[lr].T).abs().cpu()
        hist += torch.histogram(c, bins=BINS, range=(0.0, 1.0))[0].numpy()
        mx = max(mx, float(c.max()))
        n_tot += c.numel()
n_wires = int(hist[np.arange(BINS) >= BINS // 2].sum())
print(f"pairs {n_tot} | max |cos| {mx:.3f} | wires (>0.5) {n_wires}")

centers = (np.arange(BINS) + 0.5) / BINS
width = 1.0 / BINS
logf = (np.log(2) + ((D - 3) / 2) * np.log1p(-centers**2)
        - betaln(0.5, (D - 1) / 2))
null_counts = np.exp(logf) * width * n_tot

fig, ax = plt.subplots(figsize=(8.6, 3.9))
ax.bar(centers, np.maximum(hist, 0.1), width=width, color="#4C72B0",
       label=f"observed ({n_tot/1e6:.0f}M candidate pairs)")
ax.plot(centers, null_counts, color="#C44E52", lw=2,
        label="chance, expected counts (random directions, $d=768$)")
ax.axvline(0.23, color="0.4", ls=":", lw=1.4)
ax.axvline(0.5, color="0.1", ls="--", lw=1.4)
ax.text(0.237, 3e5,
        "chance ceiling $0.23$\n(95th percentile of the\nnull maximum)",
        fontsize=9, color="0.3")
ax.text(0.507, 3e3,
        f"wire threshold $0.5$\n{n_wires:,} pairs beyond,\nreaching {mx:.2f}",
        fontsize=9)
ax.set_yscale("log")
ax.set_ylim(0.5, 3e8)
ax.set_xlim(0, 1.0)
ax.set_xlabel("read–write $|\\cos|$ (writer's write vector vs. "
              "reader's read vector)")
ax.set_ylabel("neuron pairs")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig("results/figures/fig_nn_cosine_hist.png", dpi=200,
            bbox_inches="tight")
print("saved results/figures/fig_nn_cosine_hist.png")
