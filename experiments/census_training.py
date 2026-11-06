"""Table 2 at random initialization and across training checkpoints.

Reads results/map/<tag>/theory_census.json for the tags below (built by
map_build.py with --random-init or --revision stepN) and writes
results/verification/census_training.json, the LaTeX rows for the paper
table, and results/figures/fig_census_training.png (share of head pairs
above and below chance per channel against training step).

Usage: python experiments/census_training.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
MAP = REPO / "results" / "map"
SERIES = {
    "gpt2": [("gpt2-init", "init"), ("gpt2", "trained")],
    "pythia-160m": [("pythia-160m@step0", 0), ("pythia-160m@step1000", 1000),
                    ("pythia-160m@step4000", 4000),
                    ("pythia-160m@step16000", 16000),
                    ("pythia-160m@step64000", 64000),
                    ("pythia-160m", 143000)],
    # pythia-2.8b: only branches with sharded checkpoint files carry
    # genuine checkpoints (the single-file weights on other step branches
    # are the final weights); see checkpoint notes in map_build.load_tl.
    # pythia-2.8b: the step0 and step16000 branches of the hub repository
    # carry the FINAL weights in every file (census identical to the
    # released model to all digits), so they are excluded; steps 1000,
    # 4000 (sharded checkpoints) and 128000 differ from the final model.
    "pythia-2.8b": [("pythia-2.8b@step1000", 1000),
                    ("pythia-2.8b@step4000", 4000),
                    ("pythia-2.8b@step128000", 128000),
                    ("pythia-2.8b", 143000)],
}
CH = [("head_head_K", "K"), ("head_head_Q", "Q"), ("head_head_V", "V")]


def row(tag):
    f = MAP / tag / "theory_census.json"
    if not f.exists():
        return None
    c = json.load(open(f))["channels"]
    return {k: dict(above=c[k]["above2"], below=c[k]["below2"],
                    mz_above=c[k]["median_z_above"],
                    mz_below=c[k]["median_z_below"]) for k, _ in CH}


def fmt(x):
    return "--" if x is None else f"{x:+.0f}"


out, tex = {}, []
for model, pts in SERIES.items():
    out[model] = []
    for tag, step in pts:
        r = row(tag)
        if r is None:
            print(f"missing {tag}")
            continue
        out[model].append(dict(tag=tag, step=step, channels=r))
        cells = " & ".join(
            f"${r[k]['above']*100:.0f}\\%\\,({fmt(r[k]['mz_above'])})$ / "
            f"${r[k]['below']*100:.0f}\\%\\,({fmt(r[k]['mz_below'])})$"
            for k, _ in CH)
        label = f"{model} ({step})" if step != "trained" else model
        tex.append(f"{label} & {cells} \\\\")
(REPO / "results" / "verification").mkdir(exist_ok=True, parents=True)
(REPO / "results" / "verification" / "census_training.json").write_text(
    json.dumps(out, indent=1))
print("\n".join(tex))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, model in zip(axes, ["pythia-160m", "pythia-2.8b"]):
    pts = [p for p in out.get(model, []) if isinstance(p["step"], int)]
    if not pts:
        continue
    steps = [max(p["step"], 1) for p in pts]
    for k, lab in CH:
        ax.plot(steps, [p["channels"][k]["above"] * 100 for p in pts],
                "o-", label=f"{lab} above chance")
        ax.plot(steps, [p["channels"][k]["below"] * 100 for p in pts],
                "s--", label=f"{lab} below chance")
    ax.set_xscale("log")
    ax.set_xlabel("training step (0 plotted at 1)")
    ax.set_title(model)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("share of head pairs (%)")
axes[1].legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(REPO / "results" / "figures" / "fig_census_training.png", dpi=160)
print("saved figure")
