"""First-block and second-block losses behind the induction gains of the
deletion table (Application 2). Reads results/app2/cs2_pilot_*_seed0.json
and results/app2/<model>/cs2_dose.json, writes results/app2/block_losses.json
and prints LaTeX rows: model, condition, first-block loss, second-block
loss, induction gain, each before (clean) and after the deletion.

Usage: python experiments/app2_block_losses.py
"""
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A2 = REPO / "results" / "app2"
PILOT = ["gpt2", "gpt2-medium", "gpt2-large", "pythia-160m"]
DOSE = ["pythia-2.8b", "pythia-6.9b"]


def pick(d, pred):
    ks = [k for k in d if pred(k)]
    assert len(ks) == 1, ks
    return d[ks[0]]


def rows_for(name, d, rw_key, pca_key, out_key, rand_pred):
    clean = d["clean"]
    conds = [("RW-PCA (ours)", pick(d, rw_key)), ("Activation PCA", pick(d, pca_key)),
             ("Outlier dims", pick(d, out_key))]
    rnd = [d[k] for k in d if rand_pred(k)]
    res = {"clean": dict(first=clean["loss_first"], second=clean["loss_second"],
                         gain=clean["gain"])}
    tex = [f"{name} & clean & ${clean['loss_first']:.2f}$ & ${clean['loss_second']:.2f}$ & ${clean['gain']:.2f}$ \\\\"]
    for lab, c in conds:
        res[lab] = dict(first=c["loss_first"], second=c["loss_second"], gain=c["gain"])
        tex.append(f" & {lab} & ${c['loss_first']:.2f}$ & ${c['loss_second']:.2f}$ & ${c['gain']:.2f}$ \\\\")
    if rnd:
        f = statistics.median(c["loss_first"] for c in rnd)
        s = statistics.median(c["loss_second"] for c in rnd)
        g = statistics.median(c["gain"] for c in rnd)
        res["Random (median)"] = dict(first=f, second=s, gain=g)
        tex.append(f" & Random (median) & ${f:.2f}$ & ${s:.2f}$ & ${g:.2f}$ \\\\")
    return res, tex


out, tex = {}, []
for m in PILOT:
    f = A2 / f"cs2_pilot_{m}_seed0.json"
    if not f.exists():
        print("missing", f); continue
    d = json.load(open(f))
    r, t = rows_for(m, d, lambda k: k.endswith("_specific"),
                    lambda k: k.startswith("act_pca"),
                    lambda k: k.startswith("outlier_coords_") and k.count("_") == 3,
                    lambda k: k.startswith("rand"))
    out[m] = r; tex += t + ["\\midrule"]
for m in DOSE:
    f = A2 / m / "cs2_dose.json"
    if not f.exists():
        print("missing", f); continue
    d = json.load(open(f))
    r, t = rows_for(m, d, lambda k: k == "rule_top2",
                    lambda k: k.startswith("act_pca"),
                    lambda k: k.startswith("outlier_coords_"),
                    lambda k: k.startswith("rand") and k.endswith("_2dim"))
    out[m] = r; tex += t + ["\\midrule"]
(A2 / "block_losses.json").write_text(json.dumps(out, indent=1))
print("\n".join(tex[:-1]))
