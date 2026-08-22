"""Robustness of the band-selection rule (paper Section 5).

The rule ranks the pooled eigendirections by position specificity, the
ratio p/tau of positional to token coupling (cs2_common.rank_bands), and
deletes the top 2. This script checks the rule against two variants on
the stored per-band share tables of all six models: the damped ratio
p/max(tau, 1) and a legacy thresholded form (top-2 by positional share
among bands with token share < 2x chance). All three select the
identical pair on every model, so the rule carries no hidden threshold.
It also reports the top-2 by positional share alone: dropping the token
denominator changes the selection on GPT-2-large, where that pair
destroys only 31% of the induction gain (cs2_pilot_gpt2-large JSON),
so the denominator is doing real work.

Usage: uv run python experiments/app2_rule_robustness.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FILES = {
    "gpt2": "results/app2/cs2_pilot_gpt2_seed0.json",
    "gpt2-medium": "results/app2/cs2_pilot_gpt2-medium_seed0.json",
    "gpt2-large": "results/app2/cs2_pilot_gpt2-large_seed0.json",
    "pythia-160m": "results/app2/cs2_pilot_pythia-160m_seed0.json",
    "pythia-2.8b": "results/app2/pythia-2.8b/cs2_dose.json",
    "pythia-6.9b": "results/app2/pythia-6.9b/cs2_dose.json",
}

all_agree = True
print(f"{'model':14s} {'rule p/tau':>10s} {'p/max(tau,1)':>14s} "
      f"{'legacy gate':>12s} {'pos-only':>10s}")
for m, f in FILES.items():
    d = json.loads((REPO / f).read_text())
    sh = {int(b): v for b, v in d["band_shares"].items()}
    rule = sorted(sorted(sh, key=lambda b: -sh[b][0] / max(sh[b][1],
                                                           1e-12))[:2])
    maxf = sorted(sorted(sh, key=lambda b: -sh[b][0] / max(sh[b][1],
                                                           1.0))[:2])
    legacy = sorted(sorted([b for b in sh if sh[b][1] < 2.0],
                           key=lambda b: -sh[b][0])[:2])
    posonly = sorted(sorted(sh, key=lambda b: -sh[b][0])[:2])
    agree = rule == maxf == legacy
    all_agree &= agree
    note = "ok" if agree else "DIFFER"
    if posonly != rule:
        note += "  (pos-only diverges)"
    print(f"{m:14s} {str(rule):>10s} {str(maxf):>14s} {str(legacy):>12s} "
          f"{str(posonly):>10s}  {note}")
print("ALL AGREE" if all_agree else "DISAGREEMENT FOUND")
