"""Application 1, step 1: select the head-to-head edges.

Reads the raw head-head C tables from the general map (map_build.py) and
applies Application 1's selection standard: within each stratum (channel
class x layer span), the robust z against the stratum's empirical null
distribution (median / MADN), the upper-tail p-value, and per-class
Benjamini-Hochberg at q = 0.05. Head classes only; nothing else in the
map is selected.

Usage: uv run python experiments/app1_select.py MODEL [--q 0.05]
Reads  results/map/{MODEL}/head_C.csv.gz
Writes results/app1/{MODEL}/edges_heads.csv.gz, selection.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "lib"))
from commap.fdr import bh_qvalues  # noqa: E402

MAD_TO_SD = 1.4826


def med_mad_z(stat: np.ndarray, groups: np.ndarray):
    """One-sided-upper z with per-stratum central matching."""
    z = np.empty_like(stat)
    for g in np.unique(groups):
        m = groups == g
        med = np.median(stat[m])
        mad = np.median(np.abs(stat[m] - med)) * MAD_TO_SD
        z[m] = (stat[m] - med) / max(mad, 1e-12)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--q", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(REPO / "results" / "map" / args.model / "head_C.csv.gz")
    lay = lambda s: s.str.extract(r"L(\d+)H").astype(int)[0].values
    span = lay(df["reader"]) - lay(df["writer"])

    out_frames = []
    sel = {}
    for cls, g in df.groupby("cls", sort=False):
        m = (df["cls"] == cls).values
        z = med_mad_z(df.loc[m, "stat"].values, span[m])
        p = sps.norm.sf(z)
        qv = bh_qvalues(p)
        sig = qv <= args.q
        out_frames.append(pd.DataFrame(dict(
            cls=cls, writer=df.loc[m, "writer"].values,
            reader=df.loc[m, "reader"].values,
            stat=df.loc[m, "stat"].values, z=z, q=qv, sig=sig)))
        sel[cls] = dict(candidates=int(m.sum()), selected=int(sig.sum()),
                        rate=float(sig.mean()),
                        min_selected_z=float(z[sig].min()) if sig.any()
                        else None)
        print(f"  {cls}: {int(sig.sum())} of {int(m.sum())} selected "
              f"({sig.mean():.1%})")

    out = REPO / "results" / "app1" / args.model
    out.mkdir(parents=True, exist_ok=True)
    edges = pd.concat(out_frames, ignore_index=True)
    edges.to_csv(out / "edges_heads.csv.gz", index=False)
    sel["q"] = args.q
    sel["total_selected"] = int(edges["sig"].sum())
    (out / "selection.json").write_text(json.dumps(sel, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
