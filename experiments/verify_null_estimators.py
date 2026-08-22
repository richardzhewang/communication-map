"""Robust vs moment null fits: the necessity check behind Eq. 6.

The stratum score distribution is a null/signal mixture, so the null
center and scale must be estimated by estimators the signal tail cannot
corrupt (median and normalized MAD; Hampel 1974, Rousseeuw & Croux
1993). This script measures what happens otherwise: refit every
head-head stratum of the GPT-2 map with the stratum mean and standard
deviation, rerun per-class BH at q = 0.05, and compare survivor sets.

Expected outcome (quoted in Section 3): the moment fits let the signal
tail inflate the null scale by up to ~3x, and the majority of the
head-head map's edges are lost. The robust refit must reproduce the
pipeline's survivor counts exactly.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

EDGES = "results/app1/gpt2/edges_heads.csv.gz"
OUT = "results/verification/null_estimator_comparison.json"
CLASSES = ["head_head_K", "head_head_Q", "head_head_V"]
Q = 0.05
MADN = 1.4826  # Gaussian consistency factor: 1 / Phi^-1(3/4)


def bh_mask(p: np.ndarray, q: float) -> np.ndarray:
    n = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, n + 1) / n
    k = 0 if not ok.any() else np.max(np.nonzero(ok)[0]) + 1
    mask = np.zeros(n, bool)
    mask[order[:k]] = True
    return mask


def fits_and_mask(g: pd.DataFrame, how: str):
    p = np.empty(len(g))
    fits = []
    for span, idx in g.groupby("span").groups.items():
        x = g.loc[idx, "stat"].values
        if how == "robust":
            center = np.median(x)
            scale = MADN * np.median(np.abs(x - center))
        else:
            center = x.mean()
            scale = x.std(ddof=1)
        p[np.asarray(idx)] = 1 - stats.norm.cdf((x - center) / scale)
        fits.append(dict(span=int(span), n=int(len(x)),
                         center=float(center), scale=float(scale)))
    return fits, bh_mask(p, Q)


def main() -> None:
    df = pd.read_csv(EDGES)
    lw = df["writer"].str.extract(r"L(\d+)H").astype(int)[0]
    lr = df["reader"].str.extract(r"L(\d+)H").astype(int)[0]
    df["span"] = (lr - lw).values

    summary = {"q": Q, "classes": {}}
    print(f"{'class':<14} {'pipeline':>8} {'robust':>7} {'mean/SD':>8} "
          f"{'overlap':>8} {'lost':>5} {'max scale infl.':>16}")
    for cls in CLASSES:
        g = df[df["cls"] == cls].reset_index(drop=True)
        rob_fits, rob = fits_and_mask(g, "robust")
        mom_fits, mom = fits_and_mask(g, "meansd")
        infl = [m["scale"] / r["scale"] for r, m in zip(rob_fits, mom_fits)]
        row = dict(
            pipeline_sig=int(g["sig"].sum()),
            robust_refit=int(rob.sum()),
            moment_refit=int(mom.sum()),
            overlap=int((rob & mom).sum()),
            lost_by_moment=int(rob.sum() - (rob & mom).sum()),
            robust_matches_pipeline=bool(rob.sum() == g["sig"].sum()),
            max_scale_inflation=float(max(infl)),
            per_stratum=[dict(span=r["span"], n=r["n"],
                              robust_scale=r["scale"],
                              moment_scale=m["scale"],
                              inflation=m["scale"] / r["scale"])
                         for r, m in zip(rob_fits, mom_fits)],
        )
        summary["classes"][cls] = row
        print(f"{cls:<14} {row['pipeline_sig']:>8} {row['robust_refit']:>7} "
              f"{row['moment_refit']:>8} {row['overlap']:>8} "
              f"{row['lost_by_moment']:>5} {row['max_scale_inflation']:>15.2f}x")

    total_rob = sum(r["robust_refit"] for r in summary["classes"].values())
    total_lost = sum(r["lost_by_moment"] for r in summary["classes"].values())
    summary["total_robust"] = total_rob
    summary["total_lost_by_moment"] = total_lost
    summary["max_scale_inflation"] = max(
        r["max_scale_inflation"] for r in summary["classes"].values())
    assert all(r["robust_matches_pipeline"]
               for r in summary["classes"].values()), \
        "robust refit must reproduce the pipeline survivor counts"
    print(f"\ntotal: robust {total_rob}, lost under mean/SD {total_lost} "
          f"({100 * total_lost / total_rob:.0f}%), "
          f"max scale inflation {summary['max_scale_inflation']:.2f}x")
    json.dump(summary, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
