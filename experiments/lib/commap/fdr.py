"""Per-family FDR control (decision D4).

Benjamini-Hochberg is run WITHIN each edge family (= cls; strata are already
standardized by the empirical null, so pooling z-based p-values across strata
within a class is coherent), never once over the pooled edge set: a large
family would otherwise set the rejection threshold for the small ones.

BH at level q: sort p_(1) <= ... <= p_(m); k* = max{k : p_(k) <= k q / m};
reject the k* smallest. q-value of an edge = min over thresholds at which it
would be rejected = min_{j >= rank} m p_(j) / j (monotonized).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values), same order as input. [m]"""
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)          # m p_(k) / k
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]  # monotone from the right
    q = np.empty(m)
    q[order] = np.minimum(q_sorted, 1.0)
    return q


def add_fdr(
    df: pd.DataFrame, q_level: float = 0.05,
    p_col: str = "p", q_col: str = "q", sig_col: str = "sig",
) -> pd.DataFrame:
    """Adds columns: q_col (per-family BH q-value), sig_col (q <= q_level)."""
    df = df.copy()
    df[q_col] = df.groupby("cls")[p_col].transform(lambda s: bh_qvalues(s.to_numpy()))
    df[sig_col] = df[q_col] <= q_level
    return df


# --------------------------------------------------- streaming (binned) BH
#
# v1 (design S8): BH over the 1e9-edge neuron_neuron family never materializes
# or sorts the p-value vector. BH needs only, per candidate threshold, the
# COUNT of p-values below it -- so the stream accumulates a histogram of the
# statistic per stratum, each bin maps to one p-value (via the stratum's null
# fit), and the cutoff comes from bin counts. Discretization: every edge in a
# bin is assigned the bin's p (evaluated at the bin's conservative inner edge),
# so the binned BH is exactly BH on the binned p-values -- the only
# approximation is bin width (2/n_bins in cosine units), not the mechanics.


def bh_from_binned(
    p_bins: np.ndarray, counts: np.ndarray, q_level: float = 0.05
) -> tuple[float, np.ndarray, np.ndarray]:
    """BH cutoff + per-bin q-values from binned p-values.

    p_bins [B] p-value of each bin, counts [B] edges per bin (any order,
    zero-count bins fine). Returns (p_star, p_sorted, q_sorted): reject a bin
    iff its p <= p_star (p_star = -inf if nothing passes); q for a bin =
    np.interp/searchsorted into (p_sorted, q_sorted).
    """
    keep = counts > 0
    order = np.argsort(p_bins[keep], kind="stable")
    p = p_bins[keep][order]
    cum = np.cumsum(counts[keep][order])          # count(p-values <= p_k)
    m = cum[-1]
    ok = p <= (cum / m) * q_level
    p_star = float(p[ok].max()) if ok.any() else -np.inf
    q = np.minimum.accumulate((p * m / cum)[::-1])[::-1]  # monotone from right
    return p_star, p, np.minimum(q, 1.0)


def q_lookup(p_edge: np.ndarray, p_sorted: np.ndarray, q_sorted: np.ndarray) -> np.ndarray:
    """q-value for edges given their (binned) p, via the bh_from_binned table."""
    idx = np.clip(np.searchsorted(p_sorted, p_edge), 0, len(p_sorted) - 1)
    return q_sorted[idx]


def family_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Survivor counts per family (reported per D4)."""
    return (
        df.groupby("cls")
        .agg(n_edges=("stat", "size"), n_sig=("sig", "sum"),
             max_stat=("stat", "max"), max_z=("z", "max"))
        .reset_index()
    )
