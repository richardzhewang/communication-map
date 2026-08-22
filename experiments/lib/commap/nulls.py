"""Null distributions (decision D2, revised v0.1).

Two null regimes, chosen per family by SIGNAL DENSITY:

1. SPARSE-signal families (head_head_*): empirical bulk null (Efron central
   matching). With true edges sparse, the central bulk of the within-stratum
   statistic distribution is the null; fit median/MAD, score outliers.
   Anisotropy-preserving by construction. Strata = layer span.

2. DENSE-signal families (emb_*, pos_*, head_unembed): the v0 run showed the
   bulk null FAILS here -- e.g. most layer-1/2 heads genuinely read POS
   (stratum median 0.14 vs 0.03 elsewhere), so the "bulk" absorbs the signal
   and a 12-observation stratum has no power (the R4 artifact). v0.1 uses the
   HAAR ROTATION null for these families instead: rotate the interface matrix,
   recompute the statistic, moment-fit z = (obs - mean_rot)/sd_rot. This is
   the isotropic null -- anti-conservative under anisotropy -- so interface-class
   discoveries should be read with effect sizes (they are typically enormous:
   observed 0.18 vs rotation sd ~5e-4). Rotations are SHARED across all
   readers of a interface-matrix writer, so the cost is a handful of [d,d] conjugations
   plus flat matmuls, not per-edge Monte Carlo.

The edges table records which regime produced each p-value in `null_kind`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .weights import Weights

MAD_TO_SD = 1.4826  # consistency factor: MAD -> sd under Gaussian

INTERFACE_CLASSES = (
    "emb_head_K", "emb_head_Q", "emb_head_V",
    "pos_head_K", "pos_head_Q", "pos_head_V",
    "head_unembed",
)


def stratum_of(df: pd.DataFrame) -> pd.Series:
    """Stratum label per edge: span for head-head, l_r / l_w for interface classes."""
    span = (df["l_r"] - df["l_w"]).astype(int)
    out = "span" + span.astype(str)
    out = out.where(~df["cls"].str.startswith(("emb_", "pos_")), "lr" + df["l_r"].astype(str))
    out = out.where(df["cls"] != "head_unembed", "lw" + df["l_w"].astype(str))
    return out


def add_empirical_null(
    df: pd.DataFrame, stat_col: str = "stat", z_col: str = "z", p_col: str = "p"
) -> pd.DataFrame:
    """Empirical bulk null per cls x stratum (primary for head_head_*)."""
    df = df.copy()
    if "stratum" not in df.columns:
        df["stratum"] = stratum_of(df)
    med = df.groupby(["cls", "stratum"])[stat_col].transform("median")
    mad = (df[stat_col] - med).abs().groupby([df["cls"], df["stratum"]]).transform("median")
    sd = (MAD_TO_SD * mad).clip(lower=1e-12)
    df[z_col] = (df[stat_col] - med) / sd
    df[p_col] = sps.norm.sf(df[z_col])
    return df


# --------------------------------------------- shared-rotation interface nulls


def _haar(d: int, rng: np.random.Generator) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((d, d)))[0]  # [d, d]


def _frob_rotation_moments(
    G_stack: np.ndarray, Hb: np.ndarray, n_rot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Frobenius statistic rotation-null moments, one interface-matrix Gram vs many
    reader Grams, rotations shared.

    G_stack [n_r, d, d] float32 reader Grams (callers concatenate ALL slots
    into one stack so the Haar draws + conjugations are paid once); Hb [d, d]
    interface-matrix Gram. Returns (mean [n_r], sd [n_r]) of the normalized stat."""
    n_r, d, _ = G_stack.shape
    H_rot = np.empty((n_rot, d * d), dtype=np.float32)
    Hb32 = Hb.astype(np.float32)
    for i in range(n_rot):
        Q = _haar(d, rng).astype(np.float32)
        H_rot[i] = (Q @ Hb32 @ Q.T).reshape(-1)
    raw = G_stack.reshape(n_r, d * d) @ H_rot.T                 # [n_r, n_rot]
    np.maximum(raw, 0.0, out=raw)
    tr_G = np.einsum("rii->r", G_stack).astype(np.float64)      # [n_r]
    tr_H = float(np.trace(Hb))                                  # rotation-invariant
    C = np.sqrt(raw.astype(np.float64) / (tr_G[:, None] * tr_H))
    return C.mean(axis=1), C.std(axis=1)


def add_interface_rotation_null(
    df: pd.DataFrame, w: Weights, n_rot_frob: int = 500, seed: int = 0,
) -> pd.DataFrame:
    """Replace z/p for INTERFACE_CLASSES rows with shared-rotation
    Haar-null values; tag null_kind ('bulk' vs 'rotation')."""
    from .edges import interface_grams, head_grams

    df = df.copy()
    df["null_kind"] = np.where(df["cls"].isin(INTERFACE_CLASSES), "rotation", "bulk")
    rng = np.random.default_rng(seed)
    hg = head_grams(w)
    bg = interface_grams(w)
    labels = [f"L{l}H{h}" for l in range(12) for h in range(12)]

    reader_stacks = {"K": hg["G_K"], "Q": hg["G_Q"], "V": hg["G_V"]}

    def apply(cls: str, mean, sd) -> None:
        idx = df.index[df["cls"] == cls]
        order = df.loc[idx, "reader" if not cls.startswith("head_") else "writer"]
        pos = order.map({lab: i for i, lab in enumerate(labels)}).to_numpy()
        z = (df.loc[idx, "stat"].to_numpy() - mean[pos]) / np.maximum(sd[pos], 1e-12)
        df.loc[idx, "z"] = z
        df.loc[idx, "p"] = sps.norm.sf(z)

    slots = list(reader_stacks.keys())                           # ["K", "Q", "V"]
    G_all = np.concatenate([reader_stacks[s] for s in slots])    # [432, d, d]

    for wname, Hb in [("emb", bg["H_emb"].astype(np.float64)),
                      ("pos", bg["H_pos"].astype(np.float64))]:
        mean_all, sd_all = _frob_rotation_moments(G_all, Hb, n_rot_frob, rng)
        for j, slot in enumerate(slots):
            apply(f"{wname}_head_{slot}",
                  mean_all[144 * j : 144 * (j + 1)], sd_all[144 * j : 144 * (j + 1)])

    # head_unembed: rotate the READER Gram G_U against the 144 writer Grams
    G_U = bg["G_unemb"].astype(np.float64)
    mean, sd = _frob_rotation_moments(hg["H_ov"], G_U, n_rot_frob, rng)
    apply("head_unembed", mean, sd)
    return df


# ------------------------------------------------------- isotropic diagnostics


def haar_rotation_null(
    G: np.ndarray, Hm: np.ndarray, n_rot: int = 500, seed: int = 0
) -> np.ndarray:
    """Isotropic baseline (core-research-explainer.tex S3.6) for ONE edge:
    [n_rot] null stats."""
    d = G.shape[0]
    rng = np.random.default_rng(seed)
    tr_G, tr_H = np.trace(G), np.trace(Hm)
    out = np.empty(n_rot)
    for i in range(n_rot):
        Qr = _haar(d, rng)
        Hr = Qr @ Hm @ Qr.T
        out[i] = np.sqrt(max(np.sum(G * Hr), 0.0) / (tr_G * tr_H))
    return out


def widening_report(df: pd.DataFrame, w, n_pairs: int = 8, n_rot: int = 200,
                    seed: int = 0) -> pd.DataFrame:
    """Empirical vs isotropic null widths for random head_head_K pairs (D2:
    the widening factor is a reported finding; v0 measured 6-28x)."""
    from .edges import head_grams

    hg = head_grams(w)
    rng = np.random.default_rng(seed)
    hh = df[df["cls"] == "head_head_K"]
    rows = []
    for _ in range(n_pairs):
        e = hh.sample(1, random_state=rng.integers(1 << 31)).iloc[0]
        r = int(e["l_r"]) * 12 + int(e["reader"].split("H")[1])
        wi = int(e["l_w"]) * 12 + int(e["writer"].split("H")[1])
        iso = haar_rotation_null(
            hg["G_K"][r].astype(np.float64), hg["H_ov"][wi].astype(np.float64),
            n_rot=n_rot, seed=int(rng.integers(1 << 31)),
        )
        bulk = hh[hh["stratum"] == e["stratum"]]["stat"]
        emp_sd = MAD_TO_SD * (bulk - bulk.median()).abs().median()
        rows.append(
            dict(writer=e["writer"], reader=e["reader"], stratum=e["stratum"],
                 iso_mean=iso.mean(), iso_sd=iso.std(),
                 emp_med=bulk.median(), emp_sd=emp_sd,
                 widening=emp_sd / iso.std() if iso.std() > 0 else np.nan)
        )
    return pd.DataFrame(rows)
