"""Tiled neuron->neuron edge stream (v1): the 1e9-cosine class.

Design S8 mechanics, two passes over (l_w, l_r) layer-pair tiles (l_w < l_r,
66 tiles of [d_mlp, d_mlp] = 9.4e6 cosines each; a tile is one GEMM):

  PASS 1  accumulate a per-span histogram of cos(w_out, w_in) (span = l_r -
          l_w is constant within a tile, so tiling by layer pair makes the
          stratification free).
  FIT     Efron central matching per span from the histogram: med = Q50,
          sd = IQR/1.349 (quantiles interpolated from the cumulative
          histogram; bin width 2/n_bins in cosine units bounds the error).
          Isotropic baseline sd = 1/sqrt(d) closed-form (S5.1); the per-span
          widening factor emp_sd * sqrt(d) is reported.
  BH      binned BH over the whole family (fdr.bh_from_binned): map each bin
          to a p-value via its span's fit, cutoff from bin counts. Sidedness:
          TWO-SIDED on |z| by default -- a strong negative alignment is a real
          (sign-flipping) channel. PRE-REGISTRATION NOTE: freeze `two_sided`
          before the first real run.
  PASS 2  recompute tiles, keep edges in significant bins only (per-span
          cosine thresholds derived from the BH cutoff), emit survivors with
          binned z/p/q plus exact z for ranking.

Torch backend: tiles run on --device (cuda on the 5090; cpu works, ~1e12 flop
total for GPT-2 small). Histograms and survivor bookkeeping are numpy on CPU.

Memory: never holds more than one [d_mlp, d_mlp] tile (float32, 38 MB) plus
histograms [n_spans, n_bins]. Survivor arrays are the only growing object; if
a run returns >1e7 survivors the null fit is suspect -- stop and look.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from .fdr import bh_from_binned, q_lookup
from .neurons import NeuronWeights, _unit_rows, neuron_labels

MAD_TO_SD = 1.4826
IQR_TO_SD = 1.349


def _resolve_device(device: str = "auto") -> str:
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SpanFit:
    span: int
    n: int
    med: float
    sd: float          # empirical (central-matching) sd
    widening: float    # emp sd / isotropic sd (= emp_sd * sqrt(d))


def _quantile_from_hist(counts: np.ndarray, edges: np.ndarray, qs) -> np.ndarray:
    """Interpolated quantiles from one histogram (right-edge cumulative)."""
    cum = np.cumsum(counts).astype(np.float64)
    cum /= cum[-1]
    return np.interp(qs, cum, edges[1:])


def _tiles(L: int):
    for lw in range(L):
        for lr in range(lw + 1, L):  # strict: MLP_l -> MLP_l is one component
            yield lw, lr


def stream_neuron_neuron(
    nw: NeuronWeights,
    q_level: float = 0.05,
    n_bins: int = 4001,
    device: str = "auto",
    two_sided: bool = True,
    max_survivors: int = 20_000_000,
    verbose: bool = True,
) -> dict:
    """Full neuron_neuron class, streamed. Returns dict with:
    survivors (DataFrame), span_fits (DataFrame), p_star, m_total, hist,
    edges (bin edges), device."""
    import torch

    dev = _resolve_device(device)
    d = nw.W_in_r.shape[-1]
    L, N = nw.W_out_w.shape[0], nw.W_out_w.shape[1]
    Wout = torch.from_numpy(_unit_rows(nw.W_out_w).astype(np.float32)).to(dev)
    Win = torch.from_numpy(_unit_rows(nw.W_in_r).astype(np.float32)).to(dev)
    edges = np.linspace(-1.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    spans = np.arange(1, L)                       # 1 .. L-1
    hist = np.zeros((len(spans), n_bins), dtype=np.int64)

    def tile_cos(lw: int, lr: int) -> np.ndarray:
        c = Wout[lw] @ Win[lr].T                  # [N, N] on device
        return c.clamp(-1.0, 1.0).cpu().numpy()

    # ---- pass 1: per-span histograms
    for lw, lr in _tiles(L):
        c = tile_cos(lw, lr)
        hist[lr - lw - 1] += np.histogram(c.ravel(), bins=edges)[0]
        if verbose:
            print(f"  pass1 tile ({lw},{lr})", end="\r")
    m_total = int(hist.sum())

    # ---- central-matching fit per span + isotropic widening
    iso_sd = 1.0 / np.sqrt(d)
    fits: list[SpanFit] = []
    for i, s in enumerate(spans):
        q25, q50, q75 = _quantile_from_hist(hist[i], edges, [0.25, 0.5, 0.75])
        sd = max((q75 - q25) / IQR_TO_SD, 1e-12)
        fits.append(SpanFit(int(s), int(hist[i].sum()), q50, sd, sd / iso_sd))

    # ---- binned p-values per (span, bin); conservative inner-edge evaluation
    p_bins = np.empty((len(spans), n_bins))
    for i, f in enumerate(fits):
        lo, hi = edges[:-1], edges[1:]            # bin edges
        # |z| at the edge CLOSER to the median => largest p in the bin
        z_lo, z_hi = (lo - f.med) / f.sd, (hi - f.med) / f.sd
        if two_sided:
            z_in = np.minimum(np.abs(z_lo), np.abs(z_hi))
            z_in[(z_lo < 0) & (z_hi > 0)] = 0.0   # bin straddles the median
            p_bins[i] = 2.0 * sps.norm.sf(z_in)
        else:
            p_bins[i] = sps.norm.sf(z_lo)         # upper tail only
    p_star, p_sorted, q_sorted = bh_from_binned(p_bins.ravel(), hist.ravel(), q_level)

    # ---- per-span cosine acceptance regions from the significant bins
    sig_bins = p_bins <= p_star                   # [n_spans, n_bins]
    thr = []                                      # per span: (lo_thr, hi_thr)
    for i in range(len(spans)):
        up = np.where(sig_bins[i] & (centers > fits[i].med))[0]
        dn = np.where(sig_bins[i] & (centers < fits[i].med))[0]
        hi_thr = edges[up.min()] if up.size else np.inf     # left edge of first sig bin
        lo_thr = edges[dn.max() + 1] if dn.size else -np.inf
        thr.append((lo_thr, hi_thr))

    # ---- pass 2: collect survivors
    cols: dict[str, list] = {k: [] for k in ("l_w", "m_w", "l_r", "m_r", "cos")}
    n_surv = 0
    for lw, lr in _tiles(L):
        lo_thr, hi_thr = thr[lr - lw - 1]
        if not np.isfinite(hi_thr) and not np.isfinite(lo_thr):
            continue
        c = tile_cos(lw, lr)
        keep = (c >= hi_thr) | (c <= lo_thr)
        iw, ir = np.nonzero(keep)
        n_surv += iw.size
        if n_surv > max_survivors:
            raise RuntimeError(
                f"survivor count exceeded {max_survivors:.0g} -- null fit "
                "suspect (dense signal? see the R4/A1 lesson); not collecting.")
        cols["l_w"].append(np.full(iw.size, lw, dtype=np.int16))
        cols["m_w"].append(iw.astype(np.int32))
        cols["l_r"].append(np.full(ir.size, lr, dtype=np.int16))
        cols["m_r"].append(ir.astype(np.int32))
        cols["cos"].append(c[keep])
        if verbose:
            print(f"  pass2 tile ({lw},{lr}) survivors={n_surv}", end="\r")
    cat = {k: (np.concatenate(v) if v else np.array([], dtype=float))
           for k, v in cols.items()}

    # ---- binned z/p/q (consistent with the BH pass) + exact z for ranking
    span_e = (cat["l_r"] - cat["l_w"]).astype(int) if len(cat["cos"]) else np.array([], int)
    med = np.array([f.med for f in fits])
    sd = np.array([f.sd for f in fits])
    z_exact = (cat["cos"] - med[span_e - 1]) / sd[span_e - 1] if len(cat["cos"]) else cat["cos"]
    bin_idx = np.clip(np.digitize(cat["cos"], edges) - 1, 0, n_bins - 1)
    p_edge = p_bins[span_e - 1, bin_idx] if len(cat["cos"]) else cat["cos"]
    q_edge = q_lookup(p_edge, p_sorted, q_sorted) if len(cat["cos"]) else cat["cos"]

    surv = pd.DataFrame({
        "cls": "neuron_neuron",
        "writer": neuron_labels(cat["l_w"].astype(int), cat["m_w"].astype(int))
                  if len(cat["cos"]) else [],
        "l_w": cat["l_w"], "m_w": cat["m_w"],
        "reader": neuron_labels(cat["l_r"].astype(int), cat["m_r"].astype(int))
                  if len(cat["cos"]) else [],
        "l_r": cat["l_r"], "m_r": cat["m_r"],
        "stat": np.abs(cat["cos"]), "cos": cat["cos"],
        "z": z_exact, "p": p_edge, "q": q_edge, "sig": True,
    }).sort_values("q").reset_index(drop=True)

    return dict(
        survivors=surv,
        span_fits=pd.DataFrame([f.__dict__ for f in fits]),
        p_star=p_star, m_total=m_total, hist=hist, edges=edges, device=dev,
        two_sided=two_sided, n_bins=n_bins,
    )
