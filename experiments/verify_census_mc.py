"""Census of head pairs against the theoretical (rotation) null distribution.

Layer-1 measurement of the two-layer framing: for EVERY causally
eligible head-head pair and every channel (K/Q/V-composition), sample
the pair's theoretical null distribution, the coupling C recomputed
under Haar-random rotations of the writer (singular values preserved,
orientation randomized), and standardize the observed coupling
against that distribution on the C^2 scale,

    z_theory = (C_obs^2 - mean_rot[C^2]) / sd_rot[C^2],

matching map_build's closed-form census statistic.

The rotations are shared across all pairs (each pair's ensemble is
still a valid sample from its own rotation null distribution), so one
conjugated writer stack serves all readers and the whole census costs
n_rot batched matrix products on the GPU.

Outputs per channel and per layer span: the fraction of pairs
significantly ABOVE the theoretical null distribution (z >= 2, super-
coupled), significantly BELOW it (z <= -2, anti-coupled), and within
it, plus the same at |z| >= 3. These are the dense-layer numbers of
the paper. A spot check validates the shared-rotation ensembles
against the independent per-pair sampler in lib/commap/nulls.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent / "lib"))
from commap.weights import load_gpt2  # noqa: E402

N_ROT = 500
SEED = 0
OUT = "results/verification/theory_census_mc.json"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
D, DH, NH = 768, 64, 144


def chol_factor(g: np.ndarray) -> np.ndarray:
    """[H,dh,dh] PSD -> Cholesky factors, jittered for rank safety."""
    eye = np.eye(g.shape[-1])
    return np.linalg.cholesky(g + 1e-10 * eye)


def main() -> None:
    torch.manual_seed(SEED)
    w = load_gpt2("gpt2")
    fold = lambda X: X.reshape(NH, D, DH)
    Qf, Kf, Vf, Of = (fold(x) for x in (w.Q, w.K, w.V, w.O))
    g = lambda X: np.einsum("hda,hdb->hab", X, X)
    gQ, gK, gV, gO = g(Qf), g(Kf), g(Vf), g(Of)

    # Reader Grams (materialized, [NH, d, d]) and writer factors [NH, d, dh]:
    #   K-comp reader G = K gQ K^T,  Q-comp G = Q gK Q^T,  V-comp G = V gO V^T
    #   writer H = O gV O^T = F F^T with F = O chol(gV)
    sandwich = lambda X, gm: np.matmul(np.matmul(X, gm), X.transpose(0, 2, 1))
    readers = {
        "head_head_K": sandwich(Kf, gQ),
        "head_head_Q": sandwich(Qf, gK),
        "head_head_V": sandwich(Vf, gO),
    }
    F = np.matmul(Of, chol_factor(gV))  # [NH, d, dh]

    tr_H = np.einsum("hab,hab->h", gO, gV)                       # ||W_OV||^2
    tr_G = {"head_head_K": np.einsum("hab,hab->h", gQ, gK),
            "head_head_Q": np.einsum("hab,hab->h", gQ, gK),
            "head_head_V": tr_H.copy()}

    Ft = torch.tensor(F, dtype=torch.float32, device=DEV)
    Gflat = {c: torch.tensor(readers[c].reshape(NH, D * D),
                             dtype=torch.float32, device=DEV)
             for c in readers}
    # running moments of C per (reader, writer), float64 accumulators
    s1 = {c: torch.zeros(NH, NH, dtype=torch.float64, device=DEV)
          for c in readers}
    s2 = {c: torch.zeros(NH, NH, dtype=torch.float64, device=DEV)
          for c in readers}
    denom = {c: np.sqrt(tr_G[c][:, None] * tr_H[None, :]) for c in readers}
    denom_t = {c: torch.tensor(denom[c], dtype=torch.float64, device=DEV)
               for c in readers}

    gen = torch.Generator(device=DEV).manual_seed(SEED)
    for t in range(N_ROT):
        A = torch.randn(D, D, generator=gen, dtype=torch.float32, device=DEV)
        Qr, R = torch.linalg.qr(A)
        Qr = Qr * torch.sign(torch.diagonal(R))[None, :]
        X = torch.matmul(Qr, Ft)                      # [NH, d, dh] rotated F
        Hrot = torch.matmul(X, X.transpose(1, 2))     # [NH, d, d]
        Hflat = Hrot.reshape(NH, D * D)
        for c in readers:
            num = torch.clamp(Gflat[c] @ Hflat.T, min=0.0).double()
            Csq = num / denom_t[c] ** 2               # [n_readers, n_writers]
            s1[c] += Csq
            s2[c] += Csq * Csq
        if (t + 1) % 100 == 0:
            print(f"  rotation {t + 1}/{N_ROT}")

    layers = np.repeat(np.arange(12), 12)

    # observed couplings from the released map (edge-exact source of truth)
    df = pd.read_csv("results/map/gpt2/head_C.csv.gz")
    idx = lambda s: (s.str.extract(r"L(\d+)H(\d+)").astype(int)
                     .apply(lambda r: r[0] * 12 + r[1], axis=1).values)
    df["wi"], df["ri"] = idx(df["writer"]), idx(df["reader"])
    df["span"] = layers[df["ri"]] - layers[df["wi"]]

    summary = {"n_rot": N_ROT, "seed": SEED, "channels": {}}
    print(f"\n{'channel':<14} {'pairs':>6} {'z>=2':>7} {'z<=-2':>7} "
          f"{'|z|<2':>7} {'z>=3':>7} {'z<=-3':>7}")
    for c in readers:
        mean = (s1[c] / N_ROT).cpu().numpy()
        var = (s2[c] / N_ROT).cpu().numpy() - mean ** 2
        sd = np.sqrt(np.maximum(var, 1e-30))
        sub = df[df["cls"] == c]
        z = (sub["stat"].values ** 2 - mean[sub["ri"], sub["wi"]]) / \
            sd[sub["ri"], sub["wi"]]
        rows = []
        for s, gs in sub.assign(zt=z).groupby("span"):
            rows.append(dict(
                span=int(s), n=int(len(gs)),
                above2=float((gs["zt"] >= 2).mean()),
                below2=float((gs["zt"] <= -2).mean()),
                above3=float((gs["zt"] >= 3).mean()),
                below3=float((gs["zt"] <= -3).mean()),
                median_z=float(gs["zt"].median()),
            ))
        a2, b2 = float((z >= 2).mean()), float((z <= -2).mean())
        a3, b3 = float((z >= 3).mean()), float((z <= -3).mean())
        summary["channels"][c] = dict(
            n=int(len(sub)), above2=a2, below2=b2, within2=1 - a2 - b2,
            above3=a3, below3=b3, median_z=float(np.median(z)),
            per_span=rows)
        print(f"{c:<14} {len(sub):>6} {a2:>6.1%} {b2:>6.1%} "
              f"{1 - a2 - b2:>6.1%} {a3:>6.1%} {b3:>6.1%}")

    # spot-check three pairs against the independent per-pair sampler
    from commap.nulls import haar_rotation_null
    rng = np.random.default_rng(1)
    checks = []
    for _ in range(3):
        r, wi_ = int(rng.integers(12, NH)), int(rng.integers(0, NH - 12))
        if layers[wi_] >= layers[r]:
            continue
        iso = haar_rotation_null(readers["head_head_K"][r],
                                 (F[wi_] @ F[wi_].T), n_rot=200,
                                 seed=int(rng.integers(1 << 31)))
        m_ref, s_ref = (iso ** 2).mean(), (iso ** 2).std()
        m_gpu = float((s1["head_head_K"][r, wi_] / N_ROT).cpu())
        v_gpu = float((s2["head_head_K"][r, wi_] / N_ROT).cpu()) - m_gpu ** 2
        s_gpu = float(np.sqrt(max(v_gpu, 0)))
        checks.append(dict(reader=r, writer=wi_, mean_ref=float(m_ref),
                           mean_gpu=m_gpu, sd_ref=float(s_ref), sd_gpu=s_gpu))
        assert abs(m_gpu - m_ref) < 6 * s_ref / np.sqrt(200) + 1e-4, checks[-1]
        assert 0.7 < s_gpu / s_ref < 1.4, checks[-1]
    summary["spot_checks"] = checks
    print("\nspot checks vs lib/commap/nulls.haar_rotation_null: OK")

    closed = json.load(open("results/map/gpt2/theory_census.json"))
    for c in summary["channels"]:
        a, b = summary["channels"][c], closed["channels"][c]
        da = abs(a["above2"] - b["above2"])
        db = abs(a["below2"] - b["below2"])
        assert da < 0.005 and db < 0.005, (c, da, db)
    print("closed-form census matches Monte Carlo on every fraction")
    json.dump(summary, open(OUT, "w"), indent=2)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
