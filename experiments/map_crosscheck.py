"""v1 pipeline: v0.1 head map + MLP neuron classes (design S12 D5, stage 2).

SCAFFOLD -- written on the Mac, intended to run on the Linux/RTX-5090 box
(neuron_neuron is ~6e8 cosines after the causal mask; the tiles are one GEMM
each so CPU works, but GPU makes pass1+pass2 seconds). Before the first real
run, freeze the open pre-registration decisions marked PRE-REG below.

Usage:
    uv run python experiments/map_crosscheck.py [--out results/verification/crosscheck] [--q 0.05]
        [--k-global 5] [--device auto] [--n-bins 4001] [--skip-head-map]

Stages:
    1. load model once; heads (weights.py) + neurons (neurons.py)
    2. head-map families exactly as v0.1 (dual stats, bulk/rotation nulls)
    3. mixed classes head<->neuron + interface-matrix<->neuron (dense, ~1e7 edges,
       survivors kept)
    4. neuron_neuron via the tiled stream (histogram BH, survivors kept)
    5. the same for the partialled map (D3)
    6. R2-mediation query (EXPLORATORY, not pre-registered): neurons that
       significantly read an S-inhibition head's OV AND significantly write
       into a name-mover's Q-side -- the v0.1 memo's live hypothesis (i).

Outputs in --out:
    families_v1.csv          per-family counts, all classes
    survivors_mixed.csv.gz   mixed-class survivors (raw map)
    survivors_nn.csv.gz      neuron_neuron survivors (raw map)
    nn_span_fits.csv         per-span null fits + widening factors
    nn_hist.npz              the pass-1 histograms (reproducibility)
    *_partial.*              same for the partialled map
    r2_mediation.csv         exploratory two-hop table
    summary.json

PRE-REG decisions FROZEN 2026-08-08, before any real v1 run on this box:
    P1  neuron_neuron sidedness: TWO-SIDED on |z| (a strong negative
        alignment is a sign-flipping channel; cf. the L4H7 sign lesson)
    P2  pooled global directions (D3): v0's head+interface-matrix pool, unchanged,
        for comparability with the v0/v0.1 partialled maps (neuron vectors
        NOT added to the pooled second moment)
    P3  v1 recovery criterion: NONE -- R2-mediation stays an exploratory
        query (no principled pass/fail threshold exists for it yet)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "lib"))

from commap.neurons import load_neurons, mixed_class_tables, partial_neurons
from commap.partial import ahn_horenstein, global_directions, partialled_weights
from commap.recovery import NAME_MOVERS, S_INHIBITION
from commap.stream import stream_neuron_neuron
from commap.weights import load_gpt2, load_model


def head_map(w, q_level, n_rot):
    """The head-family map (same code path, D5)."""
    from commap.edges import compute_edges
    from commap.fdr import add_fdr
    from commap.nulls import add_interface_rotation_null, add_empirical_null

    df = compute_edges(w)
    df = add_empirical_null(df, "stat", "z", "p")
    df = add_interface_rotation_null(df=df, w=w, n_rot_frob=n_rot)
    df = add_fdr(df, q_level, "p", "q", "sig")
    return df


def one_map(tag, w, nw, args, out):
    """Mixed + streamed classes for one map (raw or partialled)."""
    sfx = "" if tag == "raw" else "_partial"
    print(f"  [{tag}] mixed classes (dense, survivors kept) ...")
    mixed, fam_mixed = mixed_class_tables(w, nw, q_level=args.q)
    pd.concat(mixed.values()).to_csv(out / f"survivors_mixed{sfx}.csv.gz", index=False)

    print(f"  [{tag}] neuron_neuron stream (device={args.device}) ...")
    nn = stream_neuron_neuron(nw, q_level=args.q, n_bins=args.n_bins,
                              device=args.device, two_sided=not args.one_sided)
    nn["survivors"].to_csv(out / f"survivors_nn{sfx}.csv.gz", index=False)
    nn["span_fits"].to_csv(out / f"nn_span_fits{sfx}.csv", index=False)
    np.savez_compressed(out / f"nn_hist{sfx}.npz", hist=nn["hist"], edges=nn["edges"])

    fam_nn = pd.DataFrame([dict(
        cls="neuron_neuron", n_edges=nn["m_total"],
        n_sig=len(nn["survivors"]),
        max_stat=float(nn["survivors"]["stat"].max()) if len(nn["survivors"]) else 0.0,
        max_z=float(nn["survivors"]["z"].max()) if len(nn["survivors"]) else 0.0,
        null_kind="bulk-stream",
    )])
    return mixed, pd.concat([fam_mixed, fam_nn], ignore_index=True).assign(map=tag)


def r2_mediation(mixed: dict) -> pd.DataFrame:
    """EXPLORATORY: two-hop S-inhibition -> neuron -> name-mover(Q) paths.

    hop1 = significant head_neuron edge from an S-inhibition head;
    hop2 = significant neuron_head_Q edge into a name-mover. Path score =
    min(z_hop1, z_hop2) (weakest link)."""
    h1 = mixed["head_neuron"]
    h1 = h1[h1["writer"].isin(S_INHIBITION) & h1["sig"]]
    h2 = mixed["neuron_head_Q"]
    h2 = h2[h2["reader"].isin(NAME_MOVERS) & h2["sig"]]
    paths = h1.merge(h2, left_on="reader", right_on="writer",
                     suffixes=("_hop1", "_hop2"))
    if paths.empty:
        return pd.DataFrame()
    paths["path_z"] = np.minimum(paths["z_hop1"], paths["z_hop2"])
    cols = {"writer_hop1": "s_inhib", "reader_hop1": "neuron",
            "reader_hop2": "name_mover", "stat_hop1": "stat_in",
            "stat_hop2": "stat_out", "z_hop1": "z_in", "z_hop2": "z_out"}
    return (paths.rename(columns=cols)[list(cols.values()) + ["path_z"]]
            .sort_values("path_z", ascending=False).reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/verification/crosscheck")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--k-global", type=int, default=5)
    ap.add_argument("--n-rot", type=int, default=500)
    ap.add_argument("--n-bins", type=int, default=4001)
    ap.add_argument("--device", default="auto", help="cuda | mps | cpu | auto")
    ap.add_argument("--one-sided", action="store_true",
                    help="PRE-REG P1: score neuron_neuron on the upper tail only")
    ap.add_argument("--skip-head-map", action="store_true",
                    help="skip the v0.1 head families (already recorded)")
    ap.add_argument("--model", default="gpt2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[1/6] loading model, head + neuron weights ...")
    model = load_model(args.model)
    w = load_gpt2(args.model, model=model)
    nw = load_neurons(model)
    del model

    if not args.skip_head_map:
        print("[2/6] head families (v0.1 code path) ...")
        hm = head_map(w, args.q, args.n_rot)
        hm.to_csv(out / "edges_heads_raw.csv.gz", index=False)
    else:
        print("[2/6] head families skipped")

    print("[3/6] raw map: mixed + neuron_neuron ...")
    mixed_raw, fam_raw = one_map("raw", w, nw, args, out)

    print(f"[4/6] partialled map (k={args.k_global}) ...")
    Vk, eigvals = global_directions(w, args.k_global)   # PRE-REG P2: head pool
    w_p, _ = partialled_weights(w, k=args.k_global)
    nw_p = partial_neurons(nw, Vk)
    _, fam_part = one_map("partial", w_p, nw_p, args, out)

    fam = pd.concat([fam_raw, fam_part], ignore_index=True)
    fam.to_csv(out / "families_v1.csv", index=False)

    print("[5/6] R2-mediation query (exploratory) ...")
    med = r2_mediation(mixed_raw)
    med.to_csv(out / "r2_mediation.csv", index=False)

    print("[6/6] summary ...")
    meta = dict(
        version="v1-scaffold", model=args.model, q_level=args.q,
        k_global=args.k_global, n_bins=args.n_bins,
        two_sided_nn=not args.one_sided,
        ahn_horenstein_k=ahn_horenstein(eigvals),
        n_r2_mediation_paths=int(len(med)),
        runtime_s=round(time.time() - t0, 1),
    )
    meta["families"] = {
        f"{row['map']}/{row['cls']}": dict(n_edges=int(row["n_edges"]),
                                           n_sig=int(row["n_sig"]))
        for _, row in fam.iterrows()
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
