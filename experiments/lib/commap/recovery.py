"""Pre-registered recovery acceptance test (decision D5).

FIXED BEFORE looking at the map exploratorily. The map passes v0 acceptance
iff each criterion below holds on the RAW map at family FDR q <= 0.05.

Known GPT-2-small circuitry (sources: Wang et al. 2022 IOI, arXiv:2211.00593;
standard induction-head analyses of gpt2-small; head lists are literature
values and should be re-verified causally in the validation loop):

    previous-token heads   L2H2, L4H11
    duplicate-token heads  L0H1, L0H10, L3H0
    induction heads        L5H1, L5H5, L6H9, L7H2, L7H10
    S-inhibition heads     L7H3, L7H9, L8H6, L8H10
    name-mover heads       L9H6, L9H9, L10H0

Criteria (each an edge-existence claim the weight map should reproduce):
    R1  every induction head has a significant head_head_K edge from at least
        one previous-token head (K-composition: the defining induction wiring)
    R2  every name-mover head has a significant head_head_Q edge from at least
        one S-inhibition head (Q-composition, IOI paper Fig. 3)
    R3  every S-inhibition head has a significant head_head_V edge from at
        least one duplicate-token head (V-composition, IOI paper)
    R4  every previous-token head has a significant pos_head_K edge
        (attending to the previous position is positional wiring)
"""

from __future__ import annotations

import pandas as pd

PREV_TOKEN = ["L2H2", "L4H11"]
DUP_TOKEN = ["L0H1", "L0H10", "L3H0"]
INDUCTION = ["L5H1", "L5H5", "L6H9", "L7H2", "L7H10"]
S_INHIBITION = ["L7H3", "L7H9", "L8H6", "L8H10"]
NAME_MOVERS = ["L9H6", "L9H9", "L10H0"]

CRITERIA = [
    ("R1_induction_Kcomp", "head_head_K", PREV_TOKEN, INDUCTION),
    ("R2_namemover_Qcomp", "head_head_Q", S_INHIBITION, NAME_MOVERS),
    ("R3_sinhib_Vcomp", "head_head_V", DUP_TOKEN, S_INHIBITION),
    ("R4_prevtoken_pos", "pos_head_K", ["POS"], PREV_TOKEN),
]


def evaluate(df: pd.DataFrame, suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    """df: edge table with [cls, writer, reader, stat{sfx}, z{sfx}, q{sfx}, sig{sfx}].

    Returns (per-reader detail, per-criterion summary)."""
    s, z, qc, sig = f"stat{suffix}", f"z{suffix}", f"q{suffix}", f"sig{suffix}"
    detail_rows, summary_rows = [], []
    for name, cls, writers, readers in CRITERIA:
        sub = df[(df["cls"] == cls) & df["writer"].isin(writers) & df["reader"].isin(readers)]
        n_pass = 0
        for reader in readers:
            edges = sub[sub["reader"] == reader].sort_values(qc)
            if edges.empty:  # e.g. writer layer not below reader layer
                detail_rows.append(dict(criterion=name, reader=reader, best_writer=None,
                                        stat=None, z=None, q=None, hit=False))
                continue
            best = edges.iloc[0]
            hit = bool(best[sig])
            n_pass += hit
            detail_rows.append(dict(criterion=name, reader=reader,
                                    best_writer=best["writer"], stat=best[s],
                                    z=best[z], q=best[qc], hit=hit))
        summary_rows.append(dict(criterion=name, cls=cls, n_readers=len(readers),
                                 n_pass=n_pass, passed=n_pass == len(readers)))
    summary = pd.DataFrame(summary_rows)
    summary.loc[len(summary)] = dict(criterion="ACCEPTANCE", cls="-",
                                     n_readers=summary["n_readers"].sum(),
                                     n_pass=summary["n_pass"].sum(),
                                     passed=bool(summary["passed"].all()))
    return pd.DataFrame(detail_rows), summary
