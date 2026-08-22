"""Reproduce every released artifact: the map, its census, and the two
applications.

  map     the general map + census on GPT-2 (map_build), the CPU
          cross-check (map_crosscheck), the wires figure     (~2 min)
  app1    head-to-head selection (app1_select), communities,
          Figure 2, the ablation decomposition + seed check +
          four-model replication                             (~30 min)
  app2    bands figure, deletion pilots on five models       (~40 min)
  verify  Monte Carlo nulls, both-sides bound, robust-vs-moment
          null fits, census MC validation                    (~8 min)
  test    synthetic ground-truth pipeline test               (~1 min)

Larger models run the same two scripts directly:
  uv run python experiments/map_build.py pythia-2.8b
  uv run python experiments/map_build.py pythia-6.9b --stream-load
  uv run python experiments/app1_select.py pythia-2.8b
Heavier robustness checks (app2_scale, app2_overlaps,
app2_profile_samplesize, verify_precision, verify_stream) are direct
invocations; see README.
"""
import argparse
import subprocess
import sys
from pathlib import Path

E = str(Path(__file__).parent)
PY = sys.executable

STAGES = {
    "map": [[f"{E}/map_build.py", "gpt2"],
            [f"{E}/map_crosscheck.py"],
            [f"{E}/map_fig_wires.py"]],
    "app1": [[f"{E}/app1_select.py", "gpt2"],
             [f"{E}/app1_communities.py"],
             [f"{E}/app1_fig_map.py"],
             [f"{E}/app1_ablation.py"],
             [f"{E}/app1_seed_check.py"],
             [f"{E}/app1_multimodel.py"]],
    "app2": [[f"{E}/app2_fig_bands.py"],
             [f"{E}/app2_pilot.py", "gpt2"],
             [f"{E}/app2_pilot.py", "gpt2", "1"],
             [f"{E}/app2_pilot.py", "gpt2-medium"],
             [f"{E}/app2_pilot.py", "gpt2-large"],
             [f"{E}/app2_pilot.py", "pythia-160m"],
             [f"{E}/app2_pilot.py", "gpt-neo-125M"],
             [f"{E}/app2_rule_robustness.py"]],
    "verify": [[f"{E}/verify_mc_nulls.py"],
               [f"{E}/verify_both_sides.py"],
               [f"{E}/verify_null_estimators.py"],
               [f"{E}/verify_census_mc.py"],
               [f"{E}/verify_null_shape.py"]],
    "test": [[f"{E}/test_pipeline_synthetic.py"]],
}
ORDER = ["map", "app1", "app2", "verify", "test"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ORDER),
                    help="comma-separated subset, e.g. map,app1")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        print(__doc__)
        return
    for stage in args.stages.split(","):
        for cmd in STAGES[stage.strip()]:
            print(f"== {' '.join(Path(c).name for c in cmd)}", flush=True)
            subprocess.run([PY] + cmd, check=True)


if __name__ == "__main__":
    main()
