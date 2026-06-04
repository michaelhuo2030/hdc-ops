#!/usr/bin/env python3
"""Reproduce every result table + the figures for the HDC tokamak-disruption application.
Run the data fetchers first (fetch/fetch_mast.py for MAST, fetch/fetch_golem.py for GOLEM), then:
    python run_all.py
Each step is an independent script; see README.md for the result-table <-> script <-> paper mapping.
"""
import os, sys, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("FUSION_DATA") or os.path.join(HERE, "data")
PY = sys.executable

STEPS = [
    ("R1  detection — MAST",                 "experiments/fc_m_mast_disruption.py", "mast_ctrl"),
    ("R1  detection — GOLEM (cross-machine)", "experiments/fc_b_golem.py",          "golem"),
    ("R2  control — HDC loses (boundary law)", "experiments/fc_ctrl_a_shape.py",    "mast_ctrl"),
    ("R3  provenance selective prediction",  "experiments/fc_n2_selective.py",      "mast_ctrl"),
    ("R4  traceable errors",                 "experiments/fc_n_self_iterate.py",    "mast_ctrl"),
    ("R5  weaponry ablation",                "experiments/fc_o_hdc_improve.py",     "mast_ctrl"),
    ("R7  conformal coverage guarantee",     "experiments/fc_p_conformal.py",       "mast_ctrl"),
]

def have(kind):
    if kind == "mast_ctrl":
        return len(glob.glob(os.path.join(DATA, "mast_ctrl", "*.npz")))
    return len(glob.glob(os.path.join(DATA, "golem", "*_ql_I_p.npy")))

def main():
    print(f"data dir: {DATA}")
    nm = have("mast_ctrl")
    print(f"  MAST shots cached : {nm}    (fetch with: python fetch/fetch_mast.py 11695 30474)")
    print(f"  GOLEM shots cached: {have('golem')}    (GOLEM r1 auto-fetches; or python fetch/fetch_golem.py)\n")
    for title, script, kind in STEPS:
        print("=" * 78); print(title, "  [", script, "]"); print("=" * 78)
        rc = subprocess.call([PY, os.path.join(HERE, script)], cwd=HERE)
        if rc != 0:
            print(f"  (step exited {rc} — usually means data not fetched yet; continuing)\n")
        print()
    print("=" * 78); print("figures"); print("=" * 78)
    subprocess.call([PY, "make_figures.py"], cwd=os.path.join(HERE, "figures"))
    print("\nAll steps attempted. Results -> results/ ; figures -> figures/*.pdf")

if __name__ == "__main__":
    main()
