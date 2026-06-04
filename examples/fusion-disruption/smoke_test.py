#!/usr/bin/env python3
"""Offline smoke test — proves the application runs end-to-end from a fresh clone WITHOUT downloading any data,
and proves the hdc-ops library refactor is numerically identical to the original inline encoder.

It (1) checks the encoder equivalence  bundle([bind(level,pos)]) == sign(Σ level·pos)  on random arrays;
   (2) writes a tiny synthetic dataset (MAST-shaped .npz + GOLEM-shaped .npy cache) into a temp FUSION_DATA dir;
   (3) runs every experiment against it and asserts each completes (small N -> numbers are meaningless, this only
       checks the code paths). For the REAL paper numbers, fetch the data and run run_all.py.

Usage: python smoke_test.py
"""
import os, sys, subprocess, tempfile, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))   # repo root -> hdc_ops
from hdc_ops import bind, bundle


def check_equivalence():
    rng = np.random.default_rng(0)
    D, L, NLEV = 2000, 64, 32
    levels = rng.choice(np.array([-1, 1], np.float32), (NLEV, D))
    pos = rng.choice(np.array([-1, 1], np.float32), (L, D))
    b = rng.integers(0, NLEV, L)
    inline = np.sign((levels[b] * pos).sum(0))                 # original
    lib = bundle(list(bind(levels[b], pos)))                   # library refactor used in the experiments
    ok = np.array_equal(inline, lib)
    print(f"[1] encoder equivalence  bundle([bind])==sign(Σ·)  : {'IDENTICAL' if ok else 'MISMATCH'}")
    assert ok, "library refactor changed the encoder numerics!"


def synth(data_dir, n_shots=60, L=512):
    mast = os.path.join(data_dir, "mast_ctrl"); os.makedirs(mast, exist_ok=True)
    rng = np.random.default_rng(1)
    cn = np.array([f"c{i}_coil_current" for i in range(6)]); mn = np.array([f"m{i}" for i in range(6)])
    for s in range(n_shots):
        long = s % 2 == 0
        dur_ms = float(rng.uniform(300, 600) if long else rng.uniform(120, 300))
        t = np.linspace(0.0, dur_ms / 1000.0, L).astype("f4")
        ip = (50 + 30 * np.sin(np.linspace(0, (3 if long else 7), L)) + rng.normal(0, 3, L)).astype("f4")
        coils = (rng.normal(0, 1, (6, L)) + ip[None, :] * 0.1).astype("f4")
        mag = (rng.normal(0, 1, (6, L))).astype("f4")
        np.savez_compressed(os.path.join(mast, f"{s}.npz"), ip=ip, t=t, dur_ms=dur_ms, peak=float(np.max(np.abs(ip))),
                            coils=coils, coil_names=cn, mag=mag, mag_names=mn)
    # GOLEM-shaped cache so r1_golem runs fully offline (cover fc_b_golem's ENTIRE SHOTS range, else it
    # falls through to the real network for uncached shots). fc_b_golem uses range(52740, 52902).
    golem = os.path.join(data_dir, "golem"); os.makedirs(golem, exist_ok=True)
    for i, s in enumerate(range(52740, 52902)):
        npre, npost = 400, 1200; t = np.concatenate([np.linspace(-2e-3, 0, npre), np.linspace(0, (8e-3 if i % 2 else 30e-3), npost)])
        I = np.concatenate([rng.normal(0, 0.002, npre), np.abs(np.sin(np.linspace(0, 3, npost))) * (0.5 if i % 2 else 1.0)])
        V = rng.normal(0, 0.5, len(t))
        np.save(os.path.join(golem, f"{s}_ql_I_p.npy"), np.column_stack([t, I]).astype("f4"))
        np.save(os.path.join(golem, f"{s}_V_loop.npy"), np.column_stack([t, V]).astype("f4"))
    return mast


# (script, args) — small min_shots/n so the size-gated scripts run on the tiny synthetic set
STEPS = [
    ("experiments/fc_m_mast_disruption.py", ["30"]),
    ("experiments/fc_b_golem.py",           []),
    ("experiments/fc_ctrl_a_shape.py",      ["20"]),
    ("experiments/fc_n2_selective.py",      []),
    ("experiments/fc_n_self_iterate.py",    []),
    ("experiments/fc_o_hdc_improve.py",     []),
    ("experiments/fc_p_conformal.py",       []),
    ("figures/make_figures.py",             []),
]

def main():
    check_equivalence()
    tmp = tempfile.mkdtemp(prefix="fusion_smoke_")
    synth(tmp)
    env = dict(os.environ, FUSION_DATA=tmp)
    print(f"[2] synthetic data -> {tmp}\n[3] running {len(STEPS)} steps:")
    fails = []
    for script, args in STEPS:
        cwd = os.path.join(HERE, os.path.dirname(script))
        rc = subprocess.call([sys.executable, os.path.basename(script)] + args, cwd=cwd, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        tag = "ok  " if rc == 0 else "FAIL"
        if rc != 0: fails.append(script)
        print(f"    [{tag}] {script}")
    print()
    if fails:
        print(f"SMOKE TEST FAILED: {fails}"); sys.exit(1)
    print("SMOKE TEST PASSED — every experiment runs end-to-end from a clean tree (synthetic data).")
    print("For the real paper numbers: python fetch/fetch_mast.py && python run_all.py")

if __name__ == "__main__":
    main()
