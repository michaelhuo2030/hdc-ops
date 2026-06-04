#!/usr/bin/env python3
"""Pre-fetch real GOLEM tokamak shots into data/golem/ (so r1_detection on GOLEM can use the full set).
Source: the open GOLEM tokamak (CTU Prague), one CSV per signal per shot, over plain HTTP — no key, no proxy.
Short per-request timeout so empty shot numbers don't stall. Resumable (skips cached). Pull a wide range to
accumulate the ~1,852-shot cross-machine set used in the paper.
Usage:  python fetch_golem.py [START] [END]      e.g.  python fetch_golem.py 45000 52902
"""
import os, io, sys, urllib.request
import numpy as np

_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE = os.path.join(_DATA, "golem"); os.makedirs(CACHE, exist_ok=True)
BASE = "http://golem.fjfi.cvut.cz/shotdir/{s}/Diagnostics/PlasmaDetection/{sig}.csv"


def fetch(shot, sig, timeout=8):
    fp = os.path.join(CACHE, f"{shot}_{sig}.npy")
    if os.path.exists(fp): return True
    try:
        with urllib.request.urlopen(BASE.format(s=shot, sig=sig), timeout=timeout) as r:
            txt = r.read().decode("utf-8", "ignore")
        if "<!DOCTYPE" in txt[:40] or "404" in txt[:200]: return False
        arr = np.loadtxt(io.StringIO(txt), delimiter=",")
        if arr.ndim != 2 or arr.shape[0] < 1000: return False
        np.save(fp, arr.astype(np.float32)); return True
    except Exception:
        return False


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 45000
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 52902
    print(f"[fetch_golem] {start}..{end} -> {CACHE}", flush=True)
    ok = 0
    for s in range(start, end):
        if fetch(s, "V_loop") and fetch(s, "ql_I_p"):
            ok += 1
        if s % 50 == 0:
            n = len([f for f in os.listdir(CACHE) if f.endswith("_ql_I_p.npy")])
            print(f"  up to {s}  new_ok {ok}  total_cached {n}", flush=True)
    print(f"[fetch_golem] DONE — {ok} new shot-pairs; "
          f"{len([f for f in os.listdir(CACHE) if f.endswith('_ql_I_p.npy')])} total cached", flush=True)


if __name__ == "__main__":
    main()
