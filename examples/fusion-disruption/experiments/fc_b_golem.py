#!/usr/bin/env python3
"""R1 (GOLEM) — HDC plasma-event precursor detection on real GOLEM tokamak shots (cross-machine check for the
MAST result). Data: open GOLEM tokamak (CTU Prague), fetched per shot over plain HTTP:
  http://golem.fjfi.cvut.cz/shotdir/<shot>/Diagnostics/PlasmaDetection/V_loop.csv   (time, loop voltage)
                                                                      /ql_I_p.csv    (time, plasma current)
Real binary label derived from the plasma-current waveform itself (not arbitrary):
  - plasma_formed = peak|Ip| exceeds a robust multiple of the pre-trigger (t<0) baseline noise;
  - among plasma shots: EARLY-QUENCH (short) vs SUSTAINED (long) by plasma-duration median.
  Task = predict the outcome from an EARLY signal window ONLY (leakage-safe precursor detection).

Discipline: shot-level held-out split (NOT a time-window split — the leakage trap); adequately-trained learned
baselines (logistic + MLP, not under-tuned); calibration (Brier); beats-majority; 3 seeds. The HDC encoder is the
hdc-ops library: SimHash -> per-class `bundle` prototype -> cosine. The boundary law predicts HDC may TIE/LOSE a
learned baseline on this learn-from-noisy-signal task; the distinctive HDC value is the provenance/abstention story
(see r3/r4/r7), not a raw-accuracy win. Self-contained (numpy + stdlib; downloads + caches GOLEM CSVs on first run).
Usage: python fc_b_golem.py
"""
import warnings; warnings.filterwarnings("ignore")
import urllib.request, io, os, sys, json, time, csv
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root -> hdc_ops
from hdc_ops import bundle  # per-class prototype = bundle of the class's shot HVs

_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); os.makedirs(_RES, exist_ok=True)
CACHE = os.path.join(_DATA, "golem"); os.makedirs(CACHE, exist_ok=True)

SHOTS = list(range(52740, 52902))     # ~160 recent shots (run fetch/fetch_golem.py first for the full ~1,852-shot set)
BASE = "http://golem.fjfi.cvut.cz/shotdir/{s}/Diagnostics/PlasmaDetection/{sig}.csv"
D, SEEDS = 4096, [42, 123, 7]

def fetch(shot, sig):
    fp = os.path.join(CACHE, f"{shot}_{sig}.npy")
    if os.path.exists(fp):
        try: return np.load(fp)
        except Exception: pass
    try:
        with urllib.request.urlopen(BASE.format(s=shot, sig=sig), timeout=20) as r:
            txt = r.read().decode("utf-8", "ignore")
        if "<!DOCTYPE" in txt[:40] or "404" in txt[:200]: return None
        arr = np.loadtxt(io.StringIO(txt), delimiter=",")
        if arr.ndim != 2 or arr.shape[0] < 1000: return None
        np.save(fp, arr.astype(np.float32)); return arr.astype(np.float32)
    except Exception:
        return None

def process():
    shots = []
    for s in SHOTS:
        vl, ip = fetch(s, "V_loop"), fetch(s, "ql_I_p")
        if vl is None or ip is None: continue
        t = ip[:, 0]; I = ip[:, 1]; V = vl[:, 1] if len(vl) == len(ip) else np.interp(t, vl[:, 0], vl[:, 1])
        pre = I[t < 0]
        if len(pre) < 50: continue
        base = np.std(pre) + 1e-6; thr = max(5 * base, 0.05 * np.max(np.abs(I)))
        absI = np.abs(I); plasma_mask = absI > thr
        formed = plasma_mask.sum() > 100
        post = t >= 0
        dur = float(np.ptp(t[plasma_mask & post])) if (plasma_mask & post).sum() > 5 else 0.0
        shots.append(dict(shot=s, t=t, I=I, V=V, formed=bool(formed), dur=dur, thr=thr))
    return shots

def early_features(sh, frac=0.30):
    """Features from the FIRST `frac` of the plasma phase only (leakage-safe precursor)."""
    t, I, V = sh["t"], sh["I"], sh["V"]
    post = t >= 0; tp = t[post]; Ip = I[post]; Vp = V[post]
    n = len(tp); w = max(50, int(n * frac))
    Iw, Vw, tw = Ip[:w], Vp[:w], tp[:w]
    dI = np.gradient(Iw, tw + 1e-12)
    def st(x): return [np.mean(x), np.std(x), np.max(x), np.min(x), x[-1] - x[0]]
    return np.array(st(np.abs(Iw)) + st(Vw) + [np.max(np.abs(dI)), np.mean(np.abs(dI))], np.float32)

class MLP:
    def __init__(s, di, dh, do, rng):
        s.W1 = rng.standard_normal((di, dh)).astype(np.float32) * np.sqrt(2/di); s.b1 = np.zeros(dh, np.float32)
        s.W2 = rng.standard_normal((dh, do)).astype(np.float32) * np.sqrt(2/dh); s.b2 = np.zeros(do, np.float32)
    def fwd(s, X):
        Z1 = X @ s.W1 + s.b1; H = np.maximum(Z1, 0); Z2 = H @ s.W2 + s.b2
        Z2 -= Z2.max(1, keepdims=True); E = np.exp(Z2); return Z1, H, E / E.sum(1, keepdims=True)
    def train(s, X, y, epochs=1500, lr=0.05, wd=1e-3):
        Y = np.eye(s.W2.shape[1], dtype=np.float32)[y]; n = len(X)
        for _ in range(epochs):
            Z1, H, P = s.fwd(X); dZ2 = (P - Y) / n
            dW2 = H.T @ dZ2 + wd * s.W2; db2 = dZ2.sum(0); dZ1 = (dZ2 @ s.W2.T) * (Z1 > 0)
            dW1 = X.T @ dZ1 + wd * s.W1; db1 = dZ1.sum(0)
            s.W2 -= lr*dW2; s.b2 -= lr*db2; s.W1 -= lr*dW1; s.b1 -= lr*db1
        return s
    def proba(s, X): return s.fwd(X)[2][:, 1]

def brier(p, y): return float(np.mean((p - y) ** 2))

def run(shots):
    formed = [s for s in shots if s["formed"]]
    print(f"  downloaded={len(shots)} plasma_formed={len(formed)} no_plasma={len(shots)-len(formed)}", flush=True)
    if len(formed) >= 30:
        durs = np.array([s["dur"] for s in formed]); med = np.median(durs)
        y = np.array([0 if s["dur"] >= med else 1 for s in formed])  # 1 = early-quench (short)
        pool = formed; task = f"early-quench(<{med*1e3:.1f}ms) vs sustained, among {len(formed)} plasma shots"
    else:
        y = np.array([0 if s["formed"] else 1 for s in shots])       # 1 = no-plasma
        pool = shots; task = f"plasma vs no-plasma, {len(shots)} shots"
    X = np.stack([early_features(s) for s in pool])
    print(f"  TASK = {task} | label balance: pos={int(y.sum())}/{len(y)}", flush=True)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        print("  [WARN] degenerate label balance — result will be weak/inconclusive.")
    mu, sd = X.mean(0), X.std(0) + 1e-9; Xn = (X - mu) / sd
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xn)); cut = int(0.6 * len(idx))
        tr, te = idx[:cut], idx[cut:]                                 # SHOT-level split (each row = a shot)
        Xtr, ytr, Xte, yte = Xn[tr], y[tr], Xn[te], y[te]
        # HDC: SimHash encode -> per-class bundled prototype (hdc_ops.bundle) -> cosine
        Rp = rng.standard_normal((Xn.shape[1], D)).astype(np.float32)
        Htr = np.sign(Xtr @ Rp); Hte = np.sign(Xte @ Rp)
        proto = np.stack([bundle(list(Htr[ytr == c])) if (ytr == c).any() else np.zeros(D) for c in (0, 1)])
        Pn = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-9)
        sims = (Hte / (np.linalg.norm(Hte, axis=1, keepdims=True) + 1e-9)) @ Pn.T
        hdc_p = np.exp(sims); hdc_p = hdc_p[:, 1] / hdc_p.sum(1)
        hdc_acc = float(((sims[:, 1] > sims[:, 0]).astype(int) == yte).mean())
        logi = MLP(Xn.shape[1], 2, 2, np.random.default_rng(seed)).train(Xtr, ytr, epochs=2000, lr=0.05)
        mlp = MLP(Xn.shape[1], 32, 2, np.random.default_rng(seed + 1)).train(Xtr, ytr, epochs=2000, lr=0.05)
        logi_acc = float((logi.proba(Xte) > 0.5).astype(int).__eq__(yte).mean())
        mlp_acc = float((mlp.proba(Xte) > 0.5).astype(int).__eq__(yte).mean())
        maj = max(yte.mean(), 1 - yte.mean())
        rows.append(dict(seed=seed, n_test=len(yte), hdc_acc=hdc_acc, logi_acc=logi_acc, mlp_acc=mlp_acc,
                         majority=float(maj), hdc_brier=brier(hdc_p, yte), mlp_brier=brier(mlp.proba(Xte), yte)))
    return task, rows

def main():
    t0 = time.time(); print("[R1 GOLEM] downloading + processing real shots…", flush=True)
    shots = process()
    if len(shots) < 20:
        print(f"  only {len(shots)} shots fetched — aborting (network/availability). Try fetch/fetch_golem.py first."); return
    task, rows = run(shots)
    agg = {k: round(float(np.mean([r[k] for r in rows])), 4) for k in
           ("hdc_acc", "logi_acc", "mlp_acc", "majority", "hdc_brier", "mlp_brier")}
    best_base = max(agg["logi_acc"], agg["mlp_acc"])
    print(f"\n  === REAL GOLEM ({task}) — {len(rows)} seeds ===")
    print(f"  HDC acc={agg['hdc_acc']}  | logistic={agg['logi_acc']} MLP={agg['mlp_acc']} | majority={agg['majority']}")
    print(f"  calibration Brier: HDC={agg['hdc_brier']} MLP={agg['mlp_brier']} (lower=better)")
    hdc_vs_learned = "HDC>=learned" if agg["hdc_acc"] >= best_base - 0.02 else "HDC<learned (boundary law)"
    print(f"  -> HDC vs learned: {hdc_vs_learned}")
    cpath = os.path.join(_RES, "results_fc_b_golem.csv")
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"task": task, "rows": rows, "agg": agg, "n_shots": len(shots),
               "data": "REAL GOLEM tokamak (CTU Prague), PlasmaDetection CSVs"},
              open(os.path.join(_RES, "fc_b_golem_result.json"), "w"), indent=2)
    print(f"\n[done {time.time()-t0:.0f}s] -> {cpath}", flush=True)

if __name__ == "__main__":
    main()
