#!/usr/bin/env python3
"""Exp M — REAL MAST disruption-proxy detection, LONG-PULSE analog of the GOLEM Exp B/L.
Data: data/mast_ctrl/<shot>.npz (amc/plasma_current, downsampled plasma phase, 130-820ms pulses),
fetched from the FAIR-MAST open S3 zarr by fetch/fetch_mast.py.

Applies the hard-won lessons (do NOT repeat the weak-config error):
  - FULL HDC ARSENAL: TEMPORAL waveform encoding  shot_hv = Σ_t bind(level(x_t), pos_t)  (HDC's home turf),
    NOT hand-crafted stats; readouts = nearest-centroid (stable) AND learned ridge (E10: the wall is the readout).
  - Matched baselines on the SAME shots, SAME shot-split: logistic + tiny MLP on hand-crafted features.
    (Honest caveat carried over: a waveform-CNN is the stronger NN baseline, not implemented here — noted, not hidden.)
  - NaN-robust loader: early-campaign MAST shots have dropout NaNs in the signal -> linear-interpolate the gaps.
  - Label = early-quench proxy (plasma duration < median): MAST has real disruptions but level1 has no disruption
    flag, so we use the SAME duration proxy as GOLEM for a consistent, leakage-safe target.
  - shot-split, 3 seeds, D-sweep, AUC + sensitivity@FPR. Self-contained (numpy only). 
Usage: fc_m_mast_disruption.py [min_shots]   (waits/needs >= min_shots real-plasma npz; default 600)"""
import warnings; warnings.filterwarnings("ignore")
import glob, os, sys, json, csv
import numpy as np
# repo-relative data + results dirs (override the data location with the FUSION_DATA env var)
_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root -> hdc_ops
from hdc_ops import bind, bundle  # this application is built on the hdc-ops operator library

CACHE = os.path.join(_DATA, "mast_ctrl")
SEEDS = [42, 123, 7]; FPR_TARGET = 0.10
L = 256          # temporal HV sequence length (downsample the stored 1024-pt waveform)
NLEV = 32        # value-quantization levels

# ---------- NaN-robust loader ----------
def interp_nan(a):
    a = a.astype(np.float64); n = len(a); idx = np.arange(n); good = np.isfinite(a)
    if good.sum() < 2: return None
    if not good.all(): a = np.interp(idx, idx[good], a[good])
    return a.astype(np.float32)

def load():
    shots = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        try:
            d = np.load(f); ip = interp_nan(d["ip"]);
            if ip is None: continue
            dur = float(d["dur_ms"]); pk = float(d["peak"])
        except Exception:
            continue
        if not np.isfinite(dur) or dur <= 0: continue
        shots.append(dict(shot=int(os.path.basename(f).split(".")[0]), ip=ip, dur=dur, peak=pk))
    return shots

# ---------- full-arsenal HDC: temporal waveform encoding ----------
def temporal_hv(sig, D, levels, pos):
    w = sig.astype(np.float64)
    idx = np.linspace(0, len(w) - 1, L).astype(int); w = w[idx]
    w = (w - w.mean()) / (w.std() + 1e-9)
    b = np.clip(((w - w.min()) / (np.ptp(w) + 1e-9) * (NLEV - 1)).astype(int), 0, NLEV - 1)
    return bundle(list(bind(levels[b], pos))).astype(np.float32)

# ---------- hand-crafted features for the matched MLP/logistic baseline ----------
def feats(sig):
    w = interp_nan(sig); n = len(w); e = w[:max(50, n // 3)]   # early window (leakage-safe-ish; whole pulse is the proxy)
    dI = np.gradient(w)
    def st(x): return [np.mean(x), np.std(x), np.max(x), np.min(x), x[-1] - x[0]]
    return np.array(st(np.abs(w)) + st(dI) + [np.max(np.abs(dI)), float(n)], np.float32)

# ---------- tiny numpy MLP + logistic ----------
class MLP:
    def __init__(s, di, h, rng): s.W1 = rng.standard_normal((di, h)) * 0.3; s.b1 = np.zeros(h); s.W2 = rng.standard_normal((h, 1)) * 0.3; s.b2 = 0.0
    def fwd(s, X): s.z1 = np.maximum(0, X @ s.W1 + s.b1); return 1 / (1 + np.exp(-(s.z1 @ s.W2 + s.b2).ravel()))
    def train(s, X, y, epochs=400, lr=0.1):
        for _ in range(epochs):
            p = s.fwd(X); g = (p - y) / len(y)
            gW2 = s.z1.T @ g[:, None]; gb2 = g.sum(); gh = (g[:, None] @ s.W2.T) * (s.z1 > 0)
            gW1 = X.T @ gh; gb1 = gh.sum(0)
            s.W2 -= lr * gW2; s.b2 -= lr * gb2; s.W1 -= lr * gW1; s.b1 -= lr * gb1
        return s

def auc(p, y):
    y = np.asarray(y); pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

def sens_at_fpr(p, y, ft):
    y = np.asarray(y); best = 0.0
    for thr in np.unique(p)[::-1]:
        pred = p >= thr; fpr = (pred & (y == 0)).sum() / max((y == 0).sum(), 1)
        if fpr <= ft: best = max(best, (pred & (y == 1)).sum() / max((y == 1).sum(), 1))
    return float(best)

def ridge_fit(H, y, lam=None):
    # H is binary +-1 (D cols). H.T@H diagonal ~= n_train; with D>n_train the system is rank-deficient and a
    # tiny lam over-fits/collapses (E10 / learned-readout discipline). Scale lam with n_train so regularization
    # tracks the data, not the (arbitrary) dimension D -> stable across D.
    n = H.shape[0]
    if lam is None: lam = max(10.0, 0.5 * n)
    yt = y.astype(np.float64) * 2 - 1
    return np.linalg.solve(H.T @ H + lam * np.eye(H.shape[1]), H.T @ yt)

def main():
    min_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    sh = load()
    if len(sh) < min_shots:
        print(f"[Exp M] only {len(sh)} real-plasma shots cached (< {min_shots}); not enough yet — rerun later.")
        return
    durs = np.array([x["dur"] for x in sh]); med = np.median(durs)
    y = np.array([1 if x["dur"] < med else 0 for x in sh])
    X = np.stack([feats(x["ip"]) for x in sh])
    fin = np.isfinite(X).all(1); X, y, sh = X[fin], y[fin], [s for s, ok in zip(sh, fin) if ok]
    medf = np.median(X, 0); iqr = np.percentile(X, 75, 0) - np.percentile(X, 25, 0); iqr[iqr == 0] = 1.0
    Xn = np.clip((X - medf) / iqr, -8, 8)
    print(f"[Exp M] {len(sh)} real MAST shots | early-quench(<{med:.0f}ms) pos={int(y.sum())}/{len(y)} | pulse dur med {med:.0f}ms")
    rows = []
    for D in [2048, 10000]:
        hc, hr, lg, ml = [], [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            levels = rng.choice(np.array([-1, 1], np.float32), (NLEV, D)); pos = rng.choice(np.array([-1, 1], np.float32), (L, D))
            HV = np.stack([temporal_hv(x["ip"], D, levels, pos) for x in sh])
            idx = rng.permutation(len(HV)); cut = int(0.6 * len(idx)); tr, te = idx[:cut], idx[cut:]
            P = np.stack([bundle(list(HV[tr][y[tr] == c])) if (y[tr] == c).any() else np.zeros(D) for c in (0, 1)])
            Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
            sims = (HV[te] / (np.linalg.norm(HV[te], axis=1, keepdims=True) + 1e-9)) @ Pn.T
            hc.append(auc(sims[:, 1] - sims[:, 0], y[te]))
            w = ridge_fit(HV[tr].astype(np.float64), y[tr]); hr.append(auc(HV[te] @ w, y[te]))
            lg.append(auc(MLP(Xn.shape[1], 2, np.random.default_rng(seed)).train(Xn[tr], y[tr]).fwd(Xn[te]), y[te]))
            ml.append(auc(MLP(Xn.shape[1], 32, np.random.default_rng(seed + 1)).train(Xn[tr], y[tr]).fwd(Xn[te]), y[te]))
        rows.append(dict(D=D, hdc_temporal_centroid=round(float(np.nanmean(hc)), 4),
                         hdc_temporal_ridge=round(float(np.nanmean(hr)), 4),
                         logistic=round(float(np.nanmean(lg)), 4), mlp=round(float(np.nanmean(ml)), 4)))
        r = rows[-1]
        print(f"  D={D}: HDC temporal+centroid={r['hdc_temporal_centroid']}  +ridge={r['hdc_temporal_ridge']}  | logistic={r['logistic']}  MLP={r['mlp']}")
    best = max(r["hdc_temporal_ridge"] for r in rows); mlp = max(r["mlp"] for r in rows)
    print(f"\n  best full-arsenal HDC AUC={best}  vs MLP(features)={mlp}  on REAL long-pulse MAST")
    print(f"  honest caveat: MLP is on hand-crafted features; a waveform-CNN is the stronger NN baseline (not run here).")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    cpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "results_fc_m_mast.csv"))
    with open(cpath, "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
    json.dump({"rows": rows, "best_hdc": best, "mlp": mlp, "n_shots": len(sh), "median_dur_ms": float(med)},
              open(os.path.join(_RES, "fc_m_mast.json"), "w"), indent=2)
    print(f"[done] -> {cpath}")

if __name__ == "__main__":
    main()
