#!/usr/bin/env python3
"""Exp-P — absorb ConformalHDC (arXiv 2602.21446): wrap our HDC predictor in SPLIT CONFORMAL prediction so the
provenance/abstention story becomes a DISTRIBUTION-FREE COVERAGE GUARANTEE, not just an empirical ranking.
Conformal is score-AGNOSTIC (Thm.1 holds for any score under exchangeability), so we test THREE nonconformity
scores on MAST disruption and compare coverage validity + efficiency (set size) + confident-set accuracy:
  (1) RIDGE   : S(x,y) = 1 - p_y   (p from learned ridge readout, sigmoid around train threshold)
  (2) PROV    : S(x,y) = 1 - agree_k(x,y)   (fraction of x's K nearest TRAIN shots with label y) -- OUR signal
  (3) HYBRID  : S(x,y) = 1 - agree_k(x,y)*p_y   (their discount=similarity*ratio logic, provenance as the leg)
Split BY SHOT (each shot = one whole-shot HV, so samples are shot-level exchangeable; a windowed/streaming
predictor would need group-by-shot splits -- noted). Disjoint train / calibration / test.
Binary disruption: a conformal SET of size 1 = confident call; size 2 = ambiguous (defer); size 0 = OOD/abstain.
Metrics per score @ alpha: empirical coverage (must be >=1-alpha), avg set size, %size-1 (confident),
accuracy on the size-1 confident subset (the deployable selective-prediction number).
numpy only.  Usage: fc_p_conformal.py [n_shots]"""
import warnings; warnings.filterwarnings("ignore")
import glob, os, sys, json
import numpy as np
# repo-relative data + results dirs (override the data location with the FUSION_DATA env var)
_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root -> hdc_ops
from hdc_ops import bind, bundle  # this application is built on the hdc-ops operator library

CACHE = os.path.join(_DATA, "mast_ctrl"); SEEDS = [42, 123, 7]
L = 256; NLEV = 32; D = 10000; K = 15

def interp_nan(a):
    a = np.asarray(a, np.float64); idx = np.arange(len(a)); g = np.isfinite(a)
    if g.sum() < 2: return None
    return (np.interp(idx, idx[g], a[g]) if not g.all() else a)

def load(n):
    sh = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        try:
            d = np.load(f, allow_pickle=True); ip = interp_nan(d["ip"])
            if ip is None: continue
            sh.append((ip.astype(np.float32), float(d["dur_ms"])))
            if len(sh) >= n: break
        except Exception: continue
    return sh

def thv(sig, levels, pos):
    w = sig.astype(np.float64); idx = np.linspace(0, len(w)-1, L).astype(int); w = w[idx]
    w = (w-w.mean())/(w.std()+1e-9)
    b = np.clip(((w-w.min())/(np.ptp(w)+1e-9)*(NLEV-1)).astype(int),0,NLEV-1)
    return bundle(list(bind(levels[b], pos))).astype(np.float32)

def ridge(H, y, lam):
    yt = y.astype(np.float64)*2-1
    return np.linalg.solve(H.T@H + lam*np.eye(H.shape[1]), H.T@yt)

def conformal_eval(S_cal_true, S_test_all, y_test, alpha):
    # S_cal_true: (n_cal,) nonconformity of the TRUE label on calibration; S_test_all: (n_test,2) for y in {0,1}
    n = len(S_cal_true)
    k = int(np.ceil((1-alpha)*(n+1)))
    q = np.sort(S_cal_true)[min(k, n)-1]                      # conformal quantile
    sets = S_test_all <= q                                   # (n_test,2) boolean membership
    covered = sets[np.arange(len(y_test)), y_test].mean()    # empirical marginal coverage
    size = sets.sum(1)
    conf = size == 1                                          # confident (singleton) predictions
    pred1 = np.argmax(sets, axis=1)                           # the label in singletons
    acc_conf = (pred1[conf] == y_test[conf]).mean() if conf.any() else float("nan")
    return dict(coverage=float(covered), avg_size=float(size.mean()),
                pct_singleton=float(conf.mean()), pct_abstain=float((size==0).mean()),
                pct_defer=float((size==2).mean()), acc_confident=float(acc_conf))

def main():
    n = int(sys.argv[1]) if len(sys.argv)>1 else 6000
    sh = load(n); durs = np.array([x[1] for x in sh]); med = np.median(durs)
    y = (durs < med).astype(int)
    print(f"[Exp-P] {len(sh)} MAST shots | conformal-wrapped HDC | K={K} D={D}", flush=True)
    agg = {s:{m:[] for m in ("coverage","avg_size","pct_singleton","pct_abstain","pct_defer","acc_confident")}
           for s in ("ridge","prov","hybrid")}
    ALPHA = 0.10
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        levels = rng.choice(np.array([-1,1],np.float32),(NLEV,D)); pos = rng.choice(np.array([-1,1],np.float32),(L,D))
        HV = np.stack([thv(x[0], levels, pos) for x in sh])
        idx = rng.permutation(len(HV))
        n_tr = int(0.5*len(idx)); n_cal = int(0.25*len(idx))         # disjoint train / cal / test, by shot
        tr, cal, te = idx[:n_tr], idx[n_tr:n_tr+n_cal], idx[n_tr+n_cal:]
        w = ridge(HV[tr], y[tr], max(10.0, 0.5*len(tr)))
        thr = np.median(HV[tr]@w); scale = (HV[tr]@w).std()+1e-9
        def ridge_p(H):                                              # per-class prob from ridge score
            p1 = 1/(1+np.exp(-(H@w - thr)/scale)); return np.stack([1-p1, p1], axis=1)
        # provenance: agreement among K nearest TRAIN shots
        Htn = HV[tr]/(np.linalg.norm(HV[tr],axis=1,keepdims=True)+1e-9)
        def agree(H):
            Hn = H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-9)
            nn = np.argsort(-(Hn@Htn.T), axis=1)[:, :K]
            a1 = (y[tr][nn]==1).mean(1); return np.stack([1-a1, a1], axis=1)   # (m,2) agreement per class
        for name in ("ridge","prov","hybrid"):
            if name=="ridge":
                S_cal = 1 - ridge_p(HV[cal]); S_te = 1 - ridge_p(HV[te])
            elif name=="prov":
                S_cal = 1 - agree(HV[cal]); S_te = 1 - agree(HV[te])
            else:
                S_cal = 1 - agree(HV[cal])*ridge_p(HV[cal]); S_te = 1 - agree(HV[te])*ridge_p(HV[te])
            S_cal_true = S_cal[np.arange(len(cal)), y[cal]]
            r = conformal_eval(S_cal_true, S_te, y[te], ALPHA)
            for m,v in r.items(): agg[name][m].append(v)
    print(f"\n  alpha={ALPHA} -> target coverage >= {1-ALPHA:.2f}\n", flush=True)
    print(f"  {'score':>7} | {'coverage':>8} | {'avg_size':>8} | {'%singleton':>10} | {'%defer(=2)':>10} | {'%abstain(=0)':>12} | {'acc@confident':>13}", flush=True)
    out = {}
    for name in ("ridge","prov","hybrid"):
        row = {m: float(np.mean(agg[name][m])) for m in agg[name]}
        out[name] = row
        print(f"  {name:>7} | {row['coverage']:>8.3f} | {row['avg_size']:>8.3f} | {row['pct_singleton']:>10.3f} | "
              f"{row['pct_defer']:>10.3f} | {row['pct_abstain']:>12.3f} | {row['acc_confident']:>13.3f}", flush=True)
    print(f"\n  read: coverage in [{1-ALPHA:.2f}, ~{1-ALPHA+1/( (0.25*len(sh))+1):.3f}] = valid (Thm.1); "
          f"smaller avg_size + higher %singleton at equal coverage = more EFFICIENT; "
          f"acc@confident = deployable selective-prediction accuracy on the calls it WILL make.", flush=True)
    best = max(("ridge","prov","hybrid"), key=lambda s: (out[s]['pct_singleton'], out[s]['acc_confident']))
    print(f"  -> most efficient (most confident singletons): {best}", flush=True)
    json.dump({"alpha":ALPHA,"results":out,"n":len(sh)}, open(os.path.join(_RES, "fc_p.json"),"w"), indent=2)
    print("[done]", flush=True)

if __name__=="__main__":
    main()
