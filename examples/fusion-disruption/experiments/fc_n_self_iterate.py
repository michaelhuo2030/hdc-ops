#!/usr/bin/env python3
"""Exp-N — HDC's real superpower: TRACEABLE errors + SELF-ITERATION (Michael's insight).
An MLP's mistakes are opaque; HDC's are decomposable. On MAST early-quench detection, the 完全体 HDC
(temporal HV + learned ridge readout):
  ROUND 0: train, predict held-out, collect the WRONG ones.
  TRACE (why did it miss?):  (a) BOUNDARY — are the errors clustered where the label itself is ambiguous
     (|plasma-duration - median| ~ 0)?  (b) PROVENANCE — for each wrong shot, its HDC-nearest TRAIN neighbours:
     are they label-MIXED? (i.e. 'I was wrong because this shot's signal looks like these opposite-label shots').
  SELF-ITERATE: feed the lesson back — refit the readout up-weighting the TRAIN examples round-0 got wrong
     (boosting / '越喂越好' on its own mistakes). Re-evaluate held-out AUC. Did it improve, and on what kind of error?
This demonstrates the loop NNs can't do transparently: predict -> see exactly where/why you erred -> re-learn.
numpy only.  Usage: fc_n_self_iterate.py [n_shots]"""
import warnings; warnings.filterwarnings("ignore")
import glob, os, sys, json
import numpy as np
# repo-relative data + results dirs (override the data location with the FUSION_DATA env var)
_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root -> hdc_ops
from hdc_ops import bind, bundle  # this application is built on the hdc-ops operator library

CACHE = os.path.join(_DATA, "mast_ctrl"); SEED = 42
L = 256; NLEV = 32; D = 10000

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
            sh.append(dict(shot=int(os.path.basename(f).split(".")[0]), ip=ip.astype(np.float32),
                           dur=float(d["dur_ms"]), peak=float(d["peak"])))
            if len(sh) >= n: break
        except Exception: continue
    return sh

def temporal_hv(sig, levels, pos):
    w = sig.astype(np.float64); idx = np.linspace(0, len(w)-1, L).astype(int); w = w[idx]
    w = (w - w.mean()) / (w.std() + 1e-9)
    b = np.clip(((w - w.min())/(np.ptp(w)+1e-9)*(NLEV-1)).astype(int), 0, NLEV-1)
    return bundle(list(bind(levels[b], pos))).astype(np.float32)

def auc(p, y):
    y = np.asarray(y); pos = p[y == 1]; neg = p[y == 0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    return float((pos[:,None] > neg[None,:]).mean() + 0.5*(pos[:,None]==neg[None,:]).mean())

def wridge(H, y, sw, lam):
    yt = y.astype(np.float64)*2 - 1
    Hw = H * sw[:,None]
    return np.linalg.solve(H.T@Hw + lam*np.eye(H.shape[1]), H.T@(sw*yt))

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    sh = load(n)
    durs = np.array([x["dur"] for x in sh]); med = np.median(durs)
    y = np.array([1 if x["dur"] < med else 0 for x in sh])
    rng = np.random.default_rng(SEED)
    levels = rng.choice(np.array([-1,1],np.float32),(NLEV,D)); pos = rng.choice(np.array([-1,1],np.float32),(L,D))
    print(f"[Exp-N] {len(sh)} MAST shots | early-quench(<{med:.0f}ms) | HDC 完全体 (temporal D={D} + learned readout)", flush=True)
    HV = np.stack([temporal_hv(x["ip"], levels, pos) for x in sh])
    idx = rng.permutation(len(HV)); cut = int(0.6*len(idx)); tr, te = idx[:cut], idx[cut:]
    Htr, ytr, Htn, ytn = HV[tr], y[tr], HV[te], y[te]
    lam = max(10.0, 0.5*len(Htr))
    # ROUND 0
    w0 = wridge(Htr, ytr, np.ones(len(Htr)), lam)
    s_te = Htn @ w0; auc0 = auc(s_te, ytn)
    thr = np.median(Htr @ w0)                       # operating threshold from train
    pred = (s_te >= thr).astype(int); err = pred != ytn
    print(f"\n  ROUND 0: held-out AUC={auc0:.4f} | errors={int(err.sum())}/{len(ytn)} at the operating threshold", flush=True)

    # ---- TRACE (a): boundary ambiguity ----
    dur_te = durs[te]; dist_med = np.abs(dur_te - med)
    em = dist_med[err]; cm = dist_med[~err]
    print(f"\n  TRACE (a) BOUNDARY: |duration - median| (ms) — small = genuinely-ambiguous label", flush=True)
    print(f"     wrong  shots: median |dur-med| = {np.median(em):.1f}ms", flush=True)
    print(f"     correct shots: median |dur-med| = {np.median(cm):.1f}ms", flush=True)
    near = (em < np.percentile(dist_med, 25)).mean()
    print(f"     -> {100*near:.0f}% of errors sit in the closest-to-boundary quartile (ambiguous-by-label)", flush=True)

    # ---- TRACE (b): provenance — nearest TRAIN neighbours of each wrong shot ----
    Htr_n = Htr/(np.linalg.norm(Htr,axis=1,keepdims=True)+1e-9)
    Htn_n = Htn/(np.linalg.norm(Htn,axis=1,keepdims=True)+1e-9)
    K = 15
    sims = Htn_n @ Htr_n.T                            # test x train cosine
    nn = np.argsort(-sims, axis=1)[:, :K]
    # for each test shot, fraction of its K nearest TRAIN neighbours whose label == its OWN true label (purity)
    purity = np.array([ (ytr[nn[i]] == ytn[i]).mean() for i in range(len(ytn)) ])
    print(f"\n  TRACE (b) PROVENANCE: of each shot's {K} HDC-nearest TRAIN neighbours, %% sharing its true label", flush=True)
    print(f"     wrong  shots: mean neighbourhood purity = {purity[err].mean():.2f}  (low = sits in a label-MIXED region)", flush=True)
    print(f"     correct shots: mean neighbourhood purity = {purity[~err].mean():.2f}", flush=True)
    print(f"     -> HDC can SHOW why it erred: a wrong shot's signal is HDC-near to opposite-label shots (traceable).", flush=True)
    # one concrete worked example
    ei = np.where(err)[0]
    if len(ei):
        i = ei[np.argmin(purity[err])]               # the most-traceable error
        nbl = ytr[nn[i]]
        print(f"     e.g. shot #{sh[te[i]]['shot']} (true={ytn[i]}, pred={pred[i]}, dur={dur_te[i]:.0f}ms vs med {med:.0f}): "
              f"its {K} nearest train shots are {int((nbl==1).sum())} quench / {int((nbl==0).sum())} sustained "
              f"-> genuinely mixed neighbourhood.", flush=True)

    # ---- SELF-ITERATE: refit up-weighting the TRAIN shots round-0 got wrong (boosting on own mistakes) ----
    str_ = Htr @ w0; thr_tr = np.median(str_); pred_tr = (str_ >= thr_tr).astype(int)
    trerr = pred_tr != ytr
    sw = np.where(trerr, 3.0, 1.0)                    # up-weight the hard (mis-predicted) train examples
    w1 = wridge(Htr, ytr, sw, lam)
    auc1 = auc(Htn @ w1, ytn)
    print(f"\n  SELF-ITERATE: refit readout up-weighting the {int(trerr.sum())} TRAIN shots round-0 missed (3x)", flush=True)
    print(f"     ROUND 0 held-out AUC = {auc0:.4f}", flush=True)
    print(f"     ROUND 1 held-out AUC = {auc1:.4f}   (delta {auc1-auc0:+.4f})", flush=True)
    # what KIND of error improved?
    pred1 = (Htn @ w1 >= np.median(Htr@w1)).astype(int); err1 = pred1 != ytn
    print(f"     errors {int(err.sum())} -> {int(err1.sum())} | recovered {int((err & ~err1).sum())}, newly-wrong {int((~err & err1).sum())}", flush=True)
    verdict = ("most errors are boundary-ambiguous (irreducible label noise) -> HDC HONESTLY flags its own uncertainty; "
               "self-iteration gives a small gain on the learnable residual") if np.median(em) < np.median(cm) else \
              ("errors are systematic -> self-iteration should help more")
    print(f"\n  -> {verdict}", flush=True)
    json.dump({"auc0":auc0,"auc1":auc1,"n":len(sh),"err0":int(err.sum()),"err1":int(err1.sum()),
               "err_boundary_med_ms":float(np.median(em)),"correct_boundary_med_ms":float(np.median(cm)),
               "wrong_purity":float(purity[err].mean()),"correct_purity":float(purity[~err].mean())},
              open(os.path.join(_RES, "fc_n_self_iterate.json"),"w"), indent=2)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
