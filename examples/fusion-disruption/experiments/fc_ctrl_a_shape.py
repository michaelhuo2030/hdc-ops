#!/usr/bin/env python3
"""Exp-Ctrl-A — REAL shape/position CONTROL (dimension a, ~10kHz loop), the honest negative control.
This is CONTROL (state -> ACTION), not detection. Task = behavior-clone the real MAST controller:
   state(t) = magnetic sensors + plasma_current   ->   action(t) = PF coil currents (what it commanded)
Data: data/mast_ctrl/<shot>.npz (rich pull: sensors + actuators aligned on an L-grid), via fetch/fetch_mast.py.

Boundary-law L1 PREDICTION (pre-registered, honest): learning a continuous state->action map from real noisy
data is FUNCTION LEARNING = the NN's job -> HDC (random-projection + learned linear readout) should TIE the linear
baseline and LOSE to a tuned MLP on accuracy. HDC's value here is NOT accuracy, it's EXECUTION LATENCY (Exp E:
65us reflex step) + writable cache. So: if HDC < MLP here, that's the EXPECTED boundary, and the verdict is
"HDC = fast cheap executor, NN = the learner" -> HYBRID. If HDC somehow BEATS a tuned MLP, that's an anomaly to
audit (suspect my apparatus first). Metric = held-out R^2 per coil (variance explained), shot-split, 3 seeds.
Self-contained numpy. Usage: fc_ctrl_a_shape.py [min_shots]"""
import warnings; warnings.filterwarnings("ignore")
import glob, os, sys, json, csv
import numpy as np
# repo-relative data + results dirs (override the data location with the FUSION_DATA env var)
_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); os.makedirs(_RES, exist_ok=True)
from collections import Counter

CACHE = os.path.join(_DATA, "mast_ctrl"); SEEDS = [42, 123, 7]
STRIDE = 8          # subsample timesteps (L=512 -> 64 samples/shot) to keep it tractable
D = 4096            # HDC dimension

def interp_nan(a):
    a = a.astype(np.float64); idx = np.arange(len(a)); g = np.isfinite(a)
    if g.sum() < 2: return None
    return (np.interp(idx, idx[g], a[g]) if not g.all() else a).astype(np.float32)

def load():
    shots = []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        try:
            d = np.load(f, allow_pickle=True)
            cn = list(d["coil_names"]); mn = list(d["mag_names"])
            if not cn or not mn: continue
            shots.append(dict(f=f, ip=d["ip"], coils=d["coils"], mag=d["mag"],
                              cn=[str(x) for x in cn], mn=[str(x) for x in mn]))
        except Exception:
            continue
    return shots

def aligned_set(shots):
    # use the channels present in MOST shots (consistent state/action space)
    cc = Counter(); mc = Counter()
    for s in shots:
        cc.update(s["cn"]); mc.update(s["mn"])
    coils = [c for c, n in cc.most_common() if n >= 0.8 * len(shots)]
    mags = [m for m, n in mc.most_common() if n >= 0.8 * len(shots)]
    return coils, mags

def build(shots, coils, mags, H=8):
    # AUDIT: CAUSAL framing — predict action at t+H from state at t. The earlier same-timestep version
    # (action[t] from magnetics[t]) is partly trivial because the flux-loop B-field IS produced by the
    # coils → near-deterministic inversion, not control. Predicting H steps ahead removes that instant coupling.
    X, Y, sid = [], [], []
    for i, s in enumerate(shots):
        ci = {c: k for k, c in enumerate(s["cn"])}; mi = {m: k for k, m in enumerate(s["mn"])}
        if not all(c in ci for c in coils) or not all(m in mi for m in mags): continue
        ip = interp_nan(s["ip"])
        mg = [interp_nan(s["mag"][mi[m]]) for m in mags]
        co = [interp_nan(s["coils"][ci[c]]) for c in coils]
        if ip is None or any(v is None for v in mg) or any(v is None for v in co): continue
        state = np.column_stack([ip] + mg); action = np.column_stack(co)
        if len(state) <= H: continue
        Xs, Ys = state[:len(state)-H], action[H:]    # state(t) -> action(t+H)
        sl = slice(0, len(Xs), STRIDE)
        X.append(Xs[sl]); Y.append(Ys[sl]); sid.append(np.full((Xs[sl].shape[0],), i))
    return np.vstack(X), np.vstack(Y), np.concatenate(sid)

class MLP:
    # AUDIT: strengthened baseline (suspect #1 for the HDC>MLP anomaly was an under-tuned MLP, the fc_d arm-B lesson).
    # He-init + momentum SGD + many epochs + wider hidden → a fair, properly-trained NN opponent.
    def __init__(s, di, h, do, rng):
        s.W1=rng.standard_normal((di,h))/np.sqrt(di); s.b1=np.zeros(h)
        s.W2=rng.standard_normal((h,do))/np.sqrt(h); s.b2=np.zeros(do)
        s.vW1=0.0; s.vb1=0.0; s.vW2=0.0; s.vb2=0.0
    def fwd(s, X): s.z=np.maximum(0,X@s.W1+s.b1); return s.z@s.W2+s.b2
    def train(s, X, Y, ep=1500, lr=0.02, mom=0.9):
        n=len(X)
        for _ in range(ep):
            p=s.fwd(X); g=(p-Y)/n
            gW2=s.z.T@g; gb2=g.sum(0); gh=(g@s.W2.T)*(s.z>0); gW1=X.T@gh; gb1=gh.sum(0)
            s.vW2=mom*s.vW2-lr*gW2; s.W2+=s.vW2; s.vb2=mom*s.vb2-lr*gb2; s.b2+=s.vb2
            s.vW1=mom*s.vW1-lr*gW1; s.W1+=s.vW1; s.vb1=mom*s.vb1-lr*gb1; s.b1+=s.vb1
        return s

def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum(0); ss_tot = ((true - true.mean(0)) ** 2).sum(0) + 1e-9
    return float(np.mean(1 - ss_res / ss_tot))

def main():
    min_shots = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    shots = load()
    if len(shots) < min_shots:
        print(f"[Exp-Ctrl-A] only {len(shots)} rich shots (< {min_shots}); not enough yet."); return
    coils, mags = aligned_set(shots)
    X, Y, sid = build(shots, coils, mags)
    print(f"[Exp-Ctrl-A] shape/position CONTROL | {len(set(sid))} shots, {len(X)} samples")
    print(f"  state = plasma_current + {len(mags)} magnetics ({mags}) -> action = {len(coils)} coils ({coils})")
    res = {"linear": [], "mlp": [], "hdc": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        ush = np.array(sorted(set(sid))); rng.shuffle(ush); cut = int(0.6 * len(ush))
        trsh, tesh = set(ush[:cut]), set(ush[cut:])
        tr = np.array([s in trsh for s in sid]); te = ~tr
        Xtr, Ytr, Xte, Yte = X[tr], Y[tr], X[te], Y[te]
        mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-9; my, sy = Ytr.mean(0), Ytr.std(0) + 1e-9
        Xtr_, Xte_ = (Xtr - mx) / sx, (Xte - mx) / sx; Ytr_, Yte_ = (Ytr - my) / sy, (Yte - my) / sy
        # linear ridge
        lam = 1.0; W = np.linalg.solve(Xtr_.T @ Xtr_ + lam * np.eye(Xtr_.shape[1]), Xtr_.T @ Ytr_)
        res["linear"].append(r2(Xte_ @ W, Yte_))
        # MLP (audited: wider 128-hidden, He-init + momentum, 1500 epochs — a fair opponent)
        m = MLP(Xtr_.shape[1], 128, Ytr_.shape[1], np.random.default_rng(seed)).train(Xtr_, Ytr_)
        res["mlp"].append(r2(m.fwd(Xte_), Yte_))
        # HDC: random-projection state -> sign HV -> learned ridge readout to action (HDC's home for given structure)
        R = rng.standard_normal((Xtr_.shape[1], D)).astype(np.float32)
        Htr = np.sign(Xtr_ @ R); Hte = np.sign(Xte_ @ R)
        lamh = max(10.0, 0.5 * len(Htr))
        Wh = np.linalg.solve(Htr.T @ Htr + lamh * np.eye(D), Htr.T @ Ytr_)
        res["hdc"].append(r2(Hte @ Wh, Yte_))
    A = {k: round(float(np.mean(v)), 4) for k, v in res.items()}
    print(f"  held-out R^2 (variance explained, higher=better):")
    print(f"    linear ridge : {A['linear']}")
    print(f"    MLP (128h)   : {A['mlp']}")
    print(f"    HDC (RP+ridge): {A['hdc']}")
    verdict = ("HDC≈linear, MLP leads (EXPECTED boundary L1: control-law learning is the NN's job; "
               "HDC's edge = 65us execution + writable cache, not accuracy → HYBRID)") if A['mlp'] >= A['hdc'] - 0.02 else \
              ("HDC≥MLP — ANOMALY, audit apparatus before claiming")
    print(f"  -> {verdict}")
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
    cpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "results_fc_ctrl_a.csv"))
    with open(cpath, "w", newline="") as f:
        w = csv.writer(f); w.writerow(list(A.keys())); w.writerow(list(A.values()))
    json.dump({"r2": A, "n_shots": len(set(sid)), "n_samples": len(X), "coils": coils, "mags": mags,
               "verdict": verdict}, open(os.path.join(_RES, "fc_ctrl_a.json"), "w"), indent=2)
    print(f"[done] -> {cpath}")

if __name__ == "__main__":
    main()
