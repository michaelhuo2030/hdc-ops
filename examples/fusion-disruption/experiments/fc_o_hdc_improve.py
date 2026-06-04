#!/usr/bin/env python3
"""Exp-O — improve HDC the algorithm: (A) WEAPONRY SWEEP to find the big levers, (B) DISASTER up-weighting.
On MAST early-quench detection.
(A) Ablate from a base config (temporal HV, D=10000, NLEV=32, ridge readout); vary ONE knob at a time:
    D in {2048,10000,30000}, NLEV in {8,32,64}, readout {centroid,ridge}, encoding {temporal, temporal+bigram}.
    Report held-out AUC; the biggest delta = the elephant.
(B) RARE-DISASTER regime (disaster = shortest 15% of shots = rare, safety-critical positive). Compare readout
    training: (i) unweighted, (ii) up-weight the rare disaster class (inverse-freq), (iii) +hard-example boost.
    Metric = RECALL at fixed false-alarm rate (5%,10%) — the real disruption-prediction metric — + AUC.
    Tests Michael's idea: aggregate the rare disasters + weight them up -> catch more of the critical events?
    (Last experiment already showed up-weighting boundary-ambiguous MISSES doesn't help — irreducible noise.)
numpy only.  Usage: fc_o_hdc_improve.py [n_shots]"""
import warnings; warnings.filterwarnings("ignore")
import glob, os, sys, json
import numpy as np
# repo-relative data + results dirs (override the data location with the FUSION_DATA env var)
_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"); os.makedirs(_RES, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root -> hdc_ops
from hdc_ops import bind, bundle  # this application is built on the hdc-ops operator library

CACHE = os.path.join(_DATA, "mast_ctrl"); SEEDS = [42, 123, 7]; L = 256

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

def encode(ips, D, NLEV, rng, mode):
    levels = rng.choice(np.array([-1,1],np.float32),(NLEV,D)); pos = rng.choice(np.array([-1,1],np.float32),(L,D))
    lev2 = rng.choice(np.array([-1,1],np.float32),(NLEV,D)) if mode=="bigram" else None
    out = np.empty((len(ips),D),np.float32)
    for k,sig in enumerate(ips):
        w=sig.astype(np.float64); idx=np.linspace(0,len(w)-1,L).astype(int); w=w[idx]
        w=(w-w.mean())/(w.std()+1e-9); b=np.clip(((w-w.min())/(np.ptp(w)+1e-9)*(NLEV-1)).astype(int),0,NLEV-1)
        acc=(levels[b]*pos).sum(0)
        if mode=="bigram":   # add local-dynamics bigram bindings (captures dI/dt-like structure)
            acc=acc+ (lev2[b[:-1]]*levels[b[1:]]*pos[:-1]).sum(0)
        out[k]=np.sign(acc)
    return out

def auc(p,y):
    y=np.asarray(y); P=p[y==1]; N=p[y==0]
    if len(P)==0 or len(N)==0: return float("nan")
    return float((P[:,None]>N[None,:]).mean()+0.5*(P[:,None]==N[None,:]).mean())
def tpr_at_fpr(score,y,f):
    neg=score[y==0]; thr=np.quantile(neg,1-f); return float((score[y==1]>=thr).mean())
def wridge(H,y,sw,lam):
    yt=y.astype(np.float64)*2-1; n,D=H.shape
    s=np.sqrt(sw); Hs=H*s[:,None]; ys=s*yt           # absorb sample weights into the design
    if D>n:                                          # dual form: solve the n×n system, not D×D (O(n^3) not O(D^3))
        G=Hs@Hs.T; G[np.diag_indices_from(G)]+=lam
        return Hs.T@np.linalg.solve(G,ys)
    return np.linalg.solve(Hs.T@Hs+lam*np.eye(D),Hs.T@ys)

def evalcfg(ips,durs,D,NLEV,mode,readout):
    aucs=[]
    for s in SEEDS:
        rng=np.random.default_rng(s); med=np.median(durs); y=(durs<med).astype(int)
        HV=encode(ips,D,NLEV,rng,mode)
        idx=rng.permutation(len(HV)); cut=int(0.6*len(idx)); tr,te=idx[:cut],idx[cut:]
        if readout=="centroid":
            P=np.stack([bundle(list(HV[tr][y[tr]==c])) if (y[tr]==c).any() else np.zeros(D) for c in (0,1)])
            Pn=P/(np.linalg.norm(P,axis=1,keepdims=True)+1e-9); sims=(HV[te]/(np.linalg.norm(HV[te],axis=1,keepdims=True)+1e-9))@Pn.T
            sc=sims[:,1]-sims[:,0]
        else:
            w=wridge(HV[tr],y[tr],np.ones(len(tr)),max(10.0,0.5*len(tr))); sc=HV[te]@w
        aucs.append(auc(sc,y[te]))
    return float(np.mean(aucs))

def main():
    n=int(sys.argv[1]) if len(sys.argv)>1 else 6000
    sh=load(n); ips=[x[0] for x in sh]; durs=np.array([x[1] for x in sh])
    print(f"[Exp-O] {len(sh)} MAST shots", flush=True)
    base=dict(D=10000,NLEV=32,mode="temporal",readout="ridge")
    b_auc=evalcfg(ips,durs,**base); print(f"\n  (A) WEAPONRY SWEEP  base(D=10000,NLEV=32,temporal,ridge) AUC={b_auc:.4f}", flush=True)
    sweeps=[("D",2048),("D",30000),("NLEV",8),("NLEV",64),("readout","centroid"),("mode","bigram")]
    rows=[]
    for knob,val in sweeps:
        cfg=dict(base); cfg[knob]=val; a=evalcfg(ips,durs,**cfg); rows.append((f"{knob}={val}",a,a-b_auc))
        print(f"     {knob}={val:<9} AUC={a:.4f}  (delta {a-b_auc:+.4f})", flush=True)
    rows.sort(key=lambda r:-abs(r[2]))
    print(f"     -> BIGGEST LEVER (elephant): {rows[0][0]} (|delta|={abs(rows[0][2]):.4f})", flush=True)

    # (B) DISASTER up-weighting in a RARE regime
    print(f"\n  (B) DISASTER UP-WEIGHTING — rare regime (disaster = shortest 15% = rare critical positive)", flush=True)
    q=np.quantile(durs,0.15); y=(durs<q).astype(int)
    print(f"     disasters: {int(y.sum())}/{len(y)} ({100*y.mean():.0f}%) | metric = recall(TPR) at fixed false-alarm", flush=True)
    res={k:{"auc":[],"r5":[],"r10":[]} for k in ("unweighted","disaster-upweight","+hard-boost")}
    D=base["D"]
    for s in SEEDS:
        rng=np.random.default_rng(s); HV=encode(ips,D,32,rng,"temporal")
        idx=rng.permutation(len(HV)); cut=int(0.6*len(idx)); tr,te=idx[:cut],idx[cut:]
        Htr,ytr,Hte,yte=HV[tr],y[tr],HV[te],y[te]; lam=max(10.0,0.5*len(tr))
        # (i) unweighted
        w0=wridge(Htr,ytr,np.ones(len(tr)),lam); s0=Hte@w0
        # (ii) disaster-class up-weight (inverse frequency)
        sw=np.where(ytr==1, (ytr==0).sum()/max((ytr==1).sum(),1), 1.0)
        w1=wridge(Htr,ytr,sw,lam); s1=Hte@w1
        # (iii) + hard-example boost (train shots w0 misranks near its own threshold)
        thr=np.median(Htr@w0); hard=((Htr@w0>=thr).astype(int)!=ytr); sw2=sw*np.where(hard,2.0,1.0)
        w2=wridge(Htr,ytr,sw2,lam); s2=Hte@w2
        for k,sc in (("unweighted",s0),("disaster-upweight",s1),("+hard-boost",s2)):
            res[k]["auc"].append(auc(sc,yte)); res[k]["r5"].append(tpr_at_fpr(sc,yte,0.05)); res[k]["r10"].append(tpr_at_fpr(sc,yte,0.10))
    print(f"     {'scheme':>18} | {'AUC':>6} | {'recall@5%FPR':>12} | {'recall@10%FPR':>13}", flush=True)
    for k in ("unweighted","disaster-upweight","+hard-boost"):
        a=np.mean(res[k]['auc']); r5=np.mean(res[k]['r5']); r10=np.mean(res[k]['r10'])
        print(f"     {k:>18} | {a:>6.3f} | {r5:>12.3f} | {r10:>13.3f}", flush=True)
    u=res["unweighted"]; d=res["disaster-upweight"]
    print(f"     -> disaster-upweight vs unweighted: recall@10%FPR {np.mean(u['r10']):.3f} -> {np.mean(d['r10']):.3f} "
          f"(delta {np.mean(d['r10'])-np.mean(u['r10']):+.3f}); AUC {np.mean(u['auc']):.3f}->{np.mean(d['auc']):.3f}", flush=True)
    json.dump({"sweep":[(r[0],r[1]) for r in rows],"base_auc":b_auc,
               "disaster":{k:{m:float(np.mean(v)) for m,v in res[k].items()} for k in res}},
              open(os.path.join(_RES, "fc_o.json"),"w"),indent=2)
    print("[done]", flush=True)

if __name__=="__main__":
    main()
