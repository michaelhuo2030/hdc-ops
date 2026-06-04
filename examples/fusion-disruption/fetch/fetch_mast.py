#!/usr/bin/env python3
"""Fetch real MAST tokamak shots from the open FAIR-MAST S3 store into data/mast_ctrl/<shot>.npz.

FAIR-MAST (UKAEA) publishes MAST diagnostic data as one Zarr per shot on a public, anonymous S3 bucket:
  endpoint  https://s3.echo.stfc.ac.uk        path  mast/level1/shots/<shot>.zarr   (zarr v2, consolidated)
  paper     Cummings et al., "FAIR-MAST: a FAIR data system for the MAST tokamak" (SoftwareX 2024)

Per shot we pull, aligned on a common downsampled time grid:
  ip          amc/plasma_current  (cropped to the plasma phase |Ip| > 0.2*peak)
  coils       up to 6 amc *_coil_current channels   (PF coil currents = the controller's actuators)
  mag         up to 6 amb magnetics channels        (flux loops / Mirnov = MHD precursors / sensors)
  dur_ms, peak, t, coil_names, mag_names
Only real-plasma shots are kept (peak |Ip| > 20 kA; commissioning/test shots are skipped). Resumable
(.npz + .skip markers), threaded.

NETWORK: direct access works from most of the world. If your network can't reach the UK S3 directly, set a
proxy:  MAST_PROXY=http://host:port python fetch_mast.py 28000 30474
DATA LOCATION: writes under <repo>/examples/fusion-disruption/data/mast_ctrl (override with FUSION_DATA).
DEPENDENCIES: pip install s3fs xarray zarr numpy   (also installs fsspec).
DISCOVER CHANNELS: `python fetch_mast.py --inspect <shot>` lists the groups/variables available for one shot.

Usage:  python fetch_mast.py [START] [END] [WORKERS]      e.g.  python fetch_mast.py 28000 30474 8
        python fetch_mast.py --inspect 28819
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")

PROXY = os.environ.get("MAST_PROXY")            # unset = direct (works internationally); set = route via proxy
if PROXY:
    os.environ["http_proxy"] = PROXY; os.environ["https_proxy"] = PROXY
    os.environ.pop("all_proxy", None); os.environ.pop("ALL_PROXY", None)   # SOCKS env breaks aiohttp CONNECT

import numpy as np

ENDPOINT = "https://s3.echo.stfc.ac.uk"
_DATA = os.environ.get("FUSION_DATA") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(_DATA, "mast_ctrl")
L = 512; MINPEAK = 20.0; RETRIES = 4; N_COIL = 6; N_MAG = 6


def _fs():
    import s3fs
    cfg = {"connect_timeout": 15, "read_timeout": 45, "retries": {"max_attempts": 4}}
    if PROXY:
        cfg["proxies"] = {"http": PROXY, "https": PROXY}
    return s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": ENDPOINT}, config_kwargs=cfg)


def _open(fs, sh, grp):
    import xarray as xr
    last = ""
    for _ in range(RETRIES):
        try:
            return xr.open_zarr(fs.get_mapper(f"mast/level1/shots/{sh}.zarr"), group=grp, consolidated=True)
        except Exception as e:
            last = type(e).__name__; time.sleep(1.0)
    return last                                  # str return = failure


def _dsamp(a):
    a = np.asarray(a, dtype="f4")
    if len(a) < 2: return None
    return a[np.linspace(0, len(a) - 1, L).astype(int)]


def inspect(shot):
    """Print the groups/variables available for one shot (use this if channel names ever change)."""
    fs = _fs()
    for grp in ("amc", "amb"):
        ds = _open(fs, shot, grp)
        if isinstance(ds, str):
            print(f"  group {grp!r}: open failed ({ds})"); continue
        print(f"  group {grp!r}: {list(ds.data_vars)[:40]}")
        ds.close()


def pull(sh):
    fp = os.path.join(OUT, f"{sh}.npz"); skip = os.path.join(OUT, f"{sh}.skip")
    if os.path.exists(fp) or os.path.exists(skip): return ("cached", sh, None)
    fs = _fs()
    def mark():
        try: open(skip, "w").close()
        except Exception: pass
    amc = _open(fs, sh, "amc")
    if isinstance(amc, str): return ("miss", sh, amc)                 # transient: no skip mark -> retried next sweep
    if "plasma_current" not in amc.data_vars: mark(); return ("noip", sh, None)
    ip = np.asarray(amc["plasma_current"].values, dtype="f4")
    pk = float(np.nanmax(np.abs(ip)))
    if not np.isfinite(pk) or pk <= MINPEAK: mark(); return ("noplasma", sh, pk)
    m = np.abs(ip) > 0.2 * pk; idx = np.where(m)[0]
    if len(idx) < 20: mark(); return ("noplasma", sh, pk)
    i0, i1 = int(idx[0]), int(idx[-1])
    if i1 - i0 < 10: mark(); return ("noplasma", sh, pk)
    tn = [c for c in amc.coords if "time" in c.lower()]
    t = np.asarray(amc[tn[0]].values, dtype="f4") if tn else np.arange(len(ip), dtype="f4")
    dur_ms = float((t[i1] - t[i0]) * 1000.0); sl = slice(i0, i1 + 1)
    grid = np.linspace(float(t[i0]), float(t[i1]), L)
    coils, cnames = [], []
    for v in [v for v in amc.data_vars if v.endswith("_coil_current")][:N_COIL]:
        try:
            a = np.asarray(amc[v].values, dtype="f4")
            if len(a) == len(ip): coils.append(np.interp(grid, t, a).astype("f4")); cnames.append(v)
        except Exception: pass
    amc.close()
    mags, mnames = [], []
    amb = _open(fs, sh, "amb")
    if not isinstance(amb, str):
        mt = [c for c in amb.coords if "time" in c.lower()]
        mtv = np.asarray(amb[mt[0]].values, dtype="f4") if mt else None
        for v in list(amb.data_vars)[:N_MAG]:
            try:
                a = np.asarray(amb[v].values, dtype="f4")
                if mtv is not None and len(a) == len(mtv):
                    mags.append(np.interp(grid, mtv, a).astype("f4")); mnames.append(v)
            except Exception: pass
        amb.close()
    ipd = _dsamp(ip[sl])
    if ipd is None: mark(); return ("noplasma", sh, pk)
    np.savez_compressed(fp, ip=ipd, t=grid.astype("f4"), dur_ms=dur_ms, peak=pk,
                        coils=np.stack(coils) if coils else np.zeros((0, L), "f4"), coil_names=np.array(cnames),
                        mag=np.stack(mags) if mags else np.zeros((0, L), "f4"), mag_names=np.array(mnames))
    return ("ok", sh, (len(coils), len(mags), dur_ms))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--inspect":
        inspect(int(sys.argv[2])); return
    from concurrent.futures import ThreadPoolExecutor, as_completed
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 28000
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 30474
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    os.makedirs(OUT, exist_ok=True)
    shots = list(range(start, end))
    print(f"[fetch_mast] {len(shots)} shots [{start},{end}) x{workers} -> {OUT}"
          f"{' via proxy ' + PROXY if PROXY else ' (direct S3)'}", flush=True)
    t0 = time.time(); tally = {}; done = 0; nc = []; nm = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(pull, sh): sh for sh in shots}
        for fu in as_completed(futs):
            st, sh, info = fu.result(); tally[st] = tally.get(st, 0) + 1; done += 1
            if st == "ok": nc.append(info[0]); nm.append(info[1])
            if done % 100 == 0:
                el = time.time() - t0; r = done / max(el, 1e-9)
                print(f"  {done}/{len(shots)} ok={tally.get('ok',0)} miss={tally.get('miss',0)} "
                      f"noplasma={tally.get('noplasma',0)} | {r:.2f}/s ETA {(len(shots)-done)/max(r,1e-9)/60:.0f}min", flush=True)
    print(f"\n[fetch_mast] DONE {done} in {(time.time()-t0)/60:.1f}min tally={tally} | "
          f"saved {tally.get('ok',0)} shots (avg {np.mean(nc) if nc else 0:.1f} coils + {np.mean(nm) if nm else 0:.1f} magnetics)", flush=True)


if __name__ == "__main__":
    main()
