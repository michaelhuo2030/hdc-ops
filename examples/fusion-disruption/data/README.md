# Data

This folder is populated by the fetchers (it ships empty — see `.gitignore`). Nothing here is redistributed;
both sources are open and you download them directly from the providers.

## MAST  →  `data/mast_ctrl/<shot>.npz`

Fetched by [`../fetch/fetch_mast.py`](../fetch/fetch_mast.py) from the **FAIR-MAST** open S3 store:

- endpoint `https://s3.echo.stfc.ac.uk` (anonymous), path `mast/level1/shots/<shot>.zarr` (zarr v2, consolidated)
- only real-plasma shots are kept (peak |Ip| > 20 kA; commissioning/test shots skipped)

Per-shot `.npz` schema (what the experiments read):

| key | shape | meaning |
|-----|-------|---------|
| `ip` | (512,) | plasma current waveform (`amc/plasma_current`), cropped to the plasma phase, downsampled |
| `t` | (512,) | time grid (s) |
| `dur_ms` | scalar | plasma duration (ms) — the early-quench label is derived from this |
| `peak` | scalar | peak \|Ip\| (kA) |
| `coils` | (≤6, 512) | PF coil currents (`amc/*_coil_current`) — the controller's actuators (R2) |
| `coil_names`, `mag_names` | (≤6,) | channel names |
| `mag` | (≤6, 512) | magnetics / flux loops (`amb`) — MHD precursors / sensors |

Run `python ../fetch/fetch_mast.py --inspect <shot>` to list the groups/variables for one shot if channel
names ever change in a future data release.

## GOLEM  →  `data/golem/<shot>_<signal>.npy`

Fetched by [`../fetch/fetch_golem.py`](../fetch/fetch_golem.py) (or auto-fetched by `fc_b_golem.py`) from the
open **GOLEM tokamak** (CTU Prague) over plain HTTP:
`http://golem.fjfi.cvut.cz/shotdir/<shot>/Diagnostics/PlasmaDetection/{V_loop,ql_I_p}.csv`. Each `.npy` is a
2-column `(time, value)` array.

## Attribution

- **FAIR-MAST** — N. Cummings et al., *FAIR-MAST: a FAIR data system for the MAST tokamak* (data descriptor,
  SoftwareX 2024); UK Atomic Energy Authority. Data under the FAIR-MAST terms.
- **GOLEM** — the GOLEM tokamak, FNSPE, Czech Technical University in Prague (open educational tokamak).

Please cite the data providers if you use these datasets.
