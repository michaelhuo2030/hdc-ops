# Fusion disruption: an interpretable, abstaining HDC predictor on real tokamak data

A worked application of the [`hdc-ops`](../../) operator library to a safety-critical problem: predicting
disruptions / early quenches on a tokamak, on **real public plasma data**, on commodity hardware.

It is also the reproducibility code for the paper:

> **Provenance is calibration: instance-based hyperdimensional selective prediction for tokamak disruption.**
> Xiaojie (Michael) Huo. *(preprint; arXiv link added on posting.)*

## Why this is here (and what it shows)

`hdc-ops` is the algebra; this is the algebra doing something real. The encoder is literally the library —
each shot becomes `bundle([ bind(level(xₜ), posₜ) for t in time ])` — a 1-bit, 10 000-dimensional hypervector.
On top of that, three properties an opaque neural net does **not** give you, all measured on real data:

1. **Provenance** — every call names the exact prior shots behind it (we keep individual encoded shots, not lossy
   class prototypes), so the explanation *is* the computation.
2. **Knows when it doesn't know** — neighbourhood label-agreement is an intrinsic confidence; deferring the
   shots it flags as ambiguous lifts accuracy 0.78 → 0.97, and beats the raw decision margin.
3. **A distribution-free guarantee** — wrapping it in split conformal prediction gives ≥90 % coverage on real
   MAST data while confidently resolving 84 % of shots.

**Honest framing (the same as the paper):** this is a *methodological* contribution, not an accuracy-superiority
claim. On raw AUC, HDC is competitive with — not beating — a strong neural net, and on continuous shape *control*
it loses (boundary law: HDC wins event/memory detection; neural function-learners win continuous control). The
value is the provenance + abstention + guarantee, which the AUC race doesn't capture.

## Quickstart

```bash
# from the hdc-ops repo root
pip install -e .                                  # the library (numpy only)
pip install -r examples/fusion-disruption/requirements.txt
cd examples/fusion-disruption

python smoke_test.py                              # no download — proves it runs + the encoder == the library
python fetch/fetch_mast.py 11695 30474            # ~12k real MAST shots from the FAIR-MAST open S3 (slow, resumable)
python run_all.py                                 # reproduce every table + the 3 figures
```

`smoke_test.py` needs nothing but `numpy`; the real run needs the fetch dependencies in `requirements.txt`.
Behind a restrictive network? `MAST_PROXY=http://host:port python fetch/fetch_mast.py …`.

## Results → script → paper table

All numbers are **measured on real tokamak data** (MAST 12,451 shots; GOLEM 1,852 shots, cross-machine check).

| ID | What | Script | Headline (measured) |
|----|------|--------|---------------------|
| **R1** | disruption/early-quench *detection* | `experiments/fc_m_mast_disruption.py` (MAST), `fc_b_golem.py` (GOLEM) | HDC 0.911 vs MLP 0.879 (MAST); 0.947 vs 0.936 (GOLEM) — competitive |
| **R2** | continuous state→action shape *control* | `experiments/fc_ctrl_a_shape.py` | MLP 0.836 > HDC 0.703 — the honest boundary where HDC loses |
| **R3** | provenance selective prediction | `experiments/fc_n2_selective.py` | 0.78 → **0.97** keeping the most-confident 50 %; provenance ≫ margin |
| **R4** | traceable errors | `experiments/fc_n_self_iterate.py` | errors sit at the boundary (43.9 vs 69.2 ms) in label-mixed neighbourhoods (56 % vs 84 %) |
| **R5** | what actually moves the needle | `experiments/fc_o_hdc_improve.py` | the learned readout is the biggest lever; coarse (near-ternary) quantization helps + is chip-friendly |
| **R7** | conformal coverage guarantee | `experiments/fc_p_conformal.py` | ≥90 % coverage; the hybrid score confidently resolves **84 %** of shots |

Figures (`figures/make_figures.py`) regenerate the three camera-ready plots from these measured numbers.

## Data

- **MAST** — [FAIR-MAST](https://mastapp.site) open S3 (`s3.echo.stfc.ac.uk`, `mast/level1/shots/<id>.zarr`),
  UK Atomic Energy Authority. We use the `amc` plasma current + coil currents and `amb` magnetics.
- **GOLEM** — the open [GOLEM tokamak](http://golem.fjfi.cvut.cz) (CTU Prague), plasma-detection CSVs.
- See [`data/README.md`](data/README.md) for the per-shot schema, the fetchers, and licensing/attribution.

**Label caveat:** lacking a physics-grade disruption flag in this data tier, the label is an early-quench
*proxy* (plasma duration below the median). The *methodology* is the contribution; absolute AUC is on the proxy
task. Physics-labelled data (DIII-D / EAST / J-TEXT) is the obvious follow-up.

## License

Code: MIT (same as `hdc-ops`). The MAST and GOLEM data are the property of their providers — cite them, don't
relicense them. © 2026 Xiaojie (Michael) Huo.
