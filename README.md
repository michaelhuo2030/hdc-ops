# hdc-ops

**A clean, well-documented HDC / Vector-Symbolic-Architecture operator library** — the core algebra of Hyperdimensional Computing (bind / bundle / permute / unbind) on plain `numpy`, with an optional Apple-MLX backend. Every function is one well-defined math operation plus a plain-language note on *when to use it*.

> Part of the open **[millisecond-era](https://github.com/michaelhuo2030/millisecond-era)** research program (28nm ReRAM-CIM for ternary/HDC). HDC is open science — the moat is the chip + O(1) continual append + integration, not the algebra. So this armory is open by design: take it, build with it.

## What is HDC, in three lines

Represent every concept as a long (10k+ dim) random ±1 vector. Then:
- any two random vectors are **near-orthogonal** (≈0 similarity) → astronomical room for distinct concepts;
- three operations — **bind** (reversible key↔value), **bundle** (a set/memory similar to all its members), **permute** (encode order) — compose into structured "thoughts" (key-value records, sets, sequences, graphs);
- it is 1-bit, noise-robust, one-shot, and append-forever — which is exactly why it fits cheap, imprecise in-memory-compute hardware.

## Install

```bash
git clone https://github.com/michaelhuo2030/hdc-ops
cd hdc-ops
pip install -e .          # just needs numpy;  pip install -e ".[mlx]" for the Apple-MLX backend
python examples/quickstart.py
```

## Quickstart

```python
import numpy as np
from hdc_ops import bind, bundle, unbind, similarity, search

D = 10000
rng = np.random.default_rng(0)
hv = lambda: rng.choice(np.array([-1, 1]), size=D)

a, b = hv(), hv()
similarity(a, b)                      # ~0.00  — random vectors are near-orthogonal

key, val = hv(), hv()
bound = bind(key, val)                # a new vector, unlike either
similarity(unbind(bound, key), val)   # ~1.00  — bind is reversible (key→value)

mem = bundle([hv(), hv(), val])       # a set / memory
similarity(mem, val)                  # >0     — val is "inside"
```

Output:
```
1. sim(random, random)            = -0.004   (~0: tons of room)
2. sim(bind(key,val), val)        = -0.011   (~0: looks unlike either)
   sim(unbind(bound,key), val)    = +1.000   (~1: perfectly recovered)
3. sim(memory, member x)          = +0.505   (>0: x is inside)
   sim(memory, a stranger)        = +0.008   (~0: not inside)
4. search(x, db) nearest index   = 0        (0 = x, score 1.00)
```

## What's inside

| Module | What it gives you |
|---|---|
| **`hdc_ops`** (`__init__.py`) | the ISA: `bind` `bundle` `permute` `unbind` `soft_bind` `weighted_bundle` `normalize` `threshold` `similarity` `search` `set_membership` `retrieve_role` `sequence` `compare_all` |
| **`encoding.py` / `encoding_extra.py`** | turn real data into HVs: `bipolar`, `simhash`, `sparse_ternary`, `fourier_encode`, `multi_bit` (+ benchmark notes) |
| **`structures.py`** | composite structures: sets, maps, sequences, graphs (`HVSet` / `HVMap` / `HVSequence` / `HVGraph`) |
| **`queries.py`** | high-level query patterns: nearest, role ("what did X do in this scene?"), time ("X three steps ago?"), counterfactual ("if we remove X?") |
| **`time_hdc.py`** | time-series / sequence HDC |
| **`paged_store.py`** | append-forever paged HV store (O(1) write) |
| **`diagnostics.py` / `hdc_audit.py`** | capacity, crosstalk, monosemanticity diagnostics |
| **`hdc_mlx.py`** | optional Apple-MLX backend (GPU/ANE) — import only if you `pip install ".[mlx]"` |

Docstrings are bilingual (English math + 中文大白话) and each carries a *when-to-use-it* note.

## Related

- **[hdc-neon](https://github.com/michaelhuo2030/hdc-neon)** — the same operations, NEON-SIMD accelerated (~89× over numpy).
- **[torchhd](https://github.com/michaelhuo2030/torchhd)** — torchhd fork with a ReRAM-CIM backend.
- **[millisecond-era](https://github.com/michaelhuo2030/millisecond-era)** — the chip program + interactive HDC/ReRAM explainers.

## License

MIT © 2026 Michael (Xiaojie) Huo
