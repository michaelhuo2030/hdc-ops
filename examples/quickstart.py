"""hdc-ops quickstart — the three things high-dimensional vectors can do.
Run:  python examples/quickstart.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from hdc_ops import bind, bundle, unbind, similarity, search

D = 10000
rng = np.random.default_rng(0)
def hv():
    return rng.choice(np.array([-1, 1]), size=D)

# 1) random hypervectors are near-orthogonal (the whole trick)
a, b = hv(), hv()
print(f"1. sim(random, random)            = {float(similarity(a, b)):+.3f}   (~0: tons of room)")

# 2) BIND = reversible key->value
key, val = hv(), hv()
bound = bind(key, val)
print(f"2. sim(bind(key,val), val)        = {float(similarity(bound, val)):+.3f}   (~0: looks unlike either)")
rec = unbind(bound, key)
print(f"   sim(unbind(bound,key), val)    = {float(similarity(rec, val)):+.3f}   (~1: perfectly recovered)")

# 3) BUNDLE = a set / memory you can query
x, y, z = hv(), hv(), hv()
mem = bundle([x, y, z])
print(f"3. sim(memory, member x)          = {float(similarity(mem, x)):+.3f}   (>0: x is inside)")
print(f"   sim(memory, a stranger)        = {float(similarity(mem, hv())):+.3f}   (~0: not inside)")

# 4) nearest-neighbour search over a database
db = np.stack([x, y, z])
idx, score = search(x, db, k=1)
print(f"4. search(x, db) nearest index   = {int(idx[0])}        (0 = x, score {float(score[0]):.2f})")
