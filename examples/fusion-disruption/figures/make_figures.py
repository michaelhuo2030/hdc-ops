#!/usr/bin/env python3
"""make_figures.py — generate the 3 camera-ready figures for the HDC tokamak-disruption paper, from MEASURED data.
fig1 (selective prediction): provenance vs readout-margin, accuracy at coverage {100,80,50}% (fc_n2; core-results).
fig2 (conformal): the 3 nonconformity scores, % confidently-resolved at ≥90% guaranteed coverage (fc_p.json).
fig3 (traceability): wrong-vs-correct shots — boundary distance |dur−median| and nearest-neighbour purity (fc_n.json).
All numbers are measured on real MAST data. matplotlib only. Usage: python make_figures.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- fig 1: selective prediction (measured) ----
cov = [100, 80, 50]
prov = [0.777, 0.883, 0.969]      # provenance (neighbourhood purity) confidence
marg = [0.777, 0.821, 0.895]      # raw readout-margin confidence
fig, ax = plt.subplots(figsize=(4.2, 3.2))
ax.plot(cov, prov, "o-", color="#1f6feb", lw=2, ms=7, label="provenance (neighbour purity)")
ax.plot(cov, marg, "s--", color="#999999", lw=2, ms=6, label="readout margin")
ax.set_xlabel("coverage (% of shots kept)"); ax.set_ylabel("accuracy on kept shots")
ax.set_xticks(cov); ax.invert_xaxis(); ax.set_ylim(0.74, 1.0)
ax.annotate("0.97", (50, 0.969), textcoords="offset points", xytext=(6, -2), color="#1f6feb")
ax.set_title("Selective prediction: defer the ambiguous", fontsize=11)
ax.legend(fontsize=9, loc="lower left", frameon=False)
fig.savefig("fig1_selective_prediction.pdf"); plt.close(fig)

# ---- fig 2: conformal — % confidently resolved at >=90% guaranteed coverage (fc_p.json) ----
scores = ["readout\n(1−p)", "provenance\n(1−agree)", "hybrid\n(prov×ratio)"]
singleton = [0.789, 0.798, 0.842]   # fraction of shots given a single confident label
acc_conf = [0.872, 0.885, 0.884]    # accuracy on those confident calls
fig, ax = plt.subplots(figsize=(4.4, 3.2))
bars = ax.bar(scores, [s*100 for s in singleton], color=["#999999", "#7aa6e0", "#1f6feb"], width=0.6)
ax.set_ylabel("% shots confidently resolved\n(at guaranteed ≥90% coverage)"); ax.set_ylim(0, 100)
for b, s, a in zip(bars, singleton, acc_conf):
    ax.text(b.get_x()+b.get_width()/2, s*100+1.5, f"{s*100:.0f}%\n(acc {a:.2f})", ha="center", fontsize=8.5)
ax.set_title("Conformal: the hybrid score is most efficient", fontsize=11)
fig.savefig("fig2_conformal.pdf"); plt.close(fig)

# ---- fig 3: traceability — wrong vs correct (fc_n.json) ----  [no suptitle: the LaTeX caption carries it; constrained_layout fixes the overlap]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 3.3), constrained_layout=True)
a1.bar(["wrong", "correct"], [43.9, 69.2], color=["#d1495b", "#66a182"], width=0.6)
a1.set_ylabel("median |duration − median|  (ms)", fontsize=10); a1.set_ylim(0, 80)
a1.set_title("errors sit at the boundary", fontsize=10)
for x, v in zip([0, 1], [43.9, 69.2]): a1.text(x, v+1.5, f"{v:.1f}", ha="center", fontsize=9)
a2.bar(["wrong", "correct"], [0.564, 0.842], color=["#d1495b", "#66a182"], width=0.6)
a2.set_ylabel("nearest-15 neighbour purity", fontsize=10); a2.set_ylim(0, 1.05)
a2.set_title("in label-mixed neighbourhoods", fontsize=10)
for x, v in zip([0, 1], [0.564, 0.842]): a2.text(x, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
fig.savefig("fig3_traceability.pdf"); plt.close(fig)

print("[figs] wrote fig1_selective_prediction.pdf, fig2_conformal.pdf, fig3_traceability.pdf")
