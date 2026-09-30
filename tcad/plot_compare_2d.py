"""Overlay the 2D Id-Vg curves for Lch = 80 nm and 300 nm and mark the 1D Vth."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VTH_1D = {"erased": -0.761, "neutral": 0.598, "programmed": 1.921}   # nand_cell_1d.py
COL = {"erased": "#1f77b4", "neutral": "#7f7f7f", "programmed": "#d62728"}


def load(fn):
    d = {}
    with open(os.path.join(HERE, fn)) as f:
        next(f)
        for s, vg, i in csv.reader(f):
            d.setdefault(s, []).append((float(vg), float(i)))
    return {k: np.array(v) for k, v in d.items()}


fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
for a, (fn, L) in zip(ax, [("idvg_2d.csv", 80), ("idvg_2d_L300.csv", 300)]):
    for s, r in load(fn).items():
        a.semilogy(r[:, 0], r[:, 1], color=COL[s], lw=2, label=s)
        a.axvline(VTH_1D[s], color=COL[s], ls=":", lw=1)
    a.set(xlabel="Vg [V]", title=f"Lch = {L} nm", ylim=(1e-13, 1e-3))
    a.grid(alpha=.3)
ax[0].set_ylabel("Id [A/µm]  (Vd = 0.05 V)")
ax[0].legend(title="state (dotted: 1D Vth)", fontsize=8)
fig.suptitle("2D Id–Vg of the charge-trap NAND cell: short-channel roll-off vs long channel")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "nand_cell_2d_compare.png"), dpi=130)
