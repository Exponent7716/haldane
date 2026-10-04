"""Overlay PBE and HSE06 Si bands.

HSE06 eigenvalues are exact only on the 4x4x4 SCF mesh (si_hse.chk). The path bands are
PBE (si_pbe_bands.npz) plus the band-resolved correction  d_n(k) = E_HSE06 - E_PBE
interpolated from the mesh points (periodic images + thin-plate RBF).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyscf import lib
from scipy.interpolate import RBFInterpolator

H = 27.211386245988
a = 5.431
B = a / 0.529177210903

p = lib.chkfile.load("si_scf.chk", "scf")
h = lib.chkfile.load("si_hse.chk", "scf")
Ep = np.array([np.sort(e) for e in p["mo_energy"]]) * H
Eh = np.array([np.sort(e) for e in h["mo_energy"]]) * H
kmesh = np.array(p["kpts"]) * B / (2 * np.pi)  # cartesian, units of 2pi/a
nb = 12                                        # bands to correct (4 valence + 8 conduction)
delta = (Eh - Ep)[:, :nb]

# reciprocal lattice (fcc primitive -> bcc), units 2pi/a
G = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
shifts = np.array([i*G[0] + j*G[1] + k*G[2]
                   for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
pts = np.vstack([kmesh + s for s in shifts])
vals = np.vstack([delta] * len(shifts))
rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=1e-6)

d = np.load("si_pbe_bands.npz")
Epath, xs, kcart, no = d["E"], d["xs"], d["kcart"], int(d["nocc"])
Hpath = Epath.copy()
Hpath[:, :nb] += rbf(kcart)

# sanity check: interpolation reproduces the exact mesh values at Gamma, X, L
for name, k in [("Gamma", [0, 0, 0]), ("X", [0, 1, 0]), ("L", [.5, .5, .5])]:
    i = np.argmin(np.linalg.norm(kmesh - np.array(k), axis=1))
    err = np.abs(rbf(kmesh[i][None])[0] - delta[i]).max()
    print(f"interp. error at mesh point {name}: {err:.4f} eV")

vp, vh = Epath[:, no-1].max(), Hpath[:, no-1].max()
gp, gh = Epath[:, no].min() - vp, Hpath[:, no].min() - vh
print(f"indirect gap  PBE = {gp:.3f} eV   HSE06 (interp.) = {gh:.3f} eV   exp. = 1.12 eV")
ik = Hpath[:, no].argmin()
print(f"CBM k = {kcart[ik]} (cartesian, 2pi/a)")

# tick positions: segments of 40 points, same path as si_band.py
labels = ["L", r"$\Gamma$", "X", "U|K", r"$\Gamma$"]
nseg = len(xs) // 4
tick_idx = [0, nseg - 1, 2*nseg - 1, 3*nseg + 0, 4*nseg - 1]
# X|U joins K: the figure keeps the original layout where U and K coincide on the axis
fig, ax = plt.subplots(figsize=(5.5, 6.5))
for b in range(no + 4):
    ax.plot(xs, Epath[:, b] - vp, color="0.55", lw=1.4, label="PBE" if b == 0 else None)
    ax.plot(xs, Hpath[:, b] - vh, color="C3", lw=1.4, label="HSE06 (interp.)" if b == 0 else None)
ticks = [xs[i] for i in tick_idx]
for t in ticks:
    ax.axvline(t, color="gray", lw=0.5)
ax.set_xticks(ticks); ax.set_xticklabels(labels)
ax.axhline(0, color="k", ls="--", lw=0.6)
ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-13, 8)
ax.set_ylabel(r"E - E$_{VBM}$ (eV)")
ax.set_title(f"Si bands: PBE {gp:.2f} eV vs HSE06 {gh:.2f} eV (exp. 1.12)")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("si_bands_pbe_vs_hse.png", dpi=200)
