"""Si band structure from first principles (PySCF periodic DFT, PBE + GTH pseudopotential)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyscf.pbc import gto, dft
from pyscf.data.nist import HARTREE2EV

a = 5.431  # Angstrom, experimental lattice constant
cell = gto.Cell()
cell.unit = "A"
cell.a = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])  # fcc primitive
cell.atom = [["Si", (0, 0, 0)], ["Si", (a/4, a/4, a/4)]]
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pbe"
cell.verbose = 4
cell.build()

kmesh = [4, 4, 4]
kpts = cell.make_kpts(kmesh)
mf = dft.KRKS(cell, kpts=kpts).density_fit()
mf.xc = "pbe"
mf.chkfile = "si_scf.chk"
mf.kernel()
assert mf.converged

# k-path: L - G - X - U|K - G  (cartesian units of 2pi/a)
pts = {"L": [.5, .5, .5], "G": [0, 0, 0], "X": [0, 1, 0],
       "U": [.25, 1, .25], "K": [.75, .75, 0]}
path = [("L", "G"), ("G", "X"), ("X", "U"), ("K", "G")]
nseg = 40
kcart, xs, ticks = [], [], {}
x0 = 0.0
for s, (p, q) in enumerate(path):
    P, Q = np.array(pts[p]), np.array(pts[q])
    d = np.linalg.norm(Q - P)
    t = np.linspace(0, 1, nseg, endpoint=(True))
    ks = P[None] + t[:, None]*(Q - P)[None]
    kcart.append(ks)
    xs.append(x0 + t*d)
    ticks[x0] = p
    ticks[x0 + d] = q
    x0 += d
kcart = np.vstack(kcart)
xs = np.concatenate(xs)
# cartesian (2pi/a) -> absolute k (1/Bohr) for pyscf
kabs = kcart * 2*np.pi/(a/0.529177210903)
e_k, _ = mf.get_bands(kabs, kpts=kpts)  # list of arrays (Hartree)
E = np.array([np.sort(e) for e in e_k]) * HARTREE2EV

nocc = cell.nelectron // 2  # 4 valence bands
vbm = E[:, nocc-1].max()
cbm = E[:, nocc].min()
iv, ic = E[:, nocc-1].argmax(), E[:, nocc].argmin()
print(f"VBM at x={xs[iv]:.3f} k={kcart[iv]}  CBM at x={xs[ic]:.3f} k={kcart[ic]}")
print(f"Indirect gap (PBE/gth-dzvp) = {cbm - vbm:.3f} eV")
print(f"Gamma direct gap = {E[nseg, nocc]-E[nseg, nocc-1]:.3f} eV")

np.savez("si_pbe_bands.npz", E=E, xs=xs, kcart=kcart, nocc=nocc)
E -= vbm
fig, ax = plt.subplots(figsize=(5, 6))
for b in range(nocc + 4):
    ax.plot(xs, E[:, b], color="C0" if b < nocc else "C3", lw=1.5)
for x in ticks:
    ax.axvline(x, color="gray", lw=0.6)
ax.set_xticks(list(ticks.keys()))
ax.set_xticklabels([r"$\Gamma$" if v == "G" else v for v in ticks.values()])
ax.axhline(0, color="k", ls="--", lw=0.6)
ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(-13, 8)
ax.set_ylabel("E - E$_{VBM}$ (eV)")
ax.set_title(f"Si band structure (PBE, gth-dzvp)\nindirect gap = {cbm - vbm:.2f} eV")
fig.tight_layout()
fig.savefig("si_bands.png", dpi=200)
