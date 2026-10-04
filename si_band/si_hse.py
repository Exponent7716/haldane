"""Si band structure with HSE06 (PySCF periodic DFT, gth-dzvp, 4x4x4 SCF)."""
import sys, time, numpy as np
from pyscf.pbc import gto, dft
from pyscf.data.nist import HARTREE2EV

a = 5.431
B = a / 0.529177210903
cell = gto.Cell()
cell.unit = "A"
cell.a = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
cell.atom = [["Si", (0, 0, 0)], ["Si", (a/4, a/4, a/4)]]
cell.basis = "gth-dzvp"; cell.pseudo = "gth-pbe"; cell.verbose = 3
cell.build()

nk = int(sys.argv[1]) if len(sys.argv) > 1 else 4
kpts = cell.make_kpts([nk]*3)
t = time.time()
mf = dft.KRKS(cell, kpts=kpts).density_fit()
mf.xc = "hse06"
mf.chkfile = "si_hse.chk"
mf.kernel()
print(f"HSE06 SCF converged={mf.converged} E={mf.e_tot:.6f} t={time.time()-t:.0f}s", flush=True)

pts = {"L": [.5, .5, .5], "G": [0, 0, 0], "X": [0, 1, 0], "U": [.25, 1, .25], "K": [.75, .75, 0]}
path = [("L", "G"), ("G", "X"), ("X", "U"), ("K", "G")]
nseg = 20
kcart, xs, x0 = [], [], 0.0
for p, q in path:
    P, Q = np.array(pts[p]), np.array(pts[q])
    d = np.linalg.norm(Q - P); s = np.linspace(0, 1, nseg)
    kcart.append(P + s[:, None]*(Q - P)); xs.append(x0 + s*d)
    x0 += d
kcart = np.vstack(kcart); xs = np.concatenate(xs)
e_k, _ = mf.get_bands(kcart * 2*np.pi/B, kpts=kpts)
E = np.array([np.sort(e) for e in e_k]) * HARTREE2EV
no = cell.nelectron // 2
np.savez("si_hse_bands.npz", E=E, xs=xs, kcart=kcart, nocc=no)
print(f"HSE06 indirect gap = {E[:, no].min() - E[:, no-1].max():.3f} eV  "
      f"Gamma direct = {E[nseg, no] - E[nseg, no-1]:.3f} eV", flush=True)
