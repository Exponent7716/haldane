"""HSE06 bands on a coarse k-path, reusing the converged SCF in si_hse.chk."""
import time, numpy as np
from pyscf import lib
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

kpts = cell.make_kpts([4]*3)
def fresh_mf():
    # fresh DF object per batch: reusing one after get_bands rebuilds j3c on a huge supercell
    m = dft.KRKS(cell, kpts=kpts).density_fit()
    m.xc = "hse06"
    m.__dict__.update(lib.chkfile.load("si_hse.chk", "scf"))
    return m

pts = {"L": [.5, .5, .5], "G": [0, 0, 0], "X": [0, 1, 0], "U": [.25, 1, .25], "K": [.75, .75, 0]}
path = [("L", "G"), ("G", "X"), ("X", "U"), ("K", "G")]
nseg = 8
kcart, xs, x0 = [], [], 0.0
for p, q in path:
    P, Q = np.array(pts[p]), np.array(pts[q])
    d = np.linalg.norm(Q - P); s = np.linspace(0, 1, nseg)
    kcart.append(P + s[:, None]*(Q - P)); xs.append(x0 + s*d)
    x0 += d
kcart = np.vstack(kcart); xs = np.concatenate(xs)
t = time.time()
E = np.zeros((len(kcart), cell.nao_nr()))
nb = 4
for i0 in range(0, len(kcart), nb):  # batches -> progress + partial saves
    ks = kcart[i0:i0+nb]
    e, _ = fresh_mf().get_bands(ks * 2*np.pi/B, kpts=kpts)
    for j, ej in enumerate(e):
        E[i0+j] = np.sort(ej) * HARTREE2EV
    n = i0 + len(ks)
    np.savez("si_hse_bands.npz", E=E[:n], xs=xs[:n], kcart=kcart[:n], nocc=cell.nelectron//2)
    print(f"k {n}/{len(kcart)} done t={time.time()-t:.0f}s", flush=True)
no = cell.nelectron // 2
print(f"HSE06 indirect gap = {E[:, no].min() - E[:, no-1].max():.3f} eV", flush=True)
