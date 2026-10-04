"""Convergence test of Si PBE gap w.r.t. basis and SCF k-mesh."""
import sys, time, numpy as np
from pyscf.pbc import gto, dft
from pyscf.data.nist import HARTREE2EV

a = 5.431
B = a / 0.529177210903
def build(basis):
    c = gto.Cell()
    c.unit = "A"
    c.a = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    c.atom = [["Si", (0, 0, 0)], ["Si", (a/4, a/4, a/4)]]
    c.basis = basis; c.pseudo = "gth-pbe"; c.verbose = 0
    c.build(); return c

kc = np.array([[0, 0, 0], [0, 1, 0], [0, .85, 0]]) * 2*np.pi/B  # G, X, near-CBM
for basis, nk in [("gth-dzvp", 3), ("gth-dzvp", 4), ("gth-dzvp", 6),
                  ("gth-tzvp", 3), ("gth-tzvp", 4), ("gth-tzvp", 6)]:
    t = time.time()
    cell = build(basis)
    kpts = cell.make_kpts([nk]*3)
    mf = dft.KRKS(cell, kpts=kpts).density_fit(); mf.xc = "pbe"; mf.kernel()
    e, _ = mf.get_bands(kc, kpts=kpts)
    E = np.array([np.sort(x) for x in e]) * HARTREE2EV
    no = cell.nelectron // 2
    vbm = E[0, no-1]
    print(f"{basis} k={nk}^3 conv={mf.converged} Egap_ind={E[2,no]-vbm:.3f} "
          f"Egap_X={E[1,no]-vbm:.3f} Egap_G={E[0,no]-vbm:.3f} E0={mf.e_tot:.6f} "
          f"time={time.time()-t:.0f}s", flush=True)
