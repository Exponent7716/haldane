"""Si HSE06 SCF on a 4x4x4 mesh (PySCF periodic DFT, gth-dzvp); eigenvalues saved in si_hse.chk.

Note: mf.get_bands() at k-points incommensurate with the SCF mesh makes PySCF build a huge
supercell for the range-separated exchange (43 GiB) and fail; plot_compare.py therefore
interpolates the HSE06-PBE correction from the mesh instead.
"""
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
