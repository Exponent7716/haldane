"""G0W0@PBE for Si at Gamma, X, L (PySCF KRGWAC, gth-dzvp). Usage: python si_gw.py <nk>"""
import sys, time, numpy as np
from pyscf.pbc import gto, dft, gw, df
from pyscf.data.nist import HARTREE2EV as EV

nk = int(sys.argv[1]) if len(sys.argv) > 1 else 2
a = 5.431
B = a / 0.529177210903
cell = gto.Cell()
cell.unit = "A"
cell.a = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
cell.atom = [["Si", (0, 0, 0)], ["Si", (a/4, a/4, a/4)]]
cell.basis = "gth-dzvp"; cell.pseudo = "gth-pbe"; cell.verbose = 0
cell.max_memory = 12000
cell.build()

t0 = time.time()
kpts = cell.make_kpts([nk]*3)
mf = dft.KRKS(cell, kpts=kpts).density_fit(); mf.xc = "pbe"; mf.kernel()
print(f"nk={nk} PBE SCF converged={mf.converged} t={time.time()-t0:.0f}s", flush=True)

kc = kpts * B / (2*np.pi)  # cartesian, 2pi/a
want = {"G": [0, 0, 0], "X": [0, 1, 0], "L": [.5, .5, .5]}
idx = {}
for name, k in want.items():
    # match modulo reciprocal lattice vectors (bcc, units 2pi/a)
    for i, kk in enumerate(kc):
        d = kk - np.array(k)
        r = np.linalg.solve(np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]).T, d)
        if np.allclose(r, np.round(r), atol=1e-6):
            idx[name] = i; break
print("k indices:", idx, flush=True)

nocc = cell.nelectron // 2
# PBE SCF only stores J-type (diagonal k-pair) 3c integrals; GW exchange needs all k-pairs
gdf = df.GDF(cell, kpts); gdf.build(j_only=False)
mf.with_df = gdf
g = gw.krgw_ac.KRGWAC(mf)
g.fc = True
g.kernel(orbs=[nocc-1, nocc], kptlist=sorted(idx.values()))
Eqp = np.array(g.mo_energy)  # Hartree, QP energies for evaluated orbs
Epbe = np.array(mf.mo_energy)
print("t=%.0fs" % (time.time()-t0))
res = {}
for n, i in idx.items():
    res[n] = (Epbe[i][nocc-1]*EV, Epbe[i][nocc]*EV, Eqp[i][nocc-1]*EV, Eqp[i][nocc]*EV)
    print(f"{n}: PBE v={res[n][0]:.3f} c={res[n][1]:.3f} | GW v={res[n][2]:.3f} c={res[n][3]:.3f} eV")
vp = max(r[0] for r in res.values()); vg = max(r[2] for r in res.values())
gp = min(r[1] for r in res.values()) - vp
gg = min(r[3] for r in res.values()) - vg
print(f"RESULT nk={nk} gap(mesh) PBE={gp:.3f} G0W0={gg:.3f} eV; direct Gamma PBE={res['G'][1]-res['G'][0]:.3f} G0W0={res['G'][3]-res['G'][2]:.3f}", flush=True)
