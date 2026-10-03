"""DMFT + local vertex (ED) + lattice BSE: magnetic / charge susceptibility chi(q, nu=0).

usage: python3 run_susceptibility.py [U] [beta ...]

Prints chi_m(Q=(pi,pi)), chi_m(0) and the q-average of chi_m per temperature
(the latter should reproduce the impurity value chi_m_loc up to the bath-fit error);
1/chi_m(Q) -> 0 marks the DMFT Neel temperature.
"""
import sys
import time

import numpy as np

import bse
import dmft

NB = 2          # bath sites of the ED solver
NBOX = 12       # fermionic Matsubara box: 2*NBOX frequencies
L = 32          # k grid L x L (L even); Q = (L/2, L/2)


def run(U, beta, nb=NB, nbox=NBOX, L=L, verbose=False):
    t0 = time.time()
    sol = dmft.solve(U, beta, nb=nb, L=L)
    t1 = time.time()
    vertex = bse.local_vertex(sol.imp, 0, nbox)
    t2 = time.time()
    chi = bse.lattice_chi(vertex, sol)
    loc = bse.local_physical(vertex, sol)
    if verbose:
        print(f"    DMFT {t1 - t0:.1f}s ({sol.iterations} it, converged={sol.converged}), "
              f"vertex {t2 - t1:.1f}s")
    return sol, vertex, chi, loc


if __name__ == "__main__":
    U = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
    betas = [float(b) for b in sys.argv[2:]] or [2.0, 3.0, 4.0, 5.0, 6.0]
    print(f"2D square lattice, t=1, U={U}, nb={NB}, box={2 * NBOX}, L={L}")
    print(f"{'beta':>5} {'T':>6} {'docc':>7} {'chi_m(0)':>10} {'chi_m(Q)':>10} {'1/chi_m(Q)':>11}"
          f" {'chi_c(Q)':>10} {'<chi_m>_q':>10} {'chi_m_loc':>10}")
    for beta in betas:
        sol, vertex, chi, loc = run(U, beta, verbose=True)
        q0, qpi = (0, 0), (L // 2, L // 2)
        print(f"{beta:5.2f} {1 / beta:6.3f} {sol.imp.double_occ:7.4f} {chi['m'][q0]:10.4f} "
              f"{chi['m'][qpi]:10.4f} {1 / chi['m'][qpi]:11.4f} {chi['c'][qpi]:10.4f} "
              f"{chi['m'].mean():10.4f} {loc['m']:10.4f}", flush=True)
