"""Checks for the ED / DMFT / BSE chain.  Run:  python3 -m unittest -v test_hubbard_dmft"""
import unittest

import numpy as np

import bse
import dmft
from anderson import AndersonImpurity

BETA = 4.0


class NonInteracting(unittest.TestCase):
    def test_vertex_vanishes_at_U0(self):
        imp = AndersonImpurity(0.0, [0.7, -0.7], [0.5, 0.5], BETA)
        ms = np.arange(-5, 5)
        for l in (0, 1):
            g, gn = imp.green(imp.matsubara(ms)), imp.green(imp.matsubara(ms + l))
            uu, ud = imp.chi2(0, 0, l, ms), imp.chi2(0, 1, l, ms)
            self.assertLess(abs(uu - np.diag(-BETA * g * gn)).max(), 1e-12)
            self.assertLess(abs(ud).max(), 1e-12)


class SumRule(unittest.TestCase):
    """(1/beta^2) sum chi from the 4-point function must reproduce the static ED susceptibility."""

    def test_local_sum_rule(self):
        imp = AndersonImpurity(4.0, [0.7, -0.7], [0.5, 0.5], BETA)

        class Sol:
            pass
        sol = Sol()
        sol.imp = imp
        vertex = bse.local_vertex(imp, 0, 8)
        phys = bse.local_physical(vertex, sol)
        for name, op in imp.spin_charge_ops().items():
            direct = 0.5 * imp.static_susceptibility(op, op)
            self.assertAlmostEqual(phys[name] / direct, 1.0, delta=0.01, msg=name)


class LatticeBubble(unittest.TestCase):
    def test_fft_bubble_matches_brute_force(self):
        sol = dmft.solve(4.0, BETA, nb=2, L=8)
        d_box, s_out = bse._bubbles(sol, 0, 4)
        ms = np.arange(-4, 4)
        iw = sol.imp.matsubara(ms)
        G = 1.0 / (iw[:, None, None] - sol.sigma(iw)[:, None, None] - sol.eps_k[None])
        for qx, qy in [(0, 0), (4, 4), (1, 3)]:
            ref = -BETA * (G * np.roll(G, (-qx, -qy), axis=(1, 2))).mean(axis=(1, 2))
            self.assertLess(abs(d_box[:, qx, qy] - ref).max(), 1e-12)


class DMFT(unittest.TestCase):
    def test_converges_and_is_particle_hole_symmetric(self):
        sol = dmft.solve(4.0, BETA, nb=2, L=16)
        self.assertTrue(sol.converged)
        g = sol.imp.green(sol.imp.matsubara(np.arange(8)))
        self.assertLess(abs(g.real).max(), 1e-10)
        self.assertGreater(sol.imp.double_occ, 0.05)
        self.assertLess(sol.imp.double_occ, 0.25)


if __name__ == "__main__":
    unittest.main()
