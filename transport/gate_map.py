#!/usr/bin/env python3
"""Gate-voltage / bias maps of a 1D quantum dot (Coulomb-diamond-like).

Non-interacting Landauer picture: a gate shifts the on-site energy of the
well by -alpha*Vg; a bias Vsd opens the window [mu_R, mu_L] = [-Vsd/2, +Vsd/2]:

    I(Vg, Vsd) = int T(E; Vg) [f(E - Vsd/2) - f(E + Vsd/2)] dE       (e = h = 1, so G is in e^2/h)
    G = dI/dVsd

Because there is no charging energy, the "diamonds" are single-particle
resonance diamonds; edges have slopes +-2/alpha for symmetric bias.
Results are cached in gate_map.npz (T(E, Vg) is the expensive part).
"""
import argparse
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="MUMPS")
import kwant  # noqa: E402

from nanowire import barrier_profile, build_chain  # noqa: E402


def _row(args):
    Vg, Es, well, bw, V0, alpha = args
    L = well + 2 * bw + 20
    base = barrier_profile(L, well, bw, V0)
    a = (L - (2 * bw + well)) // 2
    inside = slice(a + bw, a + bw + well)
    eps = base.copy()
    eps[inside] -= alpha * Vg
    fsyst = build_chain(L, eps)
    return [kwant.smatrix(fsyst, E).transmission(1, 0) for E in Es]


def compute_T(Vgs, Es, well, bw, V0, alpha=1.0):
    from multiprocessing import Pool
    with Pool() as pool:
        rows = pool.map(_row, [(Vg, Es, well, bw, V0, alpha) for Vg in Vgs])
    return np.array(rows)


def fermi(x, kT):
    return 0.5 * (1 - np.tanh(x / (2 * kT)))


def current(T, Es, Vsds, kT):
    dE = Es[1] - Es[0]
    f_l = fermi(Es[None, :] - Vsds[:, None] / 2, kT)
    f_r = fermi(Es[None, :] + Vsds[:, None] / 2, kT)
    win = f_l - f_r  # (nV, nE)
    return T @ (win.T) * dE  # (nVg, nV)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--well", type=int, default=6)
    p.add_argument("--bw", type=int, default=2)
    p.add_argument("--V0", type=float, default=2.0)
    p.add_argument("--kT", type=float, default=0.01)
    p.add_argument("--nVg", type=int, default=121)
    p.add_argument("--nE", type=int, default=1201)
    p.add_argument("--nV", type=int, default=241)
    p.add_argument("--Vg", type=float, nargs=2, default=(-1.5, 1.5))
    p.add_argument("--Vsd", type=float, default=1.5)
    p.add_argument("--cache", default="gate_map.npz")
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()

    Vgs = np.linspace(*a.Vg, a.nVg)
    Es = np.linspace(-1.2, 1.2, a.nE)
    if os.path.exists(a.cache) and not a.recompute:
        T = np.load(a.cache)["T"]
    else:
        T = compute_T(Vgs, Es, a.well, a.bw, a.V0)
        np.savez_compressed(a.cache, T=T, Vgs=Vgs, Es=Es)

    Vsds = np.linspace(-a.Vsd, a.Vsd, a.nV)
    I = current(T, Es, Vsds, a.kT)
    G = np.gradient(I, Vsds, axis=1)
    np.savez_compressed(a.cache.replace(".npz", "_out.npz"), Vgs=Vgs, Vsds=Vsds, I=I, G=G)
    print("T shape", T.shape, " max G [e^2/h units of 1/2pi]:", G.max())


if __name__ == "__main__":
    main()
