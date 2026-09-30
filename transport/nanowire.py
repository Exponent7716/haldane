#!/usr/bin/env python3
"""Quantum transport in a 1D nanowire / quantum dot with Kwant.

Tight-binding chain  H = sum_i eps_i |i><i| - t sum_i (|i><i+1| + h.c.)
attached to two semi-infinite leads (Landauer-Buttiker, T = Tr[t^+ t]).

Subcommands
  dot       double-barrier quantum dot: T(E) with resonant tunnelling peaks
  disorder  Anderson disorder: <ln T> vs L, localization length
  wire      multi-channel ribbon (width W): conductance staircase G/(2e^2/h)
  check     sanity tests against analytic results (no plots)

Units: hopping t = 1, lattice constant a = 1, G in units of e^2/h per spin.
"""
import argparse
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="MUMPS")
import kwant  # noqa: E402

T_HOP = 1.0


def build_chain(L, onsite, t=T_HOP):
    """1D chain of L sites; onsite is a float or array/callable of site index."""
    lat = kwant.lattice.chain(norbs=1)
    syst = kwant.Builder()
    if callable(onsite):
        syst[(lat(i) for i in range(L))] = lambda s: onsite(s.pos[0])
    else:
        eps = np.broadcast_to(np.asarray(onsite, float), (L,))
        syst[(lat(i) for i in range(L))] = lambda s: eps[s.tag[0]]
    syst[lat.neighbors()] = -t
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = 0.0
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    return syst.finalized()


def transmission(fsyst, energies):
    out = []
    for E in energies:
        out.append(kwant.smatrix(fsyst, E).transmission(1, 0))
    return np.array(out)


# ---------------------------------------------------------------- dot
def barrier_profile(L, nb, wb, V0):
    """Two barriers of width wb and height V0 enclosing a well of nb sites."""
    eps = np.zeros(L)
    a = (L - (2 * wb + nb)) // 2
    eps[a:a + wb] = V0
    eps[a + wb + nb:a + 2 * wb + nb] = V0
    return eps


def run_dot(args):
    import matplotlib.pyplot as plt
    L = args.well + 2 * args.bw + 20
    eps = barrier_profile(L, args.well, args.bw, args.V0)
    fsyst = build_chain(L, eps)
    E = np.linspace(-2 + 1e-3, 2 - 1e-3, args.npts)
    T = transmission(fsyst, E)
    # resonance energies from the isolated well+barriers region
    print(f"# double-barrier dot: well={args.well} barrier=({args.bw} x {args.V0})")
    peaks = [E[i] for i in range(1, len(E) - 1)
             if T[i] > T[i - 1] and T[i] > T[i + 1] and T[i] > 0.5]
    print("# resonance peaks (T>0.5) at E =", ", ".join(f"{p:.3f}" for p in peaks))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.semilogy(E, T)
    ax.set(xlabel="E / t", ylabel="T(E)", title="Double-barrier resonant tunnelling")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


# ----------------------------------------------------------- disorder
def run_disorder(args):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(args.seed)
    Ls = np.unique(np.geomspace(10, args.Lmax, 10).astype(int))
    mean_lnT = []
    for L in Ls:
        lnT = []
        for _ in range(args.nsamp):
            eps = args.W * (rng.random(L) - 0.5)
            fsyst = build_chain(L, eps)
            lnT.append(np.log(kwant.smatrix(fsyst, args.E).transmission(1, 0)))
        mean_lnT.append(np.mean(lnT))
        print(f"L={L:5d}  <lnT>={mean_lnT[-1]:8.3f}")
    mean_lnT = np.array(mean_lnT)
    slope = np.polyfit(Ls[len(Ls) // 2:], mean_lnT[len(Ls) // 2:], 1)[0]
    xi = -2.0 / slope  # <lnT> = -2L/xi
    # perturbative 1D result: xi = 24 (4t^2 - E^2) / W^2  (Thouless)
    xi_th = 24 * (4 * T_HOP**2 - args.E**2) / args.W**2
    print(f"# localization length xi = {xi:.1f}  (weak-disorder theory: {xi_th:.1f})")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(Ls, mean_lnT, "o-")
    ax.set(xlabel="L", ylabel="<ln T>", title=f"Anderson localization W={args.W}, E={args.E}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


# --------------------------------------------------------------- wire
def run_wire(args):
    import matplotlib.pyplot as plt
    lat = kwant.lattice.square(norbs=1)
    W, L = args.width, args.length
    syst = kwant.Builder()
    syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * T_HOP
    syst[lat.neighbors()] = -T_HOP
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[(lat(0, y) for y in range(W))] = 4 * T_HOP
    lead[lat.neighbors()] = -T_HOP
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    fsyst = syst.finalized()
    E = np.linspace(0.02, 1.8, args.npts)
    G = transmission(fsyst, E)
    print("# conductance steps (G in e^2/h):", sorted(set(np.round(G).astype(int))))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(E, G)
    ax.set(xlabel="E / t", ylabel="G  [e$^2$/h]", title=f"Conductance staircase, W={W}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


# -------------------------------------------------------------- check
def run_check(_args):
    # 1. clean chain: T = 1 inside the band, 0 outside
    f = build_chain(20, 0.0)
    assert np.allclose(transmission(f, [-1.5, 0.0, 1.0]), 1.0), "clean chain T != 1"
    assert transmission(f, [2.5])[0] < 1e-10, "T != 0 outside band"
    # 2. single-site scatterer: T = 1 / (1 + (V/(2 t sin k))^2), E = -2 t cos k
    V = 1.3
    eps = np.zeros(21)
    eps[10] = V
    f = build_chain(21, eps)
    for E in (-1.2, 0.3, 1.5):
        k = np.arccos(-E / (2 * T_HOP))
        exact = 1 / (1 + (V / (2 * T_HOP * np.sin(k))) ** 2)
        got = transmission(f, [E])[0]
        assert abs(got - exact) < 1e-9, (E, got, exact)
    # 3. symmetric double barrier hits T = 1 on resonance somewhere
    f = build_chain(60, barrier_profile(60, 8, 2, 3.0))
    from scipy.optimize import minimize_scalar
    E = np.linspace(-1.9, 1.9, 2000)
    E0 = E[np.argmax(transmission(f, E))]
    r = minimize_scalar(lambda e: -transmission(f, [e])[0], bounds=(E0 - 2e-3, E0 + 2e-3),
                        method="bounded", options={"xatol": 1e-12})
    assert -r.fun > 0.999999, "no perfect resonance"
    # 4. unitarity: T + R = 1
    sm = kwant.smatrix(f, 0.4)
    assert abs(sm.transmission(1, 0) + sm.transmission(0, 0) - 1) < 1e-10
    print("all checks passed")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dot")
    d.add_argument("--well", type=int, default=10, help="well width (sites)")
    d.add_argument("--bw", type=int, default=2, help="barrier width (sites)")
    d.add_argument("--V0", type=float, default=2.0, help="barrier height (t)")
    d.add_argument("--npts", type=int, default=1500)
    d.add_argument("--out", default="dot.png")
    d.set_defaults(fn=run_dot)

    a = sub.add_parser("disorder")
    a.add_argument("--W", type=float, default=1.0, help="disorder strength")
    a.add_argument("--E", type=float, default=0.0)
    a.add_argument("--Lmax", type=int, default=2000)
    a.add_argument("--nsamp", type=int, default=100)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="disorder.png")
    a.set_defaults(fn=run_disorder)

    w = sub.add_parser("wire")
    w.add_argument("--width", type=int, default=5)
    w.add_argument("--length", type=int, default=30)
    w.add_argument("--npts", type=int, default=300)
    w.add_argument("--out", default="wire.png")
    w.set_defaults(fn=run_wire)

    c = sub.add_parser("check")
    c.set_defaults(fn=run_check)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
