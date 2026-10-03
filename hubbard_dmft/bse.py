"""Local irreducible vertex from the impurity and the lattice Bethe-Salpeter equation.

Conventions (ph channel, Matsubara box of 2N fermionic frequencies m = -N..N-1):

    D(w)          = -beta G(w) G(w+nu)                          (bare bubble, diagonal)
    chi_c, chi_m  = chi_upup +- chi_updown
    chi^-1        = D^-1 + Gamma / beta^2                        (BSE, defines Gamma)

so that the RPA limit Gamma_m = -U gives chi_m = chi0 / (1 - U chi0).
The DMFT lattice susceptibility uses the local Gamma and the lattice bubble

    D(q, w) = -beta/N_k sum_k G(k, w) G(k+q, w+nu),   G(k,w) = 1/(iw - Sigma(w) - eps_k).

Finite frequency box
--------------------
Inverting the *truncated* chi gives an effective vertex that contains ladder
processes through the frequencies outside the box (a Schur complement).  Outside
the box the vertex is taken as its asymptotic constant g = -U (magnetic) or +U
(charge).  With that, the effect of the outside is a rank-one term that can be
removed from the local vertex and re-added for any lattice bubble in closed form:

    Gamma_box = Gamma_eff + g^2 s / (beta^2 (1 + g s / beta^2)),   s = sum_{outside} D
    chi_phys  = (1/beta^2) a / (1 + g a / beta^2),
        a     = 1^T [D_box^-1 + (Gamma_box - g)/beta^2]^-1 1  +  s_q

(1/beta^2 sum_{w,w'} chi_{w w'} = chi_phys is the static or dynamic susceptibility.)
"""
import numpy as np
from scipy.special import polygamma

M_SUM = 1000      # frequencies |m| < M_SUM summed explicitly outside the box; beyond: asymptotics


def _tail(beta):
    """sum_{|m| >= M_SUM} beta/w_m^2 with w_m = (2m+1) pi/beta."""
    return 2.0 * beta * (beta / (2 * np.pi)) ** 2 * polygamma(1, M_SUM + 0.5)


def _outside(n_box):
    """Slice-able boolean mask of m = -M..M-1 outside the box |m+1/2| < n_box."""
    m = np.arange(-M_SUM, M_SUM)
    return (m < -n_box) | (m >= n_box)


def _asymptote(imp):
    return {"m": -imp.U, "c": imp.U}


def _reduced(gamma_eff, d_box, s, g, beta):
    k = g / beta ** 2
    return gamma_eff + g ** 2 * s / (beta ** 2 * (1 + k * s))


def _susceptibility(gamma_box, d_box, s_out, g, beta):
    n = len(d_box)
    one = np.ones(n)
    a_mat = np.diag(1.0 / d_box) + (gamma_box - g) / beta ** 2
    a = one @ np.linalg.solve(a_mat, one) + s_out
    return a / (1 + g * a / beta ** 2) / beta ** 2


def local_vertex(imp, l, N):
    """Impurity generalized susceptibility and (box-corrected) irreducible vertex."""
    beta = imp.beta
    ms = np.arange(-N, N)
    uu = imp.chi2(0, 0, l, ms)
    ud = imp.chi2(0, 1, l, ms)
    big = np.arange(-M_SUM, M_SUM)
    gb = imp.green(imp.matsubara(big))
    gbn = imp.green(imp.matsubara(big + l))
    d_big = -beta * gb * gbn
    d_box = d_big[~_outside(N)]
    s_loc = d_big[_outside(N)].sum() + _tail(beta)
    out = {"ms": ms, "l": l, "D": d_box, "s_out": s_loc, "U": imp.U, "beta": beta}
    for name, chi in (("m", uu - ud), ("c", uu + ud)):
        gamma_eff = beta ** 2 * (np.linalg.inv(chi) - np.diag(1.0 / d_box))
        g = _asymptote(imp)[name]
        out["chi_" + name] = chi
        out["gamma_eff_" + name] = gamma_eff
        out["gamma_" + name] = _reduced(gamma_eff, d_box, s_loc, g, beta)
    return out


def local_physical(vertex, sol):
    """1/beta^2 sum chi for the impurity itself (sum-rule check against ED)."""
    imp = sol.imp
    out = {name: _susceptibility(vertex["gamma_" + name], vertex["D"], vertex["s_out"],
                                 _asymptote(imp)[name], imp.beta) for name in "mc"}
    return {k: v.real if vertex["l"] == 0 else v for k, v in out.items()}


def _bubbles(sol, l, n_box):
    """D(q, w) for all q on the k grid, split into box and outside parts.

    Returns d_box: (2N, L, L) and s_out: (L, L) = sum over the outside (incl. asymptotic tail).
    """
    imp, beta = sol.imp, sol.imp.beta
    L = sol.eps_k.shape[0]
    big = np.arange(-M_SUM, M_SUM)

    def green_k(shift):
        iw = imp.matsubara(big + shift)
        return 1.0 / (iw[:, None, None] - sol.sigma(iw)[:, None, None] - sol.eps_k[None])

    g0, gn = green_k(0), green_k(l)
    rev = lambda a: np.roll(a[:, ::-1, ::-1], 1, axis=(1, 2))      # a(-k)
    conv = np.fft.ifft2(np.fft.fft2(rev(g0)) * np.fft.fft2(gn), axes=(1, 2))
    d = -beta * conv / L ** 2                                         # (F, L, L)
    out = _outside(n_box)
    return d[~out], d[out].sum(axis=0) + _tail(beta)


def lattice_chi(vertex, sol):
    """DMFT susceptibilities chi_m(q), chi_c(q) at bosonic frequency nu_l on the full k grid.

    Returns arrays of shape (L, L); index (qx, qy) is the momentum 2 pi (qx, qy) / L.
    """
    imp, beta = sol.imp, sol.imp.beta
    N, l = len(vertex["ms"]) // 2, vertex["l"]
    d_box, s_out = _bubbles(sol, l, N)
    L = d_box.shape[1]
    res = {}
    for name in "mc":
        g = _asymptote(imp)[name]
        chi = np.empty((L, L), complex)
        for qx in range(L):
            for qy in range(L):
                chi[qx, qy] = _susceptibility(vertex["gamma_" + name], d_box[:, qx, qy],
                                              s_out[qx, qy], g, beta)
        res[name] = chi.real if l == 0 else chi
    return res
