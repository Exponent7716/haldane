"""ED-DMFT for the half-filled Hubbard model on the 2D square lattice.

    H = -t sum_<ij>,s c+_is c_js + U sum_i (n_i,up - 1/2)(n_i,dn - 1/2)

Particle-hole symmetric (mu = U/2 absorbed).  The bath is parametrized
symmetrically, Delta(iw) = sum_j V_j^2 [1/(iw - e_j) + 1/(iw + e_j)] (+ V_0^2/iw
for an odd number of bath sites), and fitted to the cavity function on the
Matsubara axis.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from anderson import AndersonImpurity


def kgrid_energies(L, t=1.0):
    """Dispersion on an L x L grid shifted off the zone origin (L even)."""
    k = 2 * np.pi * (np.arange(L) + 0.5) / L
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return -2 * t * (np.cos(kx) + np.cos(ky))


def unpack(x, nb):
    npair = nb // 2
    e = np.concatenate([x[:npair], -x[:npair]])
    V = np.concatenate([x[npair:2 * npair], x[npair:2 * npair]])
    if nb % 2:
        e = np.append(e, 0.0)
        V = np.append(V, x[2 * npair])
    return e, V


def hybridization(iw, e, V):
    iw = np.asarray(iw, complex)
    return (V[None, :] ** 2 / (iw.reshape(-1, 1) - e[None, :])).sum(1).reshape(iw.shape)


@dataclass
class DMFTResult:
    imp: AndersonImpurity
    e: np.ndarray
    V: np.ndarray
    eps_k: np.ndarray
    iterations: int
    converged: bool

    def delta(self, iw):
        return hybridization(iw, self.e, self.V)

    def sigma(self, iw):
        """Self-energy relative to the Hartree shift U/2 (-> 0 at large |w|)."""
        iw = np.asarray(iw, complex)
        return iw - self.delta(iw) - 1.0 / self.imp.green(iw)


def solve(U, beta, nb=2, t=1.0, L=64, n_fit=None, max_iter=200, tol=1e-6, mix=0.5,
          x0=None, verbose=False):
    eps_k = kgrid_energies(L, t)
    n_fit = n_fit or max(32, int(6.0 * beta))
    iw = 1j * (2 * np.arange(n_fit) + 1) * np.pi / beta
    npar = nb // 2 * 2 + nb % 2
    if x0 is None:
        npair = nb // 2
        x0 = np.concatenate([np.linspace(0.5, 2.0, npair) * t, np.full(npair, 0.8 * t)])
        if nb % 2:
            x0 = np.append(x0, 0.5 * t)
    x = np.array(x0, float)
    assert len(x) == npar
    delta_old = hybridization(iw, *unpack(x, nb))
    weight = 1.0 / np.sqrt(iw.imag)
    converged = False
    for it in range(1, max_iter + 1):
        e, V = unpack(x, nb)
        imp = AndersonImpurity(U, e, V, beta)
        res = DMFTResult(imp, e, V, eps_k, it, False)
        sig = res.sigma(iw)
        g_loc = (1.0 / (iw[:, None, None] - sig[:, None, None] - eps_k[None])).mean(axis=(1, 2))
        delta_new = iw - sig - 1.0 / g_loc
        target = mix * delta_new + (1 - mix) * delta_old
        fit = least_squares(
            lambda y: (hybridization(iw, *unpack(y, nb)) - target).imag * weight,
            x, bounds=([0.02] * (nb // 2) + [0.0] * (npar - nb // 2), [20.0] * npar))
        err = np.abs(fit.x - x).max()
        x = fit.x
        delta_old = hybridization(iw, *unpack(x, nb))
        if verbose:
            print(f"  iter {it:3d}  max|dx| = {err:.2e}  x = {np.round(x, 5)}")
        if err < tol:
            converged = True
            break
    e, V = unpack(x, nb)
    imp = AndersonImpurity(U, e, V, beta)
    return DMFTResult(imp, e, V, eps_k, it, converged)
