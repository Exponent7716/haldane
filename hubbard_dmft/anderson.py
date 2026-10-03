"""Finite-temperature exact diagonalization of a single-impurity Anderson model.

Hamiltonian (particle-hole symmetric form, mu = U/2 absorbed):

    H = U (n_up - 1/2)(n_dn - 1/2)                       (impurity, orbital 0)
      + sum_{k,s} eps_k n_{k s} + V_k (c+_{0 s} c_{k s} + h.c.)

Provides, on the Matsubara axis (w_m = (2m+1) pi / beta):

    * the one-particle Green function   G(iw)
    * the two-particle quantity         chi_{s s'}(w, w', nu)

        chi = (1/beta) int d^4tau  e^{i w t1 - i(w+nu) t2 + i(w'+nu) t3 - i w' t4}
                 < T c_s(t1) c+_s(t2) c_s'(t3) c+_s'(t4) >  -  beta G_s(w) G_s'(w') delta_{nu,0}

  evaluated exactly from the Lehmann representation.  The imaginary-time
  integral over each time-ordered simplex is a divided difference of exp(-beta x)
  on five points; coinciding points (degenerate levels, nu = 0) are handled
  with the confluent formula, nearly coinciding points via a matrix exponential.
"""
from itertools import permutations
from math import factorial

import numpy as np
from scipy import sparse
from scipy.linalg import expm

_NZ_TOL = 1e-12      # matrix elements below this are dropped
_E_TOL = 1e-9        # energies closer than this are one level
_NEAR = 0.05         # |lambda_i - lambda_j| below this (but not equal) -> expm path


def _parity(p):
    inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
    return -1 if inv % 2 else 1


def _expand(end, csr):
    """For every entry of `end` (row indices) list the non-zeros of that row of csr."""
    counts = csr.indptr[end + 1] - csr.indptr[end]
    rep = np.repeat(np.arange(len(end)), counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    pos = np.repeat(csr.indptr[end], counts) + offs
    return rep, csr.indices[pos], csr.data[pos]


class AndersonImpurity:
    def __init__(self, U, eps, V, beta):
        self.U = float(U)
        self.eps = np.atleast_1d(np.asarray(eps, float))
        self.V = np.atleast_1d(np.asarray(V, float))
        self.beta = float(beta)
        self.nb = len(self.eps)
        self._diagonalize()

    # ------------------------------------------------------------------ Fock space
    def _orb(self, spin, site):
        return spin * (self.nb + 1) + site

    def _diagonalize(self):
        nb, n = self.nb, 2 * (self.nb + 1)
        dim = 1 << n
        states = np.arange(dim)
        bit = lambda o: (states >> o) & 1

        c = {}
        for o in range(n):
            m = np.zeros((dim, dim))
            occ = np.nonzero(bit(o))[0]
            below = states[occ] & ((1 << o) - 1)
            sign = 1 - 2 * (np.array([bin(x).count("1") for x in below]) % 2)
            m[states[occ] ^ (1 << o), occ] = sign
            c[o] = m
        num = {o: np.diag(bit(o).astype(float)) for o in range(n)}

        H = np.zeros((dim, dim))
        o0u, o0d = self._orb(0, 0), self._orb(1, 0)
        H += self.U * (num[o0u] - 0.5 * np.eye(dim)) @ (num[o0d] - 0.5 * np.eye(dim))
        for s in (0, 1):
            for k in range(nb):
                ok = self._orb(s, k + 1)
                o0 = self._orb(s, 0)
                H += self.eps[k] * num[ok]
                hop = self.V[k] * c[o0].T @ c[ok]
                H += hop + hop.T

        nup = sum(bit(self._orb(0, i)) for i in range(nb + 1))
        ndn = sum(bit(self._orb(1, i)) for i in range(nb + 1))
        vecs = np.zeros((dim, dim))
        evals = np.zeros(dim)
        col = 0
        for a in range(nb + 2):
            for b in range(nb + 2):
                idx = np.nonzero((nup == a) & (ndn == b))[0]
                if len(idx) == 0:
                    continue
                e, v = np.linalg.eigh(H[np.ix_(idx, idx)])
                vecs[idx, col:col + len(idx)] = v
                evals[col:col + len(idx)] = e
                col += len(idx)

        evals = evals - evals.min()
        # snap (numerically) degenerate levels onto a single value / id
        order = np.argsort(evals)
        new = np.concatenate([[True], np.diff(evals[order]) > _E_TOL])
        cid_sorted = np.cumsum(new) - 1
        self.level_id = np.empty(dim, int)
        self.level_id[order] = cid_sorted
        self.levels = np.array([evals[order][cid_sorted == i].mean() for i in range(cid_sorted[-1] + 1)])
        self.E = self.levels[self.level_id]
        self.w = np.exp(-self.beta * self.E)
        self.Z = self.w.sum()
        # impurity annihilation operators in the eigenbasis
        self.A = [vecs.T @ c[self._orb(s, 0)] @ vecs for s in (0, 1)]
        self._vecs = vecs
        self._n_imp = [vecs.T @ num[self._orb(s, 0)] @ vecs for s in (0, 1)]
        self.double_occ = float(np.sum(self.w * np.diag(self._n_imp[0] @ self._n_imp[1])) / self.Z)

    # ------------------------------------------------------------------ G(iw)
    def green(self, iw):
        """G_sigma(iw) (spin symmetric) for complex frequencies `iw`."""
        iw = np.asarray(iw, complex)
        A = self.A[0]
        n, m = np.nonzero(np.abs(A) > _NZ_TOL)       # c|m> = |n>
        wt = A[n, m] ** 2 * (self.w[n] + self.w[m]) / self.Z
        pole = self.E[m] - self.E[n]
        return (wt[None, :] / (iw.reshape(-1, 1) - pole[None, :])).sum(1).reshape(iw.shape)

    def matsubara(self, m):
        return 1j * (2 * np.asarray(m) + 1) * np.pi / self.beta

    # ------------------------------------------------------------------ static check
    def static_susceptibility(self, op_a, op_b):
        """int_0^beta dtau <A(tau) B(0)>_connected for operators given in the eigenbasis."""
        dE = self.E[None, :] - self.E[:, None]       # E_b - E_a, first index a
        wa, wb = self.w[:, None], self.w[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(np.abs(dE) < 1e-9, self.beta * wa, (wa - wb) / dE)
        val = (op_a * op_b.T * f).sum() / self.Z
        mean_a = (np.diag(op_a) * self.w).sum() / self.Z
        mean_b = (np.diag(op_b) * self.w).sum() / self.Z
        return val - self.beta * mean_a * mean_b

    def spin_charge_ops(self):
        nu, nd = self._n_imp
        return {"m": nu - nd, "c": nu + nd}

    # ------------------------------------------------------------------ 4-point chains
    def _chains(self, ops):
        """Weights W(a,b,c,d) = <a|O1|b><b|O2|c><c|O3|d><d|O4|a| aggregated by level ids."""
        csr = [sparse.csr_matrix(np.where(np.abs(o) > _NZ_TOL, o, 0.0)) for o in ops[:3]]
        last = np.where(np.abs(ops[3]) > _NZ_TOL, ops[3], 0.0)
        c0 = csr[0].tocoo()
        a, b, w = c0.row, c0.col, c0.data
        rep, c, d1 = _expand(b, csr[1])
        a, b, w, c = a[rep], b[rep], w[rep] * d1, c
        rep, d, d2 = _expand(c, csr[2])
        a, b, c, w, d = a[rep], b[rep], c[rep], w[rep] * d2, d
        w = w * last[d, a]
        keep = np.abs(w) > 1e-14
        a, b, c, d, w = a[keep], b[keep], c[keep], d[keep], w[keep]
        K = len(self.levels)
        ids = [self.level_id[x] for x in (a, b, c, d)]
        key = ((ids[0] * K + ids[1]) * K + ids[2]) * K + ids[3]
        ukey, inv = np.unique(key, return_inverse=True)
        W = np.bincount(inv, weights=w)
        keep = np.abs(W) > 1e-13
        ukey, W = ukey[keep], W[keep]
        lv = [ukey // K ** 3, (ukey // K ** 2) % K, (ukey // K) % K, ukey % K]
        return [self.levels[x] for x in lv], W

    # ------------------------------------------------------------------ divided difference
    def _divided_difference(self, Ep, Sp):
        """f[l0..l4] for f(x)=exp(-beta x), l_i = Ep_i - i*pi*Sp_i/beta.

        Ep: (5, T) real, Sp: (5, F) int.  Returns (T, F) complex.
        """
        beta = self.beta
        T, F = Ep.shape[1], Sp.shape[1]
        out = np.empty((T, F), complex)
        pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        coin = np.zeros((T, F), int)
        near = np.zeros((T, F), bool)
        for bit, (i, j) in enumerate(pairs):
            dre = Ep[i][:, None] - Ep[j][:, None]
            dS = (Sp[i] - Sp[j])[None, :]
            same = (np.abs(dre) < 1e-12) & (dS == 0)
            dist = np.hypot(dre, np.pi * dS / beta)
            coin |= same.astype(int) << bit
            near |= (dist < _NEAR) & ~same
        ex = lambda i: np.exp(-beta * Ep[i])[:, None] * (1 - 2 * (Sp[i] % 2))[None, :]
        lam = lambda i, t, f: Ep[i][t] - 1j * np.pi * Sp[i][f] / beta

        # --- nearly degenerate: matrix exponential (Opitz)
        t_idx, f_idx = np.nonzero(near)
        if len(t_idx):
            B = np.zeros((len(t_idx), 5, 5), complex)
            for i in range(5):
                B[:, i, i] = lam(i, t_idx, f_idx)
                if i < 4:
                    B[:, i, i + 1] = 1.0
            out[t_idx, f_idx] = expm(-beta * B)[:, 0, 4]

        # --- well separated (with exact coincidences): Newton recursion, grouped by pattern
        good = ~near
        for code in np.unique(coin[good]):
            sel_t, sel_f = np.nonzero(good & (coin == code))
            label = list(range(5))
            for bit, (i, j) in enumerate(pairs):
                if (code >> bit) & 1:
                    label[j] = min(label[j], label[i])
            vals, expo = {}, {}
            for i in set(label):
                vals[i] = lam(i, sel_t, sel_f)
                expo[i] = ex(i)[sel_t, sel_f]
            seq = tuple(sorted(label))
            cache = {}

            def dd(s):
                if s in cache:
                    return cache[s]
                n = len(s) - 1
                if s[0] == s[-1]:
                    r = (-beta) ** n * expo[s[0]] / factorial(n)
                else:
                    r = (dd(s[1:]) - dd(s[:-1])) / (vals[s[-1]] - vals[s[0]])
                cache[s] = r
                return r

            out[sel_t, sel_f] = dd(seq)
        return out

    # ------------------------------------------------------------------ chi_{s s'}
    def chi2(self, s1, s2, l, ms, chunk=400_000):
        """Generalized susceptibility chi_{s1 s2}(w_m, w_m', nu_l) on the grid ms x ms.

        Fermionic frequency index m: w=(2m+1)pi/beta; bosonic nu=2 pi l/beta.
        Returns an array [m, m'].
        """
        ms = np.asarray(ms)
        M = len(ms)
        mm, mp = np.meshgrid(ms, ms, indexing="ij")
        mm, mp = mm.ravel(), mp.ravel()
        # integer "frequency numerators" K_j of the four phases z_j = i pi K_j / beta
        K = np.array([2 * mm + 1, -(2 * mm + 1 + 2 * l), 2 * mp + 1 + 2 * l, -(2 * mp + 1)])
        c1, c2 = self.A[s1], self.A[s2]
        ops = [c1, c1.T, c2, c2.T]
        res = np.zeros(M * M, complex)
        for p in permutations(range(4)):
            levels, W = self._chains([ops[i] for i in p])
            if len(W) == 0:
                continue
            S = np.cumsum(K[list(p)], axis=0)            # Xi_1..Xi_4 numerators
            Sp = np.vstack([np.zeros(M * M, int), S[0], S[1], S[2], np.zeros(M * M, int)])
            Ep = np.vstack([levels[0], levels[1], levels[2], levels[3], levels[0]])
            step = max(1, chunk // (M * M))
            acc = np.zeros(M * M, complex)
            for t0 in range(0, len(W), step):
                sl = slice(t0, t0 + step)
                acc += (W[sl, None] * self._divided_difference(Ep[:, sl], Sp)).sum(0)
            res += _parity(p) * acc
        chi = (res / (self.beta * self.Z)).reshape(M, M)
        if l == 0:
            g = self.green(self.matsubara(ms))
            chi = chi - self.beta * np.outer(g, g)
        return chi
