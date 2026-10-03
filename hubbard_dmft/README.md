# Hubbard model: DMFT + local vertex (ED) + lattice BSE

Half-filled 2D Hubbard model on the square lattice,

    H = -t sum_<ij>,s c+_is c_js + U sum_i (n_i,up - 1/2)(n_i,dn - 1/2).

Pipeline (the "DMFT + BSE" route, i.e. the local-vertex starting point of DΓA / TRILEX):

1. **DMFT** (`dmft.py`): finite-temperature ED impurity solver, particle-hole symmetric
   bath (`nb` bath sites), bath fitted to the cavity function on the Matsubara axis,
   self-consistency with the lattice Green function on an `L x L` k grid.
2. **Local vertex** (`anderson.py`, `bse.py`): the impurity two-particle Green function
   `chi_{ss'}(w, w', nu)` is computed exactly from the Lehmann representation
   (all 24 time orderings; the 5-point divided difference of `exp(-beta x)` is evaluated
   with the confluent formula for degenerate levels / `nu = 0`, and with a matrix
   exponential for nearly degenerate ones). The irreducible vertex follows from the
   impurity BSE `chi^-1 = D^-1 + Gamma/beta^2` in the charge and magnetic channels.
3. **Lattice BSE** (`bse.py`): the local `Gamma` and the lattice bubble `D(q, w)`
   (computed with FFTs over k) give `chi_m(q)`, `chi_c(q)` at any `q` and bosonic `nu`.

The finite Matsubara box is handled analytically: outside the box `Gamma` is replaced by
its asymptotic value `-U` (magnetic) / `+U` (charge), which gives the box-truncation
correction and the full frequency sum in closed form (see the docstring of `bse.py`).

## Usage

```bash
pip install numpy scipy
cd hubbard_dmft
python3 run_susceptibility.py 4.0 2 3 4 5 6      # U = 4, beta = 2..6
python3 -m unittest -v test_hubbard_dmft          # checks
```

Output columns: `chi_m(0)`, `chi_m(Q)` with `Q = (pi, pi)`, `1/chi_m(Q)`, `chi_c(Q)`,
the q-average of `chi_m` and the impurity value (these agree up to the bath-fit error).
`1/chi_m(Q) -> 0` marks the Néel temperature of the paramagnetic DMFT solution; below it
`chi_m(Q)` turns negative (the paramagnetic solution is unstable).

Normalization: `chi_ch(q, nu) = (1/beta^2) sum_{w,w'} chi_ch(w, w'; q, nu)` with
`chi_m = chi_upup - chi_updown`, i.e. `chi_m = (1/2) int dtau <(n_up - n_dn)(tau) (n_up - n_dn)(0)>`.

## Checks implemented

* `U = 0`: `chi_upup = -beta G G delta_{ww'}`, `chi_updown = 0` to machine precision.
* Local sum rule: `(1/beta^2) sum chi` from the 4-point function vs. the static ED
  susceptibility (agreement ~0.03% for `N = 10`, ~0.001% for `N = 20` in the magnetic channel).
* FFT bubble vs. brute-force k sum.

## Results (U = 4t, nb = 2, 24 Matsubara frequencies, 32 x 32 k grid)

| beta | T/t | chi_m(Q) | 1/chi_m(Q) |
|-----:|----:|---------:|-----------:|
| 2 | 0.50 | 1.15 | 0.87 |
| 3 | 0.33 | 3.00 | 0.33 |
| 4 | 0.25 | 15.6 | 0.064 |
| 5 | 0.20 | < 0 | (unstable) |

1/chi_m(Q) vanishes at T_N ≈ 0.24 t (DMFT mean-field Néel temperature; with only 2 bath
sites this is a rough number, not converged in `nb`).

## Limitations

* Only the **local** vertex is used (no non-local DΓA corrections), so `chi(q)` is
  DMFT-level: it has a finite-temperature Néel transition in 2D (violating Mermin–Wagner).
* Self-consistency check not fully closed: the q-average of `chi_m` exceeds the impurity
  value by ~5.6% at beta = 2 (charge channel: 0.2%). This is **not** the bath-fit error:
  going from `nb = 2` to `nb = 3` reduces |G_loc - G_imp|/|G_imp| from 3.6e-3 to 2.4e-4
  but leaves the discrepancy unchanged (5.7% -> 5.6%). Suspects: the constant-`-U`
  asymptotics outside the box and the neglected omega = +-omega' ridges of `Gamma`. Open.
* Cost: the 4-point Lehmann sum scales as `(#bath states)^3 x (2N)^2`; `nb = 2`, `N = 12`
  takes ~45 s per temperature, `nb = 3` is several times slower.
* Particle-hole symmetric (half filling) only, `nu = 0` tested most (other `nu_l` are supported
  by `chi2`/`local_vertex`).
