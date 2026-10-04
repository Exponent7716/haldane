"""Thermal effects: mirror heating, ULE zero-crossing, reticle and pellicle heating.

Mirror heating
--------------
Each multilayer mirror absorbs P_abs = P_EUV (1 - R) + P_OOB A_OOB (out-of-band
DUV/IR).  With the first illuminator mirror seeing ~ 100+ W at the IF, tens
of W are absorbed per mirror.

* lumped model:   C dT/dt = P_abs - G (T - T_cool),  tau = C / G,
  T_ss = T_cool + P_abs / G
* 1-D through-thickness diffusion (front flux q, back face held at T_cool):
  rho c dT/dt = k d2T/dz2,  -k dT/dz|_0 = q,  T(L) = T_cool
  steady state T(z) = T_cool + q (L - z) / k     (solved by Crank-Nicolson)
* 2-D in-plane map (thin plate, linear through-thickness profile) for the
  thickness-averaged rise theta(x, y):
      -k L lap(theta) + (2 k / L) theta = q(x, y)
  surface rise = 2 theta.

Thermal expansion of ULE / Zerodur
----------------------------------
Near its zero-crossing temperature T_zc the CTE is linear:
    alpha(T) = alpha0 + a (T - T_zc),  a ~ 1.6e-9 K^-2 (ULE, ~1.6 ppb/K^2)
Strain relative to the reference (cooling-water) temperature T_ref:
    eps(T) = alpha0 (T - T_ref) + a/2 [(T - T_zc)^2 - (T_ref - T_zc)^2]
For a linear depth profile T = T_ref + s (1 - z/L) the free surface rises
    h = L [alpha0 s / 2 + a/2 ((T_ref - T_zc) s + s^2 / 3)]
At normal incidence the reflected wavefront error is W = 2 h.  If the mirror
is operated at T_zc the linear term vanishes and only the tiny quadratic
term remains - this is why EUV ULE is tuned (Ti doping) to a zero-crossing
of ~ 20-30 C matched to the operating temperature.  Non-rotationally
symmetric heating (e.g. dipole illumination on pupil-near mirrors) gives
astigmatism (Z5/Z6) rather than pure focus (Z4).

Reticle heating
---------------
P_abs,mask = P_mask [1 - (f R_ML + (1-f) R_abs)] (~30-40 % for a mostly
reflective field).  In-plane expansion at mask position x: dx = eps(dT) x,
which maps to an overlay error at the wafer of dx / M (M = 4; 8 in y for
high NA).

Pellicle heating
----------------
A free-standing ~50 nm membrane in vacuum can only cool by radiation from
both faces.  Absorbed power per area (double pass, mask reflectance R_m):
    q = I (1 - T_p) (1 + T_p R_m)
Radiative equilibrium:  q = 2 eps sigma (T^4 - T_env^4)
    -> T = (q / (2 eps sigma) + T_env^4)^(1/4)
Several W/cm^2 give ~ 600-1000 C for low-emissivity membranes, which is why
emissive (metal-silicide caps, CNT) pellicles are needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve

SIGMA_SB = 5.670374419e-8      # W m^-2 K^-4
ZERO_C_K = 273.15


@dataclass(frozen=True)
class SubstrateMaterial:
    """Thermal properties; CTE alpha(T) = alpha0 + slope (T - T_zc) [1/K]."""

    name: str
    density_kg_m3: float
    heat_capacity_J_kgK: float
    conductivity_W_mK: float
    alpha0_per_K: float = 0.0
    cte_slope_per_K2: float = 0.0
    T_zc_C: float = 22.0

    @property
    def diffusivity(self) -> float:
        return self.conductivity_W_mK / (self.density_kg_m3 * self.heat_capacity_J_kgK)

    def cte(self, T_C):
        return self.alpha0_per_K + self.cte_slope_per_K2 * (np.asarray(T_C, dtype=float) - self.T_zc_C)

    def strain(self, T_C, T_ref_C: float):
        """eps = int_{T_ref}^{T} alpha dT'."""
        T = np.asarray(T_C, dtype=float)
        return (self.alpha0_per_K * (T - T_ref_C)
                + 0.5 * self.cte_slope_per_K2 * ((T - self.T_zc_C) ** 2 - (T_ref_C - self.T_zc_C) ** 2))

    def with_zero_crossing(self, T_zc_C: float) -> "SubstrateMaterial":
        return SubstrateMaterial(self.name, self.density_kg_m3, self.heat_capacity_J_kgK,
                                 self.conductivity_W_mK, self.alpha0_per_K, self.cte_slope_per_K2, T_zc_C)


ULE = SubstrateMaterial("ULE", 2210.0, 767.0, 1.31, 0.0, 1.6e-9, 22.0)
ZERODUR = SubstrateMaterial("Zerodur", 2530.0, 821.0, 1.46, 0.0, 1.0e-9, 20.0)
SILICON = SubstrateMaterial("Si", 2330.0, 700.0, 148.0, 2.6e-6, 0.0, 22.0)
MATERIALS: Dict[str, SubstrateMaterial] = {m.name: m for m in (ULE, ZERODUR, SILICON)}


def absorbed_power(P_inband_W: float, reflectivity: float, P_oob_W: float = 0.0,
                   oob_absorptance: float = 0.3) -> float:
    """P_abs = P_EUV (1 - R) + P_OOB A_OOB."""
    return P_inband_W * (1.0 - reflectivity) + P_oob_W * oob_absorptance


# --- lumped and 1-D models -------------------------------------------------------
@dataclass
class LumpedMirror:
    """Mirror as a single thermal mass coupled to a cooled frame."""

    diameter_m: float = 0.3
    thickness_m: float = 0.05
    material: SubstrateMaterial = ULE
    conductance_W_K: float = 2.0       # radiation + conduction to the frame/coolers
    T_cool_C: float = 22.0

    @property
    def heat_capacity_J_K(self) -> float:
        V = math.pi * (self.diameter_m / 2) ** 2 * self.thickness_m
        return V * self.material.density_kg_m3 * self.material.heat_capacity_J_kgK

    @property
    def time_constant_s(self) -> float:
        return self.heat_capacity_J_K / self.conductance_W_K

    def steady_state_C(self, P_abs_W: float) -> float:
        return self.T_cool_C + P_abs_W / self.conductance_W_K

    def transient_C(self, t_s, P_abs_W: float, T0_C: Optional[float] = None):
        T0 = self.T_cool_C if T0_C is None else T0_C
        Tss = self.steady_state_C(P_abs_W)
        return Tss + (T0 - Tss) * np.exp(-np.asarray(t_s, dtype=float) / self.time_constant_s)


def slab_steady_profile(q_W_m2: float, thickness_m: float, material: SubstrateMaterial,
                        T_back_C: float, n_z: int = 51) -> Tuple[np.ndarray, np.ndarray]:
    """T(z) = T_back + q (L - z) / k, z = 0 at the reflecting surface."""
    z = np.linspace(0, thickness_m, n_z)
    return z, T_back_C + q_W_m2 * (thickness_m - z) / material.conductivity_W_mK


def slab_transient(q_W_m2: float, thickness_m: float, material: SubstrateMaterial,
                   T_back_C: float, t_end_s: float, n_z: int = 51, n_t: int = 400,
                   T0_C: Optional[float] = None, n_smoothing: int = 4
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crank-Nicolson (theta = 1/2) solution of 1-D conduction with surface flux q.

    The first ``n_smoothing`` steps use backward Euler (Rannacher start-up) to
    damp the CN oscillations excited by the sudden switch-on of the flux.
    Returns (t, z, T[t, z]).
    """
    L = thickness_m
    z = np.linspace(0, L, n_z)
    dz = z[1] - z[0]
    dt = t_end_s / n_t
    r = material.diffusivity * dt / dz ** 2
    n = n_z - 1                           # unknowns: nodes 0..n-1 (node n = T_back)
    # discrete operator K (T_new - T_old = r (K T + b)); ghost node gives Neumann flux
    K = np.zeros((3, n))                  # banded storage of K (upper, diag, lower)
    K[1, :] = -2.0
    K[0, 1:] = 1.0
    K[2, :-1] = 1.0
    K[0, 1] = 2.0                         # row 0: 2 (T1 - T0)
    b = np.zeros(n)
    b[0] = 2 * dz * q_W_m2 / material.conductivity_W_mK
    b[-1] += T_back_C

    def apply_K(T):
        out = K[1] * T
        out[:-1] += K[0, 1:] * T[1:]
        out[1:] += K[2, :-1] * T[:-1]
        return out

    T = np.full(n_z, T_back_C if T0_C is None else T0_C, dtype=float)
    T[-1] = T_back_C
    out = np.empty((n_t + 1, n_z))
    out[0] = T
    for i in range(1, n_t + 1):
        theta = 1.0 if i <= n_smoothing else 0.5
        ab = -theta * r * K
        ab[1] += 1.0
        Ti = T[:n]
        rhs = Ti + (1 - theta) * r * apply_K(Ti) + r * b
        T[:n] = solve_banded((1, 1), ab, rhs)
        out[i] = T
    return np.linspace(0, t_end_s, n_t + 1), z, out


def surface_displacement(z_m: np.ndarray, T_C: np.ndarray, material: SubstrateMaterial,
                         T_ref_C: float) -> float:
    """Free surface rise h = int_0^L eps(T(z)) dz  [m]."""
    return float(np.trapezoid(material.strain(T_C, T_ref_C), z_m))


def linear_profile_displacement(surface_rise_K, thickness_m: float, material: SubstrateMaterial,
                                T_ref_C: float):
    """Closed form h for T = T_ref + s (1 - z/L):
    h = L [alpha0 s/2 + a/2 ((T_ref - T_zc) s + s^2/3)]  [m]."""
    s = np.asarray(surface_rise_K, dtype=float)
    return thickness_m * (material.alpha0_per_K * s / 2
                          + 0.5 * material.cte_slope_per_K2 * ((T_ref_C - material.T_zc_C) * s + s ** 2 / 3))


def optimal_zero_crossing_C(T_cool_C: float, surface_rise_K: float) -> float:
    """T_zc that nulls h for a uniform load: (T_ref - T_zc) + s/3 = 0."""
    return T_cool_C + surface_rise_K / 3.0


# --- Zernike decomposition -------------------------------------------------------
ZERNIKE_NAMES = {1: "piston", 2: "tilt x", 3: "tilt y", 4: "defocus", 5: "astig 45",
                 6: "astig 0", 7: "coma y", 8: "coma x", 9: "trefoil y", 10: "trefoil x",
                 11: "spherical"}


def zernike_noll(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Noll-normalised Zernike polynomials Z1..Z11."""
    r, t = rho, theta
    s3, s5, s6, s8 = math.sqrt(3), math.sqrt(5), math.sqrt(6), math.sqrt(8)
    table = {
        1: lambda: np.ones_like(r),
        2: lambda: 2 * r * np.cos(t),
        3: lambda: 2 * r * np.sin(t),
        4: lambda: s3 * (2 * r ** 2 - 1),
        5: lambda: s6 * r ** 2 * np.sin(2 * t),
        6: lambda: s6 * r ** 2 * np.cos(2 * t),
        7: lambda: s8 * (3 * r ** 3 - 2 * r) * np.sin(t),
        8: lambda: s8 * (3 * r ** 3 - 2 * r) * np.cos(t),
        9: lambda: s8 * r ** 3 * np.sin(3 * t),
        10: lambda: s8 * r ** 3 * np.cos(3 * t),
        11: lambda: s5 * (6 * r ** 4 - 6 * r ** 2 + 1),
    }
    return table[j]()


def fit_zernike(wavefront: np.ndarray, n_terms: int = 11) -> Dict[int, float]:
    """Least-squares Noll Zernike coefficients of a square map over its inscribed disc."""
    ny, nx = wavefront.shape
    y, x = np.mgrid[-1:1:ny * 1j, -1:1:nx * 1j]
    rho, th = np.hypot(x, y), np.arctan2(y, x)
    m = rho <= 1.0
    A = np.stack([zernike_noll(j, rho[m], th[m]) for j in range(1, n_terms + 1)], axis=1)
    c, *_ = np.linalg.lstsq(A, wavefront[m], rcond=None)
    return {j + 1: float(v) for j, v in enumerate(c)}


# --- 2-D mirror heating ---------------------------------------------------------
@dataclass
class MirrorHeatingResult:
    surface_rise_K: np.ndarray
    height_m: np.ndarray
    wavefront_nm: np.ndarray          # 2 h (normal incidence), piston/tilt removed
    zernike_nm: Dict[int, float]
    rms_nm: float


def mirror_heating_map(q_W_m2: np.ndarray, aperture_m: float, thickness_m: float = 0.05,
                       material: SubstrateMaterial = ULE, T_cool_C: float = 22.0,
                       incidence_deg: float = 0.0) -> MirrorHeatingResult:
    """Steady 2-D thin-plate heating of a square mirror patch (side ``aperture_m``).

    Solves -k L lap(theta) + (2k/L) theta = q with insulating edges, then
    applies the linear-profile expansion formula with surface rise 2 theta.
    """
    q = np.asarray(q_W_m2, dtype=float)
    ny, nx = q.shape
    dx = aperture_m / (nx - 1)
    k, L = material.conductivity_W_mK, thickness_m
    D = k * L / dx ** 2

    def lap1d(n):
        main = np.full(n, 2.0)
        main[0] = main[-1] = 1.0                      # Neumann edges
        return sparse.diags([-np.ones(n - 1), main, -np.ones(n - 1)], [-1, 0, 1])

    Lop = sparse.kron(sparse.identity(ny), lap1d(nx)) + sparse.kron(lap1d(ny), sparse.identity(nx))
    A = (D * Lop + (2 * k / L) * sparse.identity(nx * ny)).tocsc()
    theta = spsolve(A, q.ravel()).reshape(ny, nx)
    s = 2 * theta
    h = linear_profile_displacement(s, L, material, T_cool_C)
    wf = 2 * h * math.cos(math.radians(incidence_deg)) * 1e9
    zc = fit_zernike(wf)
    y, x = np.mgrid[-1:1:ny * 1j, -1:1:nx * 1j]
    rho, th = np.hypot(x, y), np.arctan2(y, x)
    m = rho <= 1
    resid = wf - sum(zc[j] * zernike_noll(j, rho, th) for j in (1, 2, 3))
    rms = float(np.sqrt(np.mean(resid[m] ** 2)))
    return MirrorHeatingResult(s, h, np.where(m, resid, 0.0), zc, rms)


def illumination_heat_load(shape: str, n: int = 41, total_W: float = 10.0, aperture_m: float = 0.2,
                           sigma_out: float = 0.8, sigma_in: float = 0.5,
                           pole_sigma: float = 0.2, pole_center: float = 0.7) -> np.ndarray:
    """Heat-flux map [W/m^2] on a pupil-near mirror for an illumination mode:
    'conventional', 'annular', 'dipole_x', 'dipole_y', 'uniform'."""
    y, x = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
    r = np.hypot(x, y)
    if shape == "uniform":
        w = (r <= 1).astype(float)
    elif shape == "conventional":
        w = (r <= sigma_out).astype(float)
    elif shape == "annular":
        w = ((r <= sigma_out) & (r >= sigma_in)).astype(float)
    elif shape in ("dipole_x", "dipole_y"):
        cx, cy = (pole_center, 0) if shape == "dipole_x" else (0, pole_center)
        w = ((np.hypot(x - cx, y - cy) <= pole_sigma) | (np.hypot(x + cx, y + cy) <= pole_sigma)).astype(float)
    else:
        raise ValueError(shape)
    dA = (aperture_m / (n - 1)) ** 2
    return w * total_W / (w.sum() * dA)


# --- reticle --------------------------------------------------------------------
def reticle_absorbed_fraction(reflective_fraction: float = 0.9, R_multilayer: float = 0.65,
                              R_absorber: float = 0.02) -> float:
    """A = 1 - (f R_ML + (1 - f) R_abs)."""
    return 1.0 - (reflective_fraction * R_multilayer + (1 - reflective_fraction) * R_absorber)


def reticle_overlay_error_nm(delta_T_K: float, position_at_mask_mm: float, cte_per_K: float,
                             magnification: float = 4.0) -> Tuple[float, float]:
    """(in-plane mask displacement [nm], overlay error at the wafer [nm] = dx / M)."""
    dx_nm = cte_per_K * delta_T_K * position_at_mask_mm * 1e6
    return dx_nm, dx_nm / magnification


@dataclass
class ReticleHeating:
    """Lumped reticle heating with back-side cooling (ESC + gas / clamps).

    q = P_abs / A_illum,  dT = q / h_eff;  in-plane strain eps(T) of the
    low-thermal-expansion (LTEM) substrate.
    """

    power_on_reticle_W: float = 30.0
    absorbed_fraction: float = 0.35
    illuminated_area_m2: float = 0.104 * 0.132     # scanned 104 x 132 mm
    h_eff_W_m2K: float = 200.0
    thickness_m: float = 6.35e-3
    material: SubstrateMaterial = ULE
    T_ref_C: float = 22.0
    magnification: float = 4.0

    @property
    def absorbed_W(self) -> float:
        return self.power_on_reticle_W * self.absorbed_fraction

    @property
    def delta_T_K(self) -> float:
        return self.absorbed_W / self.illuminated_area_m2 / self.h_eff_W_m2K

    @property
    def time_constant_s(self) -> float:
        return self.material.density_kg_m3 * self.material.heat_capacity_J_kgK * self.thickness_m / self.h_eff_W_m2K

    def mask_expansion_nm(self, position_at_mask_mm: float = 52.0) -> float:
        eps = float(self.material.strain(self.T_ref_C + self.delta_T_K, self.T_ref_C))
        return eps * position_at_mask_mm * 1e6

    def overlay_at_wafer_nm(self, position_at_mask_mm: float = 52.0) -> float:
        return self.mask_expansion_nm(position_at_mask_mm) / self.magnification


# --- pellicle -------------------------------------------------------------------
def pellicle_absorbed_flux(intensity_W_m2: float, transmission: float = 0.9,
                           mask_reflectance: float = 0.65) -> float:
    """q = I (1 - T) (1 + T R_mask)  (incident pass + reflected pass)."""
    return intensity_W_m2 * (1 - transmission) * (1 + transmission * mask_reflectance)


def pellicle_equilibrium_temperature(intensity_W_m2: float, emissivity: float = 0.2,
                                     transmission: float = 0.9, mask_reflectance: float = 0.65,
                                     T_env_K: float = 295.0) -> float:
    """Radiative balance of a two-sided membrane: T = (q/(2 eps sigma) + T_env^4)^(1/4)  [K]."""
    q = pellicle_absorbed_flux(intensity_W_m2, transmission, mask_reflectance)
    return (q / (2 * emissivity * SIGMA_SB) + T_env_K ** 4) ** 0.25


def pellicle_time_constant(T_K: float, emissivity: float = 0.2, thickness_m: float = 50e-9,
                           density: float = 2330.0, cp: float = 700.0) -> float:
    """Linearised radiative time constant tau = rho c d / (8 eps sigma T^3)  [s]."""
    return density * cp * thickness_m / (8 * emissivity * SIGMA_SB * T_K ** 3)
