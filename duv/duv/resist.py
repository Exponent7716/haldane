"""Wafer-side physics: thin-film stack optics, chemically amplified resist, process metrics.

This module covers what happens to the light once it reaches the wafer:

* :class:`FilmStack` -- coherent transfer-matrix (TMM) optics of the
  water / top-coat / resist / BARC / silicon stack at 193 nm: reflectivity,
  transmission, absorption and the standing-wave intensity inside the resist.
  :func:`optimize_barc_thickness` finds the BARC thickness that minimises the
  substrate reflectivity seen from the resist.
* :class:`CAResist` -- a lumped chemically amplified ArF resist model:
  Dill-C exposure (photo-acid generation), post-exposure bake (Gaussian acid
  diffusion, acid/quencher neutralisation, catalytic deprotection) and Mack
  development, or a fast threshold development model.
* Metrology and process-window helpers: :func:`measure_cd`,
  :func:`dose_to_size`, :func:`focus_exposure_matrix`, :func:`process_window`,
  :func:`bossung_curves` and the photon shot-noise LER estimate
  :func:`ler_from_shot_noise`.

Conventions follow :mod:`duv.core`: nm, mJ/cm^2, arrays indexed ``[iy, ix]``
and aerial images normalised so that a clear field has intensity 1.  Lateral
arrays may be 1D (``[ix]``, length ``grid.n``) or 2D (``[iy, ix]``); arrays
with an extra leading axis carry the resist depth ``[iz, ...]`` (top first).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import brentq, minimize_scalar

from .core import N_WATER_193, WAVELENGTH_ARF, ExposureResult, Grid, photon_energy

# --- nominal 193 nm optical constants (textbook-level values) -----------------
N_TOPCOAT_193 = 1.55 + 0.0j
N_RESIST_193 = 1.70 + 0.02j
N_BARC_193 = 1.82 + 0.34j
N_SILICON_193 = 0.88 + 2.78j

__all__ = [
    "N_TOPCOAT_193",
    "N_RESIST_193",
    "N_BARC_193",
    "N_SILICON_193",
    "Layer",
    "FilmStack",
    "default_film_stack",
    "optimize_barc_thickness",
    "CAResist",
    "measure_cd",
    "dose_to_size",
    "focus_exposure_matrix",
    "process_window",
    "bossung_curves",
    "ler_from_shot_noise",
]


# =============================================================================
# Thin-film optics
# =============================================================================


@dataclass
class Layer:
    """A homogeneous thin film.

    ``n_complex = n + i k`` with ``k >= 0`` for an absorbing film (fields vary
    as ``exp(+i kz z)`` with ``z`` pointing down into the stack).
    """

    name: str
    thickness_nm: float
    n_complex: complex


def _cos_theta(n: complex, n0_sin0: complex) -> complex:
    """cos(theta) in a medium of index ``n`` from Snell's invariant, forward branch."""
    c = np.sqrt(1.0 - (n0_sin0 / n) ** 2 + 0j)
    nc = n * c
    # forward-propagating / decaying branch: Im(n cos) > 0, or Re(n cos) > 0 if lossless
    if nc.imag < -1e-12 or (abs(nc.imag) <= 1e-12 and nc.real < 0):
        c = -c
    return complex(c)


def _fresnel(pol: str, ni: complex, nj: complex, ci: complex, cj: complex) -> tuple[complex, complex]:
    """Fresnel amplitude (r, t) for the interface i -> j."""
    if pol == "s":
        den = ni * ci + nj * cj
        return (ni * ci - nj * cj) / den, 2 * ni * ci / den
    den = nj * ci + ni * cj
    return (nj * ci - ni * cj) / den, 2 * ni * ci / den


class FilmStack:
    """Planar multilayer on a semi-infinite substrate, illuminated from ``ambient_n``.

    Parameters
    ----------
    layers:
        Films ordered from top (next to the ambient) to bottom (next to the substrate).
    substrate_n:
        Complex index of the semi-infinite substrate.
    ambient_n:
        Index of the incidence medium (immersion water by default).
    wavelength:
        Vacuum wavelength in nm.

    Angles are given in the ambient medium.  ``pol`` is ``'s'`` (TE), ``'p'``
    (TM) or ``'u'`` (unpolarised average of s and p).
    """

    def __init__(
        self,
        layers: Sequence[Layer],
        substrate_n: complex,
        ambient_n: complex = N_WATER_193,
        wavelength: float = WAVELENGTH_ARF,
    ):
        self.layers = list(layers)
        self.substrate_n = complex(substrate_n)
        self.ambient_n = complex(ambient_n)
        self.wavelength = float(wavelength)

    # ---------------------------------------------------------------- helpers
    @classmethod
    def default(cls, barc_thickness_nm: float = 80.0, resist_thickness_nm: float = 90.0) -> "FilmStack":
        """Water / top-coat (30 nm) / resist / BARC / silicon at 193 nm."""
        return default_film_stack(barc_thickness_nm, resist_thickness_nm)

    def layer_index(self, name: str) -> int:
        """Index of the first layer whose name contains ``name`` (case-insensitive)."""
        for i, layer in enumerate(self.layers):
            if name.lower() in layer.name.lower():
                return i
        raise KeyError(f"no layer named like {name!r} in stack {[l.name for l in self.layers]}")

    def with_thickness(self, name: str, thickness_nm: float) -> "FilmStack":
        """Copy of the stack with the thickness of layer ``name`` replaced."""
        new = [replace(l) for l in self.layers]
        i = self.layer_index(name)
        new[i] = replace(new[i], thickness_nm=float(thickness_nm))
        return FilmStack(new, self.substrate_n, self.ambient_n, self.wavelength)

    def below(self, name: str) -> "FilmStack":
        """Sub-stack seen from *inside* layer ``name``: that layer becomes the ambient."""
        i = self.layer_index(name)
        return FilmStack(self.layers[i + 1 :], self.substrate_n, self.layers[i].n_complex, self.wavelength)

    # -------------------------------------------------------------------- TMM
    def _solve(self, angle_rad: float, pol: str) -> dict:
        """Coherent TMM (Fresnel-matrix formulation).

        Returns r, t, the per-medium indices, cosines, kz, and the forward /
        backward amplitudes (v, w) at the top of each film.
        """
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's' or 'p' here")
        k0 = 2 * np.pi / self.wavelength
        ns = [self.ambient_n] + [complex(l.n_complex) for l in self.layers] + [self.substrate_n]
        ds = [np.inf] + [float(l.thickness_nm) for l in self.layers] + [np.inf]
        n0s = self.ambient_n * np.sin(angle_rad)
        cs = [_cos_theta(n, n0s) for n in ns]
        kz = [k0 * n * c for n, c in zip(ns, cs)]
        N = len(ns)
        rt = [_fresnel(pol, ns[i], ns[i + 1], cs[i], cs[i + 1]) for i in range(N - 1)]
        Ms = [None] * N
        Mt = np.array([[1, rt[0][0]], [rt[0][0], 1]], dtype=complex) / rt[0][1]
        for i in range(1, N - 1):
            d = kz[i] * ds[i]
            r, t = rt[i]
            Ms[i] = (
                np.array([[np.exp(-1j * d), 0], [0, np.exp(1j * d)]], dtype=complex)
                @ np.array([[1, r], [r, 1]], dtype=complex)
                / t
            )
            Mt = Mt @ Ms[i]
        r = Mt[1, 0] / Mt[0, 0]
        t = 1 / Mt[0, 0]
        vw = [None] * N
        vw[N - 1] = np.array([t, 0], dtype=complex)
        for i in range(N - 2, 0, -1):
            vw[i] = Ms[i] @ vw[i + 1]
        vw[0] = np.array([1, r], dtype=complex)
        return dict(r=r, t=t, ns=ns, cs=cs, kz=kz, ds=ds, vw=vw, pol=pol)

    def _avg(self, fn, angle_rad, pol):
        if pol == "u":
            return 0.5 * (fn(angle_rad, "s") + fn(angle_rad, "p"))
        return fn(angle_rad, pol)

    def reflection_coefficient(self, angle_rad: float = 0.0, pol: str = "s") -> complex:
        """Complex amplitude reflection coefficient r."""
        return complex(self._solve(angle_rad, pol)["r"])

    def reflectivity(self, angle_rad: float = 0.0, pol: str = "s") -> float:
        """Power reflectance R = |r|^2 seen from the ambient."""
        return float(self._avg(lambda a, p: abs(self._solve(a, p)["r"]) ** 2, angle_rad, pol))

    def transmissivity(self, angle_rad: float = 0.0, pol: str = "s") -> float:
        """Power transmittance into the substrate (normal Poynting flux ratio)."""

        def _t(a, p):
            s = self._solve(a, p)
            n0, c0, nf, cf = s["ns"][0], s["cs"][0], s["ns"][-1], s["cs"][-1]
            if p == "s":
                return abs(s["t"]) ** 2 * (nf * cf).real / (n0 * c0).real
            return abs(s["t"]) ** 2 * (nf * np.conj(cf)).real / (n0 * np.conj(c0)).real

        return float(self._avg(_t, angle_rad, pol))

    def _field_sq(self, sol: dict, i: int, z: np.ndarray) -> np.ndarray:
        """|E|^2 (relative to the incident amplitude) at depth z from the top of medium i."""
        v, w = sol["vw"][i]
        kz = sol["kz"][i]
        ef = v * np.exp(1j * kz * z)
        eb = w * np.exp(-1j * kz * z)
        if sol["pol"] == "s":
            return np.abs(ef + eb) ** 2
        c = sol["cs"][i]
        s = np.sqrt(1 - c**2 + 0j)
        return np.abs((ef - eb) * c) ** 2 + np.abs((-ef - eb) * s) ** 2

    def layer_absorption(
        self, angle_rad: float = 0.0, pol: str = "s", n_samples: int = 2001
    ) -> np.ndarray:
        """Fraction of incident power absorbed in each film (numerical field integral).

        Uses ``A_j = k0 / Re(n0 cos th0) * int Im(n_j^2) |E|^2 dz``, which is
        independent of the R/T calculation and therefore a genuine check of
        energy conservation ``R + T + sum(A) = 1``.
        """

        def _a(a, p):
            sol = self._solve(a, p)
            k0 = 2 * np.pi / self.wavelength
            inc = (sol["ns"][0] * (sol["cs"][0] if p == "s" else np.conj(sol["cs"][0]))).real
            out = []
            for j, layer in enumerate(self.layers, start=1):
                z = np.linspace(0.0, layer.thickness_nm, n_samples)
                e2 = self._field_sq(sol, j, z)
                out.append(k0 * (layer.n_complex**2).imag * np.trapezoid(e2, z) / inc)
            return np.array(out)

        return self._avg(_a, angle_rad, pol)

    def field_intensity(self, layer: str | int, z_nm, angle_rad: float = 0.0, pol: str = "s") -> np.ndarray:
        """|E|^2 relative to the incident field at depth ``z_nm`` below the top of a film."""
        i = self.layer_index(layer) if isinstance(layer, str) else int(layer)
        z = np.asarray(z_nm, dtype=float)
        return self._avg(lambda a, p: self._field_sq(self._solve(a, p), i + 1, z), angle_rad, pol)

    def intensity_in_resist(
        self,
        z_nm,
        angle_rad: float = 0.0,
        pol: str = "s",
        resist: str = "resist",
        normalize: bool = False,
    ) -> np.ndarray:
        """Standing-wave intensity |E(z)|^2 in the resist film.

        ``z_nm`` is measured down from the top of the resist.  With
        ``normalize=True`` the profile is divided by its mean over the film
        thickness (useful as a multiplicative depth factor on an aerial image).
        """
        prof = self.field_intensity(resist, z_nm, angle_rad, pol)
        if normalize:
            t = self.layers[self.layer_index(resist)].thickness_nm
            zz = np.linspace(0, t, 513)
            prof = prof / np.mean(self.field_intensity(resist, zz, angle_rad, pol))
        return prof


def default_film_stack(barc_thickness_nm: float = 80.0, resist_thickness_nm: float = 90.0) -> FilmStack:
    """Nominal immersion ArF stack: water / top-coat 30 nm / resist / BARC / Si.

    With BARC n = 1.82 + 0.34i on Si the *first* reflectivity minimum (~31 nm)
    still leaves ~5 % reflectance into the resist; the second minimum near
    80 nm (cf. the ~78 nm recommended for commercial ArF BARCs) is < 0.1 %,
    so 80 nm is the default.
    """
    return FilmStack(
        [
            Layer("topcoat", 30.0, N_TOPCOAT_193),
            Layer("resist", resist_thickness_nm, N_RESIST_193),
            Layer("barc", barc_thickness_nm, N_BARC_193),
        ],
        substrate_n=N_SILICON_193,
        ambient_n=N_WATER_193,
    )


def optimize_barc_thickness(
    stack: FilmStack | None = None,
    barc: str = "barc",
    seen_from: str = "resist",
    angle_rad: float = 0.0,
    pol: str = "s",
    t_range: tuple[float, float] = (1.0, 150.0),
    first_minimum: bool = False,
) -> tuple[float, float]:
    """BARC thickness minimising the substrate reflectivity seen from inside the resist.

    Scans ``t_range`` in 1 nm steps, takes the global minimum (or the first
    local minimum if ``first_minimum`` is True -- thinner, but for the default
    BARC only ~5 % reflectance) and refines it with a bounded
    scalar minimisation.  ``angle_rad`` is in the stack's ambient medium.

    Returns ``(thickness_nm, reflectivity)``.
    """
    stack = stack or default_film_stack()
    # convert the ambient angle to the angle inside the reference film (Snell)
    n_ref = stack.layers[stack.layer_index(seen_from)].n_complex
    sin_ref = (stack.ambient_n * np.sin(angle_rad) / n_ref).real
    a_ref = float(np.arcsin(np.clip(sin_ref, -1, 1)))

    def refl(t):
        return stack.with_thickness(barc, t).below(seen_from).reflectivity(a_ref, pol)

    ts = np.arange(t_range[0], t_range[1] + 1e-9, 1.0)
    rs = np.array([refl(t) for t in ts])
    idx = int(np.argmin(rs))
    if first_minimum:
        loc = [i for i in range(1, len(rs) - 1) if rs[i] <= rs[i - 1] and rs[i] <= rs[i + 1]]
        if loc:
            idx = loc[0]
    lo, hi = ts[max(idx - 1, 0)], ts[min(idx + 1, len(ts) - 1)]
    res = minimize_scalar(refl, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
    return float(res.x), float(res.fun)


# =============================================================================
# Chemically amplified resist
# =============================================================================


def _lateral_axes(arr: np.ndarray, has_depth: bool) -> tuple[int, ...]:
    nd = arr.ndim - (1 if has_depth else 0)
    if nd not in (1, 2):
        raise ValueError("lateral image must be 1D [ix] or 2D [iy, ix]")
    return tuple(range(arr.ndim - nd, arr.ndim))


def _gaussian_blur_fft(arr: np.ndarray, sigma_nm: float, pixel_nm: float, axes: tuple[int, ...]) -> np.ndarray:
    """Periodic Gaussian blur along ``axes`` via FFT (exact for a periodic grid)."""
    if sigma_nm <= 0:
        return np.array(arr, dtype=float, copy=True)
    F = np.fft.fftn(arr, axes=axes)
    for ax in axes:
        f = np.fft.fftfreq(arr.shape[ax], d=pixel_nm)
        shape = [1] * arr.ndim
        shape[ax] = -1
        F = F * np.exp(-2 * (np.pi * sigma_nm * f) ** 2).reshape(shape)
    return np.real(np.fft.ifftn(F, axes=axes))


@dataclass
class CAResist:
    """Lumped chemically amplified (CA) positive-tone ArF resist.

    Concentrations (acid, quencher) are relative to the initial photo-acid
    generator (PAG) concentration.  ``deprotection`` arrays hold Mack's ``m``,
    the fraction of acid-labile *protecting* groups that remain (1 = unexposed,
    0 = fully deprotected / soluble).

    Parameters
    ----------
    dill_c:
        Dill C exposure rate constant, cm^2/mJ.
    quencher:
        Base quencher loading relative to PAG (neutralises acid 1:1).
    diffusion_length_nm:
        Acid diffusion length sqrt(2 D t) reached at ``peb_time_s``.
    peb_time_s:
        Reference post-exposure-bake time.
    k_amp:
        Catalytic deprotection rate constant, 1/s (per unit relative acid).
    threshold:
        ``m`` threshold of the fast threshold development model.
    r_max, r_min:
        Mack maximum / minimum development rates, nm/s.
    mack_n:
        Mack dissolution selectivity exponent.
    m_th:
        Mack threshold inhibitor concentration.
    thickness_nm:
        Resist film thickness.
    """

    dill_c: float = 0.02
    quencher: float = 0.10
    diffusion_length_nm: float = 8.0
    peb_time_s: float = 60.0
    k_amp: float = 0.08
    threshold: float = 0.5
    r_max: float = 100.0
    r_min: float = 0.05
    mack_n: float = 5.0
    m_th: float = 0.5
    thickness_nm: float = 90.0

    # ------------------------------------------------------------- exposure
    def expose(self, aerial_image, dose_mj_cm2: float, depth_factor=None) -> np.ndarray:
        """Photo-acid concentration ``1 - exp(-C * dose * I)``.

        ``depth_factor`` (1D, length nz, e.g. from
        :meth:`FilmStack.intensity_in_resist` with ``normalize=True``) adds a
        leading depth axis: the result is then ``[iz, ...lateral]``.
        """
        img = np.clip(np.asarray(aerial_image, dtype=float), 0.0, None)
        if depth_factor is not None:
            df = np.asarray(depth_factor, dtype=float).reshape((-1,) + (1,) * img.ndim)
            img = df * img[None, ...]
        return 1.0 - np.exp(-self.dill_c * float(dose_mj_cm2) * img)

    # ------------------------------------------------------------------ PEB
    def diffusion_sigma(self, time_s: float | None = None) -> float:
        """Acid diffusion length sqrt(2 D t) (nm) for a bake of ``time_s``."""
        t = self.peb_time_s if time_s is None else time_s
        D = self.diffusion_length_nm**2 / (2.0 * self.peb_time_s)
        return float(np.sqrt(2.0 * D * t))

    def diffuse_acid(
        self, acid, grid: Grid, time_s: float | None = None, has_depth: bool = False, dz_nm: float | None = None
    ) -> np.ndarray:
        """Gaussian acid diffusion: periodic FFT laterally, reflecting walls in depth."""
        acid = np.asarray(acid, dtype=float)
        sig = self.diffusion_sigma(time_s)
        out = _gaussian_blur_fft(acid, sig, grid.pixel, _lateral_axes(acid, has_depth))
        if has_depth and acid.shape[0] > 1 and sig > 0:
            dz = dz_nm if dz_nm is not None else self.thickness_nm / acid.shape[0]
            out = gaussian_filter1d(out, sig / dz, axis=0, mode="reflect")
        return out

    def peb(
        self, acid, grid: Grid, time_s: float = 60.0, has_depth: bool = False, dz_nm: float | None = None
    ) -> np.ndarray:
        """Post-exposure bake; returns the remaining protection ``m``.

        Steps: acid diffusion (Gaussian of length sqrt(2 D t)), instantaneous
        1:1 acid-quencher neutralisation ``acid_eff = max(acid - q, 0)``, then
        catalytic deprotection ``m = exp(-k_amp * acid_eff * t)``.
        """
        h = self.diffuse_acid(acid, grid, time_s, has_depth, dz_nm)
        h_eff = np.clip(h - self.quencher, 0.0, None)
        return np.exp(-self.k_amp * h_eff * time_s)

    # ----------------------------------------------------------- development
    def development_rate(self, m) -> np.ndarray:
        """Mack (1987) development rate in nm/s for remaining protection ``m``."""
        m = np.clip(np.asarray(m, dtype=float), 0.0, 1.0)
        n = self.mack_n
        a = (n + 1.0) / (n - 1.0) * (1.0 - self.m_th) ** n
        u = (1.0 - m) ** n
        return self.r_max * (a + 1.0) * u / (a + u) + self.r_min

    def develop(self, deprotection, time_s: float = 60.0, has_depth: bool = False) -> np.ndarray:
        """Remaining resist thickness fraction (0..1) after Mack development.

        Development proceeds vertically only (no lateral undercut).  Without a
        depth axis the rate is uniform with depth and the dissolved depth is
        ``r * t``.  With a depth axis ``[iz, ...]`` the arrival time
        ``t(z) = int_0^z dz' / r(z')`` is integrated cell by cell and the
        front position at ``time_s`` is interpolated within the last cell.
        """
        r = self.development_rate(deprotection)
        T = self.thickness_nm
        if not has_depth:
            return 1.0 - np.clip(r * time_s, 0.0, T) / T
        nz = r.shape[0]
        dz = T / nz
        tcell = dz / r  # time to cross each cell
        tcum = np.cumsum(tcell, axis=0)
        tprev = np.concatenate([np.zeros_like(tcum[:1]), tcum[:-1]], axis=0)
        frac = np.clip((time_s - tprev) / tcell, 0.0, 1.0)  # fraction of each cell dissolved
        depth = dz * frac.sum(axis=0)
        return 1.0 - np.clip(depth, 0.0, T) / T

    def develop_threshold(
        self, deprotection, threshold: float | None = None, has_depth: bool = False
    ) -> np.ndarray:
        """Fast threshold model: resist remains (1) where ``m >= threshold`` else 0.

        With a depth axis the film is cleared only where every depth slice
        is below threshold (the bottom must clear too).
        """
        thr = self.threshold if threshold is None else threshold
        m = np.asarray(deprotection, dtype=float)
        if has_depth:
            m = m.max(axis=0)
        return (m >= thr).astype(float)

    # --------------------------------------------------------------- process
    def process(
        self,
        aerial_image,
        dose: float,
        grid: Grid,
        film_stack: FilmStack | None = None,
        nz: int = 16,
        model: str = "mack",
        peb_time_s: float | None = None,
        develop_time_s: float = 60.0,
        measure: bool = True,
        feature: str = "line",
    ) -> ExposureResult:
        """Expose, bake and develop; return an :class:`ExposureResult`.

        If ``film_stack`` is given the normalised standing-wave depth profile
        of its ``resist`` layer modulates the exposure on ``nz`` depth slices.
        ``model`` is ``'mack'`` (remaining-thickness profile) or
        ``'threshold'`` (binary profile; the CD is then measured on the
        continuous ``m`` latent image at the threshold level, using the
        depth-maximum of ``m`` when depth-resolved).  The CD is measured along the central row
        (``iy = ny // 2``) for 2D images.
        """
        img = np.asarray(aerial_image, dtype=float)
        t_peb = self.peb_time_s if peb_time_s is None else peb_time_s
        depth = None
        if film_stack is not None:
            z = (np.arange(nz) + 0.5) * self.thickness_nm / nz
            depth = film_stack.intensity_in_resist(z, normalize=True)
        has_depth = depth is not None
        acid = self.expose(img, dose, depth)
        m = self.peb(acid, grid, t_peb, has_depth=has_depth)
        if model == "mack":
            prof = self.develop(m, develop_time_s, has_depth=has_depth)
            cd_src, level = prof, 0.5
        elif model == "threshold":
            m_eff = m.max(axis=0) if has_depth else m
            prof = self.develop_threshold(m, has_depth=has_depth)
            cd_src, level = m_eff, self.threshold
        else:
            raise ValueError("model must be 'mack' or 'threshold'")
        cd = None
        if measure:
            row = cd_src if cd_src.ndim == 1 else cd_src[cd_src.shape[0] // 2]
            x = (np.arange(row.size) - row.size // 2) * grid.pixel
            cd = measure_cd(row, x, level=level, feature=feature)
        return ExposureResult(
            aerial_image=img,
            dose=float(dose),
            resist_profile=prof,
            cd=cd,
            info=dict(acid=acid, deprotection=m, model=model, depth_factor=depth),
        )


# =============================================================================
# Metrology and process window
# =============================================================================


def measure_cd(
    profile_1d,
    x_nm,
    level: float = 0.5,
    feature: str = "line",
    periodic: bool = True,
) -> float:
    """Width of the feature nearest ``x = 0`` from interpolated level crossings.

    ``feature='line'`` measures a region where ``profile > level`` (e.g. a
    resist line in a remaining-thickness profile); ``feature='space'``
    measures a region where ``profile < level`` (e.g. the dark line of an
    aerial image, or a developed trench).  The grid is treated as periodic
    unless ``periodic=False`` (then features touching the edges are ignored).
    Returns ``nan`` if no complete feature exists.
    """
    p = np.asarray(profile_1d, dtype=float).ravel()
    x = np.asarray(x_nm, dtype=float).ravel()
    n = p.size
    if n < 3 or x.size != n or not np.all(np.isfinite(p)):
        return float("nan")
    dx = (x[-1] - x[0]) / (n - 1)
    if feature == "line":
        g = p - level
    elif feature == "space":
        g = level - p
    else:
        raise ValueError("feature must be 'line' or 'space'")
    inside = g > 0
    if inside.all() or not inside.any():
        return float("nan")
    # roll so that index 0 is outside the feature
    i0 = int(np.argmax(~inside)) if periodic else 0
    gr = np.roll(g, -i0)
    ins = gr > 0
    period = n * dx
    best = (np.inf, float("nan"))
    j = 0
    while j < n:
        if not ins[j]:
            j += 1
            continue
        start = j
        while j < n and ins[j]:
            j += 1
        end = j - 1  # last inside index
        if start == 0 or (end == n - 1 and not periodic):
            continue
        if end == n - 1 and not ins[0]:
            nxt = 0
        else:
            nxt = end + 1
        gl0, gl1 = gr[start - 1], gr[start]
        left = (start - 1) + gl0 / (gl0 - gl1)
        gr0, gr1 = gr[end], gr[nxt]
        right = end + gr0 / (gr0 - gr1)
        width = (right - left) * dx
        centre_idx = (0.5 * (left + right) + i0) % n if periodic else 0.5 * (left + right)
        xc = x[0] + centre_idx * dx
        dist = abs(xc)
        if periodic:
            dist = abs((xc + period / 2) % period - period / 2)
        if dist < best[0]:
            best = (dist, width)
    return float(best[1])


def _get_image(image) -> np.ndarray:
    return np.asarray(image() if callable(image) else image, dtype=float)


def _cd_for_dose(image, dose, grid, resist, model, feature, process_kw) -> float:
    """CD with saturating substitutes: fully remaining -> period, fully cleared -> 0."""
    res = resist.process(image, dose, grid, model=model, measure=True, feature=feature, **process_kw)
    if np.isfinite(res.cd):
        return float(res.cd)
    src = res.resist_profile if model == "mack" else res.info["deprotection"]
    if model == "threshold" and src.ndim > (1 if image.ndim == 1 else 2):
        src = src.max(axis=0)
    row = src if src.ndim == 1 else src[src.shape[0] // 2]
    level = 0.5 if model == "mack" else resist.threshold
    above = bool(np.all(row > level))
    if feature == "space":
        above = not above
    return grid.period if above else 0.0


def dose_to_size(
    image,
    target_cd: float,
    grid: Grid,
    resist: CAResist | None = None,
    dose_range: tuple[float, float] = (1.0, 200.0),
    model: str = "threshold",
    feature: str = "line",
    n_scan: int = 40,
    **process_kw,
) -> float:
    """Dose (mJ/cm^2) at which the central feature prints at ``target_cd``.

    ``image`` is an aerial image array or a zero-argument callable returning
    one.  A log-spaced scan over ``dose_range`` brackets the solution, which is
    then refined with Brent's method.  Returns ``nan`` if no bracket exists.
    """
    resist = resist or CAResist()
    img = _get_image(image)

    def f(d):
        return _cd_for_dose(img, d, grid, resist, model, feature, process_kw) - target_cd

    ds = np.geomspace(dose_range[0], dose_range[1], n_scan)
    fs = np.array([f(d) for d in ds])
    for k in range(len(ds) - 1):
        if fs[k] == 0:
            return float(ds[k])
        if np.sign(fs[k]) != np.sign(fs[k + 1]):
            return float(brentq(f, ds[k], ds[k + 1], xtol=1e-6, rtol=1e-8))
    return float("nan")


def focus_exposure_matrix(
    image_at_focus_fn: Callable[[float], np.ndarray],
    focuses_nm,
    doses,
    grid: Grid,
    resist: CAResist | None = None,
    model: str = "threshold",
    feature: str = "line",
    **process_kw,
) -> np.ndarray:
    """CD matrix ``cd[i_focus, i_dose]`` (nm; nan where the feature does not print)."""
    resist = resist or CAResist()
    focuses = np.asarray(focuses_nm, dtype=float)
    doses = np.asarray(doses, dtype=float)
    cd = np.full((focuses.size, doses.size), np.nan)
    for i, f in enumerate(focuses):
        img = np.asarray(image_at_focus_fn(f), dtype=float)
        for j, d in enumerate(doses):
            cd[i, j] = resist.process(img, d, grid, model=model, feature=feature, **process_kw).cd
    return cd


def _dose_window(cd_row, doses, target, tol_abs) -> tuple[float, float]:
    """Contiguous in-spec dose interval (interpolated edges) around the best dose."""
    g = tol_abs - np.abs(np.asarray(cd_row, dtype=float) - target)
    g = np.where(np.isfinite(g), g, -np.inf)
    if not np.any(g >= 0):
        return float("nan"), float("nan")
    k = int(np.argmax(g))
    lo = hi = k
    while lo > 0 and g[lo - 1] >= 0:
        lo -= 1
    while hi < len(g) - 1 and g[hi + 1] >= 0:
        hi += 1

    def edge(a, b):  # a inside, b outside (or end)
        ga, gb = g[a], g[b]
        if not np.isfinite(gb):
            return doses[a]
        return doses[a] + (doses[b] - doses[a]) * ga / (ga - gb)

    dlo = edge(lo, lo - 1) if lo > 0 else doses[lo]
    dhi = edge(hi, hi + 1) if hi < len(g) - 1 else doses[hi]
    return float(dlo), float(dhi)


def process_window(
    cd_matrix,
    focuses,
    doses,
    target_cd: float,
    tol: float = 0.1,
    min_el_pct: float = 5.0,
    n_focus_interp: int = 401,
) -> dict:
    """Rectangular process window from a focus-exposure CD matrix.

    For every focus the in-spec dose interval ``|CD - target| <= tol*target``
    is found (CD interpolated in dose).  The CD matrix is first linearly
    interpolated onto ``n_focus_interp`` focus values.  For every contiguous
    focus range the common dose window gives an exposure latitude
    ``EL = (d_hi - d_lo) / mean(d_hi, d_lo) * 100``; the largest focus range
    (DOF) with ``EL >= min_el_pct`` is returned.

    Returns a dict with ``dof_nm``, ``exposure_latitude_pct`` (EL of that
    window), ``best_focus``, ``best_dose`` (window centre), ``max_el_pct``
    (largest EL at any single focus) and the ``el_vs_dof`` curve
    (array of [dof, max EL] rows).
    """
    cd = np.asarray(cd_matrix, dtype=float)
    focuses = np.asarray(focuses, dtype=float)
    doses = np.asarray(doses, dtype=float)
    order = np.argsort(focuses)
    focuses, cd = focuses[order], cd[order]
    fine = np.linspace(focuses[0], focuses[-1], max(n_focus_interp, focuses.size))
    cdf = np.empty((fine.size, doses.size))
    for j in range(doses.size):
        col = cd[:, j]
        cdf[:, j] = np.interp(fine, focuses, col)  # nan propagates into adjacent segments
    tol_abs = tol * target_cd
    win = np.array([_dose_window(cdf[i], doses, target_cd, tol_abs) for i in range(fine.size)])
    lo_all, hi_all = win[:, 0], win[:, 1]
    valid = np.isfinite(lo_all)
    result = dict(
        dof_nm=0.0,
        exposure_latitude_pct=0.0,
        best_focus=float("nan"),
        best_dose=float("nan"),
        max_el_pct=0.0,
        el_vs_dof=np.zeros((0, 2)),
    )
    if not valid.any():
        return result
    el_single = np.where(valid, (hi_all - lo_all) / (0.5 * (hi_all + lo_all)) * 100.0, -np.inf)
    result["max_el_pct"] = float(np.max(el_single))
    k = int(np.argmax(el_single))
    result["best_focus"] = float(fine[k])
    result["best_dose"] = float(0.5 * (lo_all[k] + hi_all[k]))
    result["exposure_latitude_pct"] = float(el_single[k])
    best_by_dof: dict[int, float] = {}
    best_dof = (-1, None)
    nf = fine.size
    for a in range(nf):
        if not valid[a]:
            continue
        lo, hi = lo_all[a], hi_all[a]
        for b in range(a, nf):
            if not valid[b]:
                break
            lo, hi = max(lo, lo_all[b]), min(hi, hi_all[b])
            if hi <= lo:
                break
            el = (hi - lo) / (0.5 * (hi + lo)) * 100.0
            w = b - a
            if el > best_by_dof.get(w, -np.inf):
                best_by_dof[w] = el
            if el >= min_el_pct and w > best_dof[0]:
                best_dof = (w, (a, b, lo, hi, el))
    df = fine[1] - fine[0]
    result["el_vs_dof"] = np.array(sorted((w * df, e) for w, e in best_by_dof.items()))
    if best_dof[1] is not None:
        a, b, lo, hi, el = best_dof[1]
        result.update(
            dof_nm=float(fine[b] - fine[a]),
            exposure_latitude_pct=float(el),
            best_focus=float(0.5 * (fine[a] + fine[b])),
            best_dose=float(0.5 * (lo + hi)),
        )
    return result


def bossung_curves(cd_matrix, focuses_nm, doses, fit_order: int = 2) -> list[dict]:
    """Bossung curves (CD vs focus at constant dose) with polynomial fits.

    Returns one dict per dose: ``dose``, ``focus``, ``cd`` (finite points only),
    ``coeffs`` (``np.polyfit`` order ``fit_order``, or None if too few points)
    and ``iso_focus`` (vertex of a quadratic fit, i.e. the focus of zero CD
    curvature slope, else nan).
    """
    cd = np.asarray(cd_matrix, dtype=float)
    focuses = np.asarray(focuses_nm, dtype=float)
    out = []
    for j, d in enumerate(np.asarray(doses, dtype=float)):
        ok = np.isfinite(cd[:, j])
        f, c = focuses[ok], cd[ok, j]
        coeffs, vertex = None, float("nan")
        if f.size > fit_order:
            coeffs = np.polyfit(f, c, fit_order)
            if fit_order == 2 and coeffs[0] != 0:
                vertex = float(-coeffs[1] / (2 * coeffs[0]))
        out.append(dict(dose=float(d), focus=f, cd=c, coeffs=coeffs, iso_focus=vertex))
    return out


# =============================================================================
# Stochastics
# =============================================================================


def ler_from_shot_noise(
    dose_mj_cm2: float,
    image_slope: float,
    edge_intensity: float = 0.3,
    resist: CAResist | None = None,
    absorbance_k: float = N_RESIST_193.imag,
    quantum_efficiency: float = 0.3,
    wavelength_nm: float = WAVELENGTH_ARF,
    details: bool = False,
):
    """3-sigma line-edge roughness (nm) from photon / acid shot noise.

    Model: incident photons per nm^2 ``N = dose / E_photon``; the fraction
    absorbed in the film is ``1 - exp(-alpha T)`` with ``alpha = 4 pi k /
    lambda``; acids per nm^2 at the edge ``n_a = QE * N * I_edge * A_abs``.
    Acid diffusion averages Poisson noise over an effective area
    ``4 pi sigma_d^2`` (variance reduction of a 2D Gaussian kernel), so the
    relative acid fluctuation is ``1 / sqrt(n_a * 4 pi sigma_d^2)``.  A
    relative exposure error ``eps`` moves the edge by ``eps / ILS`` with
    ``ILS = image_slope`` the image log-slope ``d ln I / dx`` at the edge
    (1/nm), so ``LER_3sigma = 3 * eps / ILS``.
    """
    resist = resist or CAResist()
    e_ph = photon_energy(wavelength_nm)
    photons_per_nm2 = dose_mj_cm2 * 1e-3 * 1e-14 / e_ph
    alpha = 4 * np.pi * absorbance_k / wavelength_nm
    a_abs = 1.0 - np.exp(-alpha * resist.thickness_nm)
    acid_per_nm2 = quantum_efficiency * photons_per_nm2 * edge_intensity * a_abs
    sig = max(resist.diffusion_length_nm, 1e-3)
    area = 4 * np.pi * sig**2
    n_acid = acid_per_nm2 * area
    eps = 1.0 / np.sqrt(n_acid) if n_acid > 0 else np.inf
    ler3 = 3.0 * eps / image_slope
    if details:
        return dict(
            ler_3sigma_nm=float(ler3),
            photons_per_nm2=float(photons_per_nm2),
            absorbed_fraction=float(a_abs),
            acids_per_nm2=float(acid_per_nm2),
            acids_in_blur_area=float(n_acid),
            relative_sigma=float(eps),
        )
    return float(ler3)
