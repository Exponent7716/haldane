"""4x catadioptric projection lens, water immersion and aerial-image formation.

Physics summary
---------------
* **Imaging** uses Abbe source-point summation (partially coherent Koehler
  illumination).  Every source point at normalised pupil coordinate
  ``(sx, sy)`` illuminates the mask with a tilted plane wave; the mask
  diffraction orders ``f`` then pass the projection pupil at ``f + s*NA/lambda``.
  The field of each source point is an incoherent contributor, so intensities
  are summed with the source weights.
* **Pupil**: hard cutoff ``|f| <= NA/lambda`` (wafer-side spatial frequency in
  cycles/nm), aberration phase ``exp(i 2 pi sum c_j Z_j(rho, theta))`` with
  Fringe Zernikes and coefficients in waves, and *exact* high-NA defocus inside
  the immersion medium ``exp(i 2 pi/lambda dz [sqrt(n^2-(lambda f)^2) - n])``.
  Because the pupil is evaluated analytically at the shifted frequencies there
  is no pupil-grid quantisation of the source.
* **Immersion**: the image is formed in water (n = 1.4366 at 193 nm), so the
  wafer-side NA can exceed 1 (NA = n sin(theta) <= n).  Dry imaging requires
  NA < 1.
* **Simplified vector model**: each plane-wave order is decomposed into TE (s,
  perpendicular to the plane of incidence spanned by the optical axis and the
  order's wavevector) and TM (p, in-plane) components.  The TE unit vector is
  unchanged by the lens; the TM unit vector is rotated to be perpendicular to
  the converging wavevector in the immersion medium, ``p_out = (cos t cos phi,
  cos t sin phi, -sin t)`` with ``sin t = lambda |f| / n``.  The image intensity
  is ``|Ex|^2 + |Ey|^2 + |Ez|^2``.  Two TM orders therefore interfere with
  contrast ``p1 . p2 = cos(angle between the two wavevectors)`` (for orders in
  the same plane), the textbook TM contrast loss at high NA.  Approximations:
  no resist/film-stack refraction or Fresnel transmission, no mask-side
  (3D-mask) polarisation effects, no lens polarisation aberrations
  (Jones pupil = identity apart from the s/p rotation); the field is evaluated
  in the immersion medium just above the wafer.
* **Normalisation**: intensities are divided by the clear-mask intensity of
  the same source/pupil, so a mask with transmission 1 images to exactly 1
  (before flare, which keeps a clear field at 1).

Supporting models: :class:`ProjectionLens` (field-dependent aberration
fingerprint and manipulators), :class:`LensHeating` (absorbed power ->
Zernike drift with time constants and feed-forward correction) and
:class:`ImmersionHood` (water temperature -> index -> focus / spherical).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from math import factorial

import numpy as np

try:  # scipy's FFT is multi-threaded; fall back to numpy if unavailable.
    import scipy.fft as _fft

    def _ifft2(a: np.ndarray) -> np.ndarray:
        return _fft.ifft2(a, axes=(-2, -1), workers=-1)

    def _fft2(a: np.ndarray) -> np.ndarray:
        return _fft.fft2(a, axes=(-2, -1), workers=-1)

except Exception:  # pragma: no cover
    def _ifft2(a: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(a, axes=(-2, -1))

    def _fft2(a: np.ndarray) -> np.ndarray:
        return np.fft.fft2(a, axes=(-2, -1))

from .core import (
    DN_DT_WATER,
    FIELD_WIDTH_MM,
    N_WATER_193,
    NA_MAX_IMMERSION,
    REDUCTION,
    WAVELENGTH_ARF,
    Grid,
    ImagingSettings,
    Mask,
    SourceMap,
)

__all__ = [
    "zernike_fringe",
    "fringe_nm",
    "zernike_fit",
    "pupil_function",
    "aerial_image",
    "image_contrast",
    "image_log_slope",
    "nils",
    "resolution_limit",
    "depth_of_focus",
    "min_pitch",
    "ProjectionLens",
    "LensHeating",
    "ImmersionHood",
    "DEFAULT_CHROMATIC_FOCUS_NM_PER_PM",
]

#: Typical longitudinal chromatic focus sensitivity of an achromatised
#: catadioptric ArF immersion lens (nm of best focus per pm of wavelength).
#: Purely refractive lenses are several times larger (~150-250 nm/pm).
DEFAULT_CHROMATIC_FOCUS_NM_PER_PM = 50.0


# =============================================================================
# Fringe Zernike polynomials
# =============================================================================


def fringe_nm(j: int) -> tuple[int, int, str]:
    """Return (n, m, 'c'|'s') of Fringe (University of Arizona) Zernike ``j``.

    Ordering: groups g = (n+m)/2 = 0, 1, 2, ...; within a group m runs from g
    down to 0, cos before sin.  j = 1..36 are the complete groups 0..5 and
    j = 37 is the extra (n=12, m=0) term.
    """
    if not 1 <= j <= 37:
        raise ValueError("Fringe Zernike index must be in 1..37")
    if j == 37:
        return 12, 0, "c"
    k = 0
    for g in range(6):
        for m in range(g, -1, -1):
            n = 2 * g - m
            for kind in (("c", "s") if m > 0 else ("c",)):
                k += 1
                if k == j:
                    return n, m, kind
    raise AssertionError("unreachable")


def _radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rho, dtype=float)
    for k in range((n - m) // 2 + 1):
        c = (-1) ** k * factorial(n - k) / (
            factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k)
        )
        out = out + c * rho ** (n - 2 * k)
    return out


def zernike_fringe(j: int, rho, theta) -> np.ndarray:
    """Fringe Zernike polynomial Z_j(rho, theta), j = 1..37.

    Convention: *un-normalised* Fringe polynomials, ``Z = R_n^m(rho) *
    {cos, sin}(m theta)`` with ``R_n^m(1) = 1``, i.e. every term has peak
    magnitude 1 at the pupil edge.  A coefficient ``c_j`` in waves therefore
    gives a wavefront of ``c_j`` waves at the edge (Z1 piston, Z2/Z3 tilt,
    Z4 = 2 rho^2 - 1 defocus, Z5/Z6 astigmatism, Z7/Z8 coma,
    Z9 = 6 rho^4 - 6 rho^2 + 1 primary spherical, Z10/Z11 trefoil, ...).
    ``rho`` is the normalised pupil radius (values outside [0, 1] are evaluated
    by the polynomial; masking is the caller's job).
    """
    n, m, kind = fringe_nm(j)
    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    r = _radial(n, m, rho)
    if m == 0:
        return r * np.ones_like(theta)
    return r * (np.cos(m * theta) if kind == "c" else np.sin(m * theta))


def zernike_fit(wavefront, rho, theta, terms=range(1, 38)) -> dict[int, float]:
    """Least-squares Fringe Zernike coefficients of ``wavefront`` sampled at
    (rho, theta) points (1D arrays, rho <= 1)."""
    terms = list(terms)
    A = np.stack([zernike_fringe(j, rho, theta).ravel() for j in terms], axis=1)
    c, *_ = np.linalg.lstsq(A, np.asarray(wavefront, float).ravel(), rcond=None)
    return {j: float(v) for j, v in zip(terms, c)}


# =============================================================================
# Pupil
# =============================================================================


def _pupil_at(fx, fy, settings: ImagingSettings, obliquity: bool = False) -> np.ndarray:
    """Pupil transmission at arbitrary wafer-side frequencies (cycles/nm)."""
    lam, na, n = settings.wavelength, settings.na, settings.n_immersion
    if na > n:
        raise ValueError(f"NA={na} exceeds the immersion index n={n}: not physical")
    fr2 = fx * fx + fy * fy
    inside = fr2 <= (na / lam) ** 2 * (1 + 1e-12)
    phase = np.zeros(np.broadcast(fx, fy).shape)
    if settings.defocus:
        kz = np.sqrt(np.maximum(n * n - lam * lam * fr2, 0.0))
        phase = phase + settings.defocus / lam * (kz - n)  # waves
    zern = {j: c for j, c in (settings.zernikes or {}).items() if c}
    if zern:
        rho = np.sqrt(fr2) * lam / na
        th = np.arctan2(fy, fx)
        for j, c in zern.items():
            phase = phase + c * zernike_fringe(j, rho, th)
    p = np.where(inside, np.exp(2j * np.pi * phase), 0.0)
    if obliquity:
        # Radiometric (energy-conservation) factor of a 4x aplanatic lens:
        # amplitude ~ [(1 - sin^2 t_mask)/(1 - sin^2 t_wafer)]^(1/4).
        s2w = np.minimum(lam * lam * fr2 / n**2, 0.999999)
        s2m = s2w * (n / REDUCTION) ** 2
        p = p * ((1 - s2m) / (1 - s2w)) ** 0.25
    return p


def pupil_function(grid: Grid, settings: ImagingSettings, sx: float = 0.0, sy: float = 0.0,
                   obliquity: bool = False) -> np.ndarray:
    """Complex pupil on the (fftshift-ed) grid frequency mesh, shifted for a
    source point at sigma (sx, sy): returns ``P(f + sigma*NA/lambda)``.

    Cutoff ``|f| <= NA/lambda``; phase = aberrations (Fringe Zernikes, waves)
    + exact immersion defocus ``dz/lambda (sqrt(n^2 - (lambda f)^2) - n)``
    (on-axis term removed so defocus adds no piston).  ``obliquity=True``
    applies the radiometric amplitude factor of a 4x reduction lens.
    """
    FX, FY = grid.freq_mesh()
    s = settings.na / settings.wavelength
    return _pupil_at(FX + sx * s, FY + sy * s, settings, obliquity)


# =============================================================================
# Aerial image (Abbe)
# =============================================================================


def _source_pol_vectors(pol: str, sx: np.ndarray, sy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Incident (ex, ey) Jones vectors per source point; a list of incoherent
    polarisation states to be averaged."""
    one, zero = np.ones_like(sx), np.zeros_like(sx)
    p = pol.lower()
    if p in ("unpolarized", "unpolarised", "none"):
        return [(one, zero), (zero, one)]
    if p == "x":
        return [(one, zero)]
    if p == "y":
        return [(zero, one)]
    if p in ("te", "azimuthal", "tm", "radial"):
        r = np.hypot(sx, sy)
        ok = r > 1e-9
        cx = np.where(ok, sx / np.where(ok, r, 1), 0.0)
        cy = np.where(ok, sy / np.where(ok, r, 1), 0.0)
        if p in ("te", "azimuthal"):
            ex, ey = -cy, cx
        else:
            ex, ey = cx, cy
        # On-axis points have no defined azimuth: treat them as unpolarised
        # by splitting into X and Y states with half weight each (handled
        # by returning two states where only on-axis points differ).
        ex1 = np.where(ok, ex, 1.0)
        ey1 = np.where(ok, ey, 0.0)
        ex2 = np.where(ok, ex, 0.0)
        ey2 = np.where(ok, ey, 1.0)
        return [(ex1, ey1), (ex2, ey2)]
    raise ValueError(f"unknown polarization '{pol}'")


def _mono_image(spec_c, kx, ky, period, N, sx, sy, w, settings, pol, obliquity, chunk):
    """Unnormalised Abbe image and clear-field normaliser for one wavelength.

    The fields are synthesised on a reduced M x M grid (M >= 4 kmax + 2), which
    represents |E|^2 (band-limited to 2 kmax) without aliasing; the summed
    intensity is Fourier-interpolated to the full N x N grid once at the end.
    """
    lam, na, n = settings.wavelength, settings.na, settings.n_immersion
    s = na / lam
    fx = kx[None, None, :] / period  # (1,1,m)
    fy = ky[None, :, None] / period  # (1,m,1)
    kmax = int(np.max(np.abs(kx)))
    M = min(N, _fast_even(4 * kmax + 2))
    idx_x = kx % M
    idx_y = ky % M
    scalar = pol.lower() == "scalar"
    states = None if scalar else _source_pol_vectors(pol, sx, sy)
    image = np.zeros((M, M))
    norm = 0.0
    for a in range(0, len(sx), chunk):
        b = min(a + chunk, len(sx))
        csx, csy, cw = sx[a:b, None, None], sy[a:b, None, None], w[a:b]
        gx = fx + csx * s  # pupil-plane frequency of each order (C,1,m)
        gy = fy + csy * s
        P = _pupil_at(gx, gy, settings, obliquity)  # (C,m,m)
        P0 = _pupil_at(csx[:, 0, 0] * s, csy[:, 0, 0] * s, settings, obliquity)
        norm += float(np.sum(cw * np.abs(P0) ** 2))
        A = P * spec_c[None]  # field amplitude per order
        if scalar:
            comps = [A]
            weights = [cw]
        else:
            g = np.sqrt(gx * gx + gy * gy)
            g = np.broadcast_to(g, A.shape)
            phi = np.arctan2(np.broadcast_to(gy, A.shape), np.broadcast_to(gx, A.shape))
            st = np.clip(lam * g / n, 0.0, 1.0)
            ct = np.sqrt(1.0 - st * st)
            cphi, sphi = np.cos(phi), np.sin(phi)
            comps, weights = [], []
            nst = len(states)
            for ex, ey in states:
                ex = ex[a:b, None, None]
                ey = ey[a:b, None, None]
                a_s = -ex * sphi + ey * cphi
                a_p = ex * cphi + ey * sphi
                comps += [A * (-a_s * sphi + a_p * ct * cphi),
                          A * (a_s * cphi + a_p * ct * sphi),
                          A * (-a_p * st)]
                weights += [cw / nst] * 3
        stack = np.zeros((len(comps) * (b - a), M, M), dtype=complex)
        allc = np.concatenate(comps, axis=0)
        stack[:, idx_y[:, None], idx_x[None, :]] = allc
        E = _ifft2(stack)
        wts = np.concatenate(weights)
        image += np.tensordot(wts, (E.real ** 2 + E.imag ** 2), axes=(0, 0))
    image *= float(M * M) ** 2
    if M < N:  # band-limited Fourier interpolation M x M -> N x N
        sp = _fft2(image) / (M * M)
        k = np.arange(-(M // 2) + 1, M // 2)  # drop the (empty) Nyquist bin
        full = np.zeros((N, N), dtype=complex)
        full[(k % N)[:, None], (k % N)[None, :]] = sp[(k % M)[:, None], (k % M)[None, :]]
        image = _ifft2(full).real * (N * N)
    return image, norm


def _fast_even(n: int) -> int:
    """Smallest even 2^a 3^b 5^c >= n (fast FFT size)."""
    m = max(2, n + (n % 2))
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1 and m % 2 == 0:
            return m
        m += 2


def aerial_image(mask: Mask, source: SourceMap, settings: ImagingSettings,
                 spectrum=None, polarization: str | None = None,
                 chromatic_focus_nm_per_pm: float = DEFAULT_CHROMATIC_FOCUS_NM_PER_PM,
                 obliquity: bool = False, source_threshold: float = 1e-3,
                 chunk: int = 32) -> np.ndarray:
    """Aerial image (normalised, clear field = 1) by Abbe source-point summation.

    Parameters
    ----------
    mask : Mask
        Complex transmission at wafer scale on a periodic :class:`Grid`.
    source : SourceMap
        Illumination; ``source.points()`` gives (sx, sy, weight).
    settings : ImagingSettings
        Wavelength, NA, immersion index, defocus, Fringe Zernikes (waves), flare.
    spectrum : (wavelengths_nm, weights) or None
        Laser spectrum.  Images are averaged over wavelengths with focus shift
        ``chromatic_focus_nm_per_pm * (lambda - settings.wavelength) * 1000``.
    polarization : str or None
        'unpolarized' (incoherent mean of X and Y), 'X', 'Y', 'TE'/'azimuthal',
        'TM'/'radial' (simplified vector model, see module docstring) or
        'scalar' (pure scalar imaging).  None uses ``source.polarization``.
    chromatic_focus_nm_per_pm : float
        Longitudinal chromatic aberration (0 for an ideal achromat).
    obliquity : bool
        Apply the radiometric pupil amplitude factor.

    Returns
    -------
    ndarray [iy, ix] of intensity; flare applied as
    ``I = (1-f) I + f * mean(|t|^2)``.
    """
    grid = mask.grid
    N = grid.n
    period = grid.period
    t = np.asarray(mask.transmission, dtype=complex)
    if t.shape != (N, N):
        raise ValueError("mask transmission shape does not match its grid")
    pol = polarization if polarization is not None else (source.polarization or "unpolarized")
    sx, sy, w = source.points(source_threshold)
    sx, sy, w = (np.asarray(v, float).ravel() for v in (sx, sy, w))

    if spectrum is None:
        lams, lw = np.array([settings.wavelength]), np.array([1.0])
    else:
        lams = np.asarray(spectrum[0], float).ravel()
        lw = np.asarray(spectrum[1], float).ravel()
        lw = lw / lw.sum()

    spec = _fft2(t) / (N * N)  # unshifted; spec[0,0] = mean transmission
    sig_max = float(np.max(np.hypot(sx, sy))) if len(sx) else 0.0
    image = np.zeros((N, N))
    for lam, wl in zip(lams, lw):
        if wl <= 0:
            continue
        st = dataclasses.replace(
            settings, wavelength=float(lam),
            defocus=settings.defocus + chromatic_focus_nm_per_pm * (lam - settings.wavelength) * 1e3)
        kmax = int(np.ceil(st.na / st.wavelength * (1 + sig_max) * period)) + 1
        kmax = min(kmax, (N - 1) // 2)
        k = np.arange(-kmax, kmax + 1)
        spec_c = spec[(k % N)[:, None], (k % N)[None, :]]
        img, norm = _mono_image(spec_c, k, k, period, N, sx, sy, w, st, pol, obliquity, chunk)
        if norm <= 0:
            raise ValueError("source lies entirely outside the pupil")
        image += wl * img / norm
    f = float(settings.flare or 0.0)
    if f:
        image = (1 - f) * image + f * float(np.mean(np.abs(t) ** 2))
    return image


# =============================================================================
# Image metrics and textbook scaling laws
# =============================================================================


def image_contrast(image_1d) -> float:
    """Michelson contrast (Imax - Imin) / (Imax + Imin)."""
    a = np.asarray(image_1d, float)
    mx, mn = a.max(), a.min()
    return float((mx - mn) / (mx + mn)) if mx + mn > 0 else 0.0


def image_log_slope(image_1d, x, edge_x: float) -> float:
    """|d ln I / dx| at ``edge_x`` (1/nm), by linear interpolation of the
    finite-difference gradient (periodic data are fine)."""
    a = np.asarray(image_1d, float)
    x = np.asarray(x, float)
    dlog = np.gradient(np.log(np.maximum(a, 1e-30)), x)
    return float(abs(np.interp(edge_x, x, dlog)))


def nils(image_1d, x, edge_x: float, cd: float) -> float:
    """Normalised image log slope ``cd * |d ln I/dx|`` at the feature edge."""
    return cd * image_log_slope(image_1d, x, edge_x)


def resolution_limit(k1: float, wavelength: float = WAVELENGTH_ARF,
                     na: float = NA_MAX_IMMERSION) -> float:
    """Rayleigh resolution (half-pitch) R = k1 * lambda / NA, in nm."""
    return k1 * wavelength / na


def min_pitch(wavelength: float = WAVELENGTH_ARF, na: float = NA_MAX_IMMERSION,
              sigma: float = 1.0) -> float:
    """Smallest pitch with any imaging (two-beam limit): lambda / (NA (1 + sigma))."""
    return wavelength / (na * (1.0 + sigma))


def depth_of_focus(k2: float, wavelength: float = WAVELENGTH_ARF,
                   na: float = NA_MAX_IMMERSION, n: float = N_WATER_193) -> float:
    """Depth of focus in nm, high-NA (Lin) form:

    ``DOF = k2 * lambda / (2 n (1 - sqrt(1 - (NA/n)^2)))``

    which reduces to the paraxial Rayleigh form ``k2 * n * lambda / NA^2``;
    at fixed NA, immersion (larger n) increases the DOF.
    """
    if na >= n:
        raise ValueError("NA must be smaller than the medium index")
    return k2 * wavelength / (2.0 * n * (1.0 - np.sqrt(1.0 - (na / n) ** 2)))


# =============================================================================
# Projection lens: aberration fingerprint + manipulators
# =============================================================================


@dataclass
class ProjectionLens:
    """4x catadioptric immersion projection lens.

    The nominal aberration fingerprint is a smooth function of the slit
    position x (wafer scale, mm, |x| <= 13): for each Fringe term
    ``c_j(u) = a_j + b_j u + q_j u^2`` with ``u = x / (slit half width)``,
    random coefficients of a few milliwaves (seeded, reproducible).
    Odd terms in x of the x-type terms are allowed (real lenses have them).

    Manipulators (movable/deformable elements) add a field-constant offset
    ``offset_j`` and, for focus (Z4) and x-tilt (Z2), also a linear-in-field
    term (``tilt_j`` per unit u), modelling element tilt / wafer-stage tilt.
    """

    na: float = NA_MAX_IMMERSION
    reduction: float = REDUCTION
    wavelength: float = WAVELENGTH_ARF
    n_immersion: float = N_WATER_193
    slit_width_mm: float = FIELD_WIDTH_MM
    rms_milliwaves: float = 5.0
    max_term: int = 16
    seed: int = 7
    manipulators: dict[int, float] = field(default_factory=dict)
    manipulator_tilts: dict[int, float] = field(default_factory=dict)
    manipulator_range_waves: float = 0.2
    drift: dict[int, float] = field(default_factory=dict)

    #: Zernike terms the manipulators can adjust (field-constant part).
    CORRECTABLE = (2, 3, 4, 5, 6, 7, 8, 9)
    #: Terms that also have a linear-in-field manipulator.
    TILTABLE = (2, 4)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        terms = np.arange(2, self.max_term + 1)
        scale = self.rms_milliwaves * 1e-3
        # Higher-order terms are typically smaller.
        decay = 1.0 / np.sqrt(1.0 + 0.15 * (terms - 2))
        self._terms = terms
        self._a = rng.normal(0, scale, len(terms)) * decay
        self._b = rng.normal(0, scale * 0.6, len(terms)) * decay
        self._q = rng.normal(0, scale * 0.6, len(terms)) * decay

    # -- fingerprint -------------------------------------------------------
    def _u(self, field_x_mm) -> np.ndarray:
        return np.asarray(field_x_mm, float) / (self.slit_width_mm / 2.0)

    def nominal_aberrations(self, field_x_mm: float) -> dict[int, float]:
        """Uncorrected lens fingerprint (waves) at slit position x (mm)."""
        u = float(self._u(field_x_mm))
        c = self._a + self._b * u + self._q * u * u
        return {int(j): float(v) for j, v in zip(self._terms, c)}

    def aberrations_at(self, field_x_mm: float, extra: dict[int, float] | None = None) -> dict[int, float]:
        """Total Zernikes (waves) at slit position x: fingerprint + manipulators
        + drift (e.g. lens heating) + ``extra``."""
        u = float(self._u(field_x_mm))
        out = self.nominal_aberrations(field_x_mm)
        for src in (self.manipulators, self.drift, extra or {}):
            for j, v in src.items():
                out[j] = out.get(j, 0.0) + v
        for j, v in self.manipulator_tilts.items():
            out[j] = out.get(j, 0.0) + v * u
        return out

    def imaging_settings(self, field_x_mm: float = 0.0, base: ImagingSettings | None = None,
                         extra: dict[int, float] | None = None) -> ImagingSettings:
        """ImagingSettings with this lens' NA and aberrations at ``field_x_mm``."""
        base = base or ImagingSettings(wavelength=self.wavelength, n_immersion=self.n_immersion)
        z = dict(base.zernikes)
        for j, v in self.aberrations_at(field_x_mm, extra).items():
            z[j] = z.get(j, 0.0) + v
        return dataclasses.replace(base, na=self.na, zernikes=z)

    # -- manipulators ------------------------------------------------------
    def set_manipulators(self, offsets: dict[int, float] | None = None,
                         tilts: dict[int, float] | None = None, **kw) -> None:
        """Set manipulator offsets (waves) per Zernike, e.g.
        ``set_manipulators({4: -0.01, 9: 0.003})`` or ``set_manipulators(Z4=-0.01)``.
        Values are clipped to +-``manipulator_range_waves``; only
        :attr:`CORRECTABLE` terms (offsets) and :attr:`TILTABLE` terms (tilts)
        are accepted."""
        offsets = dict(offsets or {})
        for k, v in kw.items():
            if k.upper().startswith("Z"):
                offsets[int(k[1:])] = v
            else:
                raise TypeError(f"unknown manipulator '{k}'")
        r = self.manipulator_range_waves
        for j, v in offsets.items():
            if j not in self.CORRECTABLE:
                raise ValueError(f"Z{j} is not correctable by the manipulators")
            self.manipulators[j] = float(np.clip(v, -r, r))
        for j, v in (tilts or {}).items():
            if j not in self.TILTABLE:
                raise ValueError(f"Z{j} has no field-tilt manipulator")
            self.manipulator_tilts[j] = float(np.clip(v, -r, r))

    def reset_manipulators(self) -> None:
        self.manipulators.clear()
        self.manipulator_tilts.clear()

    def optimize_manipulators(self, field_points_mm=None, terms=None) -> dict[int, float]:
        """Least-squares set the manipulators to minimise the field-RMS of the
        correctable terms (including current drift).  Returns residual RMS
        (waves) per term over the field points."""
        terms = tuple(terms or self.CORRECTABLE)
        xs = np.linspace(-self.slit_width_mm / 2, self.slit_width_mm / 2, 13) \
            if field_points_mm is None else np.asarray(field_points_mm, float)
        self.reset_manipulators()
        u = self._u(xs)
        offs, tilts = {}, {}
        for j in terms:
            vals = np.array([self.aberrations_at(x).get(j, 0.0) for x in xs])
            if j in self.TILTABLE and len(xs) > 1:
                slope, icpt = np.polyfit(u, vals, 1)
                tilts[j] = -slope
                offs[j] = -icpt
            else:
                offs[j] = -float(vals.mean())
        self.set_manipulators(offs, tilts)
        res = {}
        for j in terms:
            vals = np.array([self.aberrations_at(x).get(j, 0.0) for x in xs])
            res[j] = float(np.sqrt(np.mean(vals ** 2)))
        return res

    def field_rms(self, field_points_mm=None, terms=None) -> float:
        """RMS (waves) of the given terms over field points (fringe-coefficient RMS)."""
        xs = np.linspace(-self.slit_width_mm / 2, self.slit_width_mm / 2, 13) \
            if field_points_mm is None else np.asarray(field_points_mm, float)
        terms = tuple(terms or range(2, self.max_term + 1))
        v = np.array([[self.aberrations_at(x).get(j, 0.0) for j in terms] for x in xs])
        return float(np.sqrt(np.mean(v ** 2)))


# =============================================================================
# Lens heating
# =============================================================================


@dataclass
class LensHeating:
    """Lens-heating aberration drift.

    Absorbed power ``P`` (W) in the lens elements heats them; the index change
    (dn/dT of fused silica / CaF2) and expansion produce wavefront drift.
    Each Zernike follows a sum of first-order thermal responses::

        c_j(t) = P * S_j * sum_k a_k (1 - exp(-t / tau_k))

    with sensitivities ``S_j`` (waves/W at saturation) that depend on the
    illumination shape: a conventional source mostly gives focus (Z4) and
    spherical (Z9); a dipole concentrates heat in two lobes near the pupil edge
    and adds strong astigmatism (Z5, sign set by dipole orientation) and Z12;
    quadrupole adds 4-fold (Z17).  Values are illustrative but of realistic
    order (~0.1 W absorbed at high throughput gives
    ~10-25 milliwaves, dipoles mostly astigmatism).
    """

    tau_s: tuple[float, ...] = (60.0, 600.0)
    amplitudes: tuple[float, ...] = (0.6, 0.4)
    sensitivity: dict[str, dict[int, float]] = field(default_factory=lambda: {
        "conventional": {4: 0.20, 9: 0.06},
        "annular": {4: 0.18, 9: -0.04, 16: 0.02},
        "dipole_x": {4: 0.15, 5: 0.25, 9: -0.03, 12: 0.06},
        "dipole_y": {4: 0.15, 5: -0.25, 9: -0.03, 12: -0.06},
        "quadrupole": {4: 0.16, 9: -0.04, 17: 0.08},
    })
    absorption: float = 0.05  # fraction of transmitted power absorbed in the lens

    def absorbed_power(self, dose_mj_cm2: float, reticle_transmission: float = 0.5,
                       field_area_mm2: float = 26.0 * 33.0, fields_per_s: float = 6.0,
                       lens_transmission_to_wafer: float = 0.7) -> float:
        """Average absorbed power (W) for exposure at ``dose_mj_cm2``.

        Optical power at the wafer = dose * field area * fields/s; the light
        entering the lens is that divided by the lens transmission, and a
        fraction ``absorption`` of it is absorbed.  ``reticle_transmission``
        scales the open-frame power (dark-field masks heat less).
        """
        e_wafer = dose_mj_cm2 * 1e-3 * (field_area_mm2 * 1e-2)  # J per field (cm^2)
        p_wafer = e_wafer * fields_per_s * reticle_transmission
        return float(p_wafer / lens_transmission_to_wafer * self.absorption)

    def _shape(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return sum(a * (1 - np.exp(-np.maximum(t, 0) / tau))
                   for a, tau in zip(self.amplitudes, self.tau_s))

    def _sens(self, source_kind: str) -> dict[int, float]:
        key = source_kind.lower()
        if key == "dipole":
            key = "dipole_x"
        if key not in self.sensitivity:
            raise ValueError(f"unknown source kind '{source_kind}'")
        return self.sensitivity[key]

    def drift(self, t_s, power_w: float, source_kind: str = "conventional") -> dict[int, np.ndarray | float]:
        """Zernike drift (waves) after exposing for ``t_s`` seconds at constant
        absorbed power ``power_w`` from a cold lens."""
        g = self._shape(t_s)
        out = {}
        for j, s in self._sens(source_kind).items():
            v = power_w * s * g
            out[j] = float(v) if np.ndim(v) == 0 else v
        return out

    def simulate(self, times_s, power_w, source_kind: str = "conventional") -> dict[int, np.ndarray]:
        """Drift for a time-varying power trace (piecewise-constant on the
        given time samples, e.g. exposure on / wafer exchange off)."""
        t = np.asarray(times_s, float)
        p = np.broadcast_to(np.asarray(power_w, float), t.shape)
        states = np.zeros(len(self.tau_s))
        resp = np.zeros_like(t)
        for i in range(len(t)):
            if i > 0:
                dt = t[i] - t[i - 1]
                for k, (a, tau) in enumerate(zip(self.amplitudes, self.tau_s)):
                    target = a * p[i - 1]
                    states[k] = target + (states[k] - target) * np.exp(-dt / tau)
            resp[i] = states.sum()
        return {j: s * resp for j, s in self._sens(source_kind).items()}

    def feedforward(self, t_s, power_w: float, source_kind: str = "conventional",
                    model_error: float = 0.1) -> dict[str, dict[int, np.ndarray | float]]:
        """Feed-forward correction: the scanner's heating model predicts the
        drift (with relative error ``model_error``) and the manipulators apply
        the negative.  Returns {'drift', 'correction', 'residual'}."""
        d = self.drift(t_s, power_w, source_kind)
        corr = {j: -(1 + model_error) * v for j, v in d.items()}
        res = {j: d[j] + corr[j] for j in d}
        return {"drift": d, "correction": corr, "residual": res}


# =============================================================================
# Immersion hood
# =============================================================================


@dataclass
class ImmersionHood:
    """Water immersion hood (showerhead) between last lens element and wafer.

    A water gap of thickness ``gap_mm`` with index ``n`` (1.4366 at 193 nm,
    dn/dT = -1e-4 /K).  A temperature deviation changes the index by
    ``dn = dn/dT * dT``; the optical path of a plane wave with pupil radius
    rho through the gap changes by ``OPD(rho) = gap * n dn / sqrt(n^2 - NA^2 rho^2)``.
    Projected on the exact immersion-defocus function this is a focus shift
    (paraxially ``dz = -gap * dn / n``), and the high-NA residual is mainly
    spherical (Z9).  Hence water temperature must be held to ~mK.

    Other immersion effects (not modelled quantitatively): the dynamic
    meniscus at the hood edge must stay pinned at scan speeds (> 0.5 m/s),
    otherwise droplets are left on the wafer (watermark defects) and air
    bubbles can be entrained; bubbles in the gap scatter light and print as
    defects.  Water absorption at 193 nm (~0.04 /cm) causes a small dose loss.
    """

    gap_mm: float = 1.0
    n: float = N_WATER_193
    dn_dT: float = DN_DT_WATER
    na: float = NA_MAX_IMMERSION
    wavelength: float = WAVELENGTH_ARF
    absorption_per_cm: float = 0.036

    def delta_n(self, delta_T_K: float) -> float:
        return self.dn_dT * delta_T_K

    def _fit(self, delta_T_K: float):
        rho = np.linspace(0, 1, 401)
        wts = rho  # area weighting over the disk
        s = self.na * rho
        d = self.gap_mm * 1e6
        dn = self.delta_n(delta_T_K)
        opd = d * (np.sqrt((self.n + dn) ** 2 - s * s) - np.sqrt(self.n ** 2 - s * s))
        g = np.sqrt(self.n ** 2 - s * s) - self.n  # defocus OPD per nm of dz
        A = np.stack([np.ones_like(rho), g], 1) * np.sqrt(wts)[:, None]
        (piston, dz), *_ = np.linalg.lstsq(A, opd * np.sqrt(wts), rcond=None)
        resid = opd - piston - dz * g
        return dz, resid, rho, opd

    def focus_shift_nm(self, delta_T_K: float) -> float:
        """Best-fit focus shift (nm) of the image for a water temperature
        deviation ``delta_T_K`` (positive = image moves in +z of the defocus
        convention used by :func:`pupil_function`)."""
        return float(self._fit(delta_T_K)[0])

    def wavefront_zernikes(self, delta_T_K: float) -> dict[int, float]:
        """Fringe Zernike content (waves) of the gap wavefront change:
        Z4 (defocus) and Z9, Z16 (spherical)."""
        _, _, rho, opd = self._fit(delta_T_K)
        th = np.zeros_like(rho)
        c = zernike_fit(opd / self.wavelength, rho, th, terms=(1, 4, 9, 16))
        return {j: c[j] for j in (4, 9, 16)}

    def spherical_waves(self, delta_T_K: float) -> float:
        """Z9 (waves) remaining after refocus."""
        dz, resid, rho, _ = self._fit(delta_T_K)
        c = zernike_fit(resid / self.wavelength, rho, np.zeros_like(rho), terms=(1, 4, 9, 16))
        return c[9]

    def transmission(self) -> float:
        """Water-gap intensity transmission (Beer-Lambert)."""
        return float(np.exp(-self.absorption_per_cm * self.gap_mm * 0.1))
