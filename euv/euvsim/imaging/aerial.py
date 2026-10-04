"""Abbe partially-coherent aerial-image computation and image metrics.

For every source point s (sigma_x, sigma_y, weight) the mask is illuminated by a
tilted plane wave; equivalently the pupil is shifted by f_s = sigma * NA/lambda:

    E_s(x) = IFFT[ M(f) * P(f + f_s) ],     I(x) = sum_s w_s |E_s(x)|^2 / sum_s w_s |P(f_s)|^2

The denominator normalises the image so that a fully reflective (clear) mask
gives I = 1 (ARCHITECTURE contract).  Scalar imaging; polarisation/vector
effects (significant at NA 0.55 for TM) are not included.  Flare from the
optics is applied after the coherent sum.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .mask import ReflectiveMask
from .projection import ProjectionOptics


def _normalise_source(source_points: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(np.asarray(source_points, dtype=float))
    if s.shape[1] != 3:
        raise ValueError("source_points must have shape (N, 3): sigma_x, sigma_y, weight")
    s = s[s[:, 2] > 0]
    if len(s) == 0:
        raise ValueError("no source points with positive weight")
    return s


def aerial_image(pattern: np.ndarray, dx_nm: float, source_points: np.ndarray,
                 optics: ProjectionOptics, defocus_nm: float = 0.0,
                 mask: Optional[ReflectiveMask] = None, chunk: int = 64) -> np.ndarray:
    """Normalised aerial image (clear mask = 1) of a wafer-scale mask pattern.

    ``pattern``: 2-D array (1 = reflective) or a complex near field.  If ``mask``
    is given, the pattern is converted with ``mask.near_field`` (absorber
    transmission/phase and M3D shadowing); otherwise a binary thin mask with a
    perfectly dark absorber is assumed.
    """
    p = np.atleast_2d(np.asarray(pattern))
    if mask is not None:
        a = mask.near_field(p.real, dx_nm)
    else:
        a = p.astype(complex)
    ny, nx = a.shape
    A = np.fft.fft2(a)
    fx = np.fft.fftfreq(nx, dx_nm)[None, None, :]
    fy = np.fft.fftfreq(ny, dx_nm)[None, :, None]
    src = _normalise_source(source_points)
    fc = optics.cutoff_frequency
    image = np.zeros((ny, nx))
    norm = 0.0
    for i in range(0, len(src), chunk):
        s = src[i:i + chunk]
        fsx = (s[:, 0] * fc)[:, None, None]
        fsy = (s[:, 1] * fc)[:, None, None]
        w = s[:, 2]
        P = optics.pupil(fx + fsx, fy + fsy, defocus_nm)
        E = np.fft.ifft2(A[None] * P, axes=(-2, -1))
        image += np.tensordot(w, np.abs(E) ** 2, axes=1)
        P0 = optics.pupil(s[:, 0] * fc, s[:, 1] * fc, defocus_nm)
        norm += float(np.sum(w * np.abs(P0) ** 2))
    if norm <= 0:
        raise ValueError("source lies entirely outside the pupil (dark-field illumination)")
    image /= norm
    return optics.apply_flare(image, dx_nm)


# ----------------------------------------------------------------- metrics
def cutline(image: np.ndarray, axis: str = "x", index: Optional[int] = None) -> np.ndarray:
    """1-D intensity profile along x (a row) or y (a column) through the centre."""
    image = np.atleast_2d(image)
    if axis == "x":
        return image[image.shape[0] // 2 if index is None else index, :]
    return image[:, image.shape[1] // 2 if index is None else index]


def image_contrast(profile: np.ndarray) -> float:
    """Michelson contrast (Imax - Imin)/(Imax + Imin)."""
    p = np.asarray(profile)
    return float((p.max() - p.min()) / (p.max() + p.min()))


def _edges(profile: np.ndarray, dx_nm: float, threshold: float, center: Optional[int],
           tone: str):
    p = np.asarray(profile, dtype=float)
    n = len(p)
    c = n // 2 if center is None else center
    inside = (p < threshold) if tone == "dark" else (p > threshold)
    if not inside[c]:
        return None
    if inside.all():
        return None

    def walk(step):
        i, k = c, 0
        for _ in range(n):
            j = (i + step) % n
            if not inside[j]:
                # linear interpolation between i (inside) and j (outside)
                frac = (threshold - p[i]) / (p[j] - p[i])
                return k + step * frac, i, j
            i, k = j, k + step
        return None

    r, l = walk(+1), walk(-1)
    if r is None or l is None:
        return None
    return r, l


def measure_cd(profile: np.ndarray, dx_nm: float, threshold: float,
               center: Optional[int] = None, tone: str = "dark") -> float:
    """CD (nm) of the feature at ``center`` (default: profile centre) using a
    constant-threshold model.  ``tone='dark'``: feature is where I < threshold
    (e.g. an absorber line printing as resist line in a positive resist's
    unexposed region); ``'bright'``: where I > threshold.  NaN if not printed."""
    e = _edges(profile, dx_nm, threshold, center, tone)
    if e is None:
        return float("nan")
    (r, _, _), (l, _, _) = e
    return float((r - l) * dx_nm)


def nils(profile: np.ndarray, dx_nm: float, threshold: Optional[float] = None,
         center: Optional[int] = None, tone: str = "dark",
         cd_nm: Optional[float] = None) -> float:
    """Normalised image log-slope  NILS = CD * |d ln I / dx|  at the edges,
    averaged over both edges.  Default threshold: (Imax + Imin)/2.
    ``cd_nm`` defaults to the measured CD."""
    p = np.asarray(profile, dtype=float)
    if threshold is None:
        threshold = 0.5 * (p.max() + p.min())
    e = _edges(p, dx_nm, threshold, center, tone)
    if e is None:
        return 0.0
    n = len(p)
    slopes = []
    for _, i, j in e:
        slopes.append(abs(p[j % n] - p[i % n]) / dx_nm / threshold)
    (r, _, _), (l, _, _) = e
    w = cd_nm if cd_nm is not None else (r - l) * dx_nm
    return float(w * np.mean(slopes))


# --------------------------------------------------------- process window
def through_focus(pattern, dx_nm, source_points, optics, focuses_nm, mask=None):
    """List of aerial images over a focus range."""
    return [aerial_image(pattern, dx_nm, source_points, optics, f, mask) for f in focuses_nm]


def focus_exposure_matrix(pattern: np.ndarray, dx_nm: float, source_points: np.ndarray,
                          optics: ProjectionOptics, focuses_nm: Sequence[float],
                          doses: Sequence[float], threshold: float, axis: str = "x",
                          tone: str = "dark", mask: Optional[ReflectiveMask] = None) -> np.ndarray:
    """CD(focus, dose) matrix, shape (n_focus, n_dose), constant-threshold resist.

    ``doses`` are relative to nominal (1.0); the feature prints where
    dose * I crosses ``threshold``, i.e. the effective threshold is threshold/dose.
    Each row versus dose / column versus focus gives Bossung curves.
    """
    out = np.full((len(focuses_nm), len(doses)), np.nan)
    for i, f in enumerate(focuses_nm):
        prof = cutline(aerial_image(pattern, dx_nm, source_points, optics, f, mask), axis)
        for j, d in enumerate(doses):
            out[i, j] = measure_cd(prof, dx_nm, threshold / d, tone=tone)
    return out


def process_window(cd: np.ndarray, focuses_nm: Sequence[float], doses: Sequence[float],
                   target_cd_nm: float, tol: float = 0.10, el_min: float = 0.05) -> Dict[str, float]:
    """Exposure latitude and depth of focus from a FEM.

    For every focus, the in-spec dose range (|CD - target| <= tol*target) is
    found; EL = (d_max - d_min)/d_centre.  DOF = focus span over which a common
    dose window of at least ``el_min`` exists (rectangular window search).
    """
    f = np.asarray(focuses_nm, dtype=float)
    d = np.asarray(doses, dtype=float)
    ok = np.abs(np.asarray(cd) - target_cd_nm) <= tol * target_cd_nm
    ok &= np.isfinite(cd)

    def el_of(mask_row):
        idx = np.flatnonzero(mask_row)
        if len(idx) == 0:
            return 0.0
        # longest contiguous run
        best, start = (0, 0), idx[0]
        for a, b in zip(idx, np.append(idx[1:], -10)):
            if b != a + 1:
                if a - start >= best[1] - best[0]:
                    best = (start, a)
                start = b
        lo, hi = d[best[0]], d[best[1]]
        return float((hi - lo) / (0.5 * (hi + lo))) if hi > lo else 0.0

    el = np.array([el_of(r) for r in ok])
    best_i = int(np.argmax(el))
    dof = 0.0
    n = len(f)
    for i in range(n):
        for j in range(i, n):
            common = np.all(ok[i:j + 1], axis=0)
            if el_of(common) >= el_min:
                dof = max(dof, f[j] - f[i])
    return {"el_vs_focus": el, "best_focus_nm": float(f[best_i]),
            "el_best": float(el[best_i]), "dof_nm": float(dof)}


def best_focus(pattern: np.ndarray, dx_nm: float, source_points: np.ndarray,
               optics: ProjectionOptics, focuses_nm: Sequence[float],
               mask: Optional[ReflectiveMask] = None, axis: str = "x") -> float:
    """Focus (nm) of maximum image contrast, refined by a parabola fit.
    With a phase-shifting (thin-mask Kirchhoff) absorber this shows the M3D-like
    pitch-dependent best-focus shift."""
    f = np.asarray(focuses_nm, dtype=float)
    c = np.array([image_contrast(cutline(aerial_image(pattern, dx_nm, source_points, optics, z, mask), axis))
                  for z in f])
    i = int(np.argmax(c))
    if 0 < i < len(f) - 1:
        a, b, _ = np.polyfit(f[i - 1:i + 2], c[i - 1:i + 2], 2)
        if a < 0:
            return float(-b / (2 * a))
    return float(f[i])


def profile_shift_nm(profile_a: np.ndarray, profile_b: np.ndarray, dx_nm: float) -> float:
    """Lateral shift of profile_b relative to profile_a (nm), from the phase of the
    fundamental Fourier harmonic of the periodic cutline."""
    Fa = np.fft.fft(np.asarray(profile_a, dtype=float))[1]
    Fb = np.fft.fft(np.asarray(profile_b, dtype=float))[1]
    L = len(profile_a) * dx_nm
    return float(-np.angle(Fb / Fa) / (2 * np.pi) * L)


def telecentricity_error_mrad(pattern: np.ndarray, dx_nm: float, source_points: np.ndarray,
                              optics: ProjectionOptics, mask: Optional[ReflectiveMask] = None,
                              axis: str = "x", dz_nm: float = 50.0) -> float:
    """Image placement shift per unit defocus, d(x)/d(z) in mrad (nm/um), from
    images at +/- dz.  Non-zero for asymmetric (M3D-shadowed) mask near fields."""
    a = cutline(aerial_image(pattern, dx_nm, source_points, optics, -dz_nm, mask), axis)
    b = cutline(aerial_image(pattern, dx_nm, source_points, optics, +dz_nm, mask), axis)
    return 1e3 * profile_shift_nm(a, b, dx_nm) / (2 * dz_nm)
