"""Illuminator model of an ArF immersion scanner.

Light path modelled here (simplified, textbook level)::

    laser beam -> beam delivery -> pupil shaping (DOE or micro-mirror array)
      -> fly's-eye integrator -> REMA (reticle masking) blades -> condenser
      -> reticle

Two largely independent things come out of the illuminator:

* the **pupil fill** ("source"), an angular intensity distribution described by
  a :class:`~duv.core.SourceMap` in normalised sigma coordinates, and
* the **field illumination**, i.e. the spatial intensity across the slit
  (x, 26 mm) and in the scan direction (y, the ~8 mm slit profile), which after
  scanning yields the dose distribution over the exposure field.

Field quantities in this module are in millimetres at wafer scale (names end
with ``_mm``).  Pupil quantities are in normalised sigma.  2D arrays are
indexed ``[iy, ix]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear

from .core import FIELD_HEIGHT_MM, FIELD_WIDTH_MM, SLIT_HEIGHT_MM, SourceMap

__all__ = [
    "SOURCE_KINDS",
    "POLARIZATIONS",
    "sigma_grid",
    "make_source",
    "freeform_source",
    "MirrorArray",
    "DiffractiveOpticalElement",
    "pupil_metrics",
    "uniformity",
    "FlyEyeIntegrator",
    "slit_profile",
    "scanned_dose_profile",
    "REMABlades",
    "UniformityCorrector",
    "polarization_vectors",
    "polarization_state",
    "degree_of_polarization",
    "intensity_in_preferred_state",
    "laser_beam_profile",
    "Illuminator",
]

SOURCE_KINDS = (
    "conventional",
    "annular",
    "dipole_x",
    "dipole_y",
    "quadrupole",
    "quasar",
    "c_quad",
)
POLARIZATIONS = ("unpolarized", "TE", "TM", "X", "Y")


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def sigma_grid(n: int = 101) -> np.ndarray:
    """1D sigma axis with ``n`` samples spanning [-1, 1]."""
    if n < 3:
        raise ValueError("n must be >= 3")
    return np.linspace(-1.0, 1.0, n)


def _soft_step(d: np.ndarray, width: float) -> np.ndarray:
    """Smooth 0->1 step of a signed distance ``d`` (positive = inside).

    A raised-cosine ramp of total ``width`` centred on d=0, with compact
    support: exactly 0 for d <= -width/2 and exactly 1 for d >= width/2.
    ``width <= 0`` gives a hard edge.
    """
    d = np.asarray(d, dtype=float)
    if width <= 0:
        return (d >= 0).astype(float)
    f = np.clip(0.5 + d / width, 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * f)


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles (rad) to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def uniformity(profile: np.ndarray) -> float:
    """Scanner-style non-uniformity (max - min) / (max + min) of a profile."""
    p = np.asarray(profile, dtype=float)
    mx, mn = p.max(), p.min()
    if mx + mn == 0:
        return 0.0
    return float((mx - mn) / (mx + mn))


# -----------------------------------------------------------------------------
# source (pupil fill) generation
# -----------------------------------------------------------------------------
def make_source(
    kind: str,
    sigma_in: float = 0.0,
    sigma_out: float = 0.9,
    opening_angle_deg: float = 90.0,
    rotation_deg: float = 0.0,
    n: int = 101,
    polarization: str = "unpolarized",
    edge_blur: float = 0.02,
) -> SourceMap:
    """Create a parametric illumination pupil.

    Parameters
    ----------
    kind
        ``"conventional"`` (disk of radius ``sigma_out``), ``"annular"``
        (ring ``sigma_in..sigma_out``), ``"dipole_x"`` / ``"dipole_y"`` (two
        annular-sector poles on the x / y axis), ``"quadrupole"`` (four
        annular-sector poles on the x and y axes), ``"quasar"`` (four
        annular-sector poles on the 45-degree diagonals) or ``"c_quad"``
        (cross-quadrupole: four *circular* poles on the axes, centred at
        (sigma_in+sigma_out)/2 with diameter sigma_out-sigma_in).
    sigma_in, sigma_out
        Inner and outer radius of the ring that the poles are cut from.
        ``sigma_in`` is ignored for ``"conventional"``.
    opening_angle_deg
        Angular width of each pole (dipole / quadrupole / quasar).
    rotation_deg
        Extra counter-clockwise rotation of the pole pattern.
    n
        Number of samples per axis; the grid spans [-1, 1].
    polarization
        One of :data:`POLARIZATIONS`.
    edge_blur
        Width (in sigma) of the raised-cosine edge ramp, mimicking the finite
        resolution of the pupil-shaping optics.  The ramp is centred on the
        nominal edge and has compact support.

    Returns
    -------
    SourceMap with peak intensity 1.
    """
    kind = kind.lower()
    if kind not in SOURCE_KINDS:
        raise ValueError(f"unknown source kind {kind!r}; choose from {SOURCE_KINDS}")
    if polarization not in POLARIZATIONS:
        raise ValueError(f"unknown polarization {polarization!r}")
    if not 0.0 <= sigma_in < sigma_out <= 1.0 and kind != "conventional":
        raise ValueError("require 0 <= sigma_in < sigma_out <= 1")
    if kind == "conventional" and not 0.0 < sigma_out <= 1.0:
        raise ValueError("require 0 < sigma_out <= 1")

    s = sigma_grid(n)
    sx, sy = np.meshgrid(s, s)
    r = np.hypot(sx, sy)
    phi = np.arctan2(sy, sx)
    rot = np.deg2rad(rotation_deg)

    outer = _soft_step(sigma_out - r, edge_blur)
    if kind == "conventional":
        inten = outer
    elif kind == "c_quad":
        rc = 0.5 * (sigma_in + sigma_out)
        rad = 0.5 * (sigma_out - sigma_in)
        inten = np.zeros_like(r)
        for k in range(4):
            a = rot + k * np.pi / 2
            d = np.hypot(sx - rc * np.cos(a), sy - rc * np.sin(a))
            inten = np.maximum(inten, _soft_step(rad - d, edge_blur))
    else:
        ring = outer * _soft_step(r - sigma_in, edge_blur)
        if kind == "annular":
            inten = ring
        else:
            centres = {
                "dipole_x": [0.0, np.pi],
                "dipole_y": [np.pi / 2, -np.pi / 2],
                "quadrupole": [0.0, np.pi / 2, np.pi, -np.pi / 2],
                "quasar": [np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4],
            }[kind]
            half = np.deg2rad(opening_angle_deg) / 2
            ang = np.zeros_like(r)
            for c in centres:
                dphi = np.abs(_wrap_angle(phi - c - rot))
                # convert angular distance to arc length so the blur is in sigma
                ang = np.maximum(ang, _soft_step((half - dphi) * np.maximum(r, 1e-6), edge_blur))
            inten = ring * ang

    inten = np.where(r <= 1.0, inten, 0.0)
    if inten.max() > 0:
        inten = inten / inten.max()
    return SourceMap(sigma=s, intensity=inten, polarization=polarization)


def freeform_source(
    intensity_array: np.ndarray,
    sigma: np.ndarray | None = None,
    polarization: str = "unpolarized",
) -> SourceMap:
    """Wrap an arbitrary (e.g. source-mask-optimised) pupil as a SourceMap.

    ``intensity_array`` must be square ``[iy, ix]``; negative values are
    clipped, points outside the unit pupil are zeroed, and the result is
    normalised to a peak of 1.  If ``sigma`` is omitted the grid is assumed to
    span [-1, 1].
    """
    a = np.asarray(intensity_array, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("intensity_array must be a square 2D array")
    if polarization not in POLARIZATIONS:
        raise ValueError(f"unknown polarization {polarization!r}")
    s = sigma_grid(a.shape[0]) if sigma is None else np.asarray(sigma, dtype=float)
    if s.shape != (a.shape[0],):
        raise ValueError("sigma length must match intensity_array")
    sx, sy = np.meshgrid(s, s)
    a = np.where(np.hypot(sx, sy) <= 1.0, np.clip(a, 0.0, None), 0.0)
    if a.max() <= 0:
        raise ValueError("source has no intensity inside the pupil")
    return SourceMap(sigma=s, intensity=a / a.max(), polarization=polarization)


# -----------------------------------------------------------------------------
# pupil shaping hardware
# -----------------------------------------------------------------------------
class MirrorArray:
    """Programmable micro-mirror array pupil shaper (FlexRay-like).

    The laser beam is split into ``n_mirrors`` beamlets by a lens array; each
    beamlet hits one individually tiltable micro-mirror whose tilt sets where
    the beamlet lands in the pupil.  Each beamlet forms a small spot (Gaussian
    with standard deviation ``spot_sigma``, set by beam divergence and the
    lenslet focal length).  Beamlet powers follow the local laser intensity,
    so a non-uniform beam yields unequal mirror powers.

    Attributes
    ----------
    positions : (n_mirrors, 2) array of pupil targets (sigma_x, sigma_y).
    powers : (n_mirrors,) relative beamlet powers (sum 1).
    """

    def __init__(
        self,
        n_mirrors: int = 4096,
        spot_sigma: float = 0.02,
        max_sigma: float = 1.0,
        powers: np.ndarray | None = None,
        polarization: str = "unpolarized",
    ):
        if n_mirrors < 1:
            raise ValueError("n_mirrors must be positive")
        self.n_mirrors = int(n_mirrors)
        self.spot_sigma = float(spot_sigma)
        self.max_sigma = float(max_sigma)
        self.polarization = polarization
        self.positions = np.zeros((self.n_mirrors, 2))
        self.set_powers(powers)

    def set_powers(self, powers: np.ndarray | None) -> None:
        """Set beamlet powers (e.g. from the laser beam profile); None = equal."""
        if powers is None:
            p = np.ones(self.n_mirrors)
        else:
            p = np.clip(np.asarray(powers, dtype=float).ravel(), 0.0, None)
            if p.size != self.n_mirrors:
                raise ValueError("powers must have one entry per mirror")
        if p.sum() <= 0:
            raise ValueError("total beamlet power must be positive")
        self.powers = p / p.sum()

    @classmethod
    def from_beam(cls, beam_profile: np.ndarray, **kwargs) -> "MirrorArray":
        """Create an array whose mirrors sample a 2D laser ``beam_profile``."""
        b = np.asarray(beam_profile, dtype=float)
        return cls(n_mirrors=b.size, powers=b.ravel(), **kwargs)

    def set_positions(self, positions: np.ndarray) -> None:
        """Point mirrors at explicit pupil positions, clipped to ``max_sigma``."""
        p = np.asarray(positions, dtype=float).reshape(self.n_mirrors, 2)
        r = np.hypot(p[:, 0], p[:, 1])
        scale = np.where(r > self.max_sigma, self.max_sigma / np.maximum(r, 1e-12), 1.0)
        self.positions = p * scale[:, None]

    def to_source(self, n: int = 101) -> SourceMap:
        """Render the pupil fill: deposit beamlet powers and blur by spot size."""
        s = sigma_grid(n)
        ds = s[1] - s[0]
        fx = (self.positions[:, 0] + 1.0) / ds
        fy = (self.positions[:, 1] + 1.0) / ds
        img = np.zeros((n, n))
        # bilinear deposition keeps sub-pixel position information
        ix0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
        iy0 = np.clip(np.floor(fy).astype(int), 0, n - 2)
        tx = np.clip(fx - ix0, 0, 1)
        ty = np.clip(fy - iy0, 0, 1)
        w = self.powers
        np.add.at(img, (iy0, ix0), w * (1 - tx) * (1 - ty))
        np.add.at(img, (iy0, ix0 + 1), w * tx * (1 - ty))
        np.add.at(img, (iy0 + 1, ix0), w * (1 - tx) * ty)
        np.add.at(img, (iy0 + 1, ix0 + 1), w * tx * ty)
        if self.spot_sigma > 0:
            img = gaussian_filter(img, self.spot_sigma / ds, mode="constant")
        sx, sy = np.meshgrid(s, s)
        img = np.where(np.hypot(sx, sy) <= 1.0, img, 0.0)
        return SourceMap(sigma=s, intensity=img / img.max(), polarization=self.polarization)

    def fit_target(self, target: SourceMap, seed: int | None = 0) -> dict:
        """Assign mirrors so the rendered pupil approximates ``target``.

        Mirrors are allocated by inverse-CDF (quantile) sampling of the target
        intensity, weighted by each mirror's power, so every target pixel gets
        beamlet power proportional to its intensity.  Mirror order is shuffled
        first (a real controller is free to choose which mirror serves which
        pupil position).  Positions get a small random sub-pixel offset.

        Returns a dict with ``rms_error`` (normalised RMS difference between
        the rendered pupil and the spot-blurred target) and ``correlation``.
        """
        rng = np.random.default_rng(seed)
        s = target.sigma
        ds = s[1] - s[0]
        t = np.clip(target.intensity, 0, None).ravel()
        cdf = np.cumsum(t)
        cdf /= cdf[-1]
        order = rng.permutation(self.n_mirrors)
        pw = self.powers[order]
        c = np.cumsum(pw) - 0.5 * pw
        idx = np.clip(np.searchsorted(cdf, c), 0, t.size - 1)
        iy, ix = np.unravel_index(idx, target.intensity.shape)
        jitter = rng.uniform(-0.5, 0.5, size=(self.n_mirrors, 2)) * ds
        pos = np.empty((self.n_mirrors, 2))
        pos[order, 0] = s[ix] + jitter[:, 0]
        pos[order, 1] = s[iy] + jitter[:, 1]
        self.set_positions(pos)
        self.polarization = target.polarization

        rendered = self.to_source(len(s)).intensity
        ref = target.intensity
        if self.spot_sigma > 0:
            ref = gaussian_filter(ref, self.spot_sigma / ds, mode="constant")
        ref = ref / ref.max()
        rms = float(np.sqrt(np.mean((rendered - ref) ** 2)) / np.sqrt(np.mean(ref**2)))
        corr = float(np.corrcoef(rendered.ravel(), ref.ravel())[0, 1])
        return {"rms_error": rms, "correlation": corr}


class DiffractiveOpticalElement:
    """Fixed diffractive optical element (DOE) pupil shaper.

    A DOE is designed for one pupil shape (here any :func:`make_source` kind).
    Its far field is the design pattern convolved with the laser angular
    divergence, plus a small undiffracted zero-order spot at the pupil centre
    and a diffraction efficiency below 1 (lost light is not in the pupil).
    """

    def __init__(
        self,
        kind: str,
        divergence_sigma: float = 0.015,
        zero_order: float = 0.005,
        efficiency: float = 0.9,
        **source_kwargs,
    ):
        self.kind = kind
        self.divergence_sigma = float(divergence_sigma)
        self.zero_order = float(zero_order)
        self.efficiency = float(efficiency)
        self.source_kwargs = source_kwargs

    def to_source(self, n: int = 101) -> SourceMap:
        """Far-field pupil fill produced by the DOE."""
        kw = dict(self.source_kwargs)
        kw["n"] = n
        src = make_source(self.kind, **kw)
        s = src.sigma
        ds = s[1] - s[0]
        img = src.intensity / src.intensity.sum() * (1.0 - self.zero_order)
        c = n // 2
        img[c, c] += self.zero_order
        if self.divergence_sigma > 0:
            img = gaussian_filter(img, self.divergence_sigma / ds, mode="constant")
        sx, sy = np.meshgrid(s, s)
        img = np.where(np.hypot(sx, sy) <= 1.0, img, 0.0)
        return SourceMap(sigma=s, intensity=img / img.max(), polarization=src.polarization)


# -----------------------------------------------------------------------------
# pupil metrics
# -----------------------------------------------------------------------------
def pupil_metrics(source: SourceMap, fill_threshold: float = 0.1) -> dict:
    """Standard pupil-fill metrics.

    Returns a dict with

    * ``sigma_center`` - intensity-weighted mean radius,
    * ``sigma_rms`` - intensity-weighted RMS radius,
    * ``ellipticity`` - (E_h - E_v)/(E_h + E_v), where E_h is the energy in the
      two 90-degree sectors around the x axis and E_v around the y axis
      (0 for a four-fold symmetric source, +1 for an ideal x-dipole),
    * ``pole_balance`` - (max - min)/(max + min) of the energies in the four
      quadrants (0 = perfectly balanced),
    * ``telecentricity_x``, ``telecentricity_y`` and ``telecentricity`` - the
      intensity centroid (sigma units) and its magnitude; a non-zero centroid
      causes pattern shift through focus,
    * ``fill_fraction`` - fraction of the unit pupil area above
      ``fill_threshold`` x peak,
    * ``total`` - summed intensity.
    """
    s = source.sigma
    sx, sy = np.meshgrid(s, s)
    r = np.hypot(sx, sy)
    I = np.clip(source.intensity, 0, None)
    tot = I.sum()
    if tot <= 0:
        raise ValueError("empty source")
    w = I / tot
    cx = float((w * sx).sum())
    cy = float((w * sy).sum())
    horiz = np.abs(sx) > np.abs(sy)
    vert = np.abs(sy) > np.abs(sx)
    diag = np.isclose(np.abs(sx), np.abs(sy))
    eh = w[horiz].sum() + 0.5 * w[diag].sum()
    ev = w[vert].sum() + 0.5 * w[diag].sum()
    # quadrant energies; points on an axis are split between neighbours
    qx = np.where(sx > 0, 1.0, np.where(sx < 0, 0.0, 0.5))
    qy = np.where(sy > 0, 1.0, np.where(sy < 0, 0.0, 0.5))
    quads = np.array(
        [
            (w * qx * qy).sum(),
            (w * (1 - qx) * qy).sum(),
            (w * (1 - qx) * (1 - qy)).sum(),
            (w * qx * (1 - qy)).sum(),
        ]
    )
    inside = r <= 1.0
    fill = float(((I > fill_threshold * I.max()) & inside).sum() / inside.sum())
    return {
        "sigma_center": float((w * r).sum()),
        "sigma_rms": float(np.sqrt((w * r**2).sum())),
        "ellipticity": float((eh - ev) / (eh + ev)) if eh + ev > 0 else 0.0,
        "pole_balance": float((quads.max() - quads.min()) / (quads.max() + quads.min())),
        "quadrant_energy": quads,
        "telecentricity_x": cx,
        "telecentricity_y": cy,
        "telecentricity": float(np.hypot(cx, cy)),
        "fill_fraction": fill,
        "total": float(tot),
    }


# -----------------------------------------------------------------------------
# polarization
# -----------------------------------------------------------------------------
def polarization_vectors(source: SourceMap) -> tuple[np.ndarray, np.ndarray] | None:
    """Unit Jones-vector field (ex, ey) of each pupil point, indexed [iy, ix].

    'TE' is azimuthal (perpendicular to the radius, i.e. s-polarised in the
    image plane, best for high-NA contrast), 'TM' radial, 'X'/'Y' linear.
    Returns ``None`` for unpolarized light.  At the pupil centre the TE/TM
    direction is undefined and is set to x.
    """
    s = source.sigma
    sx, sy = np.meshgrid(s, s)
    r = np.hypot(sx, sy)
    safe = np.where(r > 0, r, 1.0)
    pol = source.polarization
    if pol == "unpolarized":
        return None
    if pol == "X":
        return np.ones_like(sx), np.zeros_like(sx)
    if pol == "Y":
        return np.zeros_like(sx), np.ones_like(sx)
    if pol == "TE":
        ex, ey = -sy / safe, sx / safe
    elif pol == "TM":
        ex, ey = sx / safe, sy / safe
    else:
        raise ValueError(f"unknown polarization {pol!r}")
    ex = np.where(r > 0, ex, 1.0)
    ey = np.where(r > 0, ey, 0.0)
    return ex, ey


def polarization_state(source: SourceMap, purity: float = 1.0) -> dict:
    """Stokes-parameter maps of the source.

    ``purity`` is the fraction of light in the nominal polarization state
    (the rest is unpolarized), e.g. 0.95 for a real polariser.  Returns a dict
    with Stokes maps ``S0..S3`` (linear states only, so S3 = 0), the local
    degree of polarization ``dop_map``, and the intensity-weighted scalar
    ``dop``.
    """
    I = np.clip(source.intensity, 0, None)
    vec = polarization_vectors(source)
    if vec is None:
        purity = 0.0
        ex, ey = np.ones_like(I), np.zeros_like(I)
    else:
        ex, ey = vec
    s1 = purity * I * (ex**2 - ey**2)
    s2 = purity * I * 2 * ex * ey
    s3 = np.zeros_like(I)
    with np.errstate(invalid="ignore", divide="ignore"):
        dop_map = np.where(I > 0, np.sqrt(s1**2 + s2**2 + s3**2) / np.where(I > 0, I, 1), 0.0)
    dop = float((dop_map * I).sum() / I.sum()) if I.sum() > 0 else 0.0
    return {"S0": I, "S1": s1, "S2": s2, "S3": s3, "dop_map": dop_map, "dop": dop}


def degree_of_polarization(source: SourceMap, purity: float = 1.0) -> float:
    """Intensity-weighted degree of polarization (0 unpolarized ... 1 fully)."""
    return polarization_state(source, purity)["dop"]


def intensity_in_preferred_state(
    source: SourceMap, preferred: str = "TE", purity: float = 1.0
) -> float:
    """IPS: fraction of source intensity in the ``preferred`` polarization.

    Unpolarized light gives 0.5.  E.g. X-polarised light on a y-dipole is
    pure TE, so IPS(TE) = 1.
    """
    I = np.clip(source.intensity, 0, None)
    vec = polarization_vectors(source)
    pref = polarization_vectors(SourceMap(source.sigma, I, preferred))
    if vec is None or pref is None:
        return 0.5
    cos2 = (vec[0] * pref[0] + vec[1] * pref[1]) ** 2
    local = purity * cos2 + (1 - purity) * 0.5
    return float((local * I).sum() / I.sum())


# -----------------------------------------------------------------------------
# field formation: integrator, slit, scanning, REMA, uniformity correction
# -----------------------------------------------------------------------------
class FlyEyeIntegrator:
    """Fly's-eye (lenslet array) integrator.

    The incoming beam is divided into ``n_lenslets`` sub-apertures per axis;
    the condenser images every sub-aperture onto the full field, so the field
    profile is the *average* of the beam segments, each stretched to the field
    size.  Slow beam non-uniformities (tilt, curvature) are reduced roughly as
    1/n_lenslets.

    Interference note: the excimer laser is partially coherent, so the
    superposed lenslet images interfere and create speckle / fine interference
    fringes.  These are suppressed because (a) the beam contains many mutually
    incoherent transverse modes (high etendue), and (b) many pulses with
    different speckle realisations are integrated while scanning.  See
    :meth:`speckle_contrast`.
    """

    def __init__(self, n_lenslets: int = 16, n_field: int = 201):
        if n_lenslets < 1:
            raise ValueError("n_lenslets must be >= 1")
        self.n_lenslets = int(n_lenslets)
        self.n_field = int(n_field)

    def _homogenize_1d(self, profile: np.ndarray) -> np.ndarray:
        p = np.asarray(profile, dtype=float)
        m = p.size
        xin = (np.arange(m) + 0.5) / m
        u = (np.arange(self.n_field) + 0.5) / self.n_field
        k = np.arange(self.n_lenslets)[:, None]
        coords = (k + u[None, :]) / self.n_lenslets
        return np.interp(coords, xin, p).mean(axis=0)

    def homogenize(self, input_profile: np.ndarray) -> np.ndarray:
        """Field intensity produced from a 1D or 2D (``[iy, ix]``) beam profile.

        2D input is treated as crossed cylindrical arrays (separable along
        both axes).  The output is normalised to a mean of 1.
        """
        p = np.asarray(input_profile, dtype=float)
        if p.ndim == 1:
            out = self._homogenize_1d(p)
        elif p.ndim == 2:
            tmp = np.array([self._homogenize_1d(row) for row in p])
            out = np.array([self._homogenize_1d(col) for col in tmp.T]).T
        else:
            raise ValueError("input_profile must be 1D or 2D")
        return out / out.mean()

    def field_uniformity(self, input_profile: np.ndarray) -> dict:
        """Compare non-uniformity of the beam and of the homogenised field.

        Returns dict with ``input_uniformity``, ``output_uniformity`` (both
        (max-min)/(max+min)), ``improvement`` (ratio) and ``field_profile``.
        """
        p = np.asarray(input_profile, dtype=float)
        out = self.homogenize(p)
        u_in = uniformity(p)
        u_out = uniformity(out)
        return {
            "input_uniformity": u_in,
            "output_uniformity": u_out,
            "improvement": u_in / u_out if u_out > 0 else np.inf,
            "field_profile": out,
        }

    def speckle_contrast(
        self,
        beam_size_mm: float = 20.0,
        coherence_length_mm: float = 0.5,
        n_pulses: int = 50,
    ) -> float:
        """Estimated residual speckle / interference contrast.

        The number of mutually incoherent modes reaching each field point is
        limited both by the lenslet count and by the number of spatial
        coherence cells across the beam; pulses add independent realisations.
        Contrast ~ 1/sqrt(M_modes * n_pulses).
        """
        cells = (beam_size_mm / coherence_length_mm) ** 2
        modes = min(self.n_lenslets**2, cells)
        return float(1.0 / np.sqrt(max(modes, 1.0) * max(n_pulses, 1)))


def slit_profile(
    y_mm: np.ndarray,
    slit_height_mm: float = SLIT_HEIGHT_MM,
    edge_width_mm: float = 1.0,
    smooth: bool = True,
) -> np.ndarray:
    """Static slit intensity profile in the scan direction (peak 1).

    A trapezoid whose full width at half maximum is ``slit_height_mm`` and whose
    10%-90%-like ramps are ``edge_width_mm`` wide.  With ``smooth=True`` the
    ramps are raised-cosine (no kinks), otherwise linear.  A finite edge width
    (deliberate defocus of the slit image) suppresses pulse-quantisation dose
    ripple during scanning.
    """
    y = np.asarray(y_mm, dtype=float)
    d = slit_height_mm / 2 - np.abs(y)
    if edge_width_mm <= 0:
        return (d >= 0).astype(float)
    if smooth:
        return _soft_step(d, edge_width_mm)
    return np.clip(0.5 + d / edge_width_mm, 0.0, 1.0)


def scanned_dose_profile(
    y_mm: np.ndarray,
    scan_start_mm: float | None = None,
    scan_end_mm: float | None = None,
    scan_speed_mm_s: float = 700.0,
    rep_rate_hz: float = 6000.0,
    slit_height_mm: float = SLIT_HEIGHT_MM,
    edge_width_mm: float = 1.0,
    pulse_energy: float = 1.0,
    energy_jitter: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Integrated dose along the scan direction for a pulsed, scanned exposure.

    The slit centre moves from ``scan_start_mm`` to ``scan_end_mm`` (defaults:
    from the first to the last ``y_mm`` sample) at ``scan_speed_mm_s``, firing a
    pulse every 1/``rep_rate_hz``.  Each pulse deposits ``pulse_energy`` x
    :func:`slit_profile`, with optional relative Gaussian ``energy_jitter``.

    Returns dose vs ``y_mm`` normalised so that the ideal continuous-scan dose
    is ``pulse_energy`` (i.e. divided by the number of pulses per point,
    ``slit_height * rep_rate / speed``).
    """
    y = np.asarray(y_mm, dtype=float)
    y0 = y.min() if scan_start_mm is None else scan_start_mm
    y1 = y.max() if scan_end_mm is None else scan_end_mm
    step = scan_speed_mm_s / rep_rate_hz
    if step <= 0:
        raise ValueError("scan speed and rep rate must be positive")
    centres = np.arange(y0, y1 + 0.5 * step, step)
    e = np.full(centres.size, pulse_energy, dtype=float)
    if energy_jitter > 0:
        rng = np.random.default_rng(seed)
        e *= 1.0 + energy_jitter * rng.standard_normal(centres.size)
    dose = np.zeros_like(y)
    chunk = 256
    for i in range(0, centres.size, chunk):
        c = centres[i : i + chunk]
        dose += (e[i : i + chunk, None] * slit_profile(y[None, :] - c[:, None], slit_height_mm, edge_width_mm)).sum(axis=0)
    pulses_per_point = slit_height_mm / step
    return dose / pulses_per_point


@dataclass
class REMABlades:
    """Reticle-masking (REMA) blades, in wafer-scale mm.

    The x blades are static and set the field width.  The y blades move
    synchronously with the scan: they open as the slit enters the field and
    close as it leaves, so no light reaches the wafer outside
    ``[y_min, y_max]``.  The blades are imaged onto the reticle by the REMA
    lens with finite sharpness ``edge_width_mm``.  ``scan_pos`` is the slit
    centre position in field coordinates (y).
    """

    x_min: float = -FIELD_WIDTH_MM / 2
    x_max: float = FIELD_WIDTH_MM / 2
    y_min: float = -FIELD_HEIGHT_MM / 2
    y_max: float = FIELD_HEIGHT_MM / 2
    slit_height_mm: float = SLIT_HEIGHT_MM
    slit_edge_mm: float = 1.0
    edge_width_mm: float = 0.05

    def blade_positions(self, scan_pos: float) -> tuple[float, float]:
        """Positions (lower, upper) of the y blades for slit centre ``scan_pos``."""
        half = self.slit_height_mm / 2 + self.slit_edge_mm
        lo = min(max(scan_pos - half, self.y_min), scan_pos + half)
        hi = max(min(scan_pos + half, self.y_max), scan_pos - half)
        return lo, hi

    def illuminated_region(self, x_mm: np.ndarray, y_mm: np.ndarray, scan_pos: float = 0.0) -> np.ndarray:
        """Instantaneous illumination ``[iy, ix]`` on 1D axes ``x_mm``, ``y_mm``."""
        x = np.asarray(x_mm, dtype=float)
        y = np.asarray(y_mm, dtype=float)
        bx = _soft_step(x - self.x_min, self.edge_width_mm) * _soft_step(self.x_max - x, self.edge_width_mm)
        lo, hi = self.blade_positions(scan_pos)
        by = _soft_step(y - lo, self.edge_width_mm) * _soft_step(hi - y, self.edge_width_mm)
        sy = slit_profile(y - scan_pos, self.slit_height_mm, self.slit_edge_mm)
        return (by * sy)[:, None] * bx[None, :]

    def exposed_dose_map(self, x_mm: np.ndarray, y_mm: np.ndarray, n_steps: int = 400) -> np.ndarray:
        """Dose map ``[iy, ix]`` integrated over a full scan (interior ~1)."""
        half = self.slit_height_mm / 2 + self.slit_edge_mm
        pos = np.linspace(self.y_min - half, self.y_max + half, n_steps)
        acc = np.zeros((np.size(y_mm), np.size(x_mm)))
        for p in pos:
            acc += self.illuminated_region(x_mm, y_mm, p)
        dp = pos[1] - pos[0]
        return acc * dp / self.slit_height_mm


class UniformityCorrector:
    """Finger-array slit uniformity corrector (UNICOM-like).

    ``n_fingers`` fingers with pitch ``field_width_mm / n_fingers`` can each be
    pushed into the slit edge by up to ``max_insertion_mm``.  Since the scanned
    dose is proportional to the local effective slit height, inserting finger
    k by d_k reduces the dose near its x position by a factor
    1 - d_k * phi_k(x) / slit_height, with phi_k a smooth (Gaussian) influence
    function of width ``influence_width`` x pitch (the fingers sit slightly
    out of focus).
    """

    def __init__(
        self,
        n_fingers: int = 28,
        field_width_mm: float = FIELD_WIDTH_MM,
        slit_height_mm: float = SLIT_HEIGHT_MM,
        max_insertion_mm: float = 1.0,
        influence_width: float = 0.7,
    ):
        self.n_fingers = int(n_fingers)
        self.field_width_mm = float(field_width_mm)
        self.slit_height_mm = float(slit_height_mm)
        self.max_insertion_mm = float(max_insertion_mm)
        self.influence_width = float(influence_width)
        pitch = self.field_width_mm / self.n_fingers
        self.pitch = pitch
        self.finger_x_mm = -self.field_width_mm / 2 + pitch * (np.arange(self.n_fingers) + 0.5)
        self.positions_mm = np.zeros(self.n_fingers)

    def influence(self, x_mm: np.ndarray) -> np.ndarray:
        """Influence matrix ``[ix, finger]`` (normalised so the sum is ~1)."""
        x = np.asarray(x_mm, dtype=float)
        w = self.influence_width * self.pitch
        g = np.exp(-0.5 * ((x[:, None] - self.finger_x_mm[None, :]) / w) ** 2)
        return g / (np.sqrt(2 * np.pi) * w / self.pitch)

    def transmission(self, x_mm: np.ndarray, positions_mm: np.ndarray | None = None) -> np.ndarray:
        """Relative dose factor along x for the given finger insertions."""
        d = self.positions_mm if positions_mm is None else np.asarray(positions_mm, dtype=float)
        return 1.0 - self.influence(x_mm) @ d / self.slit_height_mm

    def correct(self, x_profile: np.ndarray, x_mm: np.ndarray | None = None) -> dict:
        """Choose finger insertions that flatten an integrated dose profile.

        Solves the bounded linear least-squares problem
        P(x) * (1 - A d / H) ~= target with 0 <= d <= max_insertion, where
        the target is the minimum of P (fingers can only remove light).

        Returns dict with ``finger_positions_mm``, ``corrected_profile``,
        ``uniformity_before``, ``uniformity_after`` and ``light_loss``
        (mean fractional dose removed).  The finger positions are stored.
        """
        p = np.asarray(x_profile, dtype=float)
        if x_mm is None:
            x_mm = np.linspace(-self.field_width_mm / 2, self.field_width_mm / 2, p.size)
        x = np.asarray(x_mm, dtype=float)
        A = self.influence(x)
        target = p.min()
        M = p[:, None] * A / self.slit_height_mm
        res = lsq_linear(M, p - target, bounds=(0.0, self.max_insertion_mm), method="bvls")
        self.positions_mm = res.x
        corrected = p * self.transmission(x)
        return {
            "finger_positions_mm": res.x.copy(),
            "corrected_profile": corrected,
            "uniformity_before": uniformity(p),
            "uniformity_after": uniformity(corrected),
            "light_loss": float(1.0 - corrected.mean() / p.mean()),
        }


# -----------------------------------------------------------------------------
# laser beam and the complete illuminator
# -----------------------------------------------------------------------------
def laser_beam_profile(
    n: int = 32,
    tilt: float = 0.15,
    gaussian_width: float = 0.6,
    noise: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Synthetic excimer beam cross-section ``[iy, ix]`` on an n x n grid.

    Excimer beams are roughly flat-top in one axis and Gaussian-like in the
    other; ``tilt`` adds a linear intensity slope in x (beam pointing /
    discharge asymmetry) and ``noise`` relative random structure.
    """
    u = np.linspace(-1, 1, n)
    ux, uy = np.meshgrid(u, u)
    flat = _soft_step(0.9 - np.abs(ux), 0.2)
    gauss = np.exp(-0.5 * (uy / gaussian_width) ** 2)
    b = flat * gauss * (1.0 + tilt * ux) + 1e-3
    if noise > 0:
        rng = np.random.default_rng(seed)
        b *= 1.0 + noise * rng.standard_normal(b.shape)
    return np.clip(b, 0, None)


@dataclass
class Illuminator:
    """Complete illuminator: pupil shaper + integrator + slit + REMA + UNICOM.

    Call :meth:`run` with a laser beam profile to obtain the pupil
    :class:`SourceMap` and a summary of pupil and field metrics.
    """

    kind: str = "annular"
    sigma_in: float = 0.6
    sigma_out: float = 0.9
    opening_angle_deg: float = 90.0
    rotation_deg: float = 0.0
    polarization: str = "unpolarized"
    edge_blur: float = 0.02
    shaper: str = "mirror_array"  # or "doe" or "ideal"
    spot_sigma: float = 0.02
    n_lenslets: int = 16
    slit_height_mm: float = SLIT_HEIGHT_MM
    slit_edge_mm: float = 1.0
    use_uniformity_correction: bool = True
    n_fingers: int = 28
    rema: REMABlades = field(default_factory=REMABlades)
    target: SourceMap | None = None  # freeform target overrides parametric kind

    def target_source(self, n: int = 101) -> SourceMap:
        """The designed (ideal) pupil."""
        if self.target is not None:
            return self.target
        return make_source(
            self.kind,
            sigma_in=self.sigma_in,
            sigma_out=self.sigma_out,
            opening_angle_deg=self.opening_angle_deg,
            rotation_deg=self.rotation_deg,
            n=n,
            polarization=self.polarization,
            edge_blur=self.edge_blur,
        )

    def source(self, beam_profile: np.ndarray | None = None, n: int = 101) -> SourceMap:
        """Pupil actually produced by the chosen pupil shaper."""
        target = self.target_source(n)
        if self.shaper == "ideal":
            return target
        if self.shaper == "doe":
            if self.target is not None:
                raise ValueError("a DOE cannot realise a freeform target here; use mirror_array")
            return DiffractiveOpticalElement(
                self.kind,
                divergence_sigma=self.spot_sigma,
                sigma_in=self.sigma_in,
                sigma_out=self.sigma_out,
                opening_angle_deg=self.opening_angle_deg,
                rotation_deg=self.rotation_deg,
                polarization=self.polarization,
                edge_blur=self.edge_blur,
            ).to_source(n)
        if self.shaper == "mirror_array":
            beam = laser_beam_profile() if beam_profile is None else beam_profile
            ma = MirrorArray.from_beam(beam, spot_sigma=self.spot_sigma, polarization=self.polarization)
            ma.fit_target(target)
            return ma.to_source(n)
        raise ValueError(f"unknown shaper {self.shaper!r}")

    def run(self, beam_profile: np.ndarray | None = None, n: int = 101, n_x: int = 131) -> tuple[SourceMap, dict]:
        """Simulate the illuminator for a laser ``beam_profile`` ``[iy, ix]``.

        Returns ``(source, summary)``; summary keys: ``pupil`` (see
        :func:`pupil_metrics`), ``fit`` (pupil-shaper fidelity),
        ``dop``, ``beam_uniformity``, ``flyeye_uniformity``,
        ``corrected_uniformity``, ``finger_positions_mm``, ``light_loss``,
        ``scan_dose_ripple`` and ``speckle_contrast``.
        """
        beam = laser_beam_profile() if beam_profile is None else np.asarray(beam_profile, dtype=float)
        src = self.source(beam, n)
        target = self.target_source(n)
        ds = src.sigma[1] - src.sigma[0]
        ref = gaussian_filter(target.intensity, self.spot_sigma / ds, mode="constant") if self.shaper != "ideal" else target.intensity
        ref = ref / ref.max()
        fit = {
            "rms_error": float(np.sqrt(np.mean((src.intensity - ref) ** 2)) / np.sqrt(np.mean(ref**2))),
            "correlation": float(np.corrcoef(src.intensity.ravel(), ref.ravel())[0, 1]),
        }

        # field along the slit (x): fly-eye homogenisation of the beam x-profile
        fe = FlyEyeIntegrator(self.n_lenslets, n_field=n_x)
        beam_x = beam.sum(axis=0)
        fe_res = fe.field_uniformity(beam_x)
        x_mm = np.linspace(-FIELD_WIDTH_MM / 2, FIELD_WIDTH_MM / 2, n_x)
        x_profile = fe_res["field_profile"]
        if self.use_uniformity_correction:
            uc = UniformityCorrector(self.n_fingers, slit_height_mm=self.slit_height_mm)
            corr = uc.correct(x_profile, x_mm)
        else:
            corr = {
                "finger_positions_mm": np.zeros(0),
                "corrected_profile": x_profile,
                "uniformity_after": uniformity(x_profile),
                "light_loss": 0.0,
            }

        # scan direction: pulse-quantisation ripple in the field interior
        y = np.linspace(self.rema.y_min, self.rema.y_max, 331)
        dose_y = scanned_dose_profile(
            y,
            self.rema.y_min - self.slit_height_mm,
            self.rema.y_max + self.slit_height_mm,
            slit_height_mm=self.slit_height_mm,
            edge_width_mm=self.slit_edge_mm,
        )
        ripple = uniformity(dose_y)

        summary = {
            "pupil": pupil_metrics(src),
            "fit": fit,
            "dop": degree_of_polarization(src),
            "beam_uniformity": uniformity(beam_x),
            "flyeye_uniformity": fe_res["output_uniformity"],
            "corrected_uniformity": corr["uniformity_after"],
            "finger_positions_mm": corr["finger_positions_mm"],
            "light_loss": corr["light_loss"],
            "x_mm": x_mm,
            "x_profile": corr["corrected_profile"],
            "scan_dose_ripple": ripple,
            "speckle_contrast": fe.speckle_contrast(),
        }
        return src, summary
