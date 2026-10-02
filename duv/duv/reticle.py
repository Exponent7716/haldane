"""Reticle (photomask), pellicle and reticle-stage models.

Conventions (see :mod:`duv.core`):
    * All lateral layout dimensions are in nm at **wafer scale**.  The physical
      reticle is ``REDUCTION`` (4x) larger; use :func:`wafer_to_reticle` /
      :func:`reticle_to_wafer` to convert.
    * A *pattern* is a float array on a :class:`~duv.core.Grid`, indexed
      ``[iy, ix]``, with values in [0, 1]: ``1`` means absorber (chrome for a
      binary mask, MoSi for an attenuated PSM) is drawn on that pixel, ``0``
      means clear quartz.  Edges are *area sampled* (anti-aliased): a pixel
      half covered by absorber has value 0.5, so CDs that are not a multiple of
      the pixel size are represented exactly in the mean (zeroth order).
    * The grid is periodic, so all layouts wrap around at ``grid.period``.

The optical model behind :func:`make_mask` is the thin-mask (Kirchhoff)
approximation; :func:`mask_3d_correction` adds a simple boundary-layer
correction for the finite absorber thickness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .core import NA_MAX_IMMERSION, REDUCTION, SLIT_HEIGHT_MM, WAVELENGTH_ARF, Grid, Mask

__all__ = [
    "reticle_to_wafer",
    "wafer_to_reticle",
    "rectangles",
    "line_space",
    "isolated_line",
    "contact_array",
    "line_end",
    "elbow",
    "make_mask",
    "apply_bias",
    "simple_opc",
    "add_srafs",
    "mask_3d_correction",
    "Pellicle",
    "mask_cd_error",
    "meef",
    "ReticleStage",
    "ReticleHeating",
]

FUSED_SILICA_CTE = 0.5e-6  # 1/K
FUSED_SILICA_DENSITY = 2200.0  # kg/m^3
FUSED_SILICA_CP = 740.0  # J/(kg K)


# ---------------------------------------------------------------------------
# Scale conversions
# ---------------------------------------------------------------------------
def reticle_to_wafer(x_nm, reduction: float = REDUCTION):
    """Convert a reticle-scale length (nm) to wafer scale (divide by 4)."""
    return np.asarray(x_nm, dtype=float) / reduction if np.ndim(x_nm) else float(x_nm) / reduction


def wafer_to_reticle(x_nm, reduction: float = REDUCTION):
    """Convert a wafer-scale length (nm) to reticle scale (multiply by 4)."""
    return np.asarray(x_nm, dtype=float) * reduction if np.ndim(x_nm) else float(x_nm) * reduction


# ---------------------------------------------------------------------------
# Layout primitives (area-sampled)
# ---------------------------------------------------------------------------
def _coverage_1d(xc: np.ndarray, pixel: float, period: float, a: float, b: float) -> np.ndarray:
    """Fraction of each pixel [xc - p/2, xc + p/2] covered by the periodic interval [a, b]."""
    if b <= a:
        return np.zeros_like(xc)
    if b - a >= period:
        return np.ones_like(xc)
    lo_pix = xc - pixel / 2
    hi_pix = xc + pixel / 2
    total = np.zeros_like(xc)
    kmin = int(math.floor((lo_pix.min() - b) / period)) - 1
    kmax = int(math.ceil((hi_pix.max() - a) / period)) + 1
    for k in range(kmin, kmax + 1):
        lo = np.maximum(a + k * period, lo_pix)
        hi = np.minimum(b + k * period, hi_pix)
        total += np.clip(hi - lo, 0.0, None)
    return np.clip(total / pixel, 0.0, 1.0)


def rectangles(grid: Grid, rects) -> np.ndarray:
    """Area-sampled pattern of axis-aligned rectangles.

    Parameters
    ----------
    grid : Grid
    rects : iterable of (x0, y0, x1, y1)
        Rectangle corners in nm (wafer scale, grid coordinates centred on 0).
        Rectangles wrap periodically.  Overlaps are clipped to 1 (adjacent,
        non-overlapping rectangles are combined exactly).

    Returns
    -------
    ndarray of float, shape (n, n), values in [0, 1].
    """
    x = grid.x.astype(float)
    out = np.zeros((grid.n, grid.n))
    for r in rects:
        x0, y0, x1, y1 = (float(v) for v in r)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        cx = _coverage_1d(x, grid.pixel, grid.period, x0, x1)
        cy = _coverage_1d(x, grid.pixel, grid.period, y0, y1)
        out += np.outer(cy, cx)
    return np.clip(out, 0.0, 1.0)


def _check_pitch(grid: Grid, pitch: float) -> int:
    """Number of periods of ``pitch`` in the grid; raise if pitch does not divide the period."""
    if pitch <= 0:
        raise ValueError("pitch must be positive")
    ratio = grid.period / pitch
    nper = int(round(ratio))
    if nper < 1 or abs(ratio - nper) > 1e-6 * max(1.0, ratio):
        raise ValueError(
            f"pitch {pitch} nm does not divide the grid period {grid.period} nm "
            "(periodic simulation); choose grid.n*grid.pixel = k*pitch"
        )
    return nper


def _apply_tone(p: np.ndarray, tone: str) -> np.ndarray:
    if tone == "dark":
        return p
    if tone == "clear":
        return 1.0 - p
    raise ValueError("tone must be 'dark' or 'clear'")


def line_space(grid: Grid, cd: float, pitch: float, orientation: str = "vertical",
               tone: str = "dark", offset: float = 0.0) -> np.ndarray:
    """Periodic lines and spaces.

    Parameters
    ----------
    cd : line width in nm (wafer scale).
    pitch : line pitch in nm; must divide ``grid.period`` exactly (ValueError otherwise).
    orientation : 'vertical' (lines along y, varying in x) or 'horizontal'.
    tone : 'dark' -> absorber lines on a clear field (mean = cd/pitch);
        'clear' -> clear lines (trenches) of width ``cd`` in an absorber field.
    offset : shift of the line centres (nm); a line is centred at x = offset.
    """
    if not 0 <= cd <= pitch:
        raise ValueError("need 0 <= cd <= pitch")
    nper = _check_pitch(grid, pitch)
    half = grid.period / 2
    rects = []
    for k in range(nper):
        c = offset + k * pitch
        if orientation == "vertical":
            rects.append((c - cd / 2, -half, c + cd / 2, half))
        elif orientation == "horizontal":
            rects.append((-half, c - cd / 2, half, c + cd / 2))
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
    return _apply_tone(rectangles(grid, rects), tone)


def isolated_line(grid: Grid, cd: float, orientation: str = "vertical", tone: str = "dark",
                  center: float = 0.0) -> np.ndarray:
    """A single line of width ``cd`` (nm) centred at ``center`` (one per grid period)."""
    return line_space(grid, cd, grid.period, orientation=orientation, tone=tone, offset=center)


def contact_array(grid: Grid, size: float, pitch: float, tone: str = "clear",
                  size_y: float | None = None, pitch_y: float | None = None) -> np.ndarray:
    """Square (or rectangular) contact array.

    ``tone='clear'`` (default) gives clear contact holes in an absorber (dark)
    field, i.e. the returned pattern is 1 everywhere except the holes.
    ``tone='dark'`` gives absorber posts (pillars) on a clear field.
    Pitches must divide ``grid.period``.
    """
    sy = size if size_y is None else size_y
    py = pitch if pitch_y is None else pitch_y
    nx = _check_pitch(grid, pitch)
    ny = _check_pitch(grid, py)
    x = grid.x.astype(float)
    cx = sum(_coverage_1d(x, grid.pixel, grid.period, i * pitch - size / 2, i * pitch + size / 2)
             for i in range(nx))
    cy = sum(_coverage_1d(x, grid.pixel, grid.period, j * py - sy / 2, j * py + sy / 2)
             for j in range(ny))
    posts = np.clip(np.outer(np.clip(cy, 0, 1), np.clip(cx, 0, 1)), 0, 1)
    return _apply_tone(posts, tone)


def line_end(grid: Grid, cd: float, gap: float, length: float | None = None) -> np.ndarray:
    """Two collinear vertical absorber lines whose ends face each other across ``gap`` (nm)."""
    length = grid.period / 2 - gap / 2 if length is None else length
    return rectangles(grid, [(-cd / 2, gap / 2, cd / 2, gap / 2 + length),
                             (-cd / 2, -gap / 2 - length, cd / 2, -gap / 2)])


def elbow(grid: Grid, cd: float, arm: float) -> np.ndarray:
    """L-shaped absorber line ('elbow') of width ``cd`` with arms of length ``arm`` (nm)."""
    return rectangles(grid, [(-cd / 2, -cd / 2, -cd / 2 + arm, cd / 2),
                             (-cd / 2, cd / 2, cd / 2, -cd / 2 + arm)])


# ---------------------------------------------------------------------------
# Mask technologies
# ---------------------------------------------------------------------------
def _periodic_labels(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected-component labels (4-connectivity) with periodic boundary merge."""
    lab, n = ndimage.label(binary)
    if n == 0:
        return lab, 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a_arr, b_arr in ((lab[:, 0], lab[:, -1]), (lab[0, :], lab[-1, :])):
        for a, b in zip(a_arr, b_arr):
            if a and b:
                ra, rb = find(int(a)), find(int(b))
                if ra != rb:
                    parent[rb] = ra
    roots = sorted({find(i) for i in range(1, n + 1)})
    remap = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        remap[i] = roots.index(find(i)) + 1
    return remap[lab], len(roots)


def _alt_phase_map(clear: np.ndarray, grid: Grid, phase_rad: float, axis: str) -> np.ndarray:
    """Alternate 0/phase across neighbouring clear regions (for alternating PSM)."""
    lab, n = _periodic_labels(clear > 0.5)
    if n == 0:
        return np.zeros_like(clear)
    if axis == "auto":
        axis = "x" if np.var(clear.mean(axis=0)) >= np.var(clear.mean(axis=1)) else "y"
    X, Y = grid.mesh()
    c = X if axis == "x" else Y
    ang = 2 * np.pi * c / grid.period
    centroids = []
    for k in range(1, n + 1):
        m = lab == k
        centroids.append(np.angle(np.mean(np.exp(1j * ang[m]))))
    order = np.argsort(centroids)
    phase_of = np.zeros(n + 1)
    for rank, k in enumerate(order):
        phase_of[k + 1] = phase_rad * (rank % 2)
    # assign every pixel (incl. anti-aliased edges) to its nearest clear region
    _, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    nearest = lab[idx[0], idx[1]]
    return phase_of[nearest]


def make_mask(pattern: np.ndarray, grid: Grid, mask_type: str = "binary",
              transmission: float = 0.06, phase_deg: float = 180.0, name: str | None = None,
              shifter: np.ndarray | None = None, shift_axis: str = "auto") -> Mask:
    """Convert an absorber pattern into a thin-mask complex transmission.

    mask_type
        * ``'binary'`` (chrome on glass): absorber -> 0, clear -> 1.
        * ``'attpsm'`` (MoSi attenuated PSM, default 6 %): absorber ->
          ``sqrt(transmission) * exp(i*phase)``, clear -> 1.
        * ``'altpsm'`` (alternating aperture PSM): absorber -> 0; clear openings
          alternately 1 and ``exp(i*phase)``.  If ``shifter`` (array, 1 where
          the quartz is etched) is not given, neighbouring clear regions are
          found automatically (periodic connected components sorted along
          ``shift_axis``) and alternated.  A periodic layout needs an even
          number of openings for a conflict-free assignment.

    Anti-aliased pixels are linearly mixed (exact for the zeroth order).
    """
    p = np.clip(np.asarray(pattern, dtype=float), 0.0, 1.0)
    if p.shape != (grid.n, grid.n):
        raise ValueError("pattern shape does not match grid")
    phi = np.deg2rad(phase_deg)
    if mask_type == "binary":
        t = (1.0 - p).astype(complex)
    elif mask_type == "attpsm":
        if not 0 <= transmission <= 1:
            raise ValueError("transmission must be in [0, 1]")
        t = (1.0 - p) + p * np.sqrt(transmission) * np.exp(1j * phi)
    elif mask_type == "altpsm":
        if shifter is not None:
            ph = np.clip(np.asarray(shifter, dtype=float), 0, 1) * phi
        else:
            ph = _alt_phase_map(1.0 - p, grid, phi, shift_axis)
        t = (1.0 - p) * np.exp(1j * ph)
    else:
        raise ValueError("mask_type must be 'binary', 'attpsm' or 'altpsm'")
    return Mask(grid=grid, transmission=t.astype(complex), name=name or mask_type)


# ---------------------------------------------------------------------------
# Bias / OPC
# ---------------------------------------------------------------------------
def _shift(p: np.ndarray, s_pix: float, axis: int) -> np.ndarray:
    """Periodic shift by a fractional number of pixels (linear interpolation)."""
    n = math.floor(s_pix)
    f = s_pix - n
    a = np.roll(p, n, axis=axis)
    if f == 0:
        return a
    return (1 - f) * a + f * np.roll(p, n + 1, axis=axis)


def _morph_1d(p: np.ndarray, r_pix: float, axis: int, grow: bool) -> np.ndarray:
    op = np.maximum if grow else np.minimum
    out = p.copy()
    m = math.floor(r_pix)
    shifts = [float(k) for k in range(1, m + 1)]
    if r_pix - m > 1e-12:
        shifts.append(r_pix)
    for s in shifts:
        out = op(out, _shift(p, s, axis))
        out = op(out, _shift(p, -s, axis))
    return out


def apply_bias(pattern: np.ndarray, grid: Grid, bias_nm: float) -> np.ndarray:
    """Move every absorber edge outward by ``bias_nm`` (negative shrinks).

    The CD of a line therefore changes by ``2 * bias_nm``.  Implemented as a
    grey-scale dilation/erosion with a square structuring element, applied
    separably along x and y with sub-pixel (linearly interpolated) shifts.  For
    area-sampled Manhattan edges the area change per edge is exact to
    sub-pixel precision; corners stay square (rectilinear bias, as in mask
    data prep).
    """
    p = np.clip(np.asarray(pattern, dtype=float), 0.0, 1.0)
    if bias_nm == 0:
        return p.copy()
    r = abs(bias_nm) / grid.pixel
    grow = bias_nm > 0
    out = _morph_1d(p, r, axis=1, grow=grow)
    out = _morph_1d(out, r, axis=0, grow=grow)
    return np.clip(out, 0.0, 1.0)


def _convex_corners(binary: np.ndarray, grid: Grid) -> list[tuple[float, float, int, int]]:
    """Outer (convex) corners of a binary pattern: (x_nm, y_nm, sx, sy) where s points outward."""
    corners = []
    x = grid.x
    h = grid.pixel / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            nx_out = ~np.roll(binary, -sx, axis=1)
            ny_out = ~np.roll(binary, -sy, axis=0)
            iy, ix = np.nonzero(binary & nx_out & ny_out)
            for a, b in zip(iy, ix):
                corners.append((x[b] + sx * h, x[a] + sy * h, sx, sy))
    return corners


def simple_opc(pattern: np.ndarray, grid: Grid, bias_dense_nm: float = 0.0,
               bias_iso_nm: float = 2.0, density_threshold: float = 0.25,
               density_radius_nm: float = 150.0, serif_nm: float = 0.0) -> np.ndarray:
    """Rule-based OPC: density-dependent edge bias plus optional corner serifs.

    Each pixel is assigned the dense or isolated bias depending on the local
    absorber-pattern density (Gaussian average with sigma ``density_radius_nm``,
    periodic).  Isolated features print narrower through focus (lower image
    contrast), so they get a larger bias.  If ``serif_nm > 0`` square serifs of
    that size are centred on every convex corner to counter corner rounding
    (for line ends this acts as a hammerhead).
    """
    p = np.clip(np.asarray(pattern, dtype=float), 0.0, 1.0)
    density = ndimage.gaussian_filter(p, density_radius_nm / grid.pixel, mode="wrap")
    p_dense = apply_bias(p, grid, bias_dense_nm)
    p_iso = apply_bias(p, grid, bias_iso_nm)
    out = np.where(density >= density_threshold, p_dense, p_iso)
    if serif_nm > 0:
        corners = _convex_corners(p >= 0.5, grid)
        s = serif_nm / 2
        rects = [(cx - s, cy - s, cx + s, cy + s) for cx, cy, _, _ in corners]
        if rects:
            out = np.clip(np.maximum(out, rectangles(grid, rects)), 0, 1)
    return out


def add_srafs(grid: Grid, line_centers_nm, sraf_width: float, sraf_offset: float,
              orientation: str = "vertical", base: np.ndarray | None = None,
              both_sides: bool = True) -> np.ndarray:
    """Sub-resolution assist features (scattering bars) beside isolated lines.

    Places absorber bars of width ``sraf_width`` (should be below the
    resolution limit, typically < ~0.3 lambda/NA at wafer) whose centres are
    ``sraf_offset`` nm from each main-line centre (on both sides by default).
    They make an isolated line diffract like a denser one, improving depth of
    focus.  If ``base`` is given the SRAFs are merged into it.
    """
    half = grid.period / 2
    rects = []
    sides = (-1, 1) if both_sides else (1,)
    for c in np.atleast_1d(line_centers_nm):
        for s in sides:
            sc = float(c) + s * sraf_offset
            if orientation == "vertical":
                rects.append((sc - sraf_width / 2, -half, sc + sraf_width / 2, half))
            else:
                rects.append((-half, sc - sraf_width / 2, half, sc + sraf_width / 2))
    sraf = rectangles(grid, rects)
    if base is not None:
        sraf = np.clip(np.maximum(np.asarray(base, dtype=float), sraf), 0, 1)
    return sraf


# ---------------------------------------------------------------------------
# Mask 3D (thick mask) correction
# ---------------------------------------------------------------------------
def mask_3d_correction(pattern: np.ndarray, grid: Grid, mask_type: str = "binary",
                       transmission: float = 0.06, phase_deg: float = 180.0,
                       bl_width_nm: float = 2.5, bl_amplitude: float = 0.5,
                       bl_phase_deg: float = 90.0, effective_bias_nm: float = 0.0,
                       name: str | None = None) -> Mask:
    """Thin-mask transmission with a boundary-layer mask-3D correction.

    The Kirchhoff (thin-mask) model assumes the field right below the mask is
    the geometric shadow of the absorber.  A real absorber is ~50-70 nm thick
    (comparable to the 193 nm wavelength at 4x), so the near field deviates
    near each edge: light is partly blocked/phase-delayed in a strip along the
    edge, which acts like an effective CD bias plus a phase error that shifts
    best focus with pitch.  This function uses the boundary-layer model
    (Tirapu-Azpiroz & Yablonovitch): a strip of width ``bl_width_nm`` (wafer
    scale) on the clear side of every absorber edge gets the complex
    transmission ``t_clear * bl_amplitude * exp(i*bl_phase_deg)``.  An
    additional rectilinear ``effective_bias_nm`` may be applied first.

    This is a crude, calibration-dependent approximation; it does not replace
    rigorous EMF (RCWA/FDTD) simulation and ignores polarisation and
    incidence-angle dependence.
    """
    p = apply_bias(pattern, grid, effective_bias_nm)
    base = make_mask(p, grid, mask_type=mask_type, transmission=transmission,
                     phase_deg=phase_deg, name=name or f"{mask_type}+M3D")
    if bl_width_nm <= 0:
        return base
    band = np.clip(apply_bias(p, grid, bl_width_nm) - p, 0.0, 1.0)
    if mask_type == "altpsm":
        # unit-amplitude clear-side transmission carrying the local shifter phase
        t_clear = np.exp(1j * np.angle(base.transmission))
    else:
        t_clear = np.ones_like(base.transmission)
    t_bl = t_clear * bl_amplitude * np.exp(1j * np.deg2rad(bl_phase_deg))
    base.transmission = base.transmission * (1 - band) + band * t_bl
    return base


# ---------------------------------------------------------------------------
# Pellicle
# ---------------------------------------------------------------------------
class Pellicle:
    """Thin fluoropolymer pellicle membrane mounted at a stand-off above the chrome.

    Single free-standing film (air / film / air) with complex index
    ``n + i k``; transmission from the Airy (multiple-beam) thin-film formula.
    The default thickness is close to an anti-reflection (maximum transmission)
    condition at 193 nm: ``2 n d = m lambda`` with m = 12 -> d ~ 829 nm.
    """

    def __init__(self, thickness_nm: float = 828.7, n: float = 1.40, k: float = 0.0,
                 wavelength: float = WAVELENGTH_ARF, standoff_mm: float = 6.3):
        self.thickness_nm = float(thickness_nm)
        self.n = float(n)
        self.k = float(k)
        self.wavelength = float(wavelength)
        self.standoff_mm = float(standoff_mm)

    @staticmethod
    def ar_thickness(n: float = 1.40, order: int = 12, wavelength: float = WAVELENGTH_ARF) -> float:
        """Film thickness (nm) of the ``order``-th transmission maximum at normal incidence."""
        return order * wavelength / (2 * n)

    def amplitude(self, angle_rad, polarization: str = "s") -> np.ndarray:
        """Complex amplitude transmission coefficient for 's' or 'p' polarisation."""
        th = np.asarray(angle_rad, dtype=float)
        n0 = 1.0
        n1 = self.n + 1j * self.k
        c0 = np.cos(th) + 0j
        s1 = n0 * np.sin(th) / n1
        c1 = np.sqrt(1 - s1 ** 2)
        if polarization == "s":
            r01 = (n0 * c0 - n1 * c1) / (n0 * c0 + n1 * c1)
            t01 = 2 * n0 * c0 / (n0 * c0 + n1 * c1)
            r12 = (n1 * c1 - n0 * c0) / (n1 * c1 + n0 * c0)
            t12 = 2 * n1 * c1 / (n1 * c1 + n0 * c0)
        elif polarization == "p":
            r01 = (n1 * c0 - n0 * c1) / (n1 * c0 + n0 * c1)
            t01 = 2 * n0 * c0 / (n1 * c0 + n0 * c1)
            r12 = (n0 * c1 - n1 * c0) / (n0 * c1 + n1 * c0)
            t12 = 2 * n1 * c1 / (n0 * c1 + n1 * c0)
        else:
            raise ValueError("polarization must be 's' or 'p'")
        beta = 2 * np.pi * n1 * self.thickness_nm * c1 / self.wavelength
        return t01 * t12 * np.exp(1j * beta) / (1 + r01 * r12 * np.exp(2j * beta))

    def transmission(self, angle_rad=0.0, polarization: str = "unpolarized"):
        """Intensity transmission vs incidence angle (radians, in air at the reticle).

        ``polarization`` is 's' (TE), 'p' (TM) or 'unpolarized' (average).
        """
        if polarization == "unpolarized":
            t = 0.5 * (np.abs(self.amplitude(angle_rad, "s")) ** 2
                       + np.abs(self.amplitude(angle_rad, "p")) ** 2)
        else:
            t = np.abs(self.amplitude(angle_rad, polarization)) ** 2
        return float(t) if np.ndim(t) == 0 else t

    def transmission_vs_na(self, na_reticle: float = NA_MAX_IMMERSION / REDUCTION,
                           n_points: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """(angles, T) from normal incidence up to asin(na_reticle)."""
        th = np.linspace(0, math.asin(min(na_reticle, 1.0)), n_points)
        return th, self.transmission(th)

    def standoff_defect_blur(self, particle_size_um: float, standoff_mm: float | None = None,
                             na_reticle: float = NA_MAX_IMMERSION / REDUCTION) -> dict:
        """Why particles on the pellicle do not print.

        A particle on the pellicle is ``standoff_mm`` out of focus of the
        reticle plane.  The illumination cone (half angle asin(na_reticle))
        through a point on the chrome covers a disc of diameter
        ``2 * standoff * tan(theta)`` on the pellicle; a particle of diameter
        d therefore only removes ~(d / D)^2 of the light reaching that mask
        point (geometric-shadow estimate).

        Returns a dict with ``blur_diameter_um`` (at the pellicle),
        ``obscuration`` (fractional dose loss at the mask point, <= 1),
        ``blur_diameter_wafer_um`` (the defocused blur image size projected
        to the wafer) and ``defocus_wafer_um`` (equivalent wafer defocus,
        standoff / M^2).
        """
        z = self.standoff_mm if standoff_mm is None else standoff_mm
        th = math.asin(min(na_reticle, 1.0))
        blur_um = 2 * z * 1e3 * math.tan(th)
        obsc = min(1.0, (particle_size_um / blur_um) ** 2) if blur_um > 0 else 1.0
        return {
            "blur_diameter_um": blur_um,
            "obscuration": obsc,
            "blur_diameter_wafer_um": blur_um / REDUCTION,
            "defocus_wafer_um": z * 1e3 / REDUCTION ** 2,
        }


# ---------------------------------------------------------------------------
# Mask errors
# ---------------------------------------------------------------------------
def mask_cd_error(pattern: np.ndarray, grid: Grid, cd_error_nm: float,
                  rng: np.random.Generator | int | None = None, mode: str = "random",
                  reticle_scale: bool = False) -> np.ndarray:
    """Apply mask CD errors to an absorber pattern.

    ``cd_error_nm`` is the CD error (both edges together) at wafer scale, or at
    reticle scale if ``reticle_scale=True`` (then divided by ``REDUCTION``).

    * ``mode='systematic'``: every feature's CD changes by ``cd_error_nm``.
    * ``mode='random'``: each feature (connected absorber component) gets an
      independent CD error drawn from N(0, cd_error_nm) (i.e. ``cd_error_nm``
      is 1 sigma).  Pixels are assigned to features by nearest-feature
      (Voronoi) partition so neighbouring features are biased independently.
    """
    p = np.clip(np.asarray(pattern, dtype=float), 0.0, 1.0)
    err = cd_error_nm / REDUCTION if reticle_scale else cd_error_nm
    if mode == "systematic":
        return apply_bias(p, grid, err / 2)
    if mode != "random":
        raise ValueError("mode must be 'random' or 'systematic'")
    rng = np.random.default_rng(rng)
    lab, n = _periodic_labels(p >= 0.5)
    if n == 0:
        return p.copy()
    _, idx = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    cell = lab[idx[0], idx[1]]
    out = np.zeros_like(p)
    errs = rng.normal(0.0, err, size=n)
    for k in range(1, n + 1):
        m = cell == k
        out[m] = apply_bias(p * m, grid, errs[k - 1] / 2)[m]
    return out


def meef(cd_wafer_values, cd_mask_values, reticle_scale: bool = False) -> float:
    """Mask error enhancement factor, MEEF = dCD_wafer / dCD_mask (mask CD at wafer scale).

    Fitted as the least-squares slope of wafer CD versus mask CD.  If
    ``reticle_scale`` is True the mask CDs are given at reticle (4x) scale and
    are divided by ``REDUCTION`` first.  MEEF = 1 for a linear imaging
    regime; MEEF > 1 near the resolution limit.
    """
    w = np.asarray(cd_wafer_values, dtype=float)
    m = np.asarray(cd_mask_values, dtype=float)
    if reticle_scale:
        m = m / REDUCTION
    if w.size != m.size or w.size < 2:
        raise ValueError("need >= 2 matching (wafer, mask) CD values")
    if np.ptp(m) == 0:
        raise ValueError("mask CD values must vary")
    return float(np.polyfit(m, w, 1)[0])


# ---------------------------------------------------------------------------
# Reticle stage and reticle heating
# ---------------------------------------------------------------------------
@dataclass
class ReticleStage:
    """Scanning reticle stage synchronised to the wafer stage.

    During a scan the reticle moves ``REDUCTION`` times faster than the wafer
    and in the opposite direction (the projection lens inverts the image).
    Positions are in nm, the scan direction is y.
    """

    scan_speed_wafer_mm_s: float = 700.0
    reduction: float = REDUCTION

    def reticle_velocity(self) -> float:
        """Signed reticle velocity in mm/s (opposite sign to the wafer)."""
        return -self.reduction * self.scan_speed_wafer_mm_s

    def reticle_speed(self) -> float:
        """Reticle scan speed magnitude in mm/s (= 4 x wafer speed)."""
        return abs(self.reticle_velocity())

    def scan_time(self, field_height_mm: float = 33.0, slit_height_mm: float = SLIT_HEIGHT_MM) -> float:
        """Time (s) to scan a field: the slit must traverse field + slit height."""
        return (field_height_mm + slit_height_mm) / self.scan_speed_wafer_mm_s

    def ideal_reticle_position(self, wafer_pos_nm):
        """Reticle position (nm) that is perfectly synchronised with ``wafer_pos_nm``."""
        return -self.reduction * np.asarray(wafer_pos_nm, dtype=float)

    def synchronization_error(self, wafer_pos, reticle_pos):
        """Image placement error at the wafer (nm): wafer_pos + reticle_pos / REDUCTION.

        Zero when the reticle position equals ``-REDUCTION * wafer_pos``.
        """
        return np.asarray(wafer_pos, dtype=float) + np.asarray(reticle_pos, dtype=float) / self.reduction

    def ma_msd(self, t_s, wafer_pos, reticle_pos, slit_height_mm: float = SLIT_HEIGHT_MM):
        """Moving average (MA) and moving standard deviation (MSD) of the sync error.

        A wafer point is exposed while it crosses the slit, i.e. for
        ``slit_height / scan_speed`` seconds; MA (-> overlay) and MSD
        (-> image blur / contrast loss) are taken over that sliding window.
        Returns (ma, msd) arrays (nm), NaN where the window is incomplete.
        """
        t = np.asarray(t_s, dtype=float)
        e = self.synchronization_error(wafer_pos, reticle_pos)
        window = slit_height_mm / self.scan_speed_wafer_mm_s
        ma = np.full_like(e, np.nan)
        msd = np.full_like(e, np.nan)
        for i, ti in enumerate(t):
            sel = (t >= ti - window / 2) & (t <= ti + window / 2)
            if t[0] <= ti - window / 2 and t[-1] >= ti + window / 2:
                ma[i] = e[sel].mean()
                msd[i] = e[sel].std()
        return ma, msd


class ReticleHeating:
    """Lumped thermal model of reticle heating by absorbed exposure light.

    The 6-inch fused-silica reticle (152 x 152 x 6.35 mm) absorbs power ``P``
    (mostly in the chrome) and loses it by convection/radiation from both
    faces with coefficient ``h`` (W/m^2/K).  Temperature rise
    ``dT(t) = P/(h A) * (1 - exp(-t/tau))`` with ``tau = m c_p / (h A)``.
    Uniform expansion ``CTE * dT`` about the reticle centre is a pure
    magnification change; at wafer scale the image point (x, y) moves by
    ``CTE * dT * (x, y)`` (the 4x reduction scales length and error alike).
    Real reticles heat non-uniformly, which adds higher-order distortion; this
    model captures only the first-order magnification term.
    """

    def __init__(self, absorbed_power_w: float = 1.0, h_w_m2k: float = 10.0,
                 size_mm: float = 152.0, thickness_mm: float = 6.35,
                 cte_per_k: float = FUSED_SILICA_CTE):
        self.absorbed_power_w = float(absorbed_power_w)
        self.h = float(h_w_m2k)
        self.size_mm = float(size_mm)
        self.thickness_mm = float(thickness_mm)
        self.cte = float(cte_per_k)

    @staticmethod
    def absorbed_power(incident_power_w: float, absorber_coverage: float,
                       absorber_absorptance: float = 0.6) -> float:
        """Power (W) absorbed in the reticle for a given absorber area fraction."""
        return incident_power_w * absorber_coverage * absorber_absorptance

    @property
    def area_m2(self) -> float:
        return 2 * (self.size_mm * 1e-3) ** 2

    @property
    def heat_capacity_j_k(self) -> float:
        vol = (self.size_mm * 1e-3) ** 2 * self.thickness_mm * 1e-3
        return vol * FUSED_SILICA_DENSITY * FUSED_SILICA_CP

    @property
    def time_constant_s(self) -> float:
        return self.heat_capacity_j_k / (self.h * self.area_m2)

    def temperature_rise(self, t_s: float | np.ndarray | None = None):
        """Reticle temperature rise (K); steady state if ``t_s`` is None."""
        dt_ss = self.absorbed_power_w / (self.h * self.area_m2)
        if t_s is None:
            return dt_ss
        return dt_ss * (1 - np.exp(-np.asarray(t_s, dtype=float) / self.time_constant_s))

    def magnification_error_ppm(self, t_s=None):
        """Magnification error in ppm (CTE * dT * 1e6)."""
        return self.cte * self.temperature_rise(t_s) * 1e6

    def registration_error_nm(self, x_wafer_nm, y_wafer_nm, t_s=None):
        """(dx, dy) image displacement at the wafer (nm) for points at (x, y) wafer nm."""
        s = self.cte * self.temperature_rise(t_s)
        return s * np.asarray(x_wafer_nm, dtype=float), s * np.asarray(y_wafer_nm, dtype=float)
