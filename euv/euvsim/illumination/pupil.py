"""Illumination pupil shapes ("illumination settings") as discrete source points.

In partially coherent (Koehler) illumination every point of the illuminator
pupil is an incoherent plane-wave source illuminating the reticle at direction

    (sin θx, sin θy) = NA_reticle · (σx, σy)

with σ the normalised pupil coordinate (σ = 1 ↔ edge of the projection-lens
NA referred to the reticle).  The pupil intensity distribution J(σx, σy) is
therefore represented, per the package contract, as an ``ndarray`` of shape
``(N, 3)`` with columns ``(sigma_x, sigma_y, weight)``, |σ| ≤ 1 and Σ w = 1.

Shapes provided (public-literature names):

* conventional (top-hat disc, radius σ)
* annular (σin ≤ |σ| ≤ σout)
* dipole (two poles on x or y, annular sectors of opening angle α)
* quadrupole: ``quasar`` (poles on the diagonals 45°/135°/...) and
  ``cquad`` (poles on the x and y axes, "C-Quad")
* leaf-shaped dipole (lens-shaped poles formed by two intersecting circles),
  typical of EUV dense-line SMO solutions
* freeform via :func:`from_bitmap` / :func:`from_mask`

Shapes are defined as boolean masks ``mask(sx, sy) -> bool`` (see
:func:`shape_mask`) and sampled on a square σ grid of pitch ``step``; the same
masks drive the facet-based (FlexPupil) discretisation in :mod:`.flexpupil`.

Pupil metrics
-------------
* Pupil fill ratio (PFR): illuminated fraction of the unit pupil disc,
  ``PFR = A_lit / π``.  For weighted pupils we use the *effective* lit area
  ``A_eff = (Σw)² / Σw² · Δσ²`` (exact for top-hat pupils).
* Centroid: ``(Σ w σx, Σ w σy)`` (telecentricity-relevant pupil shift).
* Ellipticity: ``E = (I_H − I_V)/(I_H + I_V)``, with I_H the energy in the
  ±45° sectors around the x axis and I_V around the y axis.
* Pole balance: ``(max − min)/(max + min)`` of the energies in the poles
  (quadrants or half-planes).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

Mask = Callable[[np.ndarray, np.ndarray], np.ndarray]

DEFAULT_STEP = 0.02


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sigma_grid(step: float = DEFAULT_STEP) -> tuple[np.ndarray, np.ndarray]:
    """Cell centres of a square σ grid of pitch ``step`` covering the unit disc."""
    if not (0 < step <= 0.5):
        raise ValueError("step must be in (0, 0.5]")
    n = int(np.ceil(1.0 / step))
    c = (np.arange(-n, n) + 0.5) * step
    sx, sy = np.meshgrid(c, c)
    inside = sx ** 2 + sy ** 2 <= 1.0
    return sx[inside], sy[inside]


def _points(sx: np.ndarray, sy: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    sx = np.asarray(sx, float).ravel()
    sy = np.asarray(sy, float).ravel()
    w = np.ones_like(sx) if w is None else np.asarray(w, float).ravel()
    keep = w > 0
    sx, sy, w = sx[keep], sy[keep], w[keep]
    if w.size == 0:
        raise ValueError("pupil shape contains no source points; refine `step` or check parameters")
    return np.column_stack([sx, sy, w / w.sum()])


def _angle(sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    return np.arctan2(sy, sx)


def _ang_dist(a: np.ndarray, center: float) -> np.ndarray:
    """Smallest absolute angular distance (rad) between ``a`` and ``center``."""
    return np.abs((a - center + np.pi) % (2 * np.pi) - np.pi)


def _check_radii(sigma_in: float, sigma_out: float) -> None:
    if not (0 <= sigma_in < sigma_out <= 1.0):
        raise ValueError("need 0 <= sigma_in < sigma_out <= 1")


# ---------------------------------------------------------------------------
# shape masks
# ---------------------------------------------------------------------------
def conventional_mask(sigma: float = 0.8) -> Mask:
    if not (0 < sigma <= 1):
        raise ValueError("sigma must be in (0, 1]")
    return lambda sx, sy: sx ** 2 + sy ** 2 <= sigma ** 2


def annular_mask(sigma_in: float = 0.5, sigma_out: float = 0.8) -> Mask:
    _check_radii(sigma_in, sigma_out)

    def m(sx, sy):
        r2 = sx ** 2 + sy ** 2
        return (r2 >= sigma_in ** 2) & (r2 <= sigma_out ** 2)
    return m


def sector_mask(sigma_in: float, sigma_out: float, centers_deg, opening_angle_deg: float) -> Mask:
    """Union of annular sectors centred on ``centers_deg`` with full opening α."""
    ann = annular_mask(sigma_in, sigma_out)
    half = 0.5 * opening_angle_deg * np.pi / 180.0
    cs = [c * np.pi / 180.0 for c in centers_deg]

    def m(sx, sy):
        a = _angle(sx, sy)
        sel = np.zeros(np.shape(sx), bool)
        for c in cs:
            sel |= _ang_dist(a, c) <= half
        return ann(sx, sy) & sel
    return m


def dipole_mask(sigma_in: float = 0.5, sigma_out: float = 0.9, opening_angle_deg: float = 90.0,
                orientation: str = "x") -> Mask:
    """Dipole poles: annular sectors on the x axis (``'x'``, images vertical lines)
    or the y axis (``'y'``)."""
    if orientation not in ("x", "y"):
        raise ValueError("orientation must be 'x' or 'y'")
    c = (0.0, 180.0) if orientation == "x" else (90.0, 270.0)
    return sector_mask(sigma_in, sigma_out, c, opening_angle_deg)


def quadrupole_mask(sigma_in: float = 0.5, sigma_out: float = 0.9, opening_angle_deg: float = 45.0,
                    kind: str = "quasar") -> Mask:
    """``kind='quasar'``: poles at 45°, 135°, ...; ``kind='cquad'``: poles on the axes."""
    if kind == "quasar":
        c = (45.0, 135.0, 225.0, 315.0)
    elif kind in ("cquad", "c-quad"):
        c = (0.0, 90.0, 180.0, 270.0)
    else:
        raise ValueError("kind must be 'quasar' or 'cquad'")
    return sector_mask(sigma_in, sigma_out, c, opening_angle_deg)


def leaf_mask(sigma_c: float = 0.7, leaf_radius: float = 0.55, orientation: str = "x",
              sigma_out: float = 1.0) -> Mask:
    """Leaf-shaped dipole: each pole is the lens-shaped intersection of two discs.

    For an x-dipole each leaf is the intersection of two discs of radius ``R``
    centred at (±σc, ±d) with d chosen so the leaf is symmetric about the x axis,
    i.e. the pole is bounded by two circular arcs meeting at tips on the x axis
    at σx = σc ± sqrt(R² − d²).  We use d = R/√2, giving a leaf of length √2·R and
    half-width R(1 − 1/√2).  Points outside ``sigma_out`` are clipped.
    """
    if orientation not in ("x", "y"):
        raise ValueError("orientation must be 'x' or 'y'")
    R = leaf_radius
    d = R / np.sqrt(2.0)

    def m(sx, sy):
        u, v = (sx, sy) if orientation == "x" else (sy, sx)
        au = np.abs(u)
        lens = (((au - sigma_c) ** 2 + (v - d) ** 2) <= R ** 2) & \
               (((au - sigma_c) ** 2 + (v + d) ** 2) <= R ** 2)
        return lens & (sx ** 2 + sy ** 2 <= sigma_out ** 2)
    return m


_MASKS: Dict[str, Callable[..., Mask]] = {
    "conventional": conventional_mask,
    "annular": annular_mask,
    "dipole": dipole_mask,
    "quadrupole": quadrupole_mask,
    "quasar": lambda **kw: quadrupole_mask(kind="quasar", **kw),
    "cquad": lambda **kw: quadrupole_mask(kind="cquad", **kw),
    "leaf": leaf_mask,
}


def shape_mask(name: str, **params) -> Mask:
    """Boolean pupil mask for a named setting (``conventional``, ``annular``,
    ``dipole``, ``quadrupole``, ``quasar``, ``cquad``, ``leaf``)."""
    key = name.lower().replace("-", "").replace("_", "")
    if key not in _MASKS:
        raise KeyError(f"unknown pupil setting {name!r}; known: {sorted(_MASKS)}")
    return _MASKS[key](**params)


# ---------------------------------------------------------------------------
# source-point generators (public contract)
# ---------------------------------------------------------------------------
def from_mask(mask: Mask, step: float = DEFAULT_STEP) -> np.ndarray:
    """Sample a boolean pupil mask on the σ grid → uniform-weight source points."""
    sx, sy = sigma_grid(step)
    sel = np.asarray(mask(sx, sy), bool)
    return _points(sx[sel], sy[sel])


def conventional(sigma: float = 0.8, step: float = DEFAULT_STEP) -> np.ndarray:
    """Top-hat disc |σ| ≤ sigma.  PFR = σ²."""
    return from_mask(conventional_mask(sigma), step)


def annular(sigma_in: float = 0.5, sigma_out: float = 0.8, step: float = DEFAULT_STEP) -> np.ndarray:
    """Annulus σin ≤ |σ| ≤ σout.  PFR = σout² − σin²."""
    return from_mask(annular_mask(sigma_in, sigma_out), step)


def dipole(sigma_in: float = 0.5, sigma_out: float = 0.9, opening_angle_deg: float = 90.0,
           orientation: str = "x", step: float = DEFAULT_STEP) -> np.ndarray:
    """Dipole of two annular-sector poles.  PFR = (σout² − σin²)·α/180°."""
    return from_mask(dipole_mask(sigma_in, sigma_out, opening_angle_deg, orientation), step)


def quadrupole(sigma_in: float = 0.5, sigma_out: float = 0.9, opening_angle_deg: float = 45.0,
               kind: str = "quasar", step: float = DEFAULT_STEP) -> np.ndarray:
    """Quasar (diagonal poles) or C-Quad (axial poles).  PFR = (σout² − σin²)·α/90°."""
    return from_mask(quadrupole_mask(sigma_in, sigma_out, opening_angle_deg, kind), step)


def quasar(sigma_in: float = 0.5, sigma_out: float = 0.9, opening_angle_deg: float = 45.0,
           step: float = DEFAULT_STEP) -> np.ndarray:
    return quadrupole(sigma_in, sigma_out, opening_angle_deg, "quasar", step)


def cquad(sigma_in: float = 0.5, sigma_out: float = 0.9, opening_angle_deg: float = 45.0,
          step: float = DEFAULT_STEP) -> np.ndarray:
    return quadrupole(sigma_in, sigma_out, opening_angle_deg, "cquad", step)


def leaf(sigma_c: float = 0.7, leaf_radius: float = 0.55, orientation: str = "x",
         step: float = DEFAULT_STEP) -> np.ndarray:
    """Leaf-shaped dipole (see :func:`leaf_mask`)."""
    return from_mask(leaf_mask(sigma_c, leaf_radius, orientation), step)


def from_bitmap(bitmap: np.ndarray, step: float = DEFAULT_STEP, threshold: float = 0.0) -> np.ndarray:
    """Freeform pupil from a 2-D intensity bitmap ``bitmap[iy, ix]``.

    The bitmap spans σ ∈ [−1, 1] on both axes (row 0 at σy = −1).  It is
    resampled (nearest neighbour) onto the σ grid; values ≤ ``threshold`` and
    points outside the unit disc are dropped; the remaining values become weights.
    """
    b = np.asarray(bitmap, float)
    if b.ndim != 2:
        raise ValueError("bitmap must be 2-D")
    if np.any(b < 0):
        raise ValueError("bitmap intensities must be non-negative")
    ny, nx = b.shape
    sx, sy = sigma_grid(step)
    ix = np.clip(((sx + 1) / 2 * nx).astype(int), 0, nx - 1)
    iy = np.clip(((sy + 1) / 2 * ny).astype(int), 0, ny - 1)
    w = b[iy, ix]
    w = np.where(w > threshold, w, 0.0)
    return _points(sx, sy, w)


def to_bitmap(points: np.ndarray, n: int = 101) -> np.ndarray:
    """Bin source points into an ``n × n`` bitmap over σ ∈ [−1, 1] (for display)."""
    p = np.asarray(points)
    img, _, _ = np.histogram2d(p[:, 1], p[:, 0], bins=n, range=[[-1, 1], [-1, 1]], weights=p[:, 2])
    return img


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def pupil_fill_ratio(points: np.ndarray, step: float = DEFAULT_STEP) -> float:
    """PFR = effective lit area / π, with A_eff = (Σw)²/Σw² · step².

    For a top-hat pupil sampled on a grid of pitch ``step`` this is the lit
    fraction of the unit pupil disc (e.g. conventional σ → σ²).
    """
    w = np.asarray(points)[:, 2]
    n_eff = w.sum() ** 2 / np.sum(w ** 2)
    return float(n_eff * step ** 2 / np.pi)


def centroid(points: np.ndarray) -> tuple[float, float]:
    """Energy-weighted pupil centroid (σx, σy); 0 for a balanced pupil."""
    p = np.asarray(points)
    w = p[:, 2] / p[:, 2].sum()
    return float(np.sum(w * p[:, 0])), float(np.sum(w * p[:, 1]))


def ellipticity(points: np.ndarray) -> float:
    """E = (I_H − I_V)/(I_H + I_V) using ±45° sectors around the x and y axes."""
    p = np.asarray(points)
    a = _angle(p[:, 0], p[:, 1])
    h = (_ang_dist(a, 0.0) < np.pi / 4) | (_ang_dist(a, np.pi) < np.pi / 4)
    diag = np.isclose(np.abs(np.abs(a) - np.pi / 2), np.pi / 4)   # exactly on a 45° line: split
    ih = p[h & ~diag, 2].sum() + 0.5 * p[diag, 2].sum()
    iv = p[~h & ~diag, 2].sum() + 0.5 * p[diag, 2].sum()
    tot = ih + iv
    return float((ih - iv) / tot) if tot > 0 else 0.0


def pole_energies(points: np.ndarray, n_poles: int = 4, offset_deg: float = 0.0) -> np.ndarray:
    """Energy in ``n_poles`` equal angular sectors, the first centred on ``offset_deg``."""
    p = np.asarray(points)
    a = (_angle(p[:, 0], p[:, 1]) - offset_deg * np.pi / 180 + np.pi / n_poles) % (2 * np.pi)
    idx = np.minimum((a / (2 * np.pi / n_poles)).astype(int), n_poles - 1)
    return np.bincount(idx, weights=p[:, 2], minlength=n_poles)


def pole_balance(points: np.ndarray, n_poles: int = 4, offset_deg: float = 0.0) -> float:
    """Pole imbalance (max − min)/(max + min) of :func:`pole_energies` (0 = perfect).

    Use ``n_poles=2, offset_deg=0`` for an x-dipole, ``offset_deg=45`` for quasar.
    """
    e = pole_energies(points, n_poles, offset_deg)
    return float((e.max() - e.min()) / (e.max() + e.min())) if e.max() > 0 else 0.0


def pupil_metrics(points: np.ndarray, step: float = DEFAULT_STEP) -> dict:
    """Bundle of PFR, centroid, ellipticity, 2- and 4-pole balance."""
    cx, cy = centroid(points)
    return {
        "pfr": pupil_fill_ratio(points, step),
        "centroid_x": cx, "centroid_y": cy,
        "ellipticity": ellipticity(points),
        "pole_balance_2": pole_balance(points, 2, 0.0),
        "pole_balance_4": pole_balance(points, 4, 0.0),
        "n_points": int(len(points)),
    }


def validate(points: np.ndarray, tol: float = 1e-9) -> None:
    """Raise ``ValueError`` if ``points`` violates the source-point contract."""
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("source points must have shape (N, 3)")
    if np.any(p[:, 2] < 0) or abs(p[:, 2].sum() - 1) > 1e-6:
        raise ValueError("weights must be non-negative and sum to 1")
    if np.any(np.hypot(p[:, 0], p[:, 1]) > 1 + tol):
        raise ValueError("|sigma| must be <= 1")
