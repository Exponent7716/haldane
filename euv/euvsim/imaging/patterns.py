"""Test-pattern generators on a *wafer-scale* periodic grid.

All generators return ``pattern[y, x]`` with values in [0, 1] (1 = reflective
multilayer, 0 = absorber) following the ARCHITECTURE contract.  Pixels are
anti-aliased by exact area coverage, so CDs need not be multiples of ``dx_nm``.

Grid convention: pixel ``i`` is centred at ``x = i*dx`` and covers
``[(i-0.5)dx, (i+0.5)dx)``.  The grid is periodic (the FFT imaging engine
assumes periodic boundary conditions), so a grid of one pitch describes an
infinite array.  The physical mask features are ``magnification`` times larger
(see :class:`euvsim.imaging.mask.ReflectiveMask`).
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def _npix(length_nm: float, dx_nm: float) -> int:
    n = length_nm / dx_nm
    ni = int(round(n))
    if ni < 1 or abs(n - ni) > 1e-6 * max(1.0, n):
        raise ValueError(f"length {length_nm} nm is not an integer multiple of dx={dx_nm} nm")
    return ni


def coverage_1d(a_nm: float, b_nm: float, n: int, dx_nm: float) -> np.ndarray:
    """Fractional coverage of the periodic interval [a, b) on an n-pixel grid."""
    L = n * dx_nm
    w = b_nm - a_nm
    if w <= 0:
        return np.zeros(n)
    if w >= L:
        return np.ones(n)
    a = (a_nm % L)
    b = a + w
    lo = (np.arange(n) - 0.5) * dx_nm
    hi = lo + dx_nm
    cov = np.zeros(n)
    for k in (-1, 0, 1):
        cov += np.clip(np.minimum(b + k * L, hi) - np.maximum(a + k * L, lo), 0.0, None)
    return np.clip(cov / dx_nm, 0.0, 1.0)


def rectangles(rects: Iterable[Sequence[float]], size_nm: Tuple[float, float], dx_nm: float,
               value: float = 1.0, background: float = 0.0) -> np.ndarray:
    """Arbitrary axis-aligned rectangles ``(x0, y0, x1, y1)`` (nm) on a periodic field
    of ``size_nm = (Lx, Ly)``.  Overlapping rectangles are merged (max coverage)."""
    nx, ny = _npix(size_nm[0], dx_nm), _npix(size_nm[1], dx_nm)
    cov = np.zeros((ny, nx))
    for x0, y0, x1, y1 in rects:
        cx = coverage_1d(min(x0, x1), max(x0, x1), nx, dx_nm)
        cy = coverage_1d(min(y0, y1), max(y0, y1), ny, dx_nm)
        cov = np.maximum(cov, np.outer(cy, cx))
    return background + (value - background) * cov


def _tone(cov: np.ndarray, tone: str) -> np.ndarray:
    if tone == "dark":      # absorber feature on reflective background (bright field)
        return 1.0 - cov
    if tone == "bright":    # reflective feature in absorber (dark field)
        return cov
    raise ValueError("tone must be 'dark' or 'bright'")


def line_space(pitch_nm: float, cd_nm: float, dx_nm: float, orientation: str = "vertical",
               n_periods: int = 1, other_px: int = 1, tone: str = "dark") -> np.ndarray:
    """Periodic lines/spaces.

    ``orientation='vertical'`` -> lines run along y (pattern varies in x), shape
    ``(other_px, n_periods*pitch/dx)``; ``'horizontal'`` -> transposed.  Line
    centres sit at ``(k+0.5)*pitch``, so for ``n_periods=1`` a line is centred on
    the grid centre.  ``tone='dark'``: lines are absorber (0) on reflective
    spaces; ``'bright'``: lines are reflective trenches.
    """
    n = _npix(n_periods * pitch_nm, dx_nm)
    cov = np.zeros(n)
    for k in range(n_periods):
        c = (k + 0.5) * pitch_nm
        cov = np.maximum(cov, coverage_1d(c - cd_nm / 2, c + cd_nm / 2, n, dx_nm))
    row = _tone(cov, tone)
    if orientation == "vertical":
        return np.tile(row, (other_px, 1))
    if orientation == "horizontal":
        return np.tile(row[:, None], (1, other_px))
    raise ValueError("orientation must be 'vertical' or 'horizontal'")


def isolated_line(cd_nm: float, dx_nm: float, field_nm: float = 400.0,
                  orientation: str = "vertical", other_px: int = 1, tone: str = "dark") -> np.ndarray:
    """A single line centred in a (periodic) field of ``field_nm``."""
    return line_space(field_nm, cd_nm, dx_nm, orientation, 1, other_px, tone)


def contact_array(pitch_x_nm: float, cd_x_nm: float, dx_nm: float, pitch_y_nm: float | None = None,
                  cd_y_nm: float | None = None, n_x: int = 1, n_y: int = 1,
                  tone: str = "bright") -> np.ndarray:
    """Rectangular array of contact holes (default: reflective holes in absorber,
    i.e. dark-field mask).  Hole centres at ``((i+0.5)px, (j+0.5)py)``."""
    pitch_y_nm = pitch_x_nm if pitch_y_nm is None else pitch_y_nm
    cd_y_nm = cd_x_nm if cd_y_nm is None else cd_y_nm
    rects = []
    for i in range(n_x):
        for j in range(n_y):
            cx, cy = (i + 0.5) * pitch_x_nm, (j + 0.5) * pitch_y_nm
            rects.append((cx - cd_x_nm / 2, cy - cd_y_nm / 2, cx + cd_x_nm / 2, cy + cd_y_nm / 2))
    cov = rectangles(rects, (n_x * pitch_x_nm, n_y * pitch_y_nm), dx_nm)
    return _tone(cov, tone)


def tip_to_tip(line_cd_nm: float, gap_nm: float, pitch_nm: float, dx_nm: float,
               field_y_nm: float | None = None, tone: str = "bright") -> np.ndarray:
    """Vertical lines (pitch in x) each cut by a tip-to-tip gap centred at the
    field centre in y.  Default tone: reflective lines (dark-field metal layer)."""
    Ly = field_y_nm if field_y_nm is not None else max(8 * line_cd_nm, 4 * gap_nm)
    xc, yc = pitch_nm / 2, Ly / 2
    rect = (xc - line_cd_nm / 2, yc + gap_nm / 2, xc + line_cd_nm / 2, yc + Ly - gap_nm / 2)
    cov = rectangles([rect], (pitch_nm, Ly), dx_nm)
    return _tone(cov, tone)


def grid_coords(shape: Tuple[int, int], dx_nm: float) -> Tuple[np.ndarray, np.ndarray]:
    """x (columns) and y (rows) coordinate vectors in nm for a pattern grid."""
    ny, nx = shape
    return np.arange(nx) * dx_nm, np.arange(ny) * dx_nm
