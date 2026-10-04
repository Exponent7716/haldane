"""Overlay: wafer grid + intra-field models, alignment fit and budgets.

Overlay model
-------------
For a mark at wafer coordinate ``(X, Y) = (Xf + xf, Yf + yf)`` (field centre
plus intra-field position) the placement error is modelled as

    dx = Tx + Mx X - Rx Y + [higher order in X, Y] + mx xf - rx yf
    dy = Ty + My Y + Ry X + [higher order in X, Y] + my yf + ry xf

Linear wafer (inter-field) terms: translation (Tx, Ty), magnification
(Mx, My, in ppm when multiplied by 1e6), rotation (Rx, Ry in rad).  The
usual derived quantities are

    symmetric magnification  M = (Mx + My)/2,   asymmetric (Mx - My)/2
    rotation                 R = (Rx + Ry)/2
    non-orthogonality        N = Rx - Ry   (grid axes not at 90 degrees)

Higher-order wafer terms (e.g. the 3rd-order "20-parameter" model) are all
monomials X^i Y^j with 2 <= i + j <= order, independently for dx and dy.
Intra-field (lens / reticle / scan) terms: field magnification (mx, my) and
field rotation (rx, ry); the scanner corrects these per field via lens
magnification and reticle-stage rotation / scan skew.

Fit: ordinary linear least squares of all parameters to the measured
alignment (or overlay-metrology) displacements.  Overlay performance is
quoted per axis as ``|mean| + 3 sigma`` of the residuals.

Overlay budgets combine independent contributors in root-sum-square:
``OV = sqrt(sum_i OV_i^2)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

LINEAR_NAMES = ("tx", "ty", "mx", "my", "rx", "ry")
INTRA_NAMES = ("fmx", "fmy", "frx", "fry")


def _monomials(order: int) -> list[tuple[int, int]]:
    return [(i, n - i) for n in range(2, order + 1) for i in range(n, -1, -1)]


@dataclass
class GridModel:
    """Wafer-grid + intra-field overlay model parameters.

    Units: translations in m, magnifications dimensionless (1e-6 = 1 ppm),
    rotations in rad; higher order coefficients in m^(1-i-j).
    """

    tx: float = 0.0
    ty: float = 0.0
    mx: float = 0.0
    my: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    fmx: float = 0.0     # intra-field magnification x
    fmy: float = 0.0
    frx: float = 0.0     # intra-field rotation
    fry: float = 0.0
    higher_x: dict[tuple[int, int], float] = field(default_factory=dict)
    higher_y: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def magnification(self) -> float:
        return 0.5 * (self.mx + self.my)

    @property
    def rotation(self) -> float:
        return 0.5 * (self.rx + self.ry)

    @property
    def non_orthogonality(self) -> float:
        return self.rx - self.ry

    def displacement(self, X: np.ndarray, Y: np.ndarray, xf: np.ndarray | None = None,
                     yf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(dx, dy) at absolute wafer coordinates (X, Y) with intra-field (xf, yf)."""
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        xf = np.zeros_like(X) if xf is None else np.asarray(xf, float)
        yf = np.zeros_like(Y) if yf is None else np.asarray(yf, float)
        dx = self.tx + self.mx * X - self.rx * Y + self.fmx * xf - self.frx * yf
        dy = self.ty + self.my * Y + self.ry * X + self.fmy * yf + self.fry * xf
        for (i, j), c in self.higher_x.items():
            dx = dx + c * X ** i * Y ** j
        for (i, j), c in self.higher_y.items():
            dy = dy + c * X ** i * Y ** j
        return dx, dy


def design_matrices(X: np.ndarray, Y: np.ndarray, xf: np.ndarray, yf: np.ndarray,
                    order: int = 1, intrafield: bool = True
                    ) -> tuple[np.ndarray, list[str]]:
    """Joint LS design matrix for stacked observations [dx; dy].

    Columns: tx, ty, mx, my, rx, ry, (fmx, fmy, frx, fry), then for each
    monomial X^i Y^j (2 <= i+j <= order) one column for dx and one for dy.
    """
    n = X.size
    z, o = np.zeros(n), np.ones(n)
    cols = {
        "tx": (o, z), "ty": (z, o), "mx": (X, z), "my": (z, Y),
        "rx": (-Y, z), "ry": (z, X),
    }
    if intrafield:
        cols.update({"fmx": (xf, z), "fmy": (z, yf), "frx": (-yf, z), "fry": (z, xf)})
    for (i, j) in _monomials(order):
        m = X ** i * Y ** j
        cols[f"kx_{i}{j}"] = (m, z)
        cols[f"ky_{i}{j}"] = (z, m)
    names = list(cols)
    A = np.column_stack([np.concatenate(cols[k]) for k in names])
    return A, names


@dataclass
class OverlayFit:
    model: GridModel
    residual_x: np.ndarray
    residual_y: np.ndarray
    names: list[str]
    coeffs: np.ndarray

    def mean_plus_3sigma(self) -> tuple[float, float]:
        """Per-axis |mean| + 3 sigma of the residuals [m]."""
        return (mean_plus_3sigma(self.residual_x), mean_plus_3sigma(self.residual_y))


def mean_plus_3sigma(r: np.ndarray) -> float:
    r = np.asarray(r, float)
    return float(abs(np.mean(r)) + 3 * np.std(r, ddof=1 if r.size > 1 else 0))


def fit_overlay(X: np.ndarray, Y: np.ndarray, xf: np.ndarray, yf: np.ndarray,
                dx: np.ndarray, dy: np.ndarray, order: int = 1,
                intrafield: bool = True) -> OverlayFit:
    """Least-squares fit of the grid (+ intra-field) model to displacements.

    Coordinates are scaled to O(1) internally for numerical conditioning.
    """
    X, Y, xf, yf = (np.asarray(a, float) for a in (X, Y, xf, yf))
    s = max(np.max(np.abs(X)), np.max(np.abs(Y)), 1e-12)
    sf = max(np.max(np.abs(xf)), np.max(np.abs(yf)), 1e-12)
    A, names = design_matrices(X / s, Y / s, xf / sf, yf / sf, order, intrafield)
    b = np.concatenate([dx, dy])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = b - A @ c
    n = X.size
    m = GridModel()
    hx, hy = {}, {}
    for name, v in zip(names, c):
        if name in ("mx", "my", "rx", "ry"):
            setattr(m, name, v / s)
        elif name in ("fmx", "fmy", "frx", "fry"):
            setattr(m, name, v / sf)
        elif name in ("tx", "ty"):
            setattr(m, name, v)
        else:
            axis, ij = name.split("_")
            i, j = int(ij[0]), int(ij[1])
            (hx if axis == "kx" else hy)[(i, j)] = v / s ** (i + j)
    m.higher_x, m.higher_y = hx, hy
    return OverlayFit(m, res[:n], res[n:], names, c)


def alignment_mark_positions(field_centers: np.ndarray,
                             field_size_m: tuple[float, float] = (26e-3, 33e-3),
                             marks_per_field: int = 4) -> tuple[np.ndarray, ...]:
    """Mark locations (X, Y, xf, yf): 4 corner-ish marks (or centre if 1) per field."""
    fw, fh = field_size_m
    if marks_per_field == 1:
        loc = np.array([[0.0, 0.0]])
    else:
        loc = np.array([[-0.4 * fw, -0.4 * fh], [0.4 * fw, -0.4 * fh],
                        [0.4 * fw, 0.4 * fh], [-0.4 * fw, 0.4 * fh]])[:marks_per_field]
    xf = np.tile(loc[:, 0], field_centers.shape[0])
    yf = np.tile(loc[:, 1], field_centers.shape[0])
    Xc = np.repeat(field_centers[:, 0], loc.shape[0])
    Yc = np.repeat(field_centers[:, 1], loc.shape[0])
    return Xc + xf, Yc + yf, xf, yf


@dataclass
class OverlayBudget:
    """Root-sum-square overlay budget, contributions as |mean|+3 sigma [m]."""

    contributions: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value_m: float) -> None:
        self.contributions[name] = float(value_m)

    @property
    def total(self) -> float:
        return float(np.sqrt(sum(v * v for v in self.contributions.values())))

    def dominant(self) -> str:
        return max(self.contributions, key=self.contributions.get)


def matched_machine_overlay_budget(stage_ma_nm: float = 0.8,
                                   alignment_nm: float = 0.6) -> OverlayBudget:
    """Illustrative matched-machine overlay (MMO) budget of a low-NA EUV tool.

    Order-of-magnitude public values (single-machine/dedicated-chuck overlay
    ~1 nm, matched-machine ~1.5-2.5 nm for NXE:3x00-class tools).  Values in nm
    (|mean|+3 sigma), returned in metres.
    """
    b = OverlayBudget()
    nm = 1e-9
    b.add("stage synchronisation (MA)", stage_ma_nm * nm)
    b.add("alignment sensor (incl. mark asymmetry)", alignment_nm * nm)
    b.add("wafer stage grid / encoder calibration", 0.6 * nm)
    b.add("lens distortion matching (machine-to-machine)", 0.8 * nm)
    b.add("reticle heating / reticle clamping", 0.5 * nm)
    b.add("wafer heating", 0.5 * nm)
    b.add("wafer clamping / chuck matching", 0.5 * nm)
    b.add("reticle writing / registration (matched)", 0.7 * nm)
    b.add("level-sensor induced (focus-to-overlay via telecentricity)", 0.3 * nm)
    return b
