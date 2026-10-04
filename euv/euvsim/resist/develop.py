"""Development: threshold model and Mack (1987) development-rate model.

Both models return a continuous *develop field* ``F(x, y)`` whose sign tells
whether the **exposed feature** printed (F > 0) and whose zero crossing is the
resist edge (used for sub-pixel CD / LER extraction).

* "Exposed feature" = the pattern under the clear mask region: a space
  (resist removed) for a positive-tone resist and a line (resist retained)
  for a negative-tone resist.

Threshold model
    F = <m>_z - m_th, with <m>_z the depth-averaged deprotection.

Mack development-rate model
    Dissolution rate as a function of the inhibitor fraction ``u``
    (u = 1 - m for positive tone, u = m for negative tone)::

        r(u) = r_max (a + 1)(1 - u)^n / (a + (1 - u)^n) + r_min
        a    = (n + 1)/(n - 1) * (1 - m_th)^n

    Each pixel column develops vertically (1-D column approximation); the
    time needed to clear the column is T = sum_z dz / r(u_z). The column
    clears if T < t_dev, so::

        F = +ln(t_dev / T)   (positive tone: clearing = exposed feature)
        F = -ln(t_dev / T)   (negative tone: remaining = exposed feature)
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .materials import ResistMaterial

DevelopModel = Literal["threshold", "mack"]


def threshold_field(deprotection: np.ndarray, threshold: float) -> np.ndarray:
    """F = depth-averaged deprotection minus threshold."""
    m = np.asarray(deprotection, dtype=float)
    if m.ndim == 3:
        m = m.mean(axis=0)
    return m - threshold


def mack_rate(inhibitor: np.ndarray, rmax: float, rmin: float, n: float, mth: float) -> np.ndarray:
    """Mack development rate r(u) [nm/s] for inhibitor fraction u."""
    a = (n + 1.0) / (n - 1.0) * (1.0 - mth) ** n
    s = (1.0 - np.clip(inhibitor, 0.0, 1.0)) ** n
    return rmax * (a + 1.0) * s / (a + s) + rmin


def mack_field(deprotection: np.ndarray, material: ResistMaterial) -> np.ndarray:
    """Develop field from the Mack model with 1-D vertical column development."""
    m = np.asarray(deprotection, dtype=float)
    if m.ndim == 2:
        m = m[None]
    u = 1.0 - m if material.tone == "positive" else m
    r = mack_rate(u, material.mack_rmax_nm_s, material.mack_rmin_nm_s,
                  material.mack_n, material.mack_mth)
    dz = material.thickness_nm / m.shape[0]
    t_clear = np.sum(dz / r, axis=0)
    f = np.log(material.t_dev_s / t_clear)
    return f if material.tone == "positive" else -f


def develop(deprotection: np.ndarray, material: ResistMaterial,
            model: DevelopModel = "threshold") -> tuple[np.ndarray, np.ndarray]:
    """Develop the resist.

    Returns ``(field, printed)`` where ``printed = field > 0`` marks where the
    exposed (clear-mask) feature printed. Resist remaining after develop is
    ``printed`` for negative tone and ``~printed`` for positive tone.
    """
    if model == "threshold":
        f = threshold_field(deprotection, material.threshold)
    elif model == "mack":
        f = mack_field(deprotection, material)
    else:
        raise ValueError(f"unknown development model {model!r}")
    return f, f > 0
