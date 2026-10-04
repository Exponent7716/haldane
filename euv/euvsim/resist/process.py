"""Top-level resist process: exposure -> PEB -> development -> metrology.

Example
-------
>>> import numpy as np
>>> from euvsim.resist import ResistProcess, CAR
>>> x = np.arange(256) * 1.0
>>> aerial = np.tile(0.5 + 0.4 * np.cos(2 * np.pi * (x - 16) / 32), (128, 1))
>>> res = ResistProcess(CAR).expose(aerial, 1.0, 40.0, np.random.default_rng(0))
>>> res.metrics.cd_mean  # doctest: +SKIP
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from .bake import BakeState, post_exposure_bake
from .develop import DevelopModel, develop
from .exposure import ExposureState, expose_resist
from .materials import ResistMaterial
from .metrics import (ContactMetrics, DefectStats, FeatureKind, LineMetrics,
                      measure_contacts, measure_lines, monte_carlo_defects)

MetricKind = Optional[Literal["line", "contact"]]


@dataclass
class ResistResult:
    """Output of :meth:`ResistProcess.expose`.

    exposure / bake: intermediate states (photons, acid, deprotection, ...).
    latent_image: depth-averaged acid density after SE blur [nm^-3] (2-D).
    deprotection: depth-averaged deprotection after PEB (2-D).
    field: develop field (F > 0 where the exposed feature printed).
    printed: binary map of the printed exposed (clear-mask) feature.
    resist_remaining: binary map of resist left on the wafer.
    metrics: LineMetrics / ContactMetrics (or None).
    """

    dose_mj_cm2: float
    dx_nm: float
    exposure: ExposureState
    bake: BakeState
    latent_image: np.ndarray
    deprotection: np.ndarray
    field: np.ndarray
    printed: np.ndarray
    resist_remaining: np.ndarray
    metrics: Union[LineMetrics, ContactMetrics, None]

    @property
    def absorbed_photons_per_nm2(self) -> float:
        """Mean absorbed photons per nm^2 of wafer area."""
        return float(np.asarray(self.exposure.absorbed).sum(axis=0).mean() / self.dx_nm**2)


@dataclass
class ResistProcess:
    """A resist material plus process choices (development model, boundaries)."""

    material: ResistMaterial
    develop_model: DevelopModel = "threshold"
    boundary: str = "wrap"

    def expose(self, aerial: np.ndarray, dx_nm: float, dose_mj_cm2: float,
               rng: Optional[np.random.Generator] = None,
               metrics: MetricKind = "line") -> ResistResult:
        """Expose, bake and develop the resist under a normalised aerial image.

        ``rng=None`` gives the deterministic mean-field result (no shot noise).
        ``metrics`` selects the metrology: ``"line"`` (CD/LER/LWR of features
        along y), ``"contact"`` (CD/LCDU) or ``None``.
        """
        m = self.material
        ex = expose_resist(aerial, dx_nm, dose_mj_cm2, m, rng, self.boundary)
        bk = post_exposure_bake(ex.acid, dx_nm, m, ex.voxel_volume_nm3, rng, self.boundary)
        f, printed = develop(bk.deprotection, m, self.develop_model)
        remaining = printed if m.tone == "negative" else ~printed
        met: Union[LineMetrics, ContactMetrics, None] = None
        if metrics == "line":
            met = measure_lines(f, dx_nm)
        elif metrics == "contact":
            met = measure_contacts(printed, dx_nm, field=f)
        return ResistResult(dose_mj_cm2, dx_nm, ex, bk, ex.acid.mean(axis=0),
                            bk.deprotection.mean(axis=0), f, printed, remaining, met)

    def dose_to_size(self, aerial: np.ndarray, dx_nm: float, target_cd_nm: float,
                     feature: FeatureKind = "line", dose_range: tuple[float, float] = (2.0, 500.0),
                     tol_nm: float = 0.05, max_iter: int = 60) -> float:
        """Dose [mJ/cm^2] at which the mean-field CD of the exposed feature equals
        ``target_cd_nm`` (bisection in log-dose; CD increases monotonically with dose).
        """
        def cd(d: float) -> float:
            r = self.expose(aerial, dx_nm, d, None, metrics=feature)
            v = r.metrics.cd_mean
            return -np.inf if not np.isfinite(v) and r.printed.mean() < 0.5 else (
                np.inf if not np.isfinite(v) else v)

        lo, hi = math.log(dose_range[0]), math.log(dose_range[1])
        if cd(math.exp(lo)) > target_cd_nm or cd(math.exp(hi)) < target_cd_nm:
            raise ValueError("target CD not bracketed by dose_range")
        mid = 0.5 * (lo + hi)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            c = cd(math.exp(mid))
            if abs(c - target_cd_nm) < tol_nm:
                break
            if c < target_cd_nm:
                lo = mid
            else:
                hi = mid
        return math.exp(mid)

    def defect_probability(self, aerial: np.ndarray, dx_nm: float, dose_mj_cm2: float,
                           rng: np.random.Generator, n_trials: int = 20,
                           kind: FeatureKind = "line",
                           nominal: Optional[np.ndarray] = None) -> DefectStats:
        """Monte-Carlo stochastic failure probability per feature.

        ``nominal`` defaults to the mean-field printed pattern at the same dose.
        """
        if nominal is None:
            nominal = self.expose(aerial, dx_nm, dose_mj_cm2, None, metrics=None).printed
        sim = lambda g: self.expose(aerial, dx_nm, dose_mj_cm2, g, metrics=None).printed  # noqa: E731
        return monte_carlo_defects(sim, nominal, n_trials, rng, kind)
