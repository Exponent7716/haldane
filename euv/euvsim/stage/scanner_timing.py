"""Wafer exposure timeline and time-/dose-limited throughput.

Dose-limited scan speed (derivation)
------------------------------------
Let ``P`` be the EUV power arriving at the wafer [W], spread uniformly over
the slit of width ``w`` (cross-scan, 26 mm) and height ``h`` (scan
direction, ~2 mm).  Irradiance in the slit:  ``E = P / (w h)`` [W/m^2].
A wafer point moving at speed ``v`` stays under the slit for ``t = h / v``,
so it receives the dose

    D = E t = P / (w h) * h / v = P / (w v)        [J/m^2]

The slit *height cancels*: a taller slit lowers the irradiance but lengthens
the dwell by the same factor.  Hence

    v_scan = P / (D w),     with D[J/m^2] = 10 * D[mJ/cm^2].

Example: P = 2 W at wafer, D = 30 mJ/cm^2 = 300 J/m^2, w = 26 mm ->
v = 2 / (300 * 0.026) = 0.256 m/s.  The scan speed actually used is
``min(v_dose, v_stage_max)``.  (The slit height still matters for the
number of pulses per point, N = h f / v, i.e. for dose stability, and for
the exposure time per field (H + h)/v.)

Timeline
--------
Per field: ``(H + h)/v`` exposure + settle + scan reversal + (non-overlapped)
step; per wafer: sum over fields + per-wafer overhead on the exposure stage
(wafer exchange between the two chucks, a short re-alignment / chuck
calibration).  In the dual-stage concept the measure side (wafer load,
alignment of many marks, full-wafer level-sensor map) runs in parallel, so
the wafer cycle is ``max(t_expose_side, t_measure_side) + t_swap``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .trajectory import MotionLimits, field_cycle_time


def max_scan_speed(dose_mj_cm2: float, power_at_wafer_w: float,
                   slit_width_mm: float = 26.0, slit_height_mm: float = 2.0,
                   v_stage_max: float = np.inf) -> float:
    """Dose-limited wafer scan speed v = P / (D w)  [m/s], capped at the stage limit.

    ``slit_height_mm`` is accepted for completeness but cancels (see module
    docstring).  ``dose_mj_cm2`` is converted with 1 mJ/cm^2 = 10 J/m^2.
    """
    if dose_mj_cm2 <= 0 or power_at_wafer_w <= 0:
        raise ValueError("dose and power must be positive")
    D = dose_mj_cm2 * 10.0
    w = slit_width_mm * 1e-3
    irradiance = power_at_wafer_w / (w * slit_height_mm * 1e-3)
    dwell_per_dose = D / irradiance           # required time under slit
    v = slit_height_mm * 1e-3 / dwell_per_dose  # = P/(D w)
    return float(min(v, v_stage_max))


def required_power_at_wafer(dose_mj_cm2: float, scan_speed: float,
                            slit_width_mm: float = 26.0) -> float:
    """Inverse relation P = D w v [W]."""
    return dose_mj_cm2 * 10.0 * slit_width_mm * 1e-3 * scan_speed


# ------------------------------------------------------------ field layout
@dataclass
class FieldLayout:
    """Field centres on the wafer [m] and classification."""

    centers: np.ndarray        # (N, 2) all exposed fields (intersecting usable area)
    full: np.ndarray           # bool (N,) field completely inside usable area
    field_size_m: tuple[float, float]
    offset_m: tuple[float, float]

    @property
    def n_fields(self) -> int:
        return int(self.centers.shape[0])

    @property
    def n_full(self) -> int:
        return int(self.full.sum())


def _layout(offset, fw, fh, R, min_area_fraction):
    nx = int(np.ceil(R / fw)) + 2
    ny = int(np.ceil(R / fh)) + 2
    xs = (np.arange(-nx, nx + 1)) * fw + offset[0]
    ys = (np.arange(-ny, ny + 1)) * fh + offset[1]
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()
    # corner test for full fields
    cx = np.abs(X) + fw / 2
    cy = np.abs(Y) + fh / 2
    full = cx ** 2 + cy ** 2 <= R ** 2
    # area fraction inside the disk via sub-sampling
    s = (np.arange(8) + 0.5) / 8 - 0.5
    SX, SY = np.meshgrid(s * fw, s * fh)
    inside = ((X[:, None] + SX.ravel()[None]) ** 2
              + (Y[:, None] + SY.ravel()[None]) ** 2 <= R ** 2).mean(axis=1)
    keep = inside > min_area_fraction
    return np.column_stack([X[keep], Y[keep]]), full[keep]


def field_layout(field_size_mm: tuple[float, float] = (26.0, 33.0),
                 wafer_diameter_mm: float = 300.0, edge_exclusion_mm: float = 3.0,
                 min_area_fraction: float = 0.05, optimise: bool = True) -> FieldLayout:
    """Rectangular field grid on the wafer.

    A field is *exposed* if more than ``min_area_fraction`` of it lies inside
    the usable radius ``R = D/2 - edge_exclusion`` (partial edge fields are
    exposed for process uniformity/yield of edge dies), and *full* if all
    four corners lie inside.  With ``optimise`` the grid offset (0 or half a
    field in x/y) maximising the number of full fields is chosen.
    """
    fw, fh = field_size_mm[0] * 1e-3, field_size_mm[1] * 1e-3
    R = (wafer_diameter_mm / 2 - edge_exclusion_mm) * 1e-3
    offsets = [(0.0, 0.0)]
    if optimise:
        offsets = [(ox * fw, oy * fh) for ox in (0, 0.5) for oy in (0, 0.5)]
    best = None
    for off in offsets:
        c, f = _layout(off, fw, fh, R, min_area_fraction)
        key = (f.sum(), -c.shape[0])
        if best is None or key > best[0]:
            best = (key, c, f, off)
    _, c, f, off = best
    return FieldLayout(c, f, (fw, fh), off)


# ------------------------------------------------------------- throughput
@dataclass
class WaferTimeline:
    """Breakdown of a wafer cycle [s] and resulting throughput."""

    n_fields: int
    scan_speed: float
    t_field: float
    t_expose_side: float
    t_measure_side: float
    t_swap: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def t_wafer(self) -> float:
        return max(self.t_expose_side, self.t_measure_side) + self.t_swap

    @property
    def wafers_per_hour(self) -> float:
        return 3600.0 / self.t_wafer


@dataclass
class TimelineParams:
    """Overheads of the exposure and measure sides (illustrative public orders)."""

    settle_time: float = 5e-3
    step_overlap: float = 0.8           # fraction of x-step hidden in scan reversal
    swap_time: float = 5.0              # chuck exchange + start-up of exposure side [s]
    expose_side_wafer_overhead: float = 1.0  # TIS/reticle align, first-field ramp [s]
    n_alignment_marks: int = 40
    t_per_mark: float = 0.05            # move + measure per alignment mark [s]
    wafer_load_time: float = 3.0        # unload/load on measure chuck [s]
    level_sensor_width_m: float = 26e-3
    level_scan_speed: float = 0.8       # m/s
    level_overhead: float = 1.0


def wafer_timeline(scan_speed: float, limits: MotionLimits | None = None,
                   field_size_mm: tuple[float, float] = (26.0, 33.0),
                   slit_height_mm: float = 2.0, n_fields: int | None = None,
                   params: TimelineParams | None = None,
                   wafer_diameter_mm: float = 300.0) -> WaferTimeline:
    """Time-limited wafer cycle for a dual-stage (measure/expose) scanner."""
    limits = limits or MotionLimits()
    p = params or TimelineParams()
    if n_fields is None:
        n_fields = field_layout(field_size_mm, wafer_diameter_mm).n_fields
    fc = field_cycle_time(scan_speed, limits, field_size_mm[1] * 1e-3,
                          field_size_mm[0] * 1e-3, slit_height_mm * 1e-3,
                          p.settle_time, order=4, step_overlap=p.step_overlap)
    t_expose = n_fields * fc["total"] + p.expose_side_wafer_overhead
    # level sensor: full-wafer map, swath width = LS array width
    D = wafer_diameter_mm * 1e-3
    n_swaths = int(np.ceil(D / p.level_sensor_width_m))
    t_level = n_swaths * (D / p.level_scan_speed + 0.1) + p.level_overhead
    t_align = p.n_alignment_marks * p.t_per_mark
    t_measure = p.wafer_load_time + t_align + t_level
    comps = dict(fc)
    comps.update(t_align=t_align, t_level=t_level, t_load=p.wafer_load_time)
    return WaferTimeline(n_fields, scan_speed, fc["total"], t_expose, t_measure,
                         p.swap_time, comps)


def throughput_wph(dose_mj_cm2: float, power_at_wafer_w: float,
                   limits: MotionLimits | None = None,
                   params: TimelineParams | None = None,
                   field_size_mm: tuple[float, float] = (26.0, 33.0),
                   slit_width_mm: float = 26.0, slit_height_mm: float = 2.0) -> WaferTimeline:
    """Throughput with the scan speed set by dose: v = min(P/(D w), v_max)."""
    limits = limits or MotionLimits()
    v = max_scan_speed(dose_mj_cm2, power_at_wafer_w, slit_width_mm,
                       slit_height_mm, limits.v_max)
    return wafer_timeline(v, limits, field_size_mm, slit_height_mm, params=params)
