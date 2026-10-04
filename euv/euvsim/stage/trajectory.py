"""Setpoint (reference trajectory) generation for step-and-scan stages.

Method
------
We use the *convolution* construction of higher-order setpoints that is
standard in high-precision motion control (Lambrechts, Boerlage & Steinbuch,
"Trajectory planning and feedforward design for electromechanical motion
systems", Control Eng. Practice 13 (2005) 145):

A velocity rectangle of height ``h`` and length ``T1`` is convolved with
``n-1`` *normalised* boxes (unit area) of lengths ``T2 >= T3 >= ... >= Tn``.
Each convolution adds one bounded derivative:

    v_peak = h,  a_peak = h/T2,  j_peak = h/(T2 T3),  s_peak = h/(T2 T3 T4)

and the total duration is ``T1 + T2 + ... + Tn``.  Order 2 (``n=2``) is a
trapezoidal velocity (acceleration-limited), order 3 is the jerk-limited
"S-curve" and order 4 is snap-limited (used on wafer stages so that the
feedforward of the dominant flexible mode, F = m1 m2/k * snap, is bounded).

For a point-to-point move of length ``D`` the ideal lengths are
``T1 = D/v_max, T2 = v_max/a_max, T3 = a_max/j_max, T4 = j_max/s_max``; when
these violate the required ordering (short moves) adjacent lengths are pooled
to their geometric mean (pool-adjacent-violators in log space), which keeps
all prefix products, hence all derivative limits, satisfied.

Scanning
--------
During exposure the wafer stage must move at a *constant* scan speed ``v``
over ``L_exp = field_height + slit_height`` (every wafer point must cross the
whole slit) after a settle distance ``v * t_settle``.  The scan profile is the
same convolution with ``T1 = (L_exp + v t_settle)/v + T2 + ... + Tn``.

The reticle stage follows the wafer with the projection magnification
``M`` (4x for NA 0.33; 4x in x but 8x in the scan direction y for the
anamorphic NA 0.55 lens, which therefore prints a half field 26 x 16.5 mm):
    x_reticle(t) = -M_y * x_wafer(t)   (sign: image inversion)
so reticle velocity and acceleration are M_y times larger.

Units: SI (m, s) throughout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

G0 = 9.80665  # m/s^2


@dataclass
class MotionLimits:
    """Kinematic limits of a stage axis (SI units).

    Typical public values for an EUV wafer stage: v ~ 0.3-0.8 m/s scan,
    a ~ 5-10 g (50-100 m/s^2), jerk ~ 1e4-1e5 m/s^3, snap ~ 1e7-1e8 m/s^4.
    """

    v_max: float = 0.6
    a_max: float = 7.0 * G0
    j_max: float = 3.0e4
    s_max: float = 3.0e7

    def as_list(self, order: int) -> list[float]:
        lims = [self.v_max, self.a_max, self.j_max, self.s_max]
        if order < 2 or order > 4:
            raise ValueError("order must be 2, 3 or 4")
        return lims[:order]

    def scaled(self, factor: float) -> "MotionLimits":
        """Limits multiplied by ``factor`` (e.g. reticle = magnification x wafer)."""
        return MotionLimits(self.v_max * factor, self.a_max * factor,
                            self.j_max * factor, self.s_max * factor)


@dataclass
class Trajectory:
    """Sampled setpoint: time and derivatives of position (SI)."""

    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    jerk: np.ndarray
    snap: np.ndarray
    segment_times: tuple[float, ...] = field(default=())

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def peaks(self) -> dict[str, float]:
        """Peak absolute value of each derivative."""
        return {k: float(np.max(np.abs(getattr(self, k))))
                for k in ("vel", "acc", "jerk", "snap")}

    def scaled(self, factor: float) -> "Trajectory":
        """Trajectory multiplied by ``factor`` (e.g. -M for the reticle stage)."""
        return Trajectory(self.t.copy(), self.pos * factor, self.vel * factor,
                          self.acc * factor, self.jerk * factor,
                          self.snap * factor, self.segment_times)


def _pool_lengths(ideal: Sequence[float]) -> list[float]:
    """Make lengths non-increasing by pooling violators to their geometric mean.

    Pooling preserves the product of each pooled block, so the highest
    derivative limit of the block stays tight while lower ones are relaxed.
    """
    blocks: list[list[float]] = []  # [sum_log, count]
    for T in ideal:
        blocks.append([np.log(T), 1])
        while len(blocks) > 1 and (blocks[-2][0] / blocks[-2][1]
                                   < blocks[-1][0] / blocks[-1][1]):
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out: list[float] = []
    for s, c in blocks:
        out.extend([float(np.exp(s / c))] * c)
    return out


def segment_lengths(distance: float, limits: Sequence[float]) -> list[float]:
    """Box lengths [T1..Tn] for a point-to-point move of ``distance``.

    ``limits`` = [v_max, a_max, (j_max, (s_max))].  Ideal values are
    T1 = D/v, T_k = lim_{k-1}/lim_k; ordering is then enforced by pooling.
    """
    D = abs(distance)
    if D <= 0:
        raise ValueError("distance must be non-zero")
    ideal = [D / limits[0]] + [limits[k - 1] / limits[k]
                               for k in range(1, len(limits))]
    return _pool_lengths(ideal)


def _derivatives(vel: np.ndarray, dt: float):
    """Position and higher derivatives consistent with a zero-order-hold force.

    Acceleration is taken piecewise constant on [t_k, t_k+1):
    ``acc_k = (v_{k+1} - v_k)/dt`` (forward difference), so that position by
    trapezoidal integration ``x_{k+1} = x_k + (v_k + v_{k+1}) dt/2`` is
    exactly what a rigid mass driven by ``F_k = m acc_k`` (ZOH) follows.  This
    makes acceleration feedforward exact in discrete time.
    """
    pos = np.concatenate([[0.0], np.cumsum(0.5 * (vel[1:] + vel[:-1])) * dt])
    acc = np.diff(vel, append=vel[-1]) / dt
    jerk = np.diff(acc, append=acc[-1]) / dt
    snap = np.diff(jerk, append=jerk[-1]) / dt
    return pos, acc, jerk, snap


def _convolution_profile(height: float, lengths: Sequence[float], dt: float,
                         t_pad: float = 0.0) -> Trajectory:
    """Sample a convolution profile on a uniform grid of step ``dt``.

    Each box length is rounded *up* to an integer number of samples, which can
    only lower the derivative peaks (conservative w.r.t. the limits).  The
    velocity height is rescaled so that the travelled distance is preserved.
    """
    n = [max(1, int(np.ceil(T / dt - 1e-9))) for T in lengths]
    distance = height * lengths[0]
    v = np.ones(n[0]) * distance / (n[0] * dt)
    for nk in n[1:]:
        v = np.convolve(v, np.ones(nk) / nk)
    npad = int(round(t_pad / dt))
    v = np.concatenate([np.zeros(1), v, np.zeros(npad + 1)])
    pos, acc, jerk, snap = _derivatives(v, dt)
    t = np.arange(v.size) * dt
    return Trajectory(t, pos, v, acc, jerk, snap,
                      tuple(nk * dt for nk in n))


def _ramp_lengths(v: float, lims: Sequence[float]) -> list[float]:
    """Box lengths [T2..Tn] of a velocity ramp 0 -> v: [v/a, a/j, j/s] pooled."""
    ideal = [v / lims[1]] + [lims[k - 1] / lims[k] for k in range(2, len(lims))]
    return _pool_lengths(ideal)


def _peak_ratio(tr: "Trajectory", lims: Sequence[float], first: int) -> float:
    """max_k (peak_k / lim_k)^(1/(k - first + 1)) over derivatives k >= first.

    ``first`` = 0 for a point-to-point move (distance fixed: the k-th
    derivative scales as lambda^-(k+1) under time scaling by lambda), 1 for a
    velocity ramp (velocity fixed: scales as lambda^-k).
    """
    pk = [np.max(np.abs(a)) for a in (tr.vel, tr.acc, tr.jerk, tr.snap)]
    r = 0.0
    for k in range(first, len(lims)):
        r = max(r, (pk[k] / lims[k]) ** (1.0 / (k - first + 1)))
    return r


def _feasible_p2p(distance: float, lims: Sequence[float], dt: float,
                  max_iter: int = 30) -> tuple[list[float], "Trajectory"]:
    """Pooled lengths, uniformly time-scaled until the *sampled* profile obeys all limits.

    Pooling alone guarantees the limits only when same-sign impulses of the
    highest derivative do not overlap (T_i - T_{i+1} >= T_{i+2} ...); for
    short moves they do, and a uniform time scaling lambda (derivative k
    reduced by lambda^k) restores feasibility.
    """
    lengths = segment_lengths(distance, lims)
    for _ in range(max_iter):
        tr = _convolution_profile(1.0, lengths, dt)
        tr = tr.scaled(distance / tr.pos[-1])
        r = _peak_ratio(tr, lims, 0)
        if r <= 1.0 + 1e-9:
            break
        lengths = [T * r * 1.0002 for T in lengths]
    return lengths, tr


def _feasible_ramp(v: float, lims: Sequence[float], dt: float,
                   max_iter: int = 30) -> list[int]:
    """Sample counts of the ramp boxes [T2..Tn] such that the sampled ramp obeys limits."""
    ramp = _ramp_lengths(v, lims)
    for _ in range(max_iter):
        n_ramp = [max(1, int(np.ceil(T / dt - 1e-9))) for T in ramp]
        vel = np.ones(sum(n_ramp) + 2) * v
        vel = np.concatenate([np.zeros(1), vel])
        for nk in n_ramp:
            vel = np.convolve(vel, np.ones(nk) / nk)[: vel.size]
        pos, acc, jerk, snap = _derivatives(vel, dt)
        acc, jerk, snap = acc[:-2], jerk[:-3], snap[:-4]  # drop end artefacts
        tr = Trajectory(np.zeros(1), pos, vel, acc, jerk, snap)
        r = _peak_ratio(tr, lims, 1)
        if r <= 1.0 + 1e-9:
            break
        ramp = [nk * dt * r * 1.0002 for nk in n_ramp]
    return n_ramp


def point_to_point(distance: float, limits: MotionLimits, order: int = 3,
                   dt: float = 1e-4) -> Trajectory:
    """Time-near-optimal point-to-point setpoint (e.g. stepping between fields).

    Parameters
    ----------
    distance : move length [m] (sign gives direction).
    limits : kinematic limits.
    order : 2 (acc-limited), 3 (jerk-limited) or 4 (snap-limited).
    dt : sample time [s].
    """
    _, tr = _feasible_p2p(distance, limits.as_list(order), dt)
    return tr


def move_time(distance: float, limits: MotionLimits, order: int = 3) -> float:
    """Duration sum(T_k) of a point-to-point move (evaluated on a fine grid)."""
    lims = limits.as_list(order)
    T = segment_lengths(distance, lims)
    lengths, _ = _feasible_p2p(distance, lims, min(T) / 200.0)
    return float(sum(lengths))


@dataclass
class ScanProfile:
    """A complete single-field scan of the wafer stage plus reticle setpoint."""

    wafer: Trajectory
    reticle: Trajectory
    scan_speed: float
    t_exposure_start: float
    t_exposure_end: float
    magnification: float

    @property
    def exposure_time(self) -> float:
        return self.t_exposure_end - self.t_exposure_start

    @property
    def exposure_mask(self) -> np.ndarray:
        t = self.wafer.t
        return (t >= self.t_exposure_start) & (t <= self.t_exposure_end)


def exposure_time_per_field(field_height_m: float, slit_height_m: float,
                            scan_speed: float) -> float:
    """t_exp = (H_field + h_slit) / v_scan.

    Every point of the field must traverse the full slit height, so the slit
    leading edge enters the field one slit height before the field starts.
    """
    return (field_height_m + slit_height_m) / scan_speed


def scan_profile(scan_speed: float, limits: MotionLimits,
                 field_height_m: float = 33e-3, slit_height_m: float = 2e-3,
                 settle_time: float = 5e-3, order: int = 4, dt: float = 5e-5,
                 magnification: float = 4.0, t_pad: float = 0.0) -> ScanProfile:
    """Wafer + reticle setpoints for scanning one field.

    The wafer velocity profile is a rectangle of height ``v`` and length
    ``T1 = (L_exp + v t_settle)/v + sum_{k>=2} T_k`` convolved with boxes
    ``T2 = v/a, T3 = a/j, T4 = j/s`` (pooled), giving a constant velocity
    plateau of exactly ``L_exp/v + t_settle``.  The reticle setpoint is
    ``-M`` times the wafer setpoint (M = 4, or 8 in y for high-NA).
    """
    v = scan_speed
    if v > limits.v_max * (1 + 1e-12):
        raise ValueError("scan speed exceeds stage v_max")
    lims = limits.as_list(order)
    n_ramp = _feasible_ramp(v, lims, dt)
    ramp_q = [nk * dt for nk in n_ramp]
    t_plateau = exposure_time_per_field(field_height_m, slit_height_m, v) + settle_time
    # discrete convolution plateau = n1 - sum(n_k - 1) samples
    n1 = int(np.ceil(t_plateau / dt)) + 1 + sum(nk - 1 for nk in n_ramp)
    vel = np.ones(n1) * v
    for nk in n_ramp:
        vel = np.convolve(vel, np.ones(nk) / nk)
    npad = int(round(t_pad / dt))
    vel = np.concatenate([np.zeros(npad + 1), vel, np.zeros(npad + 1)])
    pos, acc, jerk, snap = _derivatives(vel, dt)
    t = np.arange(vel.size) * dt
    wafer = Trajectory(t, pos, vel, acc, jerk, snap, tuple([n1 * dt] + ramp_q))
    # constant-velocity plateau
    plateau = np.flatnonzero(np.isclose(vel, v, rtol=1e-9, atol=0))
    t0 = t[plateau[0]] + settle_time
    t1 = t0 + exposure_time_per_field(field_height_m, slit_height_m, v)
    t1 = min(t1, t[plateau[-1]])
    reticle = wafer.scaled(-magnification)
    return ScanProfile(wafer, reticle, v, float(t0), float(t1), magnification)


def field_cycle_time(scan_speed: float, limits: MotionLimits,
                     field_height_m: float = 33e-3, field_width_m: float = 26e-3,
                     slit_height_m: float = 2e-3, settle_time: float = 5e-3,
                     order: int = 4, step_overlap: float = 0.5) -> dict[str, float]:
    """Time budget of one field in a meander (alternating scan direction).

    * ``t_exposure`` = (H + h)/v
    * ``t_ramp`` = deceleration + acceleration (scan reversal) = 2 sum T_k
    * ``t_settle``
    * ``t_step`` = x step of one field width, which runs largely in parallel
      with the y scan reversal; only ``(1-step_overlap)`` of it adds time.

    Returns a dict with the components and ``total``.
    """
    lims = limits.as_list(order)
    dt_fine = min(_ramp_lengths(scan_speed, lims)) / 100.0
    ramp = sum(_feasible_ramp(scan_speed, lims, dt_fine)) * dt_fine
    t_exp = exposure_time_per_field(field_height_m, slit_height_m, scan_speed)
    t_step = move_time(field_width_m, limits, order)
    t_rev = 2.0 * ramp
    t_step_extra = max(0.0, t_step - step_overlap * t_rev)
    total = t_exp + settle_time + t_rev + t_step_extra
    return {"t_exposure": t_exp, "t_settle": settle_time, "t_reversal": t_rev,
            "t_step": t_step, "t_step_extra": t_step_extra, "total": total}
