"""Wafer stage and reticle stage motion system (TWINSCAN-like dual stage).

Physical picture
----------------
Both the wafer stage and the reticle stage are built as a *long-stroke*
linear-motor stage (coarse, um-level accuracy, carries cables and the cable
slab) carrying a magnetically levitated *short-stroke* mover (fine stage,
6-DoF Lorentz actuators, ~20 kg for the wafer side).  The short stroke is what
is servo-controlled to nanometre accuracy; it is modelled here as a rigid body
(optionally with one flexible internal mode) driven by a force actuator.

During exposure the reticle moves ``REDUCTION`` times faster than the wafer so
that the reticle image (de-magnified by the projection lens) and the wafer move
synchronously under the static exposure slit.  The quality of that
synchronisation is summarised by the moving average (MA) and moving standard
deviation (MSD) of the synchronisation error over the time a wafer point needs
to cross the slit (``slit_height / scan_speed``):

* MA  -> a placement (overlay) error of the printed image,
* MSD -> an image blur that reduces the aerial-image contrast by
  ``exp(-2 pi^2 MSD^2 / pitch^2)`` (Gaussian blur MTF at frequency 1/pitch).

The TWINSCAN dual-stage concept is modelled in :class:`DualStage`: while one
chuck is exposed under the lens, the other chuck is loaded, aligned and its
height map measured (levelled); the chucks are then swapped.

Units
-----
Lengths are in nm unless the name says otherwise (``_mm``, ``_m``); time in
seconds.  Internally :class:`ScanProfile` and :class:`ServoLoop` work in SI
units (m, m/s, m/s^2, m/s^3, N, kg); servo errors are reported in nm.

Sign convention for synchronisation
-----------------------------------
The projection lens inverts the image, so physically the reticle moves in the
opposite direction to the wafer.  In this module reticle positions/errors are
expressed in a *reticle coordinate frame whose axis is already flipped*, so a
perfectly synchronised reticle satisfies ``x_reticle = REDUCTION * x_wafer``.
With that convention::

    sync_error = wafer_error - reticle_error / REDUCTION     (wafer scale, nm)

and each stage error is defined as ``setpoint - actual position``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from .core import (
    FIELD_HEIGHT_MM,
    FIELD_WIDTH_MM,
    REDUCTION,
    SLIT_HEIGHT_MM,
    WAFER_DIAMETER_MM,
)

__all__ = [
    "ScanProfile",
    "ServoLoop",
    "PositionSensor",
    "DualStage",
    "ma_msd",
    "ma_msd_stats",
    "synchronization_error",
    "simulate_synchronization",
    "msd_contrast_loss",
    "field_layout",
    "scan_directions",
    "ramp_time",
    "ramp_distance",
    "point_to_point_time",
    "throughput",
]


# =============================================================================
# Jerk-limited (3rd-order) motion primitives
# =============================================================================
def ramp_time(v: float, a_max: float, j_max: float) -> float:
    """Time of a jerk-limited velocity ramp from 0 to ``v`` (SI units).

    Third-order (S-curve) profile: jerk +j, (constant acceleration), jerk -j.
    If ``v < a_max^2 / j_max`` the constant-acceleration phase vanishes and the
    peak acceleration is ``sqrt(v * j_max)``.
    """
    v = abs(v)
    if v <= 0:
        return 0.0
    if v >= a_max**2 / j_max:
        return v / a_max + a_max / j_max
    return 2.0 * np.sqrt(v / j_max)


def ramp_distance(v: float, a_max: float, j_max: float) -> float:
    """Distance travelled during a jerk-limited ramp 0 -> ``v`` (symmetric profile)."""
    return 0.5 * abs(v) * ramp_time(v, a_max, j_max)


def _ramp_segments(v: float, a_max: float, j_max: float, sign: float = 1.0) -> list[tuple[float, float]]:
    """(duration, jerk) segments of a jerk-limited ramp changing velocity by ``sign*v``."""
    v = abs(v)
    if v <= 0:
        return []
    if v >= a_max**2 / j_max:
        tj = a_max / j_max
        ta = v / a_max - tj
    else:
        tj = np.sqrt(v / j_max)
        ta = 0.0
    return [(tj, sign * j_max), (ta, 0.0), (tj, -sign * j_max)]


def point_to_point_time(distance_m: float, v_max: float, a_max: float, j_max: float) -> float:
    """Minimum time of a jerk-limited rest-to-rest move over ``distance_m`` (SI units)."""
    d = abs(distance_m)
    if d <= 0:
        return 0.0
    if 2 * ramp_distance(v_max, a_max, j_max) <= d:
        return 2 * ramp_time(v_max, a_max, j_max) + (d - 2 * ramp_distance(v_max, a_max, j_max)) / v_max
    lo, hi = 0.0, v_max
    for _ in range(60):  # bisection on the peak velocity
        mid = 0.5 * (lo + hi)
        if 2 * ramp_distance(mid, a_max, j_max) > d:
            hi = mid
        else:
            lo = mid
    return 2 * ramp_time(lo, a_max, j_max)


def _eval_segments(segments: list[tuple[float, float]], t: np.ndarray, x0=(0.0, 0.0, 0.0)):
    """Evaluate piecewise-constant-jerk segments exactly at times ``t``.

    Returns (pos, vel, acc, jerk) arrays.  Before the first segment the initial
    state is held; after the last one the state is extrapolated with zero jerk
    and the final acceleration (which is zero for a complete profile).
    """
    t = np.asarray(t, dtype=float)
    p = np.full_like(t, x0[0])
    v = np.full_like(t, x0[1])
    a = np.full_like(t, x0[2])
    j = np.zeros_like(t)
    p0, v0, a0 = x0
    t0 = 0.0
    for k, (dur, jk) in enumerate(segments):
        last = k == len(segments) - 1
        m = (t >= t0) & ((t < t0 + dur) | last)
        tau = t[m] - t0
        if last:
            # beyond the final segment: zero jerk continuation
            tau_c = np.minimum(tau, dur)
            ex = tau - tau_c
        else:
            tau_c, ex = tau, 0.0
        ac = a0 + jk * tau_c
        vc = v0 + a0 * tau_c + 0.5 * jk * tau_c**2
        pc = p0 + v0 * tau_c + 0.5 * a0 * tau_c**2 + jk * tau_c**3 / 6.0
        p[m] = pc + vc * ex + 0.5 * ac * ex**2
        v[m] = vc + ac * ex
        a[m] = ac
        j[m] = np.where(np.asarray(ex) > 0, 0.0, jk) if last else jk
        p0, v0, a0 = (
            p0 + v0 * dur + 0.5 * a0 * dur**2 + jk * dur**3 / 6.0,
            v0 + a0 * dur + 0.5 * jk * dur**2,
            a0 + jk * dur,
        )
        t0 += dur
    return p, v, a, j


# =============================================================================
# Scan profile
# =============================================================================
@dataclass
class ScanProfile:
    """Jerk-limited (3rd-order, S-curve) setpoint for scanning one field.

    Phases (all along the scan axis y)::

        accel ramp | settle (const v) | exposure (const v) | decel ramp

    The exposure phase covers ``field_height + slit_height`` at constant speed:
    the first field line enters the slit at its start and the last field line
    leaves the slit at its end.  The settle phase lets the servo error decay
    after the acceleration transient before the first line is exposed.

    Default numbers are for the wafer stage (scan speed 700 mm/s, 30 m/s^2,
    5000 m/s^3).  :meth:`reticle_profile` returns the matching reticle stage
    profile (everything scaled by ``REDUCTION``; timing is identical).

    Parameters
    ----------
    scan_speed_mm_s : constant scan speed during exposure (mm/s).
    max_accel_m_s2  : acceleration limit (m/s^2).
    max_jerk_m_s3   : jerk limit (m/s^3).
    field_height_mm, slit_height_mm : scanned field length and slit height (mm).
    settle_time_s   : constant-velocity settling time before exposure (s).
    direction       : +1 or -1, scan direction.
    """

    scan_speed_mm_s: float = 700.0
    max_accel_m_s2: float = 30.0
    max_jerk_m_s3: float = 5000.0
    field_height_mm: float = FIELD_HEIGHT_MM
    slit_height_mm: float = SLIT_HEIGHT_MM
    settle_time_s: float = 0.005
    direction: int = 1

    # ---- derived timing -----------------------------------------------------
    @property
    def scan_speed_m_s(self) -> float:
        return self.scan_speed_mm_s * 1e-3

    @property
    def peak_accel_m_s2(self) -> float:
        """Actually reached acceleration (may be below the limit for slow scans)."""
        v, a, j = self.scan_speed_m_s, self.max_accel_m_s2, self.max_jerk_m_s3
        return a if v >= a**2 / j else float(np.sqrt(v * j))

    def ramp_time(self) -> float:
        """Duration of the acceleration (or deceleration) ramp (s)."""
        return ramp_time(self.scan_speed_m_s, self.max_accel_m_s2, self.max_jerk_m_s3)

    def ramp_distance_mm(self) -> float:
        return 1e3 * ramp_distance(self.scan_speed_m_s, self.max_accel_m_s2, self.max_jerk_m_s3)

    def exposure_length_mm(self) -> float:
        """Constant-velocity travel needed to expose the whole field (field + slit)."""
        return self.field_height_mm + self.slit_height_mm

    def exposure_time_s(self) -> float:
        return self.exposure_length_mm() / self.scan_speed_mm_s

    def slit_time_s(self) -> float:
        """Time a wafer point spends under the slit (= MA/MSD window)."""
        return self.slit_height_mm / self.scan_speed_mm_s

    def exposure_window(self) -> tuple[float, float]:
        """(start, end) time of the exposure phase (s)."""
        t0 = self.ramp_time() + self.settle_time_s
        return t0, t0 + self.exposure_time_s()

    def constant_velocity_window(self) -> tuple[float, float]:
        """(start, end) time of the whole constant-velocity phase (settle + exposure)."""
        tr = self.ramp_time()
        return tr, tr + self.settle_time_s + self.exposure_time_s()

    def time_per_field(self) -> float:
        """Total scan time of one field including both ramps and settling (s)."""
        return 2 * self.ramp_time() + self.settle_time_s + self.exposure_time_s()

    def scan_length_mm(self) -> float:
        """Total stage travel of one scan move (mm)."""
        return 2 * self.ramp_distance_mm() + self.scan_speed_mm_s * (self.settle_time_s + self.exposure_time_s())

    # ---- setpoint generation ----------------------------------------------
    def segments(self) -> list[tuple[float, float]]:
        """(duration, jerk) list describing the whole scan move."""
        s = float(np.sign(self.direction) or 1.0)
        v, a, j = self.scan_speed_m_s, self.max_accel_m_s2, self.max_jerk_m_s3
        return (
            _ramp_segments(v, a, j, s)
            + [(self.settle_time_s + self.exposure_time_s(), 0.0)]
            + _ramp_segments(v, a, j, -s)
        )

    def setpoint(self, t) -> dict:
        """Setpoint at times ``t`` (s): dict(pos_m, vel_m_s, acc_m_s2, jerk_m_s3)."""
        p, v, a, j = _eval_segments(self.segments(), np.atleast_1d(t))
        return dict(pos_m=p, vel_m_s=v, acc_m_s2=a, jerk_m_s3=j)

    def sample(self, fs_hz: float = 20e3, pre_s: float = 0.0, post_s: float = 0.005) -> dict:
        """Sampled setpoint on a uniform grid at ``fs_hz``.

        Returns the :meth:`setpoint` arrays plus ``t``, boolean masks
        ``exposure`` and ``constant_velocity`` and the exposure window times.
        """
        dt = 1.0 / fs_hz
        t = np.arange(-pre_s, self.time_per_field() + post_s, dt)
        sp = self.setpoint(t)
        t0, t1 = self.exposure_window()
        c0, c1 = self.constant_velocity_window()
        sp.update(
            t=t,
            dt=dt,
            exposure=(t >= t0) & (t <= t1),
            constant_velocity=(t >= c0) & (t <= c1),
            t_exposure=(t0, t1),
        )
        return sp

    def reticle_profile(self, reduction: float = REDUCTION) -> "ScanProfile":
        """Matching reticle-stage profile: all lengths/speeds/accels scaled by ``reduction``."""
        return ScanProfile(
            scan_speed_mm_s=self.scan_speed_mm_s * reduction,
            max_accel_m_s2=self.max_accel_m_s2 * reduction,
            max_jerk_m_s3=self.max_jerk_m_s3 * reduction,
            field_height_mm=self.field_height_mm * reduction,
            slit_height_mm=self.slit_height_mm * reduction,
            settle_time_s=self.settle_time_s,
            direction=self.direction,
        )


# =============================================================================
# Position sensors
# =============================================================================
@dataclass
class PositionSensor:
    """Stage position sensor noise model.

    ``kind="encoder"``: planar grating encoder (short optical path in a
    controlled volume) -> white noise ``white_nm`` (default 0.03 nm rms) plus a
    small slow drift ``drift_nm`` (0.02 nm rms).

    ``kind="interferometer"``: displacement interferometer whose beam travels
    ``path_length_mm`` through air.  Refractive-index fluctuations from air
    turbulence give a slow, correlated error (AR(1) process, correlation time
    ``turbulence_tau_s``).  Index fluctuations of independent turbulent cells add
    incoherently, so the error variance grows linearly with path length::

        sigma_turb = turbulence_nm_100mm * sqrt(path_length_mm / 100)

    (default 0.8 nm per sqrt(100 mm), i.e. ~1.4 nm rms at 300 mm), plus white
    electronics noise ``white_nm``.
    """

    kind: str = "encoder"
    path_length_mm: float = 300.0
    sample_rate_hz: float = 20e3
    white_nm: float | None = None
    drift_nm: float | None = None
    turbulence_nm_100mm: float = 0.8
    turbulence_tau_s: float = 5e-3

    def __post_init__(self):
        if self.kind not in ("encoder", "interferometer"):
            raise ValueError("kind must be 'encoder' or 'interferometer'")
        if self.white_nm is None:
            self.white_nm = 0.03 if self.kind == "encoder" else 0.1
        if self.drift_nm is None:
            self.drift_nm = 0.02 if self.kind == "encoder" else 0.0

    @property
    def correlated_sigma_nm(self) -> float:
        """rms of the slow (correlated) error component (nm)."""
        if self.kind == "interferometer":
            turb = self.turbulence_nm_100mm * np.sqrt(max(self.path_length_mm, 0.0) / 100.0)
            return float(np.hypot(turb, self.drift_nm))
        return float(self.drift_nm)

    @property
    def sigma_nm(self) -> float:
        """Total rms noise (nm)."""
        return float(np.hypot(self.white_nm, self.correlated_sigma_nm))

    def noise(self, n: int, rng: np.random.Generator, dt: float | None = None) -> np.ndarray:
        """``n`` consecutive noise samples (nm) at interval ``dt`` (default 1/sample_rate)."""
        dt = 1.0 / self.sample_rate_hz if dt is None else dt
        out = self.white_nm * rng.standard_normal(n)
        sc = self.correlated_sigma_nm
        if sc > 0:
            tau = self.turbulence_tau_s if self.kind == "interferometer" else 50e-3
            phi = np.exp(-dt / tau)
            w = rng.standard_normal(n) * sc * np.sqrt(1 - phi**2)
            x = np.empty(n)
            x_prev = sc * rng.standard_normal()
            for k in range(n):  # AR(1)
                x_prev = phi * x_prev + w[k]
                x[k] = x_prev
            out = out + x
        return out

    def measure(self, true_pos_nm, rng: np.random.Generator, dt: float | None = None) -> np.ndarray:
        """Measured position (nm) for a sequence of true positions (nm)."""
        true_pos_nm = np.asarray(true_pos_nm, dtype=float)
        return true_pos_nm + self.noise(true_pos_nm.size, rng, dt).reshape(true_pos_nm.shape)


# =============================================================================
# Servo loop
# =============================================================================
@dataclass
class ServoLoop:
    """Discrete-time short-stroke servo: plant + PID feedback + acceleration feedforward.

    Plant: rigid body of mass ``mass_kg`` driven by a Lorentz actuator force.
    With ``flex_mode_hz`` set, the mover is split into an actuated body (with
    the sensor, collocated control) of mass ``(1-flex_mass_ratio)*m`` and a
    payload body (``flex_mass_ratio*m``) coupled by a spring/damper whose
    internal mode is at ``flex_mode_hz`` with damping ratio ``flex_damping``.
    The continuous plant is discretised exactly with zero-order hold (matrix
    exponential).

    Controller: PID with filtered derivative.  If gains are not given they are
    derived from ``bandwidth_hz`` (crossover w_c) with the standard motion
    tuning ``kp = m w_c^2 / 3, kd = m w_c, ki = kp w_c / 10``; the derivative
    is low-pass filtered at ``5 w_c``.  ``delay_samples`` models computation
    delay (actuator force is applied that many samples late).

    Feedforward: ``F_ff = ff_mass * a_ref`` with ``ff_mass = m * (1 + ff_mass_error)``,
    evaluated with preview to compensate for delay and ZOH.

    Disturbances (forces on the short stroke):
      * floor vibration transmitted via long-stroke / magnet coupling:
        sum of random-phase sinusoids 5-100 Hz, ``floor_force_n`` rms;
      * cable slab: ``cable_stiffness * x + cable_damping * v`` plus a
        position-periodic force ripple (``ripple_force_n`` amplitude, period
        ``ripple_pitch_mm``);
      * amplifier force noise, white, ``force_noise_n`` rms;
      * sensor noise from :class:`PositionSensor`.
    """

    mass_kg: float = 20.0
    bandwidth_hz: float = 400.0
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    feedforward: bool = True
    ff_mass_error: float = 0.002
    fs_hz: float = 20e3
    delay_samples: int = 1
    flex_mode_hz: float | None = None
    flex_mass_ratio: float = 0.3
    flex_damping: float = 0.02
    sensor: PositionSensor = field(default_factory=PositionSensor)
    floor_force_n: float = 0.02
    cable_stiffness_n_m: float = 10.0
    cable_damping_n_s_m: float = 2.0
    ripple_force_n: float = 0.05
    ripple_pitch_mm: float = 30.0
    force_noise_n: float = 0.05

    def __post_init__(self):
        wc = 2 * np.pi * self.bandwidth_hz
        m = self.mass_kg
        if self.kp is None:
            self.kp = m * wc**2 / 3.0
        if self.kd is None:
            self.kd = m * wc
        if self.ki is None:
            self.ki = self.kp * wc / 10.0
        self.tau_d = 1.0 / (5 * wc)

    # ---- plant --------------------------------------------------------------
    def _plant(self):
        """Continuous (A, B) with inputs [actuator force, disturbance force] and ZOH discretisation."""
        m = self.mass_kg
        if self.flex_mode_hz is None:
            A = np.array([[0.0, 1.0], [0.0, 0.0]])
            B = np.array([[0.0, 0.0], [1 / m, 1 / m]])
        else:
            m2 = m * self.flex_mass_ratio
            m1 = m - m2
            mr = m1 * m2 / m  # reduced mass of the internal mode
            w = 2 * np.pi * self.flex_mode_hz
            k = mr * w**2
            c = 2 * self.flex_damping * mr * w
            # state: [x1, v1, x2, v2]; actuator & sensor on body 1, disturbances split by mass
            A = np.array(
                [
                    [0, 1, 0, 0],
                    [-k / m1, -c / m1, k / m1, c / m1],
                    [0, 0, 0, 1],
                    [k / m2, c / m2, -k / m2, -c / m2],
                ],
                dtype=float,
            )
            B = np.array([[0, 0], [1 / m1, (m1 / m) / m1], [0, 0], [0, (m2 / m) / m2]], dtype=float)
        n = A.shape[0]
        dt = 1.0 / self.fs_hz
        M = np.zeros((n + 2, n + 2))
        M[:n, :n] = A
        M[:n, n:] = B
        E = expm(M * dt)
        return E[:n, :n], E[:n, n:]

    # ---- simulation -----------------------------------------------------------
    def simulate(self, profile: ScanProfile, rng: np.random.Generator | None = None, disturbances: bool = True) -> dict:
        """Simulate tracking of one field scan.

        Returns dict with ``t`` (s), ``setpoint_m``, ``position_m`` (true,
        sensor body), ``measured_m``, ``error_nm`` (setpoint - true position),
        ``force_n``, ``exposure`` mask, MA/MSD arrays ``ma_nm``/``msd_nm``, and
        their exposure maxima ``ma_max_nm``/``msd_max_nm``; with a flexible mode
        also ``error_payload_nm``.
        """
        rng = np.random.default_rng() if rng is None else rng
        dt = 1.0 / self.fs_hz
        sp = profile.sample(self.fs_hz)
        t = sp["t"]
        n = t.size
        r = sp["pos_m"]
        d = self.delay_samples
        # feedforward with preview: force computed at k acts over [k+d, k+d+1) -> centre k+d+0.5
        a_ff = profile.setpoint(t + (d + 0.5) * dt)["acc_m_s2"] if self.feedforward else np.zeros(n)
        ff_mass = self.mass_kg * (1 + self.ff_mass_error)

        Ad, Bd = self._plant()
        nx = Ad.shape[0]
        x = np.zeros(nx)

        if disturbances:
            ns = self.sensor.noise(n, rng, dt) * 1e-9
            fnoise = self.force_noise_n * rng.standard_normal(n)
            freqs = rng.uniform(5, 100, 8)
            phases = rng.uniform(0, 2 * np.pi, 8)
            amps = self.floor_force_n * np.sqrt(2.0 / 8)
            ffloor = (amps * np.sin(2 * np.pi * freqs[None, :] * t[:, None] + phases[None, :])).sum(1)
            ripple_phase = rng.uniform(0, 2 * np.pi)
        else:
            ns = np.zeros(n)
            fnoise = np.zeros(n)
            ffloor = np.zeros(n)
            ripple_phase = 0.0

        pos = np.zeros(n)
        pos2 = np.zeros(n)
        meas = np.zeros(n)
        force = np.zeros(n)
        u_queue = [0.0] * d
        integ = 0.0
        dterm = 0.0
        e_prev = 0.0
        kp, ki, kd, tau = self.kp, self.ki, self.kd, self.tau_d
        pitch = self.ripple_pitch_mm * 1e-3
        for k in range(n):
            y = x[0] + ns[k]
            e = r[k] - y
            integ += ki * e * dt
            dterm = (tau * dterm + kd * (e - e_prev)) / (tau + dt)
            e_prev = e
            u = kp * e + integ + dterm + ff_mass * a_ff[k]
            u_queue.append(u)
            u_applied = u_queue.pop(0)
            if disturbances:
                fd = (
                    ffloor[k]
                    + fnoise[k]
                    - self.cable_stiffness_n_m * x[0]
                    - self.cable_damping_n_s_m * x[1]
                    + self.ripple_force_n * np.sin(2 * np.pi * x[0] / pitch + ripple_phase)
                )
            else:
                fd = 0.0
            pos[k] = x[0]
            pos2[k] = x[2] if nx == 4 else x[0]
            meas[k] = y
            force[k] = u_applied
            x = Ad @ x + Bd @ np.array([u_applied, fd])

        err_nm = (r - pos) * 1e9
        ma, msd = ma_msd(err_nm, t, profile.scan_speed_mm_s, profile.slit_height_mm)
        stats = ma_msd_stats(err_nm, t, profile)
        out = dict(
            t=t,
            setpoint_m=r,
            position_m=pos,
            measured_m=meas,
            error_nm=err_nm,
            force_n=force,
            exposure=sp["exposure"],
            constant_velocity=sp["constant_velocity"],
            ma_nm=ma,
            msd_nm=msd,
            ma_max_nm=stats["ma_max_nm"],
            msd_max_nm=stats["msd_max_nm"],
        )
        if nx == 4:
            out["error_payload_nm"] = (r - pos2) * 1e9
        return out


# =============================================================================
# MA / MSD and synchronisation
# =============================================================================
def ma_msd(error_nm, t, scan_speed_mm_s: float, slit_height_mm: float = SLIT_HEIGHT_MM):
    """Moving average and moving standard deviation of a servo error.

    The window is the time a wafer point spends under the slit,
    ``T = slit_height / scan_speed``, centred on each sample.  Samples whose
    window would extend beyond the record are NaN.

    Returns ``(ma, msd)`` arrays (same units as ``error_nm``, same length).
    MA is the mean image displacement of the point exposed at that instant
    (-> overlay); MSD the rms image smear (-> contrast loss, fading).
    """
    e = np.asarray(error_nm, dtype=float)
    t = np.asarray(t, dtype=float)
    n = e.size
    dt = (t[-1] - t[0]) / (n - 1) if n > 1 else 1.0
    T = slit_height_mm / scan_speed_mm_s
    w = max(int(round(T / dt)), 1)
    ma = np.full(n, np.nan)
    msd = np.full(n, np.nan)
    if w > n:
        return ma, msd
    ref = np.mean(e)  # subtract mean to reduce cancellation in the variance
    x = e - ref
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = (c1[w:] - c1[:-w]) / w
    s2 = (c2[w:] - c2[:-w]) / w
    var = np.maximum(s2 - s1**2, 0.0)
    start = (w - 1) // 2  # window [k-start, k-start+w)
    ma[start : start + s1.size] = s1 + ref
    msd[start : start + s1.size] = np.sqrt(var)
    return ma, msd


def ma_msd_stats(error_nm, t, profile: ScanProfile) -> dict:
    """max |MA| and max MSD over window centres that lie fully inside exposure.

    A window centred at time ``tc`` corresponds to a wafer point exposed from
    ``tc - T/2`` to ``tc + T/2``; valid centres are within
    ``[t_exp0 + T/2, t_exp1 - T/2]``.
    """
    ma, msd = ma_msd(error_nm, t, profile.scan_speed_mm_s, profile.slit_height_mm)
    t = np.asarray(t)
    t0, t1 = profile.exposure_window()
    half = 0.5 * profile.slit_time_s()
    m = (t >= t0 + half) & (t <= t1 - half) & np.isfinite(ma)
    if not np.any(m):
        return dict(ma_max_nm=np.nan, msd_max_nm=np.nan, ma=ma, msd=msd, mask=m)
    return dict(
        ma_max_nm=float(np.max(np.abs(ma[m]))),
        msd_max_nm=float(np.max(msd[m])),
        ma=ma,
        msd=msd,
        mask=m,
    )


def synchronization_error(wafer_err_nm, reticle_err_nm, reduction: float = REDUCTION):
    """Wafer-scale synchronisation error ``wafer - reticle / reduction`` (nm).

    Both errors are ``setpoint - actual`` in their own stage frame; the reticle
    frame is defined with its axis flipped so that a synchronised reticle has
    ``x_reticle = reduction * x_wafer`` (see module docstring).  A reticle error
    of ``reduction * e`` therefore cancels a wafer error ``e`` exactly.
    """
    return np.asarray(wafer_err_nm) - np.asarray(reticle_err_nm) / reduction


def simulate_synchronization(
    profile: ScanProfile,
    wafer_loop: ServoLoop | None = None,
    reticle_loop: ServoLoop | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate wafer and reticle stage for one field and return sync error MA/MSD.

    Default reticle short stroke: 10 kg, 400 Hz bandwidth.
    """
    rng = np.random.default_rng() if rng is None else rng
    wafer_loop = wafer_loop or ServoLoop()
    reticle_loop = reticle_loop or ServoLoop(mass_kg=10.0)
    rp = profile.reticle_profile()
    w = wafer_loop.simulate(profile, rng)
    r = reticle_loop.simulate(rp, rng)
    n = min(w["t"].size, r["t"].size)
    t = w["t"][:n]
    se = synchronization_error(w["error_nm"][:n], r["error_nm"][:n])
    st = ma_msd_stats(se, t, profile)
    return dict(
        t=t,
        sync_error_nm=se,
        wafer=w,
        reticle=r,
        ma_nm=st["ma"],
        msd_nm=st["msd"],
        ma_max_nm=st["ma_max_nm"],
        msd_max_nm=st["msd_max_nm"],
    )


def msd_contrast_loss(msd_nm, pitch_nm):
    """Image contrast reduction factor from MSD: ``exp(-2 pi^2 msd^2 / pitch^2)``.

    This is the MTF of a Gaussian blur of rms ``msd`` at spatial frequency
    ``1/pitch``; 1 means no loss.
    """
    msd = np.asarray(msd_nm, dtype=float)
    return np.exp(-2 * np.pi**2 * msd**2 / np.asarray(pitch_nm, dtype=float) ** 2)


# =============================================================================
# Field layout
# =============================================================================
def field_layout(
    wafer_diameter_mm: float = WAFER_DIAMETER_MM,
    field_w: float = FIELD_WIDTH_MM,
    field_h: float = FIELD_HEIGHT_MM,
    edge_exclusion_mm: float = 3.0,
    offset_mm: tuple[float, float] = (0.0, 0.0),
    include_partial: bool = True,
) -> list[tuple[float, float]]:
    """Field centres (mm) of all fields fully or partially on the usable wafer area.

    The field grid has one field centred at ``offset_mm`` from the wafer
    centre.  A field is kept if it overlaps the disc of radius
    ``D/2 - edge_exclusion`` (or, with ``include_partial=False``, if it lies
    fully inside).  Fields are returned in meander (serpentine) order: rows of
    constant y from top to bottom, x direction alternating per row.  The scan
    (y) direction alternates field by field, see :func:`scan_directions`.
    """
    R = wafer_diameter_mm / 2 - edge_exclusion_mm
    ox, oy = offset_mm
    nx = int(np.ceil(R / field_w)) + 2
    ny = int(np.ceil(R / field_h)) + 2
    rows = []
    for iy in range(ny, -ny - 1, -1):
        cy = oy + iy * field_h
        row = []
        for ix in range(-nx, nx + 1):
            cx = ox + ix * field_w
            x0, x1 = cx - field_w / 2, cx + field_w / 2
            y0, y1 = cy - field_h / 2, cy + field_h / 2
            if include_partial:
                qx = min(max(0.0, x0), x1)
                qy = min(max(0.0, y0), y1)
                keep = qx * qx + qy * qy < R * R
            else:
                keep = max(x0 * x0, x1 * x1) + max(y0 * y0, y1 * y1) <= R * R
            if keep:
                row.append((float(cx), float(cy)))
        if row:
            rows.append(row)
    out = []
    for i, row in enumerate(rows):
        out.extend(row if i % 2 == 0 else row[::-1])
    return out


def scan_directions(n_fields: int, first: int = 1) -> np.ndarray:
    """Alternating scan directions (+1/-1) for ``n_fields`` consecutive fields."""
    return first * (1 - 2 * (np.arange(n_fields) % 2))


# =============================================================================
# Dual stage and throughput
# =============================================================================
@dataclass
class DualStage:
    """TWINSCAN-like dual-chuck timing model.

    Expose side, per wafer: for each field a scan move
    (:meth:`ScanProfile.time_per_field`) plus the *non-overlapped* part of the
    step move to the next field.  The step move (jerk-limited point-to-point,
    ``step_speed_m_s``/``step_accel_m_s2``) is performed during the y
    deceleration, turnaround and acceleration of the alternating scans, so only
    ``max(0, t_step - step_overlap * (2 t_ramp + t_settle))`` is added.  Plus a
    fixed per-wafer overhead (first approach, immersion hood edge moves, ...).

    Measure side, per wafer (runs in parallel): wafer unload/load,
    ``n_align_marks`` alignment mark captures, and leveling (height-map) scan
    of the full wafer in strips of ``level_strip_mm`` at ``level_speed_mm_s``.

    Cycle per wafer = max(expose, measure) + swap_time.  With
    ``dual=False`` (single-stage scanner) everything is serial.
    """

    profile: ScanProfile = field(default_factory=ScanProfile)
    step_speed_m_s: float = 1.0
    step_accel_m_s2: float = 30.0
    step_jerk_m_s3: float = 5000.0
    step_overlap: float = 1.0
    wafer_overhead_s: float = 0.5
    swap_time_s: float = 1.5
    load_unload_s: float = 2.5
    n_align_marks: int = 24
    align_time_per_mark_s: float = 0.08
    level_speed_mm_s: float = 1000.0
    level_strip_mm: float = FIELD_WIDTH_MM
    wafer_diameter_mm: float = WAFER_DIAMETER_MM
    dual: bool = True

    def step_time(self, dx_mm: float, dy_mm: float) -> float:
        """Non-overlapped step overhead (s) between two fields."""
        p = self.profile
        d = np.hypot(dx_mm, dy_mm) * 1e-3
        t = point_to_point_time(d, self.step_speed_m_s, self.step_accel_m_s2, self.step_jerk_m_s3)
        return max(0.0, t - self.step_overlap * (2 * p.ramp_time() + p.settle_time_s))

    def expose_breakdown(self, fields: list[tuple[float, float]] | int) -> dict:
        """Expose-side time split: scan, step overhead, per-wafer overhead (s)."""
        if isinstance(fields, (int, np.integer)):
            n = int(fields)
            steps = (n - 1) * self.step_time(FIELD_WIDTH_MM, 0.0) if n > 1 else 0.0
        else:
            n = len(fields)
            steps = sum(
                self.step_time(fields[i + 1][0] - fields[i][0], fields[i + 1][1] - fields[i][1]) for i in range(n - 1)
            )
        scan = n * self.profile.time_per_field()
        return dict(
            n_fields=n,
            scan_time_s=scan,
            exposure_time_s=n * self.profile.exposure_time_s(),
            step_overhead_s=steps,
            wafer_overhead_s=self.wafer_overhead_s,
            expose_side_s=scan + steps + self.wafer_overhead_s,
        )

    def measure_time(self) -> float:
        """Measure-side time per wafer (s): load/unload + alignment + leveling."""
        D = self.wafer_diameter_mm
        n_strips = int(np.ceil(D / self.level_strip_mm))
        # strip lengths across the disc plus a turnaround per strip (~0.1 s)
        ys = (np.arange(n_strips) + 0.5) * self.level_strip_mm - D / 2
        lengths = 2 * np.sqrt(np.maximum((D / 2) ** 2 - ys**2, 0.0))
        level = lengths.sum() / self.level_speed_mm_s + 0.1 * n_strips
        align = self.n_align_marks * self.align_time_per_mark_s
        return self.load_unload_s + align + level

    def cycle_time(self, n_fields: int | list[tuple[float, float]]) -> float:
        """Time between consecutive wafers (s)."""
        expose = self.expose_breakdown(n_fields)["expose_side_s"]
        meas = self.measure_time()
        if self.dual:
            return max(expose, meas) + self.swap_time_s
        return expose + meas


def throughput(
    scan_speed_mm_s: float = 700.0,
    n_fields: int | None = None,
    max_accel_m_s2: float = 30.0,
    max_jerk_m_s3: float = 5000.0,
    settle_time_s: float = 0.005,
    dual: bool = True,
    **dual_stage_kwargs,
) -> dict:
    """Wafers per hour of a (dual-stage) scanner.

    ``n_fields=None`` uses the meander :func:`field_layout` of a 300 mm wafer
    (26 x 33 mm fields, full + partial, ~ 100 fields).  An integer instead
    assumes that many fields with 26 mm x-steps only.

    With the defaults (700 mm/s, 30 m/s^2) the expose side takes ~13 s and the
    result is ~ 250 wph, the order of a modern immersion scanner; real tools
    reach ~300 wph with higher accelerations (> 50 m/s^2) and faster scans.
    Dose limits on the maximum scan speed (laser power) are not modelled here.

    Returns dict(wph, cycle_time_s, exposure_time_s, scan_time_s,
    step_overhead_s, wafer_overhead_s, expose_side_s, measure_side_s,
    swap_time_s, overhead_s, n_fields, limiting).
    """
    prof = ScanProfile(
        scan_speed_mm_s=scan_speed_mm_s,
        max_accel_m_s2=max_accel_m_s2,
        max_jerk_m_s3=max_jerk_m_s3,
        settle_time_s=settle_time_s,
    )
    ds = DualStage(profile=prof, dual=dual, **dual_stage_kwargs)
    fields = field_layout() if n_fields is None else int(n_fields)
    br = ds.expose_breakdown(fields)
    meas = ds.measure_time()
    cycle = ds.cycle_time(fields)
    return dict(
        wph=3600.0 / cycle,
        cycle_time_s=cycle,
        exposure_time_s=br["exposure_time_s"],
        scan_time_s=br["scan_time_s"],
        step_overhead_s=br["step_overhead_s"],
        wafer_overhead_s=br["wafer_overhead_s"],
        expose_side_s=br["expose_side_s"],
        measure_side_s=meas,
        swap_time_s=ds.swap_time_s if dual else 0.0,
        overhead_s=cycle - br["exposure_time_s"],
        n_fields=br["n_fields"],
        limiting=("expose" if br["expose_side_s"] >= meas else "measure") if dual else "serial",
    )
