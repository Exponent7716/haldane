"""Stage dynamics, servo control and reticle/wafer synchronisation (MA / MSD).

Plant
-----
Each (short-stroke, magnetically levitated, in-vacuum) stage axis is a
two-mass model: the actuated mover ``m1`` (magnet plate / coil block) and the
chuck ``m2`` carrying wafer/reticle and the position sensor targets, coupled
by stiffness ``k`` and damping ``c`` (first flexible mode):

    m1 x1'' = F - k (x1 - x2) - c (x1' - x2') + d1
    m2 x2'' =     k (x1 - x2) + c (x1' - x2') + d2

    rigid-body:   P(s) ~ 1 / ((m1+m2) s^2)
    flexible mode: f_e = (1/2pi) sqrt(k (m1+m2) / (m1 m2))

The sensor measures the chuck ``x2`` (non-collocated), plus sensor noise and
a slow thermal drift of the metrology frame.

Controller
----------
PID with lead and 2nd-order low-pass, tuned by the classical rules for a mass
line (Schmidt, Schitter, Rankers, *The Design of High Performance
Mechatronics*, 2014):

    C(s) = kp (1 + w_i/s) (1 + s/w_d)/(1 + s/w_t) * w_lp^2/(s^2 + 2 z w_lp s + w_lp^2)
    w_d = w_c/3, w_t = 3 w_c, w_i = w_c/10, w_lp = 6 w_c, kp = m w_c^2 / 3

Feedforward: ``F_ff = (m1+m2) a_ref + (m1 m2 / k) s_ref`` (acceleration plus
snap feedforward compensating the compliance of the flexible mode,
Lambrechts et al. 2005).  With FF the servo error is dominated by
disturbances instead of by the reference.

Synchronisation error and MA/MSD
--------------------------------
The image of the reticle moves with the reticle divided by the magnification.
The synchronisation (overlay-relevant) error at wafer level is

    e_s(t) = e_w(t) - e_r(t)/M_y          (e = setpoint - actual)

A point on the wafer is exposed during ``T_slit = h_slit/v_scan``; the
image it receives is blurred/shifted by the error within that window:

    MA(t)  = (1/T_slit) * int_{t-T/2}^{t+T/2} e_s dt        -> image shift (overlay)
    MSD(t) = sqrt((1/T) int (e_s - MA)^2 dt)                 -> image blur (contrast/CD)

Public specs for EUV scanners are MA < ~1 nm and MSD of a few nm.

Disturbances (``rng: numpy.random.Generator``):
* floor / frame vibration: band-limited random force (Ornstein-Uhlenbeck),
* cable slab: position-dependent spring force + broadband noise,
* thermal drift: random-walk sensor offset (very low frequency),
* sensor noise: white (encoder ~ 0.05-0.2 nm rms at servo rate).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, signal

from .trajectory import ScanProfile


@dataclass
class StagePlant:
    """Two-mass stage axis model (kg, N/m, N s/m)."""

    m1: float = 20.0      # mover / short-stroke actuator mass
    m2: float = 10.0      # chuck (sensor side)
    f_flex_hz: float = 1500.0
    zeta_flex: float = 0.02

    @property
    def mass(self) -> float:
        return self.m1 + self.m2

    @property
    def mu(self) -> float:
        """Reduced mass m1 m2/(m1+m2)."""
        return self.m1 * self.m2 / self.mass

    @property
    def k(self) -> float:
        return self.mu * (2 * np.pi * self.f_flex_hz) ** 2

    @property
    def c(self) -> float:
        return 2 * self.zeta_flex * np.sqrt(self.k * self.mu)

    def state_space(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continuous (A, B, C); state [x1, v1, x2, v2], inputs [F, d1, d2], out x2."""
        m1, m2, k, c = self.m1, self.m2, self.k, self.c
        A = np.array([[0, 1, 0, 0],
                      [-k / m1, -c / m1, k / m1, c / m1],
                      [0, 0, 0, 1],
                      [k / m2, c / m2, -k / m2, -c / m2]], float)
        B = np.array([[0, 0, 0],
                      [1 / m1, 1 / m1, 0],
                      [0, 0, 0],
                      [0, 0, 1 / m2]], float)
        C = np.array([[0, 0, 1, 0]], float)
        return A, B, C

    def discretize(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact zero-order-hold discretisation via the matrix exponential."""
        A, B, C = self.state_space()
        n, m = A.shape[0], B.shape[1]
        M = np.zeros((n + m, n + m))
        M[:n, :n] = A
        M[:n, n:] = B
        E = linalg.expm(M * dt)
        return E[:n, :n], E[:n, n:], C


@dataclass
class Controller:
    """PID + lead + low-pass position controller for a mass-line plant."""

    bandwidth_hz: float = 250.0
    mass: float = 30.0
    lp_zeta: float = 0.5

    def transfer_function(self) -> tuple[np.ndarray, np.ndarray]:
        """Continuous C(s) numerator/denominator (descending powers of s)."""
        wc = 2 * np.pi * self.bandwidth_hz
        kp = self.mass * wc ** 2 / 3.0
        wi, wd, wt, wlp = wc / 10, wc / 3, 3 * wc, 6 * wc
        num = kp * np.polymul([1, wi], [1 / wd, 1])
        num = np.polymul(num, [wlp ** 2])
        den = np.polymul([1, 0], [1 / wt, 1])
        den = np.polymul(den, [1, 2 * self.lp_zeta * wlp, wlp ** 2])
        return num, den

    def discretize(self, dt: float):
        """Tustin-discretised controller in state-space form (Ad, Bd, Cd, Dd)."""
        num, den = self.transfer_function()
        A, B, C, D = signal.tf2ss(num, den)
        Ad, Bd, Cd, Dd, _ = signal.cont2discrete((A, B, C, D), dt, method="bilinear")
        return Ad, Bd, Cd, Dd


@dataclass
class Disturbances:
    """Disturbance magnitudes (SI).  Defaults are illustrative, public-order values."""

    floor_force_rms: float = 0.012     # N, band-limited (vibration via frame / planar motor)
    floor_bandwidth_hz: float = 100.0
    cable_stiffness: float = 5.0       # N/m, cable slab residual after calibrated FF
    cable_noise_rms: float = 0.05      # N
    thermal_drift_rate: float = 0.5e-9 # m/sqrt(s) random walk of sensor offset
    sensor_noise_rms: float = 0.1e-9   # m white

    @staticmethod
    def none() -> "Disturbances":
        return Disturbances(0.0, 100.0, 0.0, 0.0, 0.0, 0.0)


def _ou_process(n: int, dt: float, rms: float, f_hz: float,
                rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck (1st-order low-pass) noise with stationary rms ``rms``."""
    if rms == 0:
        return np.zeros(n)
    a = np.exp(-2 * np.pi * f_hz * dt)
    w = rng.standard_normal(n) * rms * np.sqrt(1 - a * a)
    return signal.lfilter([1.0], [1.0, -a], w, zi=[rng.standard_normal() * rms * a])[0]


@dataclass
class ServoResult:
    t: np.ndarray
    ref: np.ndarray
    pos: np.ndarray
    error: np.ndarray         # setpoint - true chuck position [m]
    force: np.ndarray


def simulate_servo(ref_pos: np.ndarray, ref_acc: np.ndarray, ref_snap: np.ndarray,
                   dt: float, plant: StagePlant, controller: Controller,
                   disturbances: Disturbances, rng: np.random.Generator,
                   feedforward: bool = True) -> ServoResult:
    """Closed-loop time simulation of one axis tracking ``ref_pos``.

    Loop per sample k (discrete, ZOH plant, Tustin controller):
        y_meas = x2 + n_sensor + drift
        e_meas = r - y_meas ;  u = C(e_meas) + F_ff ;  x <- Ad x + Bd [u, d1, d2]
    The returned ``error`` is setpoint minus *true* chuck position, i.e. what
    the image sees (sensor drift therefore appears as a real error).
    """
    n = ref_pos.size
    Ap, Bp, Cp = plant.discretize(dt)
    Ac, Bc, Cc, Dc = controller.discretize(dt)
    Bc, Cc, Dc = Bc[:, 0], Cc[0], float(Dc[0, 0])
    d = disturbances
    floor = _ou_process(n, dt, d.floor_force_rms, d.floor_bandwidth_hz, rng)
    cable_noise = d.cable_noise_rms * rng.standard_normal(n)
    sensor = d.sensor_noise_rms * rng.standard_normal(n)
    drift = np.cumsum(rng.standard_normal(n)) * d.thermal_drift_rate * np.sqrt(dt)
    ff = (plant.mass * ref_acc + plant.m1 * plant.m2 / plant.k * ref_snap
          if feedforward else np.zeros(n))
    x = np.zeros(4)
    x[0] = x[2] = ref_pos[0]
    xc = np.zeros(Ac.shape[0])
    pos = np.empty(n)
    force = np.empty(n)
    for k in range(n):
        y = x[2]
        pos[k] = y
        e = ref_pos[k] - (y + sensor[k] + drift[k])
        u = Cc @ xc + Dc * e + ff[k]
        xc = Ac @ xc + Bc * e
        d1 = floor[k] - d.cable_stiffness * (x[0] - ref_pos[0]) + cable_noise[k]
        x = Ap @ x + Bp @ np.array([u, d1, 0.0])
        force[k] = u
    t = np.arange(n) * dt
    return ServoResult(t, ref_pos, pos, ref_pos - pos, force)


def closed_loop_stable(plant: StagePlant, controller: Controller, dt: float) -> bool:
    """True if all closed-loop discrete poles lie strictly inside the unit circle."""
    Ap, Bp, Cp = plant.discretize(dt)
    Bp = Bp[:, :1]
    Ac, Bc, Cc, Dc = controller.discretize(dt)
    # u = Cc xc + Dc (-Cp x) ; xc+ = Ac xc + Bc (-Cp x)
    top = np.hstack([Ap - Bp @ Dc @ Cp, Bp @ Cc])
    bot = np.hstack([-Bc @ Cp, Ac])
    return bool(np.all(np.abs(np.linalg.eigvals(np.vstack([top, bot]))) < 1.0))


def moving_average_std(e: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered moving average and moving standard deviation over ``window`` samples.

    Returns arrays of length ``e.size - window + 1`` (valid part).
    """
    if window < 1 or window > e.size:
        raise ValueError("invalid window")
    c1 = np.concatenate([[0.0], np.cumsum(e)])
    c2 = np.concatenate([[0.0], np.cumsum(e * e)])
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    ma = s1 / window
    var = np.maximum(s2 / window - ma * ma, 0.0)
    return ma, np.sqrt(var)


@dataclass
class SyncResult:
    """Wafer–reticle synchronisation performance over one field exposure."""

    t: np.ndarray
    sync_error: np.ndarray          # at wafer level [m]
    ma: np.ndarray                  # over exposure window [m]
    msd: np.ndarray                 # [m]
    wafer: ServoResult
    reticle: ServoResult
    window_samples: int

    @property
    def ma_max_nm(self) -> float:
        return float(np.max(np.abs(self.ma)) * 1e9)

    @property
    def msd_max_nm(self) -> float:
        return float(np.max(self.msd) * 1e9)

    @property
    def ma_3sigma_nm(self) -> float:
        """|mean| + 3 sigma of MA over the field (overlay-like statistic)."""
        return float((abs(np.mean(self.ma)) + 3 * np.std(self.ma)) * 1e9)


@dataclass
class ScannerStages:
    """Wafer and reticle stage (plant + controller) pair."""

    wafer_plant: StagePlant = field(default_factory=lambda: StagePlant(20.0, 10.0, 1500.0, 0.02))
    reticle_plant: StagePlant = field(default_factory=lambda: StagePlant(15.0, 8.0, 1800.0, 0.02))
    wafer_bw_hz: float = 250.0
    reticle_bw_hz: float = 300.0

    def controllers(self) -> tuple[Controller, Controller]:
        return (Controller(self.wafer_bw_hz, self.wafer_plant.mass),
                Controller(self.reticle_bw_hz, self.reticle_plant.mass))


def simulate_scan_sync(profile: ScanProfile, rng: np.random.Generator,
                       stages: ScannerStages | None = None,
                       disturbances: Disturbances | None = None,
                       reticle_disturbances: Disturbances | None = None,
                       slit_height_m: float = 2e-3,
                       feedforward: bool = True) -> SyncResult:
    """Simulate both stages over a scan and compute MA/MSD of the sync error.

    The MA/MSD window is ``T_slit = h_slit / v_scan`` (time a wafer point
    spends under the slit), evaluated only for window centres inside the
    exposure interval of the profile.
    """
    stages = stages or ScannerStages()
    disturbances = disturbances or Disturbances()
    reticle_disturbances = reticle_disturbances or disturbances
    cw, cr = stages.controllers()
    w, r = profile.wafer, profile.reticle
    dt = w.dt
    res_w = simulate_servo(w.pos, w.acc, w.snap, dt, stages.wafer_plant, cw,
                           disturbances, rng, feedforward)
    res_r = simulate_servo(r.pos, r.acc, r.snap, dt, stages.reticle_plant, cr,
                           reticle_disturbances, rng, feedforward)
    M = profile.magnification
    e_s = res_w.error - res_r.error / (-M)  # reticle setpoint = -M * wafer
    window = max(1, int(round(slit_height_m / profile.scan_speed / dt)))
    ma, msd = moving_average_std(e_s, window)
    t_centres = w.t[: ma.size] + (window - 1) / 2 * dt
    half = window * dt / 2
    sel = ((t_centres >= profile.t_exposure_start + half)
           & (t_centres <= profile.t_exposure_end - half))
    return SyncResult(t_centres[sel], e_s, ma[sel], msd[sel], res_w, res_r, window)
