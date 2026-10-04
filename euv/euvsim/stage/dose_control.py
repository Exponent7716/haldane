"""Slit-integrated dose and pulse-energy dose control of a scanning exposure.

Pulses per point
----------------
The LPP source fires at repetition rate ``f`` (~50-100 kHz).  A wafer point
crosses the slit of height ``h`` in ``h/v``, so it collects

    N = h f / v        pulses (e.g. 2 mm * 50 kHz / 0.3 m/s ~ 333).

Dose per point:  ``D = N * E_p / (w h) = E_p f / (w v) = P/(w v)`` with
``E_p`` the pulse energy at wafer level spread over the slit area ``w h``.

Dose error
----------
With independent pulse-energy fluctuations of relative rms ``sigma_p`` the
open-loop relative dose error is

    sigma_D / D = sigma_p / sqrt(N)

A non-integer N with a sharp (top-hat) slit edge adds a *pulse quantisation*
ripple of order 1/N (points receive floor(N) or ceil(N) pulses); a soft
(trapezoidal) slit edge suppresses it.  Slow source drifts are not averaged
by the window at all.

Closed loop (pulse-energy feedback)
-----------------------------------
An energy sensor measures each pulse (relative noise ``sigma_s``); the
controller commands the next pulse such that the running sum of the last
``N`` pulses approaches its target ``N E_t`` (a moving-window dose servo,
used in excimer DUV and EUV sources alike):

    S_k  = sum_{i=k-N+1}^{k-1} E_meas_i          (last N-1 pulses)
    u_k  = 1 + g * (N E_t - S_k - E_t) / E_t   (+ integral term on drift)
    E_k  = u_k * E_raw_k                       (clipped to actuator range)

``E_raw`` is the pulse energy the source would deliver at nominal command
(passed in by the caller; synthetic in tests).  Gain ``g < 1`` keeps the
loop well damped; the integral action removes slow drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pulses_per_point(slit_height_m: float, rep_rate_hz: float, scan_speed: float) -> float:
    """N = h f / v."""
    return slit_height_m * rep_rate_hz / scan_speed


def dose_mj_cm2(pulse_energy_at_wafer_j: float, n_pulses: float,
                slit_width_m: float = 26e-3, slit_height_m: float = 2e-3) -> float:
    """D = N E_p / (w h), converted to mJ/cm^2 (1 J/m^2 = 0.1 mJ/cm^2)."""
    return n_pulses * pulse_energy_at_wafer_j / (slit_width_m * slit_height_m) * 0.1


def open_loop_dose_error(sigma_pulse_rel: float, n_pulses: float) -> float:
    """Relative 1-sigma dose error sigma_p / sqrt(N) for white pulse noise."""
    return sigma_pulse_rel / np.sqrt(n_pulses)


def slit_profile(u: np.ndarray, kind: str = "trapezoid", edge_fraction: float = 0.2) -> np.ndarray:
    """Normalised slit intensity profile in the scan direction.

    ``u`` = position / slit height (FWHM = 1, centred).  ``flat``: top hat;
    ``trapezoid``: linear edges of total width ``edge_fraction`` centred on
    the half-maximum points.  Both integrate to 1.
    """
    u = np.abs(np.asarray(u, float))
    if kind == "flat":
        return (u < 0.5).astype(float) + 0.5 * (u == 0.5)
    if kind == "trapezoid":
        e = edge_fraction
        return np.clip((0.5 + e / 2 - u) / e, 0.0, 1.0)
    raise ValueError("kind must be 'flat' or 'trapezoid'")


def slit_integrated_dose(pulse_energies: np.ndarray, n_pulses_in_slit: float,
                         profile: str = "trapezoid", edge_fraction: float = 0.2,
                         oversample: int = 8) -> np.ndarray:
    """Dose received by wafer points as the slit sweeps over them.

    Pulse ``i`` fires when the slit centre is at ``i`` (in units of the wafer
    travel between pulses, v/f).  A wafer point at position ``p`` receives

        D(p) = (1/N) sum_i E_i * w((p - i)/N)

    with ``w`` the normalised slit profile, so constant pulses ``E0`` give
    ``D = E0`` (relative units: dose in "mean pulse energies").  Points are
    sampled ``oversample`` times per pulse spacing; only points that have
    seen the whole slit are returned.
    """
    E = np.asarray(pulse_energies, float)
    N = float(n_pulses_in_slit)
    half = 0.5 + edge_fraction / 2 if profile == "trapezoid" else 0.5
    reach = int(np.ceil(half * N)) + 1
    j = np.arange(-reach, reach + 1)
    out = []
    for k in range(oversample):
        phi = k / oversample
        w = slit_profile((j + phi) / N, profile, edge_fraction) / N
        # D(p = m + phi) = sum_i E_i w((m + phi - i)/N) ; convolution kernel over j=m-i
        d = np.convolve(E, w, mode="valid")
        out.append(d)
    n = min(len(d) for d in out)
    return np.stack([d[:n] for d in out], axis=1).ravel()


@dataclass
class DoseControlResult:
    pulse_energies: np.ndarray     # delivered (true) energies
    command: np.ndarray            # u_k
    dose: np.ndarray               # slit-integrated (relative to target pulse energy)

    @property
    def dose_error_3sigma(self) -> float:
        """Relative |mean - 1| + 3 sigma of the dose."""
        return float(abs(np.mean(self.dose) - 1.0) + 3 * np.std(self.dose))


def apply_dose_control(raw_pulse_energies: np.ndarray, n_pulses_in_slit: float,
                       rng: np.random.Generator, target: float | None = None,
                       gain: float = 0.3, integral_gain: float = 0.01,
                       sensor_noise_rel: float = 0.002, actuator_range: float = 0.3,
                       feedback: bool = True, profile: str = "trapezoid",
                       edge_fraction: float = 0.2) -> DoseControlResult:
    """Run the moving-window pulse-energy dose servo over a pulse train.

    Parameters
    ----------
    raw_pulse_energies : energies at nominal command (any units).
    n_pulses_in_slit : N = h f / v.
    target : target pulse energy E_t (default: 1.0, i.e. raw energies are
        relative to nominal).
    gain, integral_gain : window-sum and drift (integral) feedback gains.
    sensor_noise_rel : relative rms noise of the pulse-energy sensor.
    actuator_range : max relative change of the command (+/-).
    feedback : False -> open loop (u = 1).
    """
    raw = np.asarray(raw_pulse_energies, float)
    Et = 1.0 if target is None else float(target)
    n = raw.size
    Nw = max(1, int(round(n_pulses_in_slit)))
    E = np.empty(n)
    u = np.ones(n)
    meas = np.empty(n)
    sensor = sensor_noise_rel * rng.standard_normal(n)
    integ = 0.0
    window_sum = 0.0  # sum of last Nw-1 measured energies
    for k in range(n):
        if feedback:
            if k >= Nw - 1:
                err = (window_sum + Et - Nw * Et) / Et   # predicted window error (in pulses)
            else:
                err = 0.0
            uk = 1.0 - gain * err - integ
            uk = min(max(uk, 1 - actuator_range), 1 + actuator_range)
        else:
            uk = 1.0
        u[k] = uk
        E[k] = uk * raw[k]
        meas[k] = E[k] * (1 + sensor[k])
        if feedback:
            integ += integral_gain * (meas[k] - Et) / Et
        window_sum += meas[k]
        if k - (Nw - 1) >= 0:
            window_sum -= meas[k - (Nw - 1)]
    dose = slit_integrated_dose(E / Et, n_pulses_in_slit, profile, edge_fraction)
    return DoseControlResult(E, u, dose)


def synthetic_pulse_train(n: int, rng: np.random.Generator, sigma_rel: float = 0.05,
                          drift_rel: float = 0.0, drift_period_pulses: float = 5000.0
                          ) -> np.ndarray:
    """Illustrative raw pulse energies: 1 + white noise + sinusoidal drift."""
    k = np.arange(n)
    return (1.0 + sigma_rel * rng.standard_normal(n)
            + drift_rel * np.sin(2 * np.pi * k / drift_period_pulses))
