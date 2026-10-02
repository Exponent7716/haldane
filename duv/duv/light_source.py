"""ArF excimer laser light source for the DUV scanner model.

This module models, at a textbook / public-literature level, the line-narrowed
ArF excimer laser (Cymer XLR / Gigaphoton GT-class) that illuminates an
immersion scanner:

* :class:`DischargeModel` - a 0-D kinetics / gain model of the Ar/F2/Ne
  discharge: deposited energy -> ArF* formation rate -> small-signal gain ->
  extracted pulse energy, with F2 acting both as fuel (ArF* formation) and as a
  quencher / absorber.  A time-domain rate-equation integration
  (:meth:`DischargeModel.simulate_pulse`) gives the gain-switched pulse shape.
* :class:`GasManager` - F2 depletion over many pulses, the voltage servo that
  hides it, and periodic F2 injections that restore the gas.
* :class:`LineNarrowingModule` - prism beam expander plus Littrow echelle
  grating; bandwidth from angular dispersion and intracavity divergence, and
  wavelength tuning via the grating incidence angle.
* :class:`ArFLaser` - the user-facing laser: spectrum with a prescribed E95,
  pulse energies with pulse-to-pulse jitter, wavelength stabilisation loop.
* :class:`DoseController` - per-pulse energy control so that every point on
  the wafer, which sees a moving window of N pulses while it crosses the slit,
  receives the target dose.
* Helpers :func:`pulses_per_point`, :func:`dose_from_pulses`,
  :func:`window_sums`, :func:`e95_width_pm`, :func:`speckle_contrast`.

Units follow :mod:`duv.core`: wavelengths in nm (bandwidths in pm, small
wavelength errors in fm, explicitly suffixed), time in s (``_ns`` suffix where
noted), pulse energy in mJ and dose in mJ/cm^2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erf

from .core import (
    FIELD_WIDTH_MM,
    SLIT_HEIGHT_MM,
    WAVELENGTH_ARF,
    photon_energy,
)

__all__ = [
    "DischargeModel",
    "GasManager",
    "LineNarrowingModule",
    "ArFLaser",
    "DoseController",
    "DoseControlResult",
    "pulses_per_point",
    "dose_from_pulses",
    "window_sums",
    "e95_width_pm",
    "coherence_length_um",
    "speckle_contrast",
]

_LOSCHMIDT_CM3 = 2.687e19  # molecules / cm^3 at 1 atm, 0 degC (gas ~ room T, close enough)
_C_CM_PER_NS = 29.9792458


# =============================================================================
# Simple helpers
# =============================================================================
def pulses_per_point(
    slit_height_mm: float = SLIT_HEIGHT_MM,
    scan_speed_mm_s: float = 700.0,
    rep_rate_hz: float = 6000.0,
) -> float:
    """Number of laser pulses a wafer point receives while crossing the slit.

    A point moving at ``scan_speed_mm_s`` crosses a slit of height
    ``slit_height_mm`` (scan direction, wafer scale) in ``h / v`` seconds, during
    which the laser fires ``rep_rate * h / v`` pulses.  E.g. 8 mm, 700 mm/s,
    6 kHz -> 68.57 pulses.
    """
    if scan_speed_mm_s <= 0:
        raise ValueError("scan speed must be positive")
    return slit_height_mm * rep_rate_hz / scan_speed_mm_s


def dose_from_pulses(
    pulse_energies_mj,
    transmission: float = 0.08,
    slit_width_mm: float = FIELD_WIDTH_MM,
    slit_height_mm: float = SLIT_HEIGHT_MM,
) -> float:
    """Wafer dose (mJ/cm^2) delivered by a set of laser pulses to one point.

    Assumptions:

    * every pulse in ``pulse_energies_mj`` (energy at the laser exit, mJ) is
      seen by the point, i.e. the array is the window of pulses fired while
      the point was inside the slit;
    * the exposure slit is a uniform top-hat of ``slit_width_mm`` x
      ``slit_height_mm`` at wafer scale (real slits have soft, trapezoidal
      edges, which smooths pulse-quantisation but not the integrated dose);
    * ``transmission`` lumps beam delivery, illuminator (homogeniser, variable
      attenuator, REMA), reticle clear area and projection-lens transmission.
      Typical laser-to-wafer values are 5-20 %; 8 % gives ~50 mJ/cm^2 for
      69 pulses of 15 mJ.
    """
    area_cm2 = (slit_width_mm * 0.1) * (slit_height_mm * 0.1)
    return float(transmission * np.sum(pulse_energies_mj) / area_cm2)


def window_sums(values, n_window: int) -> np.ndarray:
    """Sums over all complete moving windows of ``n_window`` consecutive pulses."""
    v = np.asarray(values, dtype=float)
    n = int(n_window)
    if n < 1 or n > v.size:
        raise ValueError("window must be between 1 and len(values)")
    c = np.concatenate([[0.0], np.cumsum(v)])
    return c[n:] - c[:-n]


def e95_width_pm(wavelengths_nm, weights) -> float:
    """E95 bandwidth (pm) of a sampled spectrum: width of the central 95 % of energy.

    The samples are treated as uniform bins of equal width centred on
    ``wavelengths_nm``, so the cumulative distribution is piecewise linear
    between bin edges and the 2.5 % / 97.5 % points are interpolated.
    """
    wl = np.asarray(wavelengths_nm, dtype=float) * 1e3  # pm
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    if wl.size < 2:
        return 0.0
    d = np.diff(wl).mean()
    edges = np.concatenate([[wl[0] - d / 2], wl + d / 2])
    cdf = np.concatenate([[0.0], np.cumsum(w)])
    lo = np.interp(0.025, cdf, edges)
    hi = np.interp(0.975, cdf, edges)
    return float(hi - lo)


def coherence_length_um(
    bandwidth_fwhm_pm: float, wavelength_nm: float = WAVELENGTH_ARF
) -> float:
    """Temporal coherence length ``lambda^2 / d_lambda`` in micrometres."""
    return (wavelength_nm * 1e-9) ** 2 / (bandwidth_fwhm_pm * 1e-12) * 1e6


def speckle_contrast(
    n_pulses: float,
    n_spatial_modes: float = 1.0,
    n_temporal_modes: float = 1.0,
    pulse_correlation: float = 0.0,
) -> float:
    """Speckle contrast C = sigma_I / <I> after summing independent patterns.

    A fully developed speckle pattern has C = 1.  Adding M uncorrelated
    patterns of equal mean reduces it to ``1/sqrt(M)``.  For an excimer
    scanner the diversity comes from

    * ``n_pulses`` - pulses integrated per wafer point (each pulse has a new
      transverse-mode / spectral realisation, decorrelated further by the
      illuminator's pulse-to-pulse beam movement);
    * ``n_spatial_modes`` - independent spatial coherence cells mixed by the
      homogeniser (an excimer beam is strongly multimode, M^2 ~ 100s);
    * ``n_temporal_modes`` - roughly the optical-path spread inside the
      illuminator divided by the coherence length (see
      :func:`coherence_length_um`).

    ``pulse_correlation`` (0..1) is the correlation coefficient between
    successive pulses' patterns; the effective number of independent pulses
    is ``N / (1 + (N-1) rho)``.
    """
    n = max(float(n_pulses), 1.0)
    rho = float(np.clip(pulse_correlation, 0.0, 1.0))
    n_eff_pulses = n / (1.0 + (n - 1.0) * rho)
    m = n_eff_pulses * max(n_spatial_modes, 1.0) * max(n_temporal_modes, 1.0)
    return 1.0 / math.sqrt(m)


# =============================================================================
# Discharge kinetics
# =============================================================================
@dataclass
class DischargeModel:
    """0-D kinetics model of an ArF discharge laser (oscillator chamber).

    Physics (each step is a textbook approximation):

    1. The solid-state pulsed-power module charges a storage capacitor
       ``capacitance_f`` to the charging voltage V; a fraction
       ``transfer_efficiency`` of ``C V^2 / 2`` is deposited in the discharge
       over ``pump_duration_ns``.
    2. Electrons excite/ionise Ar; ArF* forms by the harpoon reaction
       Ar* + F2 -> ArF* + F and by Ar+ + F- recombination.  The fraction of
       deposited energy ending up as ArF* rises with F2 and saturates (F2 is
       the fluorine donor, but also captures electrons):
       ``eta_f = eta_max * x / (x + x_half)``, x = F2 mole fraction.
    3. ArF* decays radiatively (``tau_rad_ns`` ~ 4 ns) and is collisionally
       quenched by F2 and by the Ar/Ne buffer, giving an effective lifetime
       ``tau``.
    4. Small-signal gain ``g0 = sigma * R * tau`` with R the ArF* formation
       rate density.  Threshold gain ``g_th = alpha_int + alpha_F2 +
       ln(1/(R_oc R_lnm)) / (2 L)`` includes F2 absorption at 193 nm.
    5. Steady-state homogeneous extraction gives output power per unit volume
       ``I_sat (g0 - g_th) * eta_out``, with ``I_sat = h nu / (sigma tau)`` and
       output-coupling efficiency ``eta_out = mirror loss / total loss``.
       Hence ``E ~ h nu (R - g_th / (sigma tau))``: more F2 helps formation but
       raises the quench-limited threshold term, giving an F2 optimum.

    The model is calibrated to ~10-20 mJ at ~1 kV and ~0.1 % F2, consistent
    with commercial ArF lithography lasers (efficiency of order 1 %).
    """

    capacitance_f: float = 3.0e-6
    transfer_efficiency: float = 0.7
    pump_duration_ns: float = 50.0
    gain_length_cm: float = 50.0
    gain_area_cm2: float = 0.36  # 0.3 cm x 1.2 cm discharge cross-section
    cavity_length_cm: float = 100.0
    pressure_atm: float = 3.5
    f2_nominal: float = 1.0e-3  # mole fraction (0.1 %)
    eta_max: float = 0.20  # max fraction of deposited energy -> ArF* (in photon units)
    x_half: float = 1.0e-3  # F2 fraction at which formation is half saturated
    sigma_cm2: float = 2.9e-16  # ArF stimulated-emission cross-section
    tau_rad_ns: float = 4.2
    k_quench_f2_cm3_s: float = 1.9e-9
    k_quench_buffer_per_atm_s: float = 1.0e8
    sigma_f2_abs_cm2: float = 1.0e-20  # F2 absorption at 193 nm
    alpha_internal_cm: float = 0.004  # transient absorbers (F-, Ne2+, ...)
    r_output_coupler: float = 0.3
    r_lnm: float = 0.5  # effective prism + grating double-pass reflectance
    extraction_efficiency: float = 0.4  # pulse build-up/geometric overlap
    wavelength_nm: float = WAVELENGTH_ARF

    # --- kinetics pieces -----------------------------------------------------
    def deposited_energy_j(self, voltage_v):
        """Energy deposited in the discharge per pulse (J)."""
        v = np.asarray(voltage_v, dtype=float)
        return self.transfer_efficiency * 0.5 * self.capacitance_f * v**2

    def f2_density_cm3(self, f2_fraction):
        return np.asarray(f2_fraction, dtype=float) * self.pressure_atm * _LOSCHMIDT_CM3

    def lifetime_ns(self, f2_fraction):
        """Effective ArF* lifetime including F2 and buffer-gas quenching (ns)."""
        rate = (
            1.0 / (self.tau_rad_ns * 1e-9)
            + self.k_quench_f2_cm3_s * self.f2_density_cm3(f2_fraction)
            + self.k_quench_buffer_per_atm_s * self.pressure_atm
        )
        return 1e9 / rate

    def formation_efficiency(self, f2_fraction):
        x = np.asarray(f2_fraction, dtype=float)
        return self.eta_max * x / (x + self.x_half)

    def formation_rate_cm3_s(self, voltage_v, f2_fraction):
        """Mean ArF* formation rate density during the pump pulse (1/cm^3/s)."""
        vol = self.gain_area_cm2 * self.gain_length_cm
        e_ph = photon_energy(self.wavelength_nm)
        return (
            self.formation_efficiency(f2_fraction)
            * self.deposited_energy_j(voltage_v)
            / (e_ph * vol * self.pump_duration_ns * 1e-9)
        )

    def small_signal_gain(self, voltage_v, f2_fraction):
        """Small-signal gain coefficient g0 (1/cm)."""
        return (
            self.sigma_cm2
            * self.formation_rate_cm3_s(voltage_v, f2_fraction)
            * self.lifetime_ns(f2_fraction)
            * 1e-9
        )

    def mirror_loss_cm(self) -> float:
        return math.log(1.0 / (self.r_output_coupler * self.r_lnm)) / (2.0 * self.gain_length_cm)

    def threshold_gain(self, f2_fraction):
        """Threshold gain (1/cm): internal + F2 absorption + mirror losses."""
        a_f2 = self.sigma_f2_abs_cm2 * self.f2_density_cm3(f2_fraction)
        return self.alpha_internal_cm + a_f2 + self.mirror_loss_cm()

    def saturation_intensity_w_cm2(self, f2_fraction):
        return photon_energy(self.wavelength_nm) / (
            self.sigma_cm2 * self.lifetime_ns(f2_fraction) * 1e-9
        )

    def _energy_slope_mj(self, f2_fraction):
        """Return (a, b) with E[mJ] = a * g0 - b, linear in g0 (i.e. in V^2)."""
        g_th = self.threshold_gain(f2_fraction)
        eta_out = self.mirror_loss_cm() / g_th
        pref = (
            self.saturation_intensity_w_cm2(f2_fraction)
            * self.gain_area_cm2
            * self.gain_length_cm
            * eta_out
            * self.extraction_efficiency
            * self.pump_duration_ns
            * 1e-9
            * 1e3
        )
        return pref, pref * g_th

    def output_energy_mj(self, voltage_v, f2_fraction=None):
        """Output pulse energy (mJ) at charging voltage ``voltage_v`` (V)."""
        x = self.f2_nominal if f2_fraction is None else f2_fraction
        a, b = self._energy_slope_mj(x)
        return np.maximum(a * self.small_signal_gain(voltage_v, x) - b, 0.0)

    def voltage_for_energy(self, energy_mj, f2_fraction=None):
        """Charging voltage (V) needed for ``energy_mj`` (inverse of output_energy_mj)."""
        x = self.f2_nominal if f2_fraction is None else f2_fraction
        a, b = self._energy_slope_mj(x)
        g0_needed = (np.asarray(energy_mj, dtype=float) + b) / a
        g0_per_v2 = self.small_signal_gain(1.0, x)
        return np.sqrt(g0_needed / g0_per_v2)

    def optimum_f2(self, voltage_v: float = 1000.0) -> float:
        """F2 fraction maximising output energy at fixed voltage."""
        xs = np.geomspace(1e-5, 1e-2, 400)
        return float(xs[np.argmax(self.output_energy_mj(voltage_v, xs))])

    # --- time-domain rate equations -----------------------------------------
    def simulate_pulse(self, voltage_v: float = 1000.0, f2_fraction=None, n_t: int = 600) -> dict:
        """Integrate the gain/photon rate equations over one discharge pulse.

        Normalised equations (g: gain coefficient in 1/cm, I: intracavity
        intensity in units of I_sat)::

            dg/dt = (g0(t) - g) / tau - g I / tau
            dI/dt = c (L_g / L_c) (g - g_th) I + beta g

        with ``g0(t) = g0_peak * pi/2 * sin^2(pi t / T_pump)`` (same mean as the
        steady-state model) and a tiny spontaneous-emission seed ``beta``.
        The laser is gain-switched: the pulse starts after a build-up delay
        and stops when the pump falls, so it is shorter than the pump.

        Returns dict with ``t_ns``, ``pump`` (normalised), ``gain``,
        ``intensity`` (I/I_sat), ``fwhm_ns``, ``tis_ns`` (integral-square
        duration) and ``energy_rel`` (integral of I, arbitrary units).
        """
        x = self.f2_nominal if f2_fraction is None else f2_fraction
        tau = float(self.lifetime_ns(x))
        g0_peak = float(self.small_signal_gain(voltage_v, x))
        g_th = float(self.threshold_gain(x))
        tp = self.pump_duration_ns
        rate = _C_CM_PER_NS * self.gain_length_cm / self.cavity_length_cm
        beta = 1e-9

        def pump(t):
            return np.where((t >= 0) & (t <= tp), 0.5 * math.pi * np.sin(math.pi * t / tp) ** 2, 0.0)

        def rhs(t, y):
            g, intensity = y
            g0 = g0_peak * pump(t)
            dg = (g0 - g) / tau - g * intensity / tau
            di = rate * (g - g_th) * intensity + beta * g
            return [dg, di]

        t_end = 1.6 * tp
        t_eval = np.linspace(0.0, t_end, n_t)
        sol = solve_ivp(rhs, (0.0, t_end), [0.0, 0.0], t_eval=t_eval, method="LSODA",
                        rtol=1e-6, atol=[1e-10, 1e-14], max_step=tp / 200)
        g, intensity = sol.y
        intensity = np.maximum(intensity, 0.0)
        dt = t_eval[1] - t_eval[0]
        energy = float(np.sum(intensity) * dt)
        if intensity.max() > 0:
            above = np.where(intensity >= 0.5 * intensity.max())[0]
            fwhm = float((above[-1] - above[0] + 1) * dt)
            tis = float(energy**2 / (np.sum(intensity**2) * dt))
        else:
            fwhm = tis = 0.0
        return {
            "t_ns": t_eval,
            "pump": pump(t_eval),
            "gain": g,
            "intensity": intensity,
            "fwhm_ns": fwhm,
            "tis_ns": tis,
            "energy_rel": energy,
        }


# =============================================================================
# Gas management
# =============================================================================
@dataclass
class GasManager:
    """F2 depletion and injection for an excimer laser chamber.

    F2 is consumed every pulse (reaction with electrode material and chamber
    walls, formation of metal fluorides and of HF/CF4 impurities), modelled as
    first-order loss ``dx/dN = -k x`` with ``depletion_per_pulse = k``.  As F2
    drops, ArF* formation (and thus pulse energy at fixed voltage) falls; the
    laser's energy servo raises the charging voltage to compensate.  When the
    voltage exceeds ``v_inject_threshold`` (or, at constant voltage, when the
    energy has dropped by ``energy_drop_inject``), a small F2 injection
    restores the fluorine fraction to ``discharge.f2_nominal``.

    Attributes
    ----------
    f2_fraction : current F2 mole fraction (starts at nominal).
    pulse_count : pulses fired since gas fill.
    """

    discharge: DischargeModel = field(default_factory=DischargeModel)
    depletion_per_pulse: float = 3.0e-8
    target_energy_mj: float | None = None
    v_inject_threshold: float | None = None
    energy_drop_inject: float = 0.05
    f2_fraction: float = field(default=-1.0)
    pulse_count: int = 0
    injections: list = field(default_factory=list)

    def __post_init__(self):
        if self.f2_fraction < 0:
            self.f2_fraction = self.discharge.f2_nominal
        self._v_nom = 1000.0
        if self.target_energy_mj is None:
            self.target_energy_mj = float(self.discharge.output_energy_mj(self._v_nom))
        if self.v_inject_threshold is None:
            v0 = float(self.discharge.voltage_for_energy(self.target_energy_mj))
            self.v_inject_threshold = 1.04 * v0

    def step(self, n_pulses: int) -> float:
        """Deplete F2 over ``n_pulses`` pulses; returns new F2 fraction."""
        self.f2_fraction *= math.exp(-self.depletion_per_pulse * n_pulses)
        self.pulse_count += int(n_pulses)
        return self.f2_fraction

    def inject(self, to_fraction: float | None = None) -> float:
        """Inject F2 to restore ``to_fraction`` (default: nominal). Returns amount added."""
        target = self.discharge.f2_nominal if to_fraction is None else to_fraction
        added = max(target - self.f2_fraction, 0.0)
        self.f2_fraction += added
        self.injections.append((self.pulse_count, added))
        return added

    def run(
        self,
        n_pulses: int,
        chunk: int = 100_000,
        mode: str = "constant_energy",
        voltage_v: float | None = None,
        auto_inject: bool = True,
    ) -> dict:
        """Simulate gas evolution over ``n_pulses`` in chunks.

        ``mode='constant_energy'``: the voltage servo holds ``target_energy_mj``
        (voltage rises as F2 depletes; injection when V > threshold).
        ``mode='constant_voltage'``: voltage fixed at ``voltage_v`` (default:
        voltage giving the target at nominal F2); energy decays; injection when
        it has dropped by ``energy_drop_inject``.

        Returns dict of arrays: ``pulses``, ``f2_fraction``, ``voltage_v``,
        ``energy_mj``, ``injected`` (bool).
        """
        dm = self.discharge
        if voltage_v is None:
            voltage_v = float(dm.voltage_for_energy(self.target_energy_mj))
        out = {k: [] for k in ("pulses", "f2_fraction", "voltage_v", "energy_mj", "injected")}
        done = 0
        while done < n_pulses:
            n = min(chunk, n_pulses - done)
            self.step(n)
            done += n
            if mode == "constant_energy":
                v = float(dm.voltage_for_energy(self.target_energy_mj, self.f2_fraction))
                e = self.target_energy_mj
                trigger = v > self.v_inject_threshold
            elif mode == "constant_voltage":
                v = voltage_v
                e = float(dm.output_energy_mj(v, self.f2_fraction))
                trigger = e < (1.0 - self.energy_drop_inject) * self.target_energy_mj
            else:
                raise ValueError(f"unknown mode {mode!r}")
            injected = False
            if auto_inject and trigger:
                self.inject()
                injected = True
                if mode == "constant_energy":
                    v = float(dm.voltage_for_energy(self.target_energy_mj, self.f2_fraction))
                else:
                    e = float(dm.output_energy_mj(v, self.f2_fraction))
            out["pulses"].append(self.pulse_count)
            out["f2_fraction"].append(self.f2_fraction)
            out["voltage_v"].append(v)
            out["energy_mj"].append(e)
            out["injected"].append(injected)
        return {k: np.asarray(v) for k, v in out.items()}


# =============================================================================
# Line narrowing
# =============================================================================
@dataclass
class LineNarrowingModule:
    """Prism beam expander + Littrow echelle grating (LNM).

    The grating in Littrow satisfies ``m lambda = 2 d sin(theta)``; its
    angular dispersion is ``dtheta/dlambda = 2 tan(theta) / lambda``.  The
    prism expander magnifies the beam by ``magnification`` M, reducing its
    divergence on the grating by M.  A ray with angular error ``dphi`` in the
    cavity sees the grating dispersion reduced by M, and the standard
    (Duarte) single-pass linewidth estimate is
    ``dlambda = div * (M * dtheta/dlambda)^-1 = lambda * div / (2 M tan theta)``.  The light makes
    ``n_round_trips`` passes during the short gain-switched pulse; repeated
    (approximately Gaussian) filtering narrows the line by ``sqrt(n)``.

    Tuning: rotating the grating (or, in practice, a tuning prism/mirror in
    front of it, driven by a PZT/stepper) by ``dtheta`` shifts the centre
    wavelength by ``lambda / tan(theta) * dtheta``.
    """

    magnification: float = 35.0
    grooves_per_mm: float = 79.0
    order: int = 128
    divergence_mrad: float = 0.4  # full-angle intracavity divergence (FWHM)
    n_round_trips: float = 6.0
    e95_to_fwhm: float = 3.2  # E95/FWHM of ArFLaser's default 50 % Lorentz line shape
    center_wavelength_nm: float = WAVELENGTH_ARF
    tuning_range_pm: float = 400.0

    @property
    def groove_spacing_nm(self) -> float:
        return 1e6 / self.grooves_per_mm

    def littrow_angle_rad(self, wavelength_nm: float | None = None) -> float:
        lam = self.center_wavelength_nm if wavelength_nm is None else wavelength_nm
        s = self.order * lam / (2.0 * self.groove_spacing_nm)
        if not 0 < s < 1:
            raise ValueError("no Littrow solution for this order/wavelength")
        return math.asin(s)

    def wavelength_at_angle(self, theta_rad: float) -> float:
        """Wavelength (nm) retro-reflected at grating angle ``theta_rad``."""
        return 2.0 * self.groove_spacing_nm * math.sin(theta_rad) / self.order

    def angular_dispersion_rad_per_nm(self, wavelength_nm: float | None = None) -> float:
        lam = self.center_wavelength_nm if wavelength_nm is None else wavelength_nm
        return 2.0 * math.tan(self.littrow_angle_rad(lam)) / lam

    def single_pass_bandwidth_pm(self) -> float:
        div = self.divergence_mrad * 1e-3
        return 1e3 * div / (self.magnification * self.angular_dispersion_rad_per_nm())

    def bandwidth_fwhm_pm(self) -> float:
        return self.single_pass_bandwidth_pm() / math.sqrt(max(self.n_round_trips, 1.0))

    def bandwidth_e95_pm(self) -> float:
        return self.e95_to_fwhm * self.bandwidth_fwhm_pm()

    def tune(self, target_wavelength_nm: float) -> float:
        """Set the centre wavelength; returns the required grating rotation (urad)."""
        offset_pm = (target_wavelength_nm - WAVELENGTH_ARF) * 1e3
        if abs(offset_pm) > self.tuning_range_pm:
            raise ValueError("target outside LNM tuning range")
        th0 = self.littrow_angle_rad(self.center_wavelength_nm)
        th1 = self.littrow_angle_rad(target_wavelength_nm)
        self.center_wavelength_nm = target_wavelength_nm
        return (th1 - th0) * 1e6

    def wavelength_shift_pm_per_urad(self) -> float:
        """Centre-wavelength shift per microradian of grating rotation (pm/urad).

        From ``m lambda = 2 d sin(theta)``: ``dlambda/dtheta = lambda / tan(theta)``
        (half the inverse of the double-pass dispersion ``2 tan(theta)/lambda``).
        """
        lam = self.center_wavelength_nm
        return 1e3 * 1e-6 * lam / math.tan(self.littrow_angle_rad(lam))


# =============================================================================
# Spectrum shape helpers (pseudo-Voigt truncated at +-span)
# =============================================================================
def _pv_mass(h, fwhm, eta):
    """Mass of a unit-area pseudo-Voigt within [-h, h]."""
    gam = fwhm / 2.0
    sig = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return eta * (2.0 / math.pi) * np.arctan(h / gam) + (1.0 - eta) * erf(h / (sig * math.sqrt(2.0)))


def _pv_cdf(x, fwhm, eta):
    """CDF of a unit-area pseudo-Voigt centred at 0."""
    x = np.asarray(x, dtype=float)
    return 0.5 + 0.5 * np.sign(x) * _pv_mass(np.abs(x), fwhm, eta)


def _pv_e95(fwhm, eta, span):
    """E95 (full width) of the pseudo-Voigt truncated to [-span, span]."""
    total = _pv_mass(span, fwhm, eta)
    h = brentq(lambda hh: _pv_mass(hh, fwhm, eta) - 0.95 * total, 0.0, span)
    return 2.0 * h


# =============================================================================
# The laser
# =============================================================================
@dataclass
class ArFLaser:
    """Line-narrowed ArF excimer laser (MOPA / ring-amplifier class).

    Parameters
    ----------
    rep_rate_hz : pulse repetition rate (6 kHz for current immersion lasers).
    pulse_energy_mj : nominal pulse energy at ``nominal_voltage_v`` and
        nominal F2 (15 mJ x 6 kHz = 90 W).
    center_wavelength_nm : centre (vacuum) wavelength.
    bandwidth_e95_pm : spectral width containing 95 % of the energy.
    energy_sigma : relative pulse-to-pulse energy jitter (1 sigma).
    pulse_duration_ns : integral-square pulse length T_is after the optical
        pulse stretcher (lowers peak power to protect optics).
    lorentz_fraction : Lorentzian weight of the pseudo-Voigt line shape.
    spectrum_span_e95 : the line shape is truncated to (and sampled over)
        ``+- spectrum_span_e95 * E95`` around the centre.
    wavelength_sigma_fm : pulse-to-pulse centre wavelength jitter (1 sigma).
    burst_spike : relative energy overshoot at the start of a burst,
        decaying with ``burst_spike_pulses`` (thermal/acoustic transient).
    """

    rep_rate_hz: float = 6000.0
    pulse_energy_mj: float = 15.0
    center_wavelength_nm: float = WAVELENGTH_ARF
    bandwidth_e95_pm: float = 0.30
    energy_sigma: float = 0.03
    pulse_duration_ns: float = 40.0
    lorentz_fraction: float = 0.5
    spectrum_span_e95: float = 1.0
    wavelength_sigma_fm: float = 15.0
    burst_spike: float = 0.05
    burst_spike_pulses: float = 15.0
    nominal_voltage_v: float = 1000.0
    discharge: DischargeModel = field(default_factory=DischargeModel)
    f2_fraction: float | None = None

    def __post_init__(self):
        if self.f2_fraction is None:
            self.f2_fraction = self.discharge.f2_nominal
        e_model = float(self.discharge.output_energy_mj(self.nominal_voltage_v, self.discharge.f2_nominal))
        self._energy_scale = self.pulse_energy_mj / e_model

    # --- derived quantities ---------------------------------------------------
    @property
    def average_power_w(self) -> float:
        return self.pulse_energy_mj * 1e-3 * self.rep_rate_hz

    @property
    def peak_power_mw(self) -> float:
        """Approximate peak power (MW) using T_is."""
        return self.pulse_energy_mj * 1e-3 / (self.pulse_duration_ns * 1e-9) * 1e-6

    @property
    def photons_per_pulse(self) -> float:
        return self.pulse_energy_mj * 1e-3 / photon_energy(self.center_wavelength_nm)

    # --- spectrum -------------------------------------------------------------
    def line_fwhm_pm(self) -> float:
        """FWHM (pm) of the pseudo-Voigt whose truncated E95 equals ``bandwidth_e95_pm``."""
        e95 = self.bandwidth_e95_pm
        span = self.spectrum_span_e95 * e95
        eta = self.lorentz_fraction
        return brentq(lambda w: _pv_e95(w, eta, span) - e95, 1e-4 * e95, 50.0 * e95)

    def spectrum(self, n_samples: int = 21, center_nm: float | None = None):
        """Line-narrowed spectrum sampled on ``n_samples`` equal bins.

        Shape: pseudo-Voigt (``lorentz_fraction`` Lorentzian + Gaussian, same
        FWHM) truncated to ``+- spectrum_span_e95 * E95``, with FWHM chosen so
        that the E95 width equals ``bandwidth_e95_pm``.  Weights are the exact
        integrals of the shape over each bin, so they sum to 1 and
        :func:`e95_width_pm` of the samples reproduces the target E95.

        Returns ``(wavelengths_nm, weights)``.
        """
        n = int(n_samples)
        c = self.center_wavelength_nm if center_nm is None else center_nm
        if n == 1:
            return np.array([c]), np.array([1.0])
        fwhm = self.line_fwhm_pm()
        span = self.spectrum_span_e95 * self.bandwidth_e95_pm
        edges = np.linspace(-span, span, n + 1)
        cdf = _pv_cdf(edges, fwhm, self.lorentz_fraction)
        w = np.diff(cdf)
        w = w / w.sum()
        centers_pm = 0.5 * (edges[1:] + edges[:-1])
        return c + centers_pm * 1e-3, w

    def coherence_length_um(self) -> float:
        return coherence_length_um(self.line_fwhm_pm(), self.center_wavelength_nm)

    # --- energy ---------------------------------------------------------------
    def energy_at_voltage(self, voltage_v, f2_fraction=None):
        """Noise-free model pulse energy (mJ) at a charging voltage."""
        x = self.f2_fraction if f2_fraction is None else f2_fraction
        return self._energy_scale * self.discharge.output_energy_mj(voltage_v, x)

    def voltage_for_energy(self, energy_mj, f2_fraction=None):
        """Charging voltage (V) giving the model energy ``energy_mj``."""
        x = self.f2_fraction if f2_fraction is None else f2_fraction
        return self.discharge.voltage_for_energy(np.asarray(energy_mj) / self._energy_scale, x)

    def fire(self, n_pulses: int, rng=None, voltage_command=None, start_index: int = 0) -> np.ndarray:
        """Fire ``n_pulses`` pulses; returns pulse energies (mJ).

        ``voltage_command`` (scalar or per-pulse array, V) sets the charging
        voltage; default is ``nominal_voltage_v``.  Each energy is the model
        energy times the burst-start transient
        ``1 + burst_spike * exp(-k / burst_spike_pulses)`` (``k`` = pulse index
        within the burst, starting at ``start_index``) times
        ``(1 + energy_sigma * N(0, 1))``.
        """
        rng = np.random.default_rng() if rng is None else rng
        n = int(n_pulses)
        v = self.nominal_voltage_v if voltage_command is None else voltage_command
        v = np.broadcast_to(np.asarray(v, dtype=float), (n,))
        k = start_index + np.arange(n)
        spike = 1.0 + self.burst_spike * np.exp(-k / max(self.burst_spike_pulses, 1e-9))
        e = self.energy_at_voltage(v) * spike * (1.0 + self.energy_sigma * rng.standard_normal(n))
        return np.maximum(e, 0.0)

    # --- wavelength -----------------------------------------------------------
    def wavelength_stability(
        self,
        n_pulses: int = 20000,
        rng=None,
        closed_loop: bool = True,
        target_nm: float | None = None,
        drift_fm_per_pulse: float = 0.02,
        random_walk_fm: float = 1.0,
        wavemeter_sigma_fm: float = 5.0,
        loop_gain: float = 0.05,
        window: int | None = None,
    ) -> dict:
        """Per-pulse centre wavelength with/without wavemeter feedback on the LNM.

        Disturbances: slow LNM thermal drift (``drift_fm_per_pulse`` ramp plus
        a random walk of ``random_walk_fm`` per pulse) and uncorrelated
        pulse-to-pulse jitter ``wavelength_sigma_fm``.  The on-board wavemeter
        (etalon + grating spectrometer, calibrated against an atomic absorption
        reference) measures each pulse with noise ``wavemeter_sigma_fm``; an
        integral controller moves the LNM tuning actuator by
        ``-loop_gain * measured error`` after every pulse.

        Returns dict with ``wavelength_nm``, ``error_fm``, ``moving_average_fm``
        and ``moving_std_fm`` (over ``window`` pulses, default: pulses per
        point at 8 mm / 700 mm/s) and summary scalars ``ma_rms_fm``,
        ``ma_max_fm``, ``msd_mean_fm``.
        """
        rng = np.random.default_rng() if rng is None else rng
        target = self.center_wavelength_nm if target_nm is None else target_nm
        n = int(n_pulses)
        if window is None:
            window = int(round(pulses_per_point(SLIT_HEIGHT_MM, 700.0, self.rep_rate_hz)))
        k = np.arange(n)
        disturbance = drift_fm_per_pulse * k + np.cumsum(random_walk_fm * rng.standard_normal(n))
        jitter = self.wavelength_sigma_fm * rng.standard_normal(n)
        meas_noise = wavemeter_sigma_fm * rng.standard_normal(n)
        err = np.empty(n)
        u = 0.0  # actuator correction (fm)
        for i in range(n):
            err[i] = disturbance[i] + u + jitter[i]
            if closed_loop:
                u -= loop_gain * (err[i] + meas_noise[i])
        w = min(window, n)
        ma = window_sums(err, w) / w
        ma2 = window_sums(err**2, w) / w
        msd = np.sqrt(np.maximum(ma2 - ma**2, 0.0))
        return {
            "wavelength_nm": target + err * 1e-6,
            "error_fm": err,
            "moving_average_fm": ma,
            "moving_std_fm": msd,
            "ma_rms_fm": float(np.sqrt(np.mean(ma**2))),
            "ma_max_fm": float(np.max(np.abs(ma))),
            "msd_mean_fm": float(np.mean(msd)),
        }


# =============================================================================
# Dose control
# =============================================================================
@dataclass
class DoseControlResult:
    """Outcome of a :meth:`DoseController.run` burst."""

    commanded_mj: np.ndarray
    measured_mj: np.ndarray
    actual_mj: np.ndarray
    voltage_v: np.ndarray
    window_dose_mj_cm2: np.ndarray
    target_dose_mj_cm2: float
    stats: dict


@dataclass
class DoseController:
    """Pulse-by-pulse dose control for a scanning exposure.

    Each wafer point integrates the ``n_window`` consecutive pulses fired
    while it is inside the slit, so the dose error of interest is the error
    of every moving-window sum.  Per pulse ``k`` the controller

    1. computes the energy surplus of the previous ``n_window - 1`` measured
       pulses, ``S = sum(E_meas - E_t)``, and commands
       ``E_cmd = E_t - dose_gain * S`` (moving-window dose servo; for
       ``0 < dose_gain <= 1`` the loop is stable and the window error becomes
       ``noise_k + (1 - dose_gain) S``);
    2. converts ``E_cmd`` to a charging voltage via the laser model divided by
       an efficiency estimate ``a_hat`` that tracks slow laser drifts
       (burst-start spike, F2 depletion) with gain ``efficiency_gain``;
    3. fires and measures the pulse with an energy monitor of relative noise
       ``meter_sigma``.

    Open-loop mode fires at the fixed voltage that nominally gives ``E_t``.
    """

    target_energy_mj: float = 15.0
    n_window: int = 69
    dose_gain: float = 0.6
    efficiency_gain: float = 0.2
    meter_sigma: float = 0.003
    transmission: float = 0.08
    command_limits: tuple = (0.6, 1.4)

    def run(self, laser: ArFLaser, n_pulses: int = 3000, closed_loop: bool = True, rng=None) -> DoseControlResult:
        rng = np.random.default_rng() if rng is None else rng
        n = int(n_pulses)
        nw = int(self.n_window)
        et = self.target_energy_mj
        cmd = np.empty(n)
        meas = np.empty(n)
        act = np.empty(n)
        volt = np.empty(n)
        v_open = float(laser.voltage_for_energy(et))
        a_hat = 1.0
        surplus = 0.0  # running sum of (meas - et) over the last nw-1 pulses
        lo, hi = self.command_limits
        meter_noise = self.meter_sigma * rng.standard_normal(n)
        for k in range(n):
            if closed_loop:
                e_cmd = float(np.clip(et - self.dose_gain * surplus, lo * et, hi * et))
                v = float(laser.voltage_for_energy(e_cmd / a_hat))
            else:
                e_cmd = et
                v = v_open
            e = float(laser.fire(1, rng=rng, voltage_command=v, start_index=k)[0])
            m = e * (1.0 + meter_noise[k])
            if closed_loop:
                predicted = float(laser.energy_at_voltage(v))
                if predicted > 0:
                    a_hat += self.efficiency_gain * (m / predicted - a_hat)
            cmd[k], meas[k], act[k], volt[k] = e_cmd, m, e, v
            surplus += m - et
            if k - (nw - 1) >= 0:
                surplus -= meas[k - (nw - 1)] - et
        area_cm2 = (FIELD_WIDTH_MM * 0.1) * (SLIT_HEIGHT_MM * 0.1)
        win = self.transmission * window_sums(act, nw) / area_cm2
        target_dose = dose_from_pulses(np.full(nw, et), self.transmission)
        rel = win / target_dose - 1.0
        stats = {
            "mean_error": float(rel.mean()),
            "std_error": float(rel.std()),
            "three_sigma": float(3.0 * rel.std() + abs(rel.mean())),
            "max_abs_error": float(np.max(np.abs(rel))),
            "pulse_energy_std_rel": float(act.std() / act.mean()),
        }
        return DoseControlResult(cmd, meas, act, volt, win, target_dose, stats)
