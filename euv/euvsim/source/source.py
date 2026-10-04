"""Top-level LPP EUV source: power budget, pulse statistics, dose control.

Power budget (time-averaged, in-band = 13.5 nm 2 % BW)
-------------------------------------------------------

    P_IF = P_laser * T_bt * <eta_couple> * CE(I_pk) * (Omega / 2 pi) * A_emis
           * <R_coll> * F_life * T_gas * T_obsc * T_IF * s_0

  P_laser          CO2 average power at the amplifier output (~20-40 kW)
  T_bt             beam transport + final focus transmission (~0.85)
  <eta_couple>     mean fraction of the main pulse intercepted by the target
                   (Gaussian beam vs pancake, with droplet/pointing jitter)
  CE(I_pk)         conversion efficiency into 2 % BW, 2 pi sr (~5.5 %)
  Omega / 2 pi     collector solid angle relative to 2 pi (5 sr -> 0.8)
  A_emis           angular emission anisotropy over the collector cone (~0.9)
  <R_coll>         emission-weighted collector reflectance (fresh, ~0.65)
  F_life           life-averaged reflectance factor (Sn + intrinsic, ~0.9)
  T_gas            H2 absorption along plasma->collector->IF
  T_obsc           obscuration by droplet catcher, vanes, struts (~0.85)
  T_IF             IF aperture clipping / IR-filter grating loss (~0.9)
  s_0              dose-control headroom: nominal pulse-energy setpoint as
                   fraction of maximum (~0.9)

The default configuration gives ~250-280 W at IF (NXE:3400C/3600D class);
:meth:`LPPSource.roadmap` gives a ~500+ W configuration.

Dose stability
--------------
A point on the wafer integrates N = f_rep * h_slit / v_scan pulses.  Open loop
the relative dose error is sigma_dose = sigma_pulse / sqrt(N).  The dose
controller adjusts each pulse's energy request from the accumulated dose
error (integral feedback), making pulse errors anti-correlated and
reducing the windowed dose error well below sigma_pulse / sqrt(N).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import PHOTON_ENERGY_EV, SLIT_HEIGHT_MM
from .collector import EllipsoidalCollector
from .debris import DebrisModel, h2_number_density
from .drive_laser import DriveLaser
from .droplet import DropletGenerator, PrePulseTarget, coupling_fraction, sample_offsets
from .plasma import SnPlasma

H2_ABS_CROSS_SECTION_M2 = 2.0e-24   # per H2 molecule at 13.5 nm (~92 eV)


@dataclass
class DoseController:
    """Integral pulse-energy feedback.

    Per pulse k the energy request (fraction of max) is
        s_k = clip(s_0 + g * (sum_{j<k} (1 - e_j)) * s_0, s_min, s_max)
    where e_j is the delivered energy of pulse j relative to the nominal
    (setpoint) energy.  g ~ 0.3-0.7 gives good dead-beat-like behaviour.
    """

    gain: float = 0.5
    setpoint: float = 0.9      # s_0, fraction of maximum pulse energy
    s_min: float = 0.3
    s_max: float = 1.0


@dataclass
class LPPSource:
    """Laser-produced-plasma EUV source (CO2 + Sn, MOPA pre-pulse scheme)."""

    laser: DriveLaser = field(default_factory=DriveLaser)
    droplets: DropletGenerator = field(default_factory=DropletGenerator)
    target: PrePulseTarget = field(default_factory=PrePulseTarget)
    plasma: SnPlasma = field(default_factory=SnPlasma)
    collector: EllipsoidalCollector = field(default_factory=EllipsoidalCollector)
    debris: DebrisModel = field(default_factory=DebrisModel)
    dose_controller: DoseController = field(default_factory=DoseController)

    beam_transport: float = 0.85
    emission_anisotropy: float = 0.90
    obscuration_transmission: float = 0.85
    if_transmission: float = 0.90
    collector_to_if_pressure_pa: float = 10.0
    misfire_probability: float = 1e-5        # pulses with no plasma (missed droplet)
    # spectral filtering of out-of-band light (collector + IR grating)
    duv_collector_reflectance: float = 0.4
    ir_collector_reflectance: float = 1e-3
    # availability model
    mtbf_h: float = 250.0
    mttr_h: float = 6.0
    droplet_generator_swap_h: float = 4.0
    collector_swap_h: float = 36.0

    def __post_init__(self) -> None:
        if abs(self.droplets.rep_rate_hz - self.laser.rep_rate_hz) > 1e-6:
            raise ValueError("droplet and laser repetition rates must match")
        self.debris.collector_solid_angle_sr = self.collector.solid_angle_sr
        self.debris.distance_to_collector_m = self.collector.vertex_distance_m

    # -- convenience constructors --------------------------------------------------
    @classmethod
    def nxe3400(cls) -> "LPPSource":
        """Default ~250 W configuration."""
        return cls()

    @classmethod
    def roadmap(cls) -> "LPPSource":
        """~500+ W roadmap: 40 kW CO2, 6 % CE, 62.5 kHz."""
        f = 62.5e3
        return cls(laser=DriveLaser(average_power_w=40e3, rep_rate_hz=f),
                   droplets=DropletGenerator(rep_rate_hz=f, velocity_m_s=85.0),
                   plasma=SnPlasma(ce_max=0.06))

    # -- basic properties --------------------------------------------------------------
    @property
    def rep_rate_hz(self) -> float:
        return self.laser.rep_rate_hz

    @property
    def photon_energy_ev(self) -> float:
        return PHOTON_ENERGY_EV

    @property
    def peak_intensity_w_cm2(self) -> float:
        return self.laser.main_pulse.peak_intensity_w_cm2

    @property
    def electron_temperature_ev(self) -> float:
        return float(self.plasma.electron_temperature_ev(self.peak_intensity_w_cm2,
                                                         self.laser.main_wavelength_um))

    @property
    def conversion_efficiency(self) -> float:
        return float(self.plasma.conversion_efficiency(self.peak_intensity_w_cm2))

    @property
    def target_radius_m(self) -> float:
        return self.target.radius_m(self.droplets, self.laser.prepulse_energy_j,
                                    self.laser.prepulse_delay_s)

    @property
    def targeting_sigma_m(self) -> float:
        """Per-axis laser-target offset sigma: droplet jitter (+) pointing jitter."""
        return float(np.hypot(self.droplets.position_sigma_m, self.laser.pointing_sigma_m))

    def mean_coupling(self) -> float:
        """<eta> averaged over the Rayleigh-distributed offset."""
        s = self.targeting_sigma_m
        r = np.linspace(0, 8 * s, 801)
        pdf = r / s ** 2 * np.exp(-r ** 2 / (2 * s ** 2))
        eta = coupling_fraction(r, self.target_radius_m, self.laser.main_spot_radius_m)
        return float(np.trapezoid(pdf * eta, r) / np.trapezoid(pdf, r))

    # -- power budget ----------------------------------------------------------------
    def gas_transmission(self) -> float:
        """exp(-sigma n L) for plasma->collector (buffer gas) and collector->IF."""
        n1 = h2_number_density(self.debris.h2_pressure_pa, self.debris.gas_temperature_k)
        n2 = h2_number_density(self.collector_to_if_pressure_pa, self.debris.gas_temperature_k)
        L1 = self.collector.vertex_distance_m * 1.5      # average path to mirror
        L2 = self.collector.focal_separation_m + 0.1
        return float(np.exp(-H2_ABS_CROSS_SECTION_M2 * (n1 * L1 + n2 * L2)))

    def sn_deposition_nm_h(self) -> float:
        return self.debris.deposition_rate_nm_h(self.droplets.atoms_per_droplet, self.rep_rate_hz)

    def collector_lifetime_gpulses(self, max_loss_rel: float = 0.2) -> float:
        return self.debris.collector_lifetime_gpulses(self.sn_deposition_nm_h(), self.rep_rate_hz,
                                                      max_loss_rel)

    def life_averaged_reflectance_factor(self, max_loss_rel: float = 0.2) -> float:
        """Mean of R(t)/R0 over one collector lifetime (or 1e4 h if unlimited)."""
        life_g = self.collector_lifetime_gpulses(max_loss_rel)
        hours = 1e4 if not np.isfinite(life_g) else life_g * 1e9 / self.rep_rate_hz / 3600
        t = np.linspace(0, hours, 2001)
        return float(np.mean(self.debris.reflectance_factor(t, self.sn_deposition_nm_h(),
                                                            self.rep_rate_hz)))

    def _chain_after_plasma(self) -> float:
        """Collection-to-IF efficiency for in-band light emitted into 2 pi."""
        return (self.collector.collection_fraction_2pi * self.emission_anisotropy
                * self.collector.average_reflectance() * self.life_averaged_reflectance_factor()
                * self.gas_transmission() * self.obscuration_transmission * self.if_transmission)

    def inband_power_2pi_w(self, at_setpoint: bool = True) -> float:
        """In-band power emitted into 2 pi sr (W)."""
        s = self.dose_controller.setpoint if at_setpoint else 1.0
        return (self.laser.average_power_w * s * self.beam_transport * self.mean_coupling()
                * self.conversion_efficiency * (1 - self.misfire_probability))

    def inband_power_at_if_w(self, at_setpoint: bool = True) -> float:
        """Usable in-band (2 % BW) EUV power at the intermediate focus (W)."""
        return self.inband_power_2pi_w(at_setpoint) * self._chain_after_plasma()

    def nominal_pulse_energy_if_j(self) -> float:
        return self.inband_power_at_if_w() / self.rep_rate_hz

    def power_budget(self) -> dict:
        """Itemised power budget (useful for teaching / plots)."""
        return {
            "laser_average_power_w": self.laser.average_power_w,
            "dose_headroom_setpoint": self.dose_controller.setpoint,
            "beam_transport": self.beam_transport,
            "mean_coupling": self.mean_coupling(),
            "peak_intensity_w_cm2": self.peak_intensity_w_cm2,
            "conversion_efficiency": self.conversion_efficiency,
            "inband_power_2pi_w": self.inband_power_2pi_w(),
            "collection_fraction_2pi": self.collector.collection_fraction_2pi,
            "emission_anisotropy": self.emission_anisotropy,
            "collector_reflectance": self.collector.average_reflectance(),
            "life_averaged_reflectance_factor": self.life_averaged_reflectance_factor(),
            "gas_transmission": self.gas_transmission(),
            "obscuration_transmission": self.obscuration_transmission,
            "if_transmission": self.if_transmission,
            "inband_power_at_if_w": self.inband_power_at_if_w(),
        }

    # -- stochastic pulses ---------------------------------------------------------------
    def _pulse_energies(self, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """In-band IF energies (J) for pulses with energy requests ``scale`` (fraction of max)."""
        scale = np.atleast_1d(np.asarray(scale, dtype=float))
        n = scale.size
        e_laser = self.laser.sample_energies(n, rng, scale)
        offsets = sample_offsets(n, self.targeting_sigma_m, rng)
        eta = coupling_fraction(offsets, self.target_radius_m, self.laser.main_spot_radius_m)
        intensity = self.peak_intensity_w_cm2 * e_laser / self.laser.main_pulse_energy_j
        ce = self.plasma.conversion_efficiency(intensity) * \
            (1 + self.plasma.ce_sigma_rel * rng.standard_normal(n))
        fire = rng.random(n) >= self.misfire_probability
        e = e_laser * self.beam_transport * eta * np.clip(ce, 0, None) * fire
        return e * self._chain_after_plasma_cached

    def pulse_energy_series(self, n: int, rng: np.random.Generator,
                            controlled: bool = False) -> np.ndarray:
        """In-band pulse energies at IF (J) for ``n`` consecutive pulses.

        Open loop (default) every pulse is requested at the setpoint; with
        ``controlled=True`` the :class:`DoseController` adjusts each request.
        """
        self._chain_after_plasma_cached = self._chain_after_plasma()
        dc = self.dose_controller
        if not controlled:
            return self._pulse_energies(np.full(n, dc.setpoint), rng)
        e_nom = self.nominal_pulse_energy_if_j()
        out = np.empty(n)
        acc_err = 0.0
        for k in range(n):
            s = np.clip(dc.setpoint * (1 + dc.gain * acc_err), dc.s_min, dc.s_max)
            out[k] = self._pulse_energies(np.array([s]), rng)[0]
            acc_err += 1.0 - out[k] / e_nom
        return out

    def pulse_sigma_rel(self, rng: np.random.Generator, n: int = 20000) -> float:
        e = self.pulse_energy_series(n, rng)
        return float(np.std(e) / np.mean(e))

    @staticmethod
    def pulses_in_slit(rep_rate_hz: float, scan_speed_m_s: float,
                       slit_height_mm: float = SLIT_HEIGHT_MM) -> int:
        """N = f_rep * h_slit / v_scan."""
        return max(1, int(round(rep_rate_hz * slit_height_mm * 1e-3 / scan_speed_m_s)))

    def dose_error_3sigma(self, n_slit: int, rng: np.random.Generator, n_total: int = 20000,
                          controlled: bool = False) -> float:
        """3 sigma relative dose error of an N-pulse moving window (slit integration)."""
        e = self.pulse_energy_series(n_total, rng, controlled=controlled)
        c = np.concatenate([[0.0], np.cumsum(e)])
        window = c[n_slit:] - c[:-n_slit]
        return float(3 * np.std(window) / np.mean(window))

    # -- spectrum ------------------------------------------------------------------------
    def spectrum(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Time-averaged spectral power emitted by the plasma into 2 pi sr (W/nm).

        Integrates to :meth:`inband_power_2pi_w` over 13.365-13.635 nm, plus
        broad out-of-band EUV, DUV (130-400 nm) and reflected 10.6 um IR.
        """
        return self.plasma.spectrum(
            wavelengths_nm, self.inband_power_2pi_w(),
            laser_power_w=self.laser.average_power_w * self.dose_controller.setpoint,
            te_ev=self.electron_temperature_ev,
            laser_wavelength_um=self.laser.main_wavelength_um)

    def spectrum_at_if(self, wavelengths_nm: np.ndarray) -> np.ndarray:
        """Spectral power density at IF (W/nm): plasma spectrum x collector R(lambda)
        (Mo/Si between 11-16 nm; parametric DUV / IR reflectance elsewhere) x the
        geometric/gas/obscuration chain."""
        wl = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
        R = np.zeros_like(wl)
        euv = (wl >= 11.0) & (wl <= 16.0)
        if euv.any():
            R[euv] = self.collector.spectral_reflectance(wl[euv])
        R[(wl >= 100.0) & (wl <= 1000.0)] = self.duv_collector_reflectance
        R[wl > 1000.0] = self.ir_collector_reflectance
        geo = (self.collector.collection_fraction_2pi * self.emission_anisotropy
               * self.life_averaged_reflectance_factor() * self.obscuration_transmission
               * self.if_transmission)
        return self.spectrum(wl) * R * geo * np.where(euv, self.gas_transmission(), 1.0)

    # -- availability ----------------------------------------------------------------------
    def availability(self) -> float:
        """Fraction of time the source is available:
        A = 1 / (1 + MTTR/MTBF + t_dg/T_dg + t_coll/T_coll)."""
        t_dg = self.droplets.runtime_h
        life_g = self.collector_lifetime_gpulses()
        t_coll = np.inf if not np.isfinite(life_g) else life_g * 1e9 / self.rep_rate_hz / 3600
        down = self.mttr_h / self.mtbf_h + self.droplet_generator_swap_h / t_dg + \
            self.collector_swap_h / t_coll
        return float(1.0 / (1.0 + down))

    # -- etendue ----------------------------------------------------------------------------
    def etendue_mm2_sr(self) -> float:
        return self.collector.etendue_mm2_sr(self.plasma.plasma_diameter_m)
