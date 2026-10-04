"""CO2 drive laser (MOPA) with a solid-state pre-pulse.

Industrial LPP sources use a master-oscillator / power-amplifier (MOPA) CO2
laser at 10.6 um: a low-power seed is amplified by a chain of fast-axial-flow
CO2 amplifiers to ~20-40 kW average power at ~50 kHz, i.e. ~0.4-0.8 J per
main pulse of ~10-30 ns.  A weaker pre-pulse (1.06/1.03 um solid state, ns or
ps) arrives ~1-3 us earlier and flattens the Sn droplet into a pancake / mist
that is then heated by the main pulse.

Key relations
-------------
* main-pulse energy           E_mp = P_avg / f_rep
* peak power (Gaussian pulse) P_pk ~= 0.94 E / tau_FWHM
* peak intensity (TEM00)      I_pk = 2 P_pk / (pi w^2),  w = 1/e^2 radius
* critical density            n_c = eps0 m_e w_L^2 / e^2 = 1.1e21 / lambda_um^2 cm^-3
  (~1e19 cm^-3 for CO2: CO2 light is absorbed in a lower-density, less opaque
  corona than 1 um light, which is why CO2 drive gives a narrower spectrum
  and higher conversion efficiency.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAUSS_PEAK_FACTOR = 2.0 * np.sqrt(np.log(2.0) / np.pi)  # 0.939: P_pk = f E / tau_FWHM


@dataclass
class LaserPulse:
    """A single laser pulse definition."""

    wavelength_um: float
    energy_j: float
    duration_s: float          # FWHM
    spot_radius_m: float       # 1/e^2 intensity radius at focus

    @property
    def peak_power_w(self) -> float:
        return GAUSS_PEAK_FACTOR * self.energy_j / self.duration_s

    @property
    def peak_intensity_w_cm2(self) -> float:
        """I_pk = 2 P_pk / (pi w^2) in W/cm^2."""
        w_cm = self.spot_radius_m * 100.0
        return 2.0 * self.peak_power_w / (np.pi * w_cm ** 2)

    @property
    def critical_density_cm3(self) -> float:
        """Plasma critical density n_c = 1.115e21 / lambda_um^2 cm^-3."""
        return 1.115e21 / self.wavelength_um ** 2


@dataclass
class DriveLaser:
    """CO2 MOPA main pulse + solid-state pre-pulse.

    Defaults are representative of published NXE:3400-class numbers
    (Fomenkov et al., Adv. Opt. Techn. 2017; Mizoguchi et al. for Gigaphoton):
    ~22 kW average CO2 power at 50 kHz, 15 ns pulses focused to ~150 um.
    """

    rep_rate_hz: float = 50e3
    average_power_w: float = 22e3            # main pulse (CO2) average power
    main_wavelength_um: float = 10.6
    main_duration_s: float = 15e-9
    main_spot_radius_m: float = 150e-6
    prepulse_wavelength_um: float = 1.064
    prepulse_energy_j: float = 5e-3
    prepulse_duration_s: float = 10e-9       # ns pre-pulse (ps option: ~10e-12)
    prepulse_spot_radius_m: float = 40e-6
    prepulse_delay_s: float = 1.5e-6         # pre-pulse -> main-pulse delay
    energy_sigma_rel: float = 0.02           # pulse-to-pulse energy jitter (1 sigma)
    pointing_sigma_m: float = 3e-6           # beam pointing jitter at focus (per axis)
    wallplug_efficiency: float = 0.02        # CO2 MOPA wall-plug (~1-2 %)

    @property
    def main_pulse_energy_j(self) -> float:
        """E_mp = P_avg / f_rep."""
        return self.average_power_w / self.rep_rate_hz

    @property
    def main_pulse(self) -> LaserPulse:
        return LaserPulse(self.main_wavelength_um, self.main_pulse_energy_j,
                          self.main_duration_s, self.main_spot_radius_m)

    @property
    def prepulse(self) -> LaserPulse:
        return LaserPulse(self.prepulse_wavelength_um, self.prepulse_energy_j,
                          self.prepulse_duration_s, self.prepulse_spot_radius_m)

    @property
    def wallplug_power_w(self) -> float:
        """Electrical power drawn by the laser: P_avg / eta_wallplug (~1 MW class)."""
        return self.average_power_w / self.wallplug_efficiency

    def sample_energies(self, n: int, rng: np.random.Generator,
                        scale: np.ndarray | float = 1.0) -> np.ndarray:
        """Main-pulse energies (J) with Gaussian jitter; ``scale`` is a per-pulse
        energy request factor (used by the dose controller)."""
        e = self.main_pulse_energy_j * np.asarray(scale, dtype=float)
        return np.clip(e * (1.0 + self.energy_sigma_rel * rng.standard_normal(n)), 0.0, None)
