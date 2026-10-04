"""Sn debris, H2 buffer-gas mitigation, H-radical cleaning and collector lifetime.

Debris species
--------------
* fast ions (Sn^q+, ~0.1-5 keV, roughly exponential energy distribution),
* neutral atoms / vapour (thermal, ~eV),
* micro-particles / fragments (um-scale, from incomplete target heating).

Ion stopping in H2
------------------
At keV energies a heavy Sn ion in H2 loses energy by electronic stopping
(LSS: S_e ~ sqrt(E)) and a weakly energy-dependent nuclear part:

    -dE/dx = n_H2 (a sqrt(E) + b),     n_H2 = p / (k_B T)

which integrates to the range

    R(E) = 2 / (n a) [ sqrt(E) - (b / a) ln(1 + a sqrt(E) / b) ].

The range scales as 1/p: the H2 pressure (~50-150 Pa) is chosen so that the
most energetic ions are thermalised before reaching the collector
(~0.2 m away) while EUV absorption by the gas stays small (a few %).
a, b are calibrated so a 3 keV Sn ion has a range of ~0.1 m at 100 Pa,
consistent with published SRIM-based estimates.

Magnetic mitigation (option): ions with Larmor radius
r_L = sqrt(2 m E) / (q e B) smaller than half the plasma-collector
distance are confined and guided to ion catchers.

H-radical cleaning
------------------
Atomic H (from EUV/plasma dissociation of H2) etches Sn as volatile stannane
Sn + 4H -> SnH4.  The etch rate is the H flux Gamma_H = n_H v_bar / 4 times
an effective etch yield Y (redeposition included, ~1e-7..1e-5) times the
Sn atomic volume.  Because etching needs Sn to be present, we use a
coverage-limited balance for the Sn film thickness delta:

    d delta / dt = D - C (1 - exp(-delta / delta0))

which has a steady state delta_ss = -delta0 ln(1 - D / C) for D < C and grows
at (D - C) otherwise.

Reflectance loss: a Sn film of thickness delta attenuates the reflected beam
by exp(-2 alpha delta / cos theta), alpha = 4 pi k_Sn / lambda (~0.067 /nm,
i.e. ~1 nm Sn costs ~13 % of R).  An additional intrinsic degradation per
pulse (blistering, interdiffusion, H-induced damage) of ~0.1 %/Gpulse is
added.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import AMU, E_CHARGE, K_BOLTZMANN, OPTICAL_CONSTANTS_13P5, WAVELENGTH_NM

SN_MASS_KG = 118.71 * AMU
SN_ATOMIC_VOLUME_M3 = SN_MASS_KG / 7300.0      # solid Sn (beta) density 7.3 g/cm^3
H_ATOM_MASS_KG = 1.008 * AMU
SN_ALPHA_PER_NM = 4 * np.pi * OPTICAL_CONSTANTS_13P5["Sn"][1] / WAVELENGTH_NM


def h2_number_density(pressure_pa: float, temperature_k: float = 300.0) -> float:
    """n = p / (k_B T)  [m^-3]."""
    return pressure_pa / (K_BOLTZMANN * temperature_k)


@dataclass
class IonStopping:
    """Sn-ion stopping in H2: -dE/dx = n (a sqrt(E) + b), E in eV, x in m."""

    a_ev_m2: float = 2.2e-20      # eV m^2 / sqrt(eV)  (electronic, LSS-like)
    b_ev_m2: float = 3.0e-19      # eV m^2             (nuclear, ~constant)

    def stopping_power_ev_per_m(self, energy_ev: np.ndarray | float, pressure_pa: float,
                                temperature_k: float = 300.0) -> np.ndarray:
        n = h2_number_density(pressure_pa, temperature_k)
        return n * (self.a_ev_m2 * np.sqrt(np.asarray(energy_ev, dtype=float)) + self.b_ev_m2)

    def range_m(self, energy_ev: np.ndarray | float, pressure_pa: float,
                temperature_k: float = 300.0) -> np.ndarray:
        """Closed-form range R(E) (m); infinite in vacuum."""
        if pressure_pa <= 0:
            return np.full_like(np.asarray(energy_ev, dtype=float), np.inf)
        n = h2_number_density(pressure_pa, temperature_k)
        a, b = self.a_ev_m2, self.b_ev_m2
        s = np.sqrt(np.asarray(energy_ev, dtype=float))
        return 2.0 / (n * a) * (s - (b / a) * np.log1p(a * s / b))


def larmor_radius_m(energy_ev: np.ndarray | float, charge: int, b_tesla: float) -> np.ndarray:
    """r_L = sqrt(2 m E) / (q e B) for a Sn ion."""
    p = np.sqrt(2.0 * SN_MASS_KG * np.asarray(energy_ev, dtype=float) * E_CHARGE)
    return p / (charge * E_CHARGE * max(b_tesla, 1e-30))


@dataclass
class DebrisModel:
    """Debris generation, mitigation and collector-degradation balance."""

    h2_pressure_pa: float = 100.0
    gas_temperature_k: float = 300.0
    distance_to_collector_m: float = 0.22
    ion_mean_energy_ev: float = 1000.0     # exponential ion energy distribution
    ion_fraction_of_droplet: float = 0.10  # Sn atoms leaving as fast ions
    vapour_fraction_of_droplet: float = 0.02
    neutral_flow_suppression: float = 2e-3  # H2 counter-flow + diffusion transmission
    collector_solid_angle_sr: float = 5.0
    collector_area_m2: float = 0.40
    magnetic_field_t: float = 0.0          # optional magnetic mitigation
    ion_charge_after_gas: int = 1
    h_dissociation_fraction: float = 1e-3  # n_H / n_H2 near collector
    etch_yield: float = 2e-7               # effective Sn atoms removed per H atom
    coverage_scale_nm: float = 0.5         # delta0 in the coverage-limited etch
    intrinsic_loss_per_gpulse: float = 1e-3  # relative R loss per 1e9 pulses
    stopping: IonStopping = None           # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.stopping is None:
            self.stopping = IonStopping()

    # -- ions --------------------------------------------------------------------
    def ion_range_m(self, energy_ev: np.ndarray | float) -> np.ndarray:
        return self.stopping.range_m(energy_ev, self.h2_pressure_pa, self.gas_temperature_k)

    def ion_transmission(self) -> float:
        """Fraction of ions (exponential energy distribution) reaching the collector:
        those whose range exceeds the distance and (if B > 0) whose Larmor
        diameter exceeds it too."""
        e = np.linspace(1.0, 20.0 * self.ion_mean_energy_ev, 4000)
        f = np.exp(-e / self.ion_mean_energy_ev) / self.ion_mean_energy_ev
        reach = self.ion_range_m(e) > self.distance_to_collector_m
        if self.magnetic_field_t > 0:
            reach &= 2 * larmor_radius_m(e, self.ion_charge_after_gas,
                                         self.magnetic_field_t) > self.distance_to_collector_m
        return float(np.trapezoid(f * reach, e) / np.trapezoid(f, e))

    # -- deposition / cleaning ------------------------------------------------------
    def deposition_rate_nm_h(self, atoms_per_droplet: float, rep_rate_hz: float) -> float:
        """Sn deposition on the collector (nm/h) from ions + vapour."""
        geom = self.collector_solid_angle_sr / (4 * np.pi)
        atoms = atoms_per_droplet * rep_rate_hz * geom * (
            self.ion_fraction_of_droplet * self.ion_transmission()
            + self.vapour_fraction_of_droplet * self.neutral_flow_suppression)
        return atoms * SN_ATOMIC_VOLUME_M3 / self.collector_area_m2 * 1e9 * 3600.0

    def h_radical_flux(self) -> float:
        """Gamma_H = n_H v_bar / 4  [m^-2 s^-1]."""
        n_h = self.h_dissociation_fraction * 2 * h2_number_density(self.h2_pressure_pa,
                                                                   self.gas_temperature_k)
        vbar = np.sqrt(8 * K_BOLTZMANN * self.gas_temperature_k / (np.pi * H_ATOM_MASS_KG))
        return n_h * vbar / 4.0

    def cleaning_rate_nm_h(self) -> float:
        """Maximum (Sn-covered) SnH4 etch rate C = Y Gamma_H Omega_Sn (nm/h)."""
        return self.etch_yield * self.h_radical_flux() * SN_ATOMIC_VOLUME_M3 * 1e9 * 3600.0

    def equilibrium_sn_thickness_nm(self, deposition_nm_h: float) -> float:
        """delta_ss = -delta0 ln(1 - D/C); inf if deposition beats cleaning."""
        c = self.cleaning_rate_nm_h()
        if deposition_nm_h >= c:
            return np.inf
        return float(-self.coverage_scale_nm * np.log1p(-deposition_nm_h / c))

    @staticmethod
    def sn_film_transmission(thickness_nm: float, aoi_deg: float = 0.0) -> float:
        """Double-pass attenuation exp(-2 alpha delta / cos theta)."""
        return float(np.exp(-2 * SN_ALPHA_PER_NM * thickness_nm / np.cos(np.radians(aoi_deg))))

    def reflectance_factor(self, hours: float | np.ndarray, deposition_nm_h: float,
                           rep_rate_hz: float) -> np.ndarray:
        """Relative collector reflectance R(t)/R0 after ``hours`` of operation."""
        t = np.asarray(hours, dtype=float)
        c = self.cleaning_rate_nm_h()
        if deposition_nm_h < c:
            delta = np.full_like(t, self.equilibrium_sn_thickness_nm(deposition_nm_h))
            # approach to steady state on time-scale delta0 / C
            delta = delta * (1 - np.exp(-t * c / self.coverage_scale_nm))
        else:
            delta = (deposition_nm_h - c) * t
        gpulses = rep_rate_hz * t * 3600.0 / 1e9
        return np.exp(-2 * SN_ALPHA_PER_NM * delta) * (1 - self.intrinsic_loss_per_gpulse) ** gpulses

    def collector_lifetime_gpulses(self, deposition_nm_h: float, rep_rate_hz: float,
                                   max_loss_rel: float = 0.2) -> float:
        """Pulses (1e9) until R drops by ``max_loss_rel`` relative to fresh."""
        hours = np.linspace(0.0, 2e5, 200001)
        f = self.reflectance_factor(hours, deposition_nm_h, rep_rate_hz)
        idx = np.argmax(f < 1 - max_loss_rel)
        if f[idx] >= 1 - max_loss_rel:
            return np.inf
        return float(hours[idx] * 3600 * rep_rate_hz / 1e9)


__all__ = ["IonStopping", "DebrisModel", "larmor_radius_m", "h2_number_density"]
