"""Tin droplet generator, pre-pulse target shaping and laser-droplet targeting.

Droplet formation (Rayleigh-Plateau)
------------------------------------
Molten Sn (~250 C) is pushed through a few-um nozzle at ~50-100 m/s.  A
liquid jet of diameter d_j is unstable to varicose perturbations with
wavelength > pi d_j; the fastest-growing (Rayleigh) wavelength is
lambda_R = 4.508 d_j, giving the natural break-up frequency
f_R = v / lambda_R (several MHz) and droplets of diameter
D = (1.5 lambda d_j^2)^(1/3) (volume conservation: one wavelength of jet per
drop).  Industrial generators modulate the nozzle (piezo) so that ~100 of
these micro-droplets *coalesce* into one larger droplet per laser pulse, at
f_rep = v / lambda_eff with lambda_eff = v / f_rep (the droplet spacing).

The capillary time tau_c = sqrt(rho R0^3 / sigma) sets the deformation
time-scale of a droplet.

Pre-pulse target shaping
------------------------
The pre-pulse ablates the droplet surface; the recoil pressure propels the
droplet (speed U ~ E_pp^0.6, Kurilovich et al., Phys. Rev. Appl. 2016) and
radially expands it into a thin sheet.  We use a simple
constant-deceleration (capillary-retarded) form of the Villermaux-Bossa
sheet expansion:

    R(t) = R0 + Rdot0 t (1 - t / (2 t_max)),   t <= t_max,  t_max = c tau_c

with initial expansion rate Rdot0 ~ U.  Thickness follows from volume
conservation h = 4 R0^3 / (3 R^2).

Targeting
---------
A Gaussian main-pulse beam (1/e^2 radius w, i.e. sigma_b = w/2) centred at
lateral offset d from a disk target of radius R couples the fraction

    eta(d) = P[|X| < R],  X ~ N_2(d, sigma_b^2)
           = F_ncx2(R^2/sigma_b^2; k=2, lambda=d^2/sigma_b^2)

(the complement of the Marcum Q-function).  The offset d is the quadrature
sum of droplet position jitter (timing jitter x droplet velocity along the
stream, plus transverse jitter) and laser pointing jitter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import ncx2

SN_LIQUID_DENSITY = 6990.0       # kg/m^3 near melting point
SN_SURFACE_TENSION = 0.55        # N/m
SN_MOLAR_MASS = 0.11871          # kg/mol
RAYLEIGH_WAVELENGTH_FACTOR = 4.508   # lambda_opt / d_jet (inviscid)


@dataclass
class DropletGenerator:
    """Sn droplet generator with coalescence and jitter.

    Defaults: 27 um droplets at 70 m/s and 50 kHz (spacing 1.4 mm), timing
    jitter ~ 50 ns (-> ~3.5 um along-stream) and transverse jitter ~ 2 um.
    """

    droplet_diameter_m: float = 27e-6
    velocity_m_s: float = 70.0
    rep_rate_hz: float = 50e3
    timing_jitter_s: float = 50e-9
    transverse_jitter_m: float = 2e-6
    reservoir_kg: float = 1.5          # usable Sn charge per generator refill

    # -- Rayleigh-Plateau ----------------------------------------------------
    @property
    def spacing_m(self) -> float:
        """Droplet spacing lambda_eff = v / f_rep."""
        return self.velocity_m_s / self.rep_rate_hz

    @property
    def jet_diameter_m(self) -> float:
        """Jet diameter from D^3 = 1.5 lambda_eff d_j^2 (one spacing of jet per drop)."""
        return float(np.sqrt(self.droplet_diameter_m ** 3 / (1.5 * self.spacing_m)))

    @property
    def rayleigh_frequency_hz(self) -> float:
        """Natural break-up frequency f_R = v / (4.508 d_j)."""
        return self.velocity_m_s / (RAYLEIGH_WAVELENGTH_FACTOR * self.jet_diameter_m)

    @property
    def coalescence_number(self) -> float:
        """Number of Rayleigh micro-droplets merged into one droplet."""
        return self.rayleigh_frequency_hz / self.rep_rate_hz

    @property
    def droplet_mass_kg(self) -> float:
        return SN_LIQUID_DENSITY * np.pi / 6.0 * self.droplet_diameter_m ** 3

    @property
    def atoms_per_droplet(self) -> float:
        return self.droplet_mass_kg / SN_MOLAR_MASS * 6.02214076e23

    @property
    def mass_flow_kg_s(self) -> float:
        return self.droplet_mass_kg * self.rep_rate_hz

    @property
    def runtime_h(self) -> float:
        """Hours of operation per Sn reservoir fill."""
        return self.reservoir_kg / self.mass_flow_kg_s / 3600.0

    @property
    def capillary_time_s(self) -> float:
        """tau_c = sqrt(rho R0^3 / sigma)."""
        r0 = self.droplet_diameter_m / 2
        return float(np.sqrt(SN_LIQUID_DENSITY * r0 ** 3 / SN_SURFACE_TENSION))

    @property
    def position_sigma_m(self) -> float:
        """Per-axis-averaged droplet position jitter at the plasma site
        (along-stream timing jitter x v combined with transverse jitter)."""
        along = self.timing_jitter_s * self.velocity_m_s
        return float(np.sqrt(0.5 * (along ** 2 + self.transverse_jitter_m ** 2)))


@dataclass
class PrePulseTarget:
    """Pancake/mist target produced by the pre-pulse.

    ``expansion_speed_ref`` is the initial radial expansion speed (m/s) at
    ``prepulse_energy_ref`` (J); the speed scales as E^0.6.  ``t_max_factor``
    sets t_max = factor * tau_c.
    """

    expansion_speed_ref: float = 150.0
    prepulse_energy_ref: float = 5e-3
    energy_exponent: float = 0.6
    t_max_factor: float = 0.8

    def expansion_speed(self, prepulse_energy_j: float) -> float:
        return self.expansion_speed_ref * (prepulse_energy_j / self.prepulse_energy_ref) ** self.energy_exponent

    def radius_m(self, droplet: DropletGenerator, prepulse_energy_j: float, delay_s: float) -> float:
        """Target radius R(t) = R0 + Rdot0 t (1 - t / 2 t_max), saturating at t_max."""
        r0 = droplet.droplet_diameter_m / 2
        t_max = self.t_max_factor * droplet.capillary_time_s
        t = min(delay_s, t_max)
        return r0 + self.expansion_speed(prepulse_energy_j) * t * (1.0 - t / (2.0 * t_max))

    def thickness_m(self, droplet: DropletGenerator, prepulse_energy_j: float, delay_s: float) -> float:
        """Sheet thickness from volume conservation h = 4 R0^3 / (3 R^2)."""
        r0 = droplet.droplet_diameter_m / 2
        r = self.radius_m(droplet, prepulse_energy_j, delay_s)
        return 4.0 * r0 ** 3 / (3.0 * r ** 2)


def coupling_fraction(offset_m: np.ndarray | float, target_radius_m: float,
                      beam_radius_m: float) -> np.ndarray:
    """Fraction of a Gaussian beam (1/e^2 radius w) intercepted by a disk target
    of radius R whose centre is laterally offset by ``offset_m``.

    eta = F_ncx2(R^2 / s^2; 2, d^2 / s^2) with s = w / 2.
    """
    s = beam_radius_m / 2.0
    d = np.asarray(offset_m, dtype=float)
    return ncx2.cdf((target_radius_m / s) ** 2, 2, (d / s) ** 2)


def sample_offsets(n: int, sigma_m: float, rng: np.random.Generator) -> np.ndarray:
    """Radial laser-target offsets for n pulses, 2-D Gaussian with per-axis sigma."""
    xy = sigma_m * rng.standard_normal((n, 2))
    return np.hypot(xy[:, 0], xy[:, 1])


def hit_probability(sigma_m: float, tolerance_m: float) -> float:
    """Probability that the radial offset of a 2-D Gaussian (per-axis sigma)
    lies within ``tolerance_m``: P = 1 - exp(-r^2 / 2 sigma^2) (Rayleigh CDF)."""
    if sigma_m <= 0:
        return 1.0
    return float(1.0 - np.exp(-tolerance_m ** 2 / (2.0 * sigma_m ** 2)))
