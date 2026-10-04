"""Vacuum / hydrogen environment of an EUV scanner.

An EUV scanner is not a UHV tool: the optical path runs in a low-pressure
hydrogen atmosphere.  H2 is chosen because it is the most EUV-transparent gas
(sigma ~ 6e-20 cm^2 at 13.5 nm), it thermalises Sn ions in the source and it
forms cleaning H radicals when EUV ionises it.

Typical (public literature) operating points:

* source vessel: ~ 50-150 Pa H2 (gas flow tens of slm) - Sn ion stopping
* illuminator / projection optics box (POB): ~ 1-5 Pa H2 purge
* reticle stage & wafer stage compartments: ~ 1-10 Pa H2, separated from the
  POB by apertures / dynamic gas locks (DGL)

Gas-flow relations used here
----------------------------
* throughput balance  Q = S_eff * (p - p_base),   Q in Pa m^3/s
* series conductance  1/S_eff = 1/S_pump + 1/C
* molecular-flow orifice conductance  C = A * v_mean / 4
* long tube (molecular)   C = (pi/12) v_mean d^3 / L
* pump-down transient  p(t) = p_eq + (p0 - p_eq) exp(-S_eff t / V)
* Knudsen number  Kn = lambda_mfp / d,  lambda = k T / (sqrt(2) pi d_m^2 p)

Dynamic gas lock (DGL)
----------------------
H2 flows out of the POB through the DGL tube toward the wafer with mean
velocity v = Q_H2 / (p A).  A contaminant (resist outgassing) diffusing
upstream obeys the 1-D advection-diffusion equation; its steady-state
concentration decays as c(x) = c0 exp(-v x / D).  Over the lock length L the
back-diffusion suppression is therefore exp(-Pe) with Peclet number
Pe = v L / D.  Because D ~ 1/p and v ~ 1/p, Pe = Q L / (A (D p)) is
independent of the lock pressure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from ..constants import AMU, K_BOLTZMANN

#: molecular masses (amu) of the common gases
MOLECULAR_MASS_AMU: Dict[str, float] = {
    "H2": 2.016, "H": 1.008, "H2O": 18.015, "N2": 28.013, "O2": 31.998,
    "CO": 28.01, "CO2": 44.01, "CH4": 16.04, "CxHy": 100.0, "SnH4": 122.7, "Ar": 39.95,
}
#: kinetic diameters (m) for mean-free-path estimates
KINETIC_DIAMETER_M: Dict[str, float] = {
    "H2": 2.89e-10, "H2O": 2.65e-10, "N2": 3.64e-10, "CH4": 3.8e-10, "CxHy": 6.0e-10,
}
#: standard litre per minute -> Pa m^3/s  (1 slm = 101325 Pa * 1e-3 m^3 / 60 s)
SLM_TO_PA_M3_S = 101325.0 * 1e-3 / 60.0
ROOM_T_K = 295.0


def mean_speed(gas: str = "H2", T_K: float = ROOM_T_K) -> float:
    """Mean thermal speed v = sqrt(8 k T / (pi m))  [m/s]."""
    m = MOLECULAR_MASS_AMU[gas] * AMU
    return math.sqrt(8.0 * K_BOLTZMANN * T_K / (math.pi * m))


def number_density(p_Pa: float, T_K: float = ROOM_T_K) -> float:
    """Ideal gas number density n = p / (k T)  [m^-3]."""
    return p_Pa / (K_BOLTZMANN * T_K)


def impingement_flux(p_Pa: float, gas: str = "H2", T_K: float = ROOM_T_K) -> float:
    """Hertz-Knudsen wall flux Gamma = p / sqrt(2 pi m k T) = n v/4  [m^-2 s^-1]."""
    m = MOLECULAR_MASS_AMU[gas] * AMU
    return p_Pa / math.sqrt(2.0 * math.pi * m * K_BOLTZMANN * T_K)


def mean_free_path(p_Pa: float, gas: str = "H2", T_K: float = ROOM_T_K) -> float:
    """lambda = k T / (sqrt(2) pi d^2 p)  [m]."""
    d = KINETIC_DIAMETER_M.get(gas, 3.5e-10)
    return K_BOLTZMANN * T_K / (math.sqrt(2.0) * math.pi * d ** 2 * p_Pa)


def knudsen_number(p_Pa: float, length_m: float, gas: str = "H2", T_K: float = ROOM_T_K) -> float:
    """Kn = lambda / L."""
    return mean_free_path(p_Pa, gas, T_K) / length_m


def flow_regime(p_Pa: float, length_m: float, gas: str = "H2", T_K: float = ROOM_T_K) -> str:
    """'molecular' (Kn > 0.5), 'transitional' (0.01 < Kn < 0.5) or 'viscous'."""
    kn = knudsen_number(p_Pa, length_m, gas, T_K)
    if kn > 0.5:
        return "molecular"
    if kn > 0.01:
        return "transitional"
    return "viscous"


def orifice_conductance(area_m2: float, gas: str = "H2", T_K: float = ROOM_T_K) -> float:
    """Molecular-flow orifice conductance C = A v / 4  [m^3/s]."""
    return area_m2 * mean_speed(gas, T_K) / 4.0


def tube_conductance(diameter_m: float, length_m: float, gas: str = "H2",
                     T_K: float = ROOM_T_K) -> float:
    """Long round tube, molecular flow: C = (pi/12) v d^3 / L  [m^3/s]."""
    return math.pi / 12.0 * mean_speed(gas, T_K) * diameter_m ** 3 / length_m


def effective_pump_speed(pump_speed_m3_s: float, conductance_m3_s: float) -> float:
    """Series combination 1/S_eff = 1/S + 1/C."""
    return 1.0 / (1.0 / pump_speed_m3_s + 1.0 / conductance_m3_s)


@dataclass
class OutgassingSource:
    """Gas load from a surface: Q = q_specific * area  [Pa m^3/s].

    Typical specific outgassing rates (after bake / long pumping):
    stainless steel ~1e-9..1e-8, elastomers ~1e-5, cable insulation ~1e-6
    Pa m^3 s^-1 m^-2.  ``gas`` labels the species (H2O, CxHy, ...).
    """

    name: str
    q_specific_Pa_m3_s_m2: float
    area_m2: float
    gas: str = "H2O"

    @property
    def Q(self) -> float:
        return self.q_specific_Pa_m3_s_m2 * self.area_m2


def resist_outgassing_Q(power_on_wafer_W: float, molecules_per_photon: float = 1e-3,
                        photon_energy_J: float | None = None, T_K: float = ROOM_T_K) -> float:
    """Gas load of EUV-induced resist outgassing.

    Q = (P / E_ph) * eta * k T     [Pa m^3/s],
    with eta the number of volatile molecules released per absorbed photon
    (CAR resists: 1e-4 .. 1e-2; PAG fragments, protecting groups, solvent).
    """
    from ..constants import PHOTON_ENERGY_J
    e = PHOTON_ENERGY_J if photon_energy_J is None else photon_energy_J
    return power_on_wafer_W / e * molecules_per_photon * K_BOLTZMANN * T_K


@dataclass
class Chamber:
    """A pumped compartment fed by a purge flow plus outgassing loads.

    Steady state per species: p_i = p_base,i + Q_i / S_eff, where the
    effective pump speed includes the connecting conductance.
    """

    name: str
    volume_m3: float
    pump_speed_m3_s: float
    conductance_m3_s: float = math.inf
    purge_gas: str = "H2"
    purge_flow_Pa_m3_s: float = 0.0
    outgassing: List[OutgassingSource] = field(default_factory=list)
    base_pressure_Pa: float = 1e-6
    T_K: float = ROOM_T_K

    @property
    def effective_speed(self) -> float:
        if math.isinf(self.conductance_m3_s):
            return self.pump_speed_m3_s
        return effective_pump_speed(self.pump_speed_m3_s, self.conductance_m3_s)

    def gas_loads(self) -> Dict[str, float]:
        loads: Dict[str, float] = {self.purge_gas: self.purge_flow_Pa_m3_s}
        for src in self.outgassing:
            loads[src.gas] = loads.get(src.gas, 0.0) + src.Q
        return loads

    def partial_pressures(self) -> Dict[str, float]:
        """p_i = Q_i / S_eff (base pressure added to the purge gas)."""
        S = self.effective_speed
        pp = {g: q / S for g, q in self.gas_loads().items()}
        pp[self.purge_gas] = pp.get(self.purge_gas, 0.0) + self.base_pressure_Pa
        return pp

    def total_pressure(self) -> float:
        return sum(self.partial_pressures().values())

    def time_constant(self) -> float:
        """tau = V / S_eff  [s]."""
        return self.volume_m3 / self.effective_speed

    def pressure_transient(self, t_s, p0_Pa: float):
        """Pump-down / fill: p(t) = p_eq + (p0 - p_eq) exp(-t/tau)."""
        import numpy as np
        p_eq = self.total_pressure()
        return p_eq + (p0_Pa - p_eq) * np.exp(-np.asarray(t_s, dtype=float) / self.time_constant())

    @staticmethod
    def flow_for_pressure(p_target_Pa: float, effective_speed_m3_s: float) -> float:
        """Purge flow needed for a target pressure: Q = S_eff p."""
        return p_target_Pa * effective_speed_m3_s


# --- dynamic gas lock -------------------------------------------------------
#: D * p for a heavy organic (~100 amu) in H2, ~0.3 cm^2/s at 1 atm  [Pa m^2/s]
DP_ORGANIC_IN_H2 = 0.3e-4 * 101325.0
#: D * p for H2O in H2 (~0.9 cm^2/s at 1 atm)
DP_H2O_IN_H2 = 0.9e-4 * 101325.0


def diffusion_coefficient(p_Pa: float, Dp_Pa_m2_s: float = DP_ORGANIC_IN_H2,
                          T_K: float = ROOM_T_K, T_ref_K: float = 298.0) -> float:
    """Binary diffusion coefficient D = (D p)_ref / p * (T/T_ref)^1.75  [m^2/s]."""
    return Dp_Pa_m2_s / p_Pa * (T_K / T_ref_K) ** 1.75


@dataclass
class DynamicGasLock:
    """H2 counter-flow lock between POB and wafer stage.

    Parameters: H2 throughput ``Q_H2`` (Pa m^3/s) leaving the POB through a
    channel of cross-section ``area_m2`` and length ``length_m`` at pressure
    ``pressure_Pa``.
    """

    Q_H2_Pa_m3_s: float = 0.15       # ~0.09 slm -> Pe ~ 8
    area_m2: float = 3.0e-4          # e.g. ~ 30 mm x 10 mm opening above the slit
    length_m: float = 0.05
    pressure_Pa: float = 3.0
    Dp_Pa_m2_s: float = DP_ORGANIC_IN_H2
    T_K: float = ROOM_T_K

    @property
    def velocity(self) -> float:
        """Mean H2 velocity v = Q / (p A)  [m/s]."""
        return self.Q_H2_Pa_m3_s / (self.pressure_Pa * self.area_m2)

    @property
    def diffusivity(self) -> float:
        return diffusion_coefficient(self.pressure_Pa, self.Dp_Pa_m2_s, self.T_K)

    @property
    def peclet(self) -> float:
        """Pe = v L / D."""
        return self.velocity * self.length_m / self.diffusivity

    def suppression(self) -> float:
        """Back-diffusion transmission c(L)/c(0) = exp(-Pe)."""
        return dgl_suppression(self.peclet)

    def concentration_profile(self, x_m):
        """c(x)/c0 = exp(-v x / D) along the lock (x from the wafer side)."""
        import numpy as np
        return np.exp(-self.velocity * np.asarray(x_m, dtype=float) / self.diffusivity)


def dgl_suppression(peclet: float) -> float:
    """Transmitted fraction of back-diffusing contaminant through a counter-flow
    channel: exp(-Pe)."""
    return math.exp(-peclet)


def default_chambers() -> Dict[str, Chamber]:
    """Representative NXE-class compartments (public order-of-magnitude values).

    Pump speeds are H2 speeds of turbo pumps (H2 compression is poor, so large
    speeds are needed).  Flows are chosen so the pressures match the
    literature ranges given in the module docstring.
    """
    ch: Dict[str, Chamber] = {}
    S_src = 4.0                                     # m^3/s total pumping on the vessel
    ch["source"] = Chamber("source", volume_m3=1.5, pump_speed_m3_s=S_src,
                           purge_flow_Pa_m3_s=Chamber.flow_for_pressure(100.0, S_src),
                           outgassing=[OutgassingSource("vessel walls", 1e-7, 10.0, "H2O")],
                           T_K=400.0)
    for name, vol, S, p, area in [("illuminator", 1.0, 2.0, 3.0, 8.0),
                                  ("pob", 1.5, 2.0, 3.0, 12.0),
                                  ("reticle_stage", 0.8, 1.5, 5.0, 6.0),
                                  ("wafer_stage", 2.0, 3.0, 5.0, 15.0)]:
        out = [OutgassingSource(f"{name} walls", 5e-8, area, "H2O"),
               OutgassingSource(f"{name} materials", 2e-9, area, "CxHy")]
        ch[name] = Chamber(name, vol, S, purge_flow_Pa_m3_s=p * S, outgassing=out)
    return ch
