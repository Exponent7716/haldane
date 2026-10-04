"""EUV absorption by the background gas and the EUV-induced hydrogen plasma.

Beer-Lambert attenuation
------------------------
    T = exp(-sum_i sigma_i n_i L),   n_i = p_i / (k T_gas)

Photo-absorption cross-sections at 13.5 nm (91.8 eV)
----------------------------------------------------
At 92 eV absorption is dominated by inner/valence photo-ionisation and is
almost additive over atoms.  Per-atom values below are derived from the
Henke/CXRO atomic scattering factors (sigma_a = 2 r_e lambda f2) and
cross-checked against the 13.5 nm optical constants in ``constants.py``
(sigma = 4 pi k / (lambda n_atoms)):

* H2:  ~6e-20 cm^2 per molecule (Samson & Haddad photo-absorption data,
  ~0.06 Mb near 90 eV) -> 1 Pa * 1 m of H2 at 295 K absorbs ~0.15 %.
* C:   ~6e-19 cm^2 per atom  (from k_C = 0.0069, rho = 2.2 g/cc)
* O:   ~2e-18 cm^2 per atom  (from k_SiO2 - k_Si contribution)
* N:   ~1.2e-18 cm^2 per atom (interpolated)

so H2O ~ 2e-18, CH4 ~ 7e-19 cm^2: residual water / hydrocarbons are 10-30x
more absorbing per molecule than H2, but are present at ~1e-5 of the H2
partial pressure.

EUV-induced hydrogen plasma
---------------------------
Each absorbed 92 eV photon ionises H2 (H2 + hv -> H2+ + e-, photoelectron
~76 eV) and the fast electron produces further ionisations; with the mean
energy per ion pair W(H2) ~ 36 eV the total ion yield is E_ph / W ~ 2.5 per
absorbed photon.  H2+ converts quickly to H3+ (H2+ + H2 -> H3+ + H), and
dissociative excitation / recombination produce H radicals: we use a yield of
~ 2 H atoms per ion pair (order-of-magnitude, van de Kerkhof / Beckers et al.
EUV-induced plasma papers).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from ..constants import K_BOLTZMANN, PHOTON_ENERGY_EV, PHOTON_ENERGY_J

#: per-atom photo-absorption cross-sections at 13.5 nm  [cm^2]
ATOMIC_CROSS_SECTION_CM2: Dict[str, float] = {
    "H": 3.0e-20, "C": 6.0e-19, "N": 1.2e-18, "O": 2.0e-18, "Ar": 3.0e-18, "Sn": 6.5e-18,
}
#: molecular formulas (atom counts)
MOLECULE_ATOMS: Dict[str, Dict[str, int]] = {
    "H2": {"H": 2}, "H": {"H": 1}, "H2O": {"H": 2, "O": 1}, "N2": {"N": 2}, "O2": {"O": 2},
    "CO": {"C": 1, "O": 1}, "CO2": {"C": 1, "O": 2}, "CH4": {"C": 1, "H": 4},
    "CxHy": {"C": 7, "H": 8},            # generic heavy hydrocarbon (toluene-like)
    "Ar": {"Ar": 1}, "SnH4": {"Sn": 1, "H": 4},
}

W_VALUE_H2_EV = 36.0           # mean energy per ion pair in H2
H_RADICALS_PER_ION_PAIR = 2.0


def cross_section_cm2(gas: str) -> float:
    """Molecular photo-absorption cross-section at 13.5 nm [cm^2] (additivity rule).

    H2 uses the dedicated molecular value 6e-20 cm^2.
    """
    if gas == "H2":
        return 6.0e-20
    atoms = MOLECULE_ATOMS[gas]
    return sum(ATOMIC_CROSS_SECTION_CM2[a] * n for a, n in atoms.items())


def cross_section_m2(gas: str) -> float:
    return cross_section_cm2(gas) * 1e-4


def attenuation_coefficient(partial_pressures_Pa: Mapping[str, float], T_K: float = 295.0) -> float:
    """mu = sum_i sigma_i p_i / (k T)  [1/m]."""
    return sum(cross_section_m2(g) * p / (K_BOLTZMANN * T_K)
               for g, p in partial_pressures_Pa.items())


def gas_transmission(partial_pressures_Pa: Mapping[str, float] | float, path_m: float | np.ndarray,
                     T_K: float = 295.0, gas: str = "H2"):
    """Beer-Lambert transmission T = exp(-mu L).

    ``partial_pressures_Pa`` may be a dict {gas: p} or a single pressure of ``gas``.
    """
    if not isinstance(partial_pressures_Pa, Mapping):
        partial_pressures_Pa = {gas: float(partial_pressures_Pa)}
    mu = attenuation_coefficient(partial_pressures_Pa, T_K)
    return np.exp(-mu * np.asarray(path_m, dtype=float))


def absorption_length(p_Pa: float, gas: str = "H2", T_K: float = 295.0) -> float:
    """1/e absorption length L_abs = k T / (sigma p)  [m]."""
    return K_BOLTZMANN * T_K / (cross_section_m2(gas) * p_Pa)


@dataclass
class PathSegment:
    """A straight section of the optical path in one gas environment."""

    name: str
    length_m: float
    partial_pressures_Pa: Dict[str, float]
    T_K: float = 295.0

    def transmission(self) -> float:
        return float(gas_transmission(self.partial_pressures_Pa, self.length_m, self.T_K))


def path_transmission(segments: Sequence[PathSegment]) -> Tuple[float, Dict[str, float]]:
    """Total gas transmission (product) and per-segment breakdown."""
    per = {s.name: s.transmission() for s in segments}
    return float(np.prod(list(per.values()))) if per else 1.0, per


# --- EUV-induced plasma --------------------------------------------------------
@dataclass
class EUVPlasmaYield:
    """Ionisation / radical production rates in the illuminated H2 volume."""

    absorbed_photons_per_s: float
    ion_pairs_per_s: float
    h_radicals_per_s: float
    absorbed_power_W: float


def euv_plasma_production(beam_power_W: float, p_H2_Pa: float, path_m: float,
                          T_K: float = 295.0, W_eV: float = W_VALUE_H2_EV,
                          h_per_ion: float = H_RADICALS_PER_ION_PAIR) -> EUVPlasmaYield:
    """Production rates in an EUV beam crossing H2.

    N_abs = (P / E_ph) (1 - exp(-sigma n L))
    ion pairs  = N_abs * E_ph / W         (~2.5 per photon)
    H radicals = ion pairs * h_per_ion
    """
    a = 1.0 - float(gas_transmission(p_H2_Pa, path_m, T_K))
    n_abs = beam_power_W / PHOTON_ENERGY_J * a
    ions = n_abs * PHOTON_ENERGY_EV / W_eV
    return EUVPlasmaYield(n_abs, ions, ions * h_per_ion, beam_power_W * a)


def h_radical_density(production_per_s: float, volume_m3: float, surface_m2: float,
                      recombination_prob: float = 0.05, T_K: float = 295.0,
                      pump_speed_m3_s: float = 0.0) -> float:
    """Steady-state H radical density from a balance of production and losses.

    dN/dt = G - n (gamma v_H A / 4) - n S  = 0
        -> n = G / (gamma v A/4 + S)   [m^-3]
    gamma: wall recombination probability (metals ~0.1-0.2, glass ~1e-3).
    """
    from .vacuum import mean_speed
    loss = recombination_prob * mean_speed("H", T_K) * surface_m2 / 4.0 + pump_speed_m3_s
    return production_per_s / loss


def h_radical_flux(n_H_m3: float, T_K: float = 295.0) -> float:
    """Wall flux Gamma_H = n v_H / 4  [m^-2 s^-1]."""
    from .vacuum import mean_speed
    return n_H_m3 * mean_speed("H", T_K) / 4.0
