"""Optics contamination under EUV: carbon growth, oxidation, Sn deposition and
hydrogen-radical cleaning, and the resulting reflectivity loss.

Carbon growth (Hollenshead & Klebanoff, JVST B 24, 64 (2006) type model)
-----------------------------------------------------------------------
Hydrocarbons (CxHy) adsorb on the mirror with a Langmuir coverage set by the
impingement flux Gamma = p / sqrt(2 pi m k T), the residence time
tau = tau0 exp(E_d / k T) and photon-induced cracking (cross-section sigma,
which effectively includes secondary electrons from the multilayer):

    dtheta/dt = s Gamma (1-theta)/N0 - theta/tau - sigma Phi theta = 0
    theta = s Gamma / (s Gamma + N0/tau + sigma Phi N0)
    growth rate (atoms/cm^2/s) = n_C sigma Phi N0 theta

so the rate is linear in photon flux Phi and in hydrocarbon pressure at low
values, and saturates at the molecule supply limit n_C s Gamma at high
photon flux (flux-limited regime).  Thickness rate = atoms rate / rho_C.

Oxidation
---------
Water adsorbs and is dissociated by EUV/secondary electrons; the O oxidises
the Ru cap (or Si if uncapped).  The oxide is self-limiting (diffusion
barrier):  dx/dt = k_ox Phi p_H2O exp(-x / x0).  Carbon on top protects
against oxidation, which is why some hydrocarbon is sometimes tolerated.

Hydrogen-radical cleaning
-------------------------
Atomic H etches carbon (C + 4H -> CH4) and tin (Sn + 4H -> SnH4, volatile):
    etch rate (atoms/cm^2/s) = Y * Gamma_H
with chemical erosion yields Y_C ~ 1e-3..1e-2 (EUV/ion-assisted can be
higher) and Y_Sn ~ 1e-5..1e-4 at room temperature.  H also reduces RuOx.
For thin films etching is limited by coverage: the etch term is multiplied
by min(1, d/delta) with delta ~ one monolayer, which gives a finite
steady-state thickness d* = delta * growth/etch when etch > growth.

Reflectivity loss
-----------------
Computed with :mod:`euvsim.optics.multilayer` by prepending a layer
(``Layer("C", d)``, ``Layer("RuO2", d)``, ``Layer("Sn", d)``) on top of a
``mosi_mirror``.  The usual budget is ~1 % relative R loss per mirror.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import AMU, K_BOLTZMANN, OPTICAL_CONSTANTS_13P5, PHOTON_ENERGY_J
from ..optics.multilayer import Layer, MultilayerStack, mosi_mirror, reflectivity

# RuO2 is not in the shared table; estimated from Ru + O cross-sections
# (rho = 6.97 g/cc): n ~ 0.90, k ~ 0.021.  Added non-destructively.
OPTICAL_CONSTANTS_13P5.setdefault("RuO2", (0.900, 0.0210))

#: atomic densities of the contaminant films [atoms / cm^3]
ATOM_DENSITY_CM3: Dict[str, float] = {
    "C": 9.0e22,          # EUV-deposited graphitic/amorphous carbon, ~1.8 g/cc
    "Sn": 3.7e22,         # beta-Sn, 7.3 g/cc
    "RuO2": 3.15e22,      # formula units
    "SiO2": 2.2e22,
}
SURFACE_SITES_CM2 = 1.0e15
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_DAY = 86400.0


def photon_flux_cm2_s(intensity_W_cm2: float) -> float:
    """Phi = I / E_ph  [photons cm^-2 s^-1]  (1 W/cm^2 ~ 6.8e16)."""
    return intensity_W_cm2 / PHOTON_ENERGY_J


def impingement_flux_cm2_s(p_Pa: float, mass_amu: float, T_K: float = 295.0) -> float:
    """Gamma = p / sqrt(2 pi m k T)  [cm^-2 s^-1]."""
    return p_Pa / math.sqrt(2 * math.pi * mass_amu * AMU * K_BOLTZMANN * T_K) * 1e-4


def atoms_to_nm(rate_atoms_cm2: float, material: str) -> float:
    """Convert an areal atom (or formula-unit) rate to a thickness rate in nm."""
    return rate_atoms_cm2 / ATOM_DENSITY_CM3[material] * 1e7


@dataclass
class CarbonGrowthModel:
    """Photon-induced hydrocarbon cracking on an EUV-irradiated surface."""

    p_hc_Pa: float = 1e-7                 # hydrocarbon partial pressure
    mass_amu: float = 100.0
    carbons_per_molecule: float = 2.0     # net C atoms left per cracked molecule
    sigma_crack_cm2: float = 1e-17        # effective incl. secondary electrons
    desorption_energy_eV: float = 0.75
    tau0_s: float = 1e-13
    sticking: float = 1.0
    T_K: float = 295.0
    N0_cm2: float = SURFACE_SITES_CM2

    @property
    def residence_time(self) -> float:
        """tau = tau0 exp(E_d / kT)."""
        return self.tau0_s * math.exp(self.desorption_energy_eV * 1.602176634e-19
                                      / (K_BOLTZMANN * self.T_K))

    def coverage(self, intensity_W_cm2: float) -> float:
        G = self.sticking * impingement_flux_cm2_s(self.p_hc_Pa, self.mass_amu, self.T_K)
        phi = photon_flux_cm2_s(intensity_W_cm2)
        return G / (G + self.N0_cm2 / self.residence_time + self.sigma_crack_cm2 * phi * self.N0_cm2)

    def growth_rate_nm_s(self, intensity_W_cm2: float) -> float:
        """Carbon thickness growth rate [nm/s]."""
        phi = photon_flux_cm2_s(intensity_W_cm2)
        atoms = self.carbons_per_molecule * self.sigma_crack_cm2 * phi * self.N0_cm2 \
            * self.coverage(intensity_W_cm2)
        return atoms_to_nm(atoms, "C")

    def supply_limit_nm_s(self) -> float:
        """High-flux saturation n_C s Gamma  [nm/s]."""
        G = self.sticking * impingement_flux_cm2_s(self.p_hc_Pa, self.mass_amu, self.T_K)
        return atoms_to_nm(self.carbons_per_molecule * G, "C")


@dataclass
class OxidationModel:
    """Self-limiting EUV-induced oxidation of the Ru cap by water.

    dx/dt = k_ox Phi p_H2O exp(-x/x0)   [nm/s], Phi in photons/cm^2/s.
    k_ox chosen so that 1e-5 Pa H2O at 1 W/cm^2 gives ~0.01 nm/h initially.
    """

    p_h2o_Pa: float = 1e-5
    k_ox_nm_per_photon_Pa: float = 4.0e-18   # nm per (photon/cm^2) per Pa
    x0_nm: float = 0.5
    oxide: str = "RuO2"
    pilling_bedworth: float = 2.3     # oxide volume / consumed metal volume

    def rate_nm_s(self, intensity_W_cm2: float, oxide_nm: float = 0.0) -> float:
        phi = photon_flux_cm2_s(intensity_W_cm2)
        return self.k_ox_nm_per_photon_Pa * phi * self.p_h2o_Pa * math.exp(-oxide_nm / self.x0_nm)


@dataclass
class HydrogenCleaning:
    """Atomic-hydrogen etching:  rate = Y * Gamma_H / rho."""

    flux_H_cm2_s: float = 1e15
    yield_C: float = 1e-3
    yield_Sn: float = 5e-5
    yield_oxide_reduction: float = 1e-4
    monolayer_nm: float = 0.1

    def etch_rate_nm_s(self, material: str = "C") -> float:
        y = {"C": self.yield_C, "Sn": self.yield_Sn, "RuO2": self.yield_oxide_reduction,
             "SiO2": 0.0}[material]
        return atoms_to_nm(y * self.flux_H_cm2_s, material)

    def balancing_flux_C(self, growth_nm_s: float) -> float:
        """H flux at which carbon etching equals growth: Gamma_H = g rho / Y."""
        return growth_nm_s * 1e-7 * ATOM_DENSITY_CM3["C"] / self.yield_C


# --- reflectivity with overlayers ------------------------------------------------
def contaminated_stack(base: Optional[MultilayerStack] = None, carbon_nm: float = 0.0,
                       oxide_nm: float = 0.0, sn_nm: float = 0.0, oxide: str = "RuO2",
                       pilling_bedworth: float = 2.3) -> MultilayerStack:
    """Return a copy of ``base`` (default ``mosi_mirror()``) with contamination
    layers prepended: [Sn][C][oxide] + (cap thinned by oxide/PBR) + multilayer."""
    base = base if base is not None else mosi_mirror()
    layers: List[Layer] = [Layer(l.material, l.thickness_nm) for l in base.layers]
    if oxide_nm > 0:
        if layers and layers[0].material in ("Ru", "Si"):
            layers[0].thickness_nm = max(0.0, layers[0].thickness_nm - oxide_nm / pilling_bedworth)
        layers.insert(0, Layer(oxide, oxide_nm))
    if carbon_nm > 0:
        layers.insert(0, Layer("C", carbon_nm))
    if sn_nm > 0:
        layers.insert(0, Layer("Sn", sn_nm))
    return MultilayerStack(layers=layers, substrate=base.substrate, ambient=base.ambient,
                           roughness_nm=base.roughness_nm)


def overlayer_reflectivity(material: str, thickness_nm: float | Sequence[float],
                           base: Optional[MultilayerStack] = None, angle_deg: float = 0.0):
    """R of ``base`` with a ``material`` layer of given thickness on top."""
    kw = {"C": "carbon_nm", "Sn": "sn_nm", "RuO2": "oxide_nm", "SiO2": "oxide_nm"}[material]
    extra = {"oxide": material} if material in ("RuO2", "SiO2") else {}

    def one(t: float) -> float:
        return reflectivity(contaminated_stack(base, **{kw: float(t)}, **extra), angle_deg=angle_deg)

    if np.ndim(thickness_nm) == 0:
        return one(float(thickness_nm))
    return np.array([one(t) for t in thickness_nm])


def relative_reflectivity_loss(material: str, thickness_nm: float,
                               base: Optional[MultilayerStack] = None, angle_deg: float = 0.0) -> float:
    """1 - R(d)/R(0)."""
    r0 = reflectivity(base if base is not None else mosi_mirror(), angle_deg=angle_deg)
    return 1.0 - overlayer_reflectivity(material, thickness_nm, base, angle_deg) / r0


def critical_thickness(material: str = "C", budget: float = 0.01,
                       base: Optional[MultilayerStack] = None, d_max_nm: float = 5.0) -> float:
    """Overlayer thickness that consumes the relative R-loss ``budget`` (bisection)."""
    lo, hi = 0.0, d_max_nm
    if relative_reflectivity_loss(material, hi, base) < budget:
        return math.inf
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if relative_reflectivity_loss(material, mid, base) < budget:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mirror_lifetime_s(net_rate_nm_s: float, material: str = "C", budget: float = 0.01,
                      base: Optional[MultilayerStack] = None) -> float:
    """Time to reach the R-loss budget at a constant net deposition rate."""
    if net_rate_nm_s <= 0:
        return math.inf
    return critical_thickness(material, budget, base) / net_rate_nm_s


# --- time evolution -------------------------------------------------------------
@dataclass
class ContaminationHistory:
    t_s: np.ndarray
    carbon_nm: np.ndarray
    oxide_nm: np.ndarray
    reflectivity: np.ndarray
    r0: float

    @property
    def relative_loss(self) -> np.ndarray:
        return 1.0 - self.reflectivity / self.r0


def evolve_mirror_contamination(duration_s: float, intensity_W_cm2: float,
                                carbon: Optional[CarbonGrowthModel] = None,
                                oxidation: Optional[OxidationModel] = None,
                                cleaning: Optional[HydrogenCleaning] = None,
                                duty_cycle: float = 1.0, n_out: int = 50,
                                base: Optional[MultilayerStack] = None) -> ContaminationHistory:
    """Integrate carbon and oxide thickness over time.

    dC/dt = D g - e_C min(1, C/delta)
    dX/dt = D r_ox(X) (1 - min(1, C/delta)) - e_ox min(1, X/delta)
    (D = exposure duty cycle; carbon shields the cap from oxidation.)
    """
    carbon = carbon or CarbonGrowthModel()
    oxidation = oxidation or OxidationModel(p_h2o_Pa=0.0)
    cleaning = cleaning or HydrogenCleaning(flux_H_cm2_s=0.0)
    g = carbon.growth_rate_nm_s(intensity_W_cm2) * duty_cycle
    eC = cleaning.etch_rate_nm_s("C")
    eX = cleaning.etch_rate_nm_s(oxidation.oxide) if oxidation.oxide == "RuO2" else 0.0
    d = cleaning.monolayer_nm

    def rhs(_t, y):
        c, x = max(y[0], 0.0), max(y[1], 0.0)
        cov = min(1.0, c / d)
        dc = g - eC * cov
        dx = duty_cycle * oxidation.rate_nm_s(intensity_W_cm2, x) * (1 - cov) - eX * min(1.0, x / d)
        return [dc, dx]

    t = np.linspace(0.0, duration_s, n_out)
    sol = solve_ivp(rhs, (0.0, duration_s), [0.0, 0.0], t_eval=t, method="LSODA",
                    rtol=1e-6, atol=1e-9)
    c = np.clip(sol.y[0], 0, None)
    x = np.clip(sol.y[1], 0, None)
    base = base if base is not None else mosi_mirror()
    R = np.array([reflectivity(contaminated_stack(base, ci, xi, oxide=oxidation.oxide))
                  for ci, xi in zip(c, x)])
    return ContaminationHistory(t, c, x, R, reflectivity(base))


# --- Sn on the collector ---------------------------------------------------------
@dataclass
class CollectorSnModel:
    """Sn deposition on the collector vs. SnH4 etching by H radicals.

    The deposition flux is the residual Sn (atoms/cm^2/s) that passes the
    debris mitigation (H2 buffer gas stopping, magnetic/flow mitigation); it
    is conceptually ``source.debris`` output.  E.g. ~1e-4 of ~1.5e19 Sn
    atoms/s escaping mitigation, spread over ~3000 cm^2, gives ~5e11
    atoms/cm^2/s - the same order as the SnH4 etch rate (Y_Sn 5e-5 x
    1e16 H/cm^2/s), so collector life hinges on a near-balance of the two.
    """

    deposition_atoms_cm2_s: float = 5.03e11
    cleaning: HydrogenCleaning = field(default_factory=lambda: HydrogenCleaning(flux_H_cm2_s=1e16))

    def net_rate_nm_s(self) -> float:
        return atoms_to_nm(self.deposition_atoms_cm2_s, "Sn") - self.cleaning.etch_rate_nm_s("Sn")

    def lifetime_s(self, budget: float = 0.1, base: Optional[MultilayerStack] = None) -> float:
        """Time to lose ``budget`` (default 10 %) of collector reflectivity."""
        return mirror_lifetime_s(self.net_rate_nm_s(), "Sn", budget, base)

    def reflectivity_after(self, t_s: float, base: Optional[MultilayerStack] = None) -> float:
        d = max(0.0, self.net_rate_nm_s() * t_s)
        return overlayer_reflectivity("Sn", d, base)
