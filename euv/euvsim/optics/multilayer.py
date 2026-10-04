"""Mo/Si multilayer mirror reflectivity via the characteristic (transfer) matrix method.

Every EUV mirror in the scanner (collector, illuminator, mask blank, projection
optics) is a Bragg reflector of ~40 Mo/Si bilayers (period ~6.9 nm, Mo fraction
~0.4) with a Ru capping layer.  Normal-incidence peak reflectance is ~67-70 %
(theoretical ~72 %).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ..constants import RAD, WAVELENGTH_NM, complex_index


@dataclass
class Layer:
    material: str
    thickness_nm: float

    @property
    def index(self) -> complex:
        return complex_index(self.material)


@dataclass
class MultilayerStack:
    """Layers listed from the top (incidence side) down to the substrate."""

    layers: List[Layer]
    substrate: str = "Si"
    ambient: str = "vacuum"
    roughness_nm: float = 0.0   # rms interface roughness (Nevot-Croce/Debye-Waller)

    def total_thickness(self) -> float:
        return sum(l.thickness_nm for l in self.layers)


def mosi_mirror(n_pairs: int = 40, period_nm: float = 6.9, gamma: float = 0.4,
                cap: str | None = "Ru", cap_nm: float = 2.5,
                interdiffusion_nm: float = 0.0, roughness_nm: float = 0.0) -> MultilayerStack:
    """Standard Mo/Si Bragg mirror. ``gamma`` = Mo thickness / period."""
    d_mo = gamma * period_nm
    d_si = period_nm - d_mo
    layers: List[Layer] = []
    if cap:
        layers.append(Layer(cap, cap_nm))
    for _ in range(n_pairs):
        if interdiffusion_nm > 0:
            layers += [Layer("Si", d_si - interdiffusion_nm), Layer("MoSi2", interdiffusion_nm),
                       Layer("Mo", d_mo - interdiffusion_nm), Layer("MoSi2", interdiffusion_nm)]
        else:
            layers += [Layer("Si", d_si), Layer("Mo", d_mo)]
    return MultilayerStack(layers=layers, substrate="Si", roughness_nm=roughness_nm)


def _kz(N: complex, n0: complex, sin_theta0: complex, wl: float) -> complex:
    # longitudinal wavevector in a layer (Snell's law in invariant form)
    return 2 * np.pi / wl * np.sqrt(N ** 2 - (n0 * sin_theta0) ** 2 + 0j)


def reflection_coefficient(stack: MultilayerStack, wavelength_nm: float = WAVELENGTH_NM,
                           angle_deg: float = 0.0, pol: str = "s") -> complex:
    """Complex amplitude reflection coefficient r using the Parratt recursion.

    ``angle_deg`` is the angle of incidence measured from the surface normal.
    ``pol`` is 's' (TE) or 'p' (TM); 'u' (unpolarised) is not allowed here.
    """
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")
    wl = wavelength_nm
    n0 = complex_index(stack.ambient)
    s0 = np.sin(angle_deg * RAD)
    media = [n0] + [l.index for l in stack.layers] + [complex_index(stack.substrate)]
    thick = [0.0] + [l.thickness_nm for l in stack.layers] + [0.0]
    kz = [_kz(N, n0, s0, wl) for N in media]
    sigma = stack.roughness_nm

    def fresnel(i: int) -> complex:
        a, b = kz[i], kz[i + 1]
        if pol == "s":
            r = (a - b) / (a + b)
        else:
            Na2, Nb2 = media[i] ** 2, media[i + 1] ** 2
            r = (b * Na2 - a * Nb2) / (b * Na2 + a * Nb2)
        if sigma > 0:
            r *= np.exp(-2 * a * b * sigma ** 2)
        return r

    # recurse from the substrate upward
    r = 0j
    for i in range(len(media) - 2, -1, -1):
        rij = fresnel(i)
        phase = np.exp(-2j * kz[i + 1] * thick[i + 1])
        r = (rij + r * phase) / (1 + rij * r * phase)
    return complex(r)


def reflectivity(stack: MultilayerStack, wavelength_nm: float = WAVELENGTH_NM,
                 angle_deg: float = 0.0, pol: str = "u") -> float:
    """Intensity reflectance R = |r|^2. ``pol='u'`` averages s and p."""
    if pol == "u":
        return 0.5 * (reflectivity(stack, wavelength_nm, angle_deg, "s")
                      + reflectivity(stack, wavelength_nm, angle_deg, "p"))
    return abs(reflection_coefficient(stack, wavelength_nm, angle_deg, pol)) ** 2


def reflectivity_spectrum(stack: MultilayerStack, wavelengths_nm: Iterable[float],
                          angle_deg: float = 0.0, pol: str = "u") -> np.ndarray:
    return np.array([reflectivity(stack, w, angle_deg, pol) for w in wavelengths_nm])


def reflectivity_vs_angle(stack: MultilayerStack, angles_deg: Iterable[float],
                          wavelength_nm: float = WAVELENGTH_NM, pol: str = "u") -> np.ndarray:
    return np.array([reflectivity(stack, wavelength_nm, a, pol) for a in angles_deg])


def bragg_period(wavelength_nm: float = WAVELENGTH_NM, angle_deg: float = 0.0,
                 n_avg: float = 0.97) -> float:
    """First-order Bragg period d = lambda / (2 sqrt(n^2 - sin^2 theta))."""
    s = np.sin(angle_deg * RAD)
    return wavelength_nm / (2.0 * np.sqrt(n_avg ** 2 - s ** 2))


def tuned_mirror(angle_deg: float, wavelength_nm: float = WAVELENGTH_NM, **kw) -> MultilayerStack:
    """Mo/Si mirror whose period is optimised for peak R at ``angle_deg``."""
    base = bragg_period(wavelength_nm, angle_deg)
    periods = np.linspace(base * 0.97, base * 1.05, 41)
    best = max(periods, key=lambda d: reflectivity(mosi_mirror(period_nm=d, **kw),
                                                   wavelength_nm, angle_deg))
    return mosi_mirror(period_nm=float(best), **kw)


def chain_transmission(reflectances: Sequence[float]) -> float:
    """Throughput of a train of mirrors: product of their reflectances."""
    return float(np.prod(reflectances))


def bandpass_chain(stack: MultilayerStack, n_mirrors: int,
                   wavelengths_nm: np.ndarray, angle_deg: float = 0.0) -> Tuple[np.ndarray, float]:
    """Spectral transmission R(lambda)^n of ``n_mirrors`` identical mirrors and its
    integrated bandwidth (nm)."""
    R = reflectivity_spectrum(stack, wavelengths_nm, angle_deg)
    T = R ** n_mirrors
    return T, float(np.trapezoid(T, wavelengths_nm))
