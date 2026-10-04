"""Projection optics: all-reflective 6-mirror (NA 0.33) or 8-mirror anamorphic
(NA 0.55, central obscuration) projection box.

The pupil is described in *wafer-side* spatial frequency (cycles/nm); the pupil
edge is at |f| = NA/lambda.  Wavefront terms (nm of optical path):

* defocus z (nm):   W = z (1 - sqrt(1 - (lambda |f|)^2))   (non-paraxial)
* Fringe Zernikes:  W = sum_j c_j Z_j(rho, theta), c_j in nm, Z1..Z37.
  Fringe Z4 = 2 rho^2 - 1, so c4 is equivalent (paraxially) to a defocus of
  z = 4 c4 / NA^2.

Pupil amplitude: 1 inside the annulus obscuration <= rho <= 1, optionally
multiplied by a multilayer-induced apodization (1 - a rho^2).  Flare is applied
to the aerial image as a power-redistributing long-range scatter kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np

from ..constants import MAG_HIGH_NA, MAG_LOW_NA, NA_HIGH, NA_LOW, WAVELENGTH_NM
from ..optics.multilayer import mosi_mirror, reflectivity

# Fringe Zernike index -> (n, m, 'c'|'s'|'')
FRINGE_NM = {
    1: (0, 0, ""), 2: (1, 1, "c"), 3: (1, 1, "s"), 4: (2, 0, ""), 5: (2, 2, "c"),
    6: (2, 2, "s"), 7: (3, 1, "c"), 8: (3, 1, "s"), 9: (4, 0, ""), 10: (3, 3, "c"),
    11: (3, 3, "s"), 12: (4, 2, "c"), 13: (4, 2, "s"), 14: (5, 1, "c"), 15: (5, 1, "s"),
    16: (6, 0, ""), 17: (4, 4, "c"), 18: (4, 4, "s"), 19: (5, 3, "c"), 20: (5, 3, "s"),
    21: (6, 2, "c"), 22: (6, 2, "s"), 23: (7, 1, "c"), 24: (7, 1, "s"), 25: (8, 0, ""),
    26: (5, 5, "c"), 27: (5, 5, "s"), 28: (6, 4, "c"), 29: (6, 4, "s"), 30: (7, 3, "c"),
    31: (7, 3, "s"), 32: (8, 2, "c"), 33: (8, 2, "s"), 34: (9, 1, "c"), 35: (9, 1, "s"),
    36: (10, 0, ""), 37: (12, 0, ""),
}
ZERNIKE_NAMES = {1: "piston", 2: "tilt x", 3: "tilt y", 4: "defocus", 5: "astig 0/90",
                 6: "astig 45", 7: "coma x", 8: "coma y", 9: "spherical", 10: "trefoil x",
                 11: "trefoil y", 16: "2nd spherical"}


def zernike_radial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rho, dtype=float)
    for s in range((n - m) // 2 + 1):
        c = ((-1) ** s * math.factorial(n - s)
             / (math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)))
        out = out + c * rho ** (n - 2 * s)
    return out


def fringe_zernike(j: int, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Un-normalised Fringe Zernike polynomial Z_j (j = 1..37)."""
    n, m, t = FRINGE_NM[j]
    R = zernike_radial(n, m, rho)
    if t == "c":
        return R * np.cos(m * theta)
    if t == "s":
        return R * np.sin(m * theta)
    return R


@lru_cache(maxsize=8)
def _mirror_R(angle_deg: float = 0.0) -> float:
    return reflectivity(mosi_mirror(), WAVELENGTH_NM, angle_deg)


@dataclass
class ProjectionOptics:
    NA: float = NA_LOW
    magnification: Tuple[float, float] = MAG_LOW_NA
    n_mirrors: int = 6
    obscuration: float = 0.0             # central obscuration radius / pupil radius
    zernikes: Dict[int, float] = field(default_factory=dict)   # Fringe coeffs, nm
    flare: float = 0.0                   # fraction of light scattered (TIS), e.g. 0.03-0.05
    flare_range_nm: float = 1.0e4        # correlation length of the flare PSF (~um-mm)
    apodization: float = 0.0             # amplitude loss at pupil edge (1 - a rho^2)
    mirror_reflectance: float | None = None
    wavelength_nm: float = WAVELENGTH_NM

    @classmethod
    def low_na(cls, **kw) -> "ProjectionOptics":
        """NXE:3x00-class: NA 0.33, 4x isomorphic, 6 mirrors, unobscured."""
        return cls(NA=NA_LOW, magnification=MAG_LOW_NA, n_mirrors=6, obscuration=0.0, **kw)

    @classmethod
    def high_na(cls, **kw) -> "ProjectionOptics":
        """EXE:5000-class: NA 0.55, anamorphic 4x/8x, 8 mirrors, ~20 % obscuration."""
        kw.setdefault("obscuration", 0.2)
        return cls(NA=NA_HIGH, magnification=MAG_HIGH_NA, n_mirrors=8, **kw)

    # ---------------------------------------------------------------- scalars
    @property
    def cutoff_frequency(self) -> float:
        """Coherent cutoff NA/lambda (cycles/nm)."""
        return self.NA / self.wavelength_nm

    def transmission(self) -> float:
        """Projection-box throughput, product of mirror reflectances (~R^n)."""
        R = self.mirror_reflectance if self.mirror_reflectance is not None else _mirror_R(0.0)
        return R ** self.n_mirrors

    def min_pitch(self, sigma_max: float = 1.0) -> float:
        """Smallest pitch passing two diffraction orders: lambda / (NA (1 + sigma))."""
        return self.wavelength_nm / (self.NA * (1 + sigma_max))

    def k1(self, half_pitch_nm: float) -> float:
        """Rayleigh k1 = HP * NA / lambda."""
        return half_pitch_nm * self.NA / self.wavelength_nm

    def resolution(self, k1: float) -> float:
        """Half pitch CD = k1 lambda / NA."""
        return k1 * self.wavelength_nm / self.NA

    def rayleigh_dof(self, k2: float = 1.0) -> float:
        """Rayleigh depth of focus k2 lambda / NA^2 (nm)."""
        return k2 * self.wavelength_nm / self.NA ** 2

    def z4_equivalent_defocus(self, c4_nm: float) -> float:
        """Paraxial defocus (nm) equivalent to a Fringe Z4 coefficient c4 (nm)."""
        return 4.0 * c4_nm / self.NA ** 2

    # ------------------------------------------------------------------ pupil
    def wavefront_nm(self, rho: np.ndarray, theta: np.ndarray) -> np.ndarray:
        W = np.zeros_like(rho, dtype=float)
        for j, c in self.zernikes.items():
            if c:
                W = W + c * fringe_zernike(int(j), rho, theta)
        return W

    def pupil(self, fx: np.ndarray, fy: np.ndarray, defocus_nm: float = 0.0) -> np.ndarray:
        """Complex pupil function P(fx, fy) (wafer-side frequencies, cycles/nm)."""
        fx = np.asarray(fx, dtype=float)
        fy = np.asarray(fy, dtype=float)
        lam = self.wavelength_nm
        fr = np.hypot(fx, fy)
        rho = fr / self.cutoff_frequency
        inside = (rho <= 1.0 + 1e-12) & (rho >= self.obscuration)
        amp = inside.astype(float)
        if self.apodization:
            amp = amp * np.clip(1.0 - self.apodization * rho ** 2, 0.0, None)
        W = np.zeros_like(rho)
        if defocus_nm:
            s2 = np.clip(1.0 - (lam * fr) ** 2, 0.0, None)
            W = W + defocus_nm * (1.0 - np.sqrt(s2))
        if self.zernikes:
            W = W + self.wavefront_nm(np.where(inside, rho, 0.0), np.arctan2(fy, fx))
        return amp * np.exp(1j * 2 * np.pi / lam * W)

    # ------------------------------------------------------------------ flare
    def apply_flare(self, image: np.ndarray, dx_nm: float) -> np.ndarray:
        """Long-range flare: a fraction ``flare`` of the light is redistributed with a
        Lorentzian (PSD-like ~1/f^2 tail) kernel of range ``flare_range_nm``.
        Energy (the image mean, hence clear-field = 1) is conserved."""
        if not self.flare:
            return image
        ny, nx = image.shape
        fx = np.fft.fftfreq(nx, dx_nm)[None, :]
        fy = np.fft.fftfreq(ny, dx_nm)[:, None]
        S = 1.0 / (1.0 + (2 * np.pi * self.flare_range_nm) ** 2 * (fx ** 2 + fy ** 2))
        K = (1.0 - self.flare) + self.flare * S
        return np.real(np.fft.ifft2(np.fft.fft2(image) * K))
