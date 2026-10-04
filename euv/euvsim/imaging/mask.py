"""Reflective EUV mask (reticle): Mo/Si blank, absorber, mask-3D effects, pellicle.

Physics summary (public literature, e.g. Erdmann et al., Burkhardt, van Setten):

* Blank: 40 Mo/Si pairs with a Ru cap, reflectance ~0.65 at the chief-ray angle
  (CRA = 6 deg for NA 0.33, 5.355 deg for NA 0.55), computed with
  :mod:`euvsim.optics.multilayer`.
* Absorber (thin-mask / Kirchhoff): light crosses the absorber twice (down and
  up), so the absorber-region amplitude is ``r_abs = t^2 * r_ML`` with
  ``t^2 = exp(-i 2 pi/lambda (N-1) 2h / cos(CRA))`` (N = n - i k).  For 60 nm
  TaBN this gives ~2-3 % intensity and ~160 deg phase relative to the ML.
  Ni (high-k) allows a thinner absorber; a "low-n" attenuated PSM uses
  n ~ 0.9, k ~ 0.02 to get ~180 deg phase with ~5-15 % reflectance.
* Mask 3-D (M3D) effects modelled simply:
    - shadowing: with oblique incidence at CRA in the plane of incidence
      (scan direction y for NXE/EXE), absorber edges *perpendicular* to that
      plane (i.e. horizontal lines, edges along x) cast a shadow of width
      ``2 h tan(CRA)`` at the mask, i.e. ``2 h tan(CRA)/mag_y`` at wafer scale.
      Horizontal absorber lines therefore print wider (H-V bias), vertical
      lines are essentially unaffected.  The shadowed strip is not black: its
      light crosses the absorber once, so it gets the single-pass amplitude
      t = sqrt(t^2) (attenuated, phase-shifted).
    - the asymmetric shadow shifts the pattern by half the shadow width
      (pattern placement / telecentricity error) and the absorber phase gives a
      pitch-dependent best-focus shift (see
      :func:`euvsim.imaging.aerial.best_focus`).
* Pellicle: thin membrane crossed twice; double-pass intensity transmission
  ``T2 = exp(-2 * 4 pi k_eff d / (lambda cos theta))``.
* Magnification: a wafer feature of size w is ``mag * w`` on the reticle;
  for anamorphic high-NA optics mag = (4, 8) in (x, y).  Patterns in
  ``euvsim`` are on wafer scale, so magnification only enters geometry
  (shadow width, mask-side NA / CRA overlap check).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from ..constants import (CRA_DEG_LOW_NA, MAG_LOW_NA, OPTICAL_CONSTANTS_13P5, RAD,
                         WAVELENGTH_NM)
from ..optics.multilayer import mosi_mirror, reflection_coefficient


# ------------------------------------------------------------------ pellicle
@dataclass
class Pellicle:
    """Thin EUV pellicle membrane mounted a few mm above the reticle.

    ``fill_factor`` (< 1) models porous membranes such as CNT meshes, whose
    effective absorption is the bulk value scaled by the volume fill fraction.
    """

    material: str = "pSi"
    thickness_nm: float = 50.0
    fill_factor: float = 1.0
    angle_deg: float = CRA_DEG_LOW_NA

    def single_pass_transmission(self, wavelength_nm: float = WAVELENGTH_NM) -> float:
        _, k = OPTICAL_CONSTANTS_13P5[self.material]
        path = self.thickness_nm / math.cos(self.angle_deg * RAD)
        return math.exp(-4 * math.pi * k * self.fill_factor * path / wavelength_nm)

    def double_pass_transmission(self, wavelength_nm: float = WAVELENGTH_NM) -> float:
        """Intensity transmission for the reticle round trip (down + up)."""
        return self.single_pass_transmission(wavelength_nm) ** 2


PELLICLE_PRESETS = {
    "pSi": Pellicle("pSi", 50.0),                    # poly-Si core (1st gen), T2 ~ 0.84
    "SiN": Pellicle("SiN", 6.0),                     # thin SiN-based membrane, T2 ~ 0.89
    "CNT": Pellicle("CNT", 40.0, fill_factor=0.15),  # carbon-nanotube mesh, T2 ~ 0.93
}


# ------------------------------------------------------------------ absorbers
ABSORBER_PRESETS = {
    # name: (material key or (n, k), thickness nm)
    "TaBN": ("TaBN", 60.0),
    "Ni": ("Ni", 40.0),
    "lowN_PSM": ((0.90, 0.035), 34.0),   # generic low-n attPSM: ~180 deg, ~10 % R
}


@lru_cache(maxsize=64)
def _ml_r(n_pairs: int, angle_deg: float, pol: str) -> complex:
    return reflection_coefficient(mosi_mirror(n_pairs=n_pairs), WAVELENGTH_NM, angle_deg, pol)


def _shift_periodic(p: np.ndarray, shift_px: float, axis: int) -> np.ndarray:
    """Periodic shift by a fractional number of pixels (linear interpolation)."""
    i = math.floor(shift_px)
    f = shift_px - i
    a = np.roll(p, i, axis=axis)
    b = np.roll(p, i + 1, axis=axis)
    return (1 - f) * a + f * b


@dataclass
class ReflectiveMask:
    """Reflective EUV reticle model (thin mask + simple M3D corrections)."""

    absorber: str = "TaBN"
    absorber_thickness_nm: float = 60.0
    absorber_nk: Optional[Tuple[float, float]] = None   # override (n, k)
    n_pairs: int = 40
    cra_deg: float = CRA_DEG_LOW_NA
    magnification: Tuple[float, float] = MAG_LOW_NA
    incidence_plane: str = "y"          # plane of incidence contains this axis (scan dir.)
    m3d: bool = True                    # apply shadowing
    pellicle: Optional[Pellicle] = None

    @classmethod
    def preset(cls, name: str, **kw) -> "ReflectiveMask":
        mat, h = ABSORBER_PRESETS[name]
        if isinstance(mat, tuple):
            return cls(absorber=name, absorber_thickness_nm=h, absorber_nk=mat, **kw)
        return cls(absorber=mat, absorber_thickness_nm=h, **kw)

    # -- optical constants --------------------------------------------------
    @property
    def nk(self) -> Tuple[float, float]:
        if self.absorber_nk is not None:
            return self.absorber_nk
        return OPTICAL_CONSTANTS_13P5[self.absorber]

    def multilayer_r(self, pol: str = "s") -> complex:
        """Complex amplitude reflectance of the Ru-capped Mo/Si blank at the CRA."""
        return _ml_r(self.n_pairs, float(self.cra_deg), pol)

    def blank_reflectance(self) -> float:
        """Unpolarised intensity reflectance of the blank (~0.65)."""
        return 0.5 * (abs(self.multilayer_r("s")) ** 2 + abs(self.multilayer_r("p")) ** 2)

    def absorber_double_pass(self) -> complex:
        """Complex double-pass absorber transmission relative to vacuum, t^2."""
        n, k = self.nk
        N = complex(n, -k)
        path = 2 * self.absorber_thickness_nm / math.cos(self.cra_deg * RAD)
        return complex(np.exp(-1j * 2 * math.pi / WAVELENGTH_NM * (N - 1) * path))

    def absorber_r(self, pol: str = "s") -> complex:
        """Kirchhoff absorber-region reflection amplitude t^2 * r_ML."""
        return self.absorber_double_pass() * self.multilayer_r(pol)

    def absorber_reflectance_ratio(self) -> float:
        """|r_abs|^2 / |r_ML|^2 (residual reflectance of the absorber, ~2-3 % for TaBN)."""
        return abs(self.absorber_double_pass()) ** 2

    def absorber_phase_deg(self) -> float:
        """Phase of the absorber reflection relative to the multilayer (deg)."""
        return math.degrees(np.angle(self.absorber_double_pass()))

    def effective_reflectance(self) -> float:
        """Blank reflectance including the pellicle double pass."""
        T = self.pellicle.double_pass_transmission() if self.pellicle else 1.0
        return self.blank_reflectance() * T

    # -- geometry / M3D -----------------------------------------------------
    def _axis_mag(self) -> float:
        return self.magnification[1] if self.incidence_plane == "y" else self.magnification[0]

    def shadow_width_mask_nm(self) -> float:
        """Shadow of an absorber edge at the mask: 2 h tan(CRA)."""
        return 2 * self.absorber_thickness_nm * math.tan(self.cra_deg * RAD)

    def shadow_bias_wafer_nm(self) -> float:
        """H-V bias at wafer scale, 2 h tan(CRA)/mag (along the incidence plane)."""
        return self.shadow_width_mask_nm() / self._axis_mag()

    def pattern_shift_wafer_nm(self) -> float:
        """Placement shift caused by the one-sided shadow (half the shadow width)."""
        return 0.5 * self.shadow_bias_wafer_nm()

    def mask_size_nm(self, wafer_nm: float, axis: str = "x") -> float:
        return wafer_nm * (self.magnification[0] if axis == "x" else self.magnification[1])

    def apply_shadowing(self, pattern: np.ndarray, dx_nm: float) -> np.ndarray:
        """Erode reflective regions along the incidence axis by the shadow width."""
        axis = 0 if self.incidence_plane == "y" else 1
        s = self.shadow_bias_wafer_nm() / dx_nm
        return np.minimum(pattern, _shift_periodic(pattern, s, axis))

    def near_field(self, pattern: np.ndarray, dx_nm: float) -> np.ndarray:
        """Complex reflected amplitude normalised to the bare multilayer (=1)."""
        p = np.asarray(pattern, dtype=float)
        t2 = self.absorber_double_pass()
        if not self.m3d:
            return p + (1.0 - p) * t2
        pe = self.apply_shadowing(p, dx_nm)
        # shadow strip: light crosses the absorber corner once (in or out), so its
        # amplitude is ~ the single-pass transmission t = sqrt(t^2) (with phase).
        # The resulting one-sided complex edge makes the object asymmetric, which
        # produces pitch-dependent best-focus shifts and through-focus pattern
        # shifts (telecentricity error), qualitatively like rigorous M3D results.
        return pe + (p - pe) * self.shadow_strip_amplitude() + (1.0 - p) * t2

    def shadow_strip_amplitude(self) -> complex:
        """Amplitude of the shadowed strip: single-pass absorber transmission."""
        return complex(np.sqrt(self.absorber_double_pass()))


def mask_side_na(na: float, magnification: Tuple[float, float]) -> Tuple[float, float]:
    """Object-side NA in x and y: NA / mag."""
    return na / magnification[0], na / magnification[1]


def cra_overlap_ok(na: float, cra_deg: float, mag_y: float) -> bool:
    """True if incident and reflected light cones at the reticle do not overlap,
    i.e. asin(NA/mag_y) < CRA.  NA 0.33 / 4x / 6 deg -> ok; NA 0.55 / 4x would
    fail, which is why high-NA uses 8x in the scan direction."""
    return math.degrees(math.asin(na / mag_y)) < cra_deg
