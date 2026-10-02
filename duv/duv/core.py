"""Shared constants, grids and data containers for the DUV scanner model.

Conventions (every module must follow these):
    * Lengths are in nanometres (nm) unless a name says otherwise (``_mm``, ``_m``).
    * Time in seconds, energy in joules, dose in mJ/cm^2.
    * Lateral image-plane quantities are given at *wafer scale*; the reticle is
      ``REDUCTION`` times larger.
    * Pupil / source coordinates are normalised: sigma = sin(theta)/NA, |sigma| <= 1.
    * 2D arrays are indexed ``[iy, ix]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --- physical constants ------------------------------------------------------
H_PLANCK = 6.62607015e-34  # J s
C_LIGHT = 2.99792458e8  # m/s

# --- ArF scanner nominal parameters (public, textbook-level values) ----------
WAVELENGTH_ARF = 193.368  # nm, vacuum wavelength of ArF line-narrowed laser
N_WATER_193 = 1.4366  # refractive index of ultrapure water at 193 nm, 22 degC
DN_DT_WATER = -1.0e-4  # 1/K, thermo-optic coefficient of water at 193 nm
NA_MAX_IMMERSION = 1.35
REDUCTION = 4.0  # reticle-to-wafer demagnification
FIELD_WIDTH_MM = 26.0  # slit length (x) at wafer
FIELD_HEIGHT_MM = 33.0  # scanned field length (y) at wafer
SLIT_HEIGHT_MM = 8.0  # static exposure slit height in scan direction
WAFER_DIAMETER_MM = 300.0


def photon_energy(wavelength_nm: float = WAVELENGTH_ARF) -> float:
    """Energy of a single photon in joules."""
    return H_PLANCK * C_LIGHT / (wavelength_nm * 1e-9)


@dataclass(frozen=True)
class Grid:
    """Square, periodic simulation grid at wafer scale.

    ``n`` samples with spacing ``pixel`` nm, so the period is ``n * pixel``.
    The aerial image is computed on this grid with FFTs, so the layout is
    treated as periodic with that period.
    """

    n: int = 128
    pixel: float = 4.0  # nm

    @property
    def period(self) -> float:
        return self.n * self.pixel

    @property
    def x(self) -> np.ndarray:
        """1D coordinates centred on zero (nm)."""
        return (np.arange(self.n) - self.n // 2) * self.pixel

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) coordinate meshes, indexed [iy, ix]."""
        return np.meshgrid(self.x, self.x)

    @property
    def freq(self) -> np.ndarray:
        """1D spatial frequencies in cycles/nm, fftshift-ed (zero at centre)."""
        return np.fft.fftshift(np.fft.fftfreq(self.n, d=self.pixel))

    def freq_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        return np.meshgrid(self.freq, self.freq)


@dataclass
class SourceMap:
    """Illumination pupil fill (the 'source') in normalised sigma coordinates.

    ``intensity[iy, ix]`` is the relative intensity at (sigma_x, sigma_y) given
    by ``sigma`` (1D, symmetric, spanning [-1, 1]).  ``polarization`` is one of
    'unpolarized', 'TE' (azimuthal), 'TM' (radial), 'X', 'Y'.
    """

    sigma: np.ndarray
    intensity: np.ndarray
    polarization: str = "unpolarized"

    def points(self, threshold: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (sx, sy, weight) of source points above threshold, weights sum to 1."""
        sx, sy = np.meshgrid(self.sigma, self.sigma)
        mask = self.intensity > threshold * self.intensity.max()
        w = self.intensity[mask]
        return sx[mask], sy[mask], w / w.sum()


@dataclass
class Mask:
    """Complex mask transmission at wafer scale on ``grid`` (indexed [iy, ix])."""

    grid: Grid
    transmission: np.ndarray
    name: str = "mask"


@dataclass
class ImagingSettings:
    """Optical settings shared by illumination, projection and resist."""

    wavelength: float = WAVELENGTH_ARF
    na: float = NA_MAX_IMMERSION
    n_immersion: float = N_WATER_193
    defocus: float = 0.0  # nm, at wafer
    zernikes: dict[int, float] = field(default_factory=dict)  # Fringe index -> waves
    flare: float = 0.0  # fraction of intensity spread as uniform background


@dataclass
class ExposureResult:
    """What a single field exposure produced."""

    aerial_image: np.ndarray  # normalised intensity (clear field = 1)
    dose: float  # mJ/cm^2 delivered
    resist_profile: np.ndarray | None = None  # remaining thickness fraction or binary
    cd: float | None = None  # nm
    info: dict = field(default_factory=dict)
