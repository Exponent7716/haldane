"""Exposure: photon shot noise, Beer-Lambert absorption, secondary-electron blur
and acid generation.

Photon statistics
-----------------
For a normalised aerial image I(x, y) (clear = 1) and dose D [mJ/cm^2] the mean
number of photons incident on a pixel of area dx^2 is::

    <N_inc> = I * D * 1e-3 / E_ph * 1e-14 * dx^2       (~0.68 photons/nm^2 per mJ/cm^2)

Photon arrival is Poisson: N_inc ~ Poisson(<N_inc>). The film is split into
``n_layers`` voxels of thickness dz = T / n_layers; a photon reaching layer
``i`` is absorbed there with probability p = 1 - exp(-alpha dz) (Beer-Lambert),
which is sampled exactly by successive binomial thinning, so that the
absorbed count per voxel is Poisson with mean
``<N_inc> * (exp(-alpha z_i) - exp(-alpha z_{i+1}))``.

Acid generation
---------------
Each absorbed photon releases a ~80 eV photoelectron whose secondary-electron
cascade activates on average ``QY`` PAG molecules; the count is
Poisson(QY * N_abs). The cascade spreads the activation laterally, modelled
as a Gaussian blur with sigma_SE (a few nm). Finite PAG loading gives
saturation::

    h = PAG * (1 - exp(-h_raw / PAG))            [nm^-3]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from ..constants import photons_per_nm2
from .materials import ResistMaterial


@dataclass
class ExposureState:
    """Arrays produced by :func:`expose_resist` (shape ``(n_layers, ny, nx)`` unless noted).

    incident: photons incident on each pixel (2-D, ny x nx).
    absorbed: photons absorbed per voxel.
    acid: acid number density after SE blur and PAG saturation [nm^-3].
    voxel_volume_nm3: dx^2 * dz.
    dz_nm: voxel thickness.
    """

    incident: np.ndarray
    absorbed: np.ndarray
    acid: np.ndarray
    voxel_volume_nm3: float
    dz_nm: float


def mean_incident_photons(aerial: np.ndarray, dx_nm: float, dose_mj_cm2: float) -> np.ndarray:
    """Mean incident photons per pixel, <N> = I * photons_per_nm2(D) * dx^2."""
    return np.asarray(aerial, dtype=float) * photons_per_nm2(dose_mj_cm2) * dx_nm**2


def layer_absorption_fractions(material: ResistMaterial) -> np.ndarray:
    """Fraction of incident photons absorbed in each depth layer.

    f_i = exp(-alpha z_i) - exp(-alpha z_{i+1}),  sum_i f_i = 1 - exp(-alpha T).
    """
    z = np.linspace(0.0, material.thickness_nm, material.n_layers + 1)
    t = np.exp(-material.absorption_per_nm * z)
    return t[:-1] - t[1:]


def gaussian_blur(arr: np.ndarray, sigma_nm: float, dx_nm: float, mode: str = "wrap") -> np.ndarray:
    """Lateral Gaussian blur of the last two axes, sigma in nm.

    Kernel G(r) = exp(-r^2 / 2 sigma^2) / (2 pi sigma^2); MTF = exp(-2 pi^2 sigma^2 f^2).
    """
    if sigma_nm <= 0:
        return np.asarray(arr, dtype=float)
    s = sigma_nm / dx_nm
    sig = (0.0,) * (arr.ndim - 2) + (s, s)
    return ndimage.gaussian_filter(np.asarray(arr, dtype=float), sigma=sig, mode=mode)


def expose_resist(
    aerial: np.ndarray,
    dx_nm: float,
    dose_mj_cm2: float,
    material: ResistMaterial,
    rng: Optional[np.random.Generator] = None,
    boundary: str = "wrap",
) -> ExposureState:
    """Expose the resist film; returns photon counts and the acid latent image.

    If ``rng`` is ``None`` the expectation values are used (no shot noise),
    which gives the deterministic "mean-field" latent image.
    """
    aerial = np.clip(np.asarray(aerial, dtype=float), 0.0, None)
    nl = material.n_layers
    dz = material.thickness_nm / nl
    vvox = dx_nm**2 * dz
    mean_inc = mean_incident_photons(aerial, dx_nm, dose_mj_cm2)
    fr = layer_absorption_fractions(material)

    if rng is None:
        incident = mean_inc
        absorbed = fr[:, None, None] * mean_inc[None]
        raw = material.quantum_yield * absorbed
    else:
        incident = rng.poisson(mean_inc).astype(np.int64)
        p_layer = 1.0 - np.exp(-material.absorption_per_nm * dz)
        remaining = incident.copy()
        absorbed = np.empty((nl,) + aerial.shape, dtype=np.int64)
        for i in range(nl):
            absorbed[i] = rng.binomial(remaining, p_layer)
            remaining -= absorbed[i]
        raw = rng.poisson(material.quantum_yield * absorbed).astype(float)

    raw_density = gaussian_blur(raw, material.se_blur_nm, dx_nm, boundary) / vvox
    pag = material.pag_density_nm3
    acid = pag * (1.0 - np.exp(-raw_density / pag))
    return ExposureState(incident=incident, absorbed=absorbed, acid=acid,
                         voxel_volume_nm3=vvox, dz_nm=dz)
