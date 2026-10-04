"""Post-exposure bake (PEB): acid diffusion, acid-quencher neutralisation and
deprotection kinetics.

Reaction-diffusion is reduced to the standard lumped model:

1. Acid and quencher diffuse during the bake. For Fickian diffusion the
   Green's function is Gaussian with sigma = sqrt(2 D t_PEB)::

       h_d = G_{sigma_h} * h ,   q_d = G_{sigma_q} * q

2. Acid-base neutralisation (fast, stoichiometric) leaves::

       h_free = max(h_d - q_d, 0)

3. Catalytic deprotection, d[M]/dt = -k_dp h_free [M] ->::

       m = 1 - exp(-k_dp * h_free * t_PEB) = 1 - exp(-kt * h_free)

   (for MOR, ``m`` is the fraction of condensed / insolubilised sites).

The quencher is sampled per voxel as Poisson(q0 * V) when an rng is given
(chemical shot noise).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .exposure import gaussian_blur
from .materials import ResistMaterial


@dataclass
class BakeState:
    """Arrays after PEB (shape ``(n_layers, ny, nx)``).

    acid_diffused: acid density after diffusion [nm^-3].
    quencher: quencher density after diffusion [nm^-3].
    acid_free: acid remaining after neutralisation [nm^-3].
    deprotection: deprotection fraction m in [0, 1].
    """

    acid_diffused: np.ndarray
    quencher: np.ndarray
    acid_free: np.ndarray
    deprotection: np.ndarray


def deprotection_kinetics(acid_free: np.ndarray, kt_nm3: float) -> np.ndarray:
    """m = 1 - exp(-kt * h_free)."""
    return 1.0 - np.exp(-kt_nm3 * np.asarray(acid_free, dtype=float))


def post_exposure_bake(
    acid: np.ndarray,
    dx_nm: float,
    material: ResistMaterial,
    voxel_volume_nm3: float,
    rng: Optional[np.random.Generator] = None,
    boundary: str = "wrap",
) -> BakeState:
    """Run the PEB model on an acid latent image (see module docstring)."""
    h_d = gaussian_blur(acid, material.diffusion_nm, dx_nm, boundary)
    q0 = material.quencher_density_nm3
    if q0 > 0:
        if rng is None:
            q = np.full(acid.shape, q0)
        else:
            q = rng.poisson(q0 * voxel_volume_nm3, size=acid.shape) / voxel_volume_nm3
        q_d = gaussian_blur(q, material.quencher_diffusion_nm, dx_nm, boundary)
    else:
        q_d = np.zeros_like(h_d)
    h_free = np.maximum(h_d - q_d, 0.0)
    m = deprotection_kinetics(h_free, material.kt_nm3)
    return BakeState(acid_diffused=h_d, quencher=q_d, acid_free=h_free, deprotection=m)
