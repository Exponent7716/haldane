"""Illuminator mirror train transmission (IF → reticle, collector excluded).

Public-literature NXE-class illuminator layout (4 reflections):

====  ===================================  ==========================
 #    element                              coating / AOI (typical)
====  ===================================  ==========================
 1    field facet mirror (FFM)             Mo/Si ML, AOI ≈ 5–15°
 2    pupil facet mirror (PFM)             Mo/Si ML, AOI ≈ 5–15°
 3    condenser mirror (normal incidence)  Mo/Si ML, AOI ≈ 10–20°
 4    grazing-incidence (G) mirror         Ru, grazing ≈ 10–15°
====  ===================================  ==========================

Transmission T_ill = Π R_i · η_fill,FFM · η_fill,PFM · η_other.

* Multilayer R: :func:`euvsim.optics.multilayer.tuned_mirror` evaluated at the
  mirror AOI, with interface roughness and Mo/Si interdiffusion to bring the
  ideal ~72 % down to realistic ~67–69 %.
* Grazing-incidence Ru: total external reflection, computed with the Parratt
  code as a single thick (50 nm) Ru layer on Si (effectively Fresnel R of Ru):
  critical grazing angle θc ≈ √(2δ) ≈ 27° for δ = 1 − n = 0.114.
* Facet fill factors: gaps/edges between facets (and FFM coverage of the
  annular beam) lose a few % per facet mirror.

Pupil-shape-dependent losses
----------------------------
* Masking (blades/apertures in the pupil): η = PFR_target / PFR_filled.
* FlexPupil redistribution: η ≈ fraction of field facets that can address the
  target (see :mod:`.flexpupil`), ~1 for designed settings.
* Étendue limit: the illuminator/projection accepts
  G_acc = A_slit,reticle · π NA_reticle² · PFR; if the source étendue G_src
  exceeds G_acc, only η_G ≈ G_acc/G_src is transmitted (étendue cannot be
  reduced by passive optics).  This is why very small PFR (< ~20 %) costs
  light on low-NA tools.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

import numpy as np

from ..optics.multilayer import Layer, MultilayerStack, reflectivity, tuned_mirror


@lru_cache(maxsize=64)
def multilayer_reflectance(aoi_deg: float, roughness_nm: float = 0.5,
                           interdiffusion_nm: float = 0.6) -> float:
    """Peak reflectance of an AOI-tuned Mo/Si mirror (unpolarised)."""
    st = tuned_mirror(aoi_deg, roughness_nm=roughness_nm, interdiffusion_nm=interdiffusion_nm)
    return float(reflectivity(st, angle_deg=aoi_deg))


@lru_cache(maxsize=64)
def grazing_ru_reflectance(grazing_deg: float, ru_nm: float = 50.0, roughness_nm: float = 0.3) -> float:
    """Reflectance of a Ru-coated grazing-incidence mirror (grazing angle from surface)."""
    if not (0 < grazing_deg < 90):
        raise ValueError("grazing angle must be in (0, 90) degrees")
    st = MultilayerStack([Layer("Ru", ru_nm)], substrate="Si", roughness_nm=roughness_nm)
    return float(reflectivity(st, angle_deg=90.0 - grazing_deg))


@dataclass
class MirrorElement:
    name: str
    kind: str                 # 'ml' (Mo/Si multilayer) or 'gi' (grazing Ru)
    angle_deg: float          # AOI from normal for 'ml'; grazing angle for 'gi'
    fill_factor: float = 1.0  # geometric efficiency (facet gaps, edge losses)

    def reflectance(self) -> float:
        if self.kind == "ml":
            return multilayer_reflectance(self.angle_deg)
        if self.kind == "gi":
            return grazing_ru_reflectance(self.angle_deg)
        raise ValueError(f"unknown mirror kind {self.kind!r}")

    def efficiency(self) -> float:
        return self.reflectance() * self.fill_factor


def default_mirror_train() -> List[MirrorElement]:
    return [
        MirrorElement("FFM", "ml", 10.0, fill_factor=0.93),
        MirrorElement("PFM", "ml", 8.0, fill_factor=0.95),
        MirrorElement("condenser NI", "ml", 15.0),
        MirrorElement("G-mirror (Ru)", "gi", 12.0),
    ]


@dataclass
class IlluminatorTransmission:
    """Transmission budget of the illuminator mirror train."""

    mirrors: List[MirrorElement] = field(default_factory=default_mirror_train)
    other_losses: float = 0.97          # residual contamination/scatter/obscurations

    def table(self) -> List[dict]:
        return [{"name": m.name, "kind": m.kind, "angle_deg": m.angle_deg,
                 "reflectance": m.reflectance(), "fill_factor": m.fill_factor,
                 "efficiency": m.efficiency()} for m in self.mirrors]

    def total(self) -> float:
        """T = Π (R_i · fill_i) · η_other."""
        return float(np.prod([m.efficiency() for m in self.mirrors]) * self.other_losses)


# ---------------------------------------------------------------------------
# pupil-shape dependent losses
# ---------------------------------------------------------------------------
def accepted_etendue_mm2sr(field_area_mm2: float, na_reticle: float, pfr: float = 1.0) -> float:
    """G_acc = A_field · π NA² · PFR  (mm² sr; small-angle solid angle π NA²)."""
    return float(field_area_mm2 * np.pi * na_reticle ** 2 * pfr)


def etendue_efficiency(source_etendue_mm2sr: float, accepted_mm2sr: float) -> float:
    """η_G = min(1, G_acc / G_src): light that fits in the accepted phase space."""
    if source_etendue_mm2sr <= 0:
        return 1.0
    return float(min(1.0, accepted_mm2sr / source_etendue_mm2sr))


def pupil_loss(target_pfr: float, method: str = "flexpupil", filled_pfr: float = 0.8,
               flex_efficiency: float = 1.0, source_etendue_mm2sr: float = 0.0,
               field_area_mm2: float = 832.0, na_reticle: float = 0.0825) -> float:
    """Pupil-shape efficiency (≤ 1) for a target PFR.

    ``method='mask'``: η = PFR_target/PFR_filled (blading a filled pupil).
    ``method='flexpupil'``: η = flex_efficiency · η_G(PFR_target), the étendue
    limited redistribution efficiency.
    """
    if method == "mask":
        return float(min(1.0, target_pfr / filled_pfr))
    if method == "flexpupil":
        g = accepted_etendue_mm2sr(field_area_mm2, na_reticle, target_pfr)
        return float(flex_efficiency * etendue_efficiency(source_etendue_mm2sr, g))
    raise ValueError("method must be 'mask' or 'flexpupil'")
