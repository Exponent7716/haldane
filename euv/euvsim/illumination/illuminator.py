"""Top-level EUV illuminator model: IF power & étendue → reticle illumination.

Étendue bookkeeping
-------------------
The source étendue at IF (plasma size × collection solid angle),

    G_src ≈ A_plasma · Ω_coll ,   Ω_coll ≈ π sin²θ_max   (projected solid angle)

is conserved by the passive illuminator.  The scanner accepts

    G_acc = A_slit,reticle · π NA_ret² · PFR,     NA_ret = NA_wafer / M

(A_slit,reticle = (26·Mx) × (h·My) mm², e.g. 104 × 8 mm² for NXE, NA_ret =
0.0825, so G_acc(PFR = 1) ≈ 17.8 mm² sr).  For G_src ≈ 3–3.5 mm² sr this gives
PFR_min ≈ 0.2: pupils filling less than ~20 % necessarily lose light.

Power at reticle
----------------
    P_ret = P_IF · T_mirrors · η_pupil · (1 − L_UNICOM)

with T_mirrors the mirror-train transmission (:mod:`.transmission`), η_pupil
the pupil-shape efficiency ('etendue' ideal redistribution, 'flexpupil'
facet model, or 'mask' blading) and L_UNICOM the uniformity-correction loss.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from ..constants import NA_LOW, SLIT_HEIGHT_MM, SLIT_WIDTH_MM
from .facets import ArcSlit, FlyEyeIntegrator, Unicom, UnicomResult
from .flexpupil import FlexPupil, FlexPupilResult
from .pupil import DEFAULT_STEP, from_mask, pupil_metrics, shape_mask
from .transmission import (IlluminatorTransmission, accepted_etendue_mm2sr,
                           etendue_efficiency)


def source_etendue_mm2sr(plasma_diameter_mm: float = 0.2, collection_half_angle_deg: float = 80.0,
                         ) -> float:
    """Étendue of a disc-shaped plasma seen within a cone: G = π r² · π sin²θ (mm² sr).

    For a near-isotropic LPP and a large collector (θ ≈ 80°) only the projected
    solid angle π sin²θ ≤ π counts.  ``plasma_diameter_mm`` is the *effective*
    emitting diameter (plasma size blurred by collector aberrations, droplet
    position jitter); a typical IF étendue specification of ~3.3 mm² sr
    corresponds to an effective diameter of ~1.2 mm.
    """
    r = plasma_diameter_mm / 2
    return float(np.pi * r ** 2 * np.pi * np.sin(np.radians(collection_half_angle_deg)) ** 2)


@dataclass
class IlluminationResult:
    setting: str
    params: dict
    source_points: np.ndarray
    pupil_metrics: dict
    mirror_transmission: float
    pupil_efficiency: float
    pupil_efficiency_mask: float
    pupil_efficiency_flex: float
    unicom_loss: float
    total_transmission: float
    power_at_reticle_w: float
    slit_x_mm: np.ndarray
    slit_profile: np.ndarray
    uniformity_raw: float
    uniformity_corrected: float
    source_etendue_mm2sr: float
    accepted_etendue_mm2sr: float
    etendue_ok: bool
    flex: FlexPupilResult | None = field(default=None, repr=False)
    unicom: UnicomResult | None = field(default=None, repr=False)

    def summary(self) -> str:
        return (f"{self.setting} {self.params}: PFR={self.pupil_metrics['pfr']:.3f}, "
                f"T_mirrors={self.mirror_transmission:.3f}, eta_pupil={self.pupil_efficiency:.3f}, "
                f"P_reticle={self.power_at_reticle_w:.1f} W, U={self.uniformity_raw:.4f}"
                f"->{self.uniformity_corrected:.4f}, G_src={self.source_etendue_mm2sr:.2f} "
                f"{'<=' if self.etendue_ok else '>'} G_acc={self.accepted_etendue_mm2sr:.2f} mm2sr")


@dataclass
class Illuminator:
    """EUV illuminator (IF → reticle).

    Parameters
    ----------
    if_power_w : in-band power at the intermediate focus
    source_etendue_mm2sr : étendue of the IF beam (~3.3 mm² sr typical spec)
    na : wafer-side NA of the projection optics
    mag : (Mx, My) reduction (4, 4) low-NA, (4, 8) high-NA anamorphic
    pupil_method : 'etendue' (ideal lossless redistribution, limited only by
        étendue), 'flexpupil' (2-position facet model) or 'mask' (blading)
    """

    if_power_w: float = 250.0
    source_etendue_mm2sr: float = 3.3
    na: float = NA_LOW
    mag: Tuple[float, float] = (4.0, 4.0)
    slit: ArcSlit = field(default_factory=lambda: ArcSlit(SLIT_WIDTH_MM, SLIT_HEIGHT_MM, 30.0))
    n_field_facets: int = 300
    n_pupil_facets: int = 600
    facet_error_rms: float = 0.05
    pupil_method: str = "etendue"
    mask_filled_pfr: float = 0.8
    step: float = DEFAULT_STEP
    use_unicom: bool = True
    seed: int = 0
    transmission: IlluminatorTransmission = field(default_factory=IlluminatorTransmission)

    @property
    def na_reticle(self) -> float:
        """Object-side NA (along x for anamorphic systems: NA/Mx)."""
        return self.na / self.mag[0]

    @property
    def slit_area_reticle_mm2(self) -> float:
        return self.slit.width_mm * self.mag[0] * self.slit.height_mm * self.mag[1]

    def accepted_etendue(self, pfr: float = 1.0) -> float:
        """G_acc = A_slit,ret · π (NA/Mx)(NA/My) · PFR (elliptical pupil for anamorphic)."""
        na_x, na_y = self.na / self.mag[0], self.na / self.mag[1]
        return float(self.slit_area_reticle_mm2 * np.pi * na_x * na_y * pfr)

    def min_pfr_without_loss(self) -> float:
        return float(self.source_etendue_mm2sr / self.accepted_etendue(1.0))

    def illuminate(self, setting: str = "conventional", **params) -> IlluminationResult:
        """Compute pupil, power and uniformity for a named setting.

        ``setting`` ∈ {conventional, annular, dipole, quadrupole, quasar, cquad, leaf};
        ``params`` are passed to the shape (e.g. ``sigma=0.8``, ``sigma_in=0.5,
        sigma_out=0.9, opening_angle_deg=90, orientation='x'``).
        """
        rng = np.random.default_rng(self.seed)
        mask = shape_mask(setting, **params)
        pts = from_mask(mask, self.step)
        met = pupil_metrics(pts, self.step)
        pfr = met["pfr"]

        # mirror train
        t_mir = self.transmission.total()

        # pupil-shape efficiencies
        g_acc = self.accepted_etendue(pfr)
        eta_g = etendue_efficiency(self.source_etendue_mm2sr, g_acc)
        eta_mask = min(1.0, pfr / self.mask_filled_pfr) * etendue_efficiency(
            self.source_etendue_mm2sr, self.accepted_etendue(self.mask_filled_pfr))
        flex = FlexPupil(self.n_pupil_facets, seed=self.seed).configure(mask)
        eta_flex = flex.efficiency
        eta = {"etendue": eta_g, "flexpupil": eta_flex, "mask": eta_mask}.get(self.pupil_method)
        if eta is None:
            raise ValueError("pupil_method must be 'etendue', 'flexpupil' or 'mask'")

        # slit uniformity (fly's eye + optional UNICOM)
        fe = FlyEyeIntegrator(n_facets=self.n_field_facets, facet_error_rms=self.facet_error_rms,
                              slit=self.slit)
        x = fe.x_mm
        prof = fe.slit_profile(rng)
        u_raw = float((prof.max() - prof.min()) / (prof.max() + prof.min()))
        uni = None
        loss = 0.0
        u_cor = u_raw
        if self.use_unicom:
            uni = Unicom(slit_width_mm=self.slit.width_mm, slit_height_mm=self.slit.height_mm).correct(x, prof)
            loss = uni.light_loss
            u_cor = uni.uniformity_after

        total = t_mir * eta * (1 - loss)
        return IlluminationResult(
            setting=setting, params=dict(params), source_points=pts, pupil_metrics=met,
            mirror_transmission=t_mir, pupil_efficiency=float(eta),
            pupil_efficiency_mask=float(eta_mask), pupil_efficiency_flex=float(eta_flex),
            unicom_loss=float(loss), total_transmission=float(total),
            power_at_reticle_w=float(self.if_power_w * total),
            slit_x_mm=x, slit_profile=prof, uniformity_raw=u_raw, uniformity_corrected=float(u_cor),
            source_etendue_mm2sr=self.source_etendue_mm2sr, accepted_etendue_mm2sr=g_acc,
            etendue_ok=bool(self.source_etendue_mm2sr <= g_acc), flex=flex, unicom=uni)


__all__ = ["Illuminator", "IlluminationResult", "source_etendue_mm2sr", "accepted_etendue_mm2sr"]
