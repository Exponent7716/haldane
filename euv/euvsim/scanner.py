"""End-to-end EUV scanner: source -> illuminator -> reticle -> POB -> wafer.

Ties the subsystems together:

* **Photon budget**  P_wafer = P_IF · T_ill · R_mask · T_pellicle · T_POB · T_gas
* **Throughput**     scan speed v = P_wafer / (D · W_slit), limited by the wafer
                     stage; WPH from the per-field scan + step timeline.
* **Patterning**     illumination pupil -> Abbe aerial image through the
                     reflective mask and projection optics -> stochastic resist
                     -> printed CD / LER.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .constants import FIELD_SIZE_MM, SLIT_WIDTH_MM
from .environment import ScannerEnvironment
from .illumination import Illuminator
from .imaging import Pellicle, ProjectionOptics, ReflectiveMask, aerial_image, line_space
from .imaging import nils as image_nils
from .resist import CAR, ResistMaterial, ResistProcess
from .source import LPPSource
from .stage import MotionLimits, TimelineParams, matched_machine_overlay_budget, throughput_wph


@dataclass
class EUVScanner:
    """A configurable EUV lithography scanner (defaults: NXE:3400-class, NA 0.33)."""

    source: LPPSource = field(default_factory=LPPSource)
    optics: ProjectionOptics = field(default_factory=ProjectionOptics.low_na)
    mask: ReflectiveMask = field(default_factory=ReflectiveMask)
    pellicle: Optional[Pellicle] = None
    resist: ResistMaterial = field(default_factory=lambda: CAR)
    illumination: str = "dipole"
    illumination_params: dict = field(default_factory=lambda: {"orientation": "x"})
    stage_limits: MotionLimits = field(default_factory=MotionLimits)
    timeline: TimelineParams = field(default_factory=TimelineParams)
    environment: ScannerEnvironment = field(default_factory=ScannerEnvironment)
    slit_height_mm: float = 2.0
    seed: int = 0

    @classmethod
    def high_na(cls, **kw) -> "EUVScanner":
        """EXE:5000-class: NA 0.55, anamorphic 4x/8x, half field."""
        kw.setdefault("optics", ProjectionOptics.high_na())
        return cls(**kw)

    # ---------------------------------------------------------------- photons
    def illuminator(self) -> Illuminator:
        return Illuminator(na=self.optics.NA, mag=tuple(self.optics.magnification),
                           source_etendue_mm2sr=self.source.etendue_mm2_sr(), seed=self.seed)

    def illuminate(self):
        return self.illuminator().illuminate(self.illumination, **self.illumination_params)

    def gas_transmission(self) -> float:
        """H2 transmission IF -> wafer (source->IF absorption is in the source model)."""
        segs = self.environment.gas_segments()
        return float(np.prod([g.transmission() for g in segs if g.name != "source->IF"]))

    def photon_budget(self) -> dict:
        """Power (W) at each plane and the transmission of each stage."""
        p_if = self.source.inband_power_at_if_w()
        ill = self.illuminate()
        t_ill = ill.total_transmission
        r_mask = self.mask.blank_reflectance()
        t_pel = self.pellicle.double_pass_transmission() if self.pellicle else 1.0
        t_pob = self.optics.transmission()
        p_ret = p_if * t_ill
        t_gas = self.gas_transmission()
        p_wafer = p_ret * r_mask * t_pel * t_pob * t_gas
        return {"P_IF_W": p_if, "T_illuminator": t_ill, "P_reticle_W": p_ret,
                "R_mask": r_mask, "T_pellicle": t_pel, "T_POB": t_pob,
                "T_gas": t_gas, "P_wafer_W": p_wafer,
                "IF_to_wafer": p_wafer / p_if}

    # -------------------------------------------------------------- throughput
    def field_size_mm(self) -> tuple:
        fx, fy = FIELD_SIZE_MM
        if self.optics.magnification[1] > self.optics.magnification[0]:
            fy /= 2.0          # anamorphic: half field in scan direction
        return fx, fy

    def throughput(self, dose_mj_cm2: float) -> dict:
        """Wafers per hour at ``dose_mj_cm2`` (source-limited or stage-limited).

        Dose D at a wafer point is P/(W·h) · (h/v) = P/(W·v), so the
        source-limited scan speed is v = P_wafer / (D · W_slit); the wafer
        timeline (scan, step, settle, dual-stage measure side) comes from
        :mod:`euvsim.stage`.
        """
        p_wafer = self.photon_budget()["P_wafer_W"]
        v_src = p_wafer / (dose_mj_cm2 * 10.0 * SLIT_WIDTH_MM * 1e-3)
        tl = throughput_wph(dose_mj_cm2, p_wafer, self.stage_limits, self.timeline,
                            self.field_size_mm(), SLIT_WIDTH_MM, self.slit_height_mm)
        return {"dose_mJ_cm2": dose_mj_cm2, "P_wafer_W": p_wafer, "scan_speed_m_s": tl.scan_speed,
                "limited_by": "source" if v_src < self.stage_limits.v_max else "stage",
                "n_fields": tl.n_fields, "t_field_s": tl.t_field, "t_wafer_s": tl.t_wafer,
                "WPH": tl.wafers_per_hour}

    def overlay_budget_nm(self) -> float:
        """Matched-machine overlay budget (RSS of |mean|+3σ terms, nm)."""
        return matched_machine_overlay_budget().total * 1e9

    def thermal_state(self) -> dict:
        return self.environment.heating_state(self.source.inband_power_at_if_w())

    # -------------------------------------------------------------- patterning
    def print_lines(self, pitch_nm: float, cd_nm: float, dose_mj_cm2: Optional[float] = None,
                    dx_nm: float = 1.0, length_nm: float = 128.0, defocus_nm: float = 0.0,
                    stochastic: bool = True) -> dict:
        """Expose a vertical line/space pattern end-to-end and measure it.

        If ``dose_mj_cm2`` is None the dose-to-size for the target CD is used.
        Returns aerial-image and resist metrics.
        """
        ny = int(round(length_nm / dx_nm))
        pat = line_space(pitch_nm, cd_nm, dx_nm, n_periods=2, other_px=ny, tone="bright")
        src = self.illuminate().source_points
        aerial = aerial_image(pat, dx_nm, src, self.optics, defocus_nm, mask=self.mask)
        prof = aerial[ny // 2]
        proc = ResistProcess(self.resist)
        if dose_mj_cm2 is None:
            dose_mj_cm2 = proc.dose_to_size(aerial, dx_nm, cd_nm)
        rng = np.random.default_rng(self.seed) if stochastic else None
        res = proc.expose(aerial, dx_nm, dose_mj_cm2, rng, metrics="line")
        m = res.metrics
        return {"pitch_nm": pitch_nm, "target_cd_nm": cd_nm, "dose_mJ_cm2": dose_mj_cm2,
                "k1": self.optics.k1(pitch_nm / 2), "image_contrast":
                    float((prof.max() - prof.min()) / (prof.max() + prof.min())),
                "image_nils": float(image_nils(prof, dx_nm, center=int(np.argmax(prof)), tone="bright", cd_nm=cd_nm)),
                "cd_nm": m.cd_mean, "ler_3sigma_nm": m.ler_3sigma, "lwr_3sigma_nm": m.lwr_3sigma,
                "aerial": aerial, "printed": res.printed}

    def report(self, dose_mj_cm2: float = 30.0) -> str:
        b = self.photon_budget()
        t = self.throughput(dose_mj_cm2)
        lines = [f"EUV scanner  NA={self.optics.NA}  mag={tuple(self.optics.magnification)}",
                 f"  source in-band @IF      {b['P_IF_W']:8.1f} W",
                 f"  illuminator T           {b['T_illuminator']:8.3f}  -> reticle {b['P_reticle_W']:.1f} W",
                 f"  mask R / pellicle T     {b['R_mask']:8.3f} / {b['T_pellicle']:.3f}",
                 f"  POB T ({self.optics.n_mirrors} mirrors)      {b['T_POB']:8.3f}",
                 f"  power at wafer          {b['P_wafer_W']:8.2f} W  ({100 * b['IF_to_wafer']:.2f} % of IF)",
                 f"  @ {dose_mj_cm2:.0f} mJ/cm2: scan {t['scan_speed_m_s']:.3f} m/s "
                 f"({t['limited_by']}-limited), {t['n_fields']} fields, {t['WPH']:.0f} wafers/h"]
        th = self.thermal_state()
        lines.append(f"  reticle heating         {th['reticle_dT_K']:8.1f} K  -> {th['reticle_overlay_nm']:.2f} nm overlay (uncorrected)")
        lines.append(f"  matched-machine overlay {self.overlay_budget_nm():8.2f} nm")
        return "\n".join(lines)

