"""Top-level scanner environment: chambers, gas path, contamination and heating.

``ScannerEnvironment`` ties together

* the pumped H2 compartments (:mod:`.vacuum`) and the DGL,
* Beer-Lambert gas transmission along source -> IF -> reticle -> wafer
  (:mod:`.gas`),
* the EUV power budget along the mirror train, with absorbed power per mirror
  (:mod:`.thermal`) and reticle / pellicle heating,
* carbon / oxide evolution on every mirror over days-weeks, driven by the
  local hydrocarbon and water partial pressures and the H-radical flux of the
  EUV-induced plasma (:mod:`.contamination`).

All numbers are public order-of-magnitude values for an NXE-class tool.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from ..optics.multilayer import mosi_mirror, reflectivity
from .contamination import (CarbonGrowthModel, ContaminationHistory, HydrogenCleaning,
                            OxidationModel, SECONDS_PER_DAY, evolve_mirror_contamination,
                            mirror_lifetime_s)
from .gas import (PathSegment, euv_plasma_production, h_radical_density, h_radical_flux,
                  path_transmission)
from .thermal import (LumpedMirror, ReticleHeating, absorbed_power,
                      pellicle_equilibrium_temperature, reticle_absorbed_fraction)
from .vacuum import Chamber, DynamicGasLock, default_chambers, resist_outgassing_Q


@lru_cache(maxsize=None)
def _r_mosi() -> float:
    return reflectivity(mosi_mirror())


@dataclass
class MirrorSpec:
    """One reflective element in the train."""

    name: str
    chamber: str
    footprint_cm2: float                   # illuminated area (sets intensity)
    reflectivity: Optional[float] = None   # None -> computed Mo/Si value
    gas_path_before_m: float = 0.0         # gas path length preceding this mirror
    cooling_W_K: float = 2.0

    @property
    def R(self) -> float:
        return _r_mosi() if self.reflectivity is None else self.reflectivity


def default_mirror_train() -> List[MirrorSpec]:
    """Collector, 4 illuminator mirrors (one grazing), reticle, 6 POB mirrors."""
    tr = [MirrorSpec("collector", "source", 3000.0, None, 0.2, 50.0),
          MirrorSpec("FFM", "illuminator", 300.0, None, 1.6, 20.0),
          MirrorSpec("PFM", "illuminator", 300.0, None, 1.0, 20.0),
          MirrorSpec("C1", "illuminator", 200.0, None, 0.6, 5.0),
          MirrorSpec("G", "illuminator", 300.0, 0.80, 0.6, 5.0),
          MirrorSpec("reticle", "reticle_stage", 8.3, None, 0.3, 1.0)]
    for i, area in enumerate([60.0, 150.0, 300.0, 400.0, 250.0, 800.0], start=1):
        tr.append(MirrorSpec(f"M{i}", "pob", area, None, 0.5, 2.0))
    return tr


@dataclass
class PowerBudgetEntry:
    name: str
    incident_W: float
    absorbed_W: float
    intensity_W_cm2: float
    gas_transmission_before: float


@dataclass
class ScannerEnvironment:
    """Integrated vacuum / contamination / thermal state of the scanner."""

    chambers: Dict[str, Chamber] = field(default_factory=default_chambers)
    mirrors: List[MirrorSpec] = field(default_factory=default_mirror_train)
    dgl: DynamicGasLock = field(default_factory=DynamicGasLock)
    source_path_m: float = 1.6             # plasma -> collector -> IF
    wafer_path_m: float = 0.3              # last mirror -> wafer (through the DGL)
    oob_fraction: float = 0.05             # OOB power relative to in-band at IF
    pellicle_transmission: Optional[float] = 0.9
    pellicle_emissivity: float = 0.2
    resist_molecules_per_photon: float = 1e-3
    h_recombination_prob: float = 0.05
    carbon_yield: float = 1e-3
    wafer_dose_duty_cycle: float = 0.6     # fraction of time actually exposing

    # --- gas -------------------------------------------------------------------
    def partial_pressures(self, chamber: str) -> Dict[str, float]:
        return self.chambers[chamber].partial_pressures()

    def gas_segments(self) -> List[PathSegment]:
        """Source vessel segment, a segment preceding each mirror, and the wafer gap."""
        segs = [PathSegment("source->IF", self.source_path_m, self.partial_pressures("source"),
                            self.chambers["source"].T_K)]
        for m in self.mirrors[1:]:
            segs.append(PathSegment(f"->{m.name}", m.gas_path_before_m,
                                    self.partial_pressures(m.chamber), self.chambers[m.chamber].T_K))
        segs.append(PathSegment("->wafer", self.wafer_path_m, self.partial_pressures("wafer_stage")))
        return segs

    def gas_transmission(self) -> float:
        """Total gas transmission plasma -> wafer (excluding mirror losses)."""
        return path_transmission(self.gas_segments())[0]

    # --- power budget ------------------------------------------------------------
    def power_budget(self, power_at_IF_W: float = 250.0) -> List[PowerBudgetEntry]:
        """Propagate in-band power from the IF through the mirror train.

        The collector entry uses the power collected (IF power / (R T_gas)).
        """
        segs = {s.name: s.transmission() for s in self.gas_segments()}
        out: List[PowerBudgetEntry] = []
        col = self.mirrors[0]
        t_src = segs["source->IF"]
        P_col_in = power_at_IF_W / (col.R * t_src)
        out.append(PowerBudgetEntry(col.name, P_col_in,
                                    absorbed_power(P_col_in, col.R, self.oob_fraction * P_col_in),
                                    P_col_in / col.footprint_cm2, t_src))
        P = power_at_IF_W
        P_oob = self.oob_fraction * power_at_IF_W
        for m in self.mirrors[1:]:
            t = segs[f"->{m.name}"]
            P *= t
            Pabs = absorbed_power(P, m.R, P_oob)
            out.append(PowerBudgetEntry(m.name, P, Pabs, P / m.footprint_cm2, t))
            P *= m.R
            P_oob *= 0.5                     # crude OOB suppression per mirror
        return out

    def power_at_wafer(self, power_at_IF_W: float = 250.0) -> float:
        last = self.power_budget(power_at_IF_W)[-1]
        R = self.mirrors[-1].R
        t = PathSegment("w", self.wafer_path_m, self.partial_pressures("wafer_stage")).transmission()
        return last.incident_W * R * t

    # --- thermal -------------------------------------------------------------------
    def heating_state(self, power_at_IF_W: float = 250.0) -> Dict[str, float]:
        """Steady temperatures (C) of mirrors (lumped), reticle rise and pellicle T."""
        res: Dict[str, float] = {}
        budget = self.power_budget(power_at_IF_W)
        for m, b in zip(self.mirrors, budget):
            if m.name == "reticle":
                continue
            res[f"T_{m.name}_C"] = LumpedMirror(conductance_W_K=m.cooling_W_K).steady_state_C(b.absorbed_W)
        ret = next(b for b in budget if b.name == "reticle")
        rh = ReticleHeating(power_on_reticle_W=ret.incident_W,
                            absorbed_fraction=reticle_absorbed_fraction(0.9))
        res["reticle_dT_K"] = rh.delta_T_K
        res["reticle_overlay_nm"] = rh.overlay_at_wafer_nm()
        if self.pellicle_transmission is not None:
            I = ret.incident_W / (self.mirrors[[m.name for m in self.mirrors].index("reticle")].footprint_cm2 * 1e-4)
            res["pellicle_T_C"] = pellicle_equilibrium_temperature(
                I, self.pellicle_emissivity, self.pellicle_transmission) - 273.15
            res["pellicle_double_pass_T"] = self.pellicle_transmission ** 2
        return res

    # --- contamination -------------------------------------------------------------
    def hydrocarbon_pressure(self, chamber: str, power_at_IF_W: float = 250.0) -> float:
        """CxHy partial pressure; the POB also receives resist outgassing that
        survives the DGL (suppression exp(-Pe))."""
        p = self.partial_pressures(chamber).get("CxHy", 0.0)
        if chamber == "pob":
            Q = resist_outgassing_Q(self.power_at_wafer(power_at_IF_W), self.resist_molecules_per_photon)
            p += Q / self.chambers["wafer_stage"].effective_speed * self.dgl.suppression()
        return p

    def h_flux_cm2_s(self, chamber: str, beam_power_W: float, path_m: float) -> float:
        """H radical flux at the optics from the EUV-induced plasma in a chamber."""
        ch = self.chambers[chamber]
        y = euv_plasma_production(beam_power_W, ch.partial_pressures().get("H2", 0.0), path_m, ch.T_K)
        area = 6.0 * ch.volume_m3 ** (2.0 / 3.0)
        n = h_radical_density(y.h_radicals_per_s, ch.volume_m3, area, self.h_recombination_prob, ch.T_K)
        return h_radical_flux(n, ch.T_K) * 1e-4

    def evolve_contamination(self, days: float = 7.0, power_at_IF_W: float = 250.0,
                             n_out: int = 30) -> Dict[str, ContaminationHistory]:
        """Carbon/oxide thickness and reflectivity of every mirror over ``days``."""
        budget = self.power_budget(power_at_IF_W)
        hist: Dict[str, ContaminationHistory] = {}
        for m, b in zip(self.mirrors, budget):
            ch = self.chambers[m.chamber]
            pp = ch.partial_pressures()
            path = self.source_path_m if m.chamber == "source" else max(m.gas_path_before_m, 0.3) * 4
            cleaning = HydrogenCleaning(flux_H_cm2_s=self.h_flux_cm2_s(m.chamber, b.incident_W, path),
                                        yield_C=self.carbon_yield)
            carbon = CarbonGrowthModel(p_hc_Pa=self.hydrocarbon_pressure(m.chamber, power_at_IF_W))
            ox = OxidationModel(p_h2o_Pa=pp.get("H2O", 0.0))
            hist[m.name] = evolve_mirror_contamination(days * SECONDS_PER_DAY, b.intensity_W_cm2,
                                                       carbon, ox, cleaning,
                                                       duty_cycle=self.wafer_dose_duty_cycle, n_out=n_out)
        return hist

    def transmission_loss_vs_time(self, days: float = 7.0, power_at_IF_W: float = 250.0,
                                  n_out: int = 30):
        """(t_days, relative optical-train transmission prod R(t)/R(0))."""
        h = self.evolve_contamination(days, power_at_IF_W, n_out)
        t = next(iter(h.values())).t_s / SECONDS_PER_DAY
        rel = np.ones_like(t)
        for v in h.values():
            rel *= v.reflectivity / v.r0
        return t, rel

    def summary(self, power_at_IF_W: float = 250.0) -> Dict[str, float]:
        s = {f"p_{k}_Pa": v.total_pressure() for k, v in self.chambers.items()}
        s["gas_transmission"] = self.gas_transmission()
        s["dgl_peclet"] = self.dgl.peclet
        s["power_at_wafer_W"] = self.power_at_wafer(power_at_IF_W)
        s.update(self.heating_state(power_at_IF_W))
        return s
