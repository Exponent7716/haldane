"""EUV resist & process model: photon shot noise, acid generation, PEB,
development and stochastic metrology (CD, LER/LWR, LCDU, defects).

Public API
----------
ResistProcess, ResistResult      top-level exposure -> printed pattern
ResistMaterial, CAR, MOR         material presets
measure_lines, measure_cd, measure_contacts, nils,
analytic_ler_3sigma, ler_dose_scaling, monte_carlo_defects,
count_feature_defects            metrology
"""
from .materials import CAR, MOR, ResistMaterial
from .exposure import ExposureState, expose_resist, mean_incident_photons, layer_absorption_fractions
from .bake import BakeState, post_exposure_bake, deprotection_kinetics
from .develop import develop, mack_rate
from .metrics import (ContactMetrics, DefectStats, LineMetrics, analytic_ler_3sigma,
                      count_feature_defects, find_line_edges, ler_dose_scaling,
                      measure_cd, measure_contacts, measure_lines, monte_carlo_defects, nils)
from .process import ResistProcess, ResistResult

__all__ = [
    "CAR", "MOR", "ResistMaterial", "ResistProcess", "ResistResult",
    "ExposureState", "expose_resist", "mean_incident_photons", "layer_absorption_fractions",
    "BakeState", "post_exposure_bake", "deprotection_kinetics", "develop", "mack_rate",
    "LineMetrics", "ContactMetrics", "DefectStats", "measure_lines", "measure_cd",
    "measure_contacts", "find_line_edges", "nils", "analytic_ler_3sigma", "ler_dose_scaling",
    "monte_carlo_defects", "count_feature_defects",
]
