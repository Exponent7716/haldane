"""Vacuum / H2 environment, gas absorption, contamination and thermal effects.

Modules: :mod:`.vacuum` (chambers, pumping, DGL), :mod:`.gas` (Beer-Lambert
EUV absorption, EUV-induced H2 plasma), :mod:`.contamination` (carbon growth,
oxidation, Sn, H-radical cleaning, reflectivity loss, lifetime),
:mod:`.thermal` (mirror / reticle / pellicle heating, ULE zero crossing,
Zernike decomposition) and :mod:`.environment` (``ScannerEnvironment``).
"""
from .vacuum import (Chamber, DynamicGasLock, OutgassingSource, default_chambers, dgl_suppression,
                     diffusion_coefficient, effective_pump_speed, flow_regime, impingement_flux,
                     knudsen_number, mean_free_path, mean_speed, number_density,
                     orifice_conductance, resist_outgassing_Q, tube_conductance, SLM_TO_PA_M3_S)
from .gas import (EUVPlasmaYield, PathSegment, absorption_length, attenuation_coefficient,
                  cross_section_cm2, euv_plasma_production, gas_transmission, h_radical_density,
                  h_radical_flux, path_transmission)
from .contamination import (CarbonGrowthModel, CollectorSnModel, ContaminationHistory,
                            HydrogenCleaning, OxidationModel, contaminated_stack,
                            critical_thickness, evolve_mirror_contamination, mirror_lifetime_s,
                            overlayer_reflectivity, photon_flux_cm2_s, relative_reflectivity_loss)
from .thermal import (MATERIALS, SILICON, ULE, ZERODUR, LumpedMirror, MirrorHeatingResult,
                      ReticleHeating, SubstrateMaterial, absorbed_power, fit_zernike,
                      illumination_heat_load, linear_profile_displacement, mirror_heating_map,
                      optimal_zero_crossing_C, pellicle_absorbed_flux,
                      pellicle_equilibrium_temperature, pellicle_time_constant,
                      reticle_absorbed_fraction, reticle_overlay_error_nm, slab_steady_profile,
                      slab_transient, surface_displacement, zernike_noll)
from .environment import MirrorSpec, PowerBudgetEntry, ScannerEnvironment, default_mirror_train

__all__ = [n for n in dir() if not n.startswith("_")]
