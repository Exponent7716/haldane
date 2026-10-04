"""Laser-produced-plasma (LPP) EUV source: CO2 drive laser, Sn droplets, Sn
plasma emission, debris mitigation, ellipsoidal collector and dose control.

Quick start::

    import numpy as np
    from euvsim.source import LPPSource
    src = LPPSource()
    src.inband_power_at_if_w()            # ~260 W
    src.pulse_energy_series(1000, np.random.default_rng(0))
"""
from .collector import EllipsoidalCollector
from .debris import DebrisModel, IonStopping, h2_number_density, larmor_radius_m
from .drive_laser import DriveLaser, LaserPulse
from .droplet import (DropletGenerator, PrePulseTarget, coupling_fraction, hit_probability,
                      sample_offsets)
from .plasma import INBAND_HI_NM, INBAND_LO_NM, SN_UTA_CENTERS_NM, SnPlasma
from .source import DoseController, LPPSource

__all__ = [
    "LPPSource", "DoseController", "DriveLaser", "LaserPulse", "DropletGenerator",
    "PrePulseTarget", "coupling_fraction", "hit_probability", "sample_offsets", "SnPlasma",
    "SN_UTA_CENTERS_NM", "INBAND_LO_NM", "INBAND_HI_NM", "EllipsoidalCollector", "DebrisModel",
    "IonStopping", "h2_number_density", "larmor_radius_m",
]
