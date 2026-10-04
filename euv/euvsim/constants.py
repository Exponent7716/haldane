"""Physical constants and EUV-specific reference values shared by all subsystems.

Units convention used across ``euvsim``:
  * length: nanometres (nm) for optics/imaging, metres (m) for mechanics/vacuum
  * time: seconds, power: watts, energy: joules, dose: mJ/cm^2
  * angles: degrees in public APIs, radians internally
"""
import math

# --- fundamental -----------------------------------------------------------
H_PLANCK = 6.62607015e-34      # J s
C_LIGHT = 2.99792458e8         # m / s
E_CHARGE = 1.602176634e-19     # C (J per eV)
K_BOLTZMANN = 1.380649e-23     # J / K
AMU = 1.66053906660e-27        # kg

# --- EUV operating point -----------------------------------------------------
WAVELENGTH_NM = 13.5                     # central wavelength
INBAND_FWHM_NM = 0.27                    # 2% bandwidth "in-band" (13.365-13.635 nm)
PHOTON_ENERGY_J = H_PLANCK * C_LIGHT / (WAVELENGTH_NM * 1e-9)
PHOTON_ENERGY_EV = PHOTON_ENERGY_J / E_CHARGE   # ~91.8 eV

# --- optical constants at 13.5 nm: complex index n - i k ---------------------
# (CXRO / Henke tabulated values, rounded)
OPTICAL_CONSTANTS_13P5 = {
    "vacuum": (1.0, 0.0),
    "Mo": (0.9238, 0.00644),
    "Si": (0.9990, 0.00183),
    "Ru": (0.8864, 0.01710),
    "MoSi2": (0.9690, 0.00435),   # interdiffusion layer
    "SiO2": (0.9781, 0.01070),
    "TaBN": (0.9500, 0.03100),     # classic absorber
    "TaBO": (0.9600, 0.02600),     # absorber ARC
    "Ni": (0.9480, 0.07270),       # high-k absorber candidate
    "Sn": (0.9300, 0.07200),
    "C": (0.9616, 0.00691),        # carbon contamination
    "H2": (1.0, 0.0),              # gas: use absorption cross-section instead
    "pSi": (0.9990, 0.00183),      # poly-Si pellicle core
    "SiN": (0.9730, 0.00930),
    "CNT": (0.9616, 0.00691),      # carbon nanotube pellicle (graphitic C)
}


def complex_index(material: str) -> complex:
    """Complex refractive index N = n - i k at 13.5 nm (time convention exp(+i w t))."""
    n, k = OPTICAL_CONSTANTS_13P5[material]
    return complex(n, -k)


def photons_per_mj_cm2(area_cm2: float = 1.0) -> float:
    """Number of 13.5 nm photons in a dose of 1 mJ/cm^2 over ``area_cm2``."""
    return 1e-3 * area_cm2 / PHOTON_ENERGY_J


def photons_per_nm2(dose_mj_cm2: float) -> float:
    """Mean photon count per nm^2 for a given dose (1 mJ/cm^2 ~ 0.68 photons/nm^2)."""
    return dose_mj_cm2 * 1e-3 / PHOTON_ENERGY_J * 1e-14


# --- scanner reference values (NXE:3400-class / EXE:5000-class) -------------
NA_LOW = 0.33
NA_HIGH = 0.55
CRA_DEG_LOW_NA = 6.0           # chief ray angle at the reticle
CRA_DEG_HIGH_NA = 5.355
MAG_LOW_NA = (4.0, 4.0)        # (x, y) reduction
MAG_HIGH_NA = (4.0, 8.0)       # anamorphic
SLIT_WIDTH_MM = 26.0           # field width at wafer (x)
SLIT_HEIGHT_MM = 2.0           # illuminated slit height at wafer (y, scan dir.) - approx.
FIELD_SIZE_MM = (26.0, 33.0)   # full field at wafer
WAFER_DIAMETER_MM = 300.0

RAD = math.pi / 180.0
