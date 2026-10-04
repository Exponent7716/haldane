"""Normal-incidence ellipsoidal Mo/Si collector.

Geometry
--------
The collector is a prolate ellipsoid of revolution with the plasma at the
first focus F1 (origin) and the intermediate focus (IF) at F2 = (0, 0, 2c).
The mirror surface lies behind the plasma (z < 0 side and around it); the
drive laser enters through a central hole.  In polar form about F1, with
phi measured from the -z axis (towards the collector vertex):

    rho(phi) = a (1 - e^2) / (1 + e cos phi),    e = c / a,
    vertex distance  a - c,   point P = (rho sin phi, -rho cos phi).

Every ray F1 -> P is reflected to F2; the local angle of incidence is half
the angle F1-P-F2 (zero at the vertex, growing towards the rim).  The
collection solid angle is Omega = 2 pi (cos phi_min - cos phi_max).
Defaults (a - c = 220 mm, 2c = 1.6 m, phi = 10..79 deg) give a ~660 mm
diameter mirror collecting ~5 sr, AOI up to ~25 deg.

Coating
-------
The multilayer period is laterally graded so that each radius is tuned for
its local AOI (``euvsim.optics.multilayer.tuned_mirror``).  The average
reflectance is the emission-weighted mean (isotropic emission ->
dOmega = 2 pi sin phi dphi) of the unpolarised R(AOI) at 13.5 nm.

Etendue
-------
For an (approximately spherical) emitter of diameter d the collected etendue
is G = (pi d^2 / 4) Omega, conserved to the IF where the image has diameter
~ m d with local magnification m = |PF2| / |PF1|.  It must not exceed the
illuminator acceptance (~3-3.5 mm^2 sr for NXE-class scanners).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from ..constants import WAVELENGTH_NM
from ..optics.multilayer import reflectivity, tuned_mirror


@dataclass
class EllipsoidalCollector:
    vertex_distance_m: float = 0.22       # plasma to collector vertex (a - c)
    focal_separation_m: float = 1.60      # plasma to IF (2c)
    phi_min_deg: float = 10.0             # inner edge (laser hole)
    phi_max_deg: float = 79.0             # rim
    n_pairs: int = 40
    roughness_nm: float = 0.25            # rms interface roughness (graded coatings)
    n_zones: int = 24                     # radial zones for averaging

    # -- geometry ----------------------------------------------------------------
    @property
    def c(self) -> float:
        return self.focal_separation_m / 2

    @property
    def a(self) -> float:
        return self.c + self.vertex_distance_m

    @property
    def eccentricity(self) -> float:
        return self.c / self.a

    def rho(self, phi_deg: np.ndarray | float) -> np.ndarray:
        e = self.eccentricity
        return self.a * (1 - e ** 2) / (1 + e * np.cos(np.radians(phi_deg)))

    def point(self, phi_deg: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        """(r, z) of the mirror point hit by the ray at emission angle phi."""
        p = np.radians(phi_deg)
        rho = self.rho(phi_deg)
        return rho * np.sin(p), -rho * np.cos(p)

    def aoi_deg(self, phi_deg: np.ndarray | float) -> np.ndarray:
        """Local angle of incidence = half the angle F1-P-F2."""
        r, z = self.point(phi_deg)
        v1 = np.stack([-r, -z])                     # P -> F1
        v2 = np.stack([-r, self.focal_separation_m - z])   # P -> F2
        cosang = np.sum(v1 * v2, axis=0) / (np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0))
        return 0.5 * np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    def magnification(self, phi_deg: np.ndarray | float) -> np.ndarray:
        """Local imaging magnification |PF2| / |PF1|."""
        r, z = self.point(phi_deg)
        return np.hypot(r, self.focal_separation_m - z) / self.rho(phi_deg)

    @property
    def diameter_m(self) -> float:
        return float(2 * self.point(self.phi_max_deg)[0])

    @property
    def solid_angle_sr(self) -> float:
        return float(2 * np.pi * (np.cos(np.radians(self.phi_min_deg))
                                  - np.cos(np.radians(self.phi_max_deg))))

    @property
    def collection_fraction_2pi(self) -> float:
        """Collected fraction of emission into 2 pi sr (CE reference)."""
        return self.solid_angle_sr / (2 * np.pi)

    def if_half_angles_deg(self) -> tuple[float, float]:
        """(min, max) ray angles at the IF relative to the optical axis."""
        out = []
        for phi in (self.phi_min_deg, self.phi_max_deg):
            r, z = self.point(phi)
            out.append(float(np.degrees(np.arctan2(r, self.focal_separation_m - z))))
        return min(out), max(out)

    @property
    def if_na(self) -> float:
        return float(np.sin(np.radians(self.if_half_angles_deg()[1])))

    # -- coating -------------------------------------------------------------------
    def _zones(self) -> tuple[np.ndarray, np.ndarray]:
        edges = np.linspace(self.phi_min_deg, self.phi_max_deg, self.n_zones + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        w = np.cos(np.radians(edges[:-1])) - np.cos(np.radians(edges[1:]))
        return mid, w / w.sum()

    @cached_property
    def zone_reflectance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(phi_deg, aoi_deg, R) for each radial zone with a locally tuned coating."""
        phi, _ = self._zones()
        aoi = self.aoi_deg(phi)
        R = np.array([reflectivity(tuned_mirror(float(t), n_pairs=self.n_pairs,
                                                roughness_nm=self.roughness_nm),
                                   WAVELENGTH_NM, float(t), "u") for t in aoi])
        return phi, aoi, R

    def average_reflectance(self) -> float:
        """Emission-weighted mean reflectance at 13.5 nm."""
        _, w = self._zones()
        return float(np.sum(w * self.zone_reflectance[2]))

    def spectral_reflectance(self, wavelengths_nm: np.ndarray, n_sample: int = 6) -> np.ndarray:
        """Emission-weighted R(lambda) over the collector (uses a coarse zone subset)."""
        phi, w = self._zones()
        idx = np.linspace(0, len(phi) - 1, n_sample).round().astype(int)
        aoi = self.aoi_deg(phi[idx])
        ww = w[idx] / w[idx].sum()
        out = np.zeros(len(np.atleast_1d(wavelengths_nm)))
        for t, wt in zip(aoi, ww):
            m = tuned_mirror(float(t), n_pairs=self.n_pairs, roughness_nm=self.roughness_nm)
            out += wt * np.array([reflectivity(m, float(l), float(t), "u")
                                  for l in np.atleast_1d(wavelengths_nm)])
        return out

    def etendue_mm2_sr(self, plasma_diameter_m: float) -> float:
        """G = (pi d^2 / 4) * Omega in mm^2 sr."""
        d_mm = plasma_diameter_m * 1e3
        return float(np.pi * d_mm ** 2 / 4 * self.solid_angle_sr)

    def if_spot_diameter_m(self, plasma_diameter_m: float) -> float:
        """Emission-weighted mean image diameter at the IF (m d)."""
        phi, w = self._zones()
        return float(plasma_diameter_m * np.sum(w * self.magnification(phi)))
