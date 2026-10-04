"""Sn laser-produced plasma: temperature, charge state, conversion efficiency
and emission spectrum.

Physics summary
---------------
* Electron temperature.  For an inverse-bremsstrahlung-heated corona in
  steady state (flux-limited), T_e grows as T_e ~ (I lambda^2)^(2/5)
  (e.g. Basko et al. 2015; White et al. 2007).  We normalise so that the
  default CO2 intensity of ~1e11 W/cm^2 gives T_e ~ 35 eV:

      T_e = T_ref (I / I_ref)^0.4 (lambda / lambda_ref)^0.8

* Mean charge state.  In the collisional-radiative Sn plasma the
  population peaks at Sn10+ around 30-40 eV; a simple fit of published CR
  results is  Z_bar ~= 0.9 T_e[eV]^0.7  (gives 10-11 at 35 eV).

* Conversion efficiency CE = in-band (13.5 nm +/- 1 %, i.e. 2 % BW) energy
  emitted into 2 pi sr / laser energy.  Too cold -> too few Sn8+..14+
  ions; too hot -> over-ionisation past Sn14+ and more energy into
  kinetic ions.  CE therefore peaks at an optimum intensity.  We use a
  Gaussian in log10(I):

      CE(I) = CE_max exp[-(log10(I / I_opt))^2 / (2 w^2)]

  with CE_max = 5.5 % and I_opt = 1e11 W/cm^2 (CO2, pre-pulse target;
  Fomenkov 2017, Mizoguchi 2017 report 5-6 %).

* Spectrum.  The 13.5 nm emission is the unresolved transition array (UTA)
  of 4p^6 4d^m - 4p^5 4d^(m+1) + 4d^(m-1) 4f transitions of Sn8+..Sn14+,
  which fortuitously overlap at 13.2-13.9 nm (O'Sullivan et al., J. Phys. B
  2015).  We model each ion's UTA as a Gaussian centred at lambda_q
  (shifting blue with charge state), weighted by a Gaussian charge-state
  distribution around Z_bar.  Opacity is applied to the total line profile
  phi(lambda) with peak optical depth tau0:

      S(lambda) ~ 1 - exp(-tau0 phi(lambda) / phi_max)

  which saturates the line core and broadens the UTA (opacity broadening).
  A broad recombination/satellite continuum (5-40 nm) carries the
  out-of-band EUV, and out-of-band DUV (130-400 nm) and scattered/reflected
  drive-laser IR are carried as fractions of the laser energy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ..constants import INBAND_FWHM_NM, WAVELENGTH_NM

#: Approximate 4d-4f / 4p-4d UTA centroids (nm) per Sn charge state.
SN_UTA_CENTERS_NM: Dict[int, float] = {
    8: 14.05, 9: 13.82, 10: 13.62, 11: 13.50, 12: 13.40, 13: 13.33, 14: 13.28,
}
INBAND_LO_NM = WAVELENGTH_NM - INBAND_FWHM_NM / 2
INBAND_HI_NM = WAVELENGTH_NM + INBAND_FWHM_NM / 2


@dataclass
class SnPlasma:
    """Parametrised Sn LPP emission model (see module docstring)."""

    ce_max: float = 0.055
    intensity_opt_w_cm2: float = 1e11
    ce_log_width: float = 0.45            # decades (1 sigma) of the CE(log I) peak
    te_ref_ev: float = 35.0
    intensity_ref_w_cm2: float = 1e11
    wavelength_ref_um: float = 10.6
    uta_width_nm: float = 0.22            # intrinsic per-ion UTA FWHM
    charge_spread: float = 1.3            # sigma of charge-state distribution
    optical_depth: float = 1.5            # peak in-band optical depth (CO2: thin-ish)
    continuum_fraction: float = 0.55      # 5-40 nm continuum / UTA energy (spectral)
    duv_fraction_of_laser: float = 0.01   # 130-400 nm emission / laser energy
    ir_reflected_fraction: float = 0.10   # drive laser reflected/scattered by plasma
    plasma_diameter_m: float = 300e-6     # emitting region (FWHM) for etendue
    ce_sigma_rel: float = 0.05            # shot-to-shot CE noise (1 sigma)

    # -- plasma state ----------------------------------------------------------
    def electron_temperature_ev(self, intensity_w_cm2: float | np.ndarray,
                                wavelength_um: float = 10.6) -> np.ndarray:
        """T_e = T_ref (I/I_ref)^0.4 (lambda/lambda_ref)^0.8  [eV]."""
        i = np.asarray(intensity_w_cm2, dtype=float)
        return self.te_ref_ev * (i / self.intensity_ref_w_cm2) ** 0.4 * \
            (wavelength_um / self.wavelength_ref_um) ** 0.8

    @staticmethod
    def mean_charge(te_ev: float | np.ndarray) -> np.ndarray:
        """Z_bar ~= 0.9 T_e^0.7, clipped to [0, 50]."""
        return np.clip(0.9 * np.asarray(te_ev, dtype=float) ** 0.7, 0.0, 50.0)

    def conversion_efficiency(self, intensity_w_cm2: float | np.ndarray) -> np.ndarray:
        """CE into 2 % BW / 2 pi sr vs. peak laser intensity (W/cm^2)."""
        x = np.log10(np.asarray(intensity_w_cm2, dtype=float) / self.intensity_opt_w_cm2)
        return self.ce_max * np.exp(-x ** 2 / (2.0 * self.ce_log_width ** 2))

    def charge_state_distribution(self, te_ev: float) -> Dict[int, float]:
        """Normalised Gaussian population of Sn8+..Sn14+ around Z_bar(T_e)."""
        zbar = float(self.mean_charge(te_ev))
        w = {q: np.exp(-(q - zbar) ** 2 / (2 * self.charge_spread ** 2)) for q in SN_UTA_CENTERS_NM}
        s = sum(w.values()) or 1.0
        return {q: v / s for q, v in w.items()}

    # -- spectrum ---------------------------------------------------------------
    def _uta_profile(self, wl: np.ndarray, te_ev: float) -> np.ndarray:
        sig = self.uta_width_nm / 2.3548
        phi = np.zeros_like(wl)
        for q, frac in self.charge_state_distribution(te_ev).items():
            phi += frac * np.exp(-(wl - SN_UTA_CENTERS_NM[q]) ** 2 / (2 * sig ** 2))
        pmax = phi.max() if phi.max() > 0 else 1.0
        return 1.0 - np.exp(-self.optical_depth * phi / pmax)

    @staticmethod
    def _continuum(wl: np.ndarray) -> np.ndarray:
        # broad log-normal recombination/satellite hump peaking ~13 nm
        x = np.log(np.clip(wl, 1e-3, None) / 13.0)
        return np.exp(-x ** 2 / (2 * 0.35 ** 2))

    def euv_shape(self, wavelengths_nm: np.ndarray, te_ev: float = 35.0) -> np.ndarray:
        """Un-normalised EUV spectral shape (5-40 nm region), UTA + continuum."""
        wl = np.asarray(wavelengths_nm, dtype=float)
        grid = np.linspace(5.0, 40.0, 7001)
        uta_g = self._uta_profile(grid, te_ev)
        cont_g = self._continuum(grid)
        k = self.continuum_fraction * np.trapezoid(uta_g, grid) / np.trapezoid(cont_g, grid)
        return self._uta_profile(wl, te_ev) + k * self._continuum(wl)

    def inband_fraction(self, te_ev: float = 35.0) -> float:
        """Fraction of 5-40 nm EUV energy that falls in the 2 % band."""
        grid = np.linspace(5.0, 40.0, 7001)
        s = self.euv_shape(grid, te_ev)
        m = (grid >= INBAND_LO_NM) & (grid <= INBAND_HI_NM)
        return float(np.trapezoid(s[m], grid[m]) / np.trapezoid(s, grid))

    def spectrum(self, wavelengths_nm: np.ndarray, inband_power_w: float,
                 laser_power_w: float = 0.0, te_ev: float = 35.0,
                 laser_wavelength_um: float = 10.6) -> np.ndarray:
        """Spectral power density (W/nm) emitted into 2 pi sr.

        The EUV part is scaled so that its integral over 13.365-13.635 nm equals
        ``inband_power_w``.  DUV (flat 130-400 nm) and drive-laser IR (narrow
        line, 10 nm FWHM, at the laser wavelength) are added as fractions of
        ``laser_power_w``.
        """
        wl = np.asarray(wavelengths_nm, dtype=float)
        grid = np.linspace(INBAND_LO_NM, INBAND_HI_NM, 401)
        norm = np.trapezoid(self.euv_shape(grid, te_ev), grid)
        out = inband_power_w / norm * self.euv_shape(wl, te_ev)
        out = out * ((wl >= 5.0) & (wl <= 40.0))
        duv = (wl >= 130.0) & (wl <= 400.0)
        out = out + duv * self.duv_fraction_of_laser * laser_power_w / 270.0
        lam_ir = laser_wavelength_um * 1e3
        sig_ir = 10.0 / 2.3548
        out = out + self.ir_reflected_fraction * laser_power_w * \
            np.exp(-(wl - lam_ir) ** 2 / (2 * sig_ir ** 2)) / (sig_ir * np.sqrt(2 * np.pi))
        return out

    def peak_wavelength_nm(self, te_ev: float = 35.0) -> float:
        grid = np.linspace(12.0, 15.0, 3001)
        return float(grid[np.argmax(self.euv_shape(grid, te_ev))])
