"""EUV photoresist material parameters.

Two archetypes are provided, both built from public literature values:

* **CAR** - chemically amplified resist (polymer + photo-acid generator, PAG,
  + base quencher). Organic, low EUV absorption: alpha ~ 4-5 /um
  (dominated by O and F cross-sections). Positive tone: exposed regions
  are deprotected and dissolve in TMAH developer.
* **MOR** - metal-oxide resist (Sn-oxo clusters, e.g. Inpria-type). Sn has a
  large 13.5 nm cross-section, giving alpha ~ 15-20 /um. Negative tone:
  exposure cleaves Sn-C bonds and condenses the film so exposed regions
  remain after (organic solvent) development. No acid amplification and very
  short reaction "blur".

Absorption follows Beer-Lambert::

    I(z) = I0 * exp(-alpha * z),   absorbance A = 1 - exp(-alpha * T)

Chemistry is reduced to a generic *reactive-species* picture used by
:mod:`euvsim.resist.exposure` and :mod:`euvsim.resist.bake`:

    acids (or Sn-C cleavage events)  = QY x absorbed photons
    deprotection (or condensation) m = 1 - exp(-kt * h_free)

with ``kt`` the product of the catalytic rate constant and the bake time,
expressed in nm^3 (h_free is a number density in nm^-3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

Tone = Literal["positive", "negative"]


@dataclass(frozen=True)
class ResistMaterial:
    """Physical and chemical parameters of an EUV resist film.

    Attributes
    ----------
    name: label.
    tone: ``"positive"`` (exposed area dissolves) or ``"negative"``.
    absorption_per_um: Beer-Lambert absorption coefficient alpha [1/um].
    thickness_nm: film thickness T [nm].
    density_g_cm3: film mass density [g/cm^3] (informational).
    quantum_yield: reactive species (acids / cleavage events) per absorbed
        photon. Each 91.8 eV photon yields a ~80 eV photo-electron that
        produces a cascade of secondary electrons; 2-4 acids/photon for CAR.
    pag_density_nm3: PAG (or reactive-site) number density [nm^-3]; caps the
        acid yield via saturation h = PAG (1 - exp(-raw/PAG)).
    quencher_density_nm3: base quencher number density [nm^-3].
    se_blur_nm: secondary-electron blur sigma [nm] (Gaussian).
    diffusion_nm: acid reaction-diffusion length sigma during PEB [nm].
    quencher_diffusion_nm: quencher diffusion sigma during PEB [nm].
    kt_nm3: deprotection rate constant x PEB time [nm^3].
    threshold: deprotection level at which the film switches solubility
        (threshold development model).
    n_layers: number of depth voxels used to resolve Beer-Lambert attenuation.
    mack_rmax_nm_s, mack_rmin_nm_s, mack_n, mack_mth: Mack (1987) development
        rate parameters (``mack_mth`` is the threshold *inhibitor* fraction);
        t_dev_s: development time [s]. Defaults are chosen so a 30-35 nm film
        clears in t_dev at an inhibitor fraction ~0.5, i.e. consistent with
        the threshold model at m_th = 0.5.
    """

    name: str
    tone: Tone
    absorption_per_um: float
    thickness_nm: float
    density_g_cm3: float
    quantum_yield: float
    pag_density_nm3: float
    quencher_density_nm3: float
    se_blur_nm: float
    diffusion_nm: float
    quencher_diffusion_nm: float
    kt_nm3: float
    threshold: float = 0.5
    n_layers: int = 4
    mack_rmax_nm_s: float = 100.0
    mack_rmin_nm_s: float = 0.02
    mack_n: float = 15.0
    mack_mth: float = 0.3
    t_dev_s: float = 60.0

    # --- derived optical quantities ---------------------------------------
    @property
    def absorption_per_nm(self) -> float:
        """alpha in 1/nm."""
        return self.absorption_per_um * 1e-3

    @property
    def absorbance(self) -> float:
        """Fraction of incident photons absorbed in the film, A = 1 - exp(-alpha T)."""
        return 1.0 - math.exp(-self.absorption_per_nm * self.thickness_nm)

    @property
    def optical_density(self) -> float:
        """Base-10 optical density OD = alpha T / ln 10."""
        return self.absorption_per_nm * self.thickness_nm / math.log(10.0)

    @property
    def bottom_to_top_ratio(self) -> float:
        """Intensity at the resist bottom relative to the top, exp(-alpha T)."""
        return math.exp(-self.absorption_per_nm * self.thickness_nm)

    @property
    def total_blur_nm(self) -> float:
        """Combined acid blur sigma = sqrt(sigma_SE^2 + sigma_diff^2)."""
        return math.hypot(self.se_blur_nm, self.diffusion_nm)

    def with_(self, **kwargs) -> "ResistMaterial":
        """Return a copy with some parameters replaced."""
        return replace(self, **kwargs)


#: Chemically amplified resist (positive tone). alpha 4.5 /um, 35 nm,
#: ~2.5 acids per absorbed photon, ~6 nm acid diffusion. Dose-to-size for
#: dense 16 nm half-pitch spaces ~ 35-55 mJ/cm^2.
CAR = ResistMaterial(
    name="CAR",
    tone="positive",
    absorption_per_um=4.5,
    thickness_nm=35.0,
    density_g_cm3=1.2,
    quantum_yield=2.5,
    pag_density_nm3=0.30,
    quencher_density_nm3=0.03,
    se_blur_nm=2.0,
    diffusion_nm=6.0,
    quencher_diffusion_nm=3.0,
    kt_nm3=12.0,
    threshold=0.5,
    n_layers=4,
)

#: Metal-oxide (Sn-oxo) resist (negative tone). alpha 18 /um, 30 nm,
#: ~2 Sn-C cleavages per absorbed photon, very short blur (~2.5 nm).
MOR = ResistMaterial(
    name="MOR",
    tone="negative",
    absorption_per_um=18.0,
    thickness_nm=30.0,
    density_g_cm3=2.4,
    quantum_yield=2.0,
    pag_density_nm3=1.0,
    quencher_density_nm3=0.0,
    se_blur_nm=2.5,
    diffusion_nm=1.0,
    quencher_diffusion_nm=0.0,
    kt_nm3=2.5,
    threshold=0.5,
    n_layers=4,
)
