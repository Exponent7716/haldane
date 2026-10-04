"""Fly's-eye integrator (field & pupil facet mirrors), arc slit and UNICOM.

Fly's-eye (Koehler) integration
-------------------------------
The collector forms an image of the plasma at the intermediate focus (IF);
the diverging beam behind IF lands on the *field facet mirror* (FFM), whose
N arc-shaped facets each cut out a small piece of the (annular, non-uniform)
far-field beam.  Each field facet is imaged by its pupil facet (and the
condenser mirrors) onto the reticle *slit*, so all N facet images are
superimposed in the slit:

    I_slit(x) = Σ_i P_i · f_i(x)

with f_i the (normalised) intensity profile of facet i across the slit
coordinate x.  Smooth large-scale beam variations become only small gradients
*within* each facet (and partly cancel by point-symmetric facet placement),
while random facet-to-facet errors e_i(x) (coating, figure, contamination)
average down:  σ_rel(I_slit) ≈ σ_rel(e) / √N.

Arc slit
--------
The projection optics of an off-axis all-reflective EUV system has a
ring-shaped well-corrected field, so the slit is an arc of radius
R (tens of mm at wafer level) and full width 26 mm (x), height h (scan
direction y, ~1–2 mm at wafer).  The dose at a wafer point is the scan
integral of the slit intensity along y: D(x) ∝ ∫ I(x, y) dy ≈ I(x)·h(x).

UNICOM-type uniformity correction
---------------------------------
A row of fingers (pitch ~4 mm at wafer level) at the slit edge can be
inserted into the beam by δ_j, locally reducing the effective slit height:

    h_eff(x) = h · [1 − Σ_j (δ_j / h) · g(x − x_j)]

with g a blurred finger footprint (the fingers sit slightly out of focus).
The insertions are found by bounded linear least squares so that
I(x)·h_eff(x) is as flat as possible; the price is a small dose loss.
Uniformity metric: U = (max − min)/(max + min) of the scan-integrated dose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import lsq_linear

from ..constants import SLIT_HEIGHT_MM, SLIT_WIDTH_MM


def uniformity(profile: np.ndarray) -> float:
    """Slit non-uniformity U = (max − min)/(max + min) (0 = perfectly flat)."""
    p = np.asarray(profile, float)
    return float((p.max() - p.min()) / (p.max() + p.min()))


# ---------------------------------------------------------------------------
# arc-shaped slit
# ---------------------------------------------------------------------------
@dataclass
class ArcSlit:
    """Arc (ring-field) slit geometry, by default at wafer level (mm).

    The slit centre line is the circle of radius ``radius_mm`` centred at
    (0, −R + y0) — i.e. the arc bulges toward +y with its apex at y = 0 —
    and spans |x| ≤ width/2.  The slit has constant height (along y) h.
    """

    width_mm: float = SLIT_WIDTH_MM
    height_mm: float = SLIT_HEIGHT_MM
    radius_mm: float = 30.0

    def __post_init__(self) -> None:
        if self.radius_mm <= self.width_mm / 2:
            raise ValueError("arc radius must exceed half the slit width")

    def centerline_y(self, x_mm: np.ndarray) -> np.ndarray:
        """y of the arc centre line: y(x) = sqrt(R² − x²) − R (≤ 0)."""
        x = np.asarray(x_mm, float)
        return np.sqrt(self.radius_mm ** 2 - x ** 2) - self.radius_mm

    @property
    def sagitta_mm(self) -> float:
        """Arc sag across the slit: R − sqrt(R² − (W/2)²)."""
        return float(-self.centerline_y(self.width_mm / 2))

    def contains(self, x_mm: np.ndarray, y_mm: np.ndarray) -> np.ndarray:
        x = np.asarray(x_mm, float)
        y = np.asarray(y_mm, float)
        inside_x = np.abs(x) <= self.width_mm / 2
        yc = self.centerline_y(np.clip(x, -self.width_mm / 2, self.width_mm / 2))
        return inside_x & (np.abs(y - yc) <= self.height_mm / 2)

    def area_mm2(self) -> float:
        """Slit area W·h (an arc band of constant y-height has the same area as a rectangle)."""
        return self.width_mm * self.height_mm

    def scaled(self, mag_x: float = 4.0, mag_y: float = 4.0) -> "ArcSlit":
        """Slit at reticle level (approximation: radius scales with mag_x)."""
        return ArcSlit(self.width_mm * mag_x, self.height_mm * mag_y, self.radius_mm * mag_x)

    def grid_mask(self, nx: int = 261, ny: int = 81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean raster of the slit: returns (x, y, mask[y, x])."""
        x = np.linspace(-self.width_mm / 2, self.width_mm / 2, nx)
        y = np.linspace(-self.sagitta_mm - self.height_mm, self.height_mm, ny)
        X, Y = np.meshgrid(x, y)
        return x, y, self.contains(X, Y)

    def scan_integrated_height(self, nx: int = 261, ny: int = 2001) -> tuple[np.ndarray, np.ndarray]:
        """Scan-integrated slit height ∫ mask dy vs x (numerically ≈ h everywhere)."""
        x, y, m = self.grid_mask(nx, ny)
        return x, m.sum(axis=0) * (y[1] - y[0])


# ---------------------------------------------------------------------------
# field facet mirror layout
# ---------------------------------------------------------------------------
@dataclass
class FieldFacetMirror:
    """Arc-shaped field facets tiled over the annular far-field beam on the FFM.

    Facets have the aspect ratio of the slit (W/h ≈ 13 for 26 × 2 mm) and are
    arranged in rows ("bricks") inside the annulus r_in ≤ r ≤ r_out (normalised
    beam radius 1).  The facet size is chosen so ~``n_target`` facets fit.
    """

    n_target: int = 300
    r_in: float = 0.25          # central obscuration of the collector far field
    r_out: float = 1.0
    aspect: float = SLIT_WIDTH_MM / SLIT_HEIGHT_MM
    centers: np.ndarray = field(init=False, repr=False)
    facet_w: float = field(init=False)
    facet_h: float = field(init=False)

    def __post_init__(self) -> None:
        area = np.pi * (self.r_out ** 2 - self.r_in ** 2)
        # facet height h, width aspect*h; ~n facets: n*aspect*h² ≈ area*0.85 (packing)
        h = np.sqrt(0.85 * area / (self.n_target * self.aspect))
        best = None
        for hh in h * np.linspace(0.85, 1.15, 61):
            c = self._layout(hh)
            d = abs(len(c) - self.n_target)
            if best is None or d < best[0]:
                best = (d, hh, c)
        _, self.facet_h, self.centers = best
        self.facet_w = self.aspect * self.facet_h

    def _layout(self, h: float) -> np.ndarray:
        w = self.aspect * h
        rows = np.arange(-self.r_out + h / 2, self.r_out, h)
        pts = []
        for k, y in enumerate(rows):
            shift = 0.5 * w * (k % 2)          # brick pattern
            xs = np.arange(-self.r_out - w + shift, self.r_out + w, w)
            for x in xs:
                # all four corners within annulus outer edge, centre outside obscuration
                corners = [(x + sx * w / 2, y + sy * h / 2) for sx in (-1, 1) for sy in (-1, 1)]
                if all(cx ** 2 + cy ** 2 <= self.r_out ** 2 for cx, cy in corners) and \
                        x ** 2 + y ** 2 >= self.r_in ** 2:
                    pts.append((x, y))
        return np.array(pts).reshape(-1, 2)

    @property
    def n(self) -> int:
        return len(self.centers)

    def fill_factor(self) -> float:
        """Fraction of the annular beam area covered by facets (rest is lost)."""
        area = np.pi * (self.r_out ** 2 - self.r_in ** 2)
        return float(min(1.0, self.n * self.facet_w * self.facet_h / area))


def far_field_beam(u: np.ndarray, v: np.ndarray, r_in: float = 0.25, r_out: float = 1.0,
                   tilt: float = 0.05, rolloff: float = 0.3) -> np.ndarray:
    """Model collector far-field intensity on the FFM (normalised coordinates).

    Annular, with radial roll-off ``1 − rolloff·r²`` and a small linear tilt
    (source/collector misalignment).
    """
    r2 = u ** 2 + v ** 2
    I = (1 - rolloff * r2) * (1 + tilt * u)
    return np.where((r2 >= r_in ** 2) & (r2 <= r_out ** 2), I, 0.0)


# ---------------------------------------------------------------------------
# fly's-eye integrator
# ---------------------------------------------------------------------------
@dataclass
class FlyEyeIntegrator:
    """Superposition of N field-facet images in the slit.

    Parameters
    ----------
    n_facets : number of field facets (≈ target; actual from the FFM layout)
    facet_error_rms : rms relative random error of each facet's slit profile
        (smooth: Legendre orders 1..``error_orders``)
    beam_tilt, beam_rolloff : systematic far-field beam shape (see :func:`far_field_beam`)
    n_x : number of sample points across the slit
    """

    n_facets: int = 300
    facet_error_rms: float = 0.05
    error_orders: int = 6
    beam_tilt: float = 0.05
    beam_rolloff: float = 0.3
    n_x: int = 261
    slit: ArcSlit = field(default_factory=ArcSlit)
    ffm: FieldFacetMirror = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ffm = FieldFacetMirror(self.n_facets)

    @property
    def x_mm(self) -> np.ndarray:
        return np.linspace(-self.slit.width_mm / 2, self.slit.width_mm / 2, self.n_x)

    def facet_profiles(self, rng: np.random.Generator) -> np.ndarray:
        """(N, n_x) slit-intensity contribution of every field facet.

        Facet i located at (u_i, v_i) on the FFM samples the beam along its long
        axis; its image is mapped onto the slit (facet x → slit x).
        """
        xn = np.linspace(-1, 1, self.n_x)
        c = self.ffm.centers
        u = c[:, 0][:, None] + 0.5 * self.ffm.facet_w * xn[None, :]
        v = np.repeat(c[:, 1][:, None], self.n_x, axis=1)
        # beam inside the facet (no annulus clipping inside a facet: use smooth part)
        beam = (1 - self.beam_rolloff * (u ** 2 + v ** 2)) * (1 + self.beam_tilt * u)
        # random smooth facet errors (reflectivity / figure / contamination)
        n = len(c)
        k = self.error_orders
        coef = rng.normal(0.0, 1.0, (n, k + 1))
        coef[:, 0] = 0.0
        err = legendre.legval(xn, coef.T)                # (n, n_x)
        norm = np.sqrt(np.mean(err ** 2)) if np.any(err) else 1.0
        err *= self.facet_error_rms / norm
        return beam * (1 + err)

    def slit_profile(self, rng: np.random.Generator) -> np.ndarray:
        """Superposed slit intensity profile I(x), normalised to mean 1."""
        p = self.facet_profiles(rng).sum(axis=0)
        return p / p.mean()

    def uniformity(self, rng: np.random.Generator) -> float:
        return uniformity(self.slit_profile(rng))


def uniformity_vs_facets(n_list, facet_error_rms: float = 0.05, n_trials: int = 5,
                         rng: np.random.Generator | None = None, **kw) -> np.ndarray:
    """Mean slit non-uniformity for each facet count in ``n_list`` (random errors only
    unless beam parameters are passed).  Expect ∝ 1/√N."""
    rng = np.random.default_rng(0) if rng is None else rng
    kw.setdefault("beam_tilt", 0.0)
    kw.setdefault("beam_rolloff", 0.0)
    out = []
    for n in n_list:
        fe = FlyEyeIntegrator(n_facets=int(n), facet_error_rms=facet_error_rms, **kw)
        out.append(np.mean([fe.uniformity(rng) for _ in range(n_trials)]))
    return np.array(out)


# ---------------------------------------------------------------------------
# UNICOM finger correction
# ---------------------------------------------------------------------------
@dataclass
class UnicomResult:
    x_mm: np.ndarray
    finger_x_mm: np.ndarray
    insertion_mm: np.ndarray           # finger insertion depth into the slit
    transmission: np.ndarray           # h_eff(x)/h
    dose_before: np.ndarray            # normalised scan-integrated dose before
    dose_after: np.ndarray
    uniformity_before: float
    uniformity_after: float
    light_loss: float                  # 1 − mean(after)/mean(before)


@dataclass
class Unicom:
    """Finger-array uniformity corrector (UNICOM-like) at the slit edge."""

    finger_pitch_mm: float = 4.0
    slit_width_mm: float = SLIT_WIDTH_MM
    slit_height_mm: float = SLIT_HEIGHT_MM
    blur_mm: float = 1.5               # footprint blur (fingers are out of focus)
    max_insertion_frac: float = 0.15   # max δ/h
    regularisation: float = 1e-3       # ridge weight on δ/h: picks the minimum-loss solution

    @property
    def finger_x_mm(self) -> np.ndarray:
        n = int(np.ceil(self.slit_width_mm / self.finger_pitch_mm)) + 1
        return (np.arange(n) - (n - 1) / 2) * self.finger_pitch_mm

    def footprints(self, x_mm: np.ndarray) -> np.ndarray:
        """(n_fingers, n_x) footprint g_j(x): rect of width = pitch convolved with a
        Gaussian of σ = ``blur_mm`` (sum of all footprints ≈ 1)."""
        from scipy.special import erf
        xf = self.finger_x_mm[:, None]
        a = self.finger_pitch_mm / 2
        s = max(self.blur_mm, 1e-6) * np.sqrt(2)
        x = np.asarray(x_mm)[None, :]
        return 0.5 * (erf((x - xf + a) / s) - erf((x - xf - a) / s))

    def correct(self, x_mm: np.ndarray, intensity: np.ndarray) -> UnicomResult:
        """Find insertions δ_j minimising ‖I·(1 − Σ δ_j g_j/h) − T‖ with T free.

        Linear in (δ, T):  Σ_j δ_j (I g_j / h) + T = I,  0 ≤ δ_j ≤ δmax.
        A uniform insertion of all fingers only lowers T (no uniformity gain),
        so a small ridge term λ‖δ/h‖² selects the least light-loss solution.
        """
        I = np.asarray(intensity, float)
        I = I / I.mean()
        g = self.footprints(x_mm)
        h = self.slit_height_mm
        nf = len(g)
        A = np.column_stack([(I[None, :] * g / h).T, np.ones_like(I)])
        lam = np.sqrt(self.regularisation * len(I) / nf)
        R = np.column_stack([lam * np.eye(nf) / h, np.zeros(nf)])
        A = np.vstack([A, R])
        b = np.r_[I, np.zeros(nf)]
        ub = np.r_[np.full(nf, self.max_insertion_frac * h), np.inf]
        res = lsq_linear(A, b, bounds=(np.zeros(len(ub)), ub))
        d = res.x[:-1]
        t = 1 - (d[:, None] * g).sum(axis=0) / h
        after = I * t
        return UnicomResult(x_mm=np.asarray(x_mm), finger_x_mm=self.finger_x_mm, insertion_mm=d,
                            transmission=t, dose_before=I, dose_after=after,
                            uniformity_before=uniformity(I), uniformity_after=uniformity(after),
                            light_loss=float(1 - after.mean() / I.mean()))
