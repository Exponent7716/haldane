"""Resist metrology: CD, LER/LWR, local CDU, NILS, stochastic defects and the
analytic shot-noise relation.

Conventions
-----------
* Lines/spaces run along **y** (the image varies along x); each row ``y`` is
  one cut through the features.
* ``field`` is a continuous develop field (see :mod:`euvsim.resist.develop`):
  F > 0 where the exposed feature printed; edges are its zero crossings,
  located with linear sub-pixel interpolation.
* Roughness is reported as 3 sigma:  LER = 3 std(x_edge),  LWR = 3 std(CD).
  For uncorrelated edges LWR ~ sqrt(2) LER.

Shot-noise limit
----------------
The edge-placement noise produced by photon / acid counting statistics is::

    sigma_x ~ (sigma_h / h) / ILS_latent = 1 / (ILS * sqrt(N_eff))
    N_eff   = n_ph(D) * I_edge * (A / T) * QY * V_corr

with n_ph(D) ~ 0.68 D photons/nm^2, A the absorbance, V_corr the volume of
the blur-defined correlation cell (~ 4 pi sigma^2 T). Hence
LER proportional to 1 / sqrt(D * A), the classic stochastic trade-off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Callable, Literal, Optional, Sequence

import numpy as np
from scipy import ndimage

from ..constants import photons_per_nm2


# ---------------------------------------------------------------------------
# lines / spaces
# ---------------------------------------------------------------------------
@dataclass
class LineMetrics:
    """CD / roughness of a set of lines (or spaces) running along y.

    All lengths in nm. Edge arrays have shape ``(n_features, n_rows)`` and
    contain NaN where the edge could not be found (break / bridge).
    """

    cd_mean: float
    cd_std: float
    lwr_3sigma: float
    ler_3sigma: float
    centres_nm: np.ndarray
    left_edges_nm: np.ndarray
    right_edges_nm: np.ndarray
    widths_nm: np.ndarray
    failed_row_fraction: float

    @property
    def n_features(self) -> int:
        return len(self.centres_nm)


def _feature_centres(profile: np.ndarray) -> np.ndarray:
    """Pixel centres of the printed (profile > 0) segments not touching the border."""
    pos = profile > 0
    d = np.diff(pos.astype(int))
    starts = np.nonzero(d == 1)[0] + 1
    ends = np.nonzero(d == -1)[0]
    centres = []
    for s in starts:
        e = ends[ends >= s]
        if len(e):
            centres.append(0.5 * (s + e[0]))
    return np.asarray(centres)


def find_line_edges(field: np.ndarray, dx_nm: float,
                    centres_px: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-pixel left/right edge positions of each feature in every row.

    Returns ``(centres_px, left_nm, right_nm)``. The search for a feature's
    edges is restricted to half way to its neighbours; an edge outside that
    window or a feature not printed at its centre gives NaN.
    """
    f = np.atleast_2d(np.asarray(field, dtype=float))
    ny, nx = f.shape
    if centres_px is None:
        centres_px = _feature_centres(f.mean(axis=0))
    c = np.asarray(centres_px, dtype=float)
    nf = len(c)
    left = np.full((nf, ny), np.nan)
    right = np.full((nf, ny), np.nan)
    if nf == 0:
        return c, left, right
    if nf > 1:
        pitch = np.median(np.diff(c))
    else:
        pitch = float(nx)
    for k, ck in enumerate(c):
        lo = max(int(np.floor(ck - 0.5 * pitch)), 0)
        hi = min(int(np.ceil(ck + 0.5 * pitch)), nx - 1)
        ci = int(round(ck))
        seg_l = f[:, lo:ci + 1]          # columns lo..ci
        seg_r = f[:, ci:hi + 1]          # columns ci..hi
        ok = f[:, ci] > 0
        # left edge: last non-positive sample left of centre
        neg_l = seg_l <= 0
        has_l = neg_l.any(axis=1) & ok
        j = seg_l.shape[1] - 1 - np.argmax(neg_l[:, ::-1], axis=1)   # index in seg_l
        rows = np.nonzero(has_l)[0]
        jj = j[rows]
        jj = np.minimum(jj, seg_l.shape[1] - 2)
        a = seg_l[rows, jj]
        b = seg_l[rows, jj + 1]
        left[k, rows] = (lo + jj + a / (a - b)) * dx_nm
        # right edge: first non-positive sample right of centre
        neg_r = seg_r <= 0
        has_r = neg_r.any(axis=1) & ok
        j2 = np.argmax(neg_r, axis=1)
        rows = np.nonzero(has_r)[0]
        jj = np.maximum(j2[rows], 1)
        a = seg_r[rows, jj - 1]
        b = seg_r[rows, jj]
        right[k, rows] = (ci + jj - 1 + a / (a - b)) * dx_nm
    return c, left, right


def measure_lines(field: np.ndarray, dx_nm: float,
                  centres_px: Optional[Sequence[float]] = None) -> LineMetrics:
    """CD, LWR and LER (3 sigma) of the printed features in a develop field.

    LER is the mean over left/right edges of 3 std of the edge position
    (each edge referenced to its own mean, i.e. placement offset removed).
    """
    c, left, right = find_line_edges(field, dx_nm, centres_px)
    widths = right - left
    valid = np.isfinite(widths)
    failed = 1.0 - valid.mean() if widths.size else 1.0
    if not valid.any():
        nan = float("nan")
        return LineMetrics(nan, nan, nan, nan, c * dx_nm, left, right, widths, failed)
    cd_mean = float(np.nanmean(widths))
    cd_std = float(np.nanstd(widths))
    lwr = 3.0 * float(np.nanmean(np.nanstd(widths, axis=1)))
    ler_l = np.nanstd(left, axis=1)
    ler_r = np.nanstd(right, axis=1)
    ler = 3.0 * float(np.nanmean(np.concatenate([ler_l, ler_r])))
    return LineMetrics(cd_mean, cd_std, lwr, ler, c * dx_nm, left, right, widths, float(failed))


def measure_cd(field: np.ndarray, dx_nm: float) -> float:
    """Mean CD [nm] of the printed features (lines/spaces along y)."""
    return measure_lines(field, dx_nm).cd_mean


# ---------------------------------------------------------------------------
# contacts
# ---------------------------------------------------------------------------
@dataclass
class ContactMetrics:
    """Contact-hole statistics; diameters are area-equivalent, d = 2 sqrt(A/pi)."""

    diameters_nm: np.ndarray
    cd_mean: float
    lcdu_3sigma: float
    n_found: int


def measure_contacts(printed: np.ndarray, dx_nm: float, min_area_px: int = 2,
                     field: Optional[np.ndarray] = None) -> ContactMetrics:
    """Local CDU (3 sigma of contact diameters) from a printed binary map.

    If ``field`` is given the area of each contact is refined to sub-pixel
    accuracy by counting each boundary pixel with the fractional weight
    ``clip(0.5 + F / |grad F|, 0, 1)`` (signed-distance estimate).
    Contacts touching the border are excluded.
    """
    pr = np.asarray(printed, dtype=bool)
    lab, n = ndimage.label(pr)
    if n == 0:
        return ContactMetrics(np.array([]), float("nan"), float("nan"), 0)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    if field is not None:
        gy, gx = np.gradient(np.asarray(field, dtype=float))
        g = np.hypot(gx, gy) + 1e-12
        w = np.clip(0.5 + field / g, 0.0, 1.0)
        # weight for every pixel in the dilated region of each contact
        lab_d = ndimage.grey_dilation(lab, size=(3, 3))
        areas = ndimage.sum(w, lab_d, index=np.arange(1, n + 1))
    else:
        areas = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    counts = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = [i for i in range(n) if (i + 1) not in border and counts[i] >= min_area_px]
    if not keep:
        return ContactMetrics(np.array([]), float("nan"), float("nan"), 0)
    d = 2.0 * np.sqrt(np.asarray(areas)[keep] / math.pi) * dx_nm
    lcdu = 3.0 * float(np.std(d)) if len(d) > 1 else float("nan")
    return ContactMetrics(d, float(np.mean(d)), lcdu, len(d))


# ---------------------------------------------------------------------------
# image quality
# ---------------------------------------------------------------------------
def nils(profile: np.ndarray, dx_nm: float, cd_nm: float, edge_x_nm: float) -> float:
    """Normalised image log slope NILS = CD * |d ln I / dx| at ``edge_x_nm``.

    ``profile`` is a 1-D intensity (or latent-image) cut along x; the
    derivative is evaluated by central differences and linearly interpolated.
    """
    p = np.asarray(profile, dtype=float)
    lnp = np.log(np.clip(p, 1e-300, None))
    d = np.gradient(lnp, dx_nm)
    x = np.arange(len(p)) * dx_nm
    return float(cd_nm * abs(np.interp(edge_x_nm, x, d)))


# ---------------------------------------------------------------------------
# analytic shot-noise relation
# ---------------------------------------------------------------------------
def analytic_ler_3sigma(dose_mj_cm2: float, absorbance: float, thickness_nm: float,
                        ils_per_nm: float, blur_nm: float, quantum_yield: float = 1.0,
                        edge_intensity: float = 0.5) -> float:
    """Shot-noise-limited LER (3 sigma) [nm].

    N_eff = photons_per_nm2(D) * I_edge * A * QY * 4 pi sigma_b^2   (acids in
    the blur cell, A = absorbed fraction over the full film);
    LER = 3 / (ILS * sqrt(N_eff)).
    Scales as (D * A)^-1/2.
    """
    n_eff = photons_per_nm2(dose_mj_cm2) * edge_intensity * absorbance * quantum_yield \
        * 4.0 * math.pi * blur_nm**2
    return 3.0 / (ils_per_nm * math.sqrt(n_eff))


def ler_dose_scaling(ler_ref_nm: float, dose_ref: float, dose: float,
                     absorbance_ref: float = 1.0, absorbance: float = 1.0) -> float:
    """LER(D, A) = LER_ref * sqrt(D_ref A_ref / (D A))."""
    return ler_ref_nm * math.sqrt(dose_ref * absorbance_ref / (dose * absorbance))


# ---------------------------------------------------------------------------
# stochastic defects
# ---------------------------------------------------------------------------
FeatureKind = Literal["line", "contact"]


@dataclass
class DefectStats:
    """Monte-Carlo stochastic-defect estimate.

    p_fail: probability that a given nominal feature fails.
    p_open: missing contact / micro-break (feature not open across a cut).
    p_bridge: merged neighbours (bridge / kissing contacts).
    upper_95: one-sided 95 % upper bound on p_fail (rule of three when 0).
    """

    p_fail: float
    p_open: float
    p_bridge: float
    n_trials: int
    n_features: int
    n_fail: int
    upper_95: float
    per_trial_fail: np.ndarray = dc_field(repr=False, default_factory=lambda: np.array([]))


def count_feature_defects(printed: np.ndarray, nominal: np.ndarray,
                          kind: FeatureKind = "line") -> tuple[np.ndarray, np.ndarray]:
    """Per-feature open / bridge flags of a printed map vs. the nominal map.

    Nominal features are the connected components of ``nominal`` (not touching
    the border for contacts).

    * ``line``: open (micro-break) if some row of the feature footprint has no
      printed pixel; bridge to the right neighbour if in some row the gap
      between the two footprints is fully printed.
    * ``contact``: open (missing) if no printed pixel in the footprint;
      bridge if its printed component also covers another nominal contact.
    """
    pr = np.asarray(printed, dtype=bool)
    nom = np.asarray(nominal, dtype=bool)
    lab, n = ndimage.label(nom)
    opens = np.zeros(n, dtype=bool)
    bridges = np.zeros(n, dtype=bool)
    if n == 0:
        return opens, bridges
    if kind == "line":
        cols = []
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            x0, x1 = xs.min(), xs.max()
            rows = np.unique(ys)
            cols.append((x0, x1))
            opens[i - 1] = not pr[rows][:, x0:x1 + 1].any(axis=1).all()
        order = np.argsort([c[0] for c in cols])
        for a, b in zip(order[:-1], order[1:]):
            g0, g1 = cols[a][1] + 1, cols[b][0]
            if g1 > g0 and pr[:, g0:g1].all(axis=1).any():
                bridges[a] = bridges[b] = True
    else:
        plab, _ = ndimage.label(pr)
        owner = {}
        for i in range(1, n + 1):
            ids = np.unique(plab[lab == i])
            ids = ids[ids > 0]
            if len(ids) == 0:
                opens[i - 1] = True
            for j in ids:
                owner.setdefault(int(j), []).append(i - 1)
        for feats in owner.values():
            if len(feats) > 1:
                bridges[feats] = True
    return opens, bridges


def monte_carlo_defects(simulate: Callable[[np.random.Generator], np.ndarray],
                        nominal: np.ndarray, n_trials: int, rng: np.random.Generator,
                        kind: FeatureKind = "line") -> DefectStats:
    """Estimate stochastic failure probability by repeated noisy exposures.

    ``simulate(rng)`` must return a printed binary map. p = n_fail / (n_trials
    * n_features); the 95 % upper bound uses the Wilson score interval
    (or 3/N when no failure is observed).
    """
    n_open = n_bridge = n_fail = 0
    per_trial = np.zeros(n_trials)
    nf = 0
    for t in range(n_trials):
        o, b = count_feature_defects(simulate(rng), nominal, kind)
        nf = len(o)
        n_open += int(o.sum())
        n_bridge += int(b.sum())
        fails = int((o | b).sum())
        n_fail += fails
        per_trial[t] = fails
    ntot = max(n_trials * nf, 1)
    p = n_fail / ntot
    if n_fail == 0:
        ub = 3.0 / ntot
    else:
        z = 1.645
        den = 1 + z**2 / ntot
        ub = (p + z**2 / (2 * ntot) + z * math.sqrt(p * (1 - p) / ntot + z**2 / (4 * ntot**2))) / den
    return DefectStats(p, n_open / ntot, n_bridge / ntot, n_trials, nf, n_fail, ub, per_trial)
