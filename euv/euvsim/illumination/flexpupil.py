"""Facet-based pupil discretisation and the FlexPupil concept.

In a fly's-eye EUV illuminator the pupil is not a continuous distribution: it
consists of the images of the intermediate-focus (IF) plasma formed on the
individual *pupil facets* of the pupil facet mirror (PFM).  Each *field
facet* on the field facet mirror (FFM) sends its light to exactly one pupil
facet.  Which pupil facets are lit therefore defines the illumination
setting.

FlexPupil-type illuminators (public literature, e.g. NXE:3300 onwards) give
each field facet a tip/tilt actuator with a small number of discrete
positions (typically 2), each aiming at a different pupil facet.  Changing
the setting means re-pointing field facets, so (ideally) *no light is
blocked*: the pupil shape is reconfigured instead of masked.  The price is
that the reachable shapes are constrained by the facet→pupil-facet
assignment ("pairing") designed into the hardware.

Model used here
---------------
* Pupil facets: square lattice of pitch ``p`` (σ units) inside |σ| ≤ σmax,
  chosen so the count ≈ ``n_pupil_facets``.  A square lattice centred on
  half-integers is invariant under 90° rotation, which lets us build the
  classic "x-dipole ↔ y-dipole" pairing.
* Each pupil-facet image is a small disc of radius ``spot_fill · p/2``
  (sub-sampled into source points).
* Pairing ``'rotate90'``: field facet j can address pupil facet q or R₉₀·q.
  ``'random'``: random disjoint pairs.  ``positions`` > 2 → random k-tuples.
* Selection for a target mask: each field facet picks a candidate pupil facet
  inside the target (least-loaded first, for balance).  If none of its
  positions is inside the target it is *parked* on a beam dump (light lost)
  or, with ``policy='nearest'``, sent to the candidate closest to the target.

The efficiency ``η_flex`` (fraction of FFM power landing in the target) is
compared with the classical masking approach, where a filled pupil is
apertured: ``η_mask = PFR_target / PFR_filled``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pupil import Mask, _points, pupil_fill_ratio


@dataclass
class PupilFacetMirror:
    """Square lattice of pupil facets inside the unit pupil disc."""

    n_pupil_facets: int = 600
    sigma_max: float = 0.98
    spot_fill: float = 0.8          # facet-image diameter / facet pitch
    centers: np.ndarray = field(init=False, repr=False)
    pitch: float = field(init=False)

    def __post_init__(self) -> None:
        if self.n_pupil_facets < 8:
            raise ValueError("need at least 8 pupil facets")
        self.pitch = float(np.sqrt(np.pi * self.sigma_max ** 2 / self.n_pupil_facets))
        # adjust pitch so the number of facets is as close as possible to the request
        best = None
        for p in self.pitch * np.linspace(0.9, 1.1, 81):
            c = self._lattice(p)
            d = abs(len(c) - self.n_pupil_facets)
            if best is None or d < best[0]:
                best = (d, p, c)
        _, self.pitch, self.centers = best

    def _lattice(self, p: float) -> np.ndarray:
        n = int(np.ceil(self.sigma_max / p)) + 1
        g = (np.arange(-n, n) + 0.5) * p
        x, y = np.meshgrid(g, g)
        r = np.hypot(x, y) + 0.5 * self.spot_fill * p
        keep = r <= self.sigma_max + 1e-12
        return np.column_stack([x[keep], y[keep]])

    @property
    def n(self) -> int:
        return len(self.centers)

    @property
    def spot_radius(self) -> float:
        return 0.5 * self.spot_fill * self.pitch

    def index_of(self, xy: np.ndarray) -> np.ndarray:
        """Indices of the facets nearest to points ``xy`` (M, 2)."""
        d = np.linalg.norm(self.centers[None, :, :] - np.asarray(xy)[:, None, :], axis=2)
        return np.argmin(d, axis=1)


@dataclass
class FlexPupilResult:
    points: np.ndarray            # source points (N, 3)
    lit_facets: np.ndarray        # index of pupil facet chosen by each field facet (-1 = parked)
    efficiency: float             # fraction of FFM power reaching the pupil in the target
    in_target_fraction: float     # fraction of delivered power that lies inside the target
    pfr: float                    # pupil fill ratio of the realised pupil


@dataclass
class FlexPupil:
    """Field facets with multi-position actuators addressing a pupil facet mirror."""

    n_pupil_facets: int = 600
    positions: int = 2
    pairing: str = "rotate90"
    seed: int = 0
    spot_fill: float = 0.8
    pfm: PupilFacetMirror = field(init=False, repr=False)
    candidates: np.ndarray = field(init=False, repr=False)   # (n_field, positions)

    def __post_init__(self) -> None:
        self.pfm = PupilFacetMirror(self.n_pupil_facets, spot_fill=self.spot_fill)
        rng = np.random.default_rng(self.seed)
        c = self.pfm.centers
        if self.pairing == "rotate90":
            if self.positions != 2:
                raise ValueError("rotate90 pairing requires positions=2")
            rot = self.pfm.index_of(np.column_stack([-c[:, 1], c[:, 0]]))
            used = np.zeros(len(c), bool)
            pairs = []
            # orbit {q, Rq, R²q, R³q}: pair (q, Rq) and (R²q, R³q)
            for q in range(len(c)):
                if used[q]:
                    continue
                q1 = rot[q]
                q2 = rot[q1]
                q3 = rot[q2]
                pairs += [(q, q1), (q2, q3)]
                used[[q, q1, q2, q3]] = True
            self.candidates = np.array(pairs, int)
        elif self.pairing == "random":
            k = self.positions
            perm = rng.permutation(len(c))
            m = len(c) // k
            self.candidates = perm[: m * k].reshape(m, k)
        else:
            raise ValueError("pairing must be 'rotate90' or 'random'")

    @property
    def n_field_facets(self) -> int:
        return len(self.candidates)

    def configure(self, target: Mask, policy: str = "park", field_power: np.ndarray | None = None,
                  subsample: int = 3) -> FlexPupilResult:
        """Choose actuator positions to approximate the target pupil mask.

        Parameters
        ----------
        target : boolean mask ``target(sx, sy)`` (see :func:`pupil.shape_mask`)
        policy : ``'park'`` (unusable facets are dumped, light lost) or
                 ``'nearest'`` (send to the candidate nearest the target region)
        field_power : relative power of each field facet (default: equal)
        subsample : facet image sampled on a ``subsample × subsample`` grid
        """
        if policy not in ("park", "nearest"):
            raise ValueError("policy must be 'park' or 'nearest'")
        nf = self.n_field_facets
        P = np.ones(nf) if field_power is None else np.asarray(field_power, float)
        c = self.pfm.centers
        in_t = np.asarray(target(c[:, 0], c[:, 1]), bool)
        load = np.zeros(len(c))
        chosen = np.full(nf, -1, int)
        # facets with fewer options first so the greedy balance works well
        n_opts = in_t[self.candidates].sum(axis=1)
        order = np.argsort(n_opts, kind="stable")
        tgt_pts = c[in_t]
        for j in order:
            cand = self.candidates[j]
            ok = cand[in_t[cand]]
            if ok.size:
                q = ok[np.argmin(load[ok] + 1e-9 * np.arange(ok.size))]
            elif policy == "nearest" and tgt_pts.size:
                d = [np.min(np.linalg.norm(tgt_pts - c[q], axis=1)) for q in cand]
                q = cand[int(np.argmin(d))]
            else:
                continue
            chosen[j] = q
            load[q] += P[j]
        delivered = P[chosen >= 0].sum()
        eff_target = P[(chosen >= 0) & in_t[np.maximum(chosen, 0)]].sum()
        # build source points: each lit facet image = disc sampled by sub-points
        lit = np.nonzero(load > 0)[0]
        if lit.size == 0:
            raise ValueError("target pupil cannot be reached by any field facet")
        r = self.pfm.spot_radius
        u = (np.arange(subsample) + 0.5) / subsample * 2 - 1
        ux, uy = np.meshgrid(u, u)
        disc = (ux ** 2 + uy ** 2) <= 1
        ox, oy = ux[disc] * r, uy[disc] * r
        sx = (c[lit, 0][:, None] + ox[None, :]).ravel()
        sy = (c[lit, 1][:, None] + oy[None, :]).ravel()
        w = np.repeat(load[lit], ox.size)
        pts = _points(sx, sy, w)
        # effective PFR: lit facet image area / π (energy-weighted)
        n_eff = load[lit].sum() ** 2 / np.sum(load[lit] ** 2)
        pfr = float(n_eff * np.pi * r ** 2 / np.pi)
        return FlexPupilResult(points=pts, lit_facets=chosen,
                               efficiency=float(eff_target / P.sum()),
                               in_target_fraction=float(eff_target / delivered) if delivered else 0.0,
                               pfr=pfr)


def masking_efficiency(target_pfr: float, filled_pfr: float = 0.8) -> float:
    """Fraction of light kept when a filled pupil (PFR ``filled_pfr``) is apertured
    down to the target shape: η_mask = PFR_target / PFR_filled (≤ 1)."""
    if filled_pfr <= 0:
        raise ValueError("filled_pfr must be > 0")
    return float(min(1.0, target_pfr / filled_pfr))


__all__ = ["PupilFacetMirror", "FlexPupil", "FlexPupilResult", "masking_efficiency",
           "pupil_fill_ratio"]
