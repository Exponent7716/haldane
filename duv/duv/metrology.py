"""On-board metrology and overlay-correction models of a dual-stage ArF scanner.

This module models the *measure side* of a dual-stage scanner and the
overlay control built on top of it:

* :class:`WaferDeformation` -- ground-truth wafer grid distortion (linear
  terms, 3rd-order polynomial, random smooth field, random noise).
* :class:`AlignmentSensor` -- phase-grating, self-referencing alignment sensor
  (SMASH / ATHENA-like).  The mark position is encoded in the phase of the
  interference signal of the +/-1 diffraction orders; mark asymmetry produces
  a colour-dependent apparent shift that a multi-colour estimate suppresses.
* Wafer-grid models: :func:`fit_linear_wafer_model`, :func:`fit_high_order`,
  :func:`apply_model`, :func:`residuals`, plus :func:`alignment_strategy` and
  :func:`evaluate_alignment_strategy`.
* :class:`LevelSensor` -- grazing-incidence optical triangulation height
  sensor; :func:`focus_correction` and :func:`focus_residual` turn the height
  map into per-slit (z, Rx, Ry) set-points and the remaining defocus.
* :class:`Overlay` -- overlay computation, statistics and k-parameter
  intrafield models.
* :class:`CorrectionLoop` -- EWMA run-to-run (APC) controller.

Units: lateral wafer coordinates in mm, displacements / heights in nm.  A
convenient consequence is that 1 ppm magnification at 1 mm gives 1 nm and
1 urad rotation at 1 mm gives 1 nm, so linear-model coefficients are quoted
in ppm / urad (== nm/mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from duv.core import FIELD_HEIGHT_MM, FIELD_WIDTH_MM, SLIT_HEIGHT_MM, WAFER_DIAMETER_MM

WAFER_RADIUS_MM = WAFER_DIAMETER_MM / 2.0
LINEAR_PARAM_NAMES = ("Tx", "Ty", "Mx", "My", "Rx", "Ry")


# =============================================================================
# Field layout helper (independent of the stage module)
# =============================================================================
def field_centers(
    field_w_mm: float = FIELD_WIDTH_MM,
    field_h_mm: float = FIELD_HEIGHT_MM,
    wafer_diameter_mm: float = WAFER_DIAMETER_MM,
    edge_exclusion_mm: float = 3.0,
) -> np.ndarray:
    """Centres (N, 2) in mm of all full fields that fit inside the usable wafer.

    The grid is centred on the wafer; a field is kept if all four corners lie
    within ``R - edge_exclusion``.
    """
    r = wafer_diameter_mm / 2.0 - edge_exclusion_mm
    nx = int(np.ceil(wafer_diameter_mm / field_w_mm)) + 2
    ny = int(np.ceil(wafer_diameter_mm / field_h_mm)) + 2
    cx = (np.arange(nx) - (nx - 1) / 2.0) * field_w_mm
    cy = (np.arange(ny) - (ny - 1) / 2.0) * field_h_mm
    X, Y = np.meshgrid(cx, cy)
    X, Y = X.ravel(), Y.ravel()
    ok = np.ones_like(X, dtype=bool)
    for sx in (-0.5, 0.5):
        for sy in (-0.5, 0.5):
            ok &= np.hypot(X + sx * field_w_mm, Y + sy * field_h_mm) <= r
    return np.column_stack([X[ok], Y[ok]])


def _as_xy(xy) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return xy.reshape(-1, 2)


# =============================================================================
# Ground truth wafer deformation
# =============================================================================
def linear_components_to_params(
    tx: float = 0.0,
    ty: float = 0.0,
    mag_ppm: float = 0.0,
    mag_asym_ppm: float = 0.0,
    rot_urad: float = 0.0,
    nonorth_urad: float = 0.0,
) -> dict:
    """Convert physical linear grid terms into the 6-parameter model.

    Model: ``dx = Tx + Mx*x - Rx*y`` and ``dy = Ty + My*y + Ry*x`` with
    ``Mx = mag + mag_asym``, ``My = mag - mag_asym``,
    ``Rx = rot + nonorth/2`` and ``Ry = rot - nonorth/2``.
    """
    return {
        "Tx": tx,
        "Ty": ty,
        "Mx": mag_ppm + mag_asym_ppm,
        "My": mag_ppm - mag_asym_ppm,
        "Rx": rot_urad + nonorth_urad / 2.0,
        "Ry": rot_urad - nonorth_urad / 2.0,
    }


def linear_params_to_components(p: dict) -> dict:
    """Inverse of :func:`linear_components_to_params`."""
    return {
        "tx": p["Tx"],
        "ty": p["Ty"],
        "mag_ppm": (p["Mx"] + p["My"]) / 2.0,
        "mag_asym_ppm": (p["Mx"] - p["My"]) / 2.0,
        "rot_urad": (p["Rx"] + p["Ry"]) / 2.0,
        "nonorth_urad": p["Rx"] - p["Ry"],
    }


class WaferDeformation:
    """Ground-truth wafer grid distortion.

    ``displacement(x, y)`` = linear grid + 3rd-order polynomial + random smooth
    field (sum of a few low-spatial-frequency sinusoids), optionally plus
    white placement noise when an ``rng`` is supplied.

    Parameters
    ----------
    tx, ty : translation, nm.
    mag_ppm, mag_asym_ppm : symmetric / asymmetric magnification, ppm.
    rot_urad, nonorth_urad : rotation / non-orthogonality, urad.
    third_order_nm : dict with keys ``"x"``/``"y"`` mapping to dict
        ``{(i, j): c}`` of coefficients (nm) of ``(x/R)^i (y/R)^j`` with
        ``i + j == 2 or 3``; or a float giving a random draw of that rms
        amplitude.
    smooth_amp_nm : amplitude of the random smooth field (nm).
    noise_nm : 1-sigma random (non-systematic) placement noise.
    seed : seed for the random parts of the systematic field.
    """

    def __init__(
        self,
        tx: float = 0.0,
        ty: float = 0.0,
        mag_ppm: float = 0.0,
        mag_asym_ppm: float = 0.0,
        rot_urad: float = 0.0,
        nonorth_urad: float = 0.0,
        third_order_nm: dict | float | None = None,
        smooth_amp_nm: float = 0.0,
        noise_nm: float = 0.0,
        seed: int = 0,
    ):
        self.linear = linear_components_to_params(
            tx, ty, mag_ppm, mag_asym_ppm, rot_urad, nonorth_urad
        )
        rng = np.random.default_rng(seed)
        if third_order_nm is None:
            self.hot = {"x": {}, "y": {}}
        elif isinstance(third_order_nm, dict):
            self.hot = {"x": dict(third_order_nm.get("x", {})), "y": dict(third_order_nm.get("y", {}))}
        else:
            terms = [(i, n - i) for n in (2, 3) for i in range(n + 1)]
            self.hot = {
                ax: {t: float(third_order_nm * rng.standard_normal()) for t in terms}
                for ax in ("x", "y")
            }
        self.smooth_amp_nm = smooth_amp_nm
        nw = 4
        self._kx = rng.uniform(0.5, 2.0, (2, nw)) * np.pi / WAFER_RADIUS_MM
        self._ky = rng.uniform(0.5, 2.0, (2, nw)) * np.pi / WAFER_RADIUS_MM
        self._ph = rng.uniform(0, 2 * np.pi, (2, nw))
        self._amp = rng.standard_normal((2, nw)) / np.sqrt(nw)
        self.noise_nm = noise_nm

    def systematic(self, x_mm, y_mm) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic part of the displacement (nm)."""
        x = np.asarray(x_mm, dtype=float)
        y = np.asarray(y_mm, dtype=float)
        dx, dy = _linear_eval(self.linear, x, y)
        u, v = x / WAFER_RADIUS_MM, y / WAFER_RADIUS_MM
        for (i, j), c in self.hot["x"].items():
            dx = dx + c * u**i * v**j
        for (i, j), c in self.hot["y"].items():
            dy = dy + c * u**i * v**j
        if self.smooth_amp_nm:
            sm = [np.zeros(np.broadcast(x, y).shape) for _ in range(2)]
            for a in range(2):
                for k in range(self._amp.shape[1]):
                    sm[a] = sm[a] + self._amp[a, k] * np.sin(
                        self._kx[a, k] * x + self._ky[a, k] * y + self._ph[a, k]
                    )
            dx = dx + self.smooth_amp_nm * sm[0]
            dy = dy + self.smooth_amp_nm * sm[1]
        return dx, dy

    def displacement(self, x_mm, y_mm, rng: np.random.Generator | None = None):
        """Grid displacement (dx_nm, dy_nm) at wafer position(s) (x_mm, y_mm).

        Random placement noise of ``noise_nm`` is added only if ``rng`` is given.
        """
        dx, dy = self.systematic(x_mm, y_mm)
        if rng is not None and self.noise_nm > 0:
            dx = dx + self.noise_nm * rng.standard_normal(np.shape(dx))
            dy = dy + self.noise_nm * rng.standard_normal(np.shape(dy))
        return dx, dy


# =============================================================================
# Alignment sensor
# =============================================================================
class AlignmentSensor:
    """Phase-grating self-referencing alignment sensor (SMASH/ATHENA-like).

    A phase grating of pitch ``P`` is scanned under the sensor spot.  The
    self-referencing interferometer overlaps the +1 and -1 diffraction orders,
    whose relative phase changes by ``2*k_g*x`` with ``k_g = 2*pi/P``; the
    detected intensity is therefore

        ``I(s) = A + B_l * cos(2*pi*(s - x_mark - delta_l) / (P/2))``

    with ``s`` the scan coordinate.  The position is the phase of that
    fringe, extracted with a linear least-squares (single-frequency DFT) fit.
    The capture range is +/- P/4 around ``x_expected`` (coarse alignment).

    Mark asymmetry (e.g. a tilted grating floor after CMP/etch), expressed by
    the scalar ``mark_asymmetry`` (nm), adds a colour-dependent apparent shift
    ``delta_l = mark_asymmetry * sensitivity(l)``.  The stack model gives an
    oscillating sensitivity ``cos(4*pi*n*d/l)`` and diffraction efficiency
    (signal strength) ``0.1 + 0.9*sin^2(2*pi*n*d/l)`` for a mark of optical
    depth ``n*d``.  ``noise_nm`` is the 1-sigma single-colour reproducibility
    at full signal strength; weak colours are noisier as ``1/B``.
    """

    def __init__(
        self,
        wavelengths_nm=(532, 633, 780, 850),
        grating_pitch_um: float = 16.0,
        noise_nm: float = 0.1,
        mark_depth_nm: float = 300.0,
        mark_index: float = 1.46,
        asym_gain: float = 1.0,
        scan_length_um: float = 32.0,
        n_samples: int = 400,
    ):
        self.wavelengths_nm = tuple(float(w) for w in wavelengths_nm)
        self.pitch_nm = grating_pitch_um * 1e3
        self.period_nm = self.pitch_nm / 2.0  # +/-1 order interference
        self.noise_nm = noise_nm
        self.mark_depth_nm = mark_depth_nm
        self.mark_index = mark_index
        self.asym_gain = asym_gain
        self.scan_length_nm = scan_length_um * 1e3
        self.n_samples = n_samples

    # --- stack model --------------------------------------------------------
    def sensitivity(self, wavelength_nm: float) -> float:
        """Apparent shift per nm of mark asymmetry at this colour (nm/nm)."""
        opd = self.mark_index * self.mark_depth_nm
        return self.asym_gain * float(np.cos(4 * np.pi * opd / wavelength_nm))

    def signal_strength(self, wavelength_nm: float) -> float:
        """Relative fringe amplitude B (diffraction efficiency), in (0, 1]."""
        opd = self.mark_index * self.mark_depth_nm
        return 0.1 + 0.9 * float(np.sin(2 * np.pi * opd / wavelength_nm) ** 2)

    # --- signal -------------------------------------------------------------
    def signal(self, scan_nm, mark_x_nm, wavelength_nm, mark_asymmetry=0.0, rng=None):
        """Detected intensity while scanning the spot over the mark."""
        s = np.asarray(scan_nm, dtype=float)
        b = self.signal_strength(wavelength_nm)
        delta = mark_asymmetry * self.sensitivity(wavelength_nm)
        i = 1.0 + b * np.cos(2 * np.pi * (s - mark_x_nm - delta) / self.period_nm)
        if rng is not None and self.noise_nm > 0:
            # detector noise sized so the fitted position has std noise_nm / b
            sigma_i = self.noise_nm * 2 * np.pi / self.period_nm * np.sqrt(s.size / 2.0)
            i = i + sigma_i * rng.standard_normal(s.shape)
        return i

    def fit_phase(self, scan_nm, intensity, x_expected_nm: float = 0.0) -> tuple[float, float]:
        """Least-squares fit of ``A + c cos(ks) + d sin(ks)``; returns (x, B).

        The fringe phase is unwrapped to the solution closest to ``x_expected_nm``.
        """
        s = np.asarray(scan_nm, dtype=float)
        k = 2 * np.pi / self.period_nm
        G = np.column_stack([np.ones_like(s), np.cos(k * s), np.sin(k * s)])
        (a, c, d), *_ = np.linalg.lstsq(G, intensity, rcond=None)
        x0 = np.arctan2(d, c) / k
        m = np.round((x_expected_nm - x0) / self.period_nm)
        return float(x0 + m * self.period_nm), float(np.hypot(c, d))

    def measure(
        self,
        mark_x_nm: float,
        rng: np.random.Generator | None = None,
        mark_asymmetry: float = 0.0,
        x_expected_nm: float | None = None,
    ) -> dict:
        """Scan across the mark in every colour; return ``{wavelength: x_nm}``.

        ``x_expected_nm`` (default: ``mark_x_nm`` rounded to 1 um, i.e. coarse
        pre-alignment) centres the scan and resolves the fringe ambiguity.
        The fitted fringe amplitudes are stored in ``self.last_strengths``.
        """
        if x_expected_nm is None:
            x_expected_nm = float(np.round(mark_x_nm, -3))
        s = x_expected_nm + np.linspace(
            -self.scan_length_nm / 2, self.scan_length_nm / 2, self.n_samples, endpoint=False
        )
        out, strengths = {}, {}
        for w in self.wavelengths_nm:
            i = self.signal(s, mark_x_nm, w, mark_asymmetry, rng)
            out[w], strengths[w] = self.fit_phase(s, i, x_expected_nm)
        self.last_strengths = strengths
        return out

    def multi_color_estimate(self, per_color: dict, method: str = "asymmetry_fit") -> float:
        """Combine per-colour positions into one asymmetry-robust estimate.

        ``"asymmetry_fit"`` solves the weighted least-squares problem
        ``x_l = x + a * sensitivity(l)`` for the true position ``x`` and the
        asymmetry ``a`` (weights ``B_l^2``, i.e. inverse noise variance).
        ``"weighted_mean"`` is the plain ``B^2``-weighted average (no asymmetry
        correction) and ``"best"`` picks the strongest-signal colour.
        """
        w = np.array(list(per_color.keys()), dtype=float)
        x = np.array(list(per_color.values()), dtype=float)
        b = np.array([self.signal_strength(wi) for wi in w])
        if method == "best":
            return float(x[np.argmax(b)])
        if method == "weighted_mean" or len(w) < 2:
            return float(np.sum(b**2 * x) / np.sum(b**2))
        if method != "asymmetry_fit":
            raise ValueError(f"unknown method {method!r}")
        s = np.array([self.sensitivity(wi) for wi in w])
        G = np.column_stack([np.ones_like(s), s]) * b[:, None]
        sol, *_ = np.linalg.lstsq(G, x * b, rcond=None)
        return float(sol[0])

    def measure_wafer(
        self,
        deformation: WaferDeformation,
        xy_mm,
        rng: np.random.Generator,
        mark_asymmetry: float = 0.0,
        method: str = "asymmetry_fit",
    ) -> np.ndarray:
        """Measure (dx, dy) in nm at the mark positions ``xy_mm`` (N, 2).

        The true mark displacement comes from ``deformation`` (incl. its random
        placement noise); each axis is read by a separate x / y grating mark.
        """
        xy = _as_xy(xy_mm)
        tdx, tdy = deformation.displacement(xy[:, 0], xy[:, 1], rng)
        a = np.broadcast_to(np.asarray(mark_asymmetry, dtype=float), (xy.shape[0],))
        out = np.empty((xy.shape[0], 2))
        for n in range(xy.shape[0]):
            for ax, t in enumerate((tdx[n], tdy[n])):
                pc = self.measure(float(t), rng, float(a[n]))
                out[n, ax] = self.multi_color_estimate(pc, method)
        return out


# =============================================================================
# Wafer grid models
# =============================================================================
def _linear_eval(p: dict, x, y):
    dx = p["Tx"] + p["Mx"] * x - p["Rx"] * y
    dy = p["Ty"] + p["My"] * y + p["Ry"] * x
    return dx, dy


def fit_linear_wafer_model(xy_mm, dxy_nm) -> dict:
    """6-parameter linear wafer-grid least-squares fit.

    ``dx = Tx + Mx*x - Rx*y``, ``dy = Ty + My*y + Ry*x`` (x, y in mm; Tx, Ty in
    nm; Mx, My in ppm; Rx, Ry in urad).  The returned dict also contains the
    derived ``mag_ppm``, ``mag_asym_ppm``, ``rot_urad`` and ``nonorth_urad``.
    """
    xy, d = _as_xy(xy_mm), _as_xy(dxy_nm)
    x, y = xy[:, 0], xy[:, 1]
    one = np.ones_like(x)
    (tx, mx, rx), *_ = np.linalg.lstsq(np.column_stack([one, x, -y]), d[:, 0], rcond=None)
    (ty, my, ry), *_ = np.linalg.lstsq(np.column_stack([one, y, x]), d[:, 1], rcond=None)
    p = {"Tx": tx, "Ty": ty, "Mx": mx, "My": my, "Rx": rx, "Ry": ry}
    p = {k: float(v) for k, v in p.items()}
    p.update(linear_params_to_components(p))
    return p


def _poly_terms(order: int):
    return [(i, n - i) for n in range(order + 1) for i in range(n, -1, -1)]


def _poly_matrix(xy: np.ndarray, order: int, scale: float) -> np.ndarray:
    u, v = xy[:, 0] / scale, xy[:, 1] / scale
    return np.column_stack([u**i * v**j for i, j in _poly_terms(order)])


def fit_high_order(xy_mm, dxy_nm, order: int = 3, scale_mm: float = WAFER_RADIUS_MM) -> dict:
    """Full 2D polynomial wafer model up to ``order`` (per axis, all monomials).

    Coefficients (nm) multiply ``(x/scale)^i (y/scale)^j``.  Order 3 has 10
    terms per axis (20 parameters).  Needs at least that many marks.
    """
    xy, d = _as_xy(xy_mm), _as_xy(dxy_nm)
    G = _poly_matrix(xy, order, scale_mm)
    cx, *_ = np.linalg.lstsq(G, d[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(G, d[:, 1], rcond=None)
    return {"order": order, "scale_mm": scale_mm, "terms": _poly_terms(order), "cx": cx, "cy": cy}


def apply_model(params: dict, xy_mm) -> np.ndarray:
    """Evaluate a linear or high-order model at ``xy_mm``; returns (N, 2) nm."""
    xy = _as_xy(xy_mm)
    if "order" in params:
        G = _poly_matrix(xy, params["order"], params["scale_mm"])
        return np.column_stack([G @ params["cx"], G @ params["cy"]])
    dx, dy = _linear_eval(params, xy[:, 0], xy[:, 1])
    return np.column_stack([dx, dy])


def residuals(params: dict, xy_mm, dxy_nm) -> np.ndarray:
    """Measured minus modelled displacement, (N, 2) nm."""
    return _as_xy(dxy_nm) - apply_model(params, xy_mm)


def alignment_strategy(
    n_marks: int,
    fields: np.ndarray | None = None,
    mark_offset_mm=(0.0, 0.0),
) -> np.ndarray:
    """Choose ``n_marks`` alignment-mark positions (N, 2) mm across the wafer.

    Fields are chosen by greedy farthest-point sampling starting from the
    outermost field, which spreads marks evenly and favours the edge (where
    high-order terms are largest).  The mark sits at ``field centre + offset``.
    """
    f = field_centers() if fields is None else _as_xy(fields)
    n_marks = int(min(n_marks, len(f)))
    r = np.hypot(f[:, 0], f[:, 1])
    chosen = [int(np.argmax(r))]
    dmin = np.hypot(*(f - f[chosen[0]]).T)
    while len(chosen) < n_marks:
        k = int(np.argmax(dmin))
        chosen.append(k)
        dmin = np.minimum(dmin, np.hypot(*(f - f[k]).T))
    return f[chosen] + np.asarray(mark_offset_mm, dtype=float)


def evaluate_alignment_strategy(
    deformation: WaferDeformation,
    n_marks: int,
    order: int = 1,
    sensor: AlignmentSensor | None = None,
    rng: np.random.Generator | None = None,
    mark_asymmetry: float = 0.0,
) -> dict:
    """Align with ``n_marks`` and an ``order`` model; report the overlay residual.

    The model fitted to the alignment measurements is compared with the true
    (systematic) grid at a dense set of points (field corners and centres).
    Returns ``{"model", "marks", "residual" (N,2), "stats"}``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    sensor = AlignmentSensor() if sensor is None else sensor
    marks = alignment_strategy(n_marks)
    meas = sensor.measure_wafer(deformation, marks, rng, mark_asymmetry)
    model = fit_linear_wafer_model(marks, meas) if order <= 1 else fit_high_order(marks, meas, order)
    fc = field_centers()
    pts = [fc]
    for sx in (-0.45, 0.45):
        for sy in (-0.45, 0.45):
            pts.append(fc + [sx * FIELD_WIDTH_MM, sy * FIELD_HEIGHT_MM])
    pts = np.vstack(pts)
    true = np.column_stack(deformation.systematic(pts[:, 0], pts[:, 1]))
    res = true - apply_model(model, pts)
    return {"model": model, "marks": marks, "residual": res, "stats": overlay_stats(res[:, 0], res[:, 1])}


# =============================================================================
# Level sensor and focus control
# =============================================================================
class LevelSensor:
    """Grazing-incidence optical-triangulation level sensor.

    A projected spot hits the wafer at angle of incidence ``theta`` (from the
    normal).  A height change ``h`` shifts the reflected spot on the detector
    by ``2 h sin(theta)`` (times detection magnification); the sensor converts
    that back to height.  The spot averages the topography over its
    ``spot_size_mm`` square footprint.

    Process dependence: light partly reflects from buried interfaces, giving an
    apparent height offset ``stack_offset_nm`` (stack-dependent, may be a
    function of (x, y)).  A broadband source averages thin-film interference
    and reduces it (factor 0.3 vs 1.0 for ``"narrowband"``).
    """

    def __init__(
        self,
        spot_size_mm: float = 2.5,
        wavelength_band: str = "broadband",
        noise_nm: float = 2.0,
        theta_deg: float = 80.0,
        magnification: float = 1.0,
        n_sub: int = 5,
    ):
        self.spot_size_mm = spot_size_mm
        self.wavelength_band = wavelength_band
        self.noise_nm = noise_nm
        self.theta = np.deg2rad(theta_deg)
        self.magnification = magnification
        self.n_sub = n_sub

    @property
    def process_factor(self) -> float:
        return 0.3 if self.wavelength_band == "broadband" else 1.0

    def spot_shift_nm(self, height_nm):
        """Lateral detector-spot displacement for a given surface height."""
        return 2.0 * np.asarray(height_nm) * np.sin(self.theta) * self.magnification

    def scan(self, height_map_fn, xy_mm, rng: np.random.Generator | None = None, stack_offset_nm=0.0) -> np.ndarray:
        """Measured heights (nm) at the spot centres ``xy_mm`` (N, 2).

        ``height_map_fn(x_mm, y_mm) -> z_nm`` is the true topography (array in,
        array out).  ``stack_offset_nm`` is a scalar or a function of (x, y).
        """
        xy = _as_xy(xy_mm)
        o = (np.arange(self.n_sub) - (self.n_sub - 1) / 2.0) / self.n_sub * self.spot_size_mm
        ox, oy = np.meshgrid(o, o)
        X = xy[:, 0][:, None] + ox.ravel()[None, :]
        Y = xy[:, 1][:, None] + oy.ravel()[None, :]
        h = np.mean(height_map_fn(X, Y), axis=1)
        off = stack_offset_nm(xy[:, 0], xy[:, 1]) if callable(stack_offset_nm) else stack_offset_nm
        h = h + self.process_factor * np.asarray(off, dtype=float)
        shift = self.spot_shift_nm(h)
        if rng is not None and self.noise_nm > 0:
            shift = shift + self.spot_shift_nm(self.noise_nm) * rng.standard_normal(shift.shape)
        return shift / (2.0 * np.sin(self.theta) * self.magnification)


def _slit_mask(xy, center, slit_width_mm, slit_height_mm):
    return (np.abs(xy[:, 0] - center[0]) <= slit_width_mm / 2.0 + 1e-9) & (
        np.abs(xy[:, 1] - center[1]) <= slit_height_mm / 2.0 + 1e-9
    )


def focus_correction(
    height_samples,
    field_center,
    slit_width_mm: float = FIELD_WIDTH_MM,
    slit_height_mm: float = SLIT_HEIGHT_MM,
) -> tuple[float, float, float]:
    """Least-squares plane through the height samples inside the exposure slit.

    ``height_samples = (xy_mm (N, 2), z_nm (N,))``; ``field_center`` is the
    slit centre (x_mm, y_mm).  Returns ``(z_nm, rx_urad, ry_urad)`` with the
    plane ``z(x, y) = z + ry*(x - xc) + rx*(y - yc)``; i.e. ``rx = dz/dy``
    (tilt about the x axis) and ``ry = dz/dx`` (tilt about y), nm/mm == urad.
    If the samples do not span both directions the corresponding tilt is 0.
    """
    xy, z = _as_xy(height_samples[0]), np.asarray(height_samples[1], dtype=float).ravel()
    m = _slit_mask(xy, field_center, slit_width_mm, slit_height_mm)
    if not np.any(m):
        raise ValueError("no height samples inside the slit")
    dx = xy[m, 0] - field_center[0]
    dy = xy[m, 1] - field_center[1]
    cols = [np.ones_like(dx)]
    use_x = np.ptp(dx) > 1e-9
    use_y = np.ptp(dy) > 1e-9
    if use_x:
        cols.append(dx)
    if use_y:
        cols.append(dy)
    sol, *_ = np.linalg.lstsq(np.column_stack(cols), z[m], rcond=None)
    z0 = sol[0]
    ry = sol[1] if use_x else 0.0
    rx = sol[-1] if use_y else 0.0
    return float(z0), float(rx), float(ry)


def focus_residual(
    height_samples,
    field_center,
    field_height_mm: float = FIELD_HEIGHT_MM,
    slit_width_mm: float = FIELD_WIDTH_MM,
    slit_height_mm: float = SLIT_HEIGHT_MM,
    step_mm: float = 1.0,
    true_height_fn=None,
) -> dict:
    """Defocus that remains after per-slit (z, Rx, Ry) leveling during a scan.

    The slit is scanned in y over the field in ``step_mm`` steps; at every
    position the stage follows the plane from :func:`focus_correction`.  A
    point is exposed while it is inside the slit, so its defocus is the
    moving average (MA) of ``z_true - plane`` over the slit positions that
    contain it; MSD is the corresponding moving standard deviation.

    ``true_height_fn`` (optional) gives the real surface used to evaluate the
    error; otherwise the samples themselves are used.  Returns a dict with
    ``"xy"``, ``"ma"``, ``"msd"`` (per in-field sample), the slit set-points
    ``"y_slit"``, ``"z"``, ``"rx"``, ``"ry"`` and summary ``"ma_3sigma"``,
    ``"msd_mean"``.
    """
    xy, z = _as_xy(height_samples[0]), np.asarray(height_samples[1], dtype=float).ravel()
    xc, yc = field_center
    infield = (np.abs(xy[:, 0] - xc) <= slit_width_mm / 2.0 + 1e-9) & (
        np.abs(xy[:, 1] - yc) <= field_height_mm / 2.0 + 1e-9
    )
    pts = xy[infield]
    zt = true_height_fn(pts[:, 0], pts[:, 1]) if true_height_fn is not None else z[infield]
    half = (field_height_mm - slit_height_mm) / 2.0
    ys = yc + np.arange(-half - slit_height_mm / 2.0, half + slit_height_mm / 2.0 + 1e-9, step_mm)
    s1 = np.zeros(len(pts))
    s2 = np.zeros(len(pts))
    cnt = np.zeros(len(pts))
    Z, RX, RY, YS = [], [], [], []
    for y in ys:
        try:
            z0, rx, ry = focus_correction((xy, z), (xc, y), slit_width_mm, slit_height_mm)
        except ValueError:
            continue
        Z.append(z0), RX.append(rx), RY.append(ry), YS.append(y)
        m = _slit_mask(pts, (xc, y), slit_width_mm, slit_height_mm)
        e = zt[m] - (z0 + ry * (pts[m, 0] - xc) + rx * (pts[m, 1] - y))
        s1[m] += e
        s2[m] += e**2
        cnt[m] += 1
    cnt = np.maximum(cnt, 1)
    ma = s1 / cnt
    msd = np.sqrt(np.maximum(s2 / cnt - ma**2, 0.0))
    return {
        "xy": pts,
        "ma": ma,
        "msd": msd,
        "y_slit": np.array(YS),
        "z": np.array(Z),
        "rx": np.array(RX),
        "ry": np.array(RY),
        "ma_3sigma": float(3 * np.std(ma)) if len(ma) else 0.0,
        "ma_max": float(np.max(np.abs(ma))) if len(ma) else 0.0,
        "msd_mean": float(np.mean(msd)) if len(msd) else 0.0,
    }


# =============================================================================
# Overlay
# =============================================================================
def overlay_stats(dx, dy) -> dict:
    """Per-axis overlay statistics.

    Returns ``{"x": {"mean", "sigma", "3sigma", "m+3s"}, "y": {...}}`` where
    ``m+3s = |mean| + 3*sigma`` (sigma = population std, ddof=0).
    """
    out = {}
    for name, d in (("x", dx), ("y", dy)):
        d = np.asarray(d, dtype=float).ravel()
        m, s = float(np.mean(d)), float(np.std(d))
        out[name] = {"mean": m, "sigma": s, "3sigma": 3 * s, "m+3s": abs(m) + 3 * s}
    return out


# k-parameter intrafield polynomial (odd k -> dx, even k -> dy)
K_TERMS = {
    1: ("x", 0, 0), 2: ("y", 0, 0),
    3: ("x", 1, 0), 4: ("y", 0, 1),
    5: ("x", 0, 1), 6: ("y", 1, 0),
    7: ("x", 2, 0), 8: ("y", 0, 2),
    9: ("x", 1, 1), 10: ("y", 1, 1),
    11: ("x", 0, 2), 12: ("y", 2, 0),
    13: ("x", 3, 0), 14: ("y", 0, 3),
    15: ("x", 2, 1), 16: ("y", 1, 2),
    17: ("x", 1, 2), 18: ("y", 2, 1),
    19: ("x", 0, 3), 20: ("y", 3, 0),
}
"""``k -> (axis, i, j)``: term ``k * xf^i * yf^j`` (field coords in mm) in dx or dy.

k1/k2 translation, k3/k4 magnification, k5/k6 rotation slopes (dx/dy and
dy/dx; pure rotation gives k5 = -k6), k7..k12 second order (k9/k10
trapezoid / keystone xy terms), k13..k20 third order (k13/k14 3rd-order
magnification, ...)."""


class Overlay:
    """Overlay = printed position - target position.

    Construct with field-local mark coordinates and per-mark overlay, or use
    :meth:`from_positions`.  Provides wafer-level and intrafield models.
    """

    def __init__(self, xy_wafer_mm, dxy_nm, xy_field_mm=None):
        self.xy = _as_xy(xy_wafer_mm)
        self.dxy = _as_xy(dxy_nm)
        self.xy_field = None if xy_field_mm is None else _as_xy(xy_field_mm)

    @staticmethod
    def compute(printed_nm, target_nm) -> np.ndarray:
        """Overlay vectors (N, 2) = printed - target."""
        return _as_xy(printed_nm) - _as_xy(target_nm)

    @classmethod
    def from_positions(cls, xy_wafer_mm, printed_nm, target_nm, xy_field_mm=None):
        return cls(xy_wafer_mm, cls.compute(printed_nm, target_nm), xy_field_mm)

    def stats(self) -> dict:
        return overlay_stats(self.dxy[:, 0], self.dxy[:, 1])

    def wafer_model(self, order: int = 1) -> dict:
        return fit_linear_wafer_model(self.xy, self.dxy) if order <= 1 else fit_high_order(self.xy, self.dxy, order)

    @staticmethod
    def intrafield_model(xy_field_mm, dxy_nm, n_params: int = 6) -> dict:
        """Fit k-parameters ``k1..k{n_params}`` (n_params in 4, 6, 12, 20).

        ``n_params == 4`` is the symmetric model (Tx, Ty, mag, rot) returned
        as k1..k6 with k3 == k4 (mag) and k6 == -k5 (rot).  Returns a dict
        ``{"k1": ..., ...}`` plus ``"residual"`` (N, 2).
        """
        xy, d = _as_xy(xy_field_mm), _as_xy(dxy_nm)
        x, y = xy[:, 0], xy[:, 1]
        if n_params == 4:
            one, zero = np.ones_like(x), np.zeros_like(x)
            G = np.vstack([
                np.column_stack([one, zero, x, -y]),
                np.column_stack([zero, one, y, x]),
            ])
            (tx, ty, mag, rot), *_ = np.linalg.lstsq(G, np.concatenate([d[:, 0], d[:, 1]]), rcond=None)
            k = {"k1": tx, "k2": ty, "k3": mag, "k4": mag, "k5": -rot, "k6": rot}
        elif n_params in (6, 12, 20):
            k = {}
            for ax, col in (("x", 0), ("y", 1)):
                ks = [n for n in range(1, n_params + 1) if K_TERMS[n][0] == ax]
                G = np.column_stack([x ** K_TERMS[n][1] * y ** K_TERMS[n][2] for n in ks])
                c, *_ = np.linalg.lstsq(G, d[:, col], rcond=None)
                k.update({f"k{n}": float(v) for n, v in zip(ks, c)})
        else:
            raise ValueError("n_params must be 4, 6, 12 or 20")
        k = {kk: float(v) for kk, v in k.items()}
        k["residual"] = d - Overlay.apply_intrafield(k, xy)
        return k

    @staticmethod
    def apply_intrafield(k: dict, xy_field_mm) -> np.ndarray:
        xy = _as_xy(xy_field_mm)
        out = np.zeros_like(xy)
        for n, (ax, i, j) in K_TERMS.items():
            c = k.get(f"k{n}", 0.0)
            if c:
                out[:, 0 if ax == "x" else 1] += c * xy[:, 0] ** i * xy[:, 1] ** j
        return out


# =============================================================================
# Run-to-run control
# =============================================================================
@dataclass
class CorrectionLoop:
    """EWMA run-to-run (APC) controller for the 6 linear wafer parameters.

    After each lot the overlay metrology gives ``measured`` linear params of
    the *residual* overlay (with correction ``c`` applied).  The disturbance
    estimate is ``d_hat = lam*(measured + c) + (1 - lam)*d_hat`` and the next
    correction is ``c = d_hat`` (applied with opposite sign by the scanner,
    i.e. printed error = disturbance - c).
    """

    lam: float = 0.3
    names: tuple = LINEAR_PARAM_NAMES
    correction: dict = field(default_factory=dict)
    history: list = field(default_factory=list)

    def __post_init__(self):
        for n in self.names:
            self.correction.setdefault(n, 0.0)

    def update(self, measured: dict) -> dict:
        """Feed the measured residual params of the last lot; return new correction."""
        for n in self.names:
            d_obs = measured[n] + self.correction[n]
            self.correction[n] = self.lam * d_obs + (1 - self.lam) * self.correction[n]
        self.history.append(dict(measured))
        return dict(self.correction)

    def simulate(
        self,
        disturbance: dict,
        n_lots: int,
        rng: np.random.Generator,
        lot_noise: dict | float = 0.0,
        drift_per_lot: dict | None = None,
        xy_marks_mm=None,
        metrology_noise_nm: float = 0.0,
    ) -> dict:
        """Run ``n_lots`` lots against a (drifting, noisy) linear disturbance.

        Each lot: true disturbance + lot noise; printed overlay at the overlay
        marks is ``grid(disturbance - correction)`` + metrology noise; the
        linear model is fitted and fed to :meth:`update`.  Returns arrays
        ``"residual_params"`` (n_lots, 6) and ``"overlay_m3s"`` (n_lots, 2).
        """
        marks = field_centers() if xy_marks_mm is None else _as_xy(xy_marks_mm)
        drift = drift_per_lot or {}
        res_p, m3s = [], []
        for lot in range(n_lots):
            true = {}
            for n in self.names:
                sig = lot_noise.get(n, 0.0) if isinstance(lot_noise, dict) else lot_noise
                true[n] = disturbance.get(n, 0.0) + drift.get(n, 0.0) * lot + sig * rng.standard_normal()
            err = {n: true[n] - self.correction[n] for n in self.names}
            ovl = apply_model(err, marks)
            if metrology_noise_nm:
                ovl = ovl + metrology_noise_nm * rng.standard_normal(ovl.shape)
            meas = fit_linear_wafer_model(marks, ovl)
            st = overlay_stats(ovl[:, 0], ovl[:, 1])
            res_p.append([err[n] for n in self.names])
            m3s.append([st["x"]["m+3s"], st["y"]["m+3s"]])
            self.update(meas)
        return {"residual_params": np.array(res_p), "overlay_m3s": np.array(m3s)}
