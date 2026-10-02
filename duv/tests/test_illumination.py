import numpy as np
import pytest

from duv.core import SourceMap
from duv.illumination import (
    DiffractiveOpticalElement,
    FlyEyeIntegrator,
    Illuminator,
    MirrorArray,
    REMABlades,
    UniformityCorrector,
    degree_of_polarization,
    freeform_source,
    intensity_in_preferred_state,
    make_source,
    pupil_metrics,
    scanned_dose_profile,
    slit_profile,
    uniformity,
)


def _polar(src):
    sx, sy = np.meshgrid(src.sigma, src.sigma)
    return np.hypot(sx, sy), np.degrees(np.arctan2(sy, sx))


def test_annular_zero_inside_and_outside():
    blur = 0.02
    src = make_source("annular", sigma_in=0.5, sigma_out=0.8, edge_blur=blur, n=101)
    r, _ = _polar(src)
    assert src.sigma[0] == -1 and src.sigma[-1] == 1 and src.intensity.shape == (101, 101)
    assert np.all(src.intensity[r < 0.5 - blur / 2] == 0)
    assert np.all(src.intensity[r > 0.8 + blur / 2] == 0)
    assert np.allclose(src.intensity[(r > 0.55) & (r < 0.75)], 1.0)


@pytest.mark.parametrize("kind", ["conventional", "annular", "quadrupole", "quasar", "c_quad"])
def test_symmetric_sources_have_zero_ellipticity_and_telecentricity(kind):
    m = pupil_metrics(make_source(kind, sigma_in=0.5, sigma_out=0.9))
    assert abs(m["ellipticity"]) < 1e-6
    assert m["telecentricity"] < 1e-9
    assert m["pole_balance"] < 1e-9
    assert 0 < m["fill_fraction"] < 1


def test_quasar_poles_on_diagonals_quadrupole_on_axes():
    for kind, centre in (("quasar", 45.0), ("quadrupole", 0.0)):
        src = make_source(kind, sigma_in=0.6, sigma_out=0.9, opening_angle_deg=30)
        r, ang = _polar(src)
        ring = (r > 0.65) & (r < 0.85)
        assert np.all(src.intensity[ring & (np.abs(ang - centre) < 10)] > 0.99)
        assert np.all(src.intensity[ring & (np.abs(ang - centre - 45) < 20)] < 1e-9)


def test_dipole_x_ellipticity_and_rotation():
    sx_ = make_source("dipole_x", 0.6, 0.9, opening_angle_deg=40)
    sy_ = make_source("dipole_y", 0.6, 0.9, opening_angle_deg=40)
    rot = make_source("dipole_x", 0.6, 0.9, opening_angle_deg=40, rotation_deg=90)
    assert pupil_metrics(sx_)["ellipticity"] > 0.9
    assert pupil_metrics(sy_)["ellipticity"] < -0.9
    np.testing.assert_allclose(rot.intensity, sy_.intensity, atol=1e-9)
    assert pupil_metrics(sx_)["sigma_center"] == pytest.approx(0.75, abs=0.02)


def test_metrics_detect_imbalance_and_telecentricity():
    src = make_source("quasar", 0.6, 0.9)
    I = src.intensity.copy()
    sx, sy = np.meshgrid(src.sigma, src.sigma)
    I[(sx > 0) & (sy > 0)] *= 1.2
    m = pupil_metrics(freeform_source(I))
    assert m["pole_balance"] > 0.05
    assert m["telecentricity_x"] > 0 and m["telecentricity_y"] > 0


def test_freeform_source_clips_and_normalises():
    a = np.full((51, 51), 2.0)
    a[0, 0] = -1
    src = freeform_source(a, polarization="TE")
    assert src.intensity.max() == pytest.approx(1.0)
    assert src.intensity[0, 0] == 0  # corner lies outside the unit pupil
    assert src.polarization == "TE"


def test_mirror_array_approximates_target():
    target = make_source("quasar", 0.6, 0.9, opening_angle_deg=40, n=81)
    beam = np.random.default_rng(1).uniform(0.5, 1.5, size=(64, 64))
    ma = MirrorArray.from_beam(beam, spot_sigma=0.03)
    fit = ma.fit_target(target)
    assert fit["correlation"] > 0.9
    m = pupil_metrics(ma.to_source(81))
    assert m["pole_balance"] < 0.05
    assert m["telecentricity"] < 0.02
    assert np.all(np.hypot(ma.positions[:, 0], ma.positions[:, 1]) <= 1.0)


def test_doe_source_has_zero_order_and_blur():
    src = DiffractiveOpticalElement("annular", sigma_in=0.6, sigma_out=0.9, zero_order=0.01).to_source(81)
    c = 40
    assert src.intensity[c, c] > 0.01  # undiffracted zero order in the centre
    assert pupil_metrics(src)["sigma_center"] == pytest.approx(0.75, abs=0.03)


def test_polarization_helpers():
    unpol = make_source("dipole_y", 0.6, 0.9)
    xpol = make_source("dipole_y", 0.6, 0.9, opening_angle_deg=30, polarization="X")
    ypol = make_source("dipole_y", 0.6, 0.9, opening_angle_deg=30, polarization="Y")
    te = make_source("annular", 0.6, 0.9, polarization="TE")
    assert degree_of_polarization(unpol) == 0.0
    assert degree_of_polarization(te) == pytest.approx(1.0)
    assert intensity_in_preferred_state(unpol) == 0.5
    assert intensity_in_preferred_state(xpol, "TE") > 0.9
    assert intensity_in_preferred_state(ypol, "TE") < 0.1
    assert intensity_in_preferred_state(te, "TE") == pytest.approx(1.0)


def test_flyeye_improves_uniformity_with_more_lenslets():
    x = np.linspace(-1, 1, 400)
    beam = np.exp(-0.5 * (x / 0.7) ** 2) * (1 + 0.3 * x) + 0.05 * np.sin(23 * x)
    u = [FlyEyeIntegrator(n).field_uniformity(beam)["output_uniformity"] for n in (1, 4, 16, 64)]
    assert u[0] == pytest.approx(uniformity(beam), rel=0.05)
    assert u[1] < u[0] and u[2] < u[1] and u[3] < u[2]
    assert u[3] < 0.02
    fe = FlyEyeIntegrator(16)
    assert fe.speckle_contrast(n_pulses=100) < fe.speckle_contrast(n_pulses=1)
    assert fe.homogenize(np.ones((20, 30))).shape == (fe.n_field, fe.n_field)


def test_slit_profile_shape():
    y = np.linspace(-8, 8, 1601)
    p = slit_profile(y, 8.0, 1.0)
    assert p.max() == 1.0
    assert slit_profile(np.array([4.0]), 8.0, 1.0)[0] == pytest.approx(0.5)
    assert np.all(p[np.abs(y) >= 4.5] == 0) and np.all(p[np.abs(y) <= 3.5] == 1)
    assert np.trapezoid(p, y) == pytest.approx(8.0, rel=1e-3)


def test_scanned_dose_flat_and_soft_edge_reduces_ripple():
    y = np.linspace(-10, 10, 801)
    kw = dict(scan_start_mm=-20, scan_end_mm=20, scan_speed_mm_s=700, rep_rate_hz=6000)
    soft = scanned_dose_profile(y, edge_width_mm=1.0, **kw)
    hard = scanned_dose_profile(y, edge_width_mm=0.0, **kw)
    assert soft.mean() == pytest.approx(1.0, rel=0.01)
    assert uniformity(soft) < uniformity(hard)
    assert uniformity(soft) < 1e-3


def test_rema_blades_confine_exposure():
    rema = REMABlades(x_min=-5, x_max=5, y_min=-6, y_max=6, slit_height_mm=4, slit_edge_mm=0.5)
    x = np.linspace(-8, 8, 81)
    y = np.linspace(-10, 10, 101)
    inst = rema.illuminated_region(x, y, scan_pos=5.0)
    assert np.all(inst[y > 6.1, :] == 0)  # upper blade masks beyond field edge
    assert np.all(inst[:, np.abs(x) > 5.1] == 0)
    assert inst[np.argmin(np.abs(y - 5)), 40] == pytest.approx(1.0)
    dose = rema.exposed_dose_map(x, y, n_steps=300)
    assert np.all(dose[np.abs(y) > 6.1] == 0)
    interior = dose[np.abs(y) < 5][:, np.abs(x) < 4.5]
    assert interior.mean() == pytest.approx(1.0, rel=0.02)


def test_uniformity_corrector_flattens_profile():
    x = np.linspace(-13, 13, 131)
    prof = 1.0 + 0.02 * np.cos(2 * np.pi * x / 26) + 0.01 * x / 13
    uc = UniformityCorrector(n_fingers=28)
    res = uc.correct(prof, x)
    assert res["uniformity_after"] < 0.25 * res["uniformity_before"]
    assert np.all(res["finger_positions_mm"] >= 0)
    assert np.all(res["finger_positions_mm"] <= uc.max_insertion_mm)
    assert 0 < res["light_loss"] < 0.05


@pytest.mark.parametrize("shaper", ["mirror_array", "doe", "ideal"])
def test_illuminator_run(shaper):
    ill = Illuminator(kind="quasar", sigma_in=0.6, sigma_out=0.9, polarization="TE", shaper=shaper)
    src, s = ill.run(n=61)
    assert isinstance(src, SourceMap) and src.intensity.shape == (61, 61)
    assert src.polarization == "TE"
    assert s["fit"]["correlation"] > 0.9
    assert s["flyeye_uniformity"] < s["beam_uniformity"]
    assert s["corrected_uniformity"] <= s["flyeye_uniformity"] + 1e-12
    assert s["pupil"]["sigma_center"] == pytest.approx(0.75, abs=0.05)
    assert s["dop"] == pytest.approx(1.0)
