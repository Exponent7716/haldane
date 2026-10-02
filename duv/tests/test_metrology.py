"""Tests for duv.metrology."""

import numpy as np
import pytest

from duv.metrology import (
    AlignmentSensor,
    CorrectionLoop,
    LevelSensor,
    Overlay,
    WaferDeformation,
    alignment_strategy,
    apply_model,
    evaluate_alignment_strategy,
    field_centers,
    fit_high_order,
    fit_linear_wafer_model,
    focus_correction,
    focus_residual,
    linear_components_to_params,
    overlay_stats,
    residuals,
)


def test_field_centers_inside_wafer():
    fc = field_centers()
    assert 60 < len(fc) < 120
    assert np.all(np.hypot(fc[:, 0], fc[:, 1]) < 150)


def test_alignment_sensor_recovers_position_within_noise():
    rng = np.random.default_rng(1)
    s = AlignmentSensor(noise_nm=0.2)
    errs = []
    for x in rng.uniform(-3000, 3000, 30):
        pc = s.measure(x, rng)
        errs.append(s.multi_color_estimate(pc, "weighted_mean") - x)
        # strongest colour individually within a few sigma
        assert abs(pc[633.0] - x) < 5 * 0.2 / s.signal_strength(633)
    errs = np.array(errs)
    assert abs(np.mean(errs)) < 0.2
    assert 0.02 < np.std(errs) < 0.4


def test_alignment_noise_free_is_exact_and_unwraps():
    s = AlignmentSensor(noise_nm=0.0)
    for x in (0.0, 1234.5, -7777.0, 15000.3):
        pc = s.measure(x, None)
        for v in pc.values():
            assert v == pytest.approx(x, abs=1e-6)


def test_asymmetry_colour_dependent_and_multicolor_suppresses():
    rng = np.random.default_rng(2)
    s = AlignmentSensor(noise_nm=0.05)
    x, asym = 500.0, 5.0
    pc = s.measure(x, rng, mark_asymmetry=asym)
    single = np.array([v - x for v in pc.values()])
    # colour-dependent apparent shift of several nm, differing across colours
    assert np.ptp(single) > 3.0
    assert np.max(np.abs(single)) > 2.0
    for w, v in pc.items():
        assert v - x == pytest.approx(asym * s.sensitivity(w), abs=1.0)
    est = s.multi_color_estimate(pc, "asymmetry_fit")
    assert abs(est - x) < 0.5
    assert abs(est - x) < 0.2 * np.max(np.abs(single))


def test_linear_fit_exact_without_noise():
    true = dict(tx=12.0, ty=-7.5, mag_ppm=0.8, mag_asym_ppm=-0.2, rot_urad=0.35, nonorth_urad=0.1)
    d = WaferDeformation(**true)
    xy = field_centers()
    dxy = np.column_stack(d.displacement(xy[:, 0], xy[:, 1]))
    p = fit_linear_wafer_model(xy, dxy)
    for k, v in true.items():
        assert p[k] == pytest.approx(v, abs=1e-9)
    ref = linear_components_to_params(**true)
    for k, v in ref.items():
        assert p[k] == pytest.approx(v, abs=1e-9)
    assert np.max(np.abs(residuals(p, xy, dxy))) < 1e-9


def test_high_order_reduces_residual_on_cubic_wafer():
    d = WaferDeformation(tx=5, mag_ppm=0.3, third_order_nm=4.0, seed=3)
    xy = field_centers()
    dxy = np.column_stack(d.displacement(xy[:, 0], xy[:, 1]))
    r1 = residuals(fit_linear_wafer_model(xy, dxy), xy, dxy)
    p3 = fit_high_order(xy, dxy, order=3)
    r3 = residuals(p3, xy, dxy)
    assert np.std(r1) > 1.0
    assert np.max(np.abs(r3)) < 1e-8
    # order-1 polynomial reproduces the linear model
    p1 = fit_high_order(xy, dxy, order=1)
    assert np.allclose(apply_model(p1, xy), apply_model(fit_linear_wafer_model(xy, dxy), xy))


def test_alignment_strategy_more_marks_and_higher_order_help():
    marks = alignment_strategy(12)
    assert marks.shape == (12, 2)
    assert len(np.unique(marks, axis=0)) == 12
    d = WaferDeformation(tx=10, mag_ppm=0.5, rot_urad=0.2, third_order_nm=3.0, noise_nm=0.5, seed=4)
    sensor = AlignmentSensor(noise_nm=0.3)

    def m3s(n, order, seed):
        st = evaluate_alignment_strategy(d, n, order, sensor, np.random.default_rng(seed))["stats"]
        return max(st["x"]["m+3s"], st["y"]["m+3s"])

    lin = m3s(16, 1, 0)
    ho_few = np.mean([m3s(12, 3, s) for s in range(3)])
    ho_many = np.mean([m3s(40, 3, s) for s in range(3)])
    assert ho_few < lin
    assert ho_many < ho_few


def test_level_sensor_and_plane_fit_recover_tilt():
    rng = np.random.default_rng(5)
    z0, rx, ry = 40.0, 3.0, -2.0  # nm, urad (nm/mm)

    def h(x, y):
        return z0 + ry * x + rx * y

    ls = LevelSensor(noise_nm=0.5)
    xs, ys = np.meshgrid(np.arange(-12, 12.1, 2.0), np.arange(-4, 4.1, 1.0))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    z = ls.scan(h, xy, rng)
    assert np.std(z - h(xy[:, 0], xy[:, 1])) < 1.0
    zf, rxf, ryf = focus_correction((xy, z), (0.0, 0.0), 26.0, 8.0)
    assert zf == pytest.approx(z0, abs=0.5)
    assert rxf == pytest.approx(rx, abs=0.3)
    assert ryf == pytest.approx(ry, abs=0.1)


def test_level_sensor_process_offset_broadband_smaller():
    flat = lambda x, y: np.zeros_like(x)
    xy = np.zeros((1, 2))
    nb = LevelSensor(wavelength_band="narrowband", noise_nm=0).scan(flat, xy, None, 20.0)
    bb = LevelSensor(wavelength_band="broadband", noise_nm=0).scan(flat, xy, None, 20.0)
    assert nb[0] == pytest.approx(20.0)
    assert abs(bb[0]) < abs(nb[0])


def test_focus_residual_zero_for_plane_nonzero_for_bump():
    xs, ys = np.meshgrid(np.arange(-13, 13.1, 1.0), np.arange(-16.5, 16.6, 0.5))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    plane = 10 + 2 * xy[:, 0] - 1.5 * xy[:, 1]
    r = focus_residual((xy, plane), (0.0, 0.0))
    assert r["ma_max"] < 1e-9 and r["msd_mean"] < 1e-9
    assert np.allclose(r["ry"], 2.0) and np.allclose(r["rx"], -1.5)
    bump = plane + 30 * np.exp(-((xy[:, 0] - 3) ** 2 + xy[:, 1] ** 2) / 4.0)
    r2 = focus_residual((xy, bump), (0.0, 0.0))
    assert r2["ma_max"] > 3.0 and r2["msd_mean"] > 0.1


def test_overlay_stats_synthetic():
    rng = np.random.default_rng(6)
    dx = 1.5 + 2.0 * rng.standard_normal(20000)
    dy = -0.7 + 1.0 * rng.standard_normal(20000)
    st = overlay_stats(dx, dy)
    assert st["x"]["mean"] == pytest.approx(1.5, abs=0.05)
    assert st["x"]["3sigma"] == pytest.approx(6.0, rel=0.03)
    assert st["y"]["m+3s"] == pytest.approx(0.7 + 3.0, rel=0.03)
    ex = overlay_stats([1, -1, 1, -1], [2, 2, 2, 2])
    assert ex["x"]["3sigma"] == pytest.approx(3.0)
    assert ex["y"]["m+3s"] == pytest.approx(2.0)
    ovl = Overlay.compute([[3, 4]], [[1, 1]])
    assert np.allclose(ovl, [[2, 3]])


def test_intrafield_model_recovers_k_parameters():
    xs, ys = np.meshgrid(np.linspace(-13, 13, 7), np.linspace(-16.5, 16.5, 7))
    xy = np.column_stack([xs.ravel(), ys.ravel()])
    k_true = {"k1": 1.0, "k2": -2.0, "k3": 0.1, "k4": 0.05, "k5": -0.08, "k6": 0.07,
              "k9": 0.004, "k13": 1e-4, "k20": -2e-4}
    d = Overlay.apply_intrafield(k_true, xy)
    k20 = Overlay.intrafield_model(xy, d, 20)
    for k, v in k_true.items():
        assert k20[k] == pytest.approx(v, abs=1e-9)
    assert np.max(np.abs(k20["residual"])) < 1e-9
    k6 = Overlay.intrafield_model(xy, d, 6)
    assert np.std(k6["residual"]) > 1e-3
    # symmetric 4-param model on pure mag + rotation
    d4 = Overlay.apply_intrafield({"k1": 0.5, "k3": 0.2, "k4": 0.2, "k5": -0.1, "k6": 0.1}, xy)
    k4 = Overlay.intrafield_model(xy, d4, 4)
    assert k4["k3"] == pytest.approx(0.2) and k4["k6"] == pytest.approx(0.1)


def test_ewma_loop_converges():
    rng = np.random.default_rng(7)
    loop = CorrectionLoop(lam=0.4)
    dist = {"Tx": 8.0, "Ty": -5.0, "Mx": 0.6, "My": 0.4, "Rx": 0.2, "Ry": -0.1}
    out = loop.simulate(dist, 25, rng, lot_noise=0.0, metrology_noise_nm=0.2)
    rp = out["residual_params"]
    assert np.max(np.abs(rp[-1])) < 0.05 * np.max(np.abs(rp[0]))
    assert np.max(out["overlay_m3s"][-1]) < 0.1 * np.max(out["overlay_m3s"][0])
    assert all(np.max(out["overlay_m3s"][i + 1]) <= np.max(out["overlay_m3s"][i]) + 1.0 for i in range(10))
    for k, v in dist.items():
        assert loop.correction[k] == pytest.approx(v, abs=0.1 * abs(v) + 0.05)
