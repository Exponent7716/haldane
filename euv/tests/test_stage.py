import numpy as np
import pytest

from euvsim.stage import (
    G0, MotionLimits, point_to_point, move_time, scan_profile, exposure_time_per_field,
    field_cycle_time, StagePlant, Controller, ScannerStages, Disturbances,
    simulate_scan_sync, moving_average_std, closed_loop_stable,
    Interferometer, Encoder, AlignmentSensor, LevelSensor, gas_refractive_index,
    generate_wafer_topography, slit_leveling_residual, depth_of_focus,
    default_focus_budget, abbe_error,
    GridModel, fit_overlay, alignment_mark_positions, mean_plus_3sigma,
    matched_machine_overlay_budget,
    pulses_per_point, dose_mj_cm2, open_loop_dose_error, slit_integrated_dose,
    apply_dose_control, synthetic_pulse_train,
    max_scan_speed, required_power_at_wafer, field_layout, wafer_timeline, throughput_wph,
)


# ------------------------------------------------------------ trajectory
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("dist", [26e-3, 1e-3, 1e-5])
def test_point_to_point_respects_limits(order, dist):
    lim = MotionLimits(v_max=0.6, a_max=7 * G0, j_max=3e4, s_max=3e7)
    tr = point_to_point(dist, lim, order=order, dt=1e-5)
    pk = tr.peaks()
    tol = 1 + 1e-6
    assert pk["vel"] <= lim.v_max * tol
    assert pk["acc"] <= lim.a_max * tol
    if order >= 3:
        assert pk["jerk"] <= lim.j_max * tol
    if order >= 4:
        assert pk["snap"] <= lim.s_max * tol
    assert tr.pos[-1] == pytest.approx(dist, rel=1e-9)
    assert abs(tr.vel[-1]) < 1e-12
    # sampled duration close to analytic (rounding up per segment)
    assert tr.duration == pytest.approx(move_time(dist, lim, order), abs=order * 2e-5 + 1e-5)


def test_scan_profile_constant_velocity_and_reticle_scaling():
    lim = MotionLimits()
    v = 0.4
    sp = scan_profile(v, lim, field_height_m=33e-3, slit_height_m=2e-3,
                      settle_time=5e-3, magnification=4.0)
    assert sp.exposure_time == pytest.approx(exposure_time_per_field(33e-3, 2e-3, v), rel=1e-3)
    m = sp.exposure_mask
    assert np.allclose(sp.wafer.vel[m], v)
    assert np.allclose(sp.wafer.acc[m], 0, atol=1e-6)
    pk = sp.wafer.peaks()
    assert pk["acc"] <= lim.a_max * (1 + 1e-9) and pk["snap"] <= lim.s_max * (1 + 1e-9)
    assert sp.reticle.peaks()["vel"] == pytest.approx(4 * v)
    # anamorphic high NA: half field, 8x in scan direction
    sph = scan_profile(0.4, MotionLimits(), field_height_m=16.5e-3, magnification=8.0)
    assert sph.reticle.peaks()["vel"] == pytest.approx(3.2)
    assert sph.exposure_time < sp.exposure_time


def test_field_cycle_time_components():
    fc = field_cycle_time(0.4, MotionLimits())
    assert fc["t_exposure"] == pytest.approx(35e-3 / 0.4)
    assert fc["total"] > fc["t_exposure"]
    assert 0.1 < fc["total"] < 0.3


# -------------------------------------------------------------- dynamics
def test_controller_stable():
    st = ScannerStages()
    cw, cr = st.controllers()
    assert closed_loop_stable(st.wafer_plant, cw, 5e-5)
    assert closed_loop_stable(st.reticle_plant, cr, 5e-5)
    assert StagePlant().k > 0


def test_moving_average_std():
    e = np.arange(10.0)
    ma, msd = moving_average_std(e, 3)
    assert np.allclose(ma, np.arange(1, 9))
    assert np.allclose(msd, np.sqrt(2 / 3))


@pytest.fixture(scope="module")
def sync_runs():
    sp = scan_profile(0.4, MotionLimits(), magnification=4.0)
    ff = [simulate_scan_sync(sp, np.random.default_rng(s), feedforward=True) for s in range(3)]
    nff = simulate_scan_sync(sp, np.random.default_rng(0), feedforward=False)
    return ff, nff


def test_ma_msd_plausible_and_feedforward_helps(sync_runs):
    ff, nff = sync_runs
    ma = np.mean([r.ma_max_nm for r in ff])
    msd = np.mean([r.msd_max_nm for r in ff])
    assert 0.05 < ma < 2.0          # spec MA ~ < 1 nm
    assert 0.05 < msd < 5.0         # spec MSD ~ few nm
    assert nff.ma_max_nm > 10 * ma
    assert nff.msd_max_nm > 5 * msd


def test_disturbance_free_feedforward_tracks():
    sp = scan_profile(0.4, MotionLimits(), magnification=4.0)
    r = simulate_scan_sync(sp, np.random.default_rng(0), disturbances=Disturbances.none())
    assert r.ma_max_nm < 1.0 and r.msd_max_nm < 0.5


# ------------------------------------------------------------- metrology
def test_refractive_index_and_interferometer_vs_encoder():
    rng = np.random.default_rng(1)
    assert gas_refractive_index(101325, 273.15, "air") - 1 == pytest.approx(2.93e-4)
    vac = Interferometer(pressure_pa=3.0, gas="H2")
    air = Interferometer(pressure_pa=101325.0, gas="air", index_fluctuation_rel=1e-4)
    assert vac.index_error_rms() < 1e-10 < air.index_error_rms()
    x = np.linspace(0, 0.1, 2000)
    enc = Encoder()
    err = enc.measure(x, rng) - x
    assert np.std(err) < 1e-9
    assert abbe_error(1e-3, 1e-6) == pytest.approx(1e-9)
    tilted = Encoder(abbe_offset_m=2e-3).measure(x, rng, tilt_rad=1e-6) - enc.measure(x, rng)
    assert np.mean(tilted) == pytest.approx(2e-9, abs=0.5e-9)


def test_alignment_sensor_estimates_mark():
    rng = np.random.default_rng(2)
    a = AlignmentSensor()
    x0 = 1.234e-6
    est = np.array([a.measure(x0, rng) for _ in range(300)])
    assert np.mean(est) == pytest.approx(x0, abs=3 * a.precision_m())
    assert np.std(est) == pytest.approx(a.precision_m(), rel=0.3)
    off = a.measure(0.0, np.random.default_rng(3), asym_phase_rad=0.1)
    assert off == pytest.approx(a.asymmetry_offset_m(0.1), abs=5 * a.precision_m())


def test_leveling_and_focus_budget():
    rng = np.random.default_rng(4)
    topo = generate_wafer_topography(rng, pixel_m=1.5e-3)
    meas = LevelSensor().measure(topo, rng)
    res = slit_leveling_residual(topo, meas)
    raw = topo.z[~np.isnan(topo.z)]
    assert np.std(res) < 0.2 * np.std(raw)   # plane following removes most
    assert 3 * np.std(res) < 80e-9
    dof = depth_of_focus(0.33)
    assert 50e-9 < dof < 70e-9 and depth_of_focus(0.55) < dof
    b = default_focus_budget(3 * np.std(res))
    assert b.total < dof
    assert b.total >= max(b.contributions.values())


# --------------------------------------------------------------- overlay
def test_overlay_fit_recovers_grid():
    rng = np.random.default_rng(5)
    lay = field_layout()
    X, Y, xf, yf = alignment_mark_positions(lay.centers)
    R = 0.15
    true = GridModel(tx=2e-9, ty=-1.5e-9, mx=0.1e-6, my=-0.05e-6, rx=0.03e-6, ry=0.01e-6,
                     fmx=0.2e-6, fmy=-0.1e-6, frx=0.1e-6, fry=0.05e-6,
                     higher_x={(3, 0): 3e-9 / R ** 3}, higher_y={(1, 2): -2e-9 / R ** 3})
    dx, dy = true.displacement(X, Y, xf, yf)
    noise = 0.3e-9
    f = fit_overlay(X, Y, xf, yf, dx + noise * rng.standard_normal(X.size),
                    dy + noise * rng.standard_normal(X.size), order=3)
    m = f.model
    for name in ("tx", "ty"):
        assert getattr(m, name) == pytest.approx(getattr(true, name), abs=0.3e-9)
    for name in ("mx", "my", "rx", "ry"):
        assert getattr(m, name) == pytest.approx(getattr(true, name), abs=5e-9)
    for name in ("fmx", "fmy", "frx", "fry"):
        assert getattr(m, name) == pytest.approx(getattr(true, name), abs=2e-8)
    assert m.non_orthogonality == pytest.approx(true.non_orthogonality, abs=1e-8)
    assert m.higher_x[(3, 0)] * R ** 3 == pytest.approx(3e-9, abs=0.6e-9)
    rx3, ry3 = f.mean_plus_3sigma()
    assert rx3 < 4 * noise and ry3 < 4 * noise
    # a linear-only fit leaves the 3rd-order signature in the residuals
    f1 = fit_overlay(X, Y, xf, yf, dx, dy, order=1)
    assert f1.mean_plus_3sigma()[0] > rx3


def test_overlay_budget_rss():
    b = matched_machine_overlay_budget()
    vals = np.array(list(b.contributions.values()))
    assert b.total == pytest.approx(np.sqrt(np.sum(vals ** 2)))
    assert 1e-9 < b.total < 3e-9
    assert mean_plus_3sigma(np.array([1.0, 1.0, 1.0])) == pytest.approx(1.0)


# ---------------------------------------------------------- dose control
def test_pulses_per_point_and_dose():
    N = pulses_per_point(2e-3, 50e3, 0.3)
    assert N == pytest.approx(333.33, rel=1e-3)
    # D = P/(w v): 2 W, 26 mm, 0.3 m/s -> 256 J/m^2 = 25.6 mJ/cm^2
    Ep = 2.0 / 50e3
    assert dose_mj_cm2(Ep, N) == pytest.approx(2.0 / (0.026 * 0.3) * 0.1, rel=1e-6)


def test_dose_error_scales_with_pulses_open_loop():
    rng = np.random.default_rng(6)
    errs = []
    for N in (20.0, 80.0, 320.0):
        raw = synthetic_pulse_train(20000, rng, sigma_rel=0.05)
        d = apply_dose_control(raw, N, rng, feedback=False).dose
        errs.append(np.std(d))
        assert np.std(d) == pytest.approx(open_loop_dose_error(0.05, N), rel=0.35)
    assert errs[0] > errs[1] > errs[2]


def test_dose_feedback_reduces_error():
    rng = np.random.default_rng(7)
    raw = synthetic_pulse_train(20000, rng, sigma_rel=0.05, drift_rel=0.03)
    for N in (40.5, 200.5):
        ol = apply_dose_control(raw, N, rng, feedback=False)
        cl = apply_dose_control(raw, N, rng, feedback=True)
        assert cl.dose_error_3sigma < 0.3 * ol.dose_error_3sigma
        assert np.all(np.abs(cl.command - 1) <= 0.3 + 1e-12)


def test_soft_slit_edge_suppresses_quantisation():
    E = np.ones(4000)
    flat = slit_integrated_dose(E, 20.5, profile="flat")
    trap = slit_integrated_dose(E, 20.5, profile="trapezoid")
    assert np.std(flat) > 10 * np.std(trap)
    assert np.mean(trap) == pytest.approx(1.0, rel=1e-3)


# -------------------------------------------------------- scanner timing
def test_max_scan_speed_relation():
    v = max_scan_speed(30.0, 2.0)
    assert v == pytest.approx(2.0 / (300 * 0.026))
    # slit height cancels
    assert max_scan_speed(30.0, 2.0, slit_height_mm=4.0) == pytest.approx(v)
    assert required_power_at_wafer(30.0, v) == pytest.approx(2.0)
    assert max_scan_speed(1.0, 100.0, v_stage_max=0.6) == 0.6


def test_fields_per_wafer():
    lay = field_layout((26.0, 33.0), 300.0)
    assert 80 <= lay.n_fields <= 120
    assert 50 <= lay.n_full < lay.n_fields
    assert np.all(np.hypot(*lay.centers.T) < 0.15 + 0.03)


def test_throughput_timeline():
    tl = wafer_timeline(0.4)
    assert 0.1 < tl.t_field < 0.3
    assert tl.t_wafer > tl.n_fields * 35e-3 / 0.4
    assert 100 < tl.wafers_per_hour < 300
    lo = throughput_wph(60.0, 2.0)
    hi = throughput_wph(20.0, 2.0)
    assert lo.scan_speed < hi.scan_speed
    assert lo.wafers_per_hour < hi.wafers_per_hour
    assert 50 < lo.wafers_per_hour < hi.wafers_per_hour < 250
