import numpy as np
import pytest

from duv.core import REDUCTION
from duv.stage import (
    DualStage,
    PositionSensor,
    ScanProfile,
    ServoLoop,
    field_layout,
    ma_msd,
    msd_contrast_loss,
    point_to_point_time,
    scan_directions,
    simulate_synchronization,
    synchronization_error,
    throughput,
)


def test_profile_limits_and_speed():
    p = ScanProfile()
    sp = p.sample(fs_hz=50e3)
    v = sp["vel_m_s"]
    assert np.isclose(v.max(), p.scan_speed_m_s, rtol=1e-9)
    assert np.all(np.abs(sp["acc_m_s2"]) <= p.max_accel_m_s2 * (1 + 1e-9))
    assert np.all(np.abs(sp["jerk_m_s3"]) <= p.max_jerk_m_s3 * (1 + 1e-9))
    # ends at rest, travelled the documented scan length
    assert abs(v[-1]) < 1e-9 and abs(sp["acc_m_s2"][-1]) < 1e-9
    assert np.isclose(sp["pos_m"][-1] * 1e3, p.scan_length_mm(), rtol=1e-6)
    # numerical derivative consistent with analytic velocity
    dv = np.gradient(sp["pos_m"], sp["t"])
    assert np.max(np.abs(dv - v)) < 1e-3


def test_constant_velocity_covers_field_plus_slit():
    p = ScanProfile()
    sp = p.sample(fs_hz=20e3)
    exp = sp["exposure"]
    assert np.allclose(sp["vel_m_s"][exp], p.scan_speed_m_s, rtol=1e-9)
    assert np.all(np.abs(sp["acc_m_s2"][exp]) < 1e-9)
    t0, t1 = p.exposure_window()
    assert (t1 - t0) * p.scan_speed_mm_s >= p.field_height_mm + p.slit_height_mm - 1e-9
    assert p.time_per_field() == pytest.approx(2 * p.ramp_time() + p.settle_time_s + (t1 - t0))


def test_low_speed_profile_reduces_peak_accel():
    p = ScanProfile(scan_speed_mm_s=50.0, max_accel_m_s2=30, max_jerk_m_s3=5000)
    sp = p.sample(fs_hz=50e3)
    assert np.isclose(sp["vel_m_s"].max(), 0.05)
    assert np.abs(sp["acc_m_s2"]).max() <= p.peak_accel_m_s2 + 1e-9 < 30


def test_reticle_profile_scaled():
    p = ScanProfile()
    r = p.reticle_profile()
    assert r.time_per_field() == pytest.approx(p.time_per_field())
    t = np.linspace(0, p.time_per_field(), 500)
    assert np.allclose(r.setpoint(t)["pos_m"], REDUCTION * p.setpoint(t)["pos_m"])


def test_point_to_point_time_monotonic():
    times = [point_to_point_time(d, 1.0, 30.0, 5000.0) for d in (0.001, 0.01, 0.026, 0.1, 0.3)]
    assert np.all(np.diff(times) > 0)


def test_feedforward_reduces_error():
    p = ScanProfile()
    with_ff = ServoLoop(feedforward=True).simulate(p, np.random.default_rng(1))
    no_ff = ServoLoop(feedforward=False).simulate(p, np.random.default_rng(1))
    assert np.abs(with_ff["error_nm"]).max() < 0.05 * np.abs(no_ff["error_nm"]).max()
    assert with_ff["msd_max_nm"] < no_ff["msd_max_nm"]


def test_default_ma_msd_small():
    p = ScanProfile()
    res = ServoLoop().simulate(p, np.random.default_rng(2))
    assert res["msd_max_nm"] < 5.0
    assert res["ma_max_nm"] < 5.0
    sync = simulate_synchronization(p, rng=np.random.default_rng(3))
    assert sync["msd_max_nm"] < 5.0


def test_flexible_mode_stable():
    res = ServoLoop(flex_mode_hz=2500.0).simulate(ScanProfile(), np.random.default_rng(4))
    assert np.all(np.isfinite(res["error_nm"]))
    assert res["msd_max_nm"] < 10.0
    assert "error_payload_nm" in res


def test_ma_msd_constant_error():
    t = np.arange(0, 0.1, 5e-5)
    e = np.full_like(t, 3.7)
    ma, msd = ma_msd(e, t, 700.0, 8.0)
    ok = np.isfinite(ma)
    assert ok.sum() > 0.8 * t.size
    assert np.allclose(ma[ok], 3.7)
    assert np.allclose(msd[ok], 0.0, atol=1e-6)


def test_ma_msd_sine_averages_out():
    # a sine whose period equals the window has MA ~ 0 and MSD = amp/sqrt(2)
    T = 8.0 / 700.0
    t = np.arange(0, 0.2, T / 400)
    e = 2.0 * np.sin(2 * np.pi * t / T)
    ma, msd = ma_msd(e, t, 700.0, 8.0)
    ok = np.isfinite(ma)
    assert np.abs(ma[ok]).max() < 0.05
    assert np.allclose(msd[ok], 2.0 / np.sqrt(2), rtol=0.02)


def test_synchronization_and_contrast():
    assert synchronization_error(5.0, 4 * 5.0) == pytest.approx(0.0)
    assert synchronization_error(1.0, -4.0) == pytest.approx(2.0)
    assert msd_contrast_loss(0.0, 80.0) == pytest.approx(1.0)
    assert msd_contrast_loss(5.0, 80.0) == pytest.approx(np.exp(-2 * np.pi**2 * 25 / 6400))
    assert msd_contrast_loss(10.0, 80.0) < msd_contrast_loss(5.0, 80.0)


def test_position_sensor_noise():
    rng = np.random.default_rng(5)
    enc = PositionSensor("encoder")
    ifm = PositionSensor("interferometer", path_length_mm=300)
    ifm_long = PositionSensor("interferometer", path_length_mm=1200)
    assert enc.sigma_nm < 0.1 < ifm.sigma_nm < ifm_long.sigma_nm
    meas = ifm.measure(np.zeros(20000), rng)
    assert 0.5 * ifm.sigma_nm < meas.std() < 2 * ifm.sigma_nm
    with pytest.raises(ValueError):
        PositionSensor("capacitive")


def test_field_layout():
    fields = field_layout()
    full = field_layout(include_partial=False)
    assert 80 <= len(fields) <= 110
    assert 40 <= len(full) < len(fields)
    # meander: consecutive fields in a row are one field width apart
    steps = np.diff(np.array(fields), axis=0)
    same_row = steps[:, 1] == 0
    assert np.allclose(np.abs(steps[same_row, 0]), 26.0)
    d = scan_directions(len(fields))
    assert np.all(d[:-1] == -d[1:])


def test_throughput_plausible_and_scan_speed():
    res = throughput()
    assert 150 < res["wph"] < 400
    assert res["limiting"] == "expose"
    assert throughput(900)["wph"] > res["wph"] > throughput(500)["wph"]
    # dual stage beats a single-stage tool
    assert throughput(dual=False)["wph"] < res["wph"]


def test_dual_stage_cycle_time():
    ds = DualStage()
    c96 = ds.cycle_time(96)
    assert c96 == pytest.approx(max(ds.expose_breakdown(96)["expose_side_s"], ds.measure_time()) + ds.swap_time_s)
    assert ds.cycle_time(120) > c96
    # few fields -> measure side limits
    assert ds.cycle_time(5) == pytest.approx(ds.measure_time() + ds.swap_time_s)
