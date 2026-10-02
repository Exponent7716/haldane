import numpy as np
import pytest

from duv.core import WAVELENGTH_ARF
from duv.light_source import (
    ArFLaser,
    DischargeModel,
    DoseController,
    GasManager,
    LineNarrowingModule,
    dose_from_pulses,
    e95_width_pm,
    pulses_per_point,
    speckle_contrast,
    window_sums,
)


@pytest.mark.parametrize("e95", [0.15, 0.30, 0.50])
def test_spectrum_e95_matches_target(e95):
    laser = ArFLaser(bandwidth_e95_pm=e95)
    wl, w = laser.spectrum(n_samples=21)
    assert wl.shape == w.shape == (21,)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= 0)
    assert np.isclose(np.sum(wl * w), WAVELENGTH_ARF, atol=1e-9)
    assert e95_width_pm(wl, w) == pytest.approx(e95, rel=0.03)
    wl_f, w_f = laser.spectrum(n_samples=801)
    assert e95_width_pm(wl_f, w_f) == pytest.approx(e95, rel=0.005)
    # line-narrowed shape: peaked, FWHM well below E95
    assert np.argmax(w) == 10
    assert laser.line_fwhm_pm() < 0.6 * e95


def test_fire_mean_and_jitter():
    laser = ArFLaser(burst_spike=0.0)
    e = laser.fire(20000, rng=np.random.default_rng(0))
    assert e.mean() == pytest.approx(15.0, rel=0.005)
    assert e.std() / e.mean() == pytest.approx(0.03, rel=0.05)
    # higher voltage -> more energy
    e_hi = laser.fire(2000, rng=np.random.default_rng(1), voltage_command=1050.0)
    assert e_hi.mean() > 1.05 * e.mean()
    assert laser.average_power_w == pytest.approx(90.0)


def test_voltage_inverse_and_burst_spike():
    laser = ArFLaser(energy_sigma=0.0)
    v = laser.voltage_for_energy(12.0)
    assert float(laser.energy_at_voltage(v)) == pytest.approx(12.0, rel=1e-9)
    e = laser.fire(200)
    assert e[0] == pytest.approx(15.0 * 1.05, rel=1e-6)
    assert e[-1] == pytest.approx(15.0, rel=1e-3)


def test_discharge_physics():
    dm = DischargeModel()
    e1 = dm.output_energy_mj(1000.0)
    assert 5.0 < e1 < 40.0
    assert dm.output_energy_mj(1100.0) > e1
    # optimum F2 exists and is above the nominal operating point
    x_opt = dm.optimum_f2()
    assert dm.f2_nominal < x_opt < 1e-2
    assert dm.output_energy_mj(1000.0, 5e-3) < dm.output_energy_mj(1000.0, x_opt)
    # gain-switched pulse is shorter than the pump
    p = dm.simulate_pulse(1000.0)
    assert 0 < p["fwhm_ns"] < dm.pump_duration_ns
    assert p["energy_rel"] > 0
    assert dm.simulate_pulse(1100.0)["energy_rel"] > p["energy_rel"]


def test_f2_depletion_and_injection():
    gm = GasManager()
    e0 = gm.target_energy_mj
    hist = gm.run(3_000_000, chunk=100_000, mode="constant_voltage", auto_inject=False)
    assert np.all(np.diff(hist["energy_mj"]) < 0)
    assert hist["energy_mj"][-1] < 0.95 * e0
    gm.inject()
    assert gm.f2_fraction == pytest.approx(gm.discharge.f2_nominal)
    v0 = float(gm.discharge.voltage_for_energy(e0))
    assert float(gm.discharge.output_energy_mj(v0, gm.f2_fraction)) == pytest.approx(e0, rel=1e-9)


def test_gas_manager_voltage_servo_and_auto_injection():
    gm = GasManager()
    hist = gm.run(10_000_000, chunk=100_000, mode="constant_energy")
    v0 = float(gm.discharge.voltage_for_energy(gm.target_energy_mj))
    assert hist["voltage_v"].max() > v0
    assert hist["voltage_v"].max() <= gm.v_inject_threshold + 1.0
    assert hist["injected"].sum() >= 1
    assert np.allclose(hist["energy_mj"], gm.target_energy_mj)


def test_line_narrowing_module():
    lnm = LineNarrowingModule()
    theta = lnm.littrow_angle_rad()
    assert np.degrees(theta) == pytest.approx(78, abs=3)  # echelle near R5 blaze
    assert lnm.wavelength_at_angle(theta) == pytest.approx(WAVELENGTH_ARF)
    assert 0.2 < lnm.bandwidth_e95_pm() < 0.45
    narrow = LineNarrowingModule(magnification=2 * lnm.magnification)
    assert narrow.bandwidth_fwhm_pm() == pytest.approx(lnm.bandwidth_fwhm_pm() / 2)
    dtheta_urad = lnm.tune(WAVELENGTH_ARF + 0.001)  # +1 pm
    assert dtheta_urad > 0
    assert dtheta_urad * lnm.wavelength_shift_pm_per_urad() == pytest.approx(1.0, rel=1e-3)


def test_wavelength_feedback_reduces_drift():
    laser = ArFLaser()
    ol = laser.wavelength_stability(5000, rng=np.random.default_rng(3), closed_loop=False)
    cl = laser.wavelength_stability(5000, rng=np.random.default_rng(3), closed_loop=True)
    assert cl["ma_rms_fm"] < 0.1 * ol["ma_rms_fm"]
    assert cl["ma_max_fm"] < 20.0  # within typical +-20 fm MA spec
    assert cl["msd_mean_fm"] == pytest.approx(laser.wavelength_sigma_fm, rel=0.3)


def test_pulses_per_point():
    assert pulses_per_point(8.0, 700.0, 6000.0) == pytest.approx(68.571, rel=1e-4)
    assert pulses_per_point(8.0, 350.0, 6000.0) == pytest.approx(2 * 68.571, rel=1e-4)


def test_dose_from_pulses_and_window_sums():
    e = np.full(69, 15.0)
    dose = dose_from_pulses(e, transmission=0.08)
    area = 2.6 * 0.8
    assert dose == pytest.approx(0.08 * 69 * 15.0 / area)
    assert 20 < dose < 80
    ws = window_sums(np.arange(10.0), 3)
    assert np.allclose(ws, [3, 6, 9, 12, 15, 18, 21, 24])


def test_dose_controller_improves_uniformity():
    laser = ArFLaser()
    n = int(round(pulses_per_point(8.0, 700.0, laser.rep_rate_hz)))
    ctrl = DoseController(target_energy_mj=14.0, n_window=n)
    ol = ctrl.run(laser, 3000, closed_loop=False, rng=np.random.default_rng(5))
    cl = ctrl.run(laser, 3000, closed_loop=True, rng=np.random.default_rng(5))
    # open-loop window error ~ sigma/sqrt(N) plus the energy offset/spike
    assert ol.stats["std_error"] == pytest.approx(0.03 / np.sqrt(n), rel=0.35)
    assert cl.stats["three_sigma"] < 0.5 * ol.stats["three_sigma"]
    assert cl.stats["max_abs_error"] < 0.5 * ol.stats["max_abs_error"]
    assert abs(cl.stats["mean_error"]) < 1e-3
    assert cl.window_dose_mj_cm2.size == 3000 - n + 1


def test_speckle_contrast():
    assert speckle_contrast(1) == pytest.approx(1.0)
    assert speckle_contrast(100) == pytest.approx(0.1)
    assert speckle_contrast(16, n_spatial_modes=4) == pytest.approx(1 / 8)
    # fully correlated pulses give no reduction
    assert speckle_contrast(100, pulse_correlation=1.0) == pytest.approx(1.0)
    c = [speckle_contrast(n) for n in (1, 10, 100)]
    assert c[0] > c[1] > c[2]
