import math

import numpy as np
import pytest

from duv.core import REDUCTION, Grid
from duv.reticle import (
    Pellicle,
    ReticleHeating,
    ReticleStage,
    add_srafs,
    apply_bias,
    contact_array,
    isolated_line,
    line_space,
    make_mask,
    mask_3d_correction,
    mask_cd_error,
    meef,
    rectangles,
    reticle_to_wafer,
    simple_opc,
    wafer_to_reticle,
)

GRID = Grid(n=128, pixel=4.0)  # period 512 nm


def test_line_space_duty_cycle_subpixel():
    for cd in (40.0, 45.0, 37.3):
        p = line_space(GRID, cd, 128.0)
        assert p.mean() == pytest.approx(cd / 128.0, rel=1e-9)
        assert p.min() >= 0 and p.max() <= 1
    ph = line_space(GRID, 45.0, 128.0, orientation="horizontal")
    assert np.allclose(ph, line_space(GRID, 45.0, 128.0).T)
    assert line_space(GRID, 45.0, 128.0, tone="clear").mean() == pytest.approx(1 - 45 / 128)


def test_line_space_pitch_must_divide_period():
    with pytest.raises(ValueError):
        line_space(GRID, 40.0, 100.0)


def test_rectangles_area_and_wrap():
    p = rectangles(GRID, [(-10.5, -20.0, 10.5, 20.0)])
    assert p.sum() * GRID.pixel ** 2 == pytest.approx(21.0 * 40.0)
    # rectangle crossing the periodic boundary keeps its area
    h = GRID.period / 2
    q = rectangles(GRID, [(h - 10, 0, h + 10, 30)])
    assert q.sum() * GRID.pixel ** 2 == pytest.approx(20.0 * 30.0)


def test_contact_array_and_isolated_line():
    c = contact_array(GRID, 60.0, 128.0)  # clear holes in dark field
    assert c.mean() == pytest.approx(1 - 16 * 60 ** 2 / 512 ** 2)
    iso = isolated_line(GRID, 50.0)
    assert iso.mean() == pytest.approx(50.0 / 512.0)


def test_binary_vs_attpsm_transmission():
    p = line_space(GRID, 64.0, 128.0)
    b = make_mask(p, GRID, "binary")
    a = make_mask(p, GRID, "attpsm", transmission=0.06)
    on, off = p == 1, p == 0
    assert np.allclose(b.transmission[on], 0) and np.allclose(b.transmission[off], 1)
    assert np.allclose(np.abs(a.transmission[on]) ** 2, 0.06)
    assert np.allclose(np.angle(a.transmission[on]), np.pi)
    assert np.allclose(a.transmission[off], 1)


def test_altpsm_alternates_phase():
    p = line_space(GRID, 32.0, 128.0)  # 4 openings per period -> conflict free
    m = make_mask(p, GRID, "altpsm")
    row = m.transmission[0]
    clear = p[0] == 0
    phases = np.round(np.abs(np.angle(row[clear])) / np.pi).astype(int)
    assert set(phases) == {0, 1}
    # zero-order of the clear field cancels for an alt-PSM
    assert abs(m.transmission.mean()) < 1e-9
    assert np.allclose(np.abs(m.transmission), 1 - p)


def test_bias_changes_cd_exactly():
    p = line_space(GRID, 40.0, 128.0)
    for bias in (2.0, 3.3, -1.7):
        q = apply_bias(p, GRID, bias)
        assert q.mean() == pytest.approx((40.0 + 2 * bias) / 128.0, rel=1e-9)
    sq = rectangles(GRID, [(-20, -20, 20, 20)])
    assert apply_bias(sq, GRID, 4.0).sum() * 16 == pytest.approx(48.0 ** 2)


def test_opc_and_srafs():
    iso = isolated_line(GRID, 40.0)
    dense = line_space(GRID, 64.0, 128.0)
    # isolated line gets the iso bias, dense lines the dense bias
    assert simple_opc(iso, GRID, bias_dense_nm=0, bias_iso_nm=3).mean() == pytest.approx(46 / 512)
    assert simple_opc(dense, GRID, bias_dense_nm=1, bias_iso_nm=3).mean() == pytest.approx(66 / 128)
    sq = rectangles(GRID, [(-40, -40, 40, 40)])
    assert simple_opc(sq, GRID, 0, 0, serif_nm=12).sum() > sq.sum()
    s = add_srafs(GRID, [0.0], sraf_width=20.0, sraf_offset=120.0, base=iso)
    assert s.mean() == pytest.approx((40 + 2 * 20) / 512)


def test_mask_3d_correction_reduces_clear_transmission():
    p = line_space(GRID, 64.0, 128.0)
    thin = make_mask(p, GRID, "binary")
    thick = mask_3d_correction(p, GRID, "binary", bl_width_nm=4.0, bl_amplitude=0.5, bl_phase_deg=90)
    assert np.abs(thick.transmission).sum() < np.abs(thin.transmission).sum()
    assert np.abs(np.angle(thick.transmission[np.abs(thick.transmission) > 0])).max() > 0.1


def test_pellicle_transmission():
    pel = Pellicle()
    assert pel.thickness_nm == pytest.approx(Pellicle.ar_thickness(), abs=0.5)
    assert pel.transmission(0.0) > 0.99
    # off the AR condition, reflection losses appear
    assert Pellicle(thickness_nm=pel.thickness_nm + 35).transmission(0.0) < 0.92
    th, t = pel.transmission_vs_na()
    assert np.all(t > 0.85) and np.all(t <= 1 + 1e-12)
    # lossless film: T + R = 1 is implied by T <= 1; absorbing film transmits less
    assert Pellicle(k=1e-4).transmission(0.0) < pel.transmission(0.0)


def test_pellicle_particle_out_of_focus():
    pel = Pellicle()
    r = pel.standoff_defect_blur(particle_size_um=10.0, standoff_mm=6.3)
    assert r["blur_diameter_um"] == pytest.approx(2 * 6300 * math.tan(math.asin(1.35 / 4)))
    assert r["obscuration"] < 0.001


def test_mask_cd_error():
    p = line_space(GRID, 40.0, 128.0)
    sys = mask_cd_error(p, GRID, 4.0, mode="systematic", reticle_scale=True)
    assert sys.mean() == pytest.approx(41.0 / 128.0)
    rnd = mask_cd_error(p, GRID, 2.0, rng=1)
    cds = rnd.sum(axis=0).reshape(4, 32).sum(axis=1) * GRID.pixel / GRID.n
    assert not np.allclose(cds, 40.0) and np.all(np.abs(cds - 40) < 10)


def test_meef_synthetic():
    mask_cd = np.array([38.0, 40.0, 42.0, 44.0])
    wafer_cd = 3.0 * (mask_cd - 40.0) + 45.0
    assert meef(wafer_cd, mask_cd) == pytest.approx(3.0)
    assert meef(wafer_cd, mask_cd * REDUCTION, reticle_scale=True) == pytest.approx(3.0)


def test_reticle_stage_and_conversions():
    st = ReticleStage(scan_speed_wafer_mm_s=700)
    assert st.reticle_speed() == pytest.approx(4 * 700)
    assert st.reticle_velocity() < 0
    w = np.linspace(0, 1e6, 50)
    assert np.allclose(st.synchronization_error(w, st.ideal_reticle_position(w)), 0)
    t = np.linspace(0, 0.05, 2001)
    wpos = 700e6 * t
    rpos = st.ideal_reticle_position(wpos) + 4 * 2.0  # 2 nm wafer-scale offset
    ma, msd = st.ma_msd(t, wpos, rpos)
    ok = ~np.isnan(ma)
    assert ok.any() and np.allclose(ma[ok], 2.0) and np.allclose(msd[ok], 0, atol=1e-9)
    assert wafer_to_reticle(10.0) == 40.0 and reticle_to_wafer(40.0) == 10.0


def test_reticle_heating():
    rh = ReticleHeating(absorbed_power_w=1.0)
    dT = rh.temperature_rise()
    assert dT > 0
    assert rh.magnification_error_ppm() == pytest.approx(0.5 * dT)
    dx, _ = rh.registration_error_nm(13e6, 0.0)
    assert dx == pytest.approx(0.5e-6 * dT * 13e6)
    assert rh.temperature_rise(0.0) == pytest.approx(0.0)
    assert rh.temperature_rise(10 * rh.time_constant_s) == pytest.approx(dT, rel=1e-4)
