import math

import numpy as np
import pytest

from euvsim.environment import (CarbonGrowthModel, CollectorSnModel, DynamicGasLock,
                                HydrogenCleaning, LumpedMirror, OxidationModel, ReticleHeating,
                                ScannerEnvironment, ULE, absorption_length, critical_thickness,
                                cross_section_cm2, default_chambers, dgl_suppression,
                                euv_plasma_production, evolve_mirror_contamination,
                                gas_transmission, illumination_heat_load,
                                linear_profile_displacement, mirror_heating_map,
                                mirror_lifetime_s, optimal_zero_crossing_C,
                                overlayer_reflectivity, pellicle_equilibrium_temperature,
                                reticle_overlay_error_nm, slab_steady_profile, slab_transient)
from euvsim.environment.vacuum import Chamber, flow_regime


# --- gas -------------------------------------------------------------------------
def test_h2_cross_section_and_absorption_length():
    assert cross_section_cm2("H2") == pytest.approx(6e-20)
    # ~0.15 % per Pa m at room temperature -> L_abs(1 Pa) ~ 680 m
    assert 500 < absorption_length(1.0) < 900
    assert cross_section_cm2("H2O") > 10 * cross_section_cm2("H2")


def test_transmission_decreases_with_pressure_and_length():
    p = np.array([1, 10, 50, 100, 200.0])
    T = np.array([gas_transmission(pi, 1.0) for pi in p])
    assert np.all(np.diff(T) < 0)
    L = np.linspace(0.1, 3, 10)
    TL = gas_transmission(100.0, L)
    assert np.all(np.diff(TL) < 0)
    assert 0.8 < gas_transmission(100.0, 1.0) < 0.9
    # Beer-Lambert: multiplicative in length
    assert gas_transmission(50, 2.0) == pytest.approx(gas_transmission(50, 1.0) ** 2)
    # residual water absorbs more per molecule
    assert gas_transmission({"H2O": 1.0}, 1.0) < gas_transmission({"H2": 1.0}, 1.0)


def test_euv_plasma_yields():
    y = euv_plasma_production(100.0, 3.0, 2.0)
    assert y.ion_pairs_per_s == pytest.approx(y.absorbed_photons_per_s * 91.8 / 36.0, rel=0.01)
    assert y.h_radicals_per_s > y.ion_pairs_per_s


# --- vacuum ----------------------------------------------------------------------
def test_chamber_pressure_q_equals_sp():
    ch = Chamber("x", 1.0, pump_speed_m3_s=2.0, purge_flow_Pa_m3_s=6.0, base_pressure_Pa=0.0)
    assert ch.total_pressure() == pytest.approx(3.0)
    ch2 = Chamber("x", 1.0, pump_speed_m3_s=2.0, conductance_m3_s=2.0, purge_flow_Pa_m3_s=6.0,
                  base_pressure_Pa=0.0)
    assert ch2.total_pressure() == pytest.approx(6.0)
    p = ch.pressure_transient([0, 1e3], 100.0)
    assert p[0] == pytest.approx(100.0) and p[1] == pytest.approx(3.0, rel=1e-6)
    chs = default_chambers()
    assert 50 < chs["source"].total_pressure() < 150
    assert 1 < chs["pob"].total_pressure() < 10
    assert flow_regime(100.0, 0.1) == "viscous"
    assert flow_regime(1e-4, 0.1) == "molecular"


def test_dgl_suppresses_exponentially_with_peclet():
    flows = [0.02, 0.05, 0.1, 0.2]
    pe = [DynamicGasLock(Q_H2_Pa_m3_s=q).peclet for q in flows]
    sup = [DynamicGasLock(Q_H2_Pa_m3_s=q).suppression() for q in flows]
    assert np.all(np.diff(pe) > 0) and np.all(np.diff(sup) < 0)
    for p, s in zip(pe, sup):
        assert s == pytest.approx(math.exp(-p))
    # log-linear: ln(suppression) = -Pe
    assert np.allclose(np.log(sup), -np.array(pe))
    # Pe independent of lock pressure (v ~ 1/p, D ~ 1/p)
    assert DynamicGasLock(pressure_Pa=1).peclet == pytest.approx(DynamicGasLock(pressure_Pa=10).peclet)
    assert dgl_suppression(0.0) == 1.0


# --- contamination -------------------------------------------------------------------
def test_carbon_reduces_reflectivity_monotonically():
    d = np.linspace(0, 4, 9)
    R = overlayer_reflectivity("C", d)
    assert np.all(np.diff(R) < 0)
    assert 0.2 < critical_thickness("C", 0.01) < 2.0
    assert overlayer_reflectivity("Sn", 1.0) < overlayer_reflectivity("C", 1.0)
    assert overlayer_reflectivity("RuO2", 1.0) < R[0]


def test_carbon_growth_scaling_and_saturation():
    c = CarbonGrowthModel(p_hc_Pa=1e-7)
    assert c.growth_rate_nm_s(0.02) == pytest.approx(2 * c.growth_rate_nm_s(0.01), rel=0.05)
    assert CarbonGrowthModel(p_hc_Pa=2e-9).growth_rate_nm_s(0.1) == pytest.approx(
        2 * CarbonGrowthModel(p_hc_Pa=1e-9).growth_rate_nm_s(0.1), rel=0.05)
    # saturation at the molecule supply limit
    assert c.growth_rate_nm_s(1e4) == pytest.approx(c.supply_limit_nm_s(), rel=0.01)


def test_h_radical_cleaning_balances_growth():
    c = CarbonGrowthModel(p_hc_Pa=1e-7)
    I = 0.5
    g = c.growth_rate_nm_s(I)
    flux = HydrogenCleaning().balancing_flux_C(g)
    week = 7 * 86400
    no_clean = evolve_mirror_contamination(week, I, c, cleaning=HydrogenCleaning(0.0))
    clean = evolve_mirror_contamination(week, I, c, cleaning=HydrogenCleaning(3 * flux))
    assert no_clean.carbon_nm[-1] == pytest.approx(g * week, rel=1e-3)
    # with cleaning > growth carbon stays at a sub-monolayer steady state
    assert clean.carbon_nm[-1] < 0.1 and clean.carbon_nm[-1] == pytest.approx(0.1 / 3, rel=0.05)
    assert clean.relative_loss[-1] < no_clean.relative_loss[-1]
    assert mirror_lifetime_s(g - HydrogenCleaning(flux).etch_rate_nm_s("C")) == math.inf


def test_oxidation_self_limiting_and_carbon_shielding():
    ox = OxidationModel(p_h2o_Pa=1e-5)
    assert ox.rate_nm_s(1.0, 1.0) < ox.rate_nm_s(1.0, 0.0)
    h = evolve_mirror_contamination(30 * 86400, 1.0, CarbonGrowthModel(p_hc_Pa=0.0), ox)
    assert h.oxide_nm[-1] > 0 and np.all(np.diff(h.relative_loss) >= -1e-12)
    shielded = evolve_mirror_contamination(30 * 86400, 1.0, CarbonGrowthModel(p_hc_Pa=1e-6), ox)
    assert shielded.oxide_nm[-1] < h.oxide_nm[-1]


def test_collector_sn():
    s = CollectorSnModel()
    assert s.net_rate_nm_s() > 0
    assert 0 < s.lifetime_s() < math.inf
    assert s.reflectivity_after(30 * 86400) < s.reflectivity_after(0)
    more_h = CollectorSnModel(cleaning=HydrogenCleaning(flux_H_cm2_s=2e16))
    assert more_h.lifetime_s() == math.inf


# --- thermal ---------------------------------------------------------------------
def test_mirror_temperature_rises_with_power():
    m = LumpedMirror()
    T = [m.steady_state_C(p) for p in (0, 5, 10, 20)]
    assert np.all(np.diff(T) > 0) and T[0] == pytest.approx(m.T_cool_C)
    tr = m.transient_C([0, m.time_constant_s, 20 * m.time_constant_s], 10)
    assert tr[0] == pytest.approx(22.0)
    assert tr[1] == pytest.approx(22 + 5 * (1 - math.exp(-1)))
    assert tr[2] == pytest.approx(27.0, rel=1e-6)


def test_slab_transient_reaches_steady_state():
    t, z, T = slab_transient(500.0, 0.03, ULE, 22.0, t_end_s=2e4, n_t=200, n_z=31)
    _, Ts = slab_steady_profile(500.0, 0.03, ULE, 22.0, n_z=31)
    assert np.allclose(T[-1], Ts, atol=0.05 * (Ts[0] - 22))
    assert np.all(np.diff(T[:, 0]) >= -1e-9)


def test_wavefront_vanishes_at_ule_zero_crossing():
    # small uniform heating: at T_cool = T_zc only the quadratic term remains
    s, L = 0.5, 0.05
    h_zc = linear_profile_displacement(s, L, ULE, ULE.T_zc_C)
    h_off = linear_profile_displacement(s, L, ULE, ULE.T_zc_C - 5.0)
    assert abs(h_zc) < 0.05 * abs(h_off)
    assert linear_profile_displacement(s, L, ULE, ULE.T_zc_C - s / 3) == pytest.approx(0, abs=1e-20)
    assert optimal_zero_crossing_C(22.0, 3.0) == pytest.approx(23.0)
    q = illumination_heat_load("dipole_x", n=31, total_W=0.5)
    on = mirror_heating_map(q, 0.2, T_cool_C=ULE.T_zc_C)
    off = mirror_heating_map(q, 0.2, T_cool_C=ULE.T_zc_C - 5)
    assert on.rms_nm < 0.2 * off.rms_nm
    # more power -> hotter
    assert mirror_heating_map(2 * q, 0.2).surface_rise_K.max() == pytest.approx(
        2 * on.surface_rise_K.max())


def test_dipole_heating_gives_astigmatism():
    T_c = ULE.T_zc_C - 5
    dip = mirror_heating_map(illumination_heat_load("dipole_x", n=31, total_W=5), 0.2, T_cool_C=T_c)
    conv = mirror_heating_map(illumination_heat_load("conventional", n=31, total_W=5), 0.2, T_cool_C=T_c)
    assert abs(dip.zernike_nm[6]) > abs(dip.zernike_nm[4])
    assert abs(conv.zernike_nm[6]) < 1e-3 * abs(conv.zernike_nm[4])
    ydip = mirror_heating_map(illumination_heat_load("dipole_y", n=31, total_W=5), 0.2, T_cool_C=T_c)
    assert ydip.zernike_nm[6] == pytest.approx(-dip.zernike_nm[6], rel=1e-6)


def test_pellicle_radiative_equilibrium():
    I = [1e4, 3e4, 6e4]
    T = [pellicle_equilibrium_temperature(i, 0.2) for i in I]
    assert np.all(np.diff(T) > 0)
    eps = [0.1, 0.2, 0.4, 0.8]
    Te = [pellicle_equilibrium_temperature(5e4, e) for e in eps]
    assert np.all(np.diff(Te) < 0)
    assert 600 + 273 < pellicle_equilibrium_temperature(1e5, 0.1) < 1200 + 273
    assert pellicle_equilibrium_temperature(0.0, 0.2) == pytest.approx(295.0)


def test_reticle_overlay_is_expansion_over_four():
    dx, ov = reticle_overlay_error_nm(1.0, 50.0, 30e-9)
    assert dx == pytest.approx(1.5)       # 30 ppb/K * 1 K * 50 mm
    assert ov == pytest.approx(dx / 4)
    rh = ReticleHeating(material=ULE.with_zero_crossing(10.0))
    assert rh.overlay_at_wafer_nm() == pytest.approx(rh.mask_expansion_nm() / 4)
    assert rh.overlay_at_wafer_nm() > 0
    assert ReticleHeating(power_on_reticle_W=60).delta_T_K == pytest.approx(2 * ReticleHeating().delta_T_K)


# --- integration -------------------------------------------------------------------
def test_scanner_environment():
    env = ScannerEnvironment()
    s = env.summary()
    assert 0.7 < s["gas_transmission"] < 0.95
    budget = env.power_budget(250.0)
    assert budget[1].incident_W < 250.0
    assert all(b.absorbed_W > 0 for b in budget)
    hot = env.heating_state(500.0)
    cold = env.heating_state(250.0)
    assert hot["T_FFM_C"] > cold["T_FFM_C"] and hot["pellicle_T_C"] > cold["pellicle_T_C"]
    t, rel = env.transmission_loss_vs_time(days=7, n_out=10)
    assert rel[0] == pytest.approx(1.0) and np.all(np.diff(rel) <= 1e-12) and rel[-1] < 1
    # higher pressure -> lower gas transmission
    env2 = ScannerEnvironment()
    env2.chambers["source"].purge_flow_Pa_m3_s *= 2
    assert env2.gas_transmission() < env.gas_transmission()
