"""Physics sanity checks for euvsim.source."""
import numpy as np
import pytest

from euvsim.source import (DebrisModel, DriveLaser, DropletGenerator, EllipsoidalCollector,
                           IonStopping, LPPSource, PrePulseTarget, SnPlasma, coupling_fraction,
                           hit_probability, larmor_radius_m, INBAND_HI_NM, INBAND_LO_NM)


@pytest.fixture(scope="module")
def src():
    return LPPSource()


def test_photon_energy(src):
    assert 91.0 < src.photon_energy_ev < 92.5


def test_drive_laser_numbers():
    las = DriveLaser()
    assert las.main_pulse_energy_j == pytest.approx(0.44)
    assert 1e10 < las.main_pulse.peak_intensity_w_cm2 < 5e11
    assert las.main_pulse.critical_density_cm3 == pytest.approx(1e19, rel=0.05)
    rng = np.random.default_rng(1)
    e = las.sample_energies(20000, rng)
    assert np.std(e) / np.mean(e) == pytest.approx(las.energy_sigma_rel, rel=0.05)


def test_ce_peaks_at_sensible_intensity():
    p = SnPlasma()
    i = np.logspace(9, 13, 401)
    ce = p.conversion_efficiency(i)
    i_peak = i[np.argmax(ce)]
    assert 3e10 < i_peak < 3e11
    assert 0.04 < ce.max() < 0.07
    assert ce[0] < 0.2 * ce.max() and ce[-1] < 0.2 * ce.max()


def test_plasma_temperature_and_charge():
    p = SnPlasma()
    te = float(p.electron_temperature_ev(1e11))
    assert 30 <= te <= 50
    assert 8 <= p.mean_charge(te) <= 14
    assert p.electron_temperature_ev(1e12) > p.electron_temperature_ev(1e11)


def test_spectrum_peak_and_opacity_broadening():
    p = SnPlasma()
    assert abs(p.peak_wavelength_nm() - 13.5) < 0.15
    thin = SnPlasma(optical_depth=0.3).inband_fraction()
    thick = SnPlasma(optical_depth=10.0).inband_fraction()
    assert thick < thin            # opacity broadens the UTA -> less in-band
    assert 0.1 < p.inband_fraction() < 0.6


def test_source_spectrum_normalisation(src):
    wl = np.linspace(INBAND_LO_NM, INBAND_HI_NM, 201)
    inband = np.trapezoid(src.spectrum(wl), wl)
    assert inband == pytest.approx(src.inband_power_2pi_w(), rel=1e-3)
    # out-of-band DUV and IR present
    assert src.spectrum(np.array([200.0]))[0] > 0
    assert src.spectrum(np.array([10600.0]))[0] > 0
    # IR strongly suppressed at IF relative to plasma
    ratio = src.spectrum_at_if(np.array([10600.0]))[0] / src.spectrum(np.array([10600.0]))[0]
    assert ratio < 1e-2


def test_droplet_rayleigh_plateau():
    d = DropletGenerator()
    assert 20e-6 < d.droplet_diameter_m < 35e-6
    assert d.spacing_m == pytest.approx(70.0 / 50e3)
    assert 1e-6 < d.jet_diameter_m < 10e-6
    assert d.rayleigh_frequency_hz > 1e6          # MHz natural break-up
    assert d.coalescence_number > 10


def test_prepulse_expansion():
    d = DropletGenerator()
    t = PrePulseTarget()
    r1 = t.radius_m(d, 5e-3, 0.5e-6)
    r2 = t.radius_m(d, 5e-3, 1.5e-6)
    assert d.droplet_diameter_m / 2 < r1 < r2
    assert 100e-6 < r2 < 300e-6
    assert t.radius_m(d, 10e-3, 1.5e-6) > r2
    assert t.thickness_m(d, 5e-3, 1.5e-6) < 1e-6


def test_targeting():
    assert coupling_fraction(0.0, 150e-6, 150e-6) == pytest.approx(1 - np.exp(-2), rel=1e-6)
    assert coupling_fraction(100e-6, 150e-6, 150e-6) < coupling_fraction(0.0, 150e-6, 150e-6)
    assert hit_probability(2e-6, 10e-6) > hit_probability(5e-6, 10e-6)
    assert hit_probability(3e-6, 3e-6) == pytest.approx(1 - np.exp(-0.5))


def test_debris_stopping_increases_with_pressure():
    st = IonStopping()
    r_low = st.range_m(3000.0, 20.0)
    r_high = st.range_m(3000.0, 150.0)
    assert r_high < r_low
    assert st.range_m(3000.0, 100.0) == pytest.approx(r_low * 20 / 100, rel=1e-9)  # R ~ 1/p
    assert 0.03 < st.range_m(3000.0, 100.0) < 0.5
    t = [DebrisModel(h2_pressure_pa=p).ion_transmission() for p in (10, 50, 100, 150)]
    assert all(a > b for a, b in zip(t, t[1:]))


def test_magnetic_mitigation():
    assert larmor_radius_m(1000.0, 1, 1.0) == pytest.approx(0.0497, rel=0.02)
    base = DebrisModel(h2_pressure_pa=30.0)
    mag = DebrisModel(h2_pressure_pa=30.0, magnetic_field_t=1.0)
    assert mag.ion_transmission() < base.ion_transmission()


def test_cleaning_balance_and_lifetime():
    d = DebrisModel()
    assert d.cleaning_rate_nm_h() > 0
    assert np.isinf(d.equilibrium_sn_thickness_nm(10 * d.cleaning_rate_nm_h()))
    small = d.equilibrium_sn_thickness_nm(0.1 * d.cleaning_rate_nm_h())
    big = d.equilibrium_sn_thickness_nm(0.5 * d.cleaning_rate_nm_h())
    assert 0 < small < big
    # 1 nm Sn costs ~10-15 % reflectance
    assert 0.85 < DebrisModel.sn_film_transmission(1.0) < 0.9
    life_low = d.collector_lifetime_gpulses(0.1, 50e3)
    life_high = d.collector_lifetime_gpulses(2.0, 50e3)
    assert life_high < life_low


def test_collector_geometry_and_reflectance():
    c = EllipsoidalCollector()
    assert 0.6 < c.diameter_m < 0.7
    assert 4.5 < c.solid_angle_sr < 5.5
    aoi = c.aoi_deg(np.array([c.phi_min_deg, c.phi_max_deg]))
    assert aoi[0] < aoi[1] < 45
    assert 0.4 < c.average_reflectance() < 0.7
    assert 0.05 < c.if_na < 0.35
    assert c.etendue_mm2_sr(300e-6) < 3.3    # within illuminator acceptance


def test_if_power_default(src):
    p = src.inband_power_at_if_w()
    assert 200 < p < 300
    assert LPPSource.roadmap().inband_power_at_if_w() > 450


def test_pulse_series_statistics(src):
    rng = np.random.default_rng(0)
    e = src.pulse_energy_series(5000, rng)
    assert e.shape == (5000,)
    mean_power = e.mean() * src.rep_rate_hz
    assert mean_power == pytest.approx(src.inband_power_at_if_w(), rel=0.03)
    sig = np.std(e) / e.mean()
    assert 0.02 < sig < 0.15
    # reproducible
    e2 = src.pulse_energy_series(5000, np.random.default_rng(0))
    np.testing.assert_allclose(e, e2)


def test_dose_stability_improves_with_pulses_and_control(src):
    rng = np.random.default_rng(3)
    d50 = src.dose_error_3sigma(50, rng, n_total=10000)
    d400 = src.dose_error_3sigma(400, rng, n_total=10000)
    assert d400 < d50
    assert d400 == pytest.approx(d50 * np.sqrt(50 / 400), rel=0.3)
    n = src.pulses_in_slit(src.rep_rate_hz, 0.4)
    closed = src.dose_error_3sigma(n, rng, n_total=10000, controlled=True)
    assert closed < 0.5 * src.dose_error_3sigma(n, rng, n_total=10000)
    assert closed < 0.005


def test_availability(src):
    assert 0.8 < src.availability() < 1.0


def test_rep_rate_mismatch_rejected():
    with pytest.raises(ValueError):
        LPPSource(droplets=DropletGenerator(rep_rate_hz=40e3))
