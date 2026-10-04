"""Tests for euvsim.resist (synthetic aerial images only)."""
import math

import numpy as np
import pytest

from euvsim.constants import photons_per_nm2
from euvsim.resist import (CAR, MOR, ResistProcess, analytic_ler_3sigma, count_feature_defects,
                           layer_absorption_fractions, ler_dose_scaling, measure_lines, nils)


def line_space(pitch=32.0, nx=128, ny=256, dx=1.0, contrast=0.45):
    """Sinusoidal L/S aerial image (lines along y), clear peaks at pitch/2 + k*pitch."""
    x = np.arange(nx) * dx
    prof = 0.5 + contrast * np.cos(2 * np.pi * (x - pitch / 2) / pitch)
    return np.tile(prof, (ny, 1))


def contact_array(pitch=40.0, n=160, sigma=9.0):
    y, x = np.mgrid[0:n, 0:n] * 1.0
    dxp = (x % pitch) - pitch / 2
    dyp = (y % pitch) - pitch / 2
    return 0.05 + 0.85 * np.exp(-(dxp**2 + dyp**2) / (2 * sigma**2))


def test_mean_photon_count_matches_dose():
    aer = np.ones((64, 64))
    dose = 40.0
    res = ResistProcess(CAR).expose(aer, 1.0, dose, np.random.default_rng(0), metrics=None)
    per_nm2_per_dose = res.exposure.incident.mean() / dose
    assert per_nm2_per_dose == pytest.approx(0.68, rel=0.02)
    assert per_nm2_per_dose == pytest.approx(photons_per_nm2(1.0), rel=0.01)
    # absorbed fraction follows Beer-Lambert
    frac = res.exposure.absorbed.sum() / res.exposure.incident.sum()
    assert frac == pytest.approx(CAR.absorbance, rel=0.03)
    # incident counts are Poisson: variance == mean
    inc = res.exposure.incident
    assert inc.var() / inc.mean() == pytest.approx(1.0, abs=0.06)


def test_beer_lambert_layers():
    f = layer_absorption_fractions(MOR)
    assert np.all(np.diff(f) < 0)
    assert f.sum() == pytest.approx(MOR.absorbance)


def test_mor_absorbs_more_than_car():
    assert MOR.absorbance > 2.0 * CAR.absorbance
    assert 0.10 < CAR.absorbance < 0.2
    aer = np.ones((32, 32))
    rc = ResistProcess(CAR).expose(aer, 1.0, 30.0, np.random.default_rng(1), metrics=None)
    rm = ResistProcess(MOR).expose(aer, 1.0, 30.0, np.random.default_rng(1), metrics=None)
    assert rm.absorbed_photons_per_nm2 > 2.0 * rc.absorbed_photons_per_nm2


def test_ler_decreases_with_dose_inverse_sqrt():
    aer = line_space(pitch=64.0)
    pr = ResistProcess(CAR)
    doses = [30.0, 45.0, 60.0]
    lers = []
    for d in doses:
        lers.append(np.mean([pr.expose(aer, 1.0, d, np.random.default_rng(s)).metrics.ler_3sigma
                             for s in range(3)]))
    assert lers[0] > lers[1] > lers[2]
    slope = np.polyfit(np.log(doses), np.log(lers), 1)[0]
    assert -0.85 < slope < -0.25
    assert ler_dose_scaling(2.0, 30.0, 120.0) == pytest.approx(1.0)
    a1 = analytic_ler_3sigma(30.0, CAR.absorbance, CAR.thickness_nm, 0.1, 5.0, 2.5)
    a2 = analytic_ler_3sigma(120.0, CAR.absorbance, CAR.thickness_nm, 0.1, 5.0, 2.5)
    assert a1 / a2 == pytest.approx(2.0)


def test_larger_blur_lowers_latent_nils():
    pitch = 32.0
    aer = line_space(pitch, ny=4)
    vals = []
    for diff in (2.0, 5.0, 8.0):
        r = ResistProcess(CAR.with_(diffusion_nm=diff)).expose(aer, 1.0, 40.0, None, metrics=None)
        latent = r.bake.acid_diffused.mean(axis=0)[0]
        vals.append(nils(latent, 1.0, pitch / 2, pitch / 4))
    assert vals[0] > vals[1] > vals[2]
    assert nils(aer[0], 1.0, 16.0, 8.0) > vals[0]


def test_cd_increases_with_dose_for_spaces():
    aer = line_space()
    pr = ResistProcess(CAR)
    cds = [pr.expose(aer, 1.0, d, None).metrics.cd_mean for d in (26.0, 32.0, 40.0, 50.0)]
    assert np.all(np.diff(cds) > 0)
    noisy = [pr.expose(aer, 1.0, d, np.random.default_rng(2)).metrics.cd_mean for d in (26.0, 50.0)]
    assert noisy[1] > noisy[0]
    # negative-tone MOR: exposed feature is a line, also grows with dose
    pm = ResistProcess(MOR)
    cdm = [pm.expose(aer, 1.0, d, None).metrics.cd_mean for d in (30.0, 45.0, 60.0)]
    assert np.all(np.diff(cdm) > 0)


def test_dose_to_size_and_mack():
    aer = line_space()
    pr = ResistProcess(CAR)
    d = pr.dose_to_size(aer, 1.0, 16.0)
    assert 20.0 < d < 70.0
    assert pr.expose(aer, 1.0, d).metrics.cd_mean == pytest.approx(16.0, abs=0.2)
    mack = ResistProcess(CAR, develop_model="mack").expose(aer, 1.0, d).metrics.cd_mean
    assert mack == pytest.approx(16.0, abs=2.0)
    r = pr.expose(aer, 1.0, d, np.random.default_rng(0))
    assert np.all(r.resist_remaining == ~r.printed)
    m = r.metrics
    assert 0.5 < m.ler_3sigma < 6.0 and m.lwr_3sigma > m.ler_3sigma


def test_contact_lcdu_and_defects_fall_with_dose():
    aer = contact_array()
    pr = ResistProcess(CAR)
    d = pr.dose_to_size(aer, 1.0, 20.0, feature="contact")
    nom = pr.expose(aer, 1.0, d, None, metrics="contact")
    assert nom.metrics.n_found == 16
    r = pr.expose(aer, 1.0, d, np.random.default_rng(3), metrics="contact")
    assert 0.5 < r.metrics.lcdu_3sigma < 8.0
    p = [pr.defect_probability(aer, 1.0, f * d, np.random.default_rng(0), n_trials=15,
                               kind="contact", nominal=nom.printed).p_fail
         for f in (0.65, 0.75, 1.0)]
    assert p[0] > p[1] > p[2]
    assert p[0] > 0.2 and p[2] < 0.05


def test_line_defect_counting():
    nominal = np.zeros((10, 30), bool)
    nominal[:, 5:10] = nominal[:, 15:20] = True
    pr = nominal.copy()
    pr[4, 5:10] = False          # micro-break in feature 0
    pr[7, 10:15] = True          # bridge between features
    o, b = count_feature_defects(pr, nominal, "line")
    assert o.tolist() == [True, False] and b.tolist() == [True, True]


def test_measure_lines_on_ideal_field():
    x = np.arange(100.0)
    f = np.tile(np.cos(2 * np.pi * (x - 25) / 50) , (5, 1))  # widths = 25 nm
    m = measure_lines(f, 1.0)
    assert m.cd_mean == pytest.approx(25.0, abs=0.3)
    assert m.ler_3sigma < 1e-9


def test_reproducible_with_seed():
    aer = line_space(ny=64)
    pr = ResistProcess(MOR)
    a = pr.expose(aer, 1.0, 40.0, np.random.default_rng(42))
    b = pr.expose(aer, 1.0, 40.0, np.random.default_rng(42))
    c = pr.expose(aer, 1.0, 40.0, np.random.default_rng(43))
    assert np.array_equal(a.printed, b.printed)
    assert a.metrics.ler_3sigma == b.metrics.ler_3sigma
    assert not np.array_equal(a.exposure.absorbed, c.exposure.absorbed)
