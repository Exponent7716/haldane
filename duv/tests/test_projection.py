"""Tests for duv.projection (lens, immersion, Abbe imaging)."""

import dataclasses
import time

import numpy as np
import pytest

from duv.core import N_WATER_193, Grid, ImagingSettings, Mask, SourceMap
from duv.projection import (
    ImmersionHood,
    LensHeating,
    ProjectionLens,
    aerial_image,
    depth_of_focus,
    image_contrast,
    min_pitch,
    nils,
    pupil_function,
    resolution_limit,
    zernike_fringe,
)


# --- helpers (simple sources built locally; illumination.py not used) --------
def conventional(sigma, n=41, pol="unpolarized"):
    s = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(s, s)
    return SourceMap(s, (np.hypot(X, Y) <= sigma).astype(float), pol)


def dipole_x(center=0.8, radius=0.15, n=41, pol="Y"):
    s = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(s, s)
    inten = (np.hypot(X - center, Y) <= radius) | (np.hypot(X + center, Y) <= radius)
    return SourceMap(s, inten.astype(float), pol)


def line_space(pitch, n=128, periods=4):
    """Vertical lines/spaces (vary along x), 50 % duty, chrome lines on clear."""
    g = Grid(n, pitch * periods / n)
    X, _ = g.mesh()
    t = (np.mod(X, pitch) < pitch / 2).astype(complex)
    return Mask(g, t)


def cut(img):
    return img[img.shape[0] // 2]


# --- tests --------------------------------------------------------------------
def test_clear_mask_gives_unit_intensity():
    m = Mask(Grid(128, 4.0), np.ones((128, 128), complex))
    for src, pol in [(conventional(0.7), "unpolarized"), (dipole_x(), "X"),
                     (conventional(0.9), "TE"), (conventional(0.5), "scalar")]:
        st = ImagingSettings(defocus=80.0, zernikes={9: 0.05, 7: 0.03})
        I = aerial_image(m, src, st, polarization=pol)
        assert np.allclose(I, 1.0, atol=1e-9)
    # flare keeps a clear field at 1
    I = aerial_image(m, conventional(0.5), ImagingSettings(flare=0.05))
    assert np.allclose(I, 1.0, atol=1e-9)


def test_pitch_below_resolution_gives_flat_image():
    sigma = 0.5
    p_min = min_pitch(193.368, 1.35, sigma)
    assert 95 < p_min < 96
    I = aerial_image(line_space(60.0), conventional(sigma), ImagingSettings())
    assert image_contrast(cut(I)) < 1e-6
    assert np.allclose(I, 0.25, atol=1e-6)  # only the zeroth order (|0.5|^2)


def test_90nm_pitch_with_dipole_resolves():
    I = aerial_image(line_space(90.0), dipole_x(pol="Y"), ImagingSettings())
    assert image_contrast(cut(I)) > 0.5
    # conventional sigma 0.5 cannot resolve 90 nm pitch at NA 1.35
    I2 = aerial_image(line_space(90.0), conventional(0.5), ImagingSettings())
    assert image_contrast(cut(I2)) < 1e-6


def test_tm_polarization_loses_contrast_at_high_na():
    m = line_space(90.0)
    c_te = image_contrast(cut(aerial_image(m, dipole_x(), ImagingSettings(), polarization="Y")))
    c_tm = image_contrast(cut(aerial_image(m, dipole_x(), ImagingSettings(), polarization="X")))
    c_un = image_contrast(cut(aerial_image(m, dipole_x(), ImagingSettings(), polarization="unpolarized")))
    c_sc = image_contrast(cut(aerial_image(m, dipole_x(), ImagingSettings(), polarization="scalar")))
    assert c_te > 0.85 and c_tm < 0.3 and c_tm < c_un < c_te
    assert abs(c_te - c_sc) < 0.02  # TE interferes like scalar waves


def _bossung(zernikes, foci, pitch=200.0):
    m = line_space(pitch)
    src = conventional(0.3, pol="scalar")
    return np.array([image_contrast(cut(aerial_image(
        m, src, ImagingSettings(defocus=f, zernikes=zernikes)))) for f in foci])


def test_defocus_reduces_contrast_symmetrically():
    foci = np.array([-150.0, -75.0, 0.0, 75.0, 150.0])
    c = _bossung({}, foci)
    assert c[2] > c[1] > c[0] and c[2] > c[3] > c[4]
    assert np.allclose(c, c[::-1], atol=1e-9)


def test_spherical_aberration_shifts_best_focus():
    foci = np.linspace(-200, 200, 41)
    c0 = _bossung({}, foci)
    c9 = _bossung({9: 0.05}, foci)
    assert abs(foci[np.argmax(c0)]) < 1e-9
    best = foci[np.argmax(c9)]
    assert abs(best) >= 20.0
    assert not np.allclose(c9, c9[::-1], atol=1e-3)  # asymmetric Bossung


def test_zernike_orthogonality_and_convention():
    n = 200
    r = (np.arange(n) + 0.5) / n
    t = (np.arange(4 * n) + 0.5) / (4 * n) * 2 * np.pi
    R, T = np.meshgrid(r, t)
    dA = R * (1.0 / n) * (2 * np.pi / (4 * n))
    Z = np.array([zernike_fringe(j, R, T).ravel() for j in range(1, 38)])
    G = (Z * dA.ravel()) @ Z.T
    off = G - np.diag(np.diag(G))
    assert np.max(np.abs(off)) < 1e-3 * np.max(np.diag(G))
    # unit amplitude at the pupil edge, known forms
    rho = np.array([0.3, 0.8])
    assert np.allclose(zernike_fringe(4, rho, 0), 2 * rho**2 - 1)
    assert np.allclose(zernike_fringe(9, rho, 0), 6 * rho**4 - 6 * rho**2 + 1)
    assert np.allclose(zernike_fringe(7, rho, 0), 3 * rho**3 - 2 * rho)
    assert np.isclose(zernike_fringe(37, 1.0, 0.3), 1.0)
    assert np.isclose(zernike_fringe(16, 1.0, 0.0), 1.0)


def test_immersion_allows_na_above_one():
    g = Grid(128, 4.0)
    st = ImagingSettings(na=1.35, n_immersion=N_WATER_193, defocus=50.0)
    P = pupil_function(g, st)
    fx, fy = g.freq_mesh()
    fr = np.hypot(fx, fy)
    inside = np.abs(P) > 0
    assert np.all(fr[inside] <= 1.35 / st.wavelength + 1e-12)
    assert np.any(fr[inside] > 1.0 / st.wavelength)  # frequencies beyond dry limit
    assert np.allclose(np.abs(P[inside]), 1.0)
    with pytest.raises(ValueError):
        pupil_function(g, dataclasses.replace(st, n_immersion=1.0))
    # shifted pupil = P(f + sigma NA / lambda)
    Ps = pupil_function(g, ImagingSettings(), sx=0.5)
    assert np.any(np.abs(Ps) > 0) and not np.array_equal(np.abs(Ps) > 0, np.abs(P) > 0)


def test_flare_and_laser_bandwidth():
    m = line_space(200.0)
    src = conventional(0.3, pol="scalar")
    st = ImagingSettings()
    I0 = aerial_image(m, src, st)
    If = aerial_image(m, src, dataclasses.replace(st, flare=0.1))
    assert np.allclose(If, 0.9 * I0 + 0.1 * 0.5)
    lam = st.wavelength + np.array([-0.5e-3, 0.0, 0.5e-3])  # +-0.5 pm
    spec = (lam, [1, 2, 1])
    Ib = aerial_image(m, src, st, spectrum=spec, chromatic_focus_nm_per_pm=200.0)
    Ia = aerial_image(m, src, st, spectrum=spec, chromatic_focus_nm_per_pm=0.0)
    assert image_contrast(cut(Ib)) < image_contrast(cut(I0)) - 1e-3
    assert abs(image_contrast(cut(Ia)) - image_contrast(cut(I0))) < 1e-4


def test_metrics_and_scaling_laws():
    assert np.isclose(resolution_limit(0.25, 193.368, 1.35), 35.81, atol=0.01)
    dof_water = depth_of_focus(1.0, 193.368, 0.9, N_WATER_193)
    dof_dry = depth_of_focus(1.0, 193.368, 0.9, 1.0)
    assert dof_water > dof_dry  # immersion DOF gain at fixed NA
    x = np.linspace(-100, 100, 2001)
    I = 0.5 + 0.5 * np.cos(2 * np.pi * x / 200)
    assert np.isclose(image_contrast(I), 1.0)
    # edge at x = 50 (I = 0.5): slope = 0.5*2pi/200, NILS = cd * slope / I
    assert np.isclose(nils(I, x, 50.0, cd=100.0), 100 * np.pi / 200 / 0.5, rtol=1e-3)


def test_performance_128_grid_300_points():
    m = line_space(90.0)
    s = np.linspace(-1, 1, 35)
    X, Y = np.meshgrid(s, s)
    r = np.hypot(X, Y)
    src = SourceMap(s, ((r >= 0.55) & (r <= 0.9)).astype(float), "unpolarized")
    n_pts = len(src.points()[0])
    assert 200 <= n_pts <= 600
    t0 = time.perf_counter()
    aerial_image(m, src, ImagingSettings(defocus=30, zernikes={9: 0.01}))
    assert time.perf_counter() - t0 < 3.0


def test_projection_lens_fingerprint_and_manipulators():
    lens = ProjectionLens()
    a0 = lens.aberrations_at(0.0)
    a1 = lens.aberrations_at(12.0)
    assert set(range(2, 17)) <= set(a0)
    assert max(abs(v) for v in a0.values()) < 0.05  # milliwave-level
    assert any(abs(a0[j] - a1[j]) > 1e-4 for j in a0)  # varies along slit
    before = lens.field_rms(terms=range(2, 10))
    lens.optimize_manipulators()
    after = lens.field_rms(terms=range(2, 10))
    assert after < before
    lens.set_manipulators(Z4=0.01)
    assert lens.manipulators[4] == 0.01
    with pytest.raises(ValueError):
        lens.set_manipulators({20: 0.01})
    st = lens.imaging_settings(5.0)
    assert st.na == lens.na and 9 in st.zernikes


def test_lens_heating_drift_and_feedforward():
    lh = LensHeating()
    p = lh.absorbed_power(30.0, reticle_transmission=0.5)
    assert 0.01 < p < 10
    d_short = lh.drift(10.0, 1.0, "conventional")
    d_long = lh.drift(1e5, 1.0, "conventional")
    assert abs(d_long[4]) > abs(d_short[4]) > 0
    assert np.isclose(d_long[4], lh.sensitivity["conventional"][4], rtol=1e-6)
    dip = lh.drift(1e4, 1.0, "dipole_x")
    assert abs(dip[5]) > 0 and 5 not in d_long  # dipole -> astigmatism
    ff = lh.feedforward(1e4, 1.0, "dipole_x", model_error=0.1)
    assert abs(ff["residual"][5]) < 0.2 * abs(ff["drift"][5])
    # time trace: heating then cooling
    t = np.linspace(0, 3000, 301)
    pw = np.where(t < 1500, 1.0, 0.0)
    tr = lh.simulate(t, pw, "conventional")[4]
    assert tr[150] > tr[0] and tr[-1] < tr[150]


def test_immersion_hood_focus_shift():
    hood = ImmersionHood()
    dz = hood.focus_shift_nm(0.01)  # 10 mK warmer (dn = -1e-6)
    paraxial = 1e6 * 1e-6 / N_WATER_193  # -gap * dn / n
    low_na = ImmersionHood(na=0.2).focus_shift_nm(0.01)
    assert np.isclose(low_na, paraxial, rtol=0.02)
    assert dz > paraxial  # high-NA rays have longer paths in the gap
    assert np.isclose(hood.focus_shift_nm(-0.01), -dz, rtol=1e-3)
    assert abs(hood.spherical_waves(0.01)) > 0
    z = hood.wavefront_zernikes(0.01)
    assert abs(z[4]) > abs(z[9])
    assert 0.99 < hood.transmission() < 1.0
