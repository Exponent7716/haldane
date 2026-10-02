"""Tests for duv.resist using synthetic aerial images only."""

import numpy as np
import pytest

from duv.core import Grid
from duv.resist import (
    CAResist,
    FilmStack,
    Layer,
    N_BARC_193,
    N_RESIST_193,
    N_SILICON_193,
    bossung_curves,
    default_film_stack,
    dose_to_size,
    focus_exposure_matrix,
    ler_from_shot_noise,
    measure_cd,
    optimize_barc_thickness,
    process_window,
)

GRID = Grid(n=128, pixel=4.0)
PITCH = GRID.period / 2  # 256 nm, two periods on the grid


def dark_line_image(contrast=0.4, grid=GRID, pitch=PITCH):
    """Dark line centred at x = 0 (positive resist -> resist line at centre)."""
    return 0.5 - contrast * np.cos(2 * np.pi * grid.x / pitch)


def michelson(a):
    return (a.max() - a.min()) / (a.max() + a.min())


# --------------------------------------------------------------- film stack
def test_bare_interface_matches_fresnel():
    st = FilmStack([], substrate_n=1.5, ambient_n=1.0)
    assert st.reflectivity() == pytest.approx(0.04, abs=1e-12)
    assert st.transmissivity() == pytest.approx(0.96, abs=1e-12)
    # zero-thickness layer is invisible
    st2 = FilmStack([Layer("x", 0.0, 2.0 + 0.3j)], substrate_n=1.5, ambient_n=1.0)
    assert st2.reflectivity() == pytest.approx(0.04, abs=1e-12)


@pytest.mark.parametrize("pol", ["s", "p"])
@pytest.mark.parametrize("angle", [0.0, 0.4, 0.8])
def test_energy_conservation(pol, angle):
    st = default_film_stack()
    R = st.reflectivity(angle, pol)
    T = st.transmissivity(angle, pol)
    A = st.layer_absorption(angle, pol)
    assert 0 < R < 1 and 0 < T < 1 and np.all(A >= 0)
    assert R + T + A.sum() == pytest.approx(1.0, abs=1e-6)


def test_barc_optimisation_reduces_reflectivity():
    no_barc = default_film_stack().with_thickness("barc", 0.0).below("resist").reflectivity()
    t, r = optimize_barc_thickness()
    assert no_barc > 0.4  # resist on bare silicon reflects strongly
    assert r < 0.01
    assert 60 < t < 100
    # first minimum exists and is thinner but less effective
    t1, r1 = optimize_barc_thickness(first_minimum=True)
    assert 20 < t1 < 45 and r1 < 0.1


def test_standing_wave_in_resist():
    bare = default_film_stack().with_thickness("barc", 0.0)
    z = np.linspace(0, 90, 181)
    i_bare = bare.intensity_in_resist(z)
    i_barc = default_film_stack().intensity_in_resist(z)
    # standing-wave swing much larger without BARC; period ~ lambda / (2 n)
    assert michelson(i_bare) > 3 * michelson(i_barc)
    norm = bare.intensity_in_resist(z, normalize=True)
    assert np.mean(norm) == pytest.approx(1.0, rel=0.02)


# ---------------------------------------------------------------- resist
def test_acid_monotonic_in_dose():
    r = CAResist()
    img = dark_line_image()
    acids = [r.expose(img, d) for d in (5, 10, 20, 40, 80)]
    for a, b in zip(acids, acids[1:]):
        assert np.all(b >= a)
    assert np.all((acids[-1] >= 0) & (acids[-1] < 1))


def test_diffusion_lowers_contrast_and_conserves_acid():
    r = CAResist(diffusion_length_nm=20.0)
    acid = r.expose(dark_line_image(), 30.0)
    acid2d = np.tile(acid, (GRID.n, 1))
    diff = r.diffuse_acid(acid2d, GRID)
    assert michelson(diff[0]) < michelson(acid)
    assert diff.mean() == pytest.approx(acid2d.mean(), rel=1e-10)
    # a pure cosine is attenuated by exp(-2 (pi sigma f)^2)
    cosine = np.cos(2 * np.pi * GRID.x / PITCH)
    blurred = r.diffuse_acid(cosine, GRID)
    expect = np.exp(-2 * (np.pi * 20.0 / PITCH) ** 2)
    assert blurred.max() == pytest.approx(expect, rel=1e-6)


def test_mack_rate_and_develop_limits():
    r = CAResist()
    assert r.development_rate(1.0) == pytest.approx(r.r_min)
    assert r.development_rate(0.0) == pytest.approx(r.r_max + r.r_min)
    m = np.linspace(0, 1, 11)
    rem = r.develop(m)
    assert np.all(np.diff(rem) >= 0)  # more protection -> more resist remains
    assert rem[0] == 0.0 and rem[-1] > 0.9
    # depth-resolved development with uniform m equals the 2D result
    m3 = np.repeat(m[None, :], 8, axis=0)
    assert np.allclose(r.develop(m3, has_depth=True), rem, atol=1e-9)


@pytest.mark.parametrize("model", ["threshold", "mack"])
def test_cd_decreases_with_dose(model):
    r = CAResist()
    img = np.tile(dark_line_image(), (GRID.n, 1))  # 2D [iy, ix]
    cds = [r.process(img, d, GRID, model=model).cd for d in (16, 20, 24, 28)]
    assert all(np.isfinite(cds))
    assert all(b < a for a, b in zip(cds, cds[1:]))


def test_process_with_film_stack_runs():
    r = CAResist()
    res = r.process(dark_line_image(), 25.0, GRID, film_stack=default_film_stack(), nz=12)
    assert res.info["acid"].shape == (12, GRID.n)
    assert res.resist_profile.shape == (GRID.n,)
    assert 0 < res.cd < PITCH


# --------------------------------------------------------------- metrology
def test_measure_cd_exact_for_rectangle_and_trapezoid():
    x = GRID.x
    rect = (np.abs(x) < 42.0).astype(float)  # samples at multiples of 4 nm
    # crossings interpolated midway between samples 40 and 44 -> width 84
    assert measure_cd(rect, x) == pytest.approx(84.0)
    trap = np.clip((50.0 - np.abs(x)) / 10.0 + 0.5, 0, 1)
    assert measure_cd(trap, x) == pytest.approx(100.0, abs=1e-9)
    assert measure_cd(1 - trap, x, feature="space") == pytest.approx(100.0, abs=1e-9)
    # feature wrapped across the periodic boundary
    shifted = np.roll(trap, GRID.n // 2)
    assert measure_cd(shifted, x) == pytest.approx(100.0, abs=1e-9)
    assert np.isnan(measure_cd(np.ones_like(x), x))
    assert np.isnan(measure_cd(np.zeros_like(x), x))


def test_dose_to_size_hits_target():
    r = CAResist()
    img = dark_line_image()
    for model in ("threshold", "mack"):
        d = dose_to_size(img, 128.0, GRID, resist=r, model=model)
        assert np.isfinite(d)
        assert r.process(img, d, GRID, model=model).cd == pytest.approx(128.0, abs=0.05)
    assert dose_to_size(lambda: img, 100.0, GRID, resist=r) > dose_to_size(img, 128.0, GRID, resist=r)


def test_process_window_on_synthetic_defocus():
    r = CAResist()
    f0 = 150.0  # nm defocus scale of the contrast loss

    def image_at_focus(f):
        return dark_line_image(contrast=0.45 * np.exp(-((f - 20.0) / f0) ** 2))

    focuses = np.linspace(-200, 240, 23)
    d0 = dose_to_size(image_at_focus(20.0), 128.0, GRID, resist=r)
    doses = d0 * np.linspace(0.8, 1.2, 21)
    cd = focus_exposure_matrix(image_at_focus, focuses, doses, GRID, resist=r)
    assert cd.shape == (focuses.size, doses.size)
    pw = process_window(cd, focuses, doses, 128.0, tol=0.1, min_el_pct=5.0)
    assert 50.0 < pw["dof_nm"] < focuses[-1] - focuses[0]
    assert pw["exposure_latitude_pct"] >= 5.0
    assert pw["best_focus"] == pytest.approx(20.0, abs=25.0)
    assert pw["best_dose"] == pytest.approx(d0, rel=0.05)
    assert pw["max_el_pct"] >= pw["exposure_latitude_pct"]
    # tighter tolerance -> smaller window
    tight = process_window(cd, focuses, doses, 128.0, tol=0.05, min_el_pct=5.0)
    assert tight["dof_nm"] < pw["dof_nm"]
    # Bossung curves are symmetric about the best focus
    curves = bossung_curves(cd, focuses, doses)
    mid = curves[len(doses) // 2]
    assert mid["iso_focus"] == pytest.approx(20.0, abs=20.0)


def test_ler_shot_noise_scaling():
    l1 = ler_from_shot_noise(20.0, image_slope=0.05)
    l4 = ler_from_shot_noise(80.0, image_slope=0.05)
    assert l1 / l4 == pytest.approx(2.0, rel=1e-9)  # 1/sqrt(dose)
    assert ler_from_shot_noise(20.0, image_slope=0.1) == pytest.approx(l1 / 2)
    det = ler_from_shot_noise(30.0, 0.05, details=True)
    assert det["photons_per_nm2"] == pytest.approx(292.0, rel=0.01)  # 30 mJ/cm^2 at 193 nm
    assert 0.1 < det["ler_3sigma_nm"] < 10
