"""Physics sanity tests for euvsim.imaging (fast, small grids)."""
import math

import numpy as np
import pytest

from euvsim.imaging import (PELLICLE_PRESETS, Pellicle, ProjectionOptics, ReflectiveMask,
                            aerial_image, best_focus, telecentricity_error_mrad, contact_array, cra_overlap_ok, cutline,
                            focus_exposure_matrix, image_contrast, isolated_line, line_space,
                            measure_cd, nils, process_window, rectangles, tip_to_tip)

LAM = 13.5


# ---------------------------------------------------------------- sources
def dipole(sc, r=0.1, n=5, axis="x"):
    g = np.linspace(-r, r, n)
    pts = []
    for a in g:
        for b in g:
            if a * a + b * b <= r * r + 1e-12:
                for s in (-1, 1):
                    pts.append((s * sc + a, b) if axis == "x" else (a, s * sc + b))
    p = np.array(pts)
    return np.c_[p, np.full(len(p), 1.0 / len(p))]


def annular(si=0.4, so=0.8, n=21):
    g = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(g, g)
    r = np.hypot(X, Y)
    m = (r >= si) & (r <= so)
    return np.c_[X[m], Y[m], np.full(m.sum(), 1.0 / m.sum())]


def optimal_dipole_sigma(pitch, na):
    return min(LAM / (2 * pitch * na), 0.9)


LOW = ProjectionOptics.low_na()
HIGH = ProjectionOptics.high_na()


def ls_contrast(pitch, optics, src, dx=1.0, **kw):
    pat = line_space(pitch, pitch / 2, dx)
    return image_contrast(cutline(aerial_image(pat, dx, src, optics, **kw)))


# ---------------------------------------------------------------- patterns
def test_patterns_shapes_and_fill():
    p = line_space(32, 16, 1.0)
    assert p.shape == (1, 32) and abs(p.mean() - 0.5) < 1e-12
    p = line_space(32, 13.3, 1.0, orientation="horizontal", other_px=4)
    assert p.shape == (32, 4) and abs((1 - p[:, 0]).sum() - 13.3) < 1e-9   # sub-pixel CD
    c = contact_array(40, 20, 2.0, n_x=2, n_y=2)
    assert c.shape == (40, 40) and abs(c.mean() - 0.25) < 1e-12
    t = tip_to_tip(16, 20, 32, 2.0, field_y_nm=160)
    assert t.shape == (80, 16) and t[40, 8] == 0 and t[0, 8] == 1
    r = rectangles([(0.5, 0.5, 10.5, 10.5), (5.5, 5.5, 15.5, 15.5)], (40, 40), 1.0)
    assert abs(r.sum() - 175) < 1e-9
    assert isolated_line(20, 2.0, 200).shape == (1, 100)
    with pytest.raises(ValueError):
        line_space(33, 16, 2.0)


# ---------------------------------------------------------------- normalisation
def test_clear_field_is_unity():
    clear = np.ones((16, 16))
    for optics, src in ((LOW, annular()), (HIGH, dipole(0.6)),
                        (ProjectionOptics.low_na(flare=0.05, zernikes={7: 2.0, 9: 1.0}), annular())):
        I = aerial_image(clear, 1.0, src, optics, defocus_nm=50)
        assert np.allclose(I, 1.0, atol=1e-9)
    I = aerial_image(clear, 1.0, annular(), LOW, mask=ReflectiveMask())
    assert np.allclose(I, 1.0, atol=1e-9)


# ---------------------------------------------------------------- resolution
def test_low_na_resolution():
    # cutoff pitch lambda/(NA(1+sigma)) ~ 20.5 nm for sigma_max = 1
    assert LOW.min_pitch(1.0) == pytest.approx(LAM / 0.66)
    assert ls_contrast(32, LOW, dipole(optimal_dipole_sigma(32, 0.33))) > 0.6
    assert ls_contrast(32, LOW, annular()) > 0.4
    assert ls_contrast(20, LOW, dipole(0.9)) < 1e-6
    assert ls_contrast(20, LOW, annular()) < 1e-6


def test_high_na_resolves_16_to_20nm():
    for pitch in (16, 20):
        src = dipole(optimal_dipole_sigma(pitch, 0.55))
        assert ls_contrast(pitch, HIGH, src, dx=0.5) > 0.6
        assert ls_contrast(pitch, LOW, src, dx=0.5) < 1e-6
    assert not cra_overlap_ok(0.55, 5.355, 4.0) and cra_overlap_ok(0.55, 5.355, 8.0)
    assert cra_overlap_ok(0.33, 6.0, 4.0)


def test_rayleigh_k1_consistency():
    sigma = 0.8
    pmin = LOW.min_pitch(sigma)
    assert LOW.k1(pmin / 2) == pytest.approx(1 / (2 * (1 + sigma)))
    assert LOW.resolution(0.25) == pytest.approx(LAM * 0.25 / 0.33)
    # a dipole with outer sigma 0.8 images just above, not just below p_min
    src = dipole(0.75, r=0.05, n=3)
    above, below = 1.1 * pmin, 0.9 * pmin
    assert ls_contrast(above, LOW, src, dx=above / 32) > 0.2
    assert ls_contrast(below, LOW, src, dx=below / 32) < 1e-6
    assert LOW.rayleigh_dof() == pytest.approx(LAM / 0.33 ** 2)


# ---------------------------------------------------------------- focus / aberrations
def test_defocus_reduces_contrast_and_nils():
    pat = line_space(40, 20, 1.0)
    src = annular()
    I0 = cutline(aerial_image(pat, 1.0, src, LOW))
    I1 = cutline(aerial_image(pat, 1.0, src, LOW, defocus_nm=100))
    assert image_contrast(I1) < image_contrast(I0) - 0.05
    assert nils(I1, 1.0) < nils(I0, 1.0)
    assert nils(I0, 1.0) > 1.0


def test_z4_behaves_like_defocus():
    pat = line_space(40, 20, 1.0)
    src = annular()
    c4 = 4.0
    z = LOW.z4_equivalent_defocus(c4)
    Iz4 = cutline(aerial_image(pat, 1.0, src, ProjectionOptics.low_na(zernikes={4: c4})))
    Idf = cutline(aerial_image(pat, 1.0, src, LOW, defocus_nm=z))
    I0 = cutline(aerial_image(pat, 1.0, src, LOW))
    assert np.max(np.abs(Iz4 - Idf)) < 0.03
    assert image_contrast(Iz4) < image_contrast(I0) - 0.05
    # Z4 can compensate defocus
    Icomp = cutline(aerial_image(pat, 1.0, src, ProjectionOptics.low_na(zernikes={4: -c4}),
                                 defocus_nm=z))
    assert np.max(np.abs(Icomp - I0)) < 0.03


def test_flare_lowers_contrast():
    pat = line_space(32, 16, 1.0)
    src = annular()
    c0 = image_contrast(cutline(aerial_image(pat, 1.0, src, LOW)))
    c4 = image_contrast(cutline(aerial_image(pat, 1.0, src, ProjectionOptics.low_na(flare=0.04))))
    assert c4 < c0
    I = cutline(aerial_image(pat, 1.0, src, ProjectionOptics.low_na(flare=0.04)))
    assert I.min() > 0.04 * 0.5 * 0.9   # flare background ~ F * mean intensity


# ---------------------------------------------------------------- mask
def test_mask_optics():
    m = ReflectiveMask()
    assert 0.6 < m.blank_reflectance() < 0.75
    assert 0.005 < m.absorber_reflectance_ratio() < 0.06
    assert m.shadow_bias_wafer_nm() == pytest.approx(2 * 60 * math.tan(math.radians(6)) / 4)
    hn = ReflectiveMask(magnification=(4, 8), cra_deg=5.355)
    assert hn.shadow_bias_wafer_nm() < m.shadow_bias_wafer_nm()
    psm = ReflectiveMask.preset("lowN_PSM")
    assert abs(abs(psm.absorber_phase_deg()) - 180) < 15
    assert ReflectiveMask.preset("Ni").absorber_reflectance_ratio() < m.absorber_reflectance_ratio()
    assert m.mask_size_nm(16, "y") == 64


def test_pellicle_transmission_range():
    for p in PELLICLE_PRESETS.values():
        assert 0.80 <= p.double_pass_transmission() <= 0.95
    assert Pellicle("pSi", 50).double_pass_transmission() < Pellicle("pSi", 30).double_pass_transmission()
    m = ReflectiveMask(pellicle=PELLICLE_PRESETS["CNT"])
    assert m.effective_reflectance() < m.blank_reflectance()


def test_shadowing_bias_sign():
    """Incidence plane = y: horizontal absorber lines print wider than vertical."""
    pitch, cd, dx = 64, 32, 1.0
    src = annular()
    mask = ReflectiveMask()
    v = cutline(aerial_image(line_space(pitch, cd, dx, "vertical"), dx, src, LOW, mask=mask), "x")
    h = cutline(aerial_image(line_space(pitch, cd, dx, "horizontal"), dx, src, LOW, mask=mask), "y")
    th = 0.3
    cd_v, cd_h = measure_cd(v, dx, th), measure_cd(h, dx, th)
    assert cd_h > cd_v
    assert 0.3 * mask.shadow_bias_wafer_nm() < cd_h - cd_v < 2.0 * mask.shadow_bias_wafer_nm()
    nom = ReflectiveMask(m3d=False)
    h0 = cutline(aerial_image(line_space(pitch, cd, dx, "horizontal"), dx, src, LOW, mask=nom), "y")
    assert measure_cd(h0, dx, th) == pytest.approx(cd_v, abs=1e-6)


def test_m3d_best_focus_shift_and_telecentricity():
    src = annular(0.3, 0.9, 15)
    f = np.linspace(-150, 150, 21)
    hor = line_space(32, 16, 2.0, orientation="horizontal")
    ver = line_space(32, 16, 2.0, orientation="vertical")
    bf_binary = best_focus(hor, 2.0, src, LOW, f, axis="y")
    bf_m3d = best_focus(hor, 2.0, src, LOW, f, mask=ReflectiveMask(), axis="y")
    assert abs(bf_binary) < 1.0
    assert abs(bf_m3d - bf_binary) > 3.0
    # through-focus pattern shift only for shadowed (horizontal) features
    t_h = telecentricity_error_mrad(hor, 2.0, src, LOW, ReflectiveMask(), axis="y")
    t_v = telecentricity_error_mrad(ver, 2.0, src, LOW, ReflectiveMask(), axis="x")
    t_ni = telecentricity_error_mrad(hor, 2.0, src, LOW, ReflectiveMask.preset("Ni"), axis="y")
    assert abs(t_h) > 0.3 and abs(t_v) < 1e-6
    assert abs(t_ni) < abs(t_h)          # thinner high-k absorber -> less M3D


# ---------------------------------------------------------------- process window
def test_process_window():
    pat = line_space(40, 20, 1.0)
    src = annular()
    focuses = np.linspace(-300, 300, 13)
    doses = np.linspace(0.8, 1.2, 9)
    I = cutline(aerial_image(pat, 1.0, src, LOW))
    th = 0.5 * (I.max() + I.min())
    cd = focus_exposure_matrix(pat, 1.0, src, LOW, focuses, doses, th)
    assert cd.shape == (13, 9)
    assert cd[6, 4] == pytest.approx(20, abs=1.0)
    # dark lines shrink with dose (Bossung slope) and degrade through focus
    assert cd[6, -1] < cd[6, 0]
    pw = process_window(cd, focuses, doses, 20.0)
    assert abs(pw["best_focus_nm"]) <= 40
    assert pw["el_best"] > 0.05
    assert 100 <= pw["dof_nm"] <= 400          # ~ Rayleigh DOF lambda/NA^2 = 124 nm
    assert pw["el_vs_focus"][0] == 0 < pw["el_best"]


def test_misc_optics():
    assert 0.08 < LOW.transmission() < 0.2      # ~R^6
    assert HIGH.transmission() < LOW.transmission()
    # central obscuration blocks low-sigma on-axis light -> clear-field still 1, but
    # a source entirely inside the obscuration is dark field
    with pytest.raises(ValueError):
        aerial_image(np.ones((4, 4)), 1.0, np.array([[0.0, 0.0, 1.0]]), HIGH)
