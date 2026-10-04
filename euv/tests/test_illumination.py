import numpy as np
import pytest

from euvsim import illumination as il


SHAPES = [
    (il.conventional, dict(sigma=0.8)),
    (il.annular, dict(sigma_in=0.5, sigma_out=0.9)),
    (il.dipole, dict(sigma_in=0.6, sigma_out=0.9, opening_angle_deg=60, orientation="x")),
    (il.dipole, dict(sigma_in=0.6, sigma_out=0.9, opening_angle_deg=60, orientation="y")),
    (il.quadrupole, dict(kind="quasar")),
    (il.quadrupole, dict(kind="cquad")),
    (il.leaf, dict()),
]


@pytest.mark.parametrize("fn,kw", SHAPES)
def test_source_point_contract(fn, kw):
    p = fn(**kw)
    assert p.ndim == 2 and p.shape[1] == 3
    assert np.isclose(p[:, 2].sum(), 1.0)
    assert np.all(p[:, 2] >= 0)
    assert np.all(np.hypot(p[:, 0], p[:, 1]) <= 1 + 1e-12)
    il.validate(p)
    cx, cy = il.centroid(p)
    assert abs(cx) < 1e-9 and abs(cy) < 1e-9


def test_annular_hole_and_pfr():
    p = il.annular(0.5, 0.8, step=0.01)
    r = np.hypot(p[:, 0], p[:, 1])
    assert r.min() >= 0.5 and r.max() <= 0.8
    assert il.pupil_fill_ratio(p, 0.01) == pytest.approx(0.8 ** 2 - 0.5 ** 2, rel=0.03)
    assert il.pupil_fill_ratio(il.conventional(0.6, 0.01), 0.01) == pytest.approx(0.36, rel=0.03)


def test_dipole_symmetry_and_metrics():
    p = il.dipole(0.6, 0.9, 40, "x")
    # mirror symmetric in x and y
    a = {(round(x, 9), round(y, 9)) for x, y in p[:, :2]}
    assert all((round(-x, 9), round(y, 9)) in a for x, y in a)
    assert all((round(x, 9), round(-y, 9)) in a for x, y in a)
    assert il.pole_balance(p, 2, 0.0) < 1e-9
    assert il.ellipticity(p) > 0.99
    assert il.ellipticity(il.dipole(0.6, 0.9, 40, "y")) < -0.99
    assert abs(il.ellipticity(il.conventional(0.8))) < 1e-9
    q = il.quasar()
    assert il.pole_balance(q, 4, 45.0) < 1e-9
    assert abs(il.ellipticity(q)) < 1e-9


def test_from_bitmap_roundtrip():
    bm = il.to_bitmap(il.annular(0.4, 0.7), n=201)
    p = il.from_bitmap(bm)
    il.validate(p)
    r = np.hypot(p[:, 0], p[:, 1])
    assert np.all(r > 0.35) and np.all(r < 0.75)


def test_flexpupil_redistribution():
    fp = il.FlexPupil(n_pupil_facets=400)
    assert fp.n_field_facets * 2 == fp.pfm.n
    rx = fp.configure(il.shape_mask("dipole", sigma_in=0.0, sigma_out=1.0, opening_angle_deg=90))
    ry = fp.configure(il.shape_mask("dipole", sigma_in=0.0, sigma_out=1.0, opening_angle_deg=90,
                                    orientation="y"))
    # with rotate-90 pairing every field facet can serve both x- and y-dipole: no loss
    assert rx.efficiency == pytest.approx(1.0)
    assert ry.efficiency == pytest.approx(1.0)
    assert il.ellipticity(rx.points) > 0.9 and il.ellipticity(ry.points) < -0.9
    il.validate(rx.points)
    # masking a filled pupil to the same shape loses half the light
    assert il.masking_efficiency(0.4, 0.8) == pytest.approx(0.5)


def test_uniformity_improves_with_facets():
    u = il.uniformity_vs_facets([25, 100, 400], n_trials=6, rng=np.random.default_rng(1))
    assert u[0] > u[1] > u[2]
    # ~1/sqrt(N): 16x more facets -> ~4x better
    assert 2.5 < u[0] / u[2] < 7


def test_arc_slit_geometry():
    s = il.ArcSlit(26.0, 2.0, 30.0)
    assert s.sagitta_mm == pytest.approx(30 - np.sqrt(30 ** 2 - 13 ** 2))
    x, h = s.scan_integrated_height()
    assert np.allclose(h, 2.0, rtol=0.02)
    assert s.contains(0.0, 0.0) and not s.contains(0.0, -3.0)


def test_unicom_reduces_nonuniformity():
    x = np.linspace(-13, 13, 261)
    prof = 1 + 0.03 * x / 13 - 0.02 * (x / 13) ** 2 + 0.01 * np.sin(x / 2)
    res = il.Unicom().correct(x, prof)
    assert res.uniformity_after < 0.5 * res.uniformity_before
    assert 0 <= res.light_loss < 0.1
    assert np.all(res.insertion_mm >= 0)


def test_transmission_plausible():
    t = il.IlluminatorTransmission()
    assert 0.15 < t.total() < 0.35
    assert 0.6 < il.multilayer_reflectance(10.0) < 0.74
    assert 0.75 < il.grazing_ru_reflectance(12.0) < 0.95
    assert il.grazing_ru_reflectance(30.0) < il.grazing_ru_reflectance(10.0)


def test_illuminator_end_to_end():
    ill = il.Illuminator(if_power_w=250.0)
    r = ill.illuminate("conventional", sigma=0.9)
    il.validate(r.source_points)
    assert r.etendue_ok
    assert 0.15 < r.total_transmission < 0.35
    assert r.power_at_reticle_w == pytest.approx(250 * r.total_transmission)
    assert r.uniformity_corrected <= r.uniformity_raw
    assert 0.1 < ill.min_pfr_without_loss() < 0.3
    # small-PFR setting violates etendue and loses light
    small = ill.illuminate("dipole", sigma_in=0.8, sigma_out=0.9, opening_angle_deg=40)
    assert not small.etendue_ok
    assert small.power_at_reticle_w < r.power_at_reticle_w
    # FlexPupil keeps more light than masking for a dipole
    d = il.Illuminator(pupil_method="flexpupil").illuminate(
        "dipole", sigma_in=0.0, sigma_out=1.0, opening_angle_deg=90)
    assert d.pupil_efficiency_flex > d.pupil_efficiency_mask
