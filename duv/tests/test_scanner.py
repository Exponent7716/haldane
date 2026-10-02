import numpy as np
import pytest

from duv import reticle
from duv.scanner import DUVScanner, ScannerConfig, _z4_waves_to_nm


@pytest.fixture(scope="module")
def scanner():
    return DUVScanner(ScannerConfig(seed=1))


@pytest.fixture(scope="module")
def mask(scanner):
    g = scanner.config.grid
    return reticle.make_mask(reticle.line_space(g, 45, 128), g, "attpsm")


def test_clear_field_image_is_unity(scanner):
    g = scanner.config.grid
    clear = reticle.make_mask(np.zeros((g.n, g.n)), g, "binary")
    img = scanner.aerial_image(clear)
    assert np.allclose(img, 1.0, atol=1e-6)


def test_dose_to_size_prints_target(scanner, mask):
    dose = scanner.dose_to_size(mask, 45.0)
    assert 1 < dose < 200
    assert scanner.expose_field(mask, dose).cd == pytest.approx(45.0, abs=0.1)


def test_defocus_and_msd_degrade_image(scanner, mask):
    def contrast(img):
        row = img[img.shape[0] // 2]
        return (row.max() - row.min()) / (row.max() + row.min())

    c0 = contrast(scanner.aerial_image(mask))
    assert contrast(scanner.aerial_image(mask, defocus_nm=150)) < c0
    assert contrast(scanner.aerial_image(mask, msd_nm=8)) < c0


def test_z4_conversion_matches_pupil_defocus():
    # 100 nm of defocus at NA 1.35 corresponds to a sizeable negative Z4
    dz = _z4_waves_to_nm(-0.1, 1.35, 1.4366)
    assert dz > 0
    assert _z4_waves_to_nm(0.0, 1.35, 1.4366) == 0.0


def test_process_window(scanner, mask):
    pw = scanner.process_window(mask, 45.0)
    assert pw["dof_nm"] > 30
    assert abs(pw["best_focus"]) < 40


def test_expose_wafer_summary(scanner, mask):
    w = scanner.expose_wafer(mask, 45.0, n_fields=4, slit_points_mm=(0.0,))
    assert w["field_cd_nm"].shape == (4, 1)
    assert abs(w["cd_mean_nm"] - 45.0) < 2.0
    assert w["overlay_nm"].shape == (4, 2)
    assert w["overlay_stats"]["x"]["m+3s"] < 20
    assert 100 < w["throughput"]["wph"] < 400
