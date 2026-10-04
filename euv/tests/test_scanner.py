import math

import pytest

from euvsim.imaging import Pellicle
from euvsim.scanner import EUVScanner


@pytest.fixture(scope="module")
def low_na():
    return EUVScanner()


def test_photon_budget_chain(low_na):
    b = low_na.photon_budget()
    assert 200 < b["P_IF_W"] < 320
    assert 0.005 < b["IF_to_wafer"] < 0.05
    assert math.isclose(b["P_wafer_W"], b["P_IF_W"] * b["T_illuminator"] * b["R_mask"]
                        * b["T_pellicle"] * b["T_POB"] * b["T_gas"], rel_tol=1e-9)


def test_pellicle_costs_power():
    bare = EUVScanner().photon_budget()["P_wafer_W"]
    pel = EUVScanner(pellicle=Pellicle()).photon_budget()["P_wafer_W"]
    assert pel < bare


def test_throughput_falls_with_dose(low_na):
    w = [low_na.throughput(d)["WPH"] for d in (20, 40, 80)]
    assert w[0] >= w[1] > w[2]
    assert 50 < w[0] < 250
    assert low_na.throughput(80)["limited_by"] == "source"
    assert 50 < low_na.throughput(20)["n_fields"] < 130


def test_high_na_has_less_power_at_wafer(low_na):
    assert EUVScanner.high_na().photon_budget()["P_wafer_W"] < low_na.photon_budget()["P_wafer_W"]


def test_end_to_end_lines_print_at_target(low_na):
    r = low_na.print_lines(32, 16)
    assert abs(r["cd_nm"] - 16) < 2.0
    assert r["image_nils"] > 1.5
    assert 0 < r["ler_3sigma_nm"] < 6
    assert 10 < r["dose_mJ_cm2"] < 150


def test_low_na_cannot_print_16nm_pitch(low_na):
    r = low_na.print_lines(16, 8, dose_mj_cm2=40, stochastic=False)
    assert r["image_contrast"] < 0.05


def test_gas_and_thermal_from_environment(low_na):
    assert 0.8 < low_na.gas_transmission() < 1.0
    th = low_na.thermal_state()
    assert th["reticle_dT_K"] > 0
    assert 0.5 < low_na.overlay_budget_nm() < 5
