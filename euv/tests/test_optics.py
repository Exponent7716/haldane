import numpy as np
import pytest

from euvsim.constants import PHOTON_ENERGY_EV, photons_per_nm2
from euvsim.optics.multilayer import (Layer, bandpass_chain, mosi_mirror, reflectivity,
                                      tuned_mirror)


def test_photon_energy():
    assert PHOTON_ENERGY_EV == pytest.approx(91.8, abs=0.2)
    assert photons_per_nm2(1.0) == pytest.approx(0.68, abs=0.01)


def test_mosi_peak_reflectivity_near_theory():
    R = reflectivity(mosi_mirror())
    assert 0.68 < R < 0.75


def test_reflectivity_saturates_with_pairs():
    R = [reflectivity(mosi_mirror(n_pairs=n)) for n in (5, 20, 40, 60)]
    assert R[0] < R[1] < R[2]
    assert abs(R[3] - R[2]) < 0.01


def test_roughness_and_interdiffusion_reduce_R():
    R0 = reflectivity(mosi_mirror())
    assert reflectivity(mosi_mirror(roughness_nm=0.3)) < R0
    assert reflectivity(mosi_mirror(interdiffusion_nm=0.5)) < R0


def test_carbon_overlayer_reduces_R():
    m = mosi_mirror()
    m.layers.insert(0, Layer("C", 2.0))
    assert reflectivity(m) < reflectivity(mosi_mirror())


def test_tuned_mirror_at_oblique_angle():
    m = tuned_mirror(15.0)
    assert reflectivity(m, angle_deg=15.0) > reflectivity(mosi_mirror(), angle_deg=15.0)


def test_bandpass_narrows_with_more_mirrors():
    wl = np.linspace(12.8, 14.2, 141)
    _, bw2 = bandpass_chain(mosi_mirror(), 2, wl)
    _, bw10 = bandpass_chain(mosi_mirror(), 10, wl)
    assert bw10 < bw2
