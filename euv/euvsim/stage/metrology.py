"""Position metrology, wafer alignment and wafer levelling (focus) models.

1. Stage position sensors
-------------------------
*Interferometer* (double-pass, heterodyne): optical path change
    Delta_phi = 2 pi * p * n * Delta_x / lambda   (p = 2 for double pass)
so the reported displacement is ``x_meas = n_assumed/n_true * x`` plus
errors from the refractive index ``n`` of the beam path of length ``L``:
    dx_index = (n - n_assumed) * L
In air (n-1 ~ 2.7e-4) turbulence of 1e-8 in n over 0.5 m gives ~5 nm; in the
EUV vacuum (a few Pa of H2, n-1 ~ 1.4e-4 * p/p0 ~ 4e-9 at 3 Pa) the index
term is negligible, which is one reason EUV stages can use long beam paths.

*Encoder* (grating period ``g``, 2-D, sensor heads on the stage reading a
grating plate on the metrology frame): interpolated signal period ``g/2``
(or ``g/4`` with double diffraction), short dead path, so noise is small;
main errors are grating non-flatness / writing errors and thermal expansion:
    dx_thermal = CTE * dT * x

*Abbe error* (both): measurement axis offset ``d`` from the functional point
together with a stage rotation ``theta``: ``dx_abbe = d * tan(theta) ~ d theta``.

2. Alignment sensor (diffraction-based, phase grating mark)
-----------------------------------------------------------
A phase grating of period ``P`` illuminated by a spot diffracts +/-1 orders;
self-referencing interference of +1 and -1 gives an intensity

    I(x) = A + B cos(4 pi (x - x0)/P + phi_asym)

when the mark is scanned along ``x``.  A linear least-squares fit of
``a + b cos(kx) + c sin(kx)`` gives the phase, hence the mark centre
``x0 = psi * P / (4 pi)`` with ``psi = atan2(c, b)`` modulo P/2 (capture range resolved by a coarse
grating/multiple periods).  Mark asymmetry (e.g. a sidewall tilt from CMP)
adds a colour-dependent phase ``phi_asym`` -> a position offset
``-phi_asym P/(4 pi)``; using several colours detects it.  The noise-limited
precision of the fit is ``sigma_x ~ P/(4 pi) * sigma_I/(B) * sqrt(2/N)``.

3. Level sensor and focus
-------------------------
Wafer topography = global bow + random (power-law) unflatness + per-field
process topography.  The level sensor measures the height map with spot
averaging and noise; during exposure the stage follows a height/tilt plane
fitted over the *slit* (26 x 2 mm) at each scan position (z, Rx, Ry), so the
focus error is the residual topography that a plane cannot follow inside the
slit plus measurement errors.  The focus budget is the RSS of independent
contributions, compared to the depth of focus  DOF ~ k2 lambda / NA^2.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Refractivity (n - 1) at STP (0 C, 101325 Pa), visible/near-IR, rounded
REFRACTIVITY_STP = {"air": 2.93e-4, "H2": 1.39e-4, "N2": 2.98e-4, "He": 3.5e-5}
P_STP = 101325.0
T_STP = 273.15


def gas_refractive_index(pressure_pa: float, temperature_k: float = 295.0,
                         gas: str = "air") -> float:
    """Ideal-gas scaling: n - 1 = (n_STP - 1) * (p/p0) * (T0/T)."""
    return 1.0 + REFRACTIVITY_STP[gas] * (pressure_pa / P_STP) * (T_STP / temperature_k)


@dataclass
class Interferometer:
    """Heterodyne displacement interferometer (one axis)."""

    wavelength_m: float = 633e-9
    passes: int = 2
    beam_length_m: float = 0.5
    pressure_pa: float = 3.0           # EUV vacuum, H2
    gas: str = "H2"
    index_fluctuation_rel: float = 0.02  # relative rms fluctuation of (n-1)
    phase_noise_rad: float = 2e-3
    abbe_offset_m: float = 0.0

    @property
    def resolution_m(self) -> float:
        """Displacement per radian of phase: lambda / (2 pi * passes)."""
        return self.wavelength_m / (2 * np.pi * self.passes)

    def measure(self, x: np.ndarray, rng: np.random.Generator,
                tilt_rad: np.ndarray | float = 0.0) -> np.ndarray:
        """Measured position for true positions ``x`` [m]."""
        x = np.asarray(x, float)
        n_minus_1 = gas_refractive_index(self.pressure_pa, gas=self.gas) - 1.0
        dn = n_minus_1 * self.index_fluctuation_rel * rng.standard_normal(x.shape)
        index_err = dn * (self.beam_length_m + x)
        phase_err = self.phase_noise_rad * self.resolution_m * rng.standard_normal(x.shape)
        abbe = self.abbe_offset_m * np.tan(tilt_rad)
        return x + index_err + phase_err + abbe

    def index_error_rms(self) -> float:
        """rms error from refractive index fluctuations over the beam path [m]."""
        n_minus_1 = gas_refractive_index(self.pressure_pa, gas=self.gas) - 1.0
        return n_minus_1 * self.index_fluctuation_rel * self.beam_length_m


@dataclass
class Encoder:
    """Grating encoder (one axis)."""

    grating_period_m: float = 1e-6
    interpolation_noise_rel: float = 1e-4   # rms noise as fraction of signal period
    signal_division: int = 4                # optical period = g/4 (double diffraction)
    cte_per_k: float = 0.02e-6              # Zerodur-like grating substrate
    delta_t_k: float = 0.01
    grating_error_rms_m: float = 0.1e-9     # residual after calibration (map)
    abbe_offset_m: float = 0.0

    @property
    def signal_period_m(self) -> float:
        return self.grating_period_m / self.signal_division

    def measure(self, x: np.ndarray, rng: np.random.Generator,
                tilt_rad: np.ndarray | float = 0.0) -> np.ndarray:
        x = np.asarray(x, float)
        noise = self.interpolation_noise_rel * self.signal_period_m * rng.standard_normal(x.shape)
        thermal = self.cte_per_k * self.delta_t_k * x
        # smooth grating-writing error: a few spatial harmonics with random phase
        k = 2 * np.pi / np.array([3e-3, 11e-3, 37e-3])
        ph = rng.uniform(0, 2 * np.pi, 3)
        grating = self.grating_error_rms_m * np.sqrt(2 / 3) * np.sum(
            np.sin(np.outer(x, k) + ph), axis=1).reshape(x.shape)
        abbe = self.abbe_offset_m * np.tan(tilt_rad)
        return x + noise + thermal + grating + abbe


def abbe_error(offset_m: float, tilt_rad: float) -> float:
    """Abbe error d * tan(theta)."""
    return offset_m * np.tan(tilt_rad)


# ---------------------------------------------------------------- alignment
@dataclass
class AlignmentSensor:
    """Self-referencing diffraction-based alignment sensor on a phase grating."""

    mark_period_m: float = 16e-6
    n_samples: int = 200
    scan_length_periods: float = 4.0
    contrast: float = 0.8
    intensity_noise_rel: float = 0.01   # rms, relative to mean intensity

    def signal(self, x_scan: np.ndarray, x0: float,
               asym_phase_rad: float = 0.0) -> np.ndarray:
        """Noise-free I(x) = 1 + B cos(4 pi (x - x0)/P + phi_asym)."""
        k = 4 * np.pi / self.mark_period_m
        return 1.0 + self.contrast * np.cos(k * (x_scan - x0) + asym_phase_rad)

    def measure(self, x0: float, rng: np.random.Generator,
                asym_phase_rad: float = 0.0) -> float:
        """Estimate the mark position (mod P/2, assuming |x0| < P/4)."""
        P = self.mark_period_m
        x = np.linspace(-0.5, 0.5, self.n_samples) * self.scan_length_periods * P
        I = self.signal(x, x0, asym_phase_rad)
        I = I + self.intensity_noise_rel * rng.standard_normal(x.size)
        k = 4 * np.pi / P
        A = np.column_stack([np.ones_like(x), np.cos(k * x), np.sin(k * x)])
        a, b, c = np.linalg.lstsq(A, I, rcond=None)[0]
        # b cos + c sin = R cos(kx - psi), psi = atan2(c, b) = k x0 - phi
        psi = np.arctan2(c, b)
        return float(psi / k)

    def precision_m(self) -> float:
        """Cramér-Rao-like 1-sigma precision: P/(4pi) * sigma/B * sqrt(2/N)."""
        return (self.mark_period_m / (4 * np.pi) * self.intensity_noise_rel
                / self.contrast * np.sqrt(2.0 / self.n_samples))

    def asymmetry_offset_m(self, asym_phase_rad: float) -> float:
        """Systematic position offset caused by an asymmetry phase: -phi P/(4 pi)."""
        return -asym_phase_rad * self.mark_period_m / (4 * np.pi)


# ------------------------------------------------------------- levelling
@dataclass
class WaferTopography:
    """Height map z(x, y) [m] on a square grid covering the wafer."""

    x: np.ndarray       # 1-D grid [m]
    y: np.ndarray
    z: np.ndarray       # z[iy, ix], NaN outside wafer

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])


def generate_wafer_topography(rng: np.random.Generator, diameter_m: float = 0.3,
                              pixel_m: float = 1e-3, bow_m: float = 2e-6,
                              unflatness_rms_m: float = 30e-9,
                              psd_slope: float = 3.0,
                              field_size_m: tuple[float, float] = (26e-3, 33e-3),
                              field_topo_m: float = 5e-9) -> WaferTopography:
    """Synthetic wafer height map.

    z = bow (paraboloid, ``bow_m`` at edge vs centre) + random unflatness with
    isotropic power-law PSD ~ f^-psd_slope (rms ``unflatness_rms_m``) + a
    repeated intra-field process topography (``field_topo_m`` amplitude).
    Note: the chuck clamps the wafer, so the clamped bow is small; ``bow_m``
    is the residual after clamping.
    """
    n = int(round(diameter_m / pixel_m)) + 1
    x = (np.arange(n) - (n - 1) / 2) * pixel_m
    X, Y = np.meshgrid(x, x)
    R = diameter_m / 2
    r2 = (X ** 2 + Y ** 2) / R ** 2
    bow = bow_m * r2
    fx = np.fft.fftfreq(n, pixel_m)
    FX, FY = np.meshgrid(fx, fx)
    f = np.hypot(FX, FY)
    f[0, 0] = np.inf
    amp = f ** (-psd_slope / 2)
    noise = np.real(np.fft.ifft2(amp * np.fft.fft2(rng.standard_normal((n, n)))))
    noise *= unflatness_rms_m / np.std(noise)
    fw, fh = field_size_m
    field = field_topo_m * (np.cos(2 * np.pi * X / fw) * np.cos(2 * np.pi * Y / fh))
    z = bow + noise + field
    z[r2 > 1] = np.nan
    return WaferTopography(x, x.copy(), z)


@dataclass
class LevelSensor:
    """Optical triangulation level sensor (grazing incidence, broadband)."""

    spot_size_m: float = 2e-3
    noise_rms_m: float = 3e-9
    process_dependency_m: float = 5e-9  # apparent-height offset (stack dependent)

    def measure(self, topo: WaferTopography, rng: np.random.Generator) -> WaferTopography:
        """Spot-averaged (box filter) height map plus noise and process offset."""
        from scipy.ndimage import uniform_filter
        k = max(1, int(round(self.spot_size_m / topo.dx)))
        z = np.where(np.isnan(topo.z), 0.0, topo.z)
        w = (~np.isnan(topo.z)).astype(float)
        zs = uniform_filter(z, k) / np.maximum(uniform_filter(w, k), 1e-12)
        zs = zs + self.noise_rms_m * rng.standard_normal(zs.shape) + self.process_dependency_m
        zs[np.isnan(topo.z)] = np.nan
        return WaferTopography(topo.x, topo.y, zs)


def fit_plane(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LS plane z = z0 + a x + b y; returns (coeffs [z0, a, b], residuals)."""
    A = np.column_stack([np.ones_like(x), x, y])
    c = np.linalg.lstsq(A, z, rcond=None)[0]
    return c, z - A @ c


def slit_leveling_residual(topo: WaferTopography, measured: WaferTopography | None = None,
                           slit_width_m: float = 26e-3, slit_height_m: float = 2e-3,
                           edge_exclusion_m: float = 3e-3) -> np.ndarray:
    """Focus errors left after the stage follows a plane fitted per slit position.

    For each slit footprint (width x height, tiled over the wafer, fully inside
    the usable radius) a plane (z, Rx, Ry) is fitted to the *measured* map and
    the *true* topography residual to that plane is returned (all pixels).
    """
    measured = measured or topo
    X, Y = np.meshgrid(topo.x, topo.y)
    R = topo.x.max() - edge_exclusion_m
    out = []
    for yc in np.arange(topo.y.min() + slit_height_m / 2, topo.y.max(), slit_height_m):
        for xc in np.arange(topo.x.min() + slit_width_m / 2, topo.x.max(), slit_width_m):
            if np.hypot(abs(xc) + slit_width_m / 2, abs(yc) + slit_height_m / 2) > R:
                continue
            m = (np.abs(X - xc) <= slit_width_m / 2) & (np.abs(Y - yc) <= slit_height_m / 2)
            if m.sum() < 4:
                continue
            c, _ = fit_plane(X[m], Y[m], measured.z[m])
            plane = c[0] + c[1] * X[m] + c[2] * Y[m]
            out.append(topo.z[m] - plane)
    return np.concatenate(out) if out else np.zeros(0)


def depth_of_focus(na: float, wavelength_m: float = 13.5e-9, k2: float = 0.5) -> float:
    """DOF ~ k2 * lambda / NA^2 (Rayleigh, total range)."""
    return k2 * wavelength_m / na ** 2


@dataclass
class FocusBudget:
    """RSS focus budget; contributions are 1-sigma (or 3-sigma, consistently) [m]."""

    contributions: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value_m: float) -> None:
        self.contributions[name] = float(value_m)

    @property
    def total(self) -> float:
        return float(np.sqrt(sum(v * v for v in self.contributions.values())))

    def fraction_of(self, dof_m: float) -> float:
        return self.total / dof_m


def default_focus_budget(leveling_residual_3s_m: float = 15e-9) -> FocusBudget:
    """Illustrative 3-sigma focus budget of a low-NA EUV scanner (public order)."""
    b = FocusBudget()
    b.add("leveling residual (topography within slit)", leveling_residual_3s_m)
    b.add("level sensor process dependency", 8e-9)
    b.add("level sensor noise", 5e-9)
    b.add("stage z MSD/MA", 6e-9)
    b.add("lens focus drift / heating", 8e-9)
    b.add("reticle flatness / non-telecentricity", 6e-9)
    b.add("chuck flatness", 8e-9)
    return b
