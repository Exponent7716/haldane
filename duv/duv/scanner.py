"""Top-level integration: one ArF immersion scanner built from all subsystems.

The light path and the control loops are chained as in a real tool::

    laser (spectrum, pulse energy, dose control)
      -> illuminator (pupil shape, polarization, slit uniformity)
      -> reticle (mask transmission, pellicle)
      -> projection lens (aberrations, lens heating, immersion water)
      -> wafer stage (MSD blur, MA overlay)
      -> resist (CA exposure, PEB, development, CD)

and on the measure side of the dual stage::

    alignment sensor -> wafer-grid model -> overlay
    level sensor     -> focus/tilt set-points -> residual defocus

``DUVScanner.expose_field`` images one slit point; ``DUVScanner.expose_wafer``
runs a whole wafer and reports CD uniformity, overlay and throughput.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import illumination, light_source, metrology, projection, resist, reticle, stage
from .core import SLIT_HEIGHT_MM, ExposureResult, Grid, ImagingSettings, Mask


@dataclass
class ScannerConfig:
    """User-facing knobs of the scanner model."""

    grid: Grid = field(default_factory=lambda: Grid(n=128, pixel=4.0))
    # illumination
    illumination: str = "dipole_x"
    sigma_in: float = 0.7
    sigma_out: float = 0.95
    opening_angle_deg: float = 35.0
    polarization: str = "Y"  # TE for vertical lines with an x-dipole
    source_n: int = 61
    # light source
    rep_rate_hz: float = 6000.0
    bandwidth_e95_pm: float = 0.30
    chromatic_focus_nm_per_pm: float = 0.0  # catadioptric lens: ~achromatic
    # projection
    na: float = 1.35
    lens_rms_milliwaves: float = 5.0
    lens_heating_model_error: float = 0.1
    water_delta_t_k: float = 0.002
    flare: float = 0.01
    # stage
    scan_speed_mm_s: float = 700.0
    # resist
    resist_model: str = "threshold"
    seed: int = 0


class DUVScanner:
    """An ArF immersion scanner assembled from the subsystem models."""

    def __init__(self, config: ScannerConfig | None = None):
        self.config = c = config or ScannerConfig()
        self.rng = np.random.default_rng(c.seed)

        self.laser = light_source.ArFLaser(rep_rate_hz=c.rep_rate_hz, bandwidth_e95_pm=c.bandwidth_e95_pm)
        self.dose_controller = light_source.DoseController(
            n_window=max(int(round(light_source.pulses_per_point(SLIT_HEIGHT_MM, c.scan_speed_mm_s, c.rep_rate_hz))), 1)
        )
        self.illuminator = illumination.Illuminator(
            kind=c.illumination,
            sigma_in=c.sigma_in,
            sigma_out=c.sigma_out,
            opening_angle_deg=c.opening_angle_deg,
            polarization=c.polarization,
        )
        self.pellicle = reticle.Pellicle()
        self.lens = projection.ProjectionLens(na=c.na, rms_milliwaves=c.lens_rms_milliwaves)
        self.lens.optimize_manipulators()  # lens set-up: null the fingerprint
        self.lens_heating = projection.LensHeating()
        self.hood = projection.ImmersionHood(na=c.na)
        self.scan_profile = stage.ScanProfile(scan_speed_mm_s=c.scan_speed_mm_s)
        self.resist = resist.CAResist()
        self.film_stack = resist.default_film_stack()
        self.alignment_sensor = metrology.AlignmentSensor()
        self.level_sensor = metrology.LevelSensor()

        self._source, self.illumination_summary = self.illuminator.run(n=c.source_n)

    # ------------------------------------------------------------------ optics
    @property
    def source(self):
        return self._source

    def imaging_settings(self, field_x_mm: float = 0.0, defocus_nm: float = 0.0,
                         extra_zernikes: dict[int, float] | None = None) -> ImagingSettings:
        """Lens fingerprint at ``field_x_mm`` + water-temperature focus + extras."""
        c = self.config
        base = ImagingSettings(na=c.na, n_immersion=self.hood.n, flare=c.flare,
                               defocus=defocus_nm + self.hood.focus_shift_nm(c.water_delta_t_k))
        return self.lens.imaging_settings(field_x_mm, base=base, extra=extra_zernikes)

    def aerial_image(self, mask: Mask, field_x_mm: float = 0.0, defocus_nm: float = 0.0,
                     extra_zernikes: dict[int, float] | None = None, msd_nm: float = 0.0) -> np.ndarray:
        """Aerial image at the wafer including laser bandwidth and stage MSD blur."""
        settings = self.imaging_settings(field_x_mm, defocus_nm, extra_zernikes)
        chroma = self.config.chromatic_focus_nm_per_pm
        # bandwidth only matters through chromatic focus blur
        spectrum = self.laser.spectrum(11) if chroma else None
        img = projection.aerial_image(mask, self.source, settings, spectrum=spectrum,
                                      chromatic_focus_nm_per_pm=chroma)
        if msd_nm > 0:
            img = _gaussian_blur(img, mask.grid, msd_nm)
        return img

    @property
    def pellicle_transmission(self) -> float:
        return float(self.pellicle.transmission(0.0))

    def expose_field(self, mask: Mask, dose_mj_cm2: float, field_x_mm: float = 0.0,
                     defocus_nm: float = 0.0, extra_zernikes: dict[int, float] | None = None,
                     msd_nm: float = 0.0, feature: str = "line") -> ExposureResult:
        """Expose one slit point and develop the resist.

        ``dose_mj_cm2`` is the dose set on the scanner (clear-field dose at
        the wafer); the pellicle loss is already compensated by dose control.
        """
        img = self.aerial_image(mask, field_x_mm, defocus_nm, extra_zernikes, msd_nm)
        res = self.resist.process(img, dose_mj_cm2, mask.grid, model=self.config.resist_model, feature=feature)
        res.info.update(field_x_mm=field_x_mm, defocus_nm=defocus_nm, msd_nm=msd_nm)
        return res

    def dose_to_size(self, mask: Mask, target_cd: float, feature: str = "line") -> float:
        """Dose that prints ``target_cd`` at nominal focus, field centre."""
        img = self.aerial_image(mask)
        return resist.dose_to_size(img, target_cd, mask.grid, resist=self.resist,
                                   model=self.config.resist_model, feature=feature)

    def process_window(self, mask: Mask, target_cd: float, focuses_nm=None, doses=None,
                       feature: str = "line", tol: float = 0.1) -> dict:
        """Focus-exposure matrix and process window around dose-to-size."""
        d0 = self.dose_to_size(mask, target_cd, feature)
        focuses_nm = np.linspace(-120, 120, 13) if focuses_nm is None else np.asarray(focuses_nm)
        doses = d0 * np.linspace(0.85, 1.15, 13) if doses is None else np.asarray(doses)
        cd = resist.focus_exposure_matrix(lambda f: self.aerial_image(mask, defocus_nm=f), focuses_nm, doses,
                                          mask.grid, resist=self.resist, model=self.config.resist_model,
                                          feature=feature)
        pw = resist.process_window(cd, focuses_nm, doses, target_cd, tol=tol)
        pw.update(cd_matrix=cd, focuses_nm=focuses_nm, doses=doses, dose_to_size=d0)
        return pw

    # --------------------------------------------------------------- full wafer
    def expose_wafer(self, mask: Mask, target_cd: float, n_fields: int | None = None,
                     slit_points_mm=(-12.0, 0.0, 12.0), feature: str = "line",
                     wafer: metrology.WaferDeformation | None = None,
                     n_align_marks: int = 16, align_order: int = 3) -> dict:
        """Run a whole wafer through measure side and expose side.

        Returns per-field CD and overlay plus wafer-level CDU / overlay
        statistics, servo, dose and throughput figures.
        """
        rng = self.rng
        c = self.config
        dose0 = self.dose_to_size(mask, target_cd, feature)

        # --- measure side: alignment -----------------------------------------
        wafer = wafer or metrology.WaferDeformation(
            tx=20.0, ty=-15.0, mag_ppm=0.8, rot_urad=0.5, nonorth_urad=0.2,
            third_order_nm=3.0, noise_nm=0.5, seed=c.seed,
        )
        align = metrology.evaluate_alignment_strategy(wafer, n_align_marks, order=align_order,
                                                      sensor=self.alignment_sensor, rng=rng)
        model = align["model"]

        # --- measure side: leveling ------------------------------------------
        fields = stage.field_layout(include_partial=False)
        if n_fields is not None:
            idx = np.linspace(0, len(fields) - 1, min(n_fields, len(fields))).round().astype(int)
            fields = [fields[i] for i in idx]
        topo = _wafer_topography(c.seed)

        # --- expose side: stage synchronisation (one representative scan) -----
        sync = stage.simulate_synchronization(self.scan_profile, rng=rng)

        # --- light source: dose control burst ---------------------------------
        dose_run = self.dose_controller.run(self.laser, n_pulses=4000, rng=rng)
        dose_rel = dose_run.window_dose_mj_cm2 / dose_run.target_dose_mj_cm2 - 1.0
        dose_rel = dose_rel[np.isfinite(dose_rel)]

        # --- lens heating with feed-forward ------------------------------------
        power = self.lens_heating.absorbed_power(dose0)
        t_field = self.scan_profile.time_per_field()
        kind = c.illumination if c.illumination in self.lens_heating.sensitivity else "conventional"

        cds, ovl_xy, ovl, defocus_list, field_cd = [], [], [], [], []
        for i, (fx, fy) in enumerate(fields):
            # leveling: sample the field, fit focus plane, keep residual
            gx, gy = np.meshgrid(np.linspace(fx - 13, fx + 13, 9), np.linspace(fy - 4, fy + 4, 3))
            xy = np.column_stack([gx.ravel(), gy.ravel()])
            z_meas = self.level_sensor.scan(topo, xy, rng=rng)
            z0, rx, ry = metrology.focus_correction((xy, z_meas), (fx, fy))
            # lens heating after feed-forward; the Z4 residual is applied as defocus
            r = self.lens_heating.feedforward(60.0 + i * t_field, power, kind,
                                              model_error=c.lens_heating_model_error)["residual"]
            heat = {j: float(v) for j, v in r.items() if j != 4}
            heat_focus_nm = _z4_waves_to_nm(float(r.get(4, 0.0)), c.na, self.hood.n)
            # one dose sample per field from the dose-controlled burst
            d = dose0 * (1.0 + dose_rel[rng.integers(len(dose_rel))])
            row = []
            for sx in slit_points_mm:
                plane = z0 + ry * sx
                defocus = float(topo(np.array(fx + sx), np.array(fy))) - plane + heat_focus_nm
                res = self.expose_field(mask, d, field_x_mm=sx, defocus_nm=defocus,
                                        extra_zernikes=heat, msd_nm=sync["msd_max_nm"], feature=feature)
                row.append(res.cd)
                cds.append(res.cd)
                defocus_list.append(defocus)
            field_cd.append(row)
            # overlay: wafer-grid model residual + stage MA
            true = np.array(wafer.systematic(np.array([fx]), np.array([fy]))).ravel()
            fit = metrology.apply_model(model, np.array([[fx, fy]]))[0]
            ma = sync["ma_max_nm"] * rng.uniform(-1, 1, 2)
            ovl.append(true - fit + ma)
            ovl_xy.append((fx, fy))

        cds = np.asarray(cds)
        ovl = np.asarray(ovl)
        ok = np.isfinite(cds)
        return {
            "dose_to_size_mj_cm2": dose0,
            "fields": np.asarray(ovl_xy),
            "field_cd_nm": np.asarray(field_cd),
            "cd_mean_nm": float(np.mean(cds[ok])) if ok.any() else float("nan"),
            "cdu_3sigma_nm": float(3 * np.std(cds[ok])) if ok.any() else float("nan"),
            "defocus_3sigma_nm": float(3 * np.std(defocus_list)),
            "overlay_nm": ovl,
            "overlay_stats": metrology.overlay_stats(ovl[:, 0], ovl[:, 1]),
            "alignment_stats": align["stats"],
            "stage_ma_nm": sync["ma_max_nm"],
            "stage_msd_nm": sync["msd_max_nm"],
            "dose_3sigma_pct": float(300 * np.std(dose_rel)),
            "throughput": stage.throughput(scan_speed_mm_s=c.scan_speed_mm_s),
            "illumination": {k: self.illumination_summary[k] for k in ("dop", "corrected_uniformity")},
        }


def _gaussian_blur(img: np.ndarray, grid: Grid, sigma_nm: float) -> np.ndarray:
    """Periodic Gaussian blur (stage MSD acts as a random image displacement)."""
    fx, fy = grid.freq_mesh()
    kernel = np.exp(-2 * np.pi**2 * sigma_nm**2 * (fx**2 + fy**2))
    spec = np.fft.fftshift(np.fft.fft2(img))
    return np.real(np.fft.ifft2(np.fft.ifftshift(spec * kernel)))


def _z4_waves_to_nm(z4_waves: float, na: float, n: float, wavelength: float = 193.368) -> float:
    """Approximate wafer defocus equivalent to a Fringe Z4 coefficient.

    The pupil defocus phase changes by -dz (n - sqrt(n^2 - NA^2)) / lambda
    from centre to edge, Z4 = 2 rho^2 - 1 by 2 Z4 waves, so
    dz = -2 Z4 lambda / (n - sqrt(n^2 - NA^2)).
    """
    return -2.0 * z4_waves * wavelength / (n - np.sqrt(n * n - na * na))


def _wafer_topography(seed: int):
    """Smooth synthetic wafer height map (nm): bow + a few process bumps."""
    rng = np.random.default_rng(seed + 101)
    centers = rng.uniform(-120, 120, (6, 2))
    amps = rng.normal(0, 25, 6)

    def h(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        z = 3e-3 * (x**2 + y**2) * 0.05 + 0.08 * x - 0.05 * y  # bow + global tilt
        for (cx, cy), a in zip(centers, amps):
            z = z + a * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 15.0**2))
        return z

    return h


__all__ = ["ScannerConfig", "DUVScanner"]
