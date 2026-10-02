"""End-to-end demo: walk light through every subsystem of the DUV scanner.

    python3 examples/run_scanner.py            # text report
    python3 examples/run_scanner.py --plot     # also writes PNGs to examples/output/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duv import light_source, reticle, stage  # noqa: E402
from duv.scanner import DUVScanner, ScannerConfig  # noqa: E402


def main(plot: bool = False) -> None:
    rng = np.random.default_rng(0)
    scanner = DUVScanner(ScannerConfig())
    g = scanner.config.grid

    print("=== 1. Light source: ArF excimer laser ===")
    laser = scanner.laser
    wl, w = laser.spectrum(201)
    print(f"  centre {laser.center_wavelength_nm:.3f} nm, E95 {light_source.e95_width_pm(wl, w):.3f} pm, "
          f"{laser.rep_rate_hz / 1e3:.0f} kHz, {laser.average_power_w:.0f} W")
    dc = scanner.dose_controller.run(laser, n_pulses=3000, rng=rng)
    print(f"  {scanner.dose_controller.n_window} pulses/point, slit-dose 3sigma {dc.stats['three_sigma'] * 100:.2f} %")

    print("=== 2. Illuminator ===")
    pm = scanner.illumination_summary["pupil"]
    print(f"  {scanner.config.illumination} {scanner.config.polarization}-pol, sigma_c {pm['sigma_center']:.2f}, "
          f"DOP {scanner.illumination_summary['dop']:.2f}, slit uniformity after UNICOM "
          f"{scanner.illumination_summary['corrected_uniformity'] * 100:.3f} %")

    print("=== 3. Reticle: 45 nm lines / 128 nm pitch, 6% attPSM ===")
    mask = reticle.make_mask(reticle.line_space(g, 45, 128), g, "attpsm")
    print(f"  pellicle transmission {scanner.pellicle_transmission:.4f}")

    print("=== 4. Projection lens + water ===")
    img = scanner.aerial_image(mask)
    row = img[g.n // 2]
    print(f"  NA {scanner.config.na}, water n {scanner.hood.n}, aerial-image contrast "
          f"{(row.max() - row.min()) / (row.max() + row.min()):.3f}, lens RMS after set-up "
          f"{scanner.lens.field_rms() * 1e3:.2f} mwaves")

    print("=== 5. Resist ===")
    dose = scanner.dose_to_size(mask, 45.0)
    res = scanner.expose_field(mask, dose)
    print(f"  dose-to-size {dose:.1f} mJ/cm^2 -> CD {res.cd:.2f} nm")
    pw = scanner.process_window(mask, 45.0)
    print(f"  process window: DOF {pw['dof_nm']:.0f} nm at EL {pw['exposure_latitude_pct']:.1f} %")

    print("=== 6. Stages ===")
    sync = stage.simulate_synchronization(scanner.scan_profile, rng=rng)
    print(f"  {scanner.config.scan_speed_mm_s:.0f} mm/s scan, sync MA {sync['ma_max_nm']:.2f} nm, "
          f"MSD {sync['msd_max_nm']:.2f} nm")

    print("=== 7. Full wafer (metrology + exposure) ===")
    wres = scanner.expose_wafer(mask, 45.0, n_fields=12)
    o = wres["overlay_stats"]
    print(f"  CD mean {wres['cd_mean_nm']:.2f} nm, CDU 3sigma {wres['cdu_3sigma_nm']:.2f} nm, "
          f"defocus 3sigma {wres['defocus_3sigma_nm']:.1f} nm")
    print(f"  overlay |m|+3s  x {o['x']['m+3s']:.2f} nm  y {o['y']['m+3s']:.2f} nm")
    print(f"  throughput {wres['throughput']['wph']:.0f} wafers/h")

    if plot:
        _plot(scanner, mask, img, pw)


def _plot(scanner, mask, img, pw):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    g = scanner.config.grid
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    s = scanner.source
    ax[0].imshow(s.intensity, extent=[-1, 1, -1, 1], origin="lower", cmap="inferno")
    ax[0].set_title("Illumination pupil")
    ax[1].imshow(np.abs(mask.transmission) ** 2, cmap="gray")
    ax[1].set_title("Mask |t|^2")
    ax[2].plot(g.x, img[g.n // 2])
    ax[2].set_title("Aerial image (centre row)")
    ax[2].set_xlabel("x [nm]")
    for i, d in enumerate(pw["doses"][::3]):
        ax[3].plot(pw["focuses_nm"], pw["cd_matrix"][:, 3 * i], label=f"{d:.0f} mJ/cm²")
    ax[3].set_title("Bossung curves")
    ax[3].set_xlabel("focus [nm]")
    ax[3].set_ylabel("CD [nm]")
    ax[3].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "scanner_overview.png", dpi=120)
    print(f"  plots written to {out}")


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)
