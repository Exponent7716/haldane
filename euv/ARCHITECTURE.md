# euvsim architecture & inter-module contracts

`euvsim` is a physics-based, public-literature model of an EUV lithography
scanner (NXE:3x00 low-NA 0.33 and EXE:5000 high-NA 0.55 class). It is an
educational / engineering simulator built from published physics, **not** a
reproduction of any vendor's proprietary design data.

Light path:

```
CO2 drive laser ─► Sn droplet ─► LPP plasma (13.5 nm) ─► collector ─► IF
  ─► illuminator (field facet mirror, pupil facet mirror, grazing mirror)
  ─► reticle (reflective Mo/Si mask + absorber, on reticle stage, pellicle)
  ─► projection optics (6 mirrors, NA 0.33 / 8 mirrors, NA 0.55 anamorphic)
  ─► wafer (resist) on wafer stage  [all in H2-purged vacuum]
```

## Package layout (one owner per subpackage)

| subpackage | content |
|---|---|
| `euvsim/constants.py` | shared constants, optical constants at 13.5 nm (shared, do not break) |
| `euvsim/optics/multilayer.py` | Mo/Si Bragg mirror reflectivity (Parratt). Shared. |
| `euvsim/source/` | drive laser, droplet generator, Sn plasma, CE, spectrum, debris, collector, dose stability |
| `euvsim/illumination/` | collector→IF étendue, field & pupil facet mirrors, illumination modes, uniformity (UNICOM), transmission |
| `euvsim/imaging/` | reflective mask (absorber, M3D shadowing, pellicle), projection optics (NA, anamorphic, Zernike, flare), Abbe aerial image |
| `euvsim/resist/` | photon shot noise, CAR acid generation/diffusion, development, LER/LWR, CD metrology, stochastic defects |
| `euvsim/stage/` | reticle & wafer stage dynamics, scan sync (MA/MSD), alignment, leveling/focus, overlay, dose-control slit integration |
| `euvsim/environment/` | vacuum/H2 environment, gas absorption, contamination (C growth, oxidation, Sn), mirror & reticle heating, lifetime |
| `euvsim/scanner.py` | integration: throughput (WPH), dose budget, end-to-end exposure |

## Shared conventions (contract)

* Units: nm for optics/imaging, m for mechanics, W/J/s, dose mJ/cm².
* Complex index N = n − i k (see `constants.complex_index`).
* **Source points (illumination → imaging):** `numpy.ndarray` shape `(N, 3)`
  with columns `(sigma_x, sigma_y, weight)`; sigma in units of NA (pupil
  radius = 1), weights sum to 1.
* **Mask pattern (imaging):** 2-D real array `pattern[y, x]` in {0..1}, 1 = reflective
  (multilayer), 0 = absorber, defined on a *wafer-scale* grid with pixel
  size `dx_nm` (mask features are magnification × larger physically).
* **Aerial image:** 2-D real array, normalised so a fully reflective (clear)
  mask gives intensity 1.0.
* **Resist input:** aerial image (normalised) × dose (mJ/cm²) → photon
  counts per pixel via `constants.photons_per_nm2`.
* Each subpackage exposes a small public API from its `__init__.py` and
  has tests in `tests/test_<subpackage>.py` runnable with `pytest`.
* Deterministic randomness: functions that sample take `rng: numpy.random.Generator`.
