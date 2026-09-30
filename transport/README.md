# Quantum transport (Kwant)

Landauer-Büttiker transport in tight-binding nanowires using [Kwant](https://kwant-project.org).

```bash
pip install --no-build-isolation kwant   # needs numpy<2, tinyarray, cython installed first
python transport/nanowire.py check       # validate against analytic results
python transport/nanowire.py dot         # double-barrier quantum dot, T(E)
python transport/nanowire.py disorder    # Anderson localization, <ln T> vs L
python transport/nanowire.py wire        # multi-channel conductance staircase
```

Units: hopping t = 1, G in e²/h (per spin). Options: `python transport/nanowire.py <cmd> -h`.

Validation: `check` compares with T=1 for the clean chain, the exact single-site scatterer
formula, perfect resonant transmission of a symmetric double barrier, and T+R=1.
`disorder` reproduces the weak-disorder localization length ξ = 24(4t²−E²)/W² (≈101 vs 96 for W=1, E=0).
