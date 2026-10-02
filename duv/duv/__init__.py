"""duv: a physics-level simulator of an ArF immersion (193 nm) DUV lithography scanner.

Subsystems (one module each):
    core          shared constants, grids and data containers
    light_source  ArF excimer laser (discharge, line narrowing, pulse/dose control)
    illumination  beam delivery, pupil shaping, integrator, REMA, polarization
    reticle       mask layout -> complex transmission, pellicle, reticle stage
    projection    4x projection lens, immersion, Abbe imaging, aberrations, lens heating
    resist        film stack, CAR exposure, PEB, development, CD metrology
    stage         wafer/reticle stages, trajectories, servo, MA/MSD
    metrology     alignment, level sensor, overlay modelling, corrections
    scanner       top-level integration of all subsystems
"""

__version__ = "0.1.0"
