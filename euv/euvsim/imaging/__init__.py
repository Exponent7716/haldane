"""Computational imaging chain: reflective mask -> projection optics -> aerial image.

Public API::

    ProjectionOptics, ReflectiveMask, Pellicle, aerial_image,
    line_space, isolated_line, contact_array, tip_to_tip, rectangles,
    nils, measure_cd, image_contrast, cutline,
    focus_exposure_matrix, process_window, best_focus
"""
from .aerial import (aerial_image, best_focus, cutline, focus_exposure_matrix, image_contrast,
                     measure_cd, nils, process_window, profile_shift_nm,
                     telecentricity_error_mrad, through_focus)
from .mask import (ABSORBER_PRESETS, PELLICLE_PRESETS, Pellicle, ReflectiveMask, cra_overlap_ok,
                   mask_side_na)
from .patterns import (contact_array, grid_coords, isolated_line, line_space, rectangles,
                       tip_to_tip)
from .projection import FRINGE_NM, ProjectionOptics, fringe_zernike

__all__ = [
    "ProjectionOptics", "ReflectiveMask", "Pellicle", "PELLICLE_PRESETS", "ABSORBER_PRESETS",
    "aerial_image", "through_focus", "line_space", "isolated_line", "contact_array",
    "tip_to_tip", "rectangles", "grid_coords", "nils", "measure_cd", "image_contrast",
    "cutline", "profile_shift_nm", "telecentricity_error_mrad", "focus_exposure_matrix", "process_window", "best_focus", "fringe_zernike",
    "FRINGE_NM", "cra_overlap_ok", "mask_side_na",
]
