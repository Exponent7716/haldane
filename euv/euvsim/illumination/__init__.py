"""EUV illuminator: pupil shapes, fly's-eye facet integrator, UNICOM, transmission.

Public API
----------
Source points (contract: ndarray (N, 3) of sigma_x, sigma_y, weight; Σw = 1):
    conventional, annular, dipole, quadrupole, quasar, cquad, leaf,
    from_bitmap, from_mask, shape_mask
Pupil metrics: pupil_fill_ratio, centroid, ellipticity, pole_balance, pupil_metrics
Facet pupil: PupilFacetMirror, FlexPupil, masking_efficiency
Field uniformity: ArcSlit, FieldFacetMirror, FlyEyeIntegrator, Unicom, uniformity,
    uniformity_vs_facets
Transmission: IlluminatorTransmission, MirrorElement, multilayer_reflectance,
    grazing_ru_reflectance, accepted_etendue_mm2sr, etendue_efficiency, pupil_loss
Top level: Illuminator, IlluminationResult, source_etendue_mm2sr
"""
from .pupil import (annular, centroid, conventional, cquad, dipole, ellipticity, from_bitmap,
                    from_mask, leaf, pole_balance, pole_energies, pupil_fill_ratio, pupil_metrics,
                    quadrupole, quasar, shape_mask, sigma_grid, to_bitmap, validate)
from .flexpupil import FlexPupil, FlexPupilResult, PupilFacetMirror, masking_efficiency
from .facets import (ArcSlit, FieldFacetMirror, FlyEyeIntegrator, Unicom, UnicomResult,
                     far_field_beam, uniformity, uniformity_vs_facets)
from .transmission import (IlluminatorTransmission, MirrorElement, accepted_etendue_mm2sr,
                           default_mirror_train, etendue_efficiency, grazing_ru_reflectance,
                           multilayer_reflectance, pupil_loss)
from .illuminator import IlluminationResult, Illuminator, source_etendue_mm2sr

__all__ = [n for n in dir() if not n.startswith("_")]
