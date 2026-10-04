"""Stage mechatronics, metrology and exposure control of a step-and-scan EUV scanner.

Modules
-------
trajectory      jerk/snap-limited setpoints, scan profile, field cycle time
dynamics        two-mass stage plant, PID+lead servo, feedforward, MA/MSD sync error
metrology       interferometer/encoder, alignment sensor, level sensor, focus budget
overlay         wafer grid + intra-field model, LS fit, mean+3sigma, RSS budgets
dose_control    pulses per point, slit-integrated dose, pulse-energy dose servo
scanner_timing  dose-limited scan speed v = P/(D w), field layout, wafer timeline/WPH
"""
from .trajectory import (G0, MotionLimits, Trajectory, ScanProfile, point_to_point,
                         move_time, scan_profile, exposure_time_per_field,
                         field_cycle_time, segment_lengths)
from .dynamics import (StagePlant, Controller, Disturbances, ScannerStages, ServoResult,
                       SyncResult, simulate_servo, simulate_scan_sync,
                       moving_average_std, closed_loop_stable)
from .metrology import (Interferometer, Encoder, AlignmentSensor, LevelSensor,
                        WaferTopography, FocusBudget, gas_refractive_index, abbe_error,
                        generate_wafer_topography, fit_plane, slit_leveling_residual,
                        depth_of_focus, default_focus_budget)
from .overlay import (GridModel, OverlayFit, OverlayBudget, fit_overlay,
                      design_matrices, alignment_mark_positions, mean_plus_3sigma,
                      matched_machine_overlay_budget)
from .dose_control import (pulses_per_point, dose_mj_cm2, open_loop_dose_error,
                           slit_profile, slit_integrated_dose, apply_dose_control,
                           DoseControlResult, synthetic_pulse_train)
from .scanner_timing import (max_scan_speed, required_power_at_wafer, FieldLayout,
                             field_layout, WaferTimeline, TimelineParams,
                             wafer_timeline, throughput_wph)

__all__ = [n for n in dir() if not n.startswith("_")]
