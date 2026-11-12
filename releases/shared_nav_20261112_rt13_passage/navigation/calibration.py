"""Accept measured geometry or an explicitly selected robot-model assumption."""
import numpy as np


def calibration_reasons(config, head=None):
    reasons = []
    if config.get('calibration_basis') != 'robot_model_assumption':
        for flag in ('extrinsics_verified', 'imu_alignment_verified'):
            if config.get(flag) is not True:
                reasons.append(flag+'_required')
    if not config.get('calibration_evidence'):
        reasons.append('calibration_evidence_required')
    try:
        position = np.asarray(config['camera_position_body_m'], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError()
        for key in ('camera_rotation_optical_to_body', 'body_from_imu'):
            r = np.asarray(config[key], dtype=float)
            if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
                raise ValueError()
    except (KeyError, ValueError, TypeError):
        reasons.append('invalid_camera_or_imu_transform')
    if config.get('camera_mount') == 'movable_head':
        reference = config.get('head_reference_rad', {})
        if set(reference) != {'head_pitch_joint', 'head_yaw_joint'}:
            reasons.append('fixed_head_reference_required')
        elif head is not None:
            if any(name not in head or not np.isfinite(head[name]) or abs(head[name]-value) > .04 for name, value in reference.items()):
                reasons.append('head_pose_not_at_calibrated_reference')
    return reasons
