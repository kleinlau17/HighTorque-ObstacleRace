"""Verify the explicitly selected Pro-model geometry and calibration gating."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.pro_head_model import ProHeadModel
from navigation.calibration import calibration_reasons


class ModelCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'deploy/pro_model_config.json').read_text())

    def test_explicit_model_basis_accepts_unmeasured_geometry(self):
        self.assertFalse(self.config['extrinsics_verified'])
        self.assertFalse(self.config['imu_alignment_verified'])
        self.assertEqual(calibration_reasons(self.config, self.config['head_reference_rad']), [])

    def test_unselected_or_misspelled_model_basis_requires_measurements(self):
        for basis in (None, 'model'):
            self.config['calibration_basis'] = basis
            reasons = calibration_reasons(self.config)
            self.assertIn('extrinsics_verified_required', reasons)
            self.assertIn('imu_alignment_verified_required', reasons)

    def test_measured_configuration_still_works(self):
        self.config.pop('calibration_basis')
        self.config.update(extrinsics_verified=True, imu_alignment_verified=True)
        self.assertEqual(calibration_reasons(self.config), [])

    def test_assumption_keeps_head_feedback_and_transform_checks(self):
        head = dict(self.config['head_reference_rad'], head_yaw_joint=.2)
        self.assertIn('head_pose_not_at_calibrated_reference', calibration_reasons(self.config, head))
        config = copy.deepcopy(self.config)
        config['body_from_imu'][0][0] = 2.
        self.assertIn('invalid_camera_or_imu_transform', calibration_reasons(config))
        config = copy.deepcopy(self.config)
        config.pop('calibration_evidence')
        self.assertIn('calibration_evidence_required', calibration_reasons(config))

    def test_matches_urdf_and_simulation_optical_mount(self):
        evidence = self.config['calibration_evidence']
        urdf = ROOT/evidence['urdf_path']
        mount_path = ROOT/evidence['optical_mount_path']
        self.assertEqual(hashlib.sha256(urdf.read_bytes()).hexdigest(), evidence['urdf_sha256'])
        self.assertEqual(hashlib.sha256(mount_path.read_bytes()).hexdigest(), evidence['optical_mount_sha256'])
        mount = json.loads(mount_path.read_text())
        transform = ProHeadModel(urdf).camera_link_transform(self.config['head_reference_rad'])
        x, y = np.asarray(mount['xyaxes_camera_link']).reshape(2, 3)
        x = x/np.linalg.norm(x)
        y = y-x*np.dot(x, y)
        y = y/np.linalg.norm(y)
        optical = np.column_stack((x, -y, -np.cross(x, y)))
        np.testing.assert_allclose(self.config['camera_rotation_optical_to_body'], transform[:3, :3]@optical, atol=1e-10)
        np.testing.assert_allclose(self.config['camera_position_body_m'],
                                   transform[:3, 3]+transform[:3, :3]@mount['position_camera_link_m'], atol=1e-10)
        np.testing.assert_array_equal(self.config['body_from_imu'], np.eye(3))


if __name__ == '__main__':
    unittest.main()
