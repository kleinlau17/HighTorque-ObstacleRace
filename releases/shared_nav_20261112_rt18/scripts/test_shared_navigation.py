"""Shared navigation contract tests, runnable on Python 3.8 without ROS/MuJoCo."""
from pathlib import Path
import ast
import json
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import tempfile
import hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from navigation.command import AMPCommandAdapter, load_contract, verify_device_files
from navigation.runtime import CameraFrame, NavigationRuntime
from navigation.visual import Perception, VisualNavigator
from navigation.calibration import calibration_reasons


class SharedTests(unittest.TestCase):
    def test_projection_needs_head_feedback_at_image_time_even_in_shadow(self):
        from deploy.shared_navigation import frame_head_reasons
        config = json.loads((ROOT/'deploy/pro_config.json').read_text())
        captured = {name: (5., angle, 0.) for name, angle in config['head_reference_rad'].items()}
        self.assertEqual(frame_head_reasons(config, captured, 5.02), [])
        self.assertTrue(frame_head_reasons(config, captured, 4.7))
        self.assertTrue(frame_head_reasons(config, captured, 5.3))
        captured['head_yaw_joint'] = (5., .15, 0.)
        self.assertTrue(frame_head_reasons(config, captured, 5.))
        del captured['head_pitch_joint']
        self.assertTrue(frame_head_reasons(config, captured, 5.))

    def test_dependency_mismatch_is_rejected(self):
        from navigation.environment import verify_dependencies
        verify_dependencies()
        with patch('navigation.environment.cv2.__version__','5.0.0'):
            with self.assertRaisesRegex(ValueError,'Navigation dependency differs'):verify_dependencies()

    def test_existing_sim_import_is_the_same_implementation(self):
        from sim.visual_navigation import VisualNavigator as SimNavigator
        self.assertIs(SimNavigator, VisualNavigator)

    def test_core_has_no_simulator_ros_or_policy_inference_dependencies(self):
        for path in (ROOT/'navigation').glob('*.py'):
            tree = ast.parse(path.read_text(), feature_version=(3, 8))
            for node in ast.walk(tree):
                names = [v.name for v in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertNotIn(name.split('.')[0], ('mujoco', 'sim', 'rospy', 'torch', 'scipy'))

    def test_amp_adapter_starts_without_modifying_factory_deadband(self):
        c = load_contract();adapter = AMPCommandAdapter();filtered = np.zeros(3)
        for _ in range(20):
            raw = adapter.update([.22, -.10, .20], filtered)
            self.assertTrue(np.all(abs(raw) <= [.4, .4, 1.]))
            filtered = (c['alpha']*raw+(1-c['alpha'])*filtered)*c['scale']
            filtered[abs(filtered)<np.array(c['deadband'])] = 0
            np.testing.assert_allclose(adapter.filtered, filtered, atol=1e-12)
        np.testing.assert_allclose(filtered, [.22, -.10, .20], atol=1e-9)
        # The unadapted factory input still gets eaten by the deadband.
        unadapted = c['alpha']*np.array([.22, -.10, .20])
        self.assertTrue(np.all(abs(unadapted)<c['deadband']))

    def test_stop_and_disabled_never_send_reverse_cancellation(self):
        adapter = AMPCommandAdapter()
        for enabled in (True, False):
            raw = adapter.update([0, 0, 0], [.4, -.3, .8], enabled=enabled)
            np.testing.assert_array_equal(raw, 0)
        np.testing.assert_array_equal(adapter.update([.2, .2, .8], enabled=False), 0)

    def test_invalid_command_and_feedback_are_rejected(self):
        adapter = AMPCommandAdapter()
        for value in ([float('nan'), 0, 0], [1, 2]):
            with self.assertRaises(ValueError):adapter.update(value)
            with self.assertRaises(ValueError):adapter.update([0, 0, 0], value)

    def test_stale_images_or_imu_stop_without_extrapolating_commands(self):
        runtime = NavigationRuntime(32, 24, [30, 30, 15, 11], [0, 0, 0], np.eye(3))
        runtime.set_attitude(1., np.eye(3))
        runtime.last_frame_time = 1.
        runtime.good_frames = 3
        p = Perception(1., np.zeros(3), np.empty((0, 2)), np.empty((0, 4)),
                       np.empty((0, 2)), 0., np.zeros((24, 32, 3), np.uint8))
        runtime.navigator.observe(p, np.zeros(3), 0.)
        runtime.navigator.command[:] = [.22, 0, .2]
        np.testing.assert_array_equal(runtime.command(1.5, [.22, 0, .2]), 0.)
        self.assertEqual(runtime.reason, 'stale_imu')
        runtime.set_attitude(1.51, np.eye(3))
        np.testing.assert_array_equal(runtime.command(1.51), 0.)
        self.assertEqual(runtime.reason, 'stale_camera')

    def test_invalid_geometry_or_sensor_timestamps_rejected(self):
        with self.assertRaises(ValueError):NavigationRuntime(32, 24, [30, 30, 15, 11], [0, 0, 0], np.zeros((3, 3)))
        runtime = NavigationRuntime(32, 24, [30, 30, 15, 11], [0, 0, 0], np.eye(3))
        runtime.set_attitude(1., np.eye(3))
        with self.assertRaises(ValueError):runtime.set_attitude(.9, np.eye(3))
        with self.assertRaises(ValueError):runtime.observe(CameraFrame(0., np.zeros((24, 32, 3), np.uint8), np.ones((24, 32))))

    def test_frozen_amp_files_are_verified(self):
        path = ROOT/'vendor/pi_plus_pro'
        if not path.exists():self.skipTest('CPU test bundle not installed on this host')
        actual = verify_device_files(path, pd_path=path/'pd.yaml')
        self.assertEqual(actual, load_contract()['sha256'])

    def test_simulation_placeholder_cannot_enable_real_motion(self):
        config = json.loads((ROOT/'deploy/pro_config.json').read_text())
        self.assertIn('extrinsics_verified_required', calibration_reasons(config))
        self.assertIn('imu_alignment_verified_required', calibration_reasons(config))
        config.update(extrinsics_verified=True, imu_alignment_verified=True)
        self.assertIn('calibration_evidence_required', calibration_reasons(config))

    def test_camera_uses_exposure_associated_attitude_without_rewinding_control_imu(self):
        from navigation.visual import rotation2
        runtime = NavigationRuntime(32,24,[30,30,15,11],[0,0,0],np.eye(3))
        runtime.set_attitude(1.,np.eye(3))
        new = np.eye(3);new[:2,:2]=rotation2(.1)
        runtime.set_attitude(1.1,new)
        # Empty imagery cannot create a route, but association still uses the
        # older frame's own attitude while retaining the latest control yaw.
        runtime.observe(CameraFrame(1.05,np.zeros((24,32,3),np.uint8),np.full((24,32),np.nan)),np.eye(3),1.05)
        self.assertAlmostEqual(runtime.yaw,.1)
        self.assertEqual(runtime.last_imu_time,1.1)
        self.assertIsNone(runtime.observation.lane)

    def test_visual_motion_recovers_across_a_dropped_frame(self):
        import cv2
        from navigation.visual import RGBDOdometry
        rotation=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
        vo=RGBDOdometry(160,120,80.,np.zeros(3),rotation,[80,80,79.5,59.5])
        gray=np.random.default_rng(2).integers(0,256,(120,160),dtype=np.uint8)
        rgb=np.repeat(gray[:,:,None],3,axis=2)
        depth=np.ones((120,160))
        xy=vo.update(CameraFrame(0.,rgb,depth),np.eye(3),np.zeros(2))
        xy=vo.update(CameraFrame(.067,np.zeros_like(rgb),np.full_like(depth,np.nan)),np.eye(3),xy)
        self.assertFalse(vo.accepted)
        moved=cv2.warpAffine(rgb,np.array([[1,0,2],[0,1,0.]],dtype=float),(160,120))
        xy=vo.update(CameraFrame(.8,moved,depth),np.eye(3),xy)
        self.assertTrue(vo.accepted)
        self.assertAlmostEqual(np.linalg.norm(xy),.025,delta=.004)

    def test_lost_visual_motion_keeps_hidden_obstacles_and_inhibits_motion(self):
        runtime=NavigationRuntime(32,24,[30,30,15,11],[0,0,0],np.eye(3))
        runtime.set_attitude(1.,np.eye(3))
        runtime.last_frame_time=0.
        runtime.perception.obstacle_memory=[dict(center=np.array([.4,.15]),size=np.array([.2,.2]),time=0.)]
        runtime.navigator.corridor_heading=0.
        runtime.navigator.corridor_point=np.zeros(2)
        frame=CameraFrame(1.,np.zeros((24,32,3),np.uint8),np.full((24,32),np.nan))
        with patch.object(runtime.perception,'update',side_effect=AssertionError('Must not integrate at an unknown pose')):
            result=runtime.observe(frame)
        self.assertEqual(len(runtime.perception.obstacle_memory),1)
        np.testing.assert_allclose(result.obstacles,[[.4,.15,.2,.2]])
        self.assertEqual(runtime.navigator.corridor_heading,0.)
        np.testing.assert_array_equal(runtime.command(1.),0.)
        runtime.set_attitude(1.8,np.eye(3))
        runtime.observe(CameraFrame(1.8,frame.rgb,frame.depth))
        self.assertEqual(runtime.localization_reseeds,0)
        self.assertEqual(len(runtime.perception.obstacle_memory),1)

    def test_motion_preflight_rejects_placeholder_before_importing_ros(self):
        from deploy.line_trial import supervisor_role
        args=SimpleNamespace(shared_navigation=True,config=ROOT/'deploy/pro_config.json')
        with self.assertRaisesRegex(ValueError,'before changing control ownership'):
            supervisor_role(args)

    def test_release_rejects_changed_files_and_paths_outside_release(self):
        from navigation.release import verify_release
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'module.py';source.write_text('value=1\n')
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            manifest=root/'release_manifest.json'
            manifest.write_text(json.dumps(dict(files={'module.py':digest})))
            verify_release(root)
            source.write_text('value=2\n')
            with self.assertRaises(ValueError):verify_release(root)
            manifest.write_text(json.dumps(dict(files={'../outside.py':digest})))
            with self.assertRaises(ValueError):verify_release(root)

    def test_calibrated_head_requires_fresh_matching_feedback(self):
        config=json.loads((ROOT/'deploy/pro_config.json').read_text())
        config.update(extrinsics_verified=True,imu_alignment_verified=True,
            calibration_evidence='test fixture only',head_reference_rad={'head_pitch_joint':.6,'head_yaw_joint':0.})
        self.assertFalse(calibration_reasons(config,{'head_pitch_joint':.61,'head_yaw_joint':0.}))
        self.assertIn('head_pose_not_at_calibrated_reference',calibration_reasons(config,{}))
        self.assertIn('head_pose_not_at_calibrated_reference',calibration_reasons(config,{'head_pitch_joint':.8,'head_yaw_joint':0.}))


if __name__ == '__main__':
    unittest.main()
