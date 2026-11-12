import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from navigation.motion_recovery import MotionRecovery


class MotionRecoveryTests(unittest.TestCase):
    def test_stationary_requested_walk_gets_bounded_standing_recovery(self):
        r=MotionRecovery();pose=np.zeros(3);velocity=np.zeros(2);desired=np.array([.22,0,0])
        self.assertEqual(r.update(0,desired,pose,velocity,True),'ready')
        self.assertEqual(r.update(.81,desired,pose,velocity,True),'settle')
        self.assertEqual(r.update(1.30,desired,pose,velocity,True),'settle')
        self.assertEqual(r.update(1.32,desired,pose,velocity,True),'ready')
        self.assertEqual(r.update(2.2,desired,pose,velocity,True),'settle')
        r.update(2.71,desired,pose,velocity,True)
        self.assertEqual(r.update(3.6,desired,pose,velocity,True),'failed')

    def test_commanded_stop_untrusted_vision_and_real_progress_do_not_recover(self):
        for desired,velocity,enabled in (([0,0,0],[0,0],True),([.22,0,0],[.2,0],True),([.22,0,0],[0,0],False)):
            r=MotionRecovery()
            for time in range(10):
                self.assertEqual(r.update(time,np.array(desired),np.zeros(3),np.array(velocity),enabled),'ready')

    def test_verified_progress_restores_retry_budget(self):
        r=MotionRecovery();r.recoveries=2;r.anchor=np.zeros(2)
        r.update(0,np.array([.22,0,0]),np.array([.11,0,0]),np.array([.2,0]),True)
        self.assertEqual(r.recoveries,0)


if __name__=='__main__':unittest.main()
