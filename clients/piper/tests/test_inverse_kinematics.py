import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from src.replay.inverse_kinematics import solve_pose


class BoundaryFK:
    def pose(self, joints_deg):
        joints=np.asarray(joints_deg,dtype=float)
        xyz=np.array([joints[0]/1000.,0.,0.])
        quat=Rotation.from_euler('z',joints[1],degrees=True).as_quat()
        return xyz,quat


class InverseKinematicsToleranceTests(unittest.TestCase):
    def setUp(self):
        self.fk=BoundaryFK()
        self.seed=np.zeros(6)
        self.bounds=np.array([[-180.,180.],[-10.,10.]]+[[-180.,180.]]*4)

    def test_accepts_solution_within_two_mm_and_one_degree(self):
        target_quat=Rotation.from_euler('z',10.5,degrees=True).as_quat()
        solved=solve_pose(self.fk,[.1815,0.,0.],target_quat,self.seed,self.bounds)
        xyz,quat=self.fk.pose(solved)
        self.assertLessEqual(np.linalg.norm(xyz-[.1815,0.,0.])*1000,2.)
        angle=np.rad2deg((Rotation.from_quat(quat).inv()*Rotation.from_quat(target_quat)).magnitude())
        self.assertLessEqual(angle,1.)

    def test_accepts_solution_outside_former_residual_limits(self):
        identity=np.array([0.,0.,0.,1.])
        solved=solve_pose(self.fk,[.183,0.,0.],identity,self.seed,self.bounds)
        xyz,_=self.fk.pose(solved)
        self.assertGreater(np.linalg.norm(xyz-[.183,0.,0.])*1000,2.)
        self.assertTrue(np.isfinite(solved).all())
        self.assertTrue(np.all(solved>=self.bounds[:,0]))
        self.assertTrue(np.all(solved<=self.bounds[:,1]))
        target_quat=Rotation.from_euler('z',11.5,degrees=True).as_quat()
        solved=solve_pose(self.fk,[.18,0.,0.],target_quat,self.seed,self.bounds)
        _,quat=self.fk.pose(solved)
        angle=np.rad2deg((Rotation.from_quat(quat).inv()*Rotation.from_quat(target_quat)).magnitude())
        self.assertGreater(angle,1.)
        self.assertTrue(np.isfinite(solved).all())


if __name__=='__main__': unittest.main()
