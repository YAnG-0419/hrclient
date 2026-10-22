"""End-pose feedback and MIT joint streaming of precomputed Cartesian IK."""
from dataclasses import dataclass
import struct
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from ..teleop.piper_io import PiperIO
from .trajectory import PoseTarget


@dataclass
class CartesianState:
    xyz_m: np.ndarray
    quaternion_xyzw: np.ndarray
    rpy_deg: np.ndarray
    feedback_time: float


def continuous_euler(quaternion, previous=None):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-6:
        raise ValueError('目标四元数无效或不是单位四元数')
    rotation = Rotation.from_quat(q)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        base = rotation.as_euler('xyz', degrees=True)
    if previous is None: return base
    previous = np.asarray(previous)
    candidates = [base, np.array([base[0]+180, 180-base[1], base[2]+180])]
    if abs(abs(base[1])-90) < 1e-7:
        sign = np.sign(base[1])
        yaw = (sign*(previous[0]-base[0]) + previous[2]) / 2
        candidates.append(np.array([base[0]+sign*yaw, base[1], yaw]))
    candidates = [c+360*np.round((previous-c)/360) for c in candidates]
    result = min(candidates, key=lambda c: np.linalg.norm(c-previous))
    if (Rotation.from_euler('xyz',result,degrees=True).inv()*rotation).magnitude() > 1e-7:
        raise ValueError('欧拉角转换不能可靠还原目标姿态')
    return result


def pose_error(target, actual):
    position = float(np.linalg.norm(np.asarray(target.xyz_m)-actual.xyz_m)*1000)
    angle = float(np.rad2deg((Rotation.from_quat(actual.quaternion_xyzw).inv()
                             * Rotation.from_quat(target.quaternion_xyzw)).magnitude()))
    return position, angle


class ReplayIO(PiperIO):
    def __init__(self, interfaces, has_gripper=None):
        if not set(interfaces) <= {'follower_1','follower_2','follower_3'}:
            raise ValueError('ReplayIO只允许三只从动臂')
        super().__init__(interfaces, has_gripper)
        self._last_rpy = {}
        self.limits = {}

    def read_cartesian(self, name, now):
        feedback = self.feedback[name]
        values, stamps = [], []
        for can_id in (0x2A2,0x2A3,0x2A4):
            values.extend(struct.unpack('>ii',feedback._get(can_id,now)))
            stamps.append(feedback.frames[can_id][1])
        if max(stamps)-min(stamps) > .05:
            raise RuntimeError(f'{name}: 末端反馈分片时间差超过50ms')
        xyz, rpy = np.asarray(values[:3],dtype=float)/1e6, np.asarray(values[3:],dtype=float)/1000
        return CartesianState(xyz,Rotation.from_euler('xyz',rpy,degrees=True).as_quat(),rpy,min(stamps))

    def enter_cartesian(self, name, measured):
        # Same streaming mode as working teleoperation; anchor at measured joints.
        state = self.read_one(name)
        self.mit_mode(name)
        self.send_pose(name,state.pose)

    def send_cartesian(self, name, target):
        xyz = np.asarray(target.xyz_m,dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError('末端位置必须是三个有限米数值')
        quat = np.asarray(target.quaternion_xyzw,dtype=float)
        if quat.shape!=(4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>1e-6:
            raise ValueError('末端四元数无效')
        joints = np.asarray(target.command_joints_deg,dtype=float)
        if joints.shape!=(6,) or not np.isfinite(joints).all() or name not in self.limits:
            raise ValueError('必须提供已经校验的六轴IK解；禁止使用原始关节列代替')
        limits = np.asarray(self.limits[name])[:6,:2]
        if np.any(joints<limits[:,0]) or np.any(joints>limits[:,1]):
            raise ValueError('IK解超出设备关节范围')
        grip = target.gripper_mm
        if grip is not None:
            if not self.feedback[name].has_gripper or not np.isfinite(grip):
                raise ValueError(f'{name}: 无效夹爪目标')
            low, high = (self.limits[name][-1][:2] if name in self.limits else (0,100))
            if not low <= grip <= high: raise ValueError(f'{name}: 夹爪目标越界')
        self.sdks[name].JointCtrl(*(int(round(v*1000)) for v in joints))
        if grip is not None:
            self.sdks[name].GripperCtrl(int(round(grip*1000)),1000,1,0)

    def stop_cartesian(self, name):
        """Attempt hold, or software quick stop; never report a physical E-stop."""
        try:
            state = self.read_one(name)
            if state.control not in (1,2) or state.move != 1 or not state.enabled or state.teach not in (0,1,2):
                raise RuntimeError('末端控制模式/使能不再有效')
            grip = state.pose[6] if self.feedback[name].has_gripper else None
            if grip is not None:
                low, high = self.limits.get(name,[(0,100)])[-1][:2]
                grip = float(np.clip(grip,low,high))
            pose = tuple(state.pose[:6]) + ((grip,) if grip is not None else ())
            self.send_pose(name,pose)
            return {'arm':name,'action':'hold_requested','confirmed_stopped':False}
        except Exception as exc:
            try:
                self.quick_stop(name)
                return {'arm':name,'action':'quick_stop_requested','confirmed_stopped':False,'reason':str(exc)}
            except Exception as failure:
                return {'arm':name,'action':'stop_failed','confirmed_stopped':False,'reason':str(failure)}
