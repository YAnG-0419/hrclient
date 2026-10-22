"""Read saved EEF/gripper frames; retain FK for IK and feedback validation only."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from ..teleop.teleop import validate_mapping

ROLES = {'left': 'follower_1', 'right': 'follower_2', 'active_camera': 'follower_3'}


@dataclass
class PoseTarget:
    xyz_m: np.ndarray
    quaternion_xyzw: np.ndarray
    gripper_mm: float | None
    reference_joints_deg: np.ndarray
    command_joints_deg: np.ndarray | None = None


@dataclass
class ArmTrajectory:
    joints_deg: np.ndarray
    xyz_m: np.ndarray
    quaternion_xyzw: np.ndarray
    gripper_mm: np.ndarray | None


@dataclass
class Episode:
    times_s: np.ndarray
    targets: dict[str, ArmTrajectory]
    metadata: dict
    source_sha256: str


def file_hash(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ForwardKinematics:
    """Use the pinned SDK's DH=1 link transforms, retaining full rotation matrices."""
    def __init__(self, sdk_file: Path, *, robot_model='piper'):
        if robot_model not in ('piper','piper_h'):
            raise ValueError(f'未支持的运动学型号: {robot_model}')
        spec = importlib.util.spec_from_file_location('egoav_offline_fk', sdk_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.C_PiperForwardKinematics(dh_is_offset=1)
        self.robot_model=robot_model
        if robot_model=='piper_h':
            # Official pyAgxArm ROBOT_MDH_PRESET: H differs in d4/d6.
            # The older piper_sdk only provides the ordinary Piper geometry.
            self.model._d[3]=255.75
            self.model._d[5]=96.3
        self.sdk_sha256 = file_hash(sdk_file)

    def pose(self, joints_deg):
        joints = np.asarray(joints_deg, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError('FK需要六个有限关节角（度）')
        matrix = np.eye(4)
        model = self.model
        for i, angle in enumerate(np.deg2rad(joints)):
            link = model._C_PiperForwardKinematics__LinkTransformtion(
                model._alpha[i], model._a[i], angle + model._theta[i], model._d[i])
            matrix = matrix @ np.asarray(link).reshape(4, 4)
        return matrix[:3, 3] / 1000, Rotation.from_matrix(matrix[:3, :3]).as_quat()


def load_episode(path: Path, mapping: dict, sdk_file: Path, *, robot_model='piper') -> Episode:
    validate_mapping(mapping)
    before = file_hash(path)
    fk = ForwardKinematics(sdk_file,robot_model=robot_model)
    with h5py.File(path, 'r') as source:
        if not source.attrs.get('complete', False):
            raise ValueError('采集未标记成功，不能回放')
        metadata = json.loads(source.attrs['metadata_json'])
        if metadata['recording_mapping']['arm_roles'] != ROLES:
            raise ValueError('采集角色映射与当前三从动臂不一致')
        recorded = {a['id']: a for a in metadata['arm_mapping']['arms']}
        current = {a['id']: a for a in mapping['arms']}
        for name in ROLES.values():
            for key, default in (('usb_can_serial', None), ('role', None), ('has_gripper', True)):
                if recorded[name].get(key, default) != current[name].get(key, default):
                    raise ValueError(f'{name}: 采集与当前机械臂身份/配置不一致: {key}')
        stamps = source['time/sample_monotonic_ns'][:]
        if (stamps.ndim != 1 or len(stamps) < 2 or stamps.dtype.kind not in 'iu'
                or np.any(stamps <= 0) or np.any(stamps[1:] <= stamps[:-1])):
            raise ValueError('样本时间戳必须正值、严格递增且至少两帧')
        times = (stamps.astype(np.int64) - int(stamps[0])) / 1e9
        if np.max(np.diff(times)) > .1:
            raise ValueError('源采样间隔超过100ms，不能作为连续回放轨迹')
        targets = {}
        for role, name in ROLES.items():
            group = source[f'observations/{role}']
            joints = np.asarray(group['joint_position_deg'][:], dtype=float)
            if joints.shape != (len(times), 6) or not np.isfinite(joints).all():
                raise ValueError(f'{role}: 关节列长度或数值无效')
            if 'eef' not in group:
                raise ValueError(f'{role}: 缺少eef字段，请先补入末端数据；不从关节角重算轨迹')
            dataset = group['eef']
            eef = np.asarray(dataset[:], dtype=float)
            if eef.shape != (len(times), 7) or not np.isfinite(eef).all():
                raise ValueError(f'{role}: eef必须是逐帧(N,7)有限数值')
            if (list(dataset.attrs.get('columns', [])) != ['x','y','z','qx','qy','qz','qw']
                    or dataset.attrs.get('position_unit') != 'm'
                    or dataset.attrs.get('quaternion_order') != 'xyzw'):
                raise ValueError(f'{role}: eef格式必须为米制xyz和xyzw四元数')
            xyz, quat = eef[:, :3].copy(), eef[:, 3:].copy()
            if not np.allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-6, rtol=0):
                raise ValueError(f'{role}: eef四元数必须归一化')
            grip = None
            if current[name].get('has_gripper', True):
                grip = np.asarray(group['gripper_width_mm'][:], dtype=float)
                if grip.shape != (len(times),) or not np.isfinite(grip).all():
                    raise ValueError(f'{role}: 夹爪列长度或数值无效')
            elif 'gripper_width_mm' in group:
                raise ValueError('主动相机臂不应有夹爪字段')
            targets[role] = ArmTrajectory(joints, xyz, quat, grip)
    if file_hash(path) != before:
        raise ValueError('准备轨迹期间源文件发生变化')
    metadata.update(dh_is_offset=1, sdk_sha256=fk.sdk_sha256, sdk_path=str(sdk_file),
                    robot_model=robot_model,link_offsets_mm=list(fk.model._d),
                    source_path=str(path), trajectory_source='hdf5_eef',
                    coordinate_frame='each_arm_own_base_link',
                    end_frame='SDK link6, no TCP/optical offset', quaternion_order='xyzw')
    return Episode(times, targets, metadata, before)


def frame_episode(episode: Episode, index: int) -> dict[str, PoseTarget]:
    """Return an original recorded frame, without any interpolation."""
    if not 0<=index<len(episode.times_s): raise ValueError('回放帧编号越界')
    return {role:PoseTarget(arm.xyz_m[index].copy(),arm.quaternion_xyzw[index].copy(),
            float(arm.gripper_mm[index]) if arm.gripper_mm is not None else None,
            arm.joints_deg[index].copy()) for role,arm in episode.targets.items()}
