"""Bounded end-pose IK, computed before motion; no source-joint replay."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .trajectory import ROLES

def solve_pose(fk, xyz_m, quaternion_xyzw, seed_deg, bounds_deg, *, max_step_deg=None):
    """Solve one base/link6 pose near the preceding solution, within limits."""
    xyz = np.asarray(xyz_m, dtype=float)
    quat = np.asarray(quaternion_xyzw, dtype=float)
    seed = np.asarray(seed_deg, dtype=float)
    bounds = np.asarray(bounds_deg, dtype=float)
    if (xyz.shape != (3,) or quat.shape != (4,) or seed.shape != (6,)
            or bounds.shape != (6, 2) or not all(np.isfinite(x).all() for x in (xyz,quat,seed,bounds))
            or abs(np.linalg.norm(quat)-1) > 1e-6 or np.any(bounds[:,0]>=bounds[:,1])
            or np.any(seed<bounds[:,0]) or np.any(seed>bounds[:,1])):
        raise ValueError('IK输入位姿、种子或关节范围无效')
    target_rotation = Rotation.from_quat(quat)
    if max_step_deg is not None:
        if not np.isfinite(max_step_deg) or max_step_deg <= 0:
            raise ValueError('IK连续性范围必须为正数')
        bounds = np.column_stack((np.maximum(bounds[:, 0], seed-max_step_deg),
                                  np.minimum(bounds[:, 1], seed+max_step_deg)))

    def residual(joints):
        position, orientation = fk.pose(joints)
        rotation_error = (target_rotation.inv()*Rotation.from_quat(orientation)).as_rotvec()
        return np.concatenate(((position-xyz)*1000, rotation_error*100))

    def continuous_residual(joints):
        # Near a wrist singularity, pose alone cannot choose J4/J6 reliably.
        # A small seed penalty chooses the nearby solution without using recorded joints.
        return np.concatenate((residual(joints), 0.01*(joints-seed)))

    objective = residual if max_step_deg is None else continuous_residual
    result = least_squares(objective, seed, bounds=(bounds[:,0],bounds[:,1]),
                           max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError('IK求解器失败或返回非有限关节角')
    if max_step_deg is not None:
        error = residual(result.x)
        position_mm = np.linalg.norm(error[:3])
        angle_deg = np.rad2deg(np.linalg.norm(error[3:])/100)
        if position_mm > 0.1 or angle_deg > 0.1:
            raise ValueError(f'连续IK未满足末端精度: {position_mm:.4f}mm/{angle_deg:.4f}deg')
    return result.x.copy()


def prepare_ik(episode, fk, home_poses, limits):
    """Solve all original frames, seeding only from home / previous IK output."""
    result = {}
    for role,name in ROLES.items():
        source = episode.targets[role]
        seed = np.asarray(home_poses[name][:6], dtype=float)
        bounds = np.asarray(limits[name], dtype=float)[:6,:2]
        commands = []
        for index in range(len(episode.times_s)):
            try:
                solved = solve_pose(fk, source.xyz_m[index], source.quaternion_xyzw[index], seed, bounds,
                                    max_step_deg=3 if index==0 else 15)
                if np.max(np.abs(solved-seed)) > (3 if index==0 else 15):
                    raise ValueError('IK关节分支/帧间跳变过大')
            except ValueError as exc:
                raise ValueError(f'{role} 第{index}帧: {exc}') from exc
            commands.append(solved)
            seed = solved
        result[role] = np.asarray(commands)
    return result
