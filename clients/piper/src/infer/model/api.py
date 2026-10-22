"""Canonical model boundary: RGB, measured EEF, absolute EEF actions."""
import math
import numpy as np

ROLES = ('left', 'right', 'active_camera')
GRIPPERS = ('left', 'right')
CONTRACT = 'egoav.absolute_eef.xyzw.m.gripper_mm.v1'


def fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f'{label}: required fields {list(names)}')


def poses(value, shape, label):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{label}: expected finite {shape}')
    if not np.allclose(np.linalg.norm(result[..., 3:], axis=-1), 1, atol=1e-6, rtol=0):
        raise ValueError(f'{label}: quaternion must be unit xyzw')
    return result.copy()


def validate_observation(obs):
    fields(obs, ('images', 'eef', 'gripper_mm', 'prompt'), 'observation')
    fields(obs['images'], ROLES, 'images'); fields(obs['eef'], ROLES, 'eef')
    fields(obs['gripper_mm'], GRIPPERS, 'gripper_mm')
    if not isinstance(obs['prompt'], str) or not obs['prompt'].strip() or len(obs['prompt']) > 4096:
        raise ValueError('prompt must be nonempty text, <=4096 characters')
    images = {}
    for role in ROLES:
        image = np.asarray(obs['images'][role])
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
                or min(image.shape[:2]) < 1 or max(image.shape[:2]) > 4096):
            raise ValueError(f'{role}: expected HWC uint8 RGB image')
        images[role] = image.copy()
    grips = {r: float(obs['gripper_mm'][r]) for r in GRIPPERS}
    if not all(math.isfinite(x) for x in grips.values()): raise ValueError('invalid gripper feedback')
    return {'images': images, 'eef': {r: poses(obs['eef'][r], (7,), r) for r in ROLES},
            'gripper_mm': grips, 'prompt': obs['prompt']}


def validate_actions(value, *, max_steps, dt_s):
    fields(value, ('eef', 'gripper_mm', 'dt_s'), 'actions')
    fields(value['eef'], ROLES, 'action eef'); fields(value['gripper_mm'], GRIPPERS, 'action grippers')
    actual_dt = float(value['dt_s'])
    if not math.isfinite(actual_dt) or actual_dt <= 0 or not math.isclose(actual_dt, dt_s, abs_tol=1e-9, rel_tol=0):
        raise ValueError('action dt_s differs from configured/training interval')
    count = len(value['eef']['left'])
    if not 1 <= count <= max_steps: raise ValueError('action horizon empty or too large')
    eef = {r: poses(value['eef'][r], (count,7), r) for r in ROLES}
    grips = {r: np.asarray(value['gripper_mm'][r], dtype=float) for r in GRIPPERS}
    if any(v.shape != (count,) or not np.isfinite(v).all() for v in grips.values()):
        raise ValueError('gripper action length/values invalid')
    return {'eef': eef, 'gripper_mm': {r:v.copy() for r,v in grips.items()}, 'dt_s': actual_dt}
