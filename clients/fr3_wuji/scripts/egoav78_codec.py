"""FR3/Wuji 54D joints <-> EgoAV78 cam0/link8 representation."""

import numpy as np

HAND_SLOTS = np.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21])
LEFT_ARM_54, LEFT_HAND_54 = slice(0, 7), slice(7, 27)
RIGHT_ARM_54, RIGHT_HAND_54 = slice(27, 34), slice(34, 54)
LEFT_POSE_78, RIGHT_POSE_78, HEAD_POSE_78 = slice(0, 9), slice(9, 18), slice(18, 27)
LEFT_HAND_78, RIGHT_HAND_78 = slice(32, 54), slice(55, 77)
IDENTITY_POSE9 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def model_mask():
    mask = np.zeros(78, dtype=bool)
    mask[:27] = True
    mask[32 + HAND_SLOTS] = True
    mask[55 + HAND_SLOTS] = True
    return mask


def execution_mask():
    mask = model_mask()
    mask[HEAD_POSE_78] = False
    return mask


def _array(value, shape, label, dtype=float):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label}: expected finite {shape}, got {result.shape}")
    return result


def matrix_to_pose9(transform):
    transform = _array(transform, (4, 4), "transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
        raise ValueError("transform rotation is not a proper rotation")
    return np.r_[transform[:3, 3], rotation[:, 0], rotation[:, 1]].astype(np.float32)


def pose9_to_matrix(pose):
    pose = _array(pose, (9,), "xyz + rot6d")
    first, second = pose[3:6], pose[6:9]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        raise ValueError("degenerate rotation first column")
    first = first / first_norm
    second = second - first * np.dot(first, second)
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        raise ValueError("degenerate rotation second column")
    second = second / second_norm
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((first, second, np.cross(first, second)))
    transform[:3, 3] = pose[:3]
    return transform


def encode_observation(state54, head_rgb, kinematics):
    state = _array(state54, (54,), "native FR3/Wuji state", np.float32)
    rgb = np.asarray(head_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("head_rgb must be uint8 HWC RGB")
    encoded = np.zeros(78, dtype=np.float32)
    for side, source, target in (
        ("left", LEFT_ARM_54, LEFT_POSE_78),
        ("right", RIGHT_ARM_54, RIGHT_POSE_78),
    ):
        base_to_link8 = kinematics.fk(state[source])
        camera_to_link8 = np.linalg.inv(kinematics.base_camera[side]) @ base_to_link8
        encoded[target] = matrix_to_pose9(camera_to_link8)
    encoded[HEAD_POSE_78] = IDENTITY_POSE9
    encoded[32 + HAND_SLOTS] = state[LEFT_HAND_54]
    encoded[55 + HAND_SLOTS] = state[RIGHT_HAND_54]
    if np.any(encoded[~model_mask()] != 0):
        raise AssertionError("inactive EgoAV78 inputs must be zero")
    return encoded, np.ascontiguousarray(rgb)


def _fast_ik(kinematics, target, seed, lower, upper):
    """Damped geometric-Jacobian IK for latency-sensitive live inference."""
    if not hasattr(kinematics, "origins") or not hasattr(kinematics, "types"):
        raise ValueError("fast IK requires the verified FR3 kinematic chain")
    q = np.asarray(seed, dtype=np.float64).copy()
    z = np.array([0.0, 0.0, 1.0])
    error = None
    for _ in range(80):
        transform, origins, axes = np.eye(4), [], []
        joint = 0
        for origin, kind in zip(kinematics.origins, kinematics.types):
            transform = transform @ origin
            if kind == "revolute":
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ z)
                cosine, sine = np.cos(q[joint]), np.sin(q[joint])
                transform = transform @ np.array([
                    [cosine, -sine, 0, 0],
                    [sine, cosine, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ])
                joint += 1
        error = kinematics.error(transform, target)
        if np.linalg.norm(error[:3]) <= 1e-5 and np.linalg.norm(error[3:]) <= 1e-4:
            return q, error
        axes = np.asarray(axes).T
        position_jacobian = np.stack(
            [np.cross(axes[:, index], transform[:3, 3] - origins[index]) for index in range(7)],
            axis=1,
        )
        rotation_jacobian = target[:3, :3].T @ axes
        jacobian = np.vstack((position_jacobian, rotation_jacobian))
        delta = -jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1e-5 * np.eye(6), error
        )
        largest = float(np.max(np.abs(delta)))
        if largest > 0.15:
            delta *= 0.15 / largest
        q = np.clip(q + delta, lower, upper)
    raise ValueError(
        f"fast IK failed: position={np.linalg.norm(error[:3]):.6g}m, "
        f"rotation={np.linalg.norm(error[3:]):.6g}rad"
    )


def decode_actions(
    actions78,
    seed54,
    kinematics,
    lower,
    upper,
    *,
    max_step=0.3,
    fast_ik=False,
    defer_hand_limits=False,
):
    actions = np.asarray(actions78, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1:] != (78,) or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("actions must be finite [horizon,78]")
    if np.any(actions[:, ~model_mask()] != 0):
        raise ValueError("inactive EgoAV78 action slots must be zero")
    seed = _array(seed54, (54,), "IK seed")
    lower, upper = _array(lower, (54,), "lower limits"), _array(upper, (54,), "upper limits")
    if np.any(lower >= upper) or np.any(seed < lower) or np.any(seed > upper):
        raise ValueError("invalid limits or seed outside limits")
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive and finite")
    joints = np.empty((len(actions), 54), dtype=np.float64)
    position_error, rotation_error = np.empty((len(actions), 2)), np.empty((len(actions), 2))
    previous = seed.copy()
    deferred_hand_values = 0
    for index, action in enumerate(actions):
        joints[index, LEFT_HAND_54] = action[32 + HAND_SLOTS]
        joints[index, RIGHT_HAND_54] = action[55 + HAND_SLOTS]
        for arm_index, (side, source, target) in enumerate((
            ("left", LEFT_POSE_78, LEFT_ARM_54),
            ("right", RIGHT_POSE_78, RIGHT_ARM_54),
        )):
            base_to_link8 = kinematics.base_camera[side] @ pose9_to_matrix(action[source])
            try:
                solver = _fast_ik if fast_ik else kinematics.ik
                joints[index, target], error = solver(
                    kinematics, base_to_link8, previous[target], lower[target], upper[target]
                ) if fast_ik else solver(base_to_link8, previous[target], lower[target], upper[target])
            except ValueError as exc:
                raise ValueError(f"frame {index}, {side} IK: {exc}") from exc
            position_error[index, arm_index] = np.linalg.norm(error[:3])
            rotation_error[index, arm_index] = np.linalg.norm(error[3:])
        arm_indices = np.r_[0:7, 27:34]
        arm_delta = np.abs(joints[index, arm_indices] - previous[arm_indices])
        delta = float(arm_delta.max())
        if delta > max_step:
            flat_joint = int(arm_delta.argmax())
            side = "left" if flat_joint < 7 else "right"
            joint = flat_joint if flat_joint < 7 else flat_joint - 7
            state_index = int(arm_indices[flat_joint])
            raise ValueError(
                f"frame {index}, {side} joint {joint}: arm IK branch/target step "
                f"{delta:.6g}rad exceeds {max_step}; "
                f"previous={previous[state_index]:.6g}, target={joints[index, state_index]:.6g}"
            )
        outside = np.flatnonzero((joints[index] < lower) | (joints[index] > upper))
        if len(outside):
            arm_outside = outside[np.isin(outside, arm_indices)]
            if len(arm_outside) or not defer_hand_limits:
                state_index = int(arm_outside[0] if len(arm_outside) else outside[0])
                side = "left" if state_index < 27 else "right"
                kind = "arm" if state_index in arm_indices else "hand"
                local_joint = state_index if state_index < 7 else (
                    state_index - 7 if state_index < 27 else (
                        state_index - 27 if state_index < 34 else state_index - 34
                    )
                )
                raise ValueError(
                    f"frame {index}, {side} {kind} joint {local_joint}: target "
                    f"{joints[index, state_index]:.6g} outside "
                    f"[{lower[state_index]:.6g}, {upper[state_index]:.6g}]"
                )
            deferred_hand_values += len(outside)
        previous = joints[index].copy()
    return joints, {
        "max_position_error_m": float(position_error.max()),
        "max_rotation_error_rad": float(rotation_error.max()),
        "max_arm_step_rad": float(np.max(np.abs(np.diff(np.vstack((seed, joints))[:, np.r_[0:7, 27:34]], axis=0)))),
        "hand_values_deferred_to_device_projection": deferred_hand_values,
        "ik_solver": "geometric_dls" if fast_ik else "reference_least_squares",
        "ignored_action_indices": np.flatnonzero(~execution_mask()).tolist(),
    }
