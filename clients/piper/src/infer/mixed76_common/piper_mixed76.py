"""Piper adapter for the shared physical76 client; existing IK/MIT unchanged."""
import numpy as np
from mixed76_contract import encode_state, decode_actions


def encode_observation(observation):
    raw = np.concatenate((observation['eef']['left'],
                          [observation['gripper_mm']['left']],
                          observation['eef']['right'],
                          [observation['gripper_mm']['right']],
                          observation['eef']['active_camera']))
    return encode_state(raw, '3b'), observation['images']['active_camera']


def execution_actions(actions76):
    raw = decode_actions(actions76, '3b')
    return {'eef': {'left': raw[:, :7], 'right': raw[:, 8:15],
                    'active_camera': raw[:, 16:23]},
            'gripper_mm': {'left': raw[:, 7], 'right': raw[:, 15]}, 'dt_s': .05}
