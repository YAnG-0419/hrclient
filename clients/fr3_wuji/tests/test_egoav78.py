import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from egoav78_codec import (HAND_SLOTS, IDENTITY_POSE9, decode_actions,
                           encode_observation, execution_mask, model_mask,
                           pose9_to_matrix)
from inference_connection import load_remote_config, SSHTunnel


class FakeKinematics:
    def __init__(self):
        self.base_camera = {"left": np.eye(4), "right": np.eye(4)}

    def fk(self, q):
        result = np.eye(4)
        result[:3, 3] = q[:3]
        return result

    def ik(self, target, seed, lower, upper):
        result = np.asarray(seed).copy()
        result[:3] = target[:3, 3]
        error = np.zeros(6)
        return result, error


class JumpKinematics(FakeKinematics):
    def ik(self, target, seed, lower, upper):
        result, error = super().ik(target, seed, lower, upper)
        result[3] += 0.4
        return result, error


class CodecTests(unittest.TestCase):
    def test_masks_and_fixed_head(self):
        self.assertEqual(int(model_mask().sum()), 67)
        self.assertEqual(int(execution_mask().sum()), 58)
        self.assertTrue(model_mask()[18:27].all())
        self.assertFalse(execution_mask()[18:27].any())

    def test_state_and_actions_map_both_arms_and_hands(self):
        state = np.zeros(54, np.float32)
        state[:3], state[27:30] = [0.1, 0.2, 0.3], [-0.1, -0.2, 0.4]
        state[7:27] = np.arange(20) / 100
        state[34:54] = -np.arange(20) / 100
        encoded, rgb = encode_observation(state, np.zeros((4, 5, 3), np.uint8), FakeKinematics())
        np.testing.assert_array_equal(encoded[18:27], IDENTITY_POSE9)
        np.testing.assert_array_equal(encoded[32 + HAND_SLOTS], state[7:27])
        np.testing.assert_array_equal(encoded[55 + HAND_SLOTS], state[34:54])
        self.assertFalse(encoded[~model_mask()].any())
        actions = np.tile(encoded, (32, 1))
        joints, report = decode_actions(actions, state, FakeKinematics(), np.full(54, -2), np.full(54, 2))
        np.testing.assert_allclose(joints[:, 7:27], np.tile(state[7:27], (32, 1)))
        np.testing.assert_allclose(joints[:, 34:54], np.tile(state[34:54], (32, 1)))
        self.assertIn(18, report["ignored_action_indices"])
        self.assertEqual(rgb.shape, (4, 5, 3))

    def test_rot6d_degenerate_rejected(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            pose9_to_matrix(np.zeros(9))

    def test_ik_jump_reports_side_and_joint(self):
        state = np.zeros(54, np.float32)
        encoded, _ = encode_observation(state, np.zeros((4, 5, 3), np.uint8), JumpKinematics())
        with self.assertRaisesRegex(ValueError, r"frame 0, left joint 3:.*previous=0, target=0.4"):
            decode_actions(
                np.tile(encoded, (32, 1)), state, JumpKinematics(),
                np.full(54, -2), np.full(54, 2), max_step=0.3,
            )

    def test_fast_path_defers_hand_limits_to_device_projection(self):
        state = np.zeros(54, np.float32)
        encoded, _ = encode_observation(
            state, np.zeros((4, 5, 3), np.uint8), FakeKinematics()
        )
        actions = np.tile(encoded, (32, 1))
        actions[:, 32 + HAND_SLOTS[0]] = 3.0
        with self.assertRaisesRegex(
            ValueError, r"frame 0, left hand joint 0: target 3 outside \[-2, 2\]"
        ):
            decode_actions(
                actions, state, FakeKinematics(), np.full(54, -2), np.full(54, 2)
            )
        joints, report = decode_actions(
            actions,
            state,
            FakeKinematics(),
            np.full(54, -2),
            np.full(54, 2),
            defer_hand_limits=True,
        )
        self.assertTrue(np.all(joints[:, 7] == 3.0))
        self.assertEqual(report["hand_values_deferred_to_device_projection"], 32)


class ConfigTests(unittest.TestCase):
    def make_config(self, **changes):
        value = {
            "protocol": "egoav78_chunk_v1", "model_name": "joint", "ssh_host": "b300",
            "server_host": "127.0.0.1", "server_port": 18869,
            "deployment_id": "a" * 64, "status": "configured",
        }
        value.update(changes)
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "remote.json"
        path.write_text(json.dumps(value))
        return directory, path

    def test_config(self):
        directory, path = self.make_config()
        with directory:
            config = load_remote_config(path)
        self.assertEqual(config["model_name"], "joint")
        self.assertEqual(config["deployment_id"], "a" * 64)

    def test_wrong_protocol_rejected(self):
        directory, path = self.make_config(protocol="openpi_websocket_msgpack")
        with directory, self.assertRaisesRegex(ValueError, "not EgoAV78"):
            load_remote_config(path)

    def test_tunnel_rejects_shell_metacharacters(self):
        with self.assertRaisesRegex(ValueError, "ssh_host"):
            SSHTunnel("b300;bad", "127.0.0.1", 18869)


if __name__ == "__main__":
    unittest.main()
