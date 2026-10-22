"""TCP v4 client and tunnel tests; no robot, camera, model, or GPU access."""
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _ours_protocol import recv_message, send_message
from client_mixed76 import Mixed76Client
from inference_connection import SSHTunnel, inference_connection
from mixed76_contract import REPRESENTATION, embodiment_mask, encode_state, decode_actions


def metadata(model):
    return dict(protocol_version=4, model=model, representation=REPRESENTATION,
                action_dim=76, state_dim=76, action_units="physical", state_units="physical",
                action_mode="absolute", horizon=32, dt_s=.05, camera_keys=["head_rgb"],
                weights_strict=True, checkpoint="test-only", normalization_sha256="test",
                deployment_id="test", prompts={"basket": "test", "yifu": "test"},
                embodiments=["3b", "g2"])


class FakeServer:
    def __init__(self, model="joint", mutate=None):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.port = self.listener.getsockname()[1]
        self.model, self.mutate = model, mutate
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        with self.listener:
            connection, _ = self.listener.accept()
            with connection:
                while True:
                    try:
                        request = recv_message(connection)
                    except ConnectionError:
                        return
                    if request["cmd"] == "info":
                        result = metadata(self.model)
                    elif request["cmd"] == "reset":
                        result = dict(protocol_version=4, request_id=request["request_id"],
                                      deployment_id="test", session_id="s", ok=True)
                    else:
                        result = {key: request[key] for key in
                                  ("request_id", "session_id", "sample_ns", "embodiment")}
                        result.update(protocol_version=4, deployment_id="test",
                                      representation=REPRESENTATION, action_units="physical",
                                      dt_s=.05, inference_seconds=.001,
                                      action_mask=embodiment_mask(request["embodiment"]).tolist(),
                                      actions=np.tile(request["observation"]["state"], (32, 1)))
                    if self.mutate:
                        self.mutate(request, result)
                    send_message(connection, result)


class ClientTests(unittest.TestCase):
    def test_all_models_and_both_embodiments(self):
        for model in ("fastwam", "alpha", "joint", "tri", "coarse_fine", "pi05"):
            for kind in ("3b", "g2"):
                with self.subTest(model=model, kind=kind):
                    server = FakeServer(model)
                    client = Mixed76Client("127.0.0.1", server.port, model=model)
                    try:
                        actions = client.infer(np.zeros(76, np.float32),
                            np.zeros((10, 12, 3), np.uint8), task="basket", embodiment=kind)
                        self.assertEqual(actions.shape, (32, 76))
                    finally:
                        client.close()
                    server.thread.join(2)

    def test_wrong_model_and_stale_identity_are_rejected(self):
        server = FakeServer("alpha")
        with self.assertRaisesRegex(ValueError, "model"):
            Mixed76Client("127.0.0.1", server.port, model="joint")
        server.thread.join(2)
        for key, value in (("request_id", "stale"), ("deployment_id", "other"),
                           ("dt_s", .1), ("action_mask", []),
                           ("actions", np.zeros((1, 76))), ("inference_seconds", -1)):
            with self.subTest(key=key):
                def mutate(request, response):
                    if request["cmd"] == "infer":
                        response[key] = value
                server = FakeServer(mutate=mutate)
                client = Mixed76Client("127.0.0.1", server.port, model="joint")
                try:
                    with self.assertRaises(ValueError):
                        client.infer(np.zeros(76, np.float32),
                            np.zeros((2, 2, 3), np.uint8), task="basket", embodiment="3b")
                finally:
                    client.close()
                server.thread.join(2)

    def test_native_roundtrip(self):
        for kind, dim, quaternion_starts in (("3b", 23, (0, 8, 16)),
                                             ("g2", 61, (0, 29))):
            native = np.zeros(dim, np.float32)
            for start in quaternion_starts:
                native[start + 6] = 1
            encoded = encode_state(native, kind)
            np.testing.assert_allclose(decode_actions(encoded[None], kind)[0], native)


class TunnelTests(unittest.TestCase):
    def args(self, **changes):
        values = dict(inference="local", ip="127.0.0.1", port=19000, remote_config=None)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_local_mode_does_not_start_ssh(self):
        with patch("inference_connection.subprocess.run", side_effect=AssertionError("SSH called")):
            with inference_connection(self.args()) as address:
                self.assertEqual(address, ("127.0.0.1", 19000))

    def test_incomplete_remote_config_fails_before_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "remote.json"
            path.write_text(json.dumps(dict(ssh_host="host", server_host="127.0.0.1",
                                            server_port=None)))
            with patch("inference_connection.subprocess.run", side_effect=AssertionError("SSH called")):
                with self.assertRaisesRegex(ValueError, "server_port"):
                    with inference_connection(self.args(inference="remote", remote_config=path)):
                        pass

    def test_failed_ssh_has_no_local_fallback(self):
        calls = []
        def failed(command, **kwargs):
            calls.append(command)
            return SimpleNamespace(returncode=255)
        with patch("inference_connection.subprocess.run", side_effect=failed):
            with self.assertRaisesRegex(RuntimeError, "SSH"):
                with SSHTunnel("host", "127.0.0.1", 19000):
                    pass
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
