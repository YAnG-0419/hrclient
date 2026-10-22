"""Strict synchronous EgoAV78 TCP v4 client. No robot imports or commands."""

import socket
import time
import uuid

import numpy as np

from egoav78_codec import model_mask
from wire_v4 import recv_message, send_message

TASK = "franka/tomato"
EMBODIMENT = "franka"
PROMPT = "Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket."


class EgoAV78Client:
    def __init__(self, host, port, *, deployment_id, timeout=300):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.session_id = None
        try:
            self.info = self._rpc({"cmd": "info"})
            expected = {
                "protocol": "egoav78_chunk_v1",
                "protocol_version": 4,
                "representation": "egoav78_rot6d_columns",
                "state_dim": 78,
                "action_dim": 78,
                "horizon": 32,
                "dt_s": 0.05,
                "state_units": "physical",
                "action_units": "physical",
                "action_mode": "absolute",
                "wire_action_mode": "absolute_in_requested_state_frame",
                "camera_keys": ["head_rgb"],
                "deployment_id": deployment_id,
            }
            for key, value in expected.items():
                if self.info.get(key) != value:
                    raise ValueError(f"server contract mismatch: {key}")
            task = self.info.get("tasks", {}).get(TASK)
            if not isinstance(task, dict) or task.get("embodiment") != EMBODIMENT:
                raise ValueError("server has no approved franka/tomato task")
            if task.get("prompt") != PROMPT or task.get("mask") != model_mask().tolist():
                raise ValueError("franka/tomato prompt or mask changed")
            if task.get("source_frame") != "cam0_color_optical_frame" or "chunk_camera" not in self.info.get("state_frames", []):
                raise ValueError("server camera-frame contract mismatch")
            self.reset()
        except BaseException:
            self.close()
            raise

    @property
    def execution_accepted(self):
        frames = self.info.get("coordinate_frames", {})
        return self.info.get("robot_execution_accepted") is True and frames.get("robot_execution_verified") is True

    def _rpc(self, message):
        send_message(self.socket, message)
        result = recv_message(self.socket)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    def reset(self):
        request_id = uuid.uuid4().hex
        result = self._rpc({"cmd": "reset", "request_id": request_id})
        for key, value in {
            "protocol": "egoav78_chunk_v1",
            "protocol_version": 4,
            "request_id": request_id,
            "deployment_id": self.info["deployment_id"],
            "ok": True,
        }.items():
            if result.get(key) != value:
                raise ValueError(f"reset response mismatch: {key}")
        if not isinstance(result.get("session_id"), str) or not result["session_id"]:
            raise ValueError("reset response has no session_id")
        self.session_id = result["session_id"]

    def infer(self, state78, head_rgb, *, sample_ns=None, seed=42):
        state = np.asarray(state78, dtype=np.float32)
        rgb = np.asarray(head_rgb)
        if state.shape != (78,) or not np.isfinite(state).all() or np.any(state[~model_mask()] != 0):
            raise ValueError("expected finite EgoAV78 state with zero inactive slots")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("expected uint8 HWC RGB head image")
        request_id = uuid.uuid4().hex
        stamp = time.monotonic_ns() if sample_ns is None else sample_ns
        request = {
            "cmd": "infer", "request_id": request_id, "session_id": self.session_id,
            "sample_ns": stamp, "task": TASK, "embodiment": EMBODIMENT,
            "state_frame": "chunk_camera", "seed": seed,
            "observation": {"state": state, "head_rgb": np.ascontiguousarray(rgb)},
        }
        result = self._rpc(request)
        expected = {
            "protocol": "egoav78_chunk_v1", "protocol_version": 4,
            "representation": "egoav78_rot6d_columns", "request_id": request_id,
            "session_id": self.session_id, "sample_ns": stamp,
            "deployment_id": self.info["deployment_id"], "task": TASK,
            "embodiment": EMBODIMENT, "action_frame": "chunk_camera",
            "action_mode": "absolute", "action_units": "physical", "dt_s": 0.05,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(f"inference response mismatch: {key}")
        if result.get("action_mask") != model_mask().tolist():
            raise ValueError("inference action mask changed")
        anchor = np.asarray(result.get("anchor_state"), dtype=np.float32)
        if anchor.shape != (78,) or not np.array_equal(anchor, state):
            raise ValueError("response anchor_state does not match this observation")
        actions = np.asarray(result.get("actions"), dtype=np.float32)
        if actions.shape != (32, 78) or not np.isfinite(actions).all():
            raise ValueError("expected finite actions[32,78]")
        if np.any(actions[:, ~model_mask()] != 0):
            raise ValueError("server returned nonzero inactive actions")
        self.last_response = result
        return actions

    def close(self):
        self.socket.close()
