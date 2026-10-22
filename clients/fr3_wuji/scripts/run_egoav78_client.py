#!/usr/bin/env python3
"""Probe or run no-motion FR3/Wuji inference through EgoAV78 TCP v4."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from egoav78_codec import decode_actions, encode_observation
from inference_connection import egoav78_connection
from model_transport_v4 import EgoAV78Client

DEFAULT_PI05 = Path("/home/user/lpy/Pi05")
DEFAULT_REPLAY = Path("/home/user/lpy/fr3_wuji_eef_replay")
DEFAULT_CALIBRATION = Path("/home/user/tomato_eef/conversion/calibration.yaml")


def load_local_contract(replay_root, calibration_path):
    replay = replay_root.expanduser().resolve()
    if not (replay / "core.py").is_file():
        raise ValueError(f"FR3/Wuji EEF replay core not found: {replay}")
    sys.path.insert(0, str(replay))
    from core import Kinematics, read_limits

    calibration = yaml.safe_load(calibration_path.expanduser().read_text())
    camera = calibration.get("camera", {})
    if camera.get("serial_number") != "CP4N5630008Z" or camera.get("dataset_optical_frame") != "cam0_color_optical_frame":
        raise ValueError("calibration is not the approved cam0/link8 asset")
    return Kinematics(calibration), read_limits(replay / "limits.yaml"), calibration


def plan_one(client, state54, rgb, kinematics, limits, *, sample_ns=None, max_ik_step=0.3):
    state78, head_rgb = encode_observation(state54, rgb, kinematics)
    actions78 = client.infer(state78, head_rgb, sample_ns=sample_ns)
    joints54, diagnostics = decode_actions(
        actions78, state54, kinematics, limits[0], limits[1], max_step=max_ik_step
    )
    return state78, actions78, joints54, diagnostics


def recorded(args, client, kinematics, limits):
    with np.load(args.recorded_observation, allow_pickle=False) as data:
        if "state" in data.files:
            state = data["state"]
        elif "observation/state" in data.files:
            state = data["observation/state"]
        else:
            raise ValueError("recorded NPZ needs state or observation/state")
        if "head_rgb" in data.files:
            rgb = data["head_rgb"]
        elif "observation/image" in data.files:
            rgb = data["observation/image"]
        else:
            raise ValueError("recorded NPZ needs head_rgb or observation/image")
    state78, actions78, joints54, diagnostics = plan_one(
        client, state, rgb, kinematics, limits, max_ik_step=args.max_ik_step
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, state54=state, state78=state78, head_rgb=rgb,
                        actions78=actions78, planned_joints54=joints54)
    print(json.dumps({
        "mode": "recorded_no_motion", "hardware_output": False,
        "output": str(args.output), "server_execution_accepted": client.execution_accepted,
        "diagnostics": diagnostics,
    }, indent=2, ensure_ascii=False))


class LivePlanner:
    def __init__(self, client, kinematics, limits, max_ik_step):
        self.client, self.kinematics, self.limits = client, kinematics, limits
        self.max_ik_step = max_ik_step
        self.completed = 0

    def poll(self, *_):
        return None

    def __call__(self, store, obs, metadata, output):
        wall, mono = time.time(), time.monotonic()
        age = wall - metadata["target_stamp"]
        if not 0 <= age <= store.max_age:
            return
        sample_ns = int((mono - age) * 1e9)
        state = obs["observation/state"]
        rgb = obs["observation/image"]
        state78, head_rgb = encode_observation(state, rgb, self.kinematics)
        actions78 = self.client.infer(state78, head_rgb, sample_ns=sample_ns)
        self.completed = 1
        np.savez_compressed(
            Path(output) / "egoav78-response.npz",
            state54=state,
            state78=state78,
            head_rgb=head_rgb,
            actions78=actions78,
        )
        joints54, diagnostics = decode_actions(
            actions78, state, self.kinematics, self.limits[0], self.limits[1],
            max_step=self.max_ik_step,
        )
        np.savez_compressed(Path(output) / "egoav78-plan.npz", state54=state, state78=state78,
                            head_rgb=rgb, actions78=actions78, planned_joints54=joints54)
        (Path(output) / "egoav78-plan.json").write_text(json.dumps({
            "mode": "live_read_only_inference_and_ik", "hardware_output": False,
            "server_execution_accepted": self.client.execution_accepted,
            "observation_metadata": metadata, "diagnostics": diagnostics,
        }, indent=2, ensure_ascii=False) + "\n")


def live(args, client, kinematics, limits):
    pi05 = args.pi05_root.expanduser().resolve()
    observation_root = pi05 / "deploy/fr3_wuji"
    if not (observation_root / "observe.py").is_file():
        raise ValueError(f"Pi05 observation stack not found: {observation_root}")
    sys.path.insert(0, str(observation_root))
    import observe

    options = [
        "--duration", "30", "--poll-interval", "0.005", "--output", str(args.output),
        "--arm-source", args.arm_source, "--hand-source", args.hand_source,
        "--wuji-left-address", args.wuji_left_address,
        "--wuji-right-address", args.wuji_right_address,
    ]
    if args.start_cameras:
        options += ["--start-cameras", "--camera-profile", "rgb"]
    observe.main(
        options,
        consumer=LivePlanner(client, kinematics, limits, args.max_ik_step),
        single_shot=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--probe-server", action="store_true")
    modes.add_argument("--recorded-observation", type=Path)
    modes.add_argument("--live-observation", action="store_true")
    parser.add_argument("--remote-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pi05-root", type=Path, default=DEFAULT_PI05)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--arm-source", choices=("ros", "direct"), default="ros")
    parser.add_argument("--hand-source", choices=("ros", "sdk"), default="ros")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:50001")
    parser.add_argument("--max-ik-step", type=float, default=0.3)
    parser.add_argument("--start-cameras", action="store_true")
    args = parser.parse_args(argv)
    if not args.probe_server and args.output is None:
        parser.error("--output is required for recorded or live observation")
    if not np.isfinite(args.max_ik_step) or args.max_ik_step <= 0:
        parser.error("--max-ik-step must be positive and finite")
    try:
        kinematics, limits, calibration = load_local_contract(args.replay_root, args.calibration)
        with egoav78_connection(args.remote_config) as (address, config):
            client = EgoAV78Client(*address, deployment_id=config["deployment_id"])
            try:
                if args.probe_server:
                    print(json.dumps({
                        "server_compatible": True,
                        "model_name": config["model_name"],
                        "deployment_id": client.info["deployment_id"],
                        "protocol": client.info["protocol"],
                        "shape": [client.info["horizon"], client.info["action_dim"]],
                        "dt_s": client.info["dt_s"],
                        "server_execution_accepted": client.execution_accepted,
                        "calibration_id": calibration["calibration_id"],
                    }, indent=2, ensure_ascii=False))
                elif args.recorded_observation:
                    recorded(args, client, kinematics, limits)
                else:
                    live(args, client, kinematics, limits)
            finally:
                client.close()
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
