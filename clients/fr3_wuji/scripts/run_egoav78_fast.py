#!/usr/bin/env python3
"""Execute up to forty 20 Hz EgoAV78 chunks through Pi05's 100 Hz controller."""

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import time
import uuid

import yaml

DEFAULT_PI05 = Path("/home/user/lpy/Pi05")
sys.path.insert(0, str(DEFAULT_PI05))

from egoav78_fast import EgoAV78FastConsumer
from head_observation import run as run_head_observation
from inference_connection import egoav78_connection
from model_transport_v4 import EgoAV78Client
from run_egoav78_client import load_local_contract
from runtime_overlay import apply_device_overlay

DEFAULT_REFERENCE = Path("/home/user/lpy/gello-retarget")
DEFAULT_SDK = Path("/home/user/miniconda3/envs/gello-upper-body-teleop/lib/python3.10/site-packages")
DEFAULT_CALIBRATION = Path("/home/user/tomato_eef/conversion/calibration.yaml")
DEFAULT_REPLAY = Path("/home/user/lpy/fr3_wuji_eef_replay")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--supervised-trial", action="store_true", required=True)
    parser.add_argument("--remote-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument("--left-arm-ip", default="172.16.0.2")
    parser.add_argument("--right-arm-ip", default="172.16.1.2")
    parser.add_argument("--wuji-left-address", default="192.168.1.110:7447")
    parser.add_argument("--wuji-right-address", default="192.168.2.111:50001")
    parser.add_argument("--pi05-root", type=Path, default=DEFAULT_PI05)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--sdk-site", type=Path, default=DEFAULT_SDK)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--replay-root", type=Path, default=DEFAULT_REPLAY)
    args = parser.parse_args(argv)
    if not 1 <= args.rounds <= 40:
        parser.error("--rounds must be between 1 and 40 chunks")

    pi05 = args.pi05_root.resolve()
    sys.path.insert(0, str(pi05))
    from deploy.fr3_wuji_fast import deploy as fast_deploy
    from deploy.fr3_wuji_fast.ipc import RemoteDevices
    from deploy.fr3_wuji_runtime import hardware
    from experiments.weight_motion_eval.cli import safe_output
    from experiments.weight_motion_eval.reference import reference_limits

    args.reference, args.sdk_site = args.reference.resolve(), args.sdk_site.resolve()
    args.finish_policy, args.hand_control, args.continuous = "disable", "slider", False
    args.arm_motion_limits = "disabled"
    args.qualification = None
    output = safe_output(args.output)
    output.mkdir(parents=True)
    kinematics, codec_limits, calibration = load_local_contract(args.replay_root, args.calibration)
    device_limits, names = reference_limits()
    manifest = hardware.verify_overlay()
    workcell = yaml.safe_load((args.reference / "config/workcell/current.yaml").read_text())
    workcell["LEFT"]["robot_ip"], workcell["RIGHT"]["robot_ip"] = args.left_arm_ip, args.right_arm_ip
    for side in ("LEFT", "RIGHT"):
        if workcell[side]["use_fake_hardware"] != "false":
            raise ValueError("Expected the real FR3 workcell profile")
    (output / "workcell.yaml").write_text(yaml.safe_dump(workcell))
    runtime = {
        "schema_version": 1,
        "reference": str(args.reference),
        "hands": {"left": args.wuji_left_address, "right": args.wuji_right_address},
        "arms": {"left": args.left_arm_ip, "right": args.right_arm_ip},
        "limits": {key: device_limits[key].tolist() for key in ("lower", "upper")},
        "reference_hashes": device_limits["sources"],
        "joint_names": names,
        "controller_library": manifest["built_library"],
        "controller_sha256": manifest["built_library_sha256"],
        "checkpoint": {"protocol": "egoav78_chunk_v1"},
        "finish_policy": args.finish_policy,
        "hand_control": args.hand_control,
        "continuous": False,
        "arm_speed_enforced": False,
        "arm_speed_rad_s": None,
        "calibration_id": calibration["calibration_id"],
        "observation_cameras": ["cam0"],
        "hand_telemetry_overlay": "latest_only_deferred_trace_v1",
    }
    (output / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    options = {
        "rounds": args.rounds,
        "mode": "serial",
        "guidance_steps": 3,
        "trigger_fraction": 0.5,
        "rtc_initial_delay_steps": 0,
        "rtc_delay_buffer_size": 1,
        "rtc_prefix_attention_schedule": "zeros",
        "rtc_max_guidance_weight": 1.0,
        "arm_motion_limits": "disabled",
        "source_hz": 20,
        "horizon": 32,
        "admission_deadline": 2.0,
    }
    (output / "egoav78-fast-config.json").write_text(
        json.dumps(
            {
                "hardware_output": True,
                "nominal_policy_seconds": args.rounds * options["horizon"] / options["source_hz"],
                "rtc": False,
                "max_ik_step_rad": 0.6,
                "max_initial_arm_delta_rad": 0.6,
                **options,
            },
            indent=2,
        )
        + "\n"
    )
    hardware.REFERENCE, hardware.SDK = args.reference, args.sdk_site
    hardware.preflight_owners(execute=True)

    with egoav78_connection(args.remote_config) as (address, remote):
        client = EgoAV78Client(*address, deployment_id=remote["deployment_id"])
        with ExitStack() as stack:
            stack.callback(client.close)
            lock = stack.enter_context((pi05 / ".deployment/oneshot-owner.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="egoav78-fast-")))
            children = hardware.Children(output)
            stack.callback(children.close)
            commands = fast_deploy.launch_commands(args, output, ipc)
            apply_device_overlay(commands)
            (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            print("【阶段】启动 FR3 网关与 Wuji 设备接口", flush=True)
            for label, command in commands.items():
                children.launch(label, command)
            deadline = time.monotonic() + 45
            while not (ipc / "devices.sock").exists():
                children.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError("Device bridge startup timed out")
                time.sleep(0.05)
            devices = RemoteDevices(
                ipc / "devices.sock",
                execute=True,
                finish_policy="disable",
                arm_speed_enforced=False,
                arm_speed_rad_s=None,
            )
            stack.callback(devices.close)
            inventory = devices.call("inventory", timeout=3)
            (output / "devices-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            feedback = None
            while time.monotonic() < deadline:
                children.check()
                try:
                    feedback = devices.feedback(time.monotonic())
                    break
                except RuntimeError:
                    time.sleep(0.02)
            if feedback is None:
                raise RuntimeError("No fresh dual-arm/dual-hand feedback")
            consumer = EgoAV78FastConsumer(
                devices,
                client,
                kinematics,
                codec_limits,
                device_limits,
                output,
                options,
                pi05 / "experiments/weight_motion_eval/config.yaml",
            )
            try:
                observation_root = pi05 / "deploy/fr3_wuji"
                sys.path.insert(0, str(observation_root))
                import observe

                observation_timeout = max(
                    150,
                    int(
                        args.rounds
                        * (
                            options["horizon"] / options["source_hz"]
                            + options["admission_deadline"]
                        )
                        + 60
                    ),
                )
                observe_args = [
                    "--arm-source", "ros",
                    "--hand-source", "ros",
                    "--duration", str(observation_timeout),
                    "--poll-interval", "0.005",
                    "--output", str(output / "observation"),
                ]
                if args.start_cameras:
                    observe_args += ["--start-cameras", "--camera-profile", "rgb"]
                run_head_observation(observe, observe_args, consumer=consumer, single_shot=True)
                if consumer.completed != 1:
                    raise RuntimeError("EgoAV78 fast stream did not complete")
                print(
                    f"【完成】{args.rounds} 个动作块已执行，设备已停止并释放",
                    flush=True,
                )
            finally:
                consumer.close()
                if not devices.closed:
                    devices.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
