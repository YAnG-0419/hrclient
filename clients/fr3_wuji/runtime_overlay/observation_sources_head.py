#!/usr/bin/env python3
"""Pi05-compatible read-only observation source that launches only cam0."""

import argparse
import contextlib
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading

import yaml

sys.path.insert(0, "/code")
import observation_sources as base


def ros_and_head(args, sock):
    if args.arm_source != "ros":
        raise ValueError("The hrclient head-only overlay supports the ROS arm source")
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from teleop_core import contract

    rclpy.init()
    node = rclpy.create_node("hrclient_head_only_observations")
    lock, refs = threading.Lock(), []
    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    camera_log = (logs / "cameras.txt").open("w")
    reader_log = (logs / "cam0-reader.txt").open("w")
    camera = reader = None

    def joint_callback(source, message):
        base.send_record(
            sock,
            {
                "source": source,
                "stamp": message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
                "stamp_basis": "ros_joint_header",
                "names": list(message.name),
                "positions": list(message.position),
                "units": "radian",
                "measured": True,
            },
            lock=lock,
        )

    try:
        reader = subprocess.Popen(
            [
                sys.executable,
                "/code/observation_sources.py",
                "camera",
                "--camera",
                "0",
                "--socket",
                args.socket,
            ],
            stdout=reader_log,
            stderr=subprocess.STDOUT,
        )
        for side in ("left", "right"):
            for part, mode, topic in (
                ("arm", args.arm_source, contract.ARM_STATE_TOPIC),
                ("hand", args.hand_source, contract.WUJI_STATE_TOPIC),
            ):
                if mode == "ros":
                    refs.append(
                        node.create_subscription(
                            JointState,
                            topic.format(side=side),
                            lambda message, source=side + "_" + part: joint_callback(source, message),
                            qos_profile_sensor_data,
                        )
                    )
        if args.start_cameras:
            deployed = yaml.safe_load(Path(args.camera_config).read_text())["camera_bringup"]
            entry = deployed["cam0"]
            camera = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "orbbec_camera",
                    "gemini435_le.launch.py",
                    "camera_name:=cam0",
                    "serial_number:=" + str(entry["serial_number"]),
                    "sync_mode:=standalone",
                    "config_file_path:=" + str(entry["config_file"]),
                    "log_level:=" + str(deployed.get("log_level", "none")),
                    "log_file_name:=cam0.log",
                ],
                stdout=camera_log,
                stderr=subprocess.STDOUT,
            )
        while rclpy.ok():
            if reader.poll() is not None:
                raise RuntimeError("cam0 reader exited")
            if camera is not None and camera.poll() is not None:
                raise RuntimeError("cam0 launch exited")
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        base.stop(camera)
        base.stop(reader)
        reader_log.close()
        camera_log.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("hands", "ros", "camera"))
    parser.add_argument("--camera", type=int, choices=(0, 1, 2))
    parser.add_argument("--socket", required=True)
    parser.add_argument("--logs", default="/logs")
    parser.add_argument("--arm-source", choices=("ros", "direct"), default="ros")
    parser.add_argument("--hand-source", choices=("ros", "sdk"), default="ros")
    parser.add_argument("--start-cameras", action="store_true")
    parser.add_argument(
        "--camera-config", default="/workspace/franka_upper_body_teleop/data_collection/config/cameras.yaml"
    )
    parser.add_argument("--left-arm", default="172.16.0.2")
    parser.add_argument("--right-arm", default="172.16.1.2")
    parser.add_argument("--left-hand", default="192.168.1.110:7447")
    parser.add_argument("--right-hand", default="192.168.2.111:7447")
    args = parser.parse_args()
    signal.signal(signal.SIGINT, base.interrupted)
    signal.signal(signal.SIGTERM, base.interrupted)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(args.socket)
        with contextlib.suppress(KeyboardInterrupt):
            {"hands": base.hands, "ros": ros_and_head, "camera": base.camera_images}[args.kind](args, sock)


if __name__ == "__main__":
    main()
