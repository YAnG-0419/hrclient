"""Inject the hrclient-only hand runtime overlay into Pi05's device container."""

from pathlib import Path


OVERLAY = Path(__file__).resolve().parents[1] / "runtime_overlay"
CONTAINER_OVERLAY = "/hrclient_runtime_overlay"


def apply_device_overlay(commands):
    command = commands["devices"]
    sitecustomize = OVERLAY / "sitecustomize.py"
    if not sitecustomize.is_file():
        raise FileNotFoundError(sitecustomize)
    image = command.index("franka-upper-body-teleop:latest")
    command[image:image] = ["-v", f"{OVERLAY}:{CONTAINER_OVERLAY}:ro"]
    for index, value in enumerate(command[:image]):
        if value.startswith("PYTHONPATH="):
            command[index] = value.replace(
                "PYTHONPATH=", f"PYTHONPATH={CONTAINER_OVERLAY}:", 1
            )
            break
    else:
        raise ValueError("Device command has no explicit PYTHONPATH")
    command[image:image] = ["-e", "HRCLIENT_EGOAV78_FAST_OVERLAY=1"]
    return commands
