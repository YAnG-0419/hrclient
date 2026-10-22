"""Run Pi05 observation with an hrclient-local cam0-only process overlay."""

from contextlib import contextmanager
from pathlib import Path
import subprocess as host_subprocess


HEAD_KEYS = {"cam0": "observation/image"}
HEAD_SOURCE = Path(__file__).resolve().parents[1] / "runtime_overlay" / "observation_sources_head.py"


class _SubprocessProxy:
    def __getattr__(self, name):
        return getattr(host_subprocess, name)

    def Popen(self, command, *args, **kwargs):
        command = rewrite_observation_command(command)
        return host_subprocess.Popen(command, *args, **kwargs)


def rewrite_observation_command(command):
    command = list(command)
    source = "/code/observation_sources.py"
    if source in command and "ros" in command:
        image = command.index("franka-upper-body-teleop:latest")
        command[image:image] = ["-v", f"{HEAD_SOURCE}:/hrclient/observation_sources_head.py:ro"]
        command[command.index(source)] = "/hrclient/observation_sources_head.py"
    return command


@contextmanager
def _head_only(observe):
    import observation

    full_observe_keys = observe.IMAGE_KEYS
    full_observation_keys = observation.IMAGE_KEYS
    full_sources = observation.SOURCES
    original_preflight = observe.preflight
    original_subprocess = observe.subprocess

    def preflight(args):
        observe.IMAGE_KEYS = full_observe_keys
        try:
            cameras = original_preflight(args)
        finally:
            observe.IMAGE_KEYS = HEAD_KEYS
        return {"cam0": cameras["cam0"], "log_level": cameras.get("log_level", "none")}

    observe.IMAGE_KEYS = HEAD_KEYS
    observe.preflight = preflight
    observe.subprocess = _SubprocessProxy()
    observation.IMAGE_KEYS = HEAD_KEYS
    observation.SOURCES = (*observation.STATE_ORDER, *HEAD_KEYS)
    try:
        yield
    finally:
        observe.IMAGE_KEYS = full_observe_keys
        observe.preflight = original_preflight
        observe.subprocess = original_subprocess
        observation.IMAGE_KEYS = full_observation_keys
        observation.SOURCES = full_sources


def run(observe, argv, *, consumer, single_shot):
    if not HEAD_SOURCE.is_file():
        raise FileNotFoundError(HEAD_SOURCE)
    with _head_only(observe):
        return observe.main(argv, consumer=consumer, single_shot=single_shot)
