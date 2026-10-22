from pathlib import Path
import sys
import unittest
import queue

import numpy as np

PI05 = Path("/home/user/lpy/Pi05")
sys.path.insert(0, str(PI05))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from deploy.fr3_wuji_fast.timeline import Timeline, checked_chunk, prepare_initial
from deploy.fr3_wuji_fast.control import run_control
from deploy.fr3_wuji_fast.test_fast import Clock, Devices
from egoav78_fast import EgoAdmission
from head_observation import rewrite_observation_command
from runtime_overlay import apply_device_overlay


LIMITS = {
    "lower": np.full(54, -3.0),
    "upper": np.full(54, 3.0),
    "speed": np.full(54, 3.0),
}
CONFIG = PI05 / "experiments/weight_motion_eval/config.yaml"


def chunk(number=1, value=0.0, sent=10.0, received=10.1):
    actions = np.full((32, 54), value, dtype=np.float64)
    admission = EgoAdmission(sent - 0.1, sent, received, f"request-{number}", "deployment", 0)
    return checked_chunk(actions, admission, number, LIMITS, horizon=32)


class Fast20HzTests(unittest.TestCase):
    def test_hrclient_overlays_are_scoped_to_generated_commands(self):
        devices = [
            "docker", "run", "-e", "PYTHONPATH=/home/user/lpy/Pi05:/sdk",
            "franka-upper-body-teleop:latest", "python3", "-m",
            "deploy.fr3_wuji_runtime.bridge",
        ]
        commands = {"devices": devices}
        apply_device_overlay(commands)
        self.assertIn("HRCLIENT_EGOAV78_FAST_OVERLAY=1", devices)
        self.assertIn("/hrclient_runtime_overlay:ro", " ".join(devices))
        self.assertIn("PYTHONPATH=/hrclient_runtime_overlay:/home/user/lpy/Pi05:/sdk", devices)

        observation = rewrite_observation_command(
            [
                "docker", "run", "franka-upper-body-teleop:latest", "python3",
                "/code/observation_sources.py", "ros", "--socket", "/feed/feed.sock",
            ]
        )
        self.assertIn("/hrclient/observation_sources_head.py", observation)
        self.assertNotIn("/code/observation_sources.py", observation)

    def test_exact_32_node_20hz_duration(self):
        first = chunk()
        timeline = Timeline(first, rounds=1, source_hz=20, horizon=32)
        timeline.start(10.0)
        self.assertFalse(timeline.exhausted(11.59))
        self.assertTrue(timeline.exhausted(11.6))

    def test_rtg_accepts_second_32_node_chunk(self):
        timeline = Timeline(chunk(), rounds=2, source_hz=20, horizon=32, admission_deadline=2.0)
        timeline.start(10.0)
        request = timeline.request(10.8)
        self.assertEqual(request["number"], 2)
        event = timeline.install(chunk(2, 0.02, sent=10.81, received=10.95), 11.0, LIMITS)
        self.assertEqual(event["number"], 2)
        self.assertEqual(event["latency_offset_steps"], 4)

    def test_doubled_initial_admission_is_explicit(self):
        first = chunk(value=0.5)
        with self.assertRaisesRegex(ValueError, "Initial arm displacement"):
            prepare_initial(first, np.zeros(54), LIMITS, CONFIG, 10.2, source_hz=20)
        prepared = prepare_initial(
            first,
            np.zeros(54),
            LIMITS,
            CONFIG,
            10.2,
            source_hz=20,
            max_initial_arm_delta=0.6,
        )
        self.assertEqual(prepared.chunk.actions.shape, (32, 54))

    def test_four_full_serial_chunks_use_20hz_timeline(self):
        clock, incoming, events = Clock(), queue.Queue(), []
        devices = Devices(clock)
        first = prepare_initial(
            chunk(), np.zeros(54), LIMITS, CONFIG, clock(), source_hz=20,
            max_initial_arm_delta=0.6,
        )
        pending = None

        def notify(event):
            nonlocal pending
            events.append(event)
            if event["event"] == "infer":
                pending = (
                    clock() + 0.12,
                    chunk(event["number"], value=event["number"] / 100, sent=clock() + 0.01, received=clock() + 0.11),
                )

        def sleep(duration):
            nonlocal pending
            clock.sleep(duration)
            if pending is not None and clock() >= pending[0]:
                incoming.put_nowait(pending[1])
                pending = None

        run_control(
            devices,
            first,
            LIMITS,
            {
                "rounds": 4,
                "mode": "serial",
                "source_hz": 20,
                "horizon": 32,
                "admission_deadline": 2.0,
            },
            incoming,
            notify,
            lambda *args: None,
            lambda: False,
            clock=clock,
            sleep=sleep,
        )
        starts = [event for event in events if event["event"] == "chunk_started"]
        self.assertEqual([event["number"] for event in starts], [1, 2, 3, 4])
        self.assertGreaterEqual(starts[-1]["at"] - starts[0]["at"], 3 * 32 / 20)
        self.assertEqual(devices.prepares, 1)
        self.assertEqual(devices.finishes, 1)


if __name__ == "__main__":
    unittest.main()
