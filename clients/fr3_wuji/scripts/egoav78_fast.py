"""EgoAV78 inference adapter for Pi05's existing fast hardware timeline."""

from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import time
import uuid

import numpy as np

from deploy.fr3_wuji_fast.control import worker
from deploy.fr3_wuji_fast.timeline import checked_chunk, prepare_initial
from deploy.fr3_wuji_slow.continuous import StopFlag

from egoav78_codec import decode_actions, encode_observation


@dataclass(frozen=True)
class EgoAdmission:
    observation_time: float
    sent_time: float
    received_time: float
    request_id: str
    checkpoint: str
    epoch: int
    observation_budget: float = 0.14
    response_deadline: float = 2.0

    def check(self):
        values = (self.observation_time, self.sent_time, self.received_time)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Non-finite inference timestamp")
        if not self.request_id or not self.checkpoint:
            raise ValueError("Missing inference provenance")
        if not 0 <= self.sent_time - self.observation_time <= self.observation_budget:
            raise ValueError("Inference observation exceeded doubled send budget")
        if not self.sent_time <= self.received_time <= self.observation_time + self.response_deadline:
            raise ValueError("Inference response exceeded doubled admission deadline")


class EgoAV78FastConsumer:
    def __init__(
        self,
        devices,
        client,
        kinematics,
        codec_limits,
        device_limits,
        output,
        options,
        config_path,
        *,
        max_ik_step=0.6,
        max_initial_arm_delta=0.6,
    ):
        self.devices, self.client = devices, client
        self.kinematics, self.codec_limits = kinematics, codec_limits
        self.device_limits, self.output = device_limits, Path(output)
        self.options, self.config_path = options, Path(config_path)
        self.max_ik_step = max_ik_step
        self.max_initial_arm_delta = max_initial_arm_delta
        self.context = mp.get_context("spawn")
        self.incoming, self.status = self.context.Queue(1), self.context.Queue(128)
        self.stop_flag = StopFlag(self.context)
        self.process = None
        self.closed = False
        self.completed = 0
        self.request = {"number": 1}
        self.report = {
            "state": "waiting_for_observation",
            "source_hz": options["source_hz"],
            "control_hz": 100,
            "hardware_output": True,
            "rtc": False,
            "arm_motion_limits": options["arm_motion_limits"],
            "options": options,
            "deployment_id": client.info["deployment_id"],
            "chunks": [],
        }

    def save_report(self):
        (self.output / "egoav78-fast-report.json").write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False) + "\n"
        )

    def poll(self, *_):
        while True:
            try:
                event = self.status.get_nowait()
            except queue.Empty:
                break
            kind = event["event"]
            if kind == "failed":
                self.report.update(state="failed", error=event["error"], traceback=event.get("traceback"))
                self.save_report()
                raise RuntimeError(event["error"])
            if kind == "infer":
                if self.request is not None:
                    raise RuntimeError("Multiple outstanding inference requests")
                self.request = event
            elif kind == "stage":
                print("【阶段】" + event["text"], flush=True)
            elif kind == "arm_motion_limits":
                if event.get("arm_speed_enforced") is not False:
                    raise RuntimeError("EgoAV78 fast run unexpectedly enabled arm speed projection")
                print("【机械臂运动限制】disabled（保留位置/跟踪/反馈/停止保护）", flush=True)
            elif kind == "control_tick_resynchronized":
                self.report.setdefault("control_timing_events", []).append(event)
            elif kind == "chunk_started":
                self.report["chunks"].append({key: value for key, value in event.items() if key != "executed_nodes"})
                print(
                    f"【阶段】动作块 {event['number']}/{self.options['rounds']}："
                    f"20 Hz，延迟补偿跳过 {event['latency_offset_steps']} 步",
                    flush=True,
                )
            elif kind == "complete":
                self.completed = 1
                self.report.update(state="complete", result=event)
                self.save_report()
        if self.process is not None and not self.process.is_alive() and not self.completed:
            raise RuntimeError("Fast control process exited unexpectedly")

    def __call__(self, store, obs, metadata, directory):
        try:
            self.consume(obs, metadata)
        except BaseException as error:
            self.stop_flag.set()
            detail = f"{type(error).__name__}: {error}"
            self.report.update(state="failed", error=detail)
            print("【执行失败】" + detail, file=sys.stderr, flush=True)
            self.save_report()
            raise

    def consume(self, obs, metadata):
        self.poll()
        if self.completed or self.request is None:
            return
        source_names = ("left_arm", "left_hand", "right_arm", "right_hand", "cam0")
        wall = time.time()
        ages = [wall - metadata["samples"][name]["stamp"] for name in source_names]
        if any(not 0 <= age <= 0.14 for age in ages):
            return
        request = self.request
        number = request["number"]
        state54 = np.asarray(obs["observation/state"], dtype=np.float32)
        state78, rgb = encode_observation(state54, obs["observation/image"], self.kinematics)
        sent = time.monotonic()
        sample_ns = int((sent - max(ages)) * 1e9)
        print(f"【推理】第 {number} 次，使用新观测", flush=True)
        actions78 = self.client.infer(state78, rgb, sample_ns=sample_ns)
        received = time.monotonic()
        chunk_dir = self.output / f"chunk-{number:04d}"
        chunk_dir.mkdir()
        np.savez_compressed(
            chunk_dir / "inference-response.npz",
            state54=state54,
            state78=state78,
            head_rgb=rgb,
            actions78=actions78,
        )
        joints54, diagnostics = decode_actions(
            actions78,
            state54,
            self.kinematics,
            self.codec_limits[0],
            self.codec_limits[1],
            max_step=self.max_ik_step,
            fast_ik=True,
            defer_hand_limits=True,
        )
        admission = EgoAdmission(
            sent - max(ages),
            sent,
            received,
            uuid.uuid4().hex,
            self.client.info["deployment_id"],
            0,
        )
        admission.check()
        np.savez_compressed(
            chunk_dir / "inference.npz",
            state54=state54,
            state78=state78,
            head_rgb=rgb,
            actions78=actions78,
            actions54=joints54,
        )
        (chunk_dir / "admission.json").write_text(
            json.dumps(
                {
                    **asdict(admission),
                    "observation_metadata": metadata,
                    "ik": diagnostics,
                },
                indent=2,
            )
            + "\n"
        )
        chunk = checked_chunk(
            joints54,
            admission,
            number,
            self.device_limits,
            horizon=self.options["horizon"],
        )
        self.request = None
        self.poll()
        if self.process is None:
            feedback = self.devices.feedback(time.monotonic())
            initial = prepare_initial(
                chunk,
                feedback.positions,
                self.device_limits,
                self.config_path,
                time.monotonic(),
                arm_motion_limits="disabled",
                source_hz=self.options["source_hz"],
                max_initial_arm_delta=self.max_initial_arm_delta,
            )
            self.report.update(state="running", stream_digest=initial.digest, approach_seconds=initial.approach.duration)
            self.process = self.context.Process(
                target=worker,
                args=(
                    self.devices.sock,
                    self.devices.counter,
                    self.devices.finish_policy,
                    initial,
                    self.device_limits,
                    self.options,
                    self.incoming,
                    self.status,
                    self.stop_flag,
                    str(self.output),
                ),
            )
            self.process.start()
            self.devices.sock.close()
            self.devices.closed = True
        else:
            self.incoming.put_nowait(chunk)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_flag.set()
        if self.process is not None:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        if self.report["state"] not in {"complete", "failed"}:
            self.report["state"] = "stopped"
        self.save_report()
        for pipe in (self.incoming, self.status):
            pipe.cancel_join_thread()
            pipe.close()
