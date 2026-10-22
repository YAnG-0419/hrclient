"""Process-local EgoAV78 hand telemetry overlay; activated only by hrclient."""

import json
import os
from pathlib import Path
import time


def _install():
    from experiments.weight_motion_eval.oneshot import hand as module

    original_trace = module.HandTrace

    class DeferredDecimatedTrace(original_trace):
        """Keep lightweight SDK objects on motion; serialize selected frames at exit."""

        def __init__(self, capacity=4000):
            super().__init__(capacity=capacity)
            self._seen = {"state": 0, "diagnostics": 0}

        def frame(self, frame, *, diagnostic=False):
            kind = "diagnostics" if diagnostic else "state"
            self._seen[kind] += 1
            if not self.triggered and self._seen[kind] % 10:
                self.coalesce(**{kind: 1})
                return
            target = self.after if self.triggered else self.before
            if len(target) == target.maxlen:
                if self.triggered:
                    self.dropped_after += 1
                else:
                    self.dropped_before += 1
            target.append(
                {
                    "event": "_deferred_frame",
                    "received_mono": time.monotonic(),
                    "received_wall": time.time(),
                    "frame": frame,
                    "diagnostic": diagnostic,
                }
            )

        @staticmethod
        def _materialize(event):
            if event.get("event") != "_deferred_frame":
                return event
            frame, diagnostic = event["frame"], event["diagnostic"]
            joints = []
            for joint in frame.joints:
                item = {"nid": int(joint.nid)}
                if diagnostic:
                    status = joint.status_word
                    item.update(error_code=int(joint.error_code_current), ext_state=int(status.ext_state))
                    for field in ("position_limit_active", "velocity_limit_active", "current_limit_active"):
                        item[field] = bool(getattr(status, field, False))
                    for field in ("current", "vbus_v_fb", "mcu_temp_c_fb"):
                        value = getattr(joint, field, None)
                        item[field] = None if value is None else float(value)
                else:
                    item.update(position=float(joint.position), velocity=float(joint.velocity))
                    item["effort"] = None if not hasattr(joint, "effort") else float(joint.effort)
                joints.append(item)
            return {
                "event": "diagnostics" if diagnostic else "state",
                "received_mono": event["received_mono"],
                "received_wall": event["received_wall"],
                "device_timestamp_us": int(frame.header.timestamp_us),
                "device_sequence": int(frame.header.seq) if hasattr(frame.header, "seq") else None,
                "num_joints": int(frame.num_joints),
                "joints": joints,
            }

        def write(self, path, identity):
            path = Path(path)
            with path.open("x") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "metadata",
                            "schema_version": 1,
                            "identity": identity,
                            "dropped_before": self.dropped_before,
                            "dropped_after": self.dropped_after,
                            "coalesced_before": self.coalesced_before,
                            "coalesced_after": self.coalesced_after,
                            "capture": "hrclient: healthy hand frames decimated 10:1 and serialized after stop",
                        }
                    )
                    + "\n"
                )
                for event in (*self.before, *self.after):
                    stream.write(json.dumps(self._materialize(event)) + "\n")

    def latest_only_poll(self, now_mono, now_wall, timing):
        started = time.monotonic()
        newest_state = newest_diagnostic_frame = None
        state_count = diagnostic_count = 0
        try:
            for _ in range(256):
                timing.mark("state_recv_ms")
                frame = self.state_sub.recv()
                timing.mark("other_ms")
                if frame is None:
                    break
                state_count += 1
                timing.counts["state_frames"] += 1
                newest_state = frame
            for _ in range(64):
                timing.mark("diagnostic_recv_ms")
                frame = self.diagnostic_sub.recv()
                timing.mark("other_ms")
                if frame is None:
                    break
                diagnostic_count += 1
                timing.counts["diagnostic_frames"] += 1
                newest_diagnostic_frame = frame
            if newest_diagnostic_frame is not None:
                timing.mark("diagnostic_check_ms")
                by_id = {int(joint.nid): joint for joint in newest_diagnostic_frame.joints}
                if len(newest_diagnostic_frame.joints) != 20 or by_id.keys() != module.NID_SET:
                    raise ValueError("Missing or duplicate hand diagnostic joint")
                if not hasattr(self, "fault_status"):
                    self.fault_status = module.HandFaults(
                        getattr(self, "sdk", None), getattr(self, "side", "unknown")
                    )
                self.fault_status.check(newest_diagnostic_frame.joints, started)
                diagnostic_stamp = int(newest_diagnostic_frame.header.timestamp_us) / 1e6
            timing.mark("decode_ms")
            elapsed = time.monotonic() - started
            now_mono, now_wall = now_mono + elapsed, now_wall + elapsed
            if newest_state is not None:
                self.latest = module.decode_state(newest_state, now_wall, now_mono)
                self.latest_received = now_mono
            if newest_diagnostic_frame is not None:
                age = now_wall - diagnostic_stamp
                module.check_feedback_age(age, "hand diagnostics")
                self.diagnostics = (by_id, now_mono - max(0.0, age))
            if self.latest is None or now_mono - self.latest[2] > 0.15:
                raise module.HandFeedbackUnavailable("No fresh measured hand state")
            if self.diagnostics is None or now_mono - self.diagnostics[1] > 0.15:
                raise module.HandFeedbackUnavailable("No fresh hand diagnostics")
            if self.hand_control == "slider" and self.owns_enable and not self.faulted:
                self.contact_grasp.check_time(now_mono)
        except BaseException:
            if self.trace is not None:
                timing.mark("trace_ms")
                if newest_state is not None:
                    self.trace.frame(newest_state)
                if newest_diagnostic_frame is not None:
                    self.trace.frame(newest_diagnostic_frame, diagnostic=True)
                timing.mark("other_ms")
            raise
        if self.trace is not None:
            timing.mark("trace_ms")
            if newest_state is not None:
                self.trace.frame(newest_state)
            if newest_diagnostic_frame is not None:
                self.trace.frame(newest_diagnostic_frame, diagnostic=True)
            self.trace.coalesce(
                states=max(0, state_count - 1),
                diagnostics=max(0, diagnostic_count - 1),
            )
            timing.mark("other_ms")
        return self.latest

    module.HandTrace = DeferredDecimatedTrace
    module.HandOwner._poll = latest_only_poll


if os.environ.get("HRCLIENT_EGOAV78_FAST_OVERLAY") == "1":
    _install()
