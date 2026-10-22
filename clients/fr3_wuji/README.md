# FR3 + Wuji EgoAV78 client overlay

This client targets the deployed `egoav78_chunk_v1` TCP v4 services on `b300`.
It reuses the verified FR3 camera calibration and link8 IK from
`/home/user/lpy/fr3_wuji_eef_replay`; it does not use the native 54D OpenPI
WebSocket service.

## Exact interface

- Request: physical `state[78]` plus only the head `head_rgb` image.
- Response: absolute physical `actions[32,78]`, 20 Hz, in the request's
  chunk-start cam0 optical frame.
- Robot task: `franka/tomato`; embodiment: `franka`.
- Fixed cam0 occupies model slots `[18:27]`. The current deployed model requires
  these slots to be valid and set to the identity pose. The hardware adapter
  always discards their predicted motion.
- Body `[27:32]`, grippers `54/77`, and hand-local slots `3/17` are inactive.
- Each Wuji hand's 20 measured joints map to the 22 semantic slots
  `[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21]`.

The current server `info` declares `robot_execution_accepted=false` and
`coordinate_frames.robot_execution_verified=false`. The read-only entry point
surfaces both fields. The separate fast hardware entry point requires explicit
`--execute --supervised-trial`; it was added only after the device owner
confirmed these fields describe the server's lack of robot I/O rather than a
client-side prohibition.

## Environment and local assets

Reuse the Pi05 Python environment:

```bash
PYTHON=/home/user/lpy/Pi05/.venv/bin/python
"$PYTHON" clients/fr3_wuji/scripts/doctor.py
```

Required local assets:

- `/home/user/tomato_eef/conversion/calibration.yaml`: cam0 serial/IP,
  left/right `base_to_camera`, `fr3_link8` definition and FR3 kinematics.
- `/home/user/lpy/fr3_wuji_eef_replay/core.py`: checked FK, IK and limits logic.
- `/home/user/lpy/fr3_wuji_eef_replay/limits.yaml`: offline 54D bounds.
- `/home/user/lpy/Pi05/deploy/fr3_wuji`: synchronized dual-arm/dual-hand/camera
  observation stack.

The calibration is valid only if cam0 is serial `CP4N5630008Z`, its mount has
not moved, and neither FR3 base has moved since calibration.

## Server configs

Three checked-in configs bind model name, SSH target, port and approved
deployment ID. They are already usable with the local `b300` SSH alias:

```text
joint_relative -> 18868
joint          -> 18869
tri            -> 18870
```

Probe one service without inference or hardware access:

```bash
"$PYTHON" clients/fr3_wuji/scripts/run_egoav78_client.py \
  --probe-server --remote-config clients/fr3_wuji/configs/joint_remote.json
```

## Recorded-observation dry run

The NPZ contains `state` (native 54D measured joints) and either `head_rgb` or
`observation/image` (uint8 HWC RGB). This opens one tunnel, runs one inference,
converts both arms back through cam0→base→link8 IK, maps both hands, and saves
the original 78D response plus planned 54D joint targets. Nothing is sent to
the robot.

```bash
"$PYTHON" clients/fr3_wuji/scripts/run_egoav78_client.py \
  --recorded-observation /path/to/observation.npz \
  --output /path/to/result.npz \
  --remote-config clients/fr3_wuji/configs/joint_remote.json
```

## Live read-only inference

This starts the existing Pi05 observation producers and performs one inference
plus offline IK plan. It never starts the command bridge or enables motion:

```bash
"$PYTHON" clients/fr3_wuji/scripts/run_egoav78_client.py \
  --live-observation --start-cameras \
  --arm-source direct --hand-source sdk \
  --wuji-left-address 192.168.1.110:7447 \
  --wuji-right-address 192.168.2.111:50001 \
  --output /home/user/lpy/Pi05/logs/fr3_wuji/egoav78-live-check \
  --remote-config clients/fr3_wuji/configs/joint_remote.json
```

If an existing camera container already publishes cam0, omit
`--start-cameras`. The server consumes only cam0 even though the current Pi05
collector also verifies the wrist-camera streams.

The live runner saves `egoav78-response.npz` as soon as a validated server
response arrives, before IK admission. This keeps the exact state/image/action
triplet available when a joint-limit or per-step safety check rejects the plan.

## Fast supervised hardware rollout

`run_egoav78_fast.py` reuses Pi05's existing fast device bridge: 20 Hz EgoAV78
targets are converted with the calibrated FR3 IK and linearly sampled at 100 Hz.
The tested profile executes complete chunks in serial mode, without RTC and
with `arm_motion_limits=disabled`. `--rounds` accepts 1 through 40 chunks; each
chunk contains 1.6 seconds of model time, so 40 chunks contain 64 seconds.
Joint position
limits, 20 ms command lifetime, feedback freshness, 0.2 rad arm tracking,
ownership locking and stop handling remain enabled. Remote inference pauses are
held at the previous endpoint rather than hidden or counted as model time.
The fast entrypoint mounts an hrclient-local runtime overlay into only its own
containers: it launches/subscribes only cam0, consumes only the newest Wuji
state/diagnostic frame from each SDK burst, and records healthy raw hand trace
at 10 Hz with frame serialization deferred until shutdown. Pi05 source files
and other Pi05 launch paths are not modified by this overlay.

```bash
"$PYTHON" clients/fr3_wuji/scripts/run_egoav78_fast.py \
  --execute --supervised-trial --rounds 40 --start-cameras \
  --wuji-left-address 192.168.1.110:7447 \
  --wuji-right-address 192.168.2.111:50001 \
  --output /home/user/lpy/Pi05/logs/weight_motion_eval/egoav78-fast-trial \
  --remote-config clients/fr3_wuji/configs/joint_remote.json
```

The per-run IK and initial acquisition admissions are explicitly doubled from
0.3 to 0.6 rad. The fast geometric-Jacobian IK retains the same FK residual
tolerances (`1e-5 m`, `1e-4 rad`) and every decoded target is rechecked against
the installed device joint limits before entering the controller. Out-of-range
Wuji hand predictions are projected to the installed hand limits by the slider
path; arm predictions remain reject-on-limit. Each raw server response is saved
before IK/limit admission so a rejected plan remains diagnosable.
