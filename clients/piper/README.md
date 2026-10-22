# Piper mixed76 inference client

This repository preserves the Piper-side client used with the six mixed76 TCP
v4 model servers. It is an overlay for the existing EgoAV Piper runtime, not a
replacement for its camera, CAN, teleop, record, or hardware-driver modules.

## Preserved behavior

- six selectable models: FastWAM, Alpha, Joint67, TriJoint67,
  Coarse-to-fine, and Pi0.5;
- one physical76 input/output contract with explicit Piper mask;
- recorded-observation inference by default, with no motion;
- opt-in continuous robot execution through the existing three-follower
  camera/CAN/IK/replay stack;
- response deadline defaults to 5 seconds;
- Piper IK returns the best finite in-limit solution without the former
  2 mm / 1 degree residual rejection;
- inference tracking ignores Cartesian position drift but retains orientation,
  joint, timing, CAN, and 70 mm gripper tracking checks (live snapshot 2026-10-20).

These settings preserve the currently tested deployment. They do not certify
coordinate frames, hardware units, or closed-loop task quality for a new robot.

## Install into EgoAV

Copy the committed `src/infer` files and the explicitly listed `src/replay`
files into the matching paths of a compatible EgoAV checkout. Do not overwrite
the checkout's `src/record`, `src/teleop`, arm mappings, recording mappings, or
Piper SDK. Review the diff before deploying.

Copy a model config example to the same name without `.example`, then set its
SSH alias, server host, and port. Live endpoints are intentionally untracked.

## Recorded inference, no motion

```bash
python -m src.infer.mixed76_client \
  --model fastwam --task book \
  --remote-config src/infer/mixed76_configs/fastwam_mixed76_v4.json \
  --observation /path/to/book_observation.npz
```

## Continuous robot execution

```bash
python -m src.infer.mixed76_client \
  --model fastwam --task book \
  --remote-config src/infer/mixed76_configs/fastwam_mixed76_v4.json \
  --execute --cycles 100000 --steps 32 --max-response-age 5
```

Before adding `--execute`, confirm current qPOS/EEF/controller state, clear all
three follower workspaces, keep an operator beside the robot, and keep the
physical emergency stop immediately reachable. A software stop request is not
a hardware emergency stop.

## Verification

`src/infer/mixed76_common/test_unified_client.py` covers all six models, both
embodiment masks, TCP identities, malformed responses, and Piper mapping. The
IK tests under `tests/` require the compatible EgoAV project as the import chain
includes trajectory, h5py and teleop modules. Hardware execution tests
are not included; the previous deployment test hard-coded the superseded
20 mm threshold. Hardware acceptance must run in the compatible EgoAV tree.
