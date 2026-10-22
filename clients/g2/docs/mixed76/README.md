# Mixed76 shared client

`scripts/run_mixed76_client.py` is the G2 entry point for the six TCP v4
servers: FastWAM, OpenWAM Alpha, Joint67, TriJoint67, Coarse-to-fine, and
Pi0.5. The default path is recorded-observation inference and never imports or
commands robot hardware.

## Configure an endpoint

Copy the matching file from `configs/mixed76_examples/` to
`configs/<model>_mixed76_v4.json`, then replace the SSH alias, server host, and
port with the current deployment values. Runtime endpoint files are local
deployment state and are intentionally not committed.

## Recorded observation test

```bash
python scripts/run_mixed76_client.py \
  --model joint --embodiment g2 --task yifu \
  --remote-config configs/joint_mixed76_v4.json \
  --observation /path/to/observation.npz \
  --output /tmp/actions.npy
```

The NPZ must contain `state` and RGB `head_rgb`. The client verifies protocol
version, model identity, checkpoint identity fields, physical76 shape, mask,
session, request ID, sample timestamp, finite actions, 32-step horizon, and
20 Hz timing metadata.

## Robot execution

Robot motion is opt-in through `--execute`. Before using it, provide the G2
Pico root, robot SSH alias, hand configuration, and confirm current robot
state, workspace clearance, controller health, and physical emergency-stop
availability. The execution path retains the installed reset and feedback
lifecycle; this repository does not weaken those gates.

See `PROTOCOL_V4.md` for the server contract. Tests of the transport and
mapping layers do not establish closed-loop task success.
