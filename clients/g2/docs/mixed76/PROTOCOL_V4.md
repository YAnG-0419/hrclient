# Mixed76 TCP v4 — frozen server/client contract
Version: 2026-10-18. Coordinator owns shared clients; four model chats own model servers.

## Transport
Reuse pi05_server/_ours_protocol.py byte-for-byte: 4-byte unsigned big-endian payload length + UTF-8 JSON, maximum 16MiB. Arrays use {"__array__":true,"dtype":"<f4" or "|u1","shape":[...],"data":"base64 raw contiguous bytes"}. No pickle. One synchronous outstanding request per connection. TCP over existing SSH tunnel. Old v3/WebSocket endpoints do not pass v4 handshake.

## info
Request {"cmd":"info"}. Response required:
protocol_version=4, model one of fastwam/alpha/joint/tri/coarse_fine/pi05,
representation="head12_wrist18_hand46_rot6d", state_dim=76, action_dim=76,
state_units="physical", action_units="physical", action_mode="absolute",
horizon=32, dt_s=0.05, camera_keys=["head_rgb"], embodiments=["3b","g2"],
checkpoint (nonempty path/id), checkpoint_step (integer), normalization_sha256,
deployment_id (immutable nonempty identity for weights+config),
prompts (task -> exact training instruction), image_preprocess (description),
denoise_steps, inference_compile (actual bool), weights_strict=true.
Also report coordinate_frames and component_units from dataset/normalization provenance; do not guess or transform frames. Client sends physical values in exactly training convention. Piper gripper native interface is mm; verify stored training units before declaring executable. Unverified units/frame identity must be reported and not claimed live-control acceptance.

## reset
Request {"cmd":"reset","request_id":"unique string"}.
Response {"protocol_version":4,"request_id":same,"deployment_id":same,"session_id":"new unique opaque string","ok":true}.
Session belongs to connection. Clear all temporal/model caches. No robot command. infer before reset or after stale session fails.

## infer
Request {"cmd":"infer","request_id":"unique string","session_id":...,"sample_ns":client monotonic integer,"task":"yifu","embodiment":"g2","observation":{"state":float32[76],"head_rgb":uint8[H,W,3]}}.
Task is one of basket/book/diguan/yifu/dengpao. Embodiment explicit, not inferred from task. Optional prompt is not required; server resolves task through info.prompts. Raw RGB, no client crop/resize/normalization. State is unnormalized physical76 with inactive slots exactly zero.
Response echoes request_id/session_id/sample_ns and has protocol_version=4, deployment_id, embodiment, representation, action_units="physical", dt_s=.05, actions=float32[32,76], action_mask=bool[76], inference_seconds finite nonnegative.
Actions finite, inactive slots exactly zero, physical unnormalized absolute targets. Server owns training normalization and all model-specific image/text/denoising work. Do not change sampling settings merely to meet protocol. No server action queue: infer computes one entire fresh chunk from that observation. Never repeat single-step RPC32 times. Client owns execution prefix/replanning.
Errors {"error":{"code":"INVALID_REQUEST|MODEL_ERROR|STALE_SESSION","message":"..."}, "request_id":...}; do not return an action on error. Do not silently fallback or auto-replay commands after reconnect.

## Mixed76 layout (half-open)
0:9 Piper camera xyz+rot6D; 9:12 G2 head3; 12:21 left wrist xyz+rot6D; 21:30 right wrist; 30:52 left fingers22; 52 left gripper; 53:75 right fingers22; 75 right gripper.
rot6D concatenates first two rotation matrix COLUMNS. Native quaternion xyzw. G2 mask65: 9:52 and53:75; Piper mask29:0:9,12:30,52,75.
Use shared mixed76_contract.py for mapping. Server's internal padded dimensions (e.g. Alpha80) never leak onto wire. Model-native masking throughout denoising must remain as trained.

## Acceptance and ownership
Model owners: implement protocol-only adapter, preserve checkpoint/native inference/settings. Add info/reset/infer malformed and stale-session tests. Compare direct predictor and network under controlled RNG; two recorded observations (Piper basket and G2 yifu), real weights, finite32x76, exact mask; report cold/warm times and actual eager/compile. Do NOT drive robots.
Coordinator: shared transport/client and robot adapters. Do not edit real/piper client files from model threads.
Provide updated endpoint/SSH route, deployment_id, checkpoint, normalization hash, test log and server info JSON to your own report. Existing services may remain on old port until replacement tested; do not stop training.
