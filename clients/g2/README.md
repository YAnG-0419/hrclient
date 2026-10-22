# G2 mixed76 inference client

This repository is a portable snapshot of the shared mixed76 TCP v4 client
used by the G2 inference checkout. It preserves the client, protocol contract,
sanitized six-model configuration examples, and focused tests. Live endpoints,
observations, action traces, credentials, and robot data are intentionally not
tracked.

The files retain their integration paths for an `agibot_inference` checkout.
Copy them into a compatible checkout, review the diff, and create each live
configuration by removing `.example` from the selected file and filling in the
deployment endpoint.

See `docs/mixed76/README.md` for commands and `docs/mixed76/PROTOCOL_V4.md`
for the wire contract.
