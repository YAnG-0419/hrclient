"""TCP v4 wire format shared with the deployed EgoAV78 services."""

import base64
import json
import math
import socket

import numpy as np

MAX_BYTES = 16 * 1024 * 1024


def _encode(value):
    if isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)
        return {"__array__": True, "dtype": value.dtype.str, "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported type: {type(value)}")


def _decode(value):
    if value.get("__array__") is not True:
        return value
    dtype, shape = np.dtype(value["dtype"]), value["shape"]
    if dtype not in (np.dtype("uint8"), np.dtype("float32"), np.dtype("float64")):
        raise ValueError("Unsupported array dtype")
    if not isinstance(shape, list) or len(shape) > 4 or any(
        type(size) is not int or size < 0 or size > MAX_BYTES for size in shape
    ):
        raise ValueError("Invalid array shape")
    expected = math.prod(shape) * dtype.itemsize
    if expected > MAX_BYTES:
        raise ValueError("Array too large")
    raw = base64.b64decode(value["data"], validate=True)
    if len(raw) != expected:
        raise ValueError("Array byte count mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        part = sock.recv(size - len(data))
        if not part:
            raise ConnectionError("Peer disconnected")
        data.extend(part)
    return bytes(data)


def send_message(sock: socket.socket, message: dict) -> None:
    payload = json.dumps(message, default=_encode, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(payload) <= MAX_BYTES:
        raise ValueError("Message too large")
    sock.sendall(len(payload).to_bytes(4, "big") + payload)


def recv_message(sock: socket.socket) -> dict:
    size = int.from_bytes(recv_exact(sock, 4), "big")
    if not 0 < size <= MAX_BYTES:
        raise ValueError("Invalid message length")
    value = json.loads(recv_exact(sock, size), object_hook=_decode)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value
