"""Validated SSH forwarding for one EgoAV78 TCP v4 endpoint."""

from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import warnings

EXPECTED_PROTOCOL = "egoav78_chunk_v1"
HOST_PATTERN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def _host(label, value):
    if not isinstance(value, str) or HOST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be an SSH alias, IPv4 address or DNS hostname")
    return value


def load_remote_config(path):
    path = Path(path).expanduser()
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid remote config: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("Remote config must be a JSON object")
    if config.get("status") != "configured":
        raise ValueError("Copy the endpoint template, fill it, and set status to configured")
    if config.get("protocol") != EXPECTED_PROTOCOL:
        raise ValueError("Remote endpoint is not EgoAV78 TCP v4")
    model = config.get("model_name")
    if model not in {"joint_relative", "joint", "tri"}:
        raise ValueError("model_name must be joint_relative, joint or tri")
    deployment_id = config.get("deployment_id")
    if not isinstance(deployment_id, str) or not re.fullmatch(r"[0-9a-f]{64}", deployment_id):
        raise ValueError("deployment_id must be the approved 64-character SHA256 identity")
    port = config.get("server_port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("server_port must be an integer in 1..65535")
    return {
        "ssh_host": _host("ssh_host", config.get("ssh_host")),
        "server_host": _host("server_host", config.get("server_host")),
        "server_port": port,
        "model_name": model,
        "deployment_id": deployment_id,
    }


class SSHTunnel:
    def __init__(self, ssh_host, server_host, server_port):
        self.ssh_host = _host("ssh_host", ssh_host)
        self.server_host = _host("server_host", server_host)
        if type(server_port) is not int or not 1 <= server_port <= 65535:
            raise ValueError("server_port must be an integer in 1..65535")
        self.server_port = server_port
        self._directory = None

    def __enter__(self):
        if shutil.which("ssh") is None:
            raise RuntimeError("ssh executable not found")
        self._directory = Path(tempfile.mkdtemp(prefix="fr3-wuji-egoav78-ssh-", dir="/tmp"))
        self.control = self._directory / "control"
        try:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                self.local_port = reservation.getsockname()[1]
            forward = f"127.0.0.1:{self.local_port}:{self.server_host}:{self.server_port}"
            command = [
                "ssh", "-M", "-S", str(self.control), "-f", "-N", "-T",
                "-o", "ControlMaster=yes", "-o", "ControlPersist=no",
                "-o", "ExitOnForwardFailure=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-L", forward, self.ssh_host,
            ]
            error_file = self._directory / "ssh.log"
            with error_file.open("wb") as errors:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=errors,
                    check=False,
                )
            if result.returncode:
                detail = error_file.read_text(errors="replace").strip()
                raise RuntimeError(f"SSH tunnel failed ({result.returncode}): {detail}")
            print(
                f"EgoAV78 tunnel: {self.ssh_host} -> {self.server_host}:{self.server_port} "
                f"(local 127.0.0.1:{self.local_port})",
                flush=True,
            )
            return "127.0.0.1", self.local_port
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, traceback):
        if self._directory is None:
            return
        if self.control.exists():
            try:
                result = subprocess.run(
                    ["ssh", "-S", str(self.control), "-O", "exit", self.ssh_host],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode:
                    warnings.warn(f"Could not close SSH control connection {self.control}", RuntimeWarning)
                    return
            except OSError as error:
                warnings.warn(f"Could not close SSH tunnel: {error}", RuntimeWarning)
                return
        shutil.rmtree(self._directory)
        self._directory = None


@contextmanager
def egoav78_connection(config_path):
    config = load_remote_config(config_path)
    tunnel_args = {key: config[key] for key in ("ssh_host", "server_host", "server_port")}
    with SSHTunnel(**tunnel_args) as address:
        yield address, config
