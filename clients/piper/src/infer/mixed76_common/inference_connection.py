"""Select the existing TCP client's endpoint; remote mode owns one SSH tunnel.

No model loading, hardware access, deadlines, retries or automatic fallback.
SSH -f returns after authentication and local forwarding setup succeeds.
That does not assert that the remote model server is ready: the existing
OursClient handshake and its existing timeouts still handle that separately.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import warnings

DEFAULT_REMOTE_CONFIG = Path(__file__).resolve().parents[1] / 'configs/inference_remote.json'


class SSHTunnel:
    def __init__(self, ssh_host, server_host, server_port):
        for label, value in (('ssh_host', ssh_host), ('server_host', server_host)):
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', value):
                raise ValueError(f'{label} must be an SSH alias, IPv4 address or DNS hostname')
        if type(server_port) is not int or not 1 <= server_port <= 65535:
            raise ValueError('请先在远端配置中填写 server_port（服务端模型的实际 TCP 监听端口）')
        self.ssh_host, self.server_host, self.server_port = ssh_host, server_host, server_port
        self._directory = None

    def __enter__(self):
        self._directory = Path(tempfile.mkdtemp(prefix='agibot-ssh-', dir='/tmp'))
        self.control = self._directory / 'control'
        try:
            # Ask the OS for an unused loopback port. If another process takes
            # it before ssh binds, ExitOnForwardFailure reports the error.
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                self.local_port = reservation.getsockname()[1]
            forward = f'127.0.0.1:{self.local_port}:{self.server_host}:{self.server_port}'
            command = ['ssh', '-M', '-S', str(self.control), '-f', '-N', '-T',
                       '-o', 'ControlMaster=yes', '-o', 'ControlPersist=no',
                       '-o', 'ExitOnForwardFailure=yes', '-o', 'BatchMode=yes',
                       '-o', 'StrictHostKeyChecking=yes', '-L', forward, self.ssh_host]
            error_file = self._directory / 'ssh.log'
            # A file (not PIPE) prevents the background master from retaining
            # a pipe and making subprocess.run wait for the tunnel to exit.
            with error_file.open('wb') as errors:
                result = subprocess.run(command, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=errors)
            if result.returncode:
                detail = error_file.read_text(errors='replace').strip()
                raise RuntimeError(f'SSH 转发建立失败（退出码 {result.returncode}）：{detail}')
            print(f'远端推理连接：{self.ssh_host} → {self.server_host}:{self.server_port} '
                  f'（本机端口 {self.local_port}，整个 client 会话复用）', flush=True)
            return '127.0.0.1', self.local_port
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        if self._directory is None:
            return
        if self.control.exists():
            try:
                result = subprocess.run(['ssh', '-S', str(self.control), '-O', 'exit', self.ssh_host],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                if result.returncode:
                    warnings.warn(f'SSH 转发关闭失败，请检查控制连接 {self.control}', RuntimeWarning)
                    return
            except OSError as error:
                warnings.warn(f'SSH 转发关闭失败：{error}；控制连接 {self.control}', RuntimeWarning)
                return
        shutil.rmtree(self._directory)
        self._directory = None


@contextmanager
def inference_connection(args):
    if args.inference == 'local':
        if args.remote_config is not None:
            raise ValueError('--remote-config 需要同时指定 --inference remote')
        yield args.ip, args.port
        return
    if args.inference != 'remote':
        raise ValueError(f'Unknown inference mode: {args.inference}')
    path = Path(args.remote_config or DEFAULT_REMOTE_CONFIG).expanduser()
    config = json.loads(path.read_text())
    with SSHTunnel(config['ssh_host'], config['server_host'], config['server_port']) as address:
        yield address
