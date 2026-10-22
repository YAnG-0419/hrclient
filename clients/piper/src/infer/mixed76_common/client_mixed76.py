"""Model-independent synchronous mixed76 TCP v4 client. No robot imports."""
import socket
import time
import uuid
import numpy as np
from _ours_protocol import send_message, recv_message
from mixed76_contract import REPRESENTATION, embodiment_mask, validate_actions


class Mixed76Client:
    def __init__(self, host, port, *, model, timeout=300, checkpoint=None):
        self.socket = socket.create_connection((host, port), timeout=timeout)
        self.session_id = None
        try:
            self.info = self._rpc({'cmd': 'info'})
            expected = dict(protocol_version=4, model=model, representation=REPRESENTATION,
                            action_dim=76, state_dim=76, action_units='physical',
                            state_units='physical', action_mode='absolute', horizon=32,
                            dt_s=.05, camera_keys=['head_rgb'], weights_strict=True)
            for key, value in expected.items():
                if self.info.get(key) != value:
                    raise ValueError(f'Server contract mismatch: {key}')
            for key in ('checkpoint', 'normalization_sha256', 'deployment_id'):
                if not isinstance(self.info.get(key), str) or not self.info[key]:
                    raise ValueError(f'Missing server identity: {key}')
            if checkpoint is not None and self.info['checkpoint'] != checkpoint:
                raise ValueError('Checkpoint mismatch')
            self.reset()
        except BaseException:
            self.close()
            raise

    def _rpc(self, message):
        send_message(self.socket, message)
        result = recv_message(self.socket)
        if 'error' in result:
            raise RuntimeError(str(result['error']))
        return result

    def close(self):
        self.socket.close()

    def reset(self):
        request_id = uuid.uuid4().hex
        result = self._rpc({'cmd': 'reset', 'request_id': request_id})
        for key, value in dict(protocol_version=4, request_id=request_id,
                               deployment_id=self.info['deployment_id'], ok=True).items():
            if result.get(key) != value:
                raise ValueError(f'Reset mismatch: {key}')
        if not isinstance(result.get('session_id'), str) or not result['session_id']:
            raise ValueError('Missing session identity')
        self.session_id = result['session_id']

    def infer(self, state76, head_rgb, *, task, embodiment, sample_ns=None):
        state = np.asarray(state76, np.float32)
        mask = embodiment_mask(embodiment)
        rgb = np.asarray(head_rgb)
        if state.shape != (76,) or not np.isfinite(state).all() or np.any(state[~mask] != 0):
            raise ValueError('Expected finite physical76 state with zero inactive slots')
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError('Expected uint8 HWC RGB')
        if task not in self.info.get('prompts', {}) or embodiment not in self.info.get('embodiments', []):
            raise ValueError('Unsupported task or embodiment')
        request_id = uuid.uuid4().hex
        stamp = time.monotonic_ns() if sample_ns is None else sample_ns
        result = self._rpc(dict(cmd='infer', request_id=request_id, session_id=self.session_id,
                                sample_ns=stamp, task=task, embodiment=embodiment,
                                observation=dict(state=state, head_rgb=rgb)))
        expected = dict(protocol_version=4, request_id=request_id, session_id=self.session_id,
                        sample_ns=stamp, deployment_id=self.info['deployment_id'],
                        embodiment=embodiment, representation=REPRESENTATION,
                        action_units='physical', dt_s=.05)
        for key, value in expected.items():
            if result.get(key) != value:
                raise ValueError(f'Inference response mismatch: {key}')
        if result.get('action_mask') != mask.tolist():
            raise ValueError('Response mask mismatch')
        seconds = result.get('inference_seconds')
        if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError('Invalid inference timing')
        actions = validate_actions(result['actions'], embodiment, self.info['horizon'])
        self.last_response = result
        return actions
