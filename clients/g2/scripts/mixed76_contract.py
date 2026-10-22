"""Physical mixed76 wire contract; numpy/scipy only, no robot/model imports."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

REPRESENTATION = 'head12_wrist18_hand46_rot6d'


def embodiment_mask(kind: str) -> np.ndarray:
    if kind not in ('3b', 'g2'):
        raise ValueError('Expected embodiment 3b or g2')
    mask = np.zeros(76, bool)
    mask[12:30] = True
    if kind == '3b':
        mask[:9] = True
        mask[[52,75]] = True
    else:
        mask[9:12] = mask[30:52] = mask[53:75] = True
    return mask


def encode_state(raw: np.ndarray, kind: str) -> np.ndarray:
    mask = embodiment_mask(kind)
    x = np.asarray(raw, np.float32)
    if x.shape != ((23 if kind == '3b' else 61),) or not np.isfinite(x).all():
        raise ValueError('Expected one finite native physical state')
    y = np.zeros(76, np.float32)
    pairs = [(0,12),(8 if kind == '3b' else 29,21)]
    if kind == '3b':
        pairs.append((16,0)); y[[52,75]] = x[[7,15]]
    else:
        y[9:12], y[30:52], y[53:75] = x[58:61], x[7:29], x[36:58]
    for src,dst in pairs:
        q = x[src+3:src+7]
        norm = np.linalg.norm(q)
        if norm < 1e-6:
            raise ValueError('Invalid source xyzw quaternion')
        mat = Rotation.from_quat(q / norm).as_matrix()
        y[dst:dst+3] = x[src:src+3]
        y[dst+3:dst+9] = np.r_[mat[:,0],mat[:,1]]
    return np.where(mask,y,0).astype(np.float32)


def validate_actions(values: np.ndarray, kind: str, horizon: int) -> np.ndarray:
    x = np.asarray(values, np.float32)
    if x.shape != (horizon,76) or not np.isfinite(x).all():
        raise ValueError('Expected finite physical [horizon,76] actions')
    if np.any(x[:,~embodiment_mask(kind)] != 0):
        raise ValueError('Inactive mixed76 slots must be zero')
    return x


def decode_actions(values: np.ndarray, kind: str) -> np.ndarray:
    """Conversion only at the robot boundary; wire/client inference stays76D."""
    x = np.asarray(values, np.float32)
    x = validate_actions(x, kind, len(x))
    y = np.zeros((len(x),23 if kind == '3b' else 61),np.float32)
    pairs = [(12,0),(21,8 if kind == '3b' else 29)]
    if kind == '3b':
        pairs.append((0,16)); y[:,[7,15]] = x[:,[52,75]]
    else:
        y[:,58:61], y[:,7:29], y[:,36:58] = x[:,9:12], x[:,30:52], x[:,53:75]
    for src,dst in pairs:
        a,b = x[:,src+3:src+6].astype(np.float64),x[:,src+6:src+9].astype(np.float64)
        n = np.linalg.norm(a,axis=-1,keepdims=True)
        if np.any(n < 1e-8):
            raise ValueError('Degenerate predicted rotation first column')
        a = a/n; b = b-(a*b).sum(-1,keepdims=True)*a
        n = np.linalg.norm(b,axis=-1,keepdims=True)
        if np.any(n < 1e-8):
            raise ValueError('Degenerate predicted rotation second column')
        b = b/n
        q = Rotation.from_matrix(np.stack((a,b,np.cross(a,b)),axis=-1)).as_quat()
        for i in range(1,len(q)):
            if np.dot(q[i-1],q[i]) < 0:
                q[i] *= -1
        y[:,dst:dst+3],y[:,dst+3:dst+7] = x[:,src:src+3],q
    return y
