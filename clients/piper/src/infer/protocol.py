"""Bounded JSON + raw RGB base64 on loopback; never pickle."""
import base64
import json
import numpy as np
from .model.api import ROLES, validate_observation

MAX_BODY = 8 * 1024 * 1024
MAX_IMAGE = 2 * 1024 * 1024


def encode_json(value):
    def convert(item):
        if isinstance(item, np.ndarray): return item.tolist()
        if isinstance(item, np.generic): return item.item()
        raise TypeError(type(item).__name__)
    data=json.dumps(value,allow_nan=False,default=convert).encode()
    if len(data)>MAX_BODY: raise ValueError('message exceeds 8 MiB')
    return data


def decode_json(data):
    if len(data)>MAX_BODY: raise ValueError('message exceeds 8 MiB')
    def invalid(value): raise ValueError(f'invalid JSON number: {value}')
    value=json.loads(data,parse_constant=invalid)
    if not isinstance(value,dict): raise ValueError('JSON object required')
    return value


def pack_observation(observation):
    result=validate_observation(observation)
    encoded={}
    for role,image in result['images'].items():
        if image.nbytes>MAX_IMAGE: raise ValueError('image too large')
        encoded[role]={'shape':list(image.shape),'rgb':base64.b64encode(image.tobytes()).decode('ascii')}
    result['images']=encoded
    return result


def unpack_observation(value):
    images={}
    for role in ROLES:
        item=value['images'][role]; shape=item['shape']
        if (not isinstance(shape,list) or len(shape)!=3 or shape[-1]!=3
                or any(type(x) is not int or x<1 for x in shape)
                or shape[0]*shape[1]*3>MAX_IMAGE): raise ValueError('invalid RGB shape')
        if not isinstance(item['rgb'],str) or len(item['rgb'])>MAX_IMAGE*2: raise ValueError('invalid RGB data')
        raw=base64.b64decode(item['rgb'],validate=True)
        if len(raw)!=shape[0]*shape[1]*3: raise ValueError('RGB byte count mismatch')
        images[role]=np.frombuffer(raw,np.uint8).reshape(shape).copy()
    return validate_observation({**value,'images':images})
