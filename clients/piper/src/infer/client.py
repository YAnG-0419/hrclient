"""Acquire observation, call server, optionally execute through IK -> MIT."""
import argparse
import logging
import math
from pathlib import Path
import queue
import signal
import threading
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import uuid

import numpy as np
from .tasks import get_task, check_task_health
from .model.api import ROLES, validate_actions
from .protocol import MAX_BODY, decode_json, encode_json, pack_observation


def rpc(url, endpoint, value, timeout=30):
    parsed=urlparse(url)
    if (parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.username
            or parsed.path not in ('','/') or parsed.query or parsed.fragment):
        raise ValueError('only http://127.0.0.1:<port> is supported')
    request=Request(url.rstrip('/')+'/'+endpoint, data=None if value is None else encode_json(value),
                    headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=timeout) as response:
        return decode_json(response.read(MAX_BODY+1))


def check_response(response, request, *, max_age, execute):
    for key in ('request_id','session_id','sample_ns'):
        if response.get(key)!=request[key]: raise ValueError(f'response {key} mismatch')
    if request.get('deployment_id') and response.get('deployment_id') != request['deployment_id']:
        raise ValueError('response model deployment mismatch')
    age=(time.monotonic_ns()-request['sample_ns'])/1e9
    if not 0<=age<=max_age: raise ValueError(f'observation/action expired: {age:.3f}s')
    if execute and response.get('test_mode') is not False: raise ValueError('mock/unidentified model cannot control robot')
    return validate_actions(response['actions'],max_steps=256,dt_s=.05)


def guarded_work(function, timeout, monitor=lambda:None):
    """Keep control monitoring responsive while network/IK work runs elsewhere."""
    result=queue.Queue(maxsize=1)
    def worker():
        try: result.put((True,function()))
        except BaseException as exc: result.put((False,exc))
    threading.Thread(target=worker,daemon=True).start()
    deadline=time.monotonic()+timeout
    while True:
        monitor()
        if time.monotonic()>deadline: raise TimeoutError('inference/IK deadline exceeded; result discarded')
        try: ok,value=result.get(timeout=.01)
        except queue.Empty: continue
        if not ok: raise value
        return value


def synthetic_observation(prompt):
    return {'images':{r:np.zeros((480,640,3),np.uint8) for r in ROLES},
            'eef':{r:[.3,0,.4,0,0,0,1] for r in ROLES},
            'gripper_mm':{'left':30.,'right':30.},'prompt':prompt}


def sample_observation(sample,prompt):
    import cv2
    images={}
    for role in ROLES:
        bgr=cv2.imdecode(np.frombuffer(sample['images'][role]['jpeg'],np.uint8),cv2.IMREAD_COLOR)
        if bgr is None: raise ValueError(f'{role}: JPEG decode failed')
        images[role]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return {'images':images,'eef':{r:sample['observations'][r]['eef'] for r in ROLES},
            'gripper_mm':{r:sample['observations'][r]['gripper_width_mm'] for r in ('left','right')},
            'prompt':prompt}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',default='http://127.0.0.1:8768')
    parser.add_argument('--prompt',help='override task prompt')
    parser.add_argument('--model',required=True,help='model family: pi05 or fastwam (not implemented)')
    parser.add_argument('--task',required=True,help='registered task; selects prompt and requires matching Server task')
    parser.add_argument('--checkpoint',help='optional expected Server checkpoint ID, bundle, or deployment JSON')
    parser.add_argument('--synthetic',action='store_true',help='no cameras/CAN; test input only')
    execution=parser.add_mutually_exclusive_group()
    execution.add_argument('--execute',dest='execute',action='store_true',help='execute actions (default)')
    execution.add_argument('--dry-run',dest='execute',action='store_false',help='predict only; no motion commands')
    parser.set_defaults(execute=True)
    parser.add_argument('--cycles',type=int,default=1,help='number of inference cycles; 0 runs until Ctrl+C')
    parser.add_argument('--steps',type=int,default=50,help='maximum actions executed from each predicted chunk (default: 50)')
    parser.add_argument('--max-response-age',type=float,default=2.,help='seconds, including inference and IK')
    args=parser.parse_args(argv)
    task=get_task(args.task,args.model)
    if args.prompt is None: args.prompt=task['prompt']
    if not args.prompt: parser.error('provide --task TASK or --prompt')
    if args.synthetic and args.execute: parser.error('--synthetic requires --dry-run')
    if args.cycles<0 or not 1<=args.steps<=256: parser.error('invalid cycles/steps')
    if not math.isfinite(args.max_response_age) or args.max_response_age<=0: parser.error('invalid max-response-age')
    return args


def main():
    args=parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    def cancel(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,cancel)
    health=rpc(args.server,'health',None)
    if health.get('ready') is not True: raise RuntimeError('server not ready')
    check_task_health(health,args.task,args.checkpoint,model=args.model)
    deployment = health.get('deployment')
    logging.info('SERVER MODEL %s',deployment or {'task':health.get('task')})
    logging.info('MODE: %s','EXECUTE: home then IK->MIT' if args.execute else 'DRY RUN: prediction only')
    if args.execute and health.get('test_mode') is not False: raise ValueError('mock server cannot execute')
    if health.get('dt_s')!=.05: raise ValueError('this client requires 20Hz action data')
    reset=rpc(args.server,'reset',{})
    if reset.get('deployment') != deployment: raise ValueError('Server model changed during reset')
    session=reset['session_id']
    hub=robot=None
    from contextlib import ExitStack
    with ExitStack() as stack:
        try:
            if not args.synthetic:
                import json
                from ..replay.replay import runtime_locks
                from ..record.record_sources import SourceHub
                root=Path(__file__).resolve().parents[2]
                stack.enter_context(runtime_locks())
                arms=json.loads((root/'src/teleop/arm_mapping.json').read_text())
                mapping=json.loads((root/'src/record/recording_mapping.json').read_text())
                hub=SourceHub(mapping,arms); hub.wait_ready()
                if args.execute:
                    from .robot import RobotExecutor
                    robot=RobotExecutor(root,arms); robot.start()
            cycle=0
            while args.cycles==0 or cycle<args.cycles:
                if hub:
                    sample=hub.snapshot(); obs=sample_observation(sample,args.prompt)
                    stamp=sample['sample_monotonic_ns']
                else: obs=synthetic_observation(args.prompt); stamp=time.monotonic_ns()
                request={'session_id':session,'request_id':uuid.uuid4().hex,'sample_ns':stamp,
                         'observation':pack_observation(obs)}
                if deployment: request['deployment_id']=deployment['deployment_id']
                monitor=robot.monitor if robot else (lambda:None)
                response=guarded_work(lambda:rpc(args.server,'predict',request),args.max_response_age,monitor)
                actions=check_response(response,request,max_age=args.max_response_age,execute=args.execute)
                if robot:
                    plan=guarded_work(lambda:robot.plan(actions,args.steps),args.max_response_age,monitor)
                    check_response(response,request,max_age=args.max_response_age,execute=True)
                    robot.execute(plan)
                cycle+=1
                logging.info('cycle=%d received=%d executed=%d infer=%.3fs %s',cycle,
                    len(actions['eef']['left']),len(plan) if robot else 0,response['inference_s'],
                    'IK->MIT' if robot else 'READ ONLY: no motion commands')
        finally:
            signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN)
            try:
                if robot: robot.close()
            finally:
                if hub: hub.close()
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: logging.warning('cancelled; confirm physical robot state'); raise SystemExit(130)
    except Exception as exc: logging.error('%s',exc); raise SystemExit(1)
