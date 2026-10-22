"""Run with python -m src.infer.mixed76_client. Default: recorded inference only."""
import argparse
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / 'mixed76_common'))
from client_mixed76 import Mixed76Client
from mixed76_contract import encode_state
from piper_mixed76 import encode_observation, execution_actions
from inference_connection import inference_connection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['fastwam','alpha','joint','tri','coarse_fine','pi05'])
    p.add_argument('--task',required=True,choices=['basket','book','diguan','yifu','dengpao'])
    p.add_argument('--inference',choices=['local','remote'],default='remote')
    p.add_argument('--remote-config',type=Path)
    p.add_argument('--ip',default='127.0.0.1');p.add_argument('--port',type=int,default=8857)
    p.add_argument('--observation',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--steps',type=int,default=32);p.add_argument(
        '--max-response-age',type=float,default=5.,
        help='seconds, including inference and IK (default: 5)')
    args=p.parse_args()
    if args.cycles<1 or not 1<=args.steps<=32 or not np.isfinite(args.max_response_age) or args.max_response_age<=0:
        p.error('Invalid cycles, steps or response age')
    if args.execute and args.observation:p.error('Recorded observations cannot execute')
    if not args.execute and not args.observation:p.error('Provide --observation for no-motion inference')
    if args.inference=='remote' and args.remote_config is None:
        args.remote_config=Path(__file__).resolve().parent/'mixed76_configs'/f'{args.model}_mixed76_v4.json'
    with inference_connection(args) as address:
        client=Mixed76Client(*address,model=args.model)
        hub=robot=None
        try:
            print(json.dumps(client.info,ensure_ascii=False))
            if not args.execute:
                with np.load(args.observation,allow_pickle=False) as data:
                    state=data['state'];rgb=data['head_rgb']
                    if state.shape!=(76,):state=encode_state(state,'3b')
                actions=client.infer(state,rgb,task=args.task,embodiment='3b')
                if args.output:np.save(args.output,actions)
                print(json.dumps({'shape':list(actions.shape),'robot_io':False}));return
            from .client import sample_observation, guarded_work
            from .robot import RobotExecutor
            from ..replay.replay import runtime_locks
            from ..record.record_sources import SourceHub
            root=Path(__file__).resolve().parents[2]
            with ExitStack() as stack:
                stack.enter_context(runtime_locks())
                arms=json.loads((root/'src/teleop/arm_mapping.json').read_text())
                mapping=json.loads((root/'src/record/recording_mapping.json').read_text())
                hub=SourceHub(mapping,arms)
                stack.callback(hub.close);hub.wait_ready()
                robot=RobotExecutor(root,arms)
                stack.callback(robot.close);robot.start()
                for _ in range(args.cycles):
                    sample=hub.snapshot();stamp=sample['sample_monotonic_ns']
                    state,rgb=encode_observation(sample_observation(sample,client.info['prompts'][args.task]))
                    actions=guarded_work(lambda:client.infer(state,rgb,task=args.task,embodiment='3b',sample_ns=stamp),
                                         args.max_response_age,robot.monitor)
                    plan=guarded_work(lambda:robot.plan(execution_actions(actions),args.steps),
                                      args.max_response_age,robot.monitor)
                    if not 0<=(time.monotonic_ns()-stamp)/1e9<=args.max_response_age:
                        raise TimeoutError('Observation expired before execution')
                    robot.execute(plan)
        finally:client.close()


if __name__=='__main__':main()
