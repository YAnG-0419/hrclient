"""Reuse Piper H IK and replay MIT I/O; never drive leaders."""
import json
import logging
from pathlib import Path
import time
import uuid
import numpy as np

from ..teleop.piper_io import discover_interfaces
from ..teleop.teleop import Teleop
from ..replay.inverse_kinematics import solve_pose
from ..replay.replay import ReplayController, cleanup_signals
from ..replay.replay_io import ReplayIO, pose_error
from ..replay.trajectory import Episode, ArmTrajectory, PoseTarget, ROLES
from .model.api import validate_actions
from .protocol import encode_json


def build_plan(actions, seeds, limits, fk, steps):
    """Pure IK planning. Compute/validate the whole execution prefix before sending."""
    actions=validate_actions(actions,max_steps=256,dt_s=.05)
    if type(steps) is not int or steps<1: raise ValueError('invalid execution steps')
    count=min(steps,len(actions['eef']['left']))
    previous={r:np.array(seeds[r][:6],dtype=float) for r in ROLES}
    plan=[]
    for index in range(count):
        targets={}
        for role in ROLES:
            eef=actions['eef'][role][index]
            grip=None if role=='active_camera' else float(actions['gripper_mm'][role][index])
            if grip is not None:
                lo,hi=limits[role][-1][:2]
                if not lo<=grip<=hi: raise ValueError(f'{role}: gripper action outside hardware limits')
            solved=solve_pose(fk,eef[:3],eef[3:],previous[role],np.asarray(limits[role])[:6,:2])
            targets[role]=PoseTarget(eef[:3].copy(),eef[3:].copy(),grip,
                                     previous[role].copy(),solved.copy())
            previous[role]=solved
        plan.append(targets)
    return plan


class _Trace:
    def __init__(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        self.file=path.open('x',buffering=1)

    def write(self,value): self.file.write(encode_json(value).decode()+'\n')


class RobotExecutor:
    def __init__(self,root,arm_mapping):
        self.root,self.mapping=Path(root),arm_mapping
        self.io=self.controller=self.home=self.trace=None
        self.states=self.cart=None
        self.last_targets=None

    def start(self):
        profile=json.loads((self.root/'src/replay/kinematics.json').read_text())
        if profile.get('dh_is_offset')!=1 or profile.get('robot_model')!='piper_h':
            raise ValueError('expected verified Piper H DH=1 profile')
        arms=[a for a in self.mapping['arms'] if a['id'] in ROLES.values()]
        self.io=ReplayIO(discover_interfaces(arms),{a['id']:a.get('has_gripper',True) for a in arms})
        self.trace=_Trace(self.root/'data_vis/infer'/(uuid.uuid4().hex+'.jsonl'))
        self.home=Teleop(self.io,self.mapping,arm_ids=list(ROLES.values()),home_joint_speed=30.)
        episode=Episode(np.array([0.,.05]),{}, {'sdk_path':str(self.root/'third_party/piper_sdk/piper_sdk/kinematics/piper_fk.py'),
            'robot_model':'piper_h'},'live-controller-feedback')
        self.controller=ReplayController(self.io,self.home,episode,self.trace)
        self.controller.wait_stationary()
        self.io.connect_commands()
        self.io.limits=self.io.load_limits()
        states,cart=self.controller.wait_stationary()
        for name,state in states.items():
            xyz,q=self.controller.fk.pose(state.pose[:6])
            pos,angle=pose_error(PoseTarget(xyz,q,None,np.asarray(state.pose[:6])),cart[name])
            if pos>5 or angle>2: raise RuntimeError(f'{name}: controller/FK mismatch {pos:.2f}mm {angle:.2f}deg')
        logging.warning('MOVING THREE FOLLOWERS: home 30 deg/s, stable 1s. Clear area; physical E-stop ready.')
        self.home.home(before_motion=self.controller.wait_stationary)
        self.io.limits=self.home.limits
        states,cart=self.controller.wait_stationary()
        for role,name in ROLES.items():
            state=states[name]; pose=cart[name]
            episode.targets[role]=ArmTrajectory(np.tile(state.pose[:6],(2,1)),
                np.tile(pose.xyz_m,(2,1)),np.tile(pose.quaternion_xyzw,(2,1)),
                None if role=='active_camera' else np.full(2,state.pose[6]))
        self.controller.handoff()
        self.monitor()
        # Monitor drift while waiting for inference, including before the first action.
        self.last_targets={r:PoseTarget(self.cart[n].xyz_m.copy(),self.cart[n].quaternion_xyzw.copy(),
            None if r=='active_camera' else self.states[n].pose[6],
            np.asarray(self.states[n].pose[:6]),np.asarray(self.states[n].pose[:6])) for r,n in ROLES.items()}

    def monitor(self):
        self.io.poll(.002)
        states,cart=self.controller.read_feedback(require_mode=True)
        now=time.monotonic()
        for role,name in ROLES.items():
            state=states[name]; bounds=np.asarray(self.io.limits[name])[:6,:2]
            q=np.asarray(state.pose[:6])
            if np.any(q<bounds[:,0]-1) or np.any(q>bounds[:,1]+1): raise RuntimeError(f'{name}: measured joint limit')
            if self.last_targets:
                target=self.last_targets[role]; pos,angle=pose_error(target,cart[name])
                joint=float(np.max(np.abs(q-target.command_joints_deg)))
                grip=0 if target.gripper_mm is None else abs(target.gripper_mm-state.pose[6])
                if self.controller.guard.update(name,now,pos,angle,joint,grip,check_position=False):
                    raise RuntimeError(f'{name}: tracking exceeded {pos:.1f}mm/{angle:.1f}deg')
        self.states,self.cart=states,cart

    def plan(self,actions,steps):
        # Snapshots are replaced, never mutated by the main monitoring thread.
        states=self.states
        seeds={r:tuple(states[n].pose) for r,n in ROLES.items()}
        limits={r:self.io.limits[n] for r,n in ROLES.items()}
        return build_plan(actions,seeds,limits,self.controller.fk,steps)

    def execute(self,plan):
        self.monitor()
        if not plan: raise ValueError('empty execution plan')
        begin=time.monotonic(); previous=begin
        for index,targets in enumerate(plan):
            deadline=begin+index*.05
            while time.monotonic()<deadline:
                self.monitor(); time.sleep(min(.002,max(0,deadline-time.monotonic())))
            now=time.monotonic()
            if now-deadline>=.05 or now-previous>.1: raise RuntimeError('20Hz deadline missed; no catch-up burst')
            self.monitor()
            try:
                for role,name in ROLES.items(): self.io.send_cartesian(name,targets[role])
            except BaseException:
                self.controller.stop('partial inference action send failed')
                raise
            self.last_targets=targets; previous=now
            if time.monotonic()-now>.1: raise RuntimeError('send cycle exceeded 100ms')
            self.trace.write({'stage':'INFER_EXECUTE','index':index,'monotonic_s':now,
                'targets':{r:{'xyz_m':t.xyz_m,'quaternion_xyzw':t.quaternion_xyzw,
                    'command_joints_deg':t.command_joints_deg,'gripper_mm':t.gripper_mm} for r,t in targets.items()},
                'actual':{r:{'joints_deg':self.states[n].pose[:6],
                    'xyz_m':self.cart[n].xyz_m,'quaternion_xyzw':self.cart[n].quaternion_xyzw} for r,n in ROLES.items()}})
        # Let the final sent action occupy its 50ms slot; next observation is then fresh.
        while time.monotonic()<begin+len(plan)*.05:
            self.monitor(); time.sleep(.002)

    def close(self):
        with cleanup_signals():
            try:
                if self.controller: self.controller.stop('inference ended/cancelled/failed')
            finally:
                try:
                    if self.io: self.io.close()
                finally:
                    if self.trace: self.trace.file.close()
