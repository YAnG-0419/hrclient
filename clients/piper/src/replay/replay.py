"""Three-follower Cartesian replay. Running without --check MOVES THE ROBOT."""
import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np

from ..teleop.piper_io import discover_interfaces
from ..teleop.teleop import Teleop
from .replay_io import ReplayIO, pose_error
from .inverse_kinematics import prepare_ik
from .trajectory import ForwardKinematics, PoseTarget, ROLES, file_hash, frame_episode
from .lerobot_trajectory import load_lerobot_episode, resolve_dataset

LOG = logging.getLogger('replay')
ROOT = Path(__file__).resolve().parents[2]
HZ = 20.0
GRIPPER_CONTINUITY_MM = 5.0
POSITION_TRACKING_MM = 70.0
GRIPPER_TRACKING_MM = 70.0
GRIPPER_ARRIVAL_MM = 5.0
POSITION_ARRIVAL_MM = 12.0


@contextmanager
def cleanup_signals():
    """Repeated Ctrl+C must not skip another arm's stop request."""
    previous={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGINT,signal.SIGTERM)}
    try: yield
    finally:
        for sig,handler in previous.items(): signal.signal(sig,handler)


@contextmanager
def runtime_locks(directory=Path('/tmp')):
    with ExitStack() as stack:
        for name in ('teleop','record'):
            stream = stack.enter_context((directory/f'egoav-{name}-{os.getuid()}.lock').open('a'))
            try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'{name}正在运行；请先退出遥操/采集，再启动回放') from exc
        yield


class RunWriter:
    """Separate trace file; persistence errors are not silently ignored."""
    def __init__(self,directory,metadata):
        self.directory = directory
        directory.mkdir(parents=True,exist_ok=False)
        self.stream = (directory/'frames.jsonl').open('x',buffering=65536)
        self.metadata, self.count, self.last_flush = metadata, 0, time.monotonic()
        self.errors = {r: {'count':0,'sum_position_mm':0.,'max_position_mm':0.,
                           'sum_angle_deg':0.,'max_angle_deg':0.,'gripper_count':0,
                           'sum_gripper_mm':0.,'max_gripper_mm':0.} for r in ROLES}

    def write(self,record):
        line=json.dumps(record,ensure_ascii=False,allow_nan=False,
                        default=lambda value: value.tolist() if isinstance(value,np.ndarray) else float(value))
        self.stream.write(line+'\n'); self.count+=1
        for role,arm in record.get('arms',{}).items():
            stats=self.errors[role]; pos,angle=arm['error_mm_deg']
            stats['count']+=1; stats['sum_position_mm']+=pos; stats['sum_angle_deg']+=angle
            stats['max_position_mm']=max(stats['max_position_mm'],pos)
            stats['max_angle_deg']=max(stats['max_angle_deg'],angle)
            grip=arm.get('gripper_error_mm')
            if grip is not None:
                stats['gripper_count']+=1; stats['sum_gripper_mm']+=grip
                stats['max_gripper_mm']=max(stats['max_gripper_mm'],grip)
        if time.monotonic()-self.last_flush>=1:
            self.stream.flush(); self.last_flush=time.monotonic()

    def finish(self,outcome,error=None):
        try: self.stream.flush(); os.fsync(self.stream.fileno())
        finally: self.stream.close()
        for stats in self.errors.values():
            count=stats['count']
            stats['mean_position_mm']=stats['sum_position_mm']/count if count else None
            stats['mean_angle_deg']=stats['sum_angle_deg']/count if count else None
            stats['mean_gripper_mm']=(stats['sum_gripper_mm']/stats['gripper_count']
                                      if stats['gripper_count'] else None)
        with (self.directory/'summary.json').open('x') as stream:
            json.dump({'outcome':outcome,'error':error,'records':self.count,
                       'metadata':self.metadata,'tracking_errors':self.errors},stream,ensure_ascii=False,indent=2)
            stream.flush(); os.fsync(stream.fileno())


class TrackingGuard:
    def __init__(self): self.since={}
    def update(self,name,now,position_mm,angle_deg,joint_error_deg,gripper_error_mm=0.,*,check_position=True):
        if ((not check_position or position_mm<=POSITION_TRACKING_MM) and angle_deg<=10 and joint_error_deg<=20
                and gripper_error_mm<=GRIPPER_TRACKING_MM):
            self.since.pop(name,None); return False
        self.since.setdefault(name,now)
        return now-self.since[name]>=.25-1e-9


class ReplayController:
    def __init__(self,io,home,episode,writer,clock=time.monotonic):
        if set(home.names)!=set(ROLES.values()): raise ValueError('回放必须且只能选择三只从动臂')
        self.io,self.home,self.episode,self.writer,self.clock=io,home,episode,writer,clock
        self.stage='PREFLIGHT'; self.cartesian=set(); self.guard=TrackingGuard()
        self.pending_record=None
        self.ik_commands=None
        self.fk=ForwardKinematics(Path(episode.metadata['sdk_path']),
                                 robot_model=episode.metadata.get('robot_model','piper'))

    def read_feedback(self,require_mode=False):
        states=self.io.read_all(); now=self.clock()
        if set(states)!=set(ROLES.values()): raise RuntimeError('反馈机械臂集合不一致')
        cart={}
        for name,state in states.items():
            if not 0<=now-state.feedback_time<=.15: raise RuntimeError(f'{name}: 关节反馈过期')
            if require_mode and (state.control not in (1,2) or state.move!=1 or not state.enabled
                                 or state.teach not in (0,1,2)):
                raise RuntimeError(f'{name}: 回放末端控制模式改变')
            actual=self.io.read_cartesian(name,now)
            if abs(actual.feedback_time-state.feedback_time)>.05:
                raise RuntimeError(f'{name}: 关节与末端反馈时间差超过50ms')
            cart[name]=actual
        return states,cart

    def wait_stationary(self):
        """Wait without commands for fresh non-teaching, stationary feedback."""
        deadline=self.clock()+3.
        while True:
            self.io.poll(.01)
            try:
                states,cart=self.read_feedback()
            except RuntimeError:
                if self.clock()>=deadline: raise
                continue
            for name,state in states.items():
                if state.teach not in (0,2):
                    raise RuntimeError(f'{name}: 请退出示教，未接管')
            if all(all(abs(v)<=1 for v in s.velocity) for s in states.values()):
                return states,cart
            if self.clock()>=deadline:
                raise RuntimeError('等待机械臂静止超时，未接管')

    def preflight(self):
        states,cart=self.read_feedback()
        self.io.limits=self.io.load_limits()
        states,cart=self.wait_stationary()
        first=frame_episode(self.episode,0)
        for role,name in ROLES.items():
            state=states[name]
            if state.teach not in (0,2) or any(abs(v)>1 for v in state.velocity):
                raise RuntimeError(f'{name}: 请退出示教并保持静止，未接管')
            xyz,q=self.fk.pose(state.pose[:6])
            error=pose_error(PoseTarget(xyz,q,None,np.asarray(state.pose[:6])),cart[name])
            if error[0]>5 or error[1]>2:
                raise RuntimeError(f'{name}: FK与控制器末端不一致，位置{error[0]:.2f}mm/姿态{error[1]:.2f}°；检查DH/TCP')
            limits=np.asarray(self.io.limits[name])
            data=self.episode.targets[role]
            if np.any(data.joints_deg<limits[:6,0]-1) or np.any(data.joints_deg>limits[:6,1]+1):
                raise RuntimeError(f'{name}: 源关节反馈超出设备范围，不发送轨迹')
            if data.gripper_mm is not None and (np.any(data.gripper_mm<limits[-1,0]) or np.any(data.gripper_mm>limits[-1,1])):
                raise RuntimeError(f'{name}: 源夹爪宽度超出设备范围，不发送轨迹')
            if np.max(np.abs(first[role].reference_joints_deg-np.asarray(self.home.home_poses[name][:6])))>3:
                raise RuntimeError(f'{name}: 首帧与配置初始关节位姿差超过3°，不自动跳转')
            if (first[role].gripper_mm is not None
                    and abs(first[role].gripper_mm-self.home.home_poses[name][6])>GRIPPER_CONTINUITY_MM):
                raise RuntimeError(f'{name}: 首帧夹爪与配置初始宽度差超过{GRIPPER_CONTINUITY_MM:g}mm，不自动跳转')
        LOG.info('回放预检查通过：控制器末端与DH=1一致；仅控制三从动臂')
        LOG.info('运动前求解全部末端帧IK；不使用原始关节列作为发送目标')
        self.ik_commands=prepare_ik(self.episode,self.fk,self.home.home_poses,self.io.limits)
        # IK may take seconds; discard queued feedback and recheck before takeover.
        self.wait_stationary()

    def handoff(self):
        states,anchors=self.read_feedback()
        first=frame_episode(self.episode,0)
        for role,name in ROLES.items():
            position,angle=pose_error(first[role],anchors[name])
            if position>15 or angle>5 or np.max(np.abs(first[role].reference_joints_deg-states[name].pose[:6]))>3:
                raise RuntimeError(f'{name}: 归位后与首帧不连续（{position:.2f}mm/{angle:.2f}°），不开始回放')
            if (first[role].gripper_mm is not None
                    and abs(first[role].gripper_mm-states[name].pose[6])>GRIPPER_CONTINUITY_MM):
                raise RuntimeError(f'{name}: 归位后夹爪与首帧差超过{GRIPPER_CONTINUITY_MM:g}mm，不开始回放')
        began=self.clock()
        for name in ROLES.values():
            self.cartesian.add(name)  # Includes partially sent mode changes.
            self.io.enter_cartesian(name,anchors[name])
        while self.clock()-began<4:
            self.io.poll(.01)
            states,cart=self.read_feedback()
            valid=True
            for name in ROLES.values():
                state=states[name]
                position,angle=pose_error(anchors[name],cart[name])
                if position>5 or angle>2: raise RuntimeError(f'{name}: 末端模式交接出现异常位移')
                valid &= (state.control in (1,2) and state.move==1 and state.enabled
                          and state.status_time>began and cart[name].feedback_time>began)
            if valid: return
        raise RuntimeError('末端模式反馈确认超时')

    def run(self):
        try:
            self.preflight()
            self.stage='HOMING'
            LOG.warning('将移动三从动臂与左右夹爪：30°/秒平滑归位，稳定1秒后原速回放；请清空运动区域')
            self.home.home(before_motion=self.wait_stationary)
            self.io.limits=self.home.limits
            self.stage='HANDOFF'; self.handoff()
            self.stage='REPLAY'; begin=self.clock(); previous=begin; next_tick=begin
            frame_index=0; frame_count=len(self.episode.times_s)
            last_targets=None; last_index=None
            duration=(frame_count-1)/HZ; final_sent=None; settled=None
            LOG.info('开始同步末端回放：末端IK→MIT关节跟随，20Hz，%d帧，%.3f秒，无插值',frame_count,duration)
            while True:
                while self.clock()<next_tick:
                    self.io.poll(min(.01,next_tick-self.clock()))
                now=self.clock()
                if now-previous>.1: raise RuntimeError('回放调度停顿超过100ms，不突发补发')
                if final_sent is None and now-next_tick>=1/HZ:
                    raise RuntimeError('已错过一个20Hz发送周期，不跳帧、不突发补发')
                previous=now; elapsed=duration if now>=begin+duration else now-begin
                index=min(frame_index,frame_count-1)
                targets=frame_episode(self.episode,index)
                for role,target in targets.items():
                    target.command_joints_deg=self.ik_commands[role][index].copy()
                states,cart=self.read_feedback(require_mode=True)
                record={'stage':self.stage,'monotonic_s':now,'elapsed_s':elapsed,
                        'frame_index':index,'source_time_s':float(self.episode.times_s[index]),'arms':{}}
                good=True
                for role,name in ROLES.items():
                    target=targets[role]; state=states[name]
                    pos,angle=pose_error(target,cart[name])
                    joint_error=float(np.max(np.abs(target.reference_joints_deg-np.asarray(state.pose[:6]))))
                    grip_error=abs(target.gripper_mm-state.pose[6]) if target.gripper_mm is not None else None
                    tracked=last_targets[role] if last_targets is not None else target
                    track_pos,track_angle=pose_error(tracked,cart[name])
                    track_joint=float(np.max(np.abs(tracked.command_joints_deg-np.asarray(state.pose[:6]))))
                    track_grip=abs(tracked.gripper_mm-state.pose[6]) if tracked.gripper_mm is not None else None
                    record['arms'][role]={'target':asdict(target),'actual_cartesian':asdict(cart[name]),
                        'actual_joint_state':asdict(state),'error_mm_deg':[pos,angle],
                        'reference_joint_error_deg':joint_error,'gripper_error_mm':grip_error,
                        'send_status':'not_attempted','tracking_frame_index':last_index,
                        'tracking_error_mm_deg':[track_pos,track_angle],
                        'tracking_joint_error_deg':track_joint,'tracking_gripper_error_mm':track_grip}
                    if last_targets is not None and self.guard.update(name,now,track_pos,track_angle,track_joint,track_grip or 0.):
                        self.writer.write(record)
                        raise RuntimeError(f'{name}: 已发送帧{last_index}跟踪误差持续超限（{track_pos:.1f}mm/{track_angle:.1f}°/关节{track_joint:.1f}°/夹爪{track_grip}mm）')
                    limits=np.asarray(self.io.limits[name])
                    if np.any(np.asarray(state.pose[:6])<limits[:6,0]-1) or np.any(np.asarray(state.pose[:6])>limits[:6,1]+1):
                        raise RuntimeError(f'{name}: 实测关节越界')
                    good &= pos<=POSITION_ARRIVAL_MM and angle<=2 and all(abs(v)<=3 for v in state.velocity)
                    good &= grip_error is None or grip_error<=GRIPPER_ARRIVAL_MM
                self.pending_record=record
                for role,name in (ROLES.items() if final_sent is None else ()):
                    arm_record=record['arms'][role]
                    arm_record.update(send_monotonic_s=self.clock(),send_status='attempted')
                    try: self.io.send_cartesian(name,targets[role])
                    except BaseException as exc:
                        arm_record.update(send_status='failed',send_error=str(exc) or type(exc).__name__)
                        raise
                    arm_record.update(send_status='completed',send_completed_monotonic_s=self.clock())
                    if self.clock()-now>.1: raise RuntimeError('一轮发送超过100ms')
                self.writer.write(record)
                self.pending_record=None
                if final_sent is None: last_targets=targets; last_index=index
                if index==frame_count-1:
                    if final_sent is None: final_sent=self.clock(); self.stage='FINAL_SETTLE'
                    settled=(now if settled is None else settled) if good else None
                    if settled is not None and now-settled>=.5 and now>final_sent: break
                    if now-final_sent>5: raise RuntimeError('末帧到位确认超时')
                if final_sent is None:
                    frame_index+=1
                    next_tick=begin+frame_index/HZ
                else: next_tick=now+1/HZ
            self.stage='DONE'; LOG.info('同步末端回放完成，末帧已到位；不自动复位')
        except BaseException as exc:
            self.stop(str(exc) or type(exc).__name__)
            # Stop the entire group before attempting potentially failing disk I/O.
            if self.pending_record is not None:
                self.pending_record.update(interrupted=True,error=str(exc) or type(exc).__name__)
                try: self.writer.write(self.pending_record)
                except Exception: LOG.exception('中断发送轮次无法落盘；停止请求已完成')
            raise

    def stop(self,reason):
        with cleanup_signals(): self._stop(reason)

    def _stop(self,reason):
        self.stage='STOPPING'; results=[]
        for name in ROLES.values():
            if name not in self.cartesian: continue
            try: results.append(self.io.stop_cartesian(name))
            except Exception as exc: results.append({'arm':name,'action':'stop_failed','reason':str(exc)})
        self.home.touched.difference_update(self.cartesian)
        try: self.home.stop()
        except Exception as exc: results.append({'action':'home_stop_failed','reason':str(exc)})
        self.stage='STOPPED'
        LOG.warning('回放已中止：%s；软件停止请求=%s。这不是硬件急停。',reason,results)
        try: self.writer.write({'stage':'STOPPED','reason':reason,'stop_results':results})
        except Exception: LOG.exception('停止日志写入失败；请以终端告警和现场状态为准')


def positive_int(value):
    try: number=int(value)
    except ValueError as exc: raise argparse.ArgumentTypeError('episode_num必须为正整数') from exc
    if number<1: raise argparse.ArgumentTypeError('episode_num必须为正整数')
    return number


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='basket')
    parser.add_argument('--episode_num',type=positive_int,required=True,
                        help='LeRobot 中第几条（从1开始；48对应episode_index=47）')
    parser.add_argument('--dataset',type=Path,
                        help='LeRobot任务目录；默认优先data_lerobot_versions/latest/<task>，否则data_lerobot/<task>')
    parser.add_argument('--check',action='store_true',help='仅离线准备末端轨迹，不连接CAN，不运动')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',datefmt='%H:%M:%S')
    dataset=resolve_dataset(ROOT,args.task,args.dataset)
    mapping=json.loads((ROOT/'src/teleop/arm_mapping.json').read_text())
    sdk_file=ROOT/'third_party/piper_sdk/piper_sdk/kinematics/piper_fk.py'
    profile_path=Path(__file__).with_name('kinematics.json')
    profile=json.loads(profile_path.read_text())
    if profile.get('dh_is_offset')!=1: raise ValueError('本回放只支持已验证的DH=1')
    episode=load_lerobot_episode(dataset,args.episode_num,mapping,sdk_file,robot_model=profile['robot_model'])
    episode.metadata.update(kinematics_profile=profile,kinematics_profile_sha256=file_hash(profile_path))
    LOG.info('轨迹准备完成（直接读取LeRobot action，不使用state作为目标）：%s；第%d条/index=%d；%d帧，%.3f秒，型号=%s，DH=1，三从动臂',
             episode.metadata['source_path'],args.episode_num,episode.metadata['episode_index'],
             len(episode.times_s),episode.times_s[-1],profile['robot_model'])
    if args.check:
        LOG.info('离线检查通过；没有连接CAN、没有发送运动命令；尚未验证实机末端坐标定义和跟踪能力')
        return 0
    with runtime_locks():
        directory=ROOT/'data_vis/replay'/args.task/f'episode{args.episode_num}'/(time.strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        metadata=dict(episode.metadata,source_sha256=episode.source_sha256,speed=1.0,send_hz=HZ,
            interpolation='none',playback_duration_s=(len(episode.times_s)-1)/HZ,
            home_joint_speed_deg_s=30,home_stable_s=1,mode='end-pose IK -> MOVE_J MIT 0xAD',
            tracking_reference='last successfully sent frame',
            limits={'startup_mm_deg':[5,2],'tracking_mm_deg':[POSITION_TRACKING_MM,10],'tracking_hold_s':.25,'joint_branch_deg':20,
                    'gripper_continuity_mm':GRIPPER_CONTINUITY_MM,'gripper_tracking_mm':GRIPPER_TRACKING_MM,
                    'gripper_arrival_mm':GRIPPER_ARRIVAL_MM,'position_arrival_mm':POSITION_ARRIVAL_MM})
        writer=RunWriter(directory,metadata); io=None; outcome='FAILED'; error=None
        old_signals={}
        def cancel(signum,frame): raise KeyboardInterrupt(f'收到信号{signum}')
        for sig in (signal.SIGINT,signal.SIGTERM):
            old_signals[sig]=signal.signal(sig,cancel)
        try:
            arms=[a for a in mapping['arms'] if a['id'] in ROLES.values()]
            interfaces=discover_interfaces(arms)
            io=ReplayIO(interfaces,{a['id']:a.get('has_gripper',True) for a in arms})
            deadline=time.monotonic()+3
            while True:
                io.poll(.01)
                try:
                    io.read_all()
                    for name in ROLES.values(): io.read_cartesian(name,time.monotonic())
                    break
                except RuntimeError:
                    if time.monotonic()>deadline: raise
            io.connect_commands()
            home=Teleop(io,mapping,arm_ids=list(ROLES.values()),home_joint_speed=30.)
            ctl=ReplayController(io,home,episode,writer)
            LOG.info('本次回放日志：%s',directory)
            ctl.run(); outcome='DONE'
        except BaseException as exc:
            error=str(exc) or type(exc).__name__
            outcome='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED'
            raise
        finally:
            # Do not let repeated Ctrl+C interrupt cleanup.
            for sig in old_signals: signal.signal(sig,signal.SIG_IGN)
            try:
                if io is not None: io.close()
                writer.finish(outcome,error)
            finally:
                for sig,handler in old_signals.items(): signal.signal(sig,handler)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.warning('用户取消回放；请确认实机状态，软件停止不是硬件急停')
        raise SystemExit(130)
    except Exception as exc:
        LOG.error('%s',exc)
        raise SystemExit(1)
