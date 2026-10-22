"""Run recorded NPZ through installed CLI and local mock protocol (no model/GPU claim)."""
import argparse
import subprocess
import sys
from test_unified_client import FakeServer

p=argparse.ArgumentParser()
p.add_argument('--piper',action='store_true')
p.add_argument('--observation',required=True)
a=p.parse_args()
server=FakeServer('joint')
entry=['-m','src.infer.mixed76_client'] if a.piper else ['scripts/run_mixed76_client.py','--embodiment','g2']
result=subprocess.run([sys.executable,*entry,'--model','joint','--task','basket' if a.piper else 'yifu',
                       '--inference','local','--ip','127.0.0.1','--port',str(server.port),
                       '--observation',a.observation],capture_output=True,text=True,timeout=30)
print(result.stdout)
print(result.stderr)
server.thread.join(2)
if result.returncode:raise SystemExit(result.returncode)
assert '"robot_io": false' in result.stdout
assert '"shape": [32, 76]' in result.stdout
