import socket
import threading
import unittest
import numpy as np
from client_mixed76 import Mixed76Client
from _ours_protocol import send_message, recv_message
from mixed76_contract import REPRESENTATION, embodiment_mask, encode_state, decode_actions
from piper_mixed76 import encode_observation, execution_actions


def metadata(model):
    return dict(protocol_version=4, model=model, representation=REPRESENTATION,
                action_dim=76, state_dim=76, action_units='physical', state_units='physical',
                action_mode='absolute', horizon=32, dt_s=.05, camera_keys=['head_rgb'],
                weights_strict=True, checkpoint='test-only', normalization_sha256='test',
                deployment_id='test', prompts={'basket':'test','yifu':'test'}, embodiments=['3b','g2'])


class FakeServer:
    def __init__(self, model='joint', mutate=None):
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1',0)); self.listener.listen()
        self.port=self.listener.getsockname()[1]
        self.model,self.mutate=model,mutate
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def run(self):
        with self.listener:
            conn,_=self.listener.accept()
            with conn:
                while True:
                    try: req=recv_message(conn)
                    except ConnectionError: return
                    if req['cmd']=='info': result=metadata(self.model)
                    elif req['cmd']=='reset':
                        result=dict(protocol_version=4,request_id=req['request_id'],deployment_id='test',session_id='s',ok=True)
                    else:
                        result={k:req[k] for k in ('request_id','session_id','sample_ns','embodiment')}
                        result.update(protocol_version=4,deployment_id='test',representation=REPRESENTATION,
                                      action_units='physical',dt_s=.05,inference_seconds=.001,
                                      action_mask=embodiment_mask(req['embodiment']).tolist(),
                                      actions=np.tile(req['observation']['state'],(32,1)))
                    if self.mutate: self.mutate(req,result)
                    send_message(conn,result)


class ContractTests(unittest.TestCase):
    def test_all_models_both_embodiments(self):
        for model in ('fastwam','alpha','joint','tri','coarse_fine','pi05'):
            for kind in ('3b','g2'):
                with self.subTest(model=model,kind=kind):
                    server=FakeServer(model)
                    c=Mixed76Client('127.0.0.1',server.port,model=model)
                    try:
                        a=c.infer(np.zeros(76),np.zeros((10,12,3),np.uint8),task='basket',embodiment=kind)
                        self.assertEqual(a.shape,(32,76)); c.reset()
                    finally: c.close()
                    server.thread.join(2)

    def test_wrong_model(self):
        s=FakeServer('alpha')
        with self.assertRaisesRegex(ValueError,'model'): Mixed76Client('127.0.0.1',s.port,model='joint')

    def test_bad_responses(self):
        for key,value in [('request_id','stale'),('session_id','old'),('deployment_id','other'),
                          ('dt_s',.1),('action_units','normalized'),('action_mask',[]),
                          ('actions',np.zeros((1,76))),('inference_seconds',-1)]:
            with self.subTest(key=key):
                def mutate(req,res):
                    if req['cmd']=='infer': res[key]=value
                s=FakeServer(mutate=mutate);c=Mixed76Client('127.0.0.1',s.port,model='joint')
                try:
                    with self.assertRaises(ValueError):
                        c.infer(np.zeros(76),np.zeros((2,2,3),np.uint8),task='basket',embodiment='3b')
                finally:c.close()
                s.thread.join(2)

    def test_roundtrip_and_piper_adapter(self):
        for kind,dim,starts in [('3b',23,(0,8,16)),('g2',61,(0,29))]:
            raw=np.zeros(dim,np.float32)
            for start in starts: raw[start+6]=1
            encoded=encode_state(raw,kind)
            np.testing.assert_allclose(decode_actions(encoded[None],kind)[0],raw)
        obs=dict(eef={r:[0,0,0,0,0,0,1] for r in ('left','right','active_camera')},
                 gripper_mm={'left':30,'right':40},images={'active_camera':np.zeros((2,2,3),np.uint8)})
        state,rgb=encode_observation(obs);result=execution_actions(np.tile(state,(32,1)))
        self.assertEqual(result['gripper_mm']['left'][0],30)
        self.assertEqual(result['eef']['active_camera'].shape,(32,7))

    def test_invalid_input(self):
        s=FakeServer();c=Mixed76Client('127.0.0.1',s.port,model='joint')
        try:
            with self.assertRaises(ValueError): c.infer(np.ones(76),np.zeros((2,2,3),np.uint8),task='basket',embodiment='3b')
            with self.assertRaises(ValueError): c.infer(np.zeros(76),np.zeros((2,2,3)),task='basket',embodiment='3b')
        finally:c.close()


if __name__=='__main__': unittest.main()
