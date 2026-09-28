"""Export the learned 24-parameter feedback policy; no legacy decoder allowed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T
from diagnostics.backward_phase_search import CpuActor, phase_delta, balance_delta, balance_features, capture_features, capture_delta


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--up-gate',help='smoothly relinquish gait correction as body tips: high,low up-vector Z')
    a=p.parse_args()
    config=json.loads(a.controller.read_text())
    if a.up_gate:
        high,low=[float(x) for x in a.up_gate.split(',')]
        if not .9 <= low < high <= .995:
            raise ValueError('invalid balance gate')
        config['up_gate']=[high,low]
    weights=np.asarray(config['weights'],dtype=np.float32).reshape(6,3)
    balance=config.get('balance_weights')
    capture=config.get('capture_weights')
    if capture is not None and (balance is None or len(balance)!=4 or len(capture)!=4):
        raise ValueError('capture correction requires four support-contact balance gains')
    deadband=float(config.get('balance_deadband_rad',.10))
    if balance is not None and len(balance)==4:
        config.update(ankle_gain_rad=.06,ankle_support_gate='native foot-body/floor geometry contact; no swing ankle balance correction')
    feedback=np.asarray(config.get('feedback_weights',np.zeros(6)),dtype=np.float32)
    kernel=np.stack([weights[:,1],weights[:,2],feedback])
    mapping=np.zeros((6,14),dtype=np.float32)
    mapping[np.arange(6),[0,1,2,9,10,11]]=1.
    mapping[2,4]=-1.
    mapping[5,13]=-1.
    constants={
        'kernel':kernel,'bias':weights[:,0],'gains':np.asarray([.06,.035,.06,.06,.035,.06],dtype=np.float32),
        'mapping':mapping,'starts':np.array([0],dtype=np.int64),'ends':np.array([3],dtype=np.int64),
        'ramp_start':np.array([3],dtype=np.int64),'ramp_end':np.array([4],dtype=np.int64),'axes':np.array([1],dtype=np.int64),
    }
    nodes=[h.make_node('Slice',['obs','starts','ends','axes'],['features']),
           h.make_node('Gemm',['features','kernel','bias'],['raw']),h.make_node('Tanh',['raw'],['bounded']),
           h.make_node('Mul',['bounded','gains'],['angles']),
           h.make_node('Slice',['obs','ramp_start','ramp_end','axes'],['ramp']),
           h.make_node('Mul',['angles','ramp'],['ramped']),
           h.make_node('MatMul',['ramped','mapping'],['ungated' if config.get('up_gate') else 'phase_delta' if balance is not None else 'joint_delta_rad'])]
    if config.get('up_gate'):
        high,low=config['up_gate']
        constants.update(up_start=np.array([4],dtype=np.int64),up_end=np.array([5],dtype=np.int64),up_low=np.array([low],dtype=np.float32),up_width=np.array([high-low],dtype=np.float32),zero=np.array([0],dtype=np.float32),one=np.array([1],dtype=np.float32))
        nodes += [h.make_node('Slice',['obs','up_start','up_end','axes'],['up']),h.make_node('Sub',['up','up_low'],['margin']),h.make_node('Div',['margin','up_width'],['ratio']),h.make_node('Clip',['ratio','zero','one'],['gate']),h.make_node('Mul',['ungated','gate'],['phase_delta' if balance is not None else 'joint_delta_rad'])]
    input_size=5 if config.get('up_gate') else 4
    if balance is not None:
        bm=np.zeros((1,14),dtype=np.float32)
        bm[0,[2,11]]=.06
        bm[0,[4,13]]=-.06
        constants.update(balance_start=np.array([input_size],dtype=np.int64),balance_end=np.array([input_size+2],dtype=np.int64),balance_kernel=np.asarray(balance[:2],dtype=np.float32).reshape(2,1),balance_mapping=bm)
        nodes += [h.make_node('Slice',['obs','balance_start','balance_end','axes'],['balance_features']),h.make_node('MatMul',['balance_features','balance_kernel'],['balance_raw']),h.make_node('Tanh',['balance_raw'],['balance_bounded']),h.make_node('Mul',['balance_bounded','ramp'],['balance_ramped']),h.make_node('MatMul',['balance_ramped','balance_mapping'],['balance_delta']),h.make_node('Add',['phase_delta','balance_delta'],['hip_plus_phase' if len(balance)==4 else 'joint_delta_rad'])]
        input_size+=2
        if len(balance)==4:
            am=np.zeros((2,14),dtype=np.float32)
            am[0,4]=.06
            am[1,13]=.06
            constants.update(contact_start=np.array([input_size],dtype=np.int64),contact_end=np.array([input_size+2],dtype=np.int64),ankle_kernel=np.asarray(balance[2:],dtype=np.float32).reshape(2,1),ankle_mapping=am)
            nodes += [h.make_node('MatMul',['balance_features','ankle_kernel'],['ankle_raw']),h.make_node('Tanh',['ankle_raw'],['ankle_bounded']),h.make_node('Mul',['ankle_bounded','ramp'],['ankle_ramped']),h.make_node('Slice',['obs','contact_start','contact_end','axes'],['contacts']),h.make_node('Mul',['ankle_ramped','contacts'],['supported_ankles']),h.make_node('MatMul',['supported_ankles','ankle_mapping'],['ankle_delta']),h.make_node('Add',['hip_plus_phase','ankle_delta'],['joint_delta_rad'])]
            input_size+=2
    if capture is not None:
        for node in nodes:
            for index,name in enumerate(node.output):
                if name=='joint_delta_rad':
                    node.output[index]='prior_delta'
        cm=np.zeros((1,14),dtype=np.float32)
        cm[0,[2,11]]=.04
        cm[0,[4,13]]=-.04
        cam=np.zeros((2,14),dtype=np.float32)
        cam[0,4]=.04
        cam[1,13]=.04
        constants.update(capture_start=np.array([input_size],dtype=np.int64),capture_end=np.array([input_size+2],dtype=np.int64),capture_ramp_start=np.array([input_size+2],dtype=np.int64),capture_ramp_end=np.array([input_size+3],dtype=np.int64),capture_hip_kernel=np.asarray(capture[:2],dtype=np.float32).reshape(2,1),capture_ankle_kernel=np.asarray(capture[2:],dtype=np.float32).reshape(2,1),capture_hip_mapping=cm,capture_ankle_mapping=cam)
        nodes += [h.make_node('Slice',['obs','capture_start','capture_end','axes'],['capture_features']),h.make_node('Slice',['obs','capture_ramp_start','capture_ramp_end','axes'],['capture_ramp']),h.make_node('MatMul',['capture_features','capture_hip_kernel'],['capture_hip_raw']),h.make_node('Tanh',['capture_hip_raw'],['capture_hip_bounded']),h.make_node('Mul',['capture_hip_bounded','capture_ramp'],['capture_hip_ramped']),h.make_node('MatMul',['capture_hip_ramped','capture_hip_mapping'],['capture_hip_delta']),h.make_node('MatMul',['capture_features','capture_ankle_kernel'],['capture_ankle_raw']),h.make_node('Tanh',['capture_ankle_raw'],['capture_ankle_bounded']),h.make_node('Mul',['capture_ankle_bounded','capture_ramp'],['capture_ankle_ramped']),h.make_node('Mul',['capture_ankle_ramped','contacts'],['capture_supported_ankles']),h.make_node('MatMul',['capture_supported_ankles','capture_ankle_mapping'],['capture_ankle_delta']),h.make_node('Add',['prior_delta','capture_hip_delta'],['prior_plus_capture_hip']),h.make_node('Add',['prior_plus_capture_hip','capture_ankle_delta'],['joint_delta_rad'])]
        input_size+=3
    graph=h.make_graph(nodes,'phase_leg_feedback',[h.make_tensor_value_info('obs',T.FLOAT,[1,input_size])],[h.make_tensor_value_info('joint_delta_rad',T.FLOAT,[1,14])],[nh.from_array(v,k) for k,v in constants.items()])
    model=h.make_model(graph,opset_imports=[h.make_opsetid('',15)],producer_name='OpenDuckNativePhaseSearch')
    model.ir_version=8
    onnx.checker.check_model(model)
    a.output.mkdir(parents=True,exist_ok=True)
    path=a.output/'corrector.onnx'
    onnx.save(model,path)
    actor=CpuActor(path)
    rng=np.random.default_rng(88)
    maximum=0.
    for _ in range(1000):
        phase=rng.uniform(-np.pi,np.pi)
        error=rng.uniform(-np.pi,np.pi)
        ramp=rng.uniform(0.,1.)
        obs=np.array([np.cos(phase),np.sin(phase),np.sin(error),ramp],dtype=np.float32)
        expected=phase_delta(np.asarray(config['weights']),obs[:2],float(obs[3]),config.get('feedback_weights'),error)
        if config.get('up_gate'):
            up=rng.uniform(.9,1.)
            obs=np.append(obs,np.float32(up)).astype(np.float32)
            high,low=config['up_gate']
            expected*=np.clip((float(obs[4])-low)/(high-low),0.,1.)
        if balance is not None:
            pitch,rate=rng.uniform(-.7,.7),rng.uniform(-4.,4.)
            obs=np.concatenate([obs,balance_features(pitch,rate,deadband)]).astype(np.float32)
            contacts=(0.,0.)
            if len(balance)==4:
                contacts=rng.integers(0,2,2).astype(np.float32)
                obs=np.concatenate([obs,contacts]).astype(np.float32)
            expected+=balance_delta(balance,pitch,rate,float(obs[3]),deadband,contacts)
        if capture is not None:
            cp_features=capture_features(rng.uniform(-.1,.1),rng.uniform(-1.,1.))
            cp_ramp=rng.uniform(0,1)
            obs=np.concatenate([obs,cp_features,[cp_ramp]]).astype(np.float32)
            expected+=capture_delta(capture,cp_features,cp_ramp,contacts)
        maximum=max(maximum,float(np.max(abs(actor.infer(obs)-expected))))
    if maximum>1e-6:
        raise ValueError(f'ONNX parity failed: {maximum}')
    config.update(onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),onnx_input='phase_cos,phase_sin,sin_relative_heading_error,startup_ramp'+(',up_vector_z' if config.get('up_gate') else '')+(',deadband_pitch_scaled,clipped_pitch_rate_scaled' if balance is not None else '')+(',left_foot_contact,right_foot_contact' if balance is not None and len(balance)==4 else '')+(',capture_error_scaled,capture_filtered_rate_scaled,capture_activation_ramp' if capture is not None else ''),onnx_output='14 joint deltas in radians',legacy_motor_action_decoder_allowed=False,history_contract='fresh_three_motor_actions',step_contract='infer_target_then_physics_then_phase',onnx_parity_max_error_rad=maximum)
    (a.output/'controller_contract.json').write_text(json.dumps(config,indent=2))
    print('EXPORTED',path,'MAX ERROR RAD:',maximum,flush=True)


if __name__=='__main__':
    main()
