"""Temporally coherent action exploration, with 50 Hz physical target updates."""
import numpy as np
from diagnostics.getup_native_curriculum import NativeEpisode


def advance_macro(episode,action,repeat,discount_per_step):
    if not (1<=repeat<=10 and 0<discount_per_step<=1): raise ValueError('bad macro contract')
    reward=0.
    for index in range(repeat):
        obs,r,done,terminated,bootstrap,info=episode.step(action)
        reward+=(discount_per_step**index)*r
        if done: break
    return obs,float(reward),done,terminated,bootstrap,info,index+1


def macro_advantage(rewards,values,next_values,done,terminated,counts,repeat,gamma=.99,lam=.95):
    advantages=np.zeros_like(rewards,dtype=np.float32)
    running=np.zeros(rewards.shape[1],np.float32)
    discount=np.power(gamma,counts/repeat)
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+discount[t]*next_values[t]*(1-terminated[t])-values[t]
        running=delta+discount[t]*lam*(1-done[t])*running
        advantages[t]=running
    return advantages,advantages+values


def pipe_worker(connection,scene,stand,starts,number,seed,tilt_max,repeat):
    import os
    os.environ['JAX_PLATFORMS']='cpu'
    try:
        envs=[NativeEpisode(scene,stand,starts,seed+i,tilt_max) for i in range(number)]
        # Preserve the R10/R11 worker's initial RNG/reset sequence.
        connection.send(('ready',np.stack([e.reset() for e in envs])))
        while True:
            operation,payload=connection.recv()
            if operation=='close': break
            if operation!='step': raise ValueError('unknown macro operation')
            connection.send(('ok',[advance_macro(e,a,repeat,.99**(1/repeat)) for e,a in zip(envs,payload)]))
    except Exception as exc:
        import traceback
        connection.send(('error',type(exc).__name__+': '+str(exc)+'\n'+traceback.format_exc()))
    finally: connection.close()
