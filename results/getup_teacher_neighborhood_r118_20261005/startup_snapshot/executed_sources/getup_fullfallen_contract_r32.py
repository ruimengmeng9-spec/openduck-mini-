"""Dependency-light shaping, decoding and acceptance for full-fallen R32."""
import numpy as np

POSES=('prone','supine','left_side','right_side')
DECISION_CONTROLS=5
GAMMA=.99

def potential(metric,gravity_x):
    """Training shaping only: no potential value declares success."""
    up=float(np.clip(metric['up_z'],-1.,1.))
    angle=np.arccos(up)
    lift=np.exp(-((metric['height_m']-.165)/.05)**2)
    upright=np.exp(-(angle/.65)**2)
    home=np.exp(-(metric['joint_home_error_mean_rad']/.6)**2)
    loaded=metric['foot_load_fraction']
    flat=np.clip(min(metric['foot_up_alignment_to_home']),0.,1.)
    roll=-.25*gravity_x*max(.5-up,0.)
    return float(1.4*up-2.*abs(metric['height_m']-.165)
                 +1.2*max(up,0.)*loaded*flat+2.*lift*upright*home*loaded+roll)

def decode_action(action,lower,upper):
    action=np.asarray(action,dtype=float)
    if action.shape!=(14,) or not np.isfinite(action).all():
        raise ValueError('14 finite normalized joint targets required')
    if np.abs(action).max()>1.+1e-6:
        raise ValueError('Policy output violates normalized target contract')
    return (lower+upper)/2+np.clip(action,-1.,1.)*(upper-lower)/2

def accepted(initial_fallen,entry_reached,valid,tail_controls,dt):
    return bool(initial_fallen and entry_reached and valid
                and tail_controls*dt>=30.-1e-8)

def bootstrap_observation(obs,terminated):
    """A true terminal has no next value; never feed invalid state to critic."""
    obs=np.asarray(obs,dtype=np.float32)
    if obs.shape!=(50,):
        raise ValueError('Bootstrap observation must retain the 50D contract')
    if terminated:
        return np.zeros_like(obs)
    if not np.isfinite(obs).all():
        raise ValueError('A nonterminal transition has nonfinite observations')
    return obs.copy()

def generalized_advantage(rewards,values,next_values,done,terminated,gamma=GAMMA,lam=.95):
    """Timeouts bootstrap their final state, but no GAE crosses a reset."""
    rewards,values,next_values=(np.asarray(x,dtype=np.float32) for x in (rewards,values,next_values))
    done,terminated=(np.asarray(x,dtype=bool) for x in (done,terminated))
    if rewards.ndim!=2 or any(x.shape!=rewards.shape for x in (values,next_values,done,terminated)):
        raise ValueError('GAE arrays require matching [time,environment] shapes')
    if not all(np.isfinite(x).all() for x in (rewards,values,next_values)):
        raise ValueError('Nonfinite GAE inputs must be diagnosed, not optimized')
    if np.any(terminated & ~done):
        raise ValueError('A true terminal must also be a done transition')
    advantage=np.zeros_like(rewards)
    running=np.zeros(rewards.shape[1],dtype=np.float32)
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*np.where(terminated[t],0.,next_values[t])-values[t]
        running=delta+gamma*lam*np.where(done[t],0.,running)
        advantage[t]=running
    return advantage,advantage+values

def completion_summary(rows):
    summary={}
    for pose in POSES:
        trials=[r for r in rows if r['pose']==pose]
        successes=sum(bool(r['success']) for r in trials)
        unique=len({r['seed'] for r in trials})
        summary[pose]=dict(trials=len(trials),unique_seeds=unique,successes=successes,
                           passed=len(trials)==20 and unique==20 and successes>=18)
    return summary,all(s['passed'] for s in summary.values())
