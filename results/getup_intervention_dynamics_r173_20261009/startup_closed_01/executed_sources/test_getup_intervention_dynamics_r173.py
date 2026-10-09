"""Mathematical/causal regression before any new physical rollout."""
import numpy as np
from diagnostics import train_getup_intervention_dynamics_r173 as run

def main():
    plans,rng=run.instruments();other,other_rng=run.instruments()
    np.testing.assert_array_equal(plans,other);assert rng==other_rng
    assert plans.shape==(7,529,3) and np.abs(plans).max()==run.AMPLITUDE
    np.testing.assert_array_equal(plans[0],np.zeros((529,3)))
    for a in range(3):
        np.testing.assert_array_equal(plans[1+2*a],-plans[2+2*a])
        assert not np.any(plans[1+2*a,:,np.arange(3)!=a])
    nominal=np.zeros(55,dtype=np.float32);current=nominal.copy();current[15]=.02
    for k in (0,1,50,528,529,2278):
        for p in plans:
            extra,activation=run.intervention(nominal,nominal,k,p)
            np.testing.assert_array_equal(extra,np.zeros(3));assert activation==0
    for k in (0,529,2278):np.testing.assert_array_equal(run.intervention(current,nominal,k,plans[1])[0],np.zeros(3))
    plus,activation=run.intervention(current,nominal,1,plans[1]);minus,_=run.intervention(current,nominal,1,plans[2])
    np.testing.assert_array_equal(plus,-minus);assert 0<np.abs(plus).max()<=run.AMPLITUDE and 0<activation<1
    changed=current.copy();changed[34:]=1.e9
    np.testing.assert_array_equal(run.intervention(changed,nominal,1,plans[1])[0],plus)
    before=[p.copy() for p in (current,nominal,plans[1])];run.intervention(current,nominal,1,plans[1])
    for a,b in zip(before,(current,nominal,plans[1])):np.testing.assert_array_equal(a,b)
    target=np.zeros(14);lower=np.full(14,-1.);upper=-lower;prev=np.zeros(14)
    assert run.local.merge_target(target,target,np.zeros(3),np.array([9,10,11]),lower,upper) is target
    merged=run.local.merge_target(target,target,plus,np.array([9,10,11]),lower,upper)
    assert np.abs(merged-target).max()<=.18
    bounded=run.planner.apply_limits(np.full(14,1.),prev,lower,upper)
    np.testing.assert_array_equal(bounded,np.full(14,5.24*.02))
    for bad in (np.nan,np.inf):
        x=current.copy();x[0]=bad
        try:run.intervention(x,nominal,1,plans[1])
        except ValueError:pass
        else:raise AssertionError('Nonfinite input accepted')
    try:run.intervention(current,nominal,1,plans[1]*2)
    except ValueError:pass
    else:raise AssertionError('Over-amplitude instrument accepted')
    assert run.dynamics.regressions()['passed']
    print('R173_12_INSTRUMENT_REGRESSION_CLASSES_AND_R172_10_MATH_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
