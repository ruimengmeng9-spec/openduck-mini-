"""R6: joint-aligned extension with a strict locomotion-policy entry gate.

Only search shaping/entry conditions change; recovery success is still checked
by continuous dynamics and sustained supported standing after the handoff.
"""
import json
from pathlib import Path
import sys

import numpy as np

from diagnostics import getup_crouch_extension as search
from diagnostics import getup_dynamic_beam as dynamic
from diagnostics.getup_dynamic_beam import SelfCollisionSim
from diagnostics.getup_independent_native import digest


class AlignedSim(SelfCollisionSim):
    def measure(self):
        m=super().measure()
        error=np.abs(self.data.qpos[self.qadr]-self.home)
        m['joint_home_error_max_rad']=float(error.max())
        m['joint_home_error_mean_rad']=float(error.mean())
        m['motor_target_home_error_max_rad']=float(np.abs(self.prev-self.home).max())
        return m


def strict_entry(m):
    return bool(m['up_z']>.97 and .14<m['height_m']<.23 and all(m['feet'])
                and not m['torso_contact'] and m['linear_speed_mps']<.08
                and m['angular_speed_rad_s']<.5 and m['self_penetration_m']<.002
                and m['joint_home_error_max_rad']<.45
                and m['motor_target_home_error_max_rad']<.45)


_base_evaluate=search.evaluate


def aligned_evaluate(job):
    result=_base_evaluate(job)
    tail=result['measurements'][-min(8,len(result['measurements'])):]
    # The standing actor was trained near home; height alone is not a valid
    # reason to enter it with reversed knees or a distant motor target.
    penalty=np.mean([2.*np.clip((m['up_z']-.75)/.2,0.,1.)
                     *np.clip((m['height_m']-.08)/.06,0.,1.)
                     *m['joint_home_error_mean_rad'] for m in tail])
    result['score']-=float(penalty)
    return result


def main():
    search.SelfCollisionSim=AlignedSim
    search.ready_to_handoff=strict_entry
    search.evaluate=aligned_evaluate
    dynamic.ready_to_handoff=strict_entry
    search.main()
    output=Path(sys.argv[sys.argv.index('--output')+1])
    results=json.loads((output/'results.json').read_text())
    results['method']='strict joint-aligned crouch-to-stand search (R6)'
    results['entry_gate']='up>.97, height .14-.23m, two feet, low velocity, no torso support, joint and motor home errors <.45rad'
    results['hashes'][str(Path(__file__))]=digest(__file__)
    (output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
