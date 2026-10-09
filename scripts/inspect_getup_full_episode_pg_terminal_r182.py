"""Compact read-only terminal results; no parameter dumps."""
import json
from diagnostics import train_getup_full_episode_pg_r181 as run
from diagnostics import audit_getup_full_episode_pg_terminal_r182 as audit

terminal=audit.read(run.OUTPUT/'results.json');closed=audit.read(run.OUTPUT/'training_closed.json')
baseline={r['case_seed']:r for r in terminal['baseline']['rows']}
for h in closed['history']:
    report=h['complete_development']
    print(json.dumps(dict(update=h['update'],successes=report['successes'],physical_failures=report['physical_failures'],rescued=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and r['success'] and not baseline[r['case_seed']]['success']],regressed=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and not r['success'] and baseline[r['case_seed']]['success']],invalid=[dict(case=r['case_seed'],controls=r['controls'],peaks=r['peaks'],maximum_extra=r['maximum_pg_extra_rad'],combined_cap=r['maximum_combined_14_joint_correction_rad']) for r in report['rows'] if not r['valid']],gradient_norms=h['gradient_norms'],changes=h['maximum_parameter_changes'],score_loss=h['score_loss'])))
print(json.dumps(dict(candidate=terminal['candidate']['successes'],baseline=terminal['baseline']['successes'],invalid=terminal['candidate']['physical_failures'],gate=terminal['original_development_gate'],attempts=terminal['formal_dynamic_attempts'])))
if (audit.OUTPUT/'results.json').exists():
    a=audit.read(audit.OUTPUT/'results.json');s=audit.read(audit.SMOKE/'results.json')
    assert a['programs'][:4]==s['programs'] and a['terminal_pairing']==s['terminal_pairing']
    print(json.dumps(dict(audit_scalar_attempts=a['scalar_attempts'],learner_updates=len(a['learner_updates']),paired=len(a['terminal_pairing']),source_hashes_unchanged=a['source_hashes_unchanged'],smoke_stats_exact=True)))
