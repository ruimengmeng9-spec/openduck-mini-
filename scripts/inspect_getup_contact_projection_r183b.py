"""Read-only progress inspector; no environment construction or integration."""
import json
from diagnostics import probe_getup_contact_projection_r183b as run

for output in (run.SMOKE,run.OUTPUT):
    print(str(output),flush=True)
    for name in ('startup_closed.json','results.json','training_closed.json'):
        path=output/name
        if path.exists():
            data=json.loads(path.read_text());print(name,{k:v for k,v in data.items() if k not in ('parity','projection','candidate','baseline')})
            for key in ('parity','projection','candidate','baseline'):
                if key in data:
                    report=data[key];print(key,{k:report[k] for k in ('successes','physical_failures','nominal_success')})
                    for r in report['rows']:
                        print({k:r[k] for k in ('case','controls','success','valid','entry_time_s','strict_tail_s','maximum_projection_extra_rad','active_contact_controls') if k in r})
    if output.exists():
        files=list(output.rglob('result.json'));print('saved_cases',len(files),'failures',len(list(output.rglob('failure.json'))))
        for path in files:
            data=json.loads(path.read_text());print(str(path.relative_to(output)),{k:data[k] for k in ('controls','valid','success','maximum_projection_extra_rad','active_contact_controls') if k in data})
