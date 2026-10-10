"""R193 closed evidence verification and unique append-only terminal archive."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
import numpy as np
from diagnostics import train_getup_shared_horizon_teacher_r193 as run
ROOT=run.ROOT;F=run.OUTPUT;S=run.SMOKE
L=ROOT/'outputs/getup_shared_horizon_teacher_launcher_r193_20261010'
REPO=ROOT/'github/openduck-mini-'
PROOF=ROOT/'tmp/getup_shared_horizon_teacher_r193_terminal_verification_20261010.json'
def read(p):return json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def inventory(root):return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
def write(p,value):
 with Path(p).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def idle(base):
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==base
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
 for p in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
  first=Path(p.split()[0]).name if p.split() else ''
  if 'python' not in first and first!='git':continue
  assert not ('-m diagnostics.' in p and any(x in p for x in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_')))
  assert not ('publish_getup' in p or ('git' in p and 'push' in p and 'server-publish' in p))
def same_arrays(a,b,old_only=False):
 with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y:
  if not old_only:assert x.files==y.files
  for k in y.files:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
def sourcecheck(manifest,folder):
 assert manifest
 for p,v in manifest.items():assert digest(p)==v['sha256']==digest(folder/v['copy']),p
def aggregate(rows):
 return dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
 nominal_success=next(r['success'] for r in rows if r['case_seed'] is None),
 physical_failures=sum(not r['valid'] for r in rows),return_sum=float(sum(r['return_sum'] for r in rows)))
def retain(r,b):
 preserved=[];rescued=[];regressed=[]
 for x,y in zip(r['rows'],b['rows']):
  assert x['case_seed']==y['case_seed'] and x['initial_hash']==y['initial_hash']
  if x['success'] and y['success']:preserved.append(x['case_seed'])
  if x['success'] and not y['success']:rescued.append(x['case_seed'])
  if y['success'] and not x['success']:regressed.append(x['case_seed'])
 return dict(preserved=preserved,rescued=rescued,regressed=regressed,eligible=bool(r['nominal_success'] and not r['physical_failures'] and not regressed and r['successes']>b['successes']))
def choose(reports):
 e=[retain(r,reports[0]) for r in reports];allowed=[i for i in range(1,len(e)) if e[i]['eligible']]
 def rank(i):
  v=[x['return_sum'] for x in reports[i]['rows']]
  return reports[i]['successes'],min(v),sum(v),-i
 return (max(allowed,key=rank) if allowed else 0),e
def verify():
 launch=read(L/'result.json')
 assert launch==dict(natural_exit=True,smoke_exit_code=0,formal_exit_code=0,source_hashes_unchanged=True,full_task_completed=False,hardware_readiness=False)
 manifests=0
 for root in (F,S):
  c=read(root/'contract.json');a=read(root/'input_hashes_after.json')
  assert a['source_and_model_files_unchanged'] and a['hashes']==c['hashes'] and a['model_files']==c['model_files']
  for p,sha in {**c['hashes'],**c['model_files']}.items():assert digest(p)==sha,p
  sourcecheck(c['sources'],root/'executed_sources');manifests+=1
  for p in root.glob('worker_source_manifest_*.json'):
   sourcecheck(read(p),root/('worker_executed_sources_'+p.stem.split('_')[-1]));manifests+=1
 sourcecheck(read(L/'contract.json')['sources'],L/'executed_sources');manifests+=1
 for label in ('initial','formal'):
  sourcecheck(read(L/label/'test_source_manifest.json'),L/label/'executed_sources');manifests+=1
  log=(L/label/'regression.log').read_text();assert 'Ran 21 tests' in log and '\nOK\n' in log
 rng=np.random.default_rng(293);d=np.clip(rng.normal(0,1e-5,size=(2,10,4)),-1e-4,1e-4)
 proposals=[np.zeros((10,4)),d[0],-d[0],d[1],-d[1]];state=rng.bit_generator.state
 for root in (F,S):
  saved=read(root/'proposal_rng.json');assert saved['seed']==293 and saved['rng']==state
  np.testing.assert_array_equal(saved['coefficients'],proposals)
 rows=[];invalid=[];controls=0;parity_count=0
 for root,count in ((F,181),(S,6)):
  paths=sorted(root.glob('**/case_*/result.json'));assert len(paths)==count
  assert not list(root.rglob('failure.json')) and not list(root.rglob('partial_trajectory.npz'))
  for p in paths:
   r=read(p);n=r['controls'];controls+=n
   assert r['physics_sha256']=='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae' and r['physics_unchanged']
   assert r['compiled_model_sha256']=='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3' and r['compiled_model_unchanged']
   assert r['root_edits_during_recovery']==0
   assert r['success']==bool(r['valid'] and n==2279 and r['entry_time_s'] is not None and r['entry_time_s']<=12 and r['strict_tail_s']>=30-1e-8)
   old=run.selector.OUTPUT/'candidate'/p.parent.name;original=read(old/'result.json')
   assert r['initial_hash']==original['initial_hash']
   with np.load(p.parent/'trajectory.npz',allow_pickle=False) as z,np.load(old/'trajectory.npz',allow_pickle=False) as oz:
    assert z['observations'].shape==(n,55) and z['construction_preparation_sensors'].shape==(40,50) and z['preparation_sensors'].shape==(40,50)
    np.testing.assert_array_equal(z['applied'][0],oz['applied'][0])
    np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
    np.testing.assert_array_equal(z['teacher_request_rad'][0],np.zeros(14))
    if n>529:np.testing.assert_array_equal(z['teacher_request_rad'][529:],np.zeros_like(z['teacher_request_rad'][529:]))
    assert np.isfinite(z['teacher_request_rad']).all()
    assert float(np.abs(z['teacher_request_rad']).max())==r['maximum_raw_teacher_request_rad']
    assert float(np.abs(z['planned_before_integration_rad']-z['same_state_unperturbed_planned_rad']).max())==r['maximum_same_state_post_slew_direct_difference_rad']
    if not r['enabled'] or r['case_seed'] is None:
     for k in oz.files:np.testing.assert_array_equal(z[k],oz[k],err_msg=k)
     assert r['peaks']==original['peaks']
     annotations={'baseline','global_program_choice','gains','no_case_metadata_in_controller','original_complete_trace_bitwise_equal','model_group','selected_fixed_R133_program_bitwise_equal','R134_baseline_exact'}
     assert set(original)-set(r)<=annotations
     for k in set(original)&set(r):assert r[k]==original[k],k
     parity_count+=1
   if not r['valid']:invalid.append(dict(identity=str(p.parent.relative_to(ROOT/'outputs')),case=r['case_seed'],controls=n,peaks=r['peaks'],raw=r['maximum_raw_teacher_request_rad'],post=r['maximum_same_state_post_slew_direct_difference_rad'],total=r['maximum_combined_14_joint_correction_rad']))
   rows.append(dict(identity=str(p.parent.relative_to(ROOT/'outputs')),controls=n,valid=r['valid'],success=r['success'],initial_hash=r['initial_hash'],peaks=r['peaks'],result_sha256=digest(p),trajectory_sha256=digest(p.parent/'trajectory.npz')))
 for name,cases in [('zero_parity',[None,769000,773004]),('teacher_smoke',[None,769002,773004])]:
  for case in cases:
   a=F/name/f'case_{case}';b=S/name/f'case_{case}'
   same_arrays(a/'trajectory.npz',b/'trajectory.npz');assert read(a/'result.json')==read(b/'result.json')
 reports=[]
 for i in range(5):
  path=F/'teacher_training'/f'program_{i:02d}'
  r=[read(path/f'case_{case}'/'result.json') for case in run.CASES];report=aggregate(r)
  assert report==read(path/'results.json')
  for x in r:np.testing.assert_array_equal(x['teacher_coefficients_rad'],proposals[i])
  reports.append(report);closed=read(F/f'closed_program_{i:02d}.json');choice,e=choose(reports)
  assert closed['reports']==reports and closed['selected']==choice and closed['evidence']==e and closed['rng']==state and closed['full_training_attempts']==25*(i+1)
  np.testing.assert_array_equal(closed['coefficients'],proposals[i])
 selected,evidence=choose(reports);training=read(F/'training_closed.json');final=read(F/'results.json')
 assert training['reports']==reports and training['retention']==evidence and training['selected_program']==selected and training['rng']==state
 for name in ('baseline','candidate'):
  report=aggregate([read(F/('development_'+name)/f'case_{c}'/'result.json') for c in run.CASES])
  assert report==final[name]==read(F/('development_'+name)/'results.json')
 for case in run.CASES:
  a=F/'development_baseline'/f'case_{case}';b=F/'development_candidate'/f'case_{case}'
  same_arrays(a/'trajectory.npz',b/'trajectory.npz');assert read(a/'result.json')==read(b/'result.json')
 assert selected==0 and final['selected_program']==0 and final['retention']==evidence and final['final_retention']==retain(final['candidate'],final['baseline'])
 assert final['terminal_result_saved'] and not any(final[k] for k in ('original_development_gate','full_task_completed','hardware_readiness','expanded_development_run','independent_qualification_run'))
 with np.load(F/'selected_teacher.npz',allow_pickle=False) as z:
  np.testing.assert_array_equal(z['coefficients'],proposals[selected])
  basis=np.sin(np.pi*np.arange(529)[:,None]/528*np.arange(1,5)[None,:]);basis[[0,-1]]=0.
  np.testing.assert_array_equal(z['basis'],basis)
 live={r.name:sum(p.stat().st_size for p in r.rglob('*') if p.is_file()) for r in (F,S,L)}
 assert 3*sum(live.values())<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
 return dict(terminal_result_saved=True,natural_exit=True,formal_attempts=181,independent_attempts=6,controls=controls,full_length_attempts=sum(r['controls']==2279 for r in rows),invalid=invalid,rows=rows,retention=evidence,selected_program=selected,proposal_rng_independently_equal=True,closed_programs=5,source_manifests_verified=manifests,R157_full_old_fields_parity_attempts=parity_count,independent_smoke_full_array_equal=True,final_25_full_array_equal=True,live_bytes=live,source_inventory={r.name:inventory(r) for r in (F,S,L)},new_dynamic_attempts=0,full_scalar_planning_audit=False,full_task_completed=False,hardware_readiness=False,qualification=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--verify',action='store_true');a=p.parse_args();idle(a.base)
 proof=verify()
 if a.verify:
  assert not PROOF.exists();write(PROOF,proof)
  print(json.dumps({k:v for k,v in proof.items() if k not in ('rows','source_inventory')},indent=2));return
 assert read(PROOF)==proof
 before_git=sum(p.stat().st_size for p in (REPO/'.git').rglob('*') if p.is_file())
 for root in (S,F,L):
  target=REPO/'results'/root.name/'terminal_snapshot';assert not target.exists()
  shutil.copytree(root,target);assert inventory(root)==proof['source_inventory'][root.name]==inventory(target)
  log=root.with_suffix('.log') if root!=L else ROOT/'tmp/getup_shared_horizon_teacher_r193_launcher_20261010.log'
  shutil.copy2(log,target/'process.log')
  write(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,full_task_completed=False,hardware_readiness=False,qualification=False,historical_gaps_recovered=False))
  write(target/'artifact_manifest.json',inventory(target))
  subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
 failure=ROOT/'tmp/getup_shared_horizon_teacher_r193_archive_verification_failure_01.json'
 target=REPO/'results'/F.name/'archive_verification_failure_01.json';assert not target.exists();shutil.copy2(failure,target)
 subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
 target=REPO/'results'/F.name/'terminal_verification.json';assert not target.exists();shutil.copy2(PROOF,target)
 subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
 names=('archive_getup_shared_horizon_teacher_terminal_r193.py','shared_horizon_teacher_kernel_r193.py','train_getup_shared_horizon_teacher_r193.py','test_getup_shared_horizon_teacher_r193.py','launch_getup_shared_horizon_teacher_r193.py',Path(__file__).name)
 for name in names:
  target=REPO/'scripts'/name;assert not target.exists();shutil.copy2(Path(__file__).with_name(name),target)
  subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
 wrapper='publish_getup_shared_horizon_teacher_terminal_r193_20261010.py'
 target=REPO/'scripts'/wrapper;assert not target.exists();shutil.copy2(ROOT/'tmp'/wrapper,target)
 subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
 for name in ('GETUP_SHARED_HORIZON_TEACHER_PLAN_R193_20261010.md','GETUP_SHARED_HORIZON_TEACHER_TERMINAL_R193_20261010.md'):
  assert not (REPO/name).exists();shutil.copy2(ROOT/'tmp'/name,REPO/name)
  subprocess.run(['git','add',name],cwd=REPO,check=True)
 subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R193 closed full horizon common teacher search and all failures'],cwd=REPO,check=True)
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
 after_git=sum(p.stat().st_size for p in (REPO/'.git').rglob('*') if p.is_file())
 archived=sum(p.stat().st_size for root in (S,F,L) for p in (REPO/'results'/root.name).rglob('*') if p.is_file())
 assert sum(proof['live_bytes'].values())+archived+max(0,after_git-before_git)<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
 print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),git_increment_bytes=after_git-before_git,archive_bytes=archived,live_bytes=sum(proof['live_bytes'].values()),free_bytes=shutil.disk_usage(ROOT).free)),flush=True)
if __name__=='__main__':main()

