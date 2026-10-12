"""Standalone R205 B: independent saved mathematics, actual distinct provenance contracts.
No A import/call; immutable failed A retained; no dynamics or production fitting.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import numpy as np
import mujoco

ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
BASE='d485cfc1c8eb31a4762ad5b711f8ae4d6fde6866'
NUMSRC=ROOT/'tmp/getup_total_target_cacla_numeric_r205_20261011'
NUMERIC=ROOT/'outputs/getup_total_target_cacla_numeric_r205_20261011'
PRE=ROOT/'outputs/getup_total_target_cacla_r205_preflight_20261011'
LAUNCH=ROOT/'outputs/getup_total_target_cacla_launcher_r205_20261011'
SMOKE=ROOT/'outputs/getup_total_target_cacla_r205_smoke_20261011'
FORMAL=ROOT/'outputs/getup_total_target_cacla_r205_left_20261011'
OLD=ROOT/'outputs/getup_set_decision_r157_left_20261006/candidate'
FAILED_A=ROOT/'outputs/getup_total_target_cacla_r205_terminal_verification_20261011'
DEST=ROOT/'outputs/getup_total_target_cacla_r205b_terminal_verification_20261012'
PROOF=ROOT/'tmp/getup_total_target_cacla_r205b_terminal_verification_20261012.json'
WRAPPER=ROOT/'tmp/terminal_verification_wrapper_r205b_20261012.py'
CAP=4294967296
LOCAL_CONTINUATION_BOUND=16*1024**2
NEW_EVIDENCE_BOUND=32*1024**2
DIRECTORIES=(NUMSRC,NUMERIC,PRE,LAUNCH,SMOKE,FORMAL,FAILED_A)
CASES=[None,*range(769000,769008),*range(773000,773016)]
SPEC=[('zero_parity',[None,769000,773004]),('initial_standard',[None]),('initial_sample',[769002,773004])]
GAMMA=.99**.2
MJB='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
PHYSICS='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'
NAMES=[g+'_'+s+str(i) for g in ('actor','value') for i in (1,2,3) for s in ('w','b')]

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def enc(x):return json.dumps(x,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def new(p,x):
    data=enc(x)+b'\n';resource(len(data))
    with Path(p).open('xb') as f:f.write(data)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def eq(a,b,label=''):
    assert np.asarray(a).shape==np.asarray(b).shape and np.asarray(a).dtype==np.asarray(b).dtype,label
    np.testing.assert_array_equal(a,b,err_msg=label)
def ah(a):
    a=np.ascontiguousarray(a)
    return [list(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest()]
def bundle(a):return hashlib.sha256(enc({k:ah(v) for k,v in sorted(a.items())})).hexdigest()
def inv(d):return {str(f.relative_to(d)):digest(f) for f in sorted(d.rglob('*')) if f.is_file()}
def resource(additional=0,start=False):
    files=set()
    for d in (ROOT/'tmp',ROOT/'outputs',PROJECT/'diagnostics'):
        for p in d.iterdir():
            if 'r205' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(q for q in p.rglob('*') if q.is_file())
    live=sum(p.stat().st_size for p in files);free=shutil.disk_usage(ROOT).free
    # Retain original minima, additionally cover a complete future copy of all live evidence.
    # Git conservatively allows loose objects AND a temporary pack plus index/ref overhead.
    archive_bound=max(640*1024**2,live+LOCAL_CONTINUATION_BOUND+NEW_EVIDENCE_BOUND+16*1024**2)
    git_bound=max(640*1024**2,2*archive_bound+64*1024**2)
    future=archive_bound+git_bound+128*1024**2
    assert live+future+LOCAL_CONTINUATION_BOUND+additional<=CAP,'Complete R205 4 GiB resource gate; no cleanup or bypass'
    assert free>=(12 if start else 10)*1024**3
    return dict(live_bytes=live,future_archive_bytes=archive_bound,future_git_including_temporary_pack_index_ref_bytes=git_bound,future_extra_bytes=128*1024**2,future_archive_git_extra_bytes=future,original_future_minimum_bytes=1476395008,additional_bound_bytes=additional,local_continuation_future_upload_bound_bytes=LOCAL_CONTINUATION_BOUND,total_bytes=live+future+LOCAL_CONTINUATION_BOUND+additional,storage_cap_bytes=CAP,free_bytes=free)
def idle():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in args and any(t in args for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),args
        assert 'publish_getup' not in args and 'git push' not in args,args

def compact_inventory(original):
    # Lossless directory trie, interned names/digests, shared identical nodes.
    # Every original relative filename and full SHA remains recoverable, not a tree-only checksum.
    names=[];hashes=[];nodes=[];ni={};hi={};memo={};roots={}
    def index(v,a,m):
        if v not in m:m[v]=len(a);a.append(v)
        return m[v]
    def node(tree):
        files=[];children=[]
        for n,v in sorted(tree.items()):
            name=index(n,names,ni)
            if isinstance(v,dict):children.append([name,node(v)])
            else:files.append([name,index(v,hashes,hi)])
        data=[files,children];key=enc(data)
        if key not in memo:memo[key]=len(nodes);nodes.append(data)
        return memo[key]
    for root,entries in original.items():
        tree={}
        for path,sha in entries.items():
            parts=path.split('/');d=tree
            for part in parts[:-1]:d=d.setdefault(part,{})
            d[parts[-1]]=sha
        roots[root]=node(tree)
    result=dict(encoding='lossless_trie_v1: node=[files(name_id,sha_id),children(name_id,node_id)]',names=names,sha256=hashes,nodes=nodes,roots=roots)
    restored={}
    def expand(i,prefix,out):
        files,children=nodes[i]
        for n,h in files:out[prefix+names[n]]=hashes[h]
        for n,j in children:expand(j,prefix+names[n]+'/',out)
    for root,i in roots.items():
        out={};expand(i,'',out);restored[root]=out
    assert restored==original
    result['full_flat_canonical_sha256']=hashlib.sha256(enc(original)).hexdigest()
    result['original_file_count']=sum(map(len,original.values()))
    return result

def source_check(m,folder):
    for p,v in m.items():assert digest(p)==v['sha256']==digest(folder/v['copy']),p
    return len(m)
def capture(existing=None):
    paths={Path(__file__).resolve(),WRAPPER}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    result={} if existing is None else dict(existing);folder=DEST/'executed_sources'
    additions=[p for p in sorted(paths) if str(p) not in result]
    resource(sum(p.stat().st_size for p in additions)+120000)
    for p in additions:
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        assert not (folder/name).exists();shutil.copy2(p,folder/name)
        result[str(p)]=dict(sha256=digest(p),copy=name)
    source_check(result,folder)
    return result
def provenance():
    evidence=[]
    numeric=read(NUMERIC/'contract.json')
    assert 'sources' not in numeric and numeric['stage']=='synthetic_numeric_only'
    assert numeric['fixed_tests']==24 and numeric['new_dynamic_attempts']==0
    assert numeric['startup_and_additional_runtime_imported_python_captured']
    assert numeric['plan_sha256']==digest(NUMSRC/'GETUP_TOTAL_TARGET_CACLA_PLAN_R205_20261011.json')
    # Numeric has separate original startup and complete manifests, not contract.sources.
    startup=read(NUMERIC/'startup_source_manifest.json')
    runtime=read(NUMERIC/'source_manifest.json')
    assert len(startup)==189 and len(runtime)==190
    assert all(runtime[p]==v for p,v in startup.items())
    assert not (PRE/'contract.json').exists()
    assert read(LAUNCH/'contract.json')['total_cap']==162
    for folder in (SMOKE,FORMAL):
        c=read(folder/'contract.json')
        assert c['seed']==305 and c['total_attempt_cap']==162 and c['formal_cap']==156
        assert c['training_rounds']==2 and c['training_attempts']==50
        assert c['total_value_update_cap']==c['total_actor_update_cap']==128
        assert c['policy_features']==101 and c['total_parameters']==7771
        assert c['learning_rate']==1e-4 and c['sampling_sigma']==1e-4
        assert c['storage_budget_gib']==2 # Immutable historic training contract, not B's cap.
        assert c['reserve_gib']==10 and c['unseen_320_never_loaded']
    for folder in (NUMERIC,PRE,LAUNCH,SMOKE,FORMAL,FAILED_A):
        contract=folder/'contract.json'
        if contract.exists():
            c=read(contract)
            if folder!=NUMERIC:
                assert isinstance(c['sources'],dict) and len(c['sources'])==491
                evidence.append(dict(path=str(contract),kind='actual_contract_startup',
                    sources=source_check(c['sources'],folder/'executed_sources'),sha256=digest(contract)))
            for p,sha in {**c.get('model_files',{}),**c.get('hashes',{})}.items():assert digest(p)==sha
        for f in sorted(folder.rglob('*source_manifest*.json')):
            if 'executed_sources' in str(f):continue
            m=read(f);assert isinstance(m,dict) and m
            for v in m.values():assert set(v)=={'sha256','copy'}
            if f.name.startswith('worker_source_manifest_'):
                copy=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            elif f.parent.name.startswith('case_'):
                copies=[p for p in sorted(folder.glob('worker_executed_sources_*')) if all((p/v['copy']).is_file() for v in m.values())]
                assert copies;copy=copies[0]
                assert len(m)==504
            else:copy=f.parent/'executed_sources'
            evidence.append(dict(path=str(f),kind='actual_separate_manifest',sources=source_check(m,copy),
                copy_directory=str(copy),sha256=digest(f)))
    assert sum('/case_' in e['path'] for e in evidence)==162
    failed=read(ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_exit_20261011.json')
    assert failed['actual_subprocess_return_observed'] and failed['natural_exit'] and failed['exit_code']==1
    assert failed['source_unchanged'] and not failed['proof_saved'] and failed['new_dynamic_attempts']==0
    assert len(read(FAILED_A/'source_manifest.json'))==181
    assert not (FAILED_A/'runtime_source_manifest.json').exists()
    assert not (ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_20261011.json').exists()
    assert "KeyError: 'sources'" in read(FAILED_A/'failure.json')['traceback']
    assert failed['source_sha256']==digest(PROJECT/'diagnostics/verify_getup_total_target_cacla_terminal_r205.py')
    assert failed['wrapper_sha256']==digest(ROOT/'tmp/terminal_verification_wrapper_r205_20261011.py')
    assert failed['log_sha256']==digest(ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_20261011.log')
    return evidence

def phi(o,initial,k):return np.concatenate((np.tanh(o[:50].astype(np.float64)),np.tanh(initial[:50].astype(np.float64)),np.array([min(k/528.,1.)])))
def net(w,x,g):
    h=np.tanh(x@w[g+'_w1']+w[g+'_b1']);v=np.tanh(h@w[g+'_w2']+w[g+'_b2']);raw=v@w[g+'_w3']+w[g+'_b3']
    return (np.tanh(raw) if g=='actor' else raw),(x,h,v,raw)
def fresh():
    rng=np.random.default_rng(305);w={}
    for g,last in [('actor',10),('value',1)]:
        d=[101,32,16,last]
        for i in (1,2,3):
            w[g+'_w'+str(i)]=np.zeros((d[i-1],d[i])) if g=='actor' and i==3 else rng.normal(size=(d[i-1],d[i]))/np.sqrt(float(d[i-1]))
            w[g+'_b'+str(i)]=np.zeros(d[i])
    assert sum(v.size for v in w.values())==7771
    return w
def weights_eq(a,b):
    assert set(a)==set(b)==set(NAMES)
    for k in NAMES:eq(a[k],b[k],k)
def state(ac,vc,rc,rng):return dict(actor_count=ac,value_count=vc,round_count=rc,learning_rng=rng.bit_generator.state,adam_or_target_network=False)
def checkpoint(w,i,rng,ac,vc):
    stem=FORMAL/f'checkpoint_{i:04d}';weights_eq(load(str(stem)+'.npz'),w)
    expected=state(ac,vc,i,rng);assert read(str(stem)+'.json')==expected
    saved=read(str(stem)+'_sampling_rng.json');assert saved['no_synthetic_checkpoint_reused'] and saved['actual_learning_rng_state']==rng.bit_generator.state
    paths=[FORMAL/n/f'case_{c}/rng.json' for n,cs in SPEC for c in cs] if i==0 else [FORMAL/f'training_{i:04d}/case_{c}/rng.json' for c in CASES]
    assert saved['actual_episode_sampling_rng_records']=={str(p):dict(sha256=digest(p),actual_saved_state=read(p)) for p in paths}
    return dict(full_raw_weight_bundle_sha256=bundle(w),full_state=expected,sampling_rng_record_file_sha256=digest(str(stem)+'_sampling_rng.json'),all_original_sampling_records_exact=True)

def replay(round_no):
    path=FORMAL/f'training_{round_no:04d}';episodes=[]
    for c in CASES:
        z=load(path/f'case_{c}/trajectory.npz');r=z['original_step_reward'].astype(np.float64);o=z['observations'];n=len(o)
        ids=np.arange(1,min(529,n),dtype=np.int64);x=np.stack([phi(o[k],o[0],int(k)) for k in ids]);nx=np.zeros_like(x);boot=np.zeros(len(ids));tail=np.zeros(len(ids))
        eq(z['policy_active'],(np.arange(n)>0)&(np.arange(n)<529))
        for j,k in enumerate(ids):
            eq(x[j],z['policy_features'][k])
            if k<528 and k+1<n:nx[j]=phi(o[k+1],o[0],int(k+1));boot[j]=1.
            if k==528 and n>529:tail[j]=np.sum(r[529:]*GAMMA**np.arange(n-529,dtype=np.float64))
        row=read(path/f'case_{c}/result.json');assert not row['valid'] or n==2279
        episodes.append(dict(phi=x,nextphi=nx,actions=z['policy_requested_action'][ids].copy(),rewards=r[ids].copy(),bootstrap=boot,home_tail=tail,controls=ids,terminal=boot==0.,full_rewards=r.copy(),invalid=not row['valid']))
    keys=['phi','nextphi','actions','rewards','bootstrap','home_tail','controls','terminal']
    combined={k:np.concatenate([e[k] for e in episodes]) for k in keys}
    combined.update(episode_offsets=np.cumsum([0]+[len(e['controls']) for e in episodes],dtype=np.int64),full_reward_offsets=np.cumsum([0]+[len(e['full_rewards']) for e in episodes],dtype=np.int64),full_rewards=np.concatenate([e['full_rewards'] for e in episodes]),invalid=np.array([e['invalid'] for e in episodes],dtype=bool))
    saved=load(path/'learning/replay.npz');assert set(saved)==set(combined)
    for k,v in combined.items():eq(v,saved[k],k)
    contract=read(path/'replay_contract.json')
    for k in ['current_round_new_training_only','original_home_tail_included','invalid_original_terminal','reward_unchanged']:assert contract[k]
    assert not contract['old_data_used'] and not contract['evaluation_data_used']
    assert contract['active_samples']==len(combined['phi']) and contract['full_reward_samples']==len(combined['full_rewards'])
    assert contract['terminal_bootstrap_count']==int((combined['bootstrap']==0.).sum())==25
    return combined
def reverse(w,cache,d,g):
    x,h,v,raw=cache
    z=d*(1.-np.tanh(raw)**2) if g=='actor' else d
    result={g+'_w3':v.T@z,g+'_b3':z.sum(axis=0)}
    hidden=(z@w[g+'_w3'].T)*(1.-v*v)
    result[g+'_w2']=h.T@hidden;result[g+'_b2']=hidden.sum(axis=0)
    first=(hidden@w[g+'_w2'].T)*(1.-h*h)
    result[g+'_w1']=x.T@first;result[g+'_b1']=first.sum(axis=0)
    return result
def learning(i,w,rng,ac,vc):
    data=replay(i);path=FORMAL/f'training_{i:04d}/learning'
    weights_eq(load(path/'before.npz'),w);assert read(path/'before.json')==state(ac,vc,i-1,rng)
    batches=[]
    for step in range(64):
        original=load(path/f'update_{step:03d}.npz');meta=read(path/f'update_{step:03d}.json');assert meta['state_before']==state(ac,vc,i-1,rng)
        indices=rng.integers(0,len(data['phi']),size=512,dtype=np.int64)
        x=data['phi'][indices];nx=data['nextphi'][indices];actions=data['actions'][indices]
        value,cachev=net(w,x,'value');nv,_=net(w,nx,'value')
        target=data['rewards'][indices]+GAMMA*(data['bootstrap'][indices]*nv[:,0]+data['home_tail'][indices])
        delta=target-value[:,0];mask=delta>0.;mean,cachea=net(w,x,'actor');error=mean-actions
        raw={**reverse(w,cachea,error*mask[:,None]/512,'actor'),**reverse(w,cachev,(-delta/512)[:,None],'value')}
        clipped={};norm={};scale={}
        for g in ('actor','value'):
            keys=[g+'_'+n for n in ('w3','b3','w2','b2','w1','b1')]
            norm[g]=float(np.sqrt(sum(float(np.sum(raw[k]*raw[k])) for k in keys)))
            scale[g]=min(1.,1./norm[g]) if norm[g]>0. else 1.
            clipped.update({k:raw[k]*scale[g] for k in keys})
        updated=bool(np.any(mask));after={k:(w[k]-1e-4*clipped[k] if k.startswith('value_') or updated else w[k].copy()) for k in NAMES}
        expected=dict(indices=indices,y=target,pred=value[:,0],nextvalue=nv[:,0],delta=delta,positive=mask,mean=mean,rawmean=cachea[3],action_error=error)
        for prefix,a in [('before',w),('after',after),('raw_gradients',raw),('clipped_gradients',clipped)]:expected.update({prefix+'__'+k:v for k,v in a.items()})
        assert set(original)==set(expected)
        for k,v in expected.items():eq(v,original[k],f'round{i} batch{step} '+k)
        assert meta['actor_updated']==updated and meta['norm']==norm and meta['scale']==scale
        assert meta['value_loss']==float(.5*np.mean(delta*delta)) and meta['actor_loss']==float(.5*np.mean(np.sum(error*error*mask[:,None],axis=1)))
        ac+=int(updated);vc+=1;assert meta['state_after']==state(ac,vc,i-1,rng)
        batches.append(dict(step=step,full_original_npz_sha256=digest(path/f'update_{step:03d}.npz'),all_named_arrays_exact_bundle_sha256=bundle(expected),positive_count=int(mask.sum()),actor_updated=updated,full_state_before=meta['state_before'],full_state_after=meta['state_after'],norm=norm,scale=scale,value_loss=meta['value_loss'],actor_loss=meta['actor_loss']))
        w=after
    weights_eq(load(path/'after.npz'),w);assert read(path/'after.json')==state(ac,vc,i,rng)
    cp=checkpoint(w,i,rng,ac,vc)
    print('R205B_READONLY_ROUND_VERIFIED',i,len(data['phi']),ac,vc,flush=True)
    return w,ac,vc,dict(iteration=i,active_samples=len(data['phi']),full_reward_samples=len(data['full_rewards']),replay_arrays={k:ah(v) for k,v in data.items()},batch_array_names_and_shapes={k:[list(v.shape),str(v.dtype)] for k,v in expected.items()},batches=batches,checkpoint=cp,current_round_only=True,original_home_tail_terminal_exact=True,actor_all512_denominator_strict_sign_only=True,value_current_semi_TD_and_both_before_updates=True)

def report(path,cases):
    saved=read(path/'results.json');rows=[read(path/f'case_{c}/result.json') for c in cases];assert rows==saved['rows']
    values=dict(successes=sum(bool(r['success']) for r in rows if r['case_seed'] is not None),nominal_success=next((bool(r['success']) for r in rows if r['case_seed'] is None),None),physical_failures=sum(not r['valid'] for r in rows),return_sum=sum(r['return_sum'] for r in rows))
    for k,v in values.items():assert saved[k]==v,k
    return saved
def good(r,b):
    old={row['case_seed']:row for row in b['rows']}
    return bool(r['nominal_success'] and r['physical_failures']==0 and r['successes']>b['successes'] and all(not old[x['case_seed']]['success'] or x['success'] for x in r['rows']))

def verify():
    before={str(d):inv(d) for d in DIRECTORIES}
    external=[p for d in (ROOT/'tmp',ROOT/'outputs',PROJECT/'diagnostics') for p in d.iterdir() if p.is_file() and 'r205' in p.name.lower() and 'terminal_verification' not in p.name.lower()]
    external += [PROJECT/'diagnostics/verify_getup_total_target_cacla_terminal_r205.py',ROOT/'tmp/terminal_verification_wrapper_r205_20261011.py',ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_exit_20261011.json',ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_20261011.log']
    before['external_original_files']={str(p):digest(p) for p in sorted(external)}
    source_evidence=provenance()
    ne=read(ROOT/'tmp/getup_total_target_cacla_r205_numeric_exit_20261011.json');assert ne['natural_exit'] and ne['actual_subprocess_return_observed'] and ne['exit_code']==0 and ne['source_hashes_unchanged']
    for p,sha in ne['source_hashes'].items():assert digest(NUMSRC/p)==sha
    nr=read(NUMERIC/'result.json');assert nr['passed'] and nr['tests_run']==24 and nr['skipped']==nr['failures']==nr['errors']==0
    for folder in (PRE,LAUNCH/'initial',LAUNCH/'formal'):
        r=read(folder/'interface_result.json');assert r['tests_run']==16 and r['skipped']==r['failures']==r['errors']==0 and r['source_hashes_unchanged']
        log=(folder/'regression.log').read_text();assert 'Ran 16 tests' in log and '\nOK\n' in log and 'skipped' not in log
    for p in [LAUNCH/'initial/regression_natural_exit.json',LAUNCH/'formal/regression_natural_exit.json',LAUNCH/'smoke_natural_exit.json',LAUNCH/'formal_natural_exit.json']:
        r=read(p);assert r['natural_exit'] and r['exit_code']==0
    exitrow=read(ROOT/'tmp/getup_total_target_cacla_r205_launcher_exit_20261011.json')
    assert exitrow['actual_launcher_subprocess_return_observed'] and exitrow['natural_exit'] and exitrow['exit_code']==0 and exitrow['source_hashes_unchanged']
    for p,sha in exitrow['source_hashes'].items():assert digest(p)==sha
    assert not (FORMAL/'failure.json').exists() and not (LAUNCH/'failure.json').exists()
    rng=np.random.default_rng(305001);w=fresh();ac=vc=0
    for folder in (SMOKE,FORMAL):
        weights_eq(load(folder/'initial.npz'),w);assert read(folder/'initial.json')==state(0,0,0,rng)
    initial=checkpoint(w,0,rng,0,0);weights=[w];learn=[]
    for i in (1,2):
        w,ac,vc,e=learning(i,w,rng,ac,vc);weights.append(w);learn.append(e)
    assert ac==vc==128
    c=read(SMOKE/'contract.json');model=mujoco.MjModel.from_xml_path(str(ROOT/'training/getup_decomposed_r4/model/scene.xml'))
    buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,buffer=buffer);assert hashlib.sha256(buffer.tobytes()).hexdigest()==MJB
    legs=np.array([model.actuator(side+'_'+part).id for side in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
    head=np.array([i for i in range(14) if i not in legs]);joints=model.actuator_trnid[:,0];lower,upper=model.jnt_range[joints].T.copy();home=model.keyframe('home').qpos[model.jnt_qposadr[joints]]
    reference=load(ROOT/'outputs/getup_path_audit_r99_20261005/frozen_path.npz')['targets'];selector=load(SMOKE/'frozen/initial_selector.npz');nominal=load(SMOKE/'frozen/nominal_sensor_trajectory.npz')['observations']
    rows=[];count=controls=baseline_count=0
    for folder in (SMOKE,FORMAL):
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];n=row['controls'];group=path.parent.name;count+=1;controls+=n
            wi=1 if group in ('training_0002','check_0001') else 2 if group=='check_0002' else 0;w=weights[wi]
            weightfile=FORMAL/f'checkpoint_{wi:04d}.npz' if group in ('training_0001','training_0002','check_0001','check_0002') else folder/'initial.npz'
            assert row['policy_model_sha256']==digest(weightfile) and row['physics_sha256']==PHYSICS and row['compiled_model_sha256']==MJB and row['physics_unchanged'] and row['compiled_model_unchanged']
            rr=np.random.default_rng(305000000+row['sampling_seed']);sampling=read(path/'policy_sampling_rng_records.json');expected_records=[]
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                z={k:archive[k] for k in archive.files if k not in ('qpos','qvel')};obs=z['observations'];assert len(obs)==n
                assert z['construction_preparation_sensors'].shape==z['preparation_sensors'].shape==(40,50) and z['construction_preparation_times'].shape==(40,)
                eq(obs[1:,34:48],(z['applied'][:-1]-home).astype(np.float32))
                sensor=z['preparation_sensors'][-1].astype(float);logits=(sensor-selector['mean'])/selector['std']@selector['weight']+selector['bias'];choice=int(np.argmax(logits));gains=selector['global_gains'][choice].copy()
                assert logits.tolist()==row['initial_sensor_logits'] and choice==row['base_choice'] and gains.tolist()==row['base_gains']
                for k in range(n):
                    ids=np.array([9,10,11]);current=obs[k].astype(np.float32);nom=nominal[k].astype(np.float32)
                    position=(current[6+ids]-nom[6+ids]).astype(float)/.05;velocity=(current[20+ids]-nom[20+ids]).astype(float)/.05
                    right=-.18*np.tanh(gains[:3]*position+gains[3:]*velocity) if k<529 else np.zeros(3);eq(right,z['local_hip_extra_rad'][k])
                    active=mode!='baseline' and 0<k<529;assert bool(z['policy_active'][k])==active
                    x=np.zeros(101);mean=np.zeros(10);noise=np.zeros(10);proposal=np.zeros(10);requested=np.zeros(10);value=np.asarray(0.);raw=np.zeros(10)
                    if active:
                        x=phi(obs[k],obs[0],k);a,cache=net(w,x[None,:],'actor');mean=a[0];raw=cache[3][0];value=net(w,x[None,:],'value')[0][0,0]
                        rb=None if mode!='sample' else json.loads(json.dumps(rr.bit_generator.state))
                        if mode=='sample':noise=1e-4*rr.normal(size=10)
                        proposal=mean+noise;requested=np.clip(proposal,-1.,1.)
                        expected_records.append(dict(control=k,mode=mode,rng_before=rb,rng_after=None if mode!='sample' else json.loads(json.dumps(rr.bit_generator.state))))
                    for key,val in [('policy_features',x),('policy_raw_mean',raw),('policy_mean',mean),('policy_value',value),('policy_noise',noise),('policy_unclipped_proposal',proposal),('policy_requested_action',requested)]:eq(val,z[key][k],key)
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    if active:adjusted[legs]=np.clip(reference[k,legs]+.18*requested,lower[legs],upper[legs])
                    eq(adjusted,z['adjusted_double_pre_slew_target_rad'][k]);eq(adjusted[head],fixed[head])
                    if k:
                        prev=z['applied'][k-1];planned=np.clip(np.clip(adjusted,lower,upper),prev-5.24*.02,prev+5.24*.02);base=np.clip(np.clip(fixed,lower,upper),prev-5.24*.02,prev+5.24*.02)
                        eq(planned,z['planned_before_integration_rad'][k]);eq(base,z['same_state_unperturbed_planned_rad'][k])
                eq(z['planned_before_integration_rad'],z['applied']);assert float(np.abs(z['adjusted_double_pre_slew_target_rad']-reference[:n]).max())==row['maximum_combined_14_joint_correction_rad']
                assert abs(sum(z['original_step_reward'])-row['return_sum'])<1e-10
                assert row['success']==bool(row['valid'] and n==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8)
                oldrow=read(OLD/f"case_{row['case_seed']}/result.json");assert row['initial_hash']==oldrow['initial_hash']
                with np.load(OLD/f"case_{row['case_seed']}/trajectory.npz",allow_pickle=False) as old:
                    eq(z['applied'][0],old['applied'][0])
                    if mode=='baseline':
                        baseline_count+=1
                        for key in old.files:eq(archive[key] if key in ('qpos','qvel') else z[key],old[key],key)
                        for key in ('pose','case_seed','initial_hash','actual_fallen_start','valid','training_success','controls','strict_tail_s','entry_time_s','return_sum','final','peaks','prefix_start','root_edits_after_initialization'):assert row[key]==oldrow[key],key
            assert sampling==expected_records and rr.bit_generator.state==read(path/'rng.json')
            rows.append(dict(path=str(path.relative_to(ROOT)),case=row['case_seed'],mode=mode,controls=n,success=row['success'],valid=row['valid'],original_peaks=row['peaks'],full_final_rng=rr.bit_generator.state,all_active_full_rng_records_exact=True,active_rng_record_count=len(sampling),rng_records_original_sha256=digest(path/'policy_sampling_rng_records.json')))
            print('R205B_READONLY_PATH_VERIFIED',count,group,row['case_seed'],n,flush=True)
    assert count==162 and controls==367222 and baseline_count==56
    for folder in (SMOKE,FORMAL):
        startup=read(folder/'startup_closed.json');assert startup['smoke_gate_passed'] and startup['attempts']==6
        for name,cs in SPEC:assert report(folder/name,cs)==startup['reports'][name]
    for name,cs in SPEC:
        for case in cs:
            a=FORMAL/name/f'case_{case}';b=SMOKE/name/f'case_{case}';za=load(a/'trajectory.npz');zb=load(b/'trajectory.npz');assert za.keys()==zb.keys()
            for key in za:eq(za[key],zb[key],key)
            for key in ['result.json','rng.json','policy_sampling_rng_records.json']:assert (a/key).read_bytes()==(b/key).read_bytes()
    baseline=report(FORMAL/'development_baseline',CASES);candidate=report(FORMAL/'candidate',CASES);result=read(FORMAL/'results.json')
    assert result['baseline']==baseline and result['candidate']==candidate and result['selected_checkpoint'] is None and result['selected_R157_fallback']
    assert baseline['successes']==candidate['successes']==18 and baseline['nominal_success'] and candidate['nominal_success'] and baseline['physical_failures']==candidate['physical_failures']==0
    for case in CASES:
        a=FORMAL/'development_baseline'/f'case_{case}';b=FORMAL/'candidate'/f'case_{case}';za=load(a/'trajectory.npz');zb=load(b/'trajectory.npz');assert za.keys()==zb.keys()
        for k in za:eq(za[k],zb[k],k)
        ra=read(a/'result.json');rb=read(b/'result.json')
        for k in ra:
            if k!='sampling_seed':assert ra[k]==rb[k],k
    history=read(FORMAL/'training_closed_history.json');assert history==result['history'] and len(history)==2
    for i,h in enumerate(history,1):
        t=report(FORMAL/f'training_{i:04d}',CASES);d=report(FORMAL/f'check_{i:04d}',CASES)
        assert t==h['training'] and d==h['check'] and h['eligible']==good(d,baseline)==False
        assert t['successes']==(8 if i==1 else 11) and not t['nominal_success'] and t['physical_failures']==(1 if i==1 else 0)
        assert d['successes']==7 and d['nominal_success'] and d['physical_failures']==0
        assert h['value_updates_this_round']==64 and h['total_value_updates']==h['total_actor_updates']==64*i
        assert h['active_samples']==learn[i-1]['active_samples'] and h['complete_reward_samples']==learn[i-1]['full_reward_samples']
        data=load(FORMAL/f'training_{i:04d}/learning/replay.npz');change=float(np.abs(net(weights[i],data['phi'],'actor')[0]-net(weights[i-1],data['phi'],'actor')[0]).max());assert h['max_saved_action_change']==change
    assert not result['original_development_gate'] and result['actual_dynamic_attempts']==156
    for k in ('expanded_development_run','independent_qualification_run','qualification_passed','full_task_completed','hardware_readiness'):assert not result[k]
    after={str(d):inv(d) for d in DIRECTORIES};after['external_original_files']={str(p):digest(p) for p in sorted(external)};assert before==after
    idle()
    return dict(audit_complete=True,standalone_verifier_B=True,failed_A_preserved_and_not_executed=True,storage_cap_bytes=CAP,local_continuation_future_upload_bound_bytes=LOCAL_CONTINUATION_BOUND,new_dynamic_attempts=0,closed_saved_paths=count,controls=controls,baseline_paths_all_original_fields_exact=baseline_count,initial=initial,independent_CACLA_rounds=learn,rows=rows,source_manifest_counts=source_evidence,source_inventory=compact_inventory(before),full_model_MJB_sha256=MJB,physics_original_sha256=PHYSICS,all_source_assets_and_original_arrays_unchanged=True,all_causal101_rawmean_mean_value_scaled_iid_noise_proposal_boxrequests_full_RNG_exact=True,det_no_RNG_consumption=True,all_original_128_raw_clipped_gradients_norm_SGD_beforeafter_weights_counts_rng_exact=True,selector_right_independent_exact=True,post_IMU_right_saved_double_ten_leg_replacement_head_exact=True,recursive_joint_slew_planning_after_first_exact=True,first_actual_R157_applied_exact=True,planned_equals_applied=True,constructor_reset_each40x50=True,independent_formal6_full_arrays_results_rng_exact=True,full25_R157_fallback_exact=True,baseline_successes=18,deterministic_check_successes=[7,7],common_eligible=[False,False],selected_checkpoint=None,fallback_successes=18,original_invalid769007_303_labels_peaks_retained=True,formal_launcher_actual_natural_exit0=True,original_before_request_full_IMU_double_reconstructed=False,first_joint_slew_independent_planning_reconstructed=False,production_kernel_imported_or_called=False,production_fitting_or_old_init_worker_invoked=False,environment_MjData_forward_integrate_contact_force_relabel_invoked=False,static_MjModel_only=True,native_binary_snapshot=False,cross_implementation_bitwise_safety_Markov_convergence_monotonic_improvement_guarantee=False,all_original_traces_preserved_no_large_new_array_copy=True,dynamic162_budget_closed=True,archive_publisher_not_yet_implemented=True,best_remains_R15718=True,full_task_completed=False,hardware_readiness=False)

def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle();resource(start=True)
    assert not DEST.exists() and not PROOF.exists()
    paths={Path(__file__).resolve(),WRAPPER}
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    # Exact startup source bytes plus bounded full proof/manifests/log/exit reserve, before any output creation.
    resource(sum(p.stat().st_size for p in paths)+NEW_EVIDENCE_BOUND,start=True)
    DEST.mkdir();(DEST/'executed_sources').mkdir();startup=capture();new(DEST/'source_manifest.json',startup)
    try:
        result=verify();complete=capture(startup);new(DEST/'runtime_source_manifest.json',complete)
        result.update(verifier_source_sha256=digest(__file__),startup_python_sources=len(startup),complete_python_sources=len(complete),startup_and_runtime_original_copy_sha_exact=True,resources=resource())
        new(PROOF,result);resource()
        print('R205B_TERMINAL_VERIFICATION_COMPLETE',162,367222,digest(PROOF),flush=True)
    except Exception:
        new(DEST/'failure.json',dict(traceback=traceback.format_exc(),new_dynamic_attempts=0,proof_saved=PROOF.exists()))
        raise
if __name__=='__main__':main()
