"""Append-only R205 archive: lossless evidence checks only, no mathematical re-execution."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
BASE='d485cfc1c8eb31a4762ad5b711f8ae4d6fde6866'
CAP=4294967296
MIB=1024**2
PROOF=ROOT/'tmp/getup_total_target_cacla_r205b_terminal_verification_20261012.json'
PROOF_SHA='06e94cf49e9203879aa02114a8fbb9c3807d3435c35e0cd421685fcb047dfb86'
EXIT=ROOT/'tmp/getup_total_target_cacla_r205b_terminal_verification_exit_20261012.json'
VERIFY=ROOT/'outputs/getup_total_target_cacla_r205b_terminal_verification_20261012'
FAILED=ROOT/'outputs/getup_total_target_cacla_r205_terminal_verification_20261011'
NUMSRC=ROOT/'tmp/getup_total_target_cacla_numeric_r205_20261011'
FORMAL=ROOT/'outputs/getup_total_target_cacla_r205_left_20261011'
CONT=ROOT/'tmp/getup_total_target_cacla_r205_continuation_20261012'
CONT_MANIFEST=ROOT/'tmp/getup_total_target_cacla_r205_continuation_manifest_20261012.json'
TERMINAL='GETUP_TOTAL_TARGET_CACLA_TERMINAL_R205_20261012.json'
PLAN='GETUP_TOTAL_TARGET_CACLA_PLAN_R205_20261011.json'
RESULT=ROOT/'tmp/getup_total_target_cacla_r205_archive_result_20261012.json'
ARCHIVE_LOG=ROOT/'tmp/getup_total_target_cacla_r205_archive_20261012.log'
ARCHIVE_EXIT=ROOT/'tmp/getup_total_target_cacla_r205_archive_exit_20261012.json'
PUBLISHER=ROOT/'tmp/publish_getup_total_target_cacla_terminal_r205_20261012.py'
PREP=ROOT/'tmp/getup_total_target_cacla_r205_archive_preparation_20261012.json'
PROTECTED=2664890
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def enc(v):return json.dumps(v,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def inventory(p):
    p=Path(p)
    assert p.is_dir() and not p.is_symlink(),p
    files=sorted(f for f in p.rglob('*') if f.is_file())
    assert all(not f.is_symlink() for f in files),p
    return {f.relative_to(p).as_posix():digest(f) for f in files}
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def size(p):
    p=Path(p)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=REPO,text=True).strip()
def idle():
    assert git('rev-parse','HEAD')==BASE
    assert not git('status','--porcelain')
    assert (Path('/proc')/str(PROTECTED)/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()=='.venv/bin/python -u -m open_duck_agent.live_server --host 127.0.0.1 --port 18766'
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        tokens=line.split();first=Path(tokens[0]).name if tokens else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(t in line for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),line
        assert 'publish_getup' not in line and 'git push' not in line,line
    for d in (ROOT/'tmp',ROOT/'outputs',PROJECT/'diagnostics'):
        assert not any(re.search(r'r2(?:0[6-9]|[1-9][0-9])',p.name.lower()) for p in d.iterdir()),d
def decode(v):
    assert v['encoding'].startswith('lossless_trie_v1:')
    names=v['names'];hashes=v['sha256'];nodes=v['nodes'];flat={}
    def walk(i,prefix,out,ancestors):
        assert i not in ancestors
        files,children=nodes[i]
        for n,h in files:
            key=prefix+names[n]
            assert key not in out
            out[key]=hashes[h]
        for n,j in children:walk(j,prefix+names[n]+'/',out,ancestors|{i})
    for root,node in v['roots'].items():
        out={};walk(node,'',out,set());flat[root]=dict(sorted(out.items()))
    assert sum(map(len,flat.values()))==v['original_file_count']==10815
    assert hashlib.sha256(enc(flat)).hexdigest()==v['full_flat_canonical_sha256']=='6be63385b4870ccc9a45b29306572d57aa8ae844cfe240804158aafa85ece0c9'
    for root,items in flat.items():
        if root=='external_original_files':
            assert all(digest(p)==sha for p,sha in items.items())
        else:assert inventory(root)==items,root
    return flat
def check_sources(manifest,folder):
    for p,v in manifest.items():assert digest(p)==v['sha256']==digest(Path(folder)/v['copy']),p
def completed():
    assert digest(PROOF)==PROOF_SHA
    proof=read(PROOF);e=read(EXIT)
    assert e['actual_subprocess_return_observed'] and e['natural_exit'] and e['exit_code']==0 and e['source_unchanged'] and e['proof_saved'] and e['proof_sha256']==PROOF_SHA
    assert e['source_sha256']==proof['verifier_source_sha256']==digest(PROJECT/'diagnostics/verify_getup_total_target_cacla_terminal_r205b.py')=='7cdfc7f2615389c57200dea8f73fb4ea98fc0797705fc71409f617e0f1e2ae2c'
    assert e['wrapper_sha256']==digest(ROOT/'tmp/terminal_verification_wrapper_r205b_20261012.py')=='526fac79f21931738b70c84a8d3313dbc008f6e15a4e277c9a345ba0e8c9530f'
    assert proof['audit_complete'] and proof['closed_saved_paths']==162 and proof['controls']==367222
    assert proof['new_dynamic_attempts']==0 and proof['dynamic162_budget_closed'] and proof['formal_launcher_actual_natural_exit0']
    assert proof['fallback_successes']==18 and proof['deterministic_check_successes']==[7,7] and proof['common_eligible']==[False,False] and proof['selected_checkpoint'] is None
    assert len(proof['independent_CACLA_rounds'])==2 and proof['all_original_128_raw_clipped_gradients_norm_SGD_beforeafter_weights_counts_rng_exact']
    assert not proof['full_task_completed'] and not proof['hardware_readiness']
    original=decode(proof['source_inventory'])
    assert len(proof['source_manifest_counts'])==188
    for row in proof['source_manifest_counts']:
        p=Path(row['path']);assert digest(p)==row['sha256']
        if row['kind']=='actual_contract_startup':manifest=read(p)['sources'];folder=p.parent/'executed_sources'
        else:
            assert row['kind']=='actual_separate_manifest'
            manifest=read(p);folder=Path(row['copy_directory'])
        assert len(manifest)==row['sources'];check_sources(manifest,folder)
    for name,count,sha in [('source_manifest.json',182,'8940ebf24005482c2a8c269355bcff41dbd53f182bddba0ade5475aab41dcd3f'),('runtime_source_manifest.json',244,'7a82bdbd20125004e38bace68d8b7070d33b0590eb91699c0251d6a8280cf2a7')]:
        p=VERIFY/name;assert digest(p)==sha;manifest=read(p);assert len(manifest)==count;check_sources(manifest,VERIFY/'executed_sources')
    assert not (VERIFY/'failure.json').exists()
    assert not (FAILED/'runtime_source_manifest.json').exists()
    assert not (ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_20261011.json').exists()
    a=read(ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_exit_20261011.json')
    assert a['natural_exit'] and a['exit_code']==1 and not a['proof_saved']
    assert digest(NUMSRC/PLAN)=='2d36c19fda9c44740cbd9e476ea30ed72dd20332a2f021327af5c9985e9a452a'
    return proof,original
def sources():return sorted(p for p in (PROJECT/'diagnostics').iterdir() if p.is_file() and 'r205' in p.name.lower() and p.suffix=='.py')
def live():
    files=set(sources())
    for d in (ROOT/'tmp',ROOT/'outputs'):
        for p in d.iterdir():
            if 'r205' in p.name.lower():
                files.add(p) if p.is_file() else files.update(f for f in p.rglob('*') if f.is_file())
    return sum(f.stat().st_size for f in files)
def assets():
    result={}
    for folder in (FORMAL,ROOT/'outputs/getup_total_target_cacla_r205_smoke_20261011'):
        c=read(folder/'contract.json')
        for p,h in {**c['model_files'],**c['hashes']}.items():
            assert digest(p)==h
            if not Path(p).is_relative_to(folder):result[p]=h
    assert len(result)==32
    return dict(sorted(result.items()))
def gate(archive_bound,additional=0,start=False):
    live_bytes=live()
    archive_bound=max(640*MIB,archive_bound)
    git_bound=max(640*MIB,2*archive_bound+64*MIB)
    total=live_bytes+archive_bound+git_bound+128*MIB+16*MIB+additional
    free=shutil.disk_usage(ROOT).free
    assert total<CAP,(live_bytes,archive_bound,git_bound,total)
    assert free>=(12 if start else 10)*1024**3
    assert free-additional-archive_bound-git_bound>=10*1024**3
    return dict(live_bytes=live_bytes,archive_or_future_bound_bytes=archive_bound,git_loose_plus_future_temporary_pack_index_ref_bound_bytes=git_bound,extra_reserved_bytes=128*MIB,local_future_continuation_bound_bytes=16*MIB,additional_bound_bytes=additional,total_all_inclusive_bound_bytes=total,cap_bytes=CAP,free_bytes=free)
def dirs_and_inventory(proof,original):
    dirs=[Path(p) for p in original if p!='external_original_files']+[VERIFY]
    assert len(dirs)==8 and NUMSRC in dirs
    return dirs,{str(p):inventory(p) for p in dirs}
def prepare():
    idle()
    assert not (ROOT/'tmp'/TERMINAL).exists() and not RESULT.exists() and not PREP.exists() and not CONT.exists()
    proof,original=completed();dirs,before=dirs_and_inventory(proof,original);asset=assets()
    r=read(FORMAL/'results.json')
    assert r['terminal_result_saved'] and r['actual_dynamic_attempts']==156 and r['selected_R157_fallback'] and r['selected_checkpoint'] is None
    planned=live()+sum(Path(p).stat().st_size for p in asset)+64*MIB
    resource=gate(planned,additional=16*MIB,start=True)
    omit={'initial','independent_CACLA_rounds','rows','source_manifest_counts','source_inventory','resources'}
    summary={k:v for k,v in proof.items() if k not in omit}
    document=dict(date='2026-10-12',stage='R205_completed_no_improvement_independent_B_proof_archive_preparation',saved_paths=162,controls=367222,formal_saved_paths=156,independent_smoke_paths=6,dynamic_budget_closed=True,new_dynamic_attempts=0,actor_updates=128,value_updates=128,current_round_training_only=True,training_successes=[8,11],training_standard_successes=[False,False],training_invalid_counts=[1,0],deterministic_successes=[7,7],common_eligible=[False,False],selected_checkpoint=None,complete_R157_fallback_successes=18,final_standard_success=True,final_invalid_count=0,best='R157 18/24',new_improvement=False,original_invalid=read(FORMAL/'training_0001/case_769007/result.json'),failed_A_source_contract_KeyError_preserved=True,failed_A_exit_sha256=digest(ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_exit_20261011.json'),failed_A_proof_and_runtime_manifest_missing_not_fabricated=True,independent_proof_summary=summary,completed_full_proof_file=str(PROOF),completed_full_proof_sha256=PROOF_SHA,actual_B_exit_sha256=digest(EXIT),full_lossless_original_inventory_canonical_sha256=proof['source_inventory']['full_flat_canonical_sha256'],full_lossless_original_inventory_file_count=10815,original_source_contract_or_manifest_count=188,B_startup_source_count=182,B_complete_runtime_source_count=244,full_162_decisions_128_gradients_checkpoints_0_1_2_raw_RNG_in_separate_complete_proof=True,archive_source_sha256={str(p):digest(p) for p in sources()},original_before_request_full_IMU_double_gap_preserved=True,first_joint_slew_independent_planning_gap_preserved=True,qpos_qvel_only_isolated_equality=True,physical_safety_Markov_cross_implementation_bitwise_convergence_monotonic_improvement_claim=False,full_task_completed=False,hardware_readiness=False,qualification=False,unread320_not_read=True,hardware_not_connected=True,protected_live_server_not_operated=True,source_plan_immutable=True,storage_cap_bytes=CAP,storage_authorization_sha256=digest(ROOT/'tmp/r205_storage_cap_authorized_4gib_20261012.json'),immutable_preparation_archive_complete=False,immutable_preparation_publication_complete=False,publication_requires_actual_exit_independent_no_push_and_explicit_commit_full_GitHub_text_match=True,resource_preparation_snapshot=resource)
    write(ROOT/'tmp'/TERMINAL,document)
    write(PREP,dict(terminal_sha256=digest(ROOT/'tmp'/TERMINAL),full_proof_sha256=PROOF_SHA,directory_inventory_sha256={str(p):hashlib.sha256(enc(before[str(p)])).hexdigest() for p in dirs},assets=asset,resources=resource,archive_source_sha256=digest(__file__),new_dynamic_attempts=0))
    print(json.dumps(dict(preparation_saved=True,terminal_sha256=digest(ROOT/'tmp'/TERMINAL),terminal_bytes=size(ROOT/'tmp'/TERMINAL),preparation_sha256=digest(PREP),resource=resource,new_dynamic_attempts=0)),flush=True)
def copy(p,target):
    p=Path(p);target=Path(target);assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert digest(p)==digest(target)
def add(target):subprocess.run(['git','-c','gc.auto=0','add','-f','--',str(target.relative_to(REPO))],cwd=REPO,check=True)
def archive():
    idle();assert not RESULT.exists() and CONT.is_dir() and CONT_MANIFEST.is_file()
    proof,original=completed();dirs,before=dirs_and_inventory(proof,original);prep=read(PREP);asset=assets()
    assert asset==prep['assets'] and prep['archive_source_sha256']==digest(__file__)
    assert prep['terminal_sha256']==digest(ROOT/'tmp'/TERMINAL)==digest(CONT/TERMINAL)
    assert digest(CONT/'terminal_verification_b_proof_20261012.json')==PROOF_SHA
    continuation=inventory(CONT);assert continuation==read(CONT_MANIFEST)['files']
    assert sum((CONT/p).stat().st_size for p in continuation)<16*MIB
    for p in dirs:assert hashlib.sha256(enc(before[str(p)])).hexdigest()==prep['directory_inventory_sha256'][str(p)]
    doc=read(ROOT/'tmp'/TERMINAL)
    for p,h in doc['archive_source_sha256'].items():assert digest(p)==h
    outputs=[p for p in dirs if p!=NUMSRC];assert len(outputs)==7
    for p in outputs:assert not (REPO/'results'/p.name).exists()
    names=sources();assert Path(__file__) in names
    for p in names:assert not (REPO/'scripts'/p.name).exists()
    for name in (TERMINAL,PLAN,PUBLISHER.name):assert not (REPO/('scripts/'+name if name==PUBLISHER.name else name)).exists()
    tmpfiles=sorted(p for p in (ROOT/'tmp').iterdir() if p.is_file() and 'r205' in p.name.lower() and p not in (ARCHIVE_LOG,ARCHIVE_EXIT,RESULT))
    logs=sorted(p for p in (ROOT/'outputs').iterdir() if p.is_file() and 'r205' in p.name.lower())
    external={str(p):digest(p) for p in [*tmpfiles,*logs]}
    predicted=sum(size(p) for p in dirs)+sum(Path(p).stat().st_size for p in asset)+size(CONT)+sum(size(p) for p in [*tmpfiles,*logs,*names,PUBLISHER,NUMSRC/PLAN,ROOT/'tmp'/TERMINAL])+32*MIB
    resource_before=gate(predicted,start=True)
    print('R205_APPEND_ONLY_ARCHIVE_BEGIN',json.dumps(resource_before),flush=True)
    targets=[]
    for p in outputs:
        root=REPO/'results'/p.name
        shutil.copytree(p,root/'terminal_snapshot');assert inventory(root/'terminal_snapshot')==before[str(p)]
        write(root/'snapshot_metadata.json',dict(original_R205_terminal_bytes_preserved=True,completed_B_proof_sha256=PROOF_SHA,new_dynamic_attempts=0,original_A_failure_invalids_source_gaps_preserved=True,full_task_completed=False,hardware_readiness=False))
        targets.append(root)
    numeric=REPO/'results'/outputs[0].name/'numeric_source_terminal_snapshot'
    shutil.copytree(NUMSRC,numeric);assert inventory(numeric)==before[str(NUMSRC)]
    record=REPO/'results'/FORMAL.name/'terminal_evidence';record.mkdir()
    for p in [*tmpfiles,*logs]:
        copy(p,record/'external_files'/p.relative_to(ROOT))
    for p,h in original['external_original_files'].items():
        target=record/'original_proof_external_files'/Path(p).relative_to(ROOT);copy(p,target);assert digest(target)==h
    asset_mapping={}
    for p,h in asset.items():
        target=record/'full_model_and_reference_assets'/Path(p).relative_to(ROOT);copy(p,target);assert digest(target)==h
        asset_mapping[p]=dict(sha256=h,archive_relative_path=target.relative_to(REPO).as_posix())
    shutil.copytree(CONT,record/'local_continuation_original_bytes');assert inventory(record/'local_continuation_original_bytes')==continuation
    write(record/'full_asset_manifest.json',asset_mapping)
    write(record/'artifact_manifest.json',inventory(record))
    for p in names:
        target=REPO/'scripts'/p.name;copy(p,target);targets.append(target)
    target=REPO/'scripts'/PUBLISHER.name;copy(PUBLISHER,target);targets.append(target)
    for name,p in [(TERMINAL,ROOT/'tmp'/TERMINAL),(PLAN,NUMSRC/PLAN)]:
        target=REPO/name;copy(p,target);targets.append(target)
    archive_bytes=sum(size(p) for p in targets)
    assert archive_bytes<predicted
    gate(archive_bytes)
    for target in targets:add(target)
    staged=git('diff','--cached','--name-only').splitlines()
    allowed=[p.relative_to(REPO).as_posix()+('/' if p.is_dir() else '') for p in targets]
    assert staged and all(any(n.startswith(a) if a.endswith('/') else n==a for a in allowed) for n in staged)
    assert before=={str(p):inventory(p) for p in dirs}
    assert all(digest(p)==h for p,h in external.items()) and inventory(CONT)==continuation
    assert git('rev-parse','HEAD')==BASE
    gate(archive_bytes)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R205 CACLA 162 paths, 128 updates, failed A, full independent B proof and R157 fallback'],cwd=REPO,check=True)
    head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==BASE and not git('status','--porcelain')
    object_ids=[line.split()[0] for line in git('rev-list','--objects',head,'^'+BASE).splitlines()]
    loose=[REPO/'.git/objects'/h[:2]/h[2:] for h in object_ids]
    assert all(p.is_file() for p in loose)
    reachable=sum(size(p) for p in loose)
    index_bytes=size(REPO/'.git/index')
    temp_packs=sum(size(p) for p in (REPO/'.git/objects/pack').iterdir() if p.name.startswith(('tmp_','tmp-')))
    accounted_git=reachable+index_bytes+temp_packs+MIB
    resource_after=gate(archive_bytes,additional=64*MIB)
    assert accounted_git<resource_after['git_loose_plus_future_temporary_pack_index_ref_bound_bytes']
    assert before=={str(p):inventory(p) for p in dirs} and all(digest(p)==h for p,h in external.items()) and inventory(CONT)==continuation
    for p,h in asset.items():assert digest(p)==h==digest(REPO/asset_mapping[p]['archive_relative_path'])
    result=dict(commit=head,parent=BASE,repository_clean=True,completed_proof_sha256=PROOF_SHA,terminal_sha256=digest(ROOT/'tmp'/TERMINAL),archive_source_sha256=digest(__file__),independent_verifier_A_or_B_not_reexecuted=True,new_dynamic_attempts=0,source_hashes_unchanged=True,original_invalid_A_failure_and_all_source_planning_gaps_preserved=True,saved_paths=162,controls=367222,actor_updates=128,value_updates=128,final_R157_fallback_successes=18,new_improvement=False,output_snapshot_count=7,full_model_reference_assets_count=32,local_continuation_files=len(continuation),local_continuation_bytes=size(CONT),archive_bytes=archive_bytes,new_reachable_loose_git_objects_bytes=reachable,entire_index_bytes=index_bytes,actual_temporary_pack_bytes=temp_packs,git_actual_accounted_with_1MiB_ref_log_bound_bytes=accounted_git,conservative_all_inclusive_resources_after=resource_after,resource_before=resource_before,full_task_completed=False,hardware_readiness=False,qualification=False,published=False)
    write(RESULT,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else archive()

