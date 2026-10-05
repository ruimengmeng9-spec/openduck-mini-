"""Append immutable, fully verified teacher records without live-file races."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'
RUN=ROOT/'outputs/getup_case_teachers_r109_left_20261005'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    records=[]
    for folder in sorted((RUN/'existing_teachers').glob('case_*')):
        if (folder/'result.json').exists():records.append((folder,False))
    for result in sorted((RUN/'complete_teacher_checks').glob('case_*/generation_*/result.json')):
        row=json.loads(result.read_text())
        if row['qualified_teacher']:records.append((result.parent,True))
    assert len(records)>=22
    terminal=json.loads((RUN/'results.json').read_text())
    assert terminal['completed_generations']==1 and terminal['uncovered']==[]
    assert len(terminal['teachers'])==25 and len(records)==25
    target=REPO/'results'/RUN.name/'verified_teacher_snapshot_0001'
    assert not target.exists()
    verified=[];seeds=set()
    for folder,is_new in records:
        row=json.loads((folder/'result.json').read_text())
        assert row['success'] and row['qualified_teacher'] and row['valid'] and row['controls']==2279
        assert row['strict_tail_s']>=30.-1e-8 and row['entry_time_s']<=12.
        assert row['case_seed'] not in seeds;seeds.add(row['case_seed'])
        with np.load(folder/'trajectory.npz') as data:
            assert data['observations'].shape==(2279,55)
            assert data['normalized_residual'].shape==(2279,10)
            assert np.isfinite(data['observations']).all() and np.isfinite(data['normalized_residual']).all()
        verified.append(dict(case_seed=row['case_seed'],new_teacher=is_new,strict_tail_s=row['strict_tail_s'],
            entry_time_s=row['entry_time_s'],initial_hash=row['initial_hash'],source=str(folder),
            trajectory_sha256=digest(folder/'trajectory.npz')))
    target.mkdir(parents=True)
    for folder,_ in records:shutil.copytree(folder,target/f"case_{json.loads((folder/'result.json').read_text())['case_seed']}")
    for name in ('contract.json','frozen_profiles.npz'):shutil.copy2(RUN/name,target/name)
    for name in ('results.json','progress.json','history.json','parity.json'):
        shutil.copy2(RUN/name,target/name)
    shutil.copytree(RUN/'checkpoints',target/'checkpoints')
    shutil.copytree(RUN/'executed_sources',target/'executed_sources')
    shutil.copy2(RUN.with_suffix('.log'),target/'parent.log')
    scope=dict(records=verified,teacher_data_only=True,unified_policy_success=False,
        independent_qualification_run=False,full_task_completed=False,hardware_readiness=False,
        complete_search_terminal=True)
    (target/'snapshot.json').write_text(json.dumps(scope,indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f)
        for f in target.rglob('*') if f.is_file()},indent=2))
    doc='GETUP_CASE_TEACHERS_VERIFIED_SNAPSHOT_R109_20261005.md'
    assert not (REPO/doc).exists()
    text='# R109 已完整复测的教师数据快照\n\n'
    text+=f'本快照保存 {len(records)} 条标准或既有开发起点的完整成功教师轨迹，其中新增 {sum(x["new_teacher"] for x in verified)} 个两套固定控制此前都失败的案例。'
    text+='每条均从原完整倒地起点运行 2279 步，满足原 12 秒进入期限、连续严格站稳至少 30 秒及原物理审计有效。\n\n'
    text+='搜索第一代就补齐五个此前未覆盖开发案例，因全部教师找到而结束，未用满24代预算。保存完整检查点、搜索历史与随机数状态。这些是逐案例优化得到的教师数据，不是统一策略成绩或未见资格验证；不得按案例编号在部署时选择轨迹。下一阶段需要训练只读取当前观测及相位（如使用起始传感器上下文，必须是因果采样而非案例编号）的统一策略，并重新完成完整闭环验证。\n\n'
    text+='| 案例 | 新找到的教师 | 连续严格站稳秒 | 首次进入秒 |\n| --- | --- | --- | --- |\n'
    for row in verified:text+=f'| {row["case_seed"]} | {row["new_teacher"]} | {row["strict_tail_s"]:.2f} | {row["entry_time_s"]:.2f} |\n'
    (REPO/doc).write_text(text)
    name=Path(__file__).name;shutil.copy2(Path(__file__),REPO/'scripts'/name)
    subprocess.run(['git','add',doc,'scripts/'+name],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve fully verified complete-fall teacher dataset snapshot R109'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_verified_teachers_r109_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        records=len(records),new_teacher_cases=[r['case_seed'] for r in verified if r['new_teacher']])),flush=True)


if __name__=='__main__':main()
