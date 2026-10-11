$ErrorActionPreference='Stop'
$taskDestination='C:/Users/86185/.codex/visualizations/2026/09/03/01a064da-da1b-7163-9ac6-b555b9e6d159/getup_r202_20261011'
$taskServerHashes=@'
import pathlib,hashlib,json
root=pathlib.Path("/data/shijinsheng/open_duck")
pairs=[["tmp/getup_total_target_vmpo_r202b_numeric_exit_20261011.json","numeric_b_exit_20261011.json"],["tmp/getup_total_target_vmpo_r202b_numeric_20261011.log","numeric_b_wrapper_20261011.log"],["outputs/getup_total_target_vmpo_numeric_r202b_20261011/result.json","numeric_b_result_20261011.json"],["outputs/getup_total_target_vmpo_numeric_r202b_20261011/regression.log","numeric_b_full_regression_20261011.log"],["tmp/getup_total_target_vmpo_r202_preflight_exit_20261011.json","preflight_exit_20261011.json"],["outputs/getup_total_target_vmpo_r202_preflight_20261011/regression.log","preflight_full_regression_20261011.log"],["tmp/getup_total_target_vmpo_r202_launch_request_20261011.json","launch_request_exit_20261011.json"],["outputs/getup_total_target_vmpo_launcher_r202_20261011/initial/regression.log","original_launcher_initial_skipped_20261011.log"],["outputs/getup_total_target_vmpo_launcher_r202_20261011/formal/regression.log","original_launcher_formal_skipped_20261011.log"],["outputs/getup_total_target_vmpo_r202_smoke_20261011/results.json","smoke_results_20261011.json"],["outputs/getup_total_target_vmpo_r202_smoke_20261011.log","smoke_full_20261011.log"],["outputs/getup_total_target_vmpo_r202_left_20261011.log","formal_interrupted_full_20261011.log"],["outputs/getup_total_target_vmpo_r202_left_20261011/development_baseline/results.json","formal_baseline_saved_20261011.json"],["outputs/getup_total_target_vmpo_r202_left_20261011/training_0001/results.json","formal_training1_saved_20261011.json"],["outputs/getup_total_target_vmpo_r202_left_20261011/training_0001/case_769002/result.json","formal_training1_invalid_769002_saved_20261011.json"],["tmp/getup_total_target_vmpo_r202_user_authorized_task_stop_20261011.json","user_authorized_task_stop_20261011.json"],["tmp/getup_total_target_vmpo_r202b_gate_repair_exit_20261011.json","gate_repair_exit_20261011.json"],["outputs/getup_total_target_vmpo_launcher_r202b_20261011/result.json","gate_repair_result_20261011.json"],["outputs/getup_total_target_vmpo_launcher_r202b_20261011/complete_gate/regression.log","gate_repair_full_regression_20261011.log"],["tmp/getup_total_target_vmpo_r202_interrupted_gate_repaired_20261011.json","interrupted_gate_repaired_20261011.json"]]
records=[]
for remote,local in pairs:
 p=root/remote;assert p.is_file(),str(p)
 records.append(dict(remote=str(p),local=local,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
print(json.dumps(records))
'@
$taskHashOutput=$taskServerHashes | ssh -o BatchMode=yes -o ConnectTimeout=15 5090x8 '/data/shijinsheng/open_duck/projects/Open_Duck_Playground/.venv/bin/python -'
if ($LASTEXITCODE -ne 0) {throw 'Remote hash inventory failed'}
$taskEntries=($taskHashOutput -join "`n") | ConvertFrom-Json
foreach($taskEntry in $taskEntries) {
  $taskTarget=Join-Path $taskDestination $taskEntry.local
  if(Test-Path -LiteralPath $taskTarget){throw "Existing destination: $taskTarget"}
}
function Count-TaskLeaves($value) {
  if($null -eq $value){return 1}
  if($value -is [System.Management.Automation.PSCustomObject]){
    $taskN=0;foreach($prop in $value.PSObject.Properties){$taskN+=Count-TaskLeaves $prop.Value};return $taskN
  }
  if($value -is [System.Array]){$taskN=0;foreach($item in $value){$taskN+=Count-TaskLeaves $item};return $taskN}
  return 1
}
$taskReadRecords=@()
foreach($taskEntry in $taskEntries){
  $taskTarget=Join-Path $taskDestination $taskEntry.local
  scp -o BatchMode=yes -o ConnectTimeout=15 ("5090x8:"+$taskEntry.remote) $taskTarget
  if($LASTEXITCODE -ne 0){throw "Download failed: $($taskEntry.local)"}
  $taskActual=(Get-FileHash -LiteralPath $taskTarget -Algorithm SHA256).Hash.ToLowerInvariant()
  if($taskActual -ne $taskEntry.sha256){throw "SHA mismatch: $($taskEntry.local)"}
  $taskText=Get-Content -LiteralPath $taskTarget -Raw -Encoding UTF8
  $taskLeaves=0
  if($taskEntry.local.EndsWith('.json')){$taskLeaves=Count-TaskLeaves ($taskText|ConvertFrom-Json)}
  $taskReadRecords+= [PSCustomObject]@{file=$taskTarget;remote=$taskEntry.remote;sha256=$taskActual;bytes=$taskEntry.bytes;characters=$taskText.Length;json_leaves=$taskLeaves;raw_bytes_saved=$true}
}
$taskReadRecords | ConvertTo-Json -Depth 5 -Compress

