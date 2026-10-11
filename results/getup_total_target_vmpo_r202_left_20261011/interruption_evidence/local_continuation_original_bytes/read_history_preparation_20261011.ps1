$ErrorActionPreference = 'Stop'
$taskRoot = 'C:/Users/86185/.codex/visualizations/2026/09/03/01a064da-da1b-7163-9ac6-b555b9e6d159'
$catalog = Get-Content -Raw -LiteralPath "$taskRoot/getup_r201_20261011/evidence_preparation_numeric_20261011.json" | ConvertFrom-Json
$paths = @($catalog.files.path)
$paths += @(Get-ChildItem -File -LiteralPath "$taskRoot/getup_r201_20261011" | ForEach-Object { $_.FullName })
$paths += "$taskRoot/open_duck_storage_sparse_20261011/all_archives_complete_20261011.json"
function Count-Leaves($v) {
    if ($null -eq $v) { return 1 }
    if ($v -is [System.Management.Automation.PSCustomObject]) {
        $n = 0; foreach ($p in $v.PSObject.Properties) { $n += Count-Leaves $p.Value }; return $n
    }
    if ($v -is [array]) { $n = 0; foreach ($p in $v) { $n += Count-Leaves $p }; return $n }
    return 1
}
$rows = @(); $characters = 0; $leaves = 0
foreach ($p in ($paths | Sort-Object -Unique)) {
    $content = Get-Content -Raw -LiteralPath $p -Encoding UTF8
    $n = $null
    if ([IO.Path]::GetExtension($p) -eq '.json') { $n = Count-Leaves ($content | ConvertFrom-Json); $leaves += $n }
    $characters += $content.Length
    $rows += [ordered]@{ path=$p; sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower(); characters=$content.Length; parsed_leaves=$n }
}
[ordered]@{ operation='full-file-read-and-json-traversal-only'; files=$rows; file_count=$rows.Count; characters=$characters; json_leaves=$leaves; original_evidence_written=$false; old_verifier_executed=$false; new_dynamic=0; rng_values_not_reserialized=$true } | ConvertTo-Json -Depth 12 -Compress
