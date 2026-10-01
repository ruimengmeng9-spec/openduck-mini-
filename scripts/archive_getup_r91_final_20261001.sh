#!/usr/bin/env bash
set -euo pipefail
root=/data/shijinsheng/open_duck
source_repo="$root/projects/Open_Duck_Playground"
publication="$root/github/openduck-mini-"
base=60552aaa81ab1354ff36cb3fb1ec1097f82e1730
run=getup_head_tolerance_r91_left_20261001
cd "$publication"
test "$(git rev-parse HEAD)" = "$base"
test -z "$(git status --porcelain)"
test -f "$root/outputs/$run/results.json"
test ! -e "results/$run/final"
mkdir "results/$run/final"
cp -r "$root/outputs/$run/." "results/$run/final/"
cp "$root/outputs/$run.log" "results/$run/final/console.log"
cd "$source_repo"
.venv/bin/python - "$publication/results/$run/final" <<'PY'
import hashlib,json
from pathlib import Path
import sys
d=Path(sys.argv[1]);r=json.loads((d/'results.json').read_text());rows=r['rows']
assert len(rows)==17 and rows[0]['successes']==13 and rows[0]['nominal']['success']
assert sum(x['nominal']['success'] for x in rows[1:])==3
assert max(x['successes'] for x in rows[1:])==13
assert not r['promotion_allowed'] and r['no_independent_seeds']
hashes={str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in d.rglob('*') if p.is_file()}
(d/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print('R91_FINAL',len(rows),'arms; diagnostic only',flush=True)
PY
cd "$publication"
for file in GETUP_HEAD_TOLERANCE_R91_20261001.md GETUP_SHORTPATH_CONTINUATION_R69_R70_20260930.md scripts/archive_getup_r91_final_20261001.sh; do
    cp "$source_repo/$file" "$file"
    git add "$file"
done
git add -f "results/$run/final"
git commit -m "Record complete R91 getup micro-command sensitivity and retain all failed trials" | tail -n 5
git bundle create "$root/tmp/getup_r91_final_20261001.bundle" HEAD "^$base"
git rev-parse HEAD
