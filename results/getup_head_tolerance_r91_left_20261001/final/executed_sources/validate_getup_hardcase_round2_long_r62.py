"""Long-hold validation on R61's fresh forty-seed group."""
import json
from pathlib import Path
import sys

from diagnostics.getup_independent_native import digest
import diagnostics.validate_getup_hardcase_long_r60 as validation


validation.HELDOUT_SEEDS = tuple(range(762000, 762040))


def main():
    validation.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    path = output / 'results.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['stage'] = ('captured transition round-2 long hold; '
                       'not full recovery')
    report['driver_source_sha256'] = digest(__file__)
    report['parent_validator_source_sha256'] = report['source_sha256']
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
