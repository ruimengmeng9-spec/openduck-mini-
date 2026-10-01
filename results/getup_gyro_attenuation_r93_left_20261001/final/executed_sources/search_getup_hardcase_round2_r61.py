"""Second independent hard-case refinement round built on the R59 winner."""
import json
from pathlib import Path
import sys

from diagnostics.getup_independent_native import digest
import diagnostics.search_getup_hardcase_refine_r59 as search


# Seven R60 long-hold failures plus five successes to limit forgetting.
search.TRAIN_SEEDS = (761004, 761017, 761019, 761020, 761021, 761025,
                      761030, 761000, 761001, 761002, 761003, 761005)
search.HELDOUT_SEEDS = tuple(range(762000, 762040))


def main():
    search.main()
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    path = output / 'results.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['stage'] = ('captured transition hard-case refinement round 2; '
                       'not full recovery')
    report['driver_source_sha256'] = digest(__file__)
    report['parent_algorithm_source_sha256'] = report['source_sha256']
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
