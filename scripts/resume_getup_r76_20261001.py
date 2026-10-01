"""Resume only the verified, user-authorized R76 simulation process family."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal

PIDS = (1886895, 1886896, 1887024, 1887025, 1887026, 1887027, 1887028, 1887031)
ROOT = Path('/data/shijinsheng/open_duck')
MODULE = b'diagnostics.search_getup_prefix_timing_r75'
OUTPUT = str(ROOT / 'outputs/getup_prefix_timing_cem_r76_20261001').encode()


def inspect(pid):
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != os.getuid():
        raise RuntimeError(f'Wrong process owner: {pid}')
    argv = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    if MODULE not in argv or OUTPUT not in argv or b'--train' not in argv:
        raise RuntimeError(f'Not the expected R76 simulation command: {pid}')
    stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] not in ('T', 't'):
        raise RuntimeError(f'Process is not stopped: {pid}: {stat[0]}')
    return {'pid': pid, 'state': stat[0], 'ppid': int(stat[1]),
            'start_ticks': stat[19],
            'command_sha256': hashlib.sha256(b'\0'.join(argv)).hexdigest()}


def main():
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Need PID handles to avoid PID reuse races')
    handles = {}
    try:
        for pid in PIDS:
            handles[pid] = os.pidfd_open(pid)
        before = [inspect(pid) for pid in PIDS]
        if before[1]['ppid'] != PIDS[0] or any(row['ppid'] != PIDS[1] for row in before[2:]):
            raise RuntimeError('Unexpected R76 parent-child relationship')
        boot = int(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                        if line.startswith('btime ')))
        expected_start = datetime(2026, 10, 1, 2, 12, 38, tzinfo=timezone.utc).timestamp()
        for row in before:
            actual_start = boot + int(row['start_ticks']) / os.sysconf('SC_CLK_TCK')
            if not expected_start - 1 <= actual_start <= expected_start + 3:
                raise RuntimeError(f"Unexpected process start time: {row['pid']}")
        log = ROOT / 'outputs/getup_prefix_timing_cem_r76_20261001.log'
        generations = [int(value) for value in re.findall(r'^GEN (\d+) ', log.read_text(), re.M)]
        now = datetime.now(timezone.utc)
        record_dir = ROOT / 'outputs' / f'getup_r76_resume_{now:%Y%m%d_%H%M%S_%f}'
        record_dir.mkdir(exist_ok=False)
        report = {'observed_at_utc': now.isoformat(), 'simulation_only': True,
                  'action': 'SIGCONT_only_no_restart_no_parameter_change',
                  'generation_before': max(generations, default=0),
                  'log_bytes_before': log.stat().st_size,
                  'processes_before': before, 'resumed_pids': [],
                  'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        record = record_dir / 'resume.json'
        record.write_text(json.dumps(report, indent=2) + '\n')
        for pid in PIDS:
            signal.pidfd_send_signal(handles[pid], signal.SIGCONT)
            report['resumed_pids'].append(pid)
            record.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'resumed_pids': report['resumed_pids'], 'record': str(record)}))
    finally:
        for fd in handles.values():
            os.close(fd)


if __name__ == '__main__':
    main()
