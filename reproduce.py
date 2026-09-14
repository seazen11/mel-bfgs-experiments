"""Cross-platform entry point. Published results are never overwritten."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parent


def verify():
    expected = json.loads((ROOT / 'SHA256SUMS.json').read_text())
    for path, digest in expected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    folder = ROOT / 'paper_results/expanded'
    rows = json.loads((folder / 'summary.json').read_text())
    assert len(rows) == 77
    count = 0
    directions = 0
    max_ratio = 0.0
    for row in rows:
        assert len(row['times']) == 3
        assert row['median'] == sorted(row['times'])[1]
        assert row['times'][row['median_run']] == row['median']
        for rep in range(3):
            name = f"{row['problem']}_{row['test']}_{row['method']}_r{rep}.json"
            run = json.loads((folder / name).read_text())
            assert run['status'] == row['statuses'][rep] == 'converged', name
            assert run['checked_gap'] <= run['tolerance'], name
            assert run['checked_gap'] == row['gaps'][rep], name
            assert run['seconds'] == row['times'][rep], name
            assert len(run['history']) - 1 == row['iterations'][rep], name
            for step in run['history']:
                if 'eta' in step:
                    ratio = step['relative_model_residual'] / step['eta']
                    assert ratio <= 1.001, name
                    max_ratio = max(max_ratio, ratio)
                    directions += 1
            count += 1
    assert count == 231
    with zipfile.ZipFile(ROOT / 'paper_results/supporting.zip') as archive:
        assert archive.testzip() is None
    print(f'PASS: {len(expected)} file hashes; 77 settings; {count} certified records.')
    print(f'PASS: {directions} nested directions; maximum residual/tolerance = {max_ratio:.6f}.')
    current = ROOT / 'paper_results/value_review'
    audits = json.loads((current / 'independent_audit.json').read_text())
    assert len(audits) == 192
    successes = 0
    for item in audits:
        run = json.loads((current / item['file']).read_text())
        assert run['status'] == item['status']
        assert abs(run['gap'] - item['independent_gap']) <= 1e-12 * max(1.0, abs(run['gap']))
        if run['status'] == 'converged':
            assert item['independent_gap'] <= run['tolerance']
            successes += 1
        else:
            assert item['independent_gap'] > run['tolerance']
        assert run['dense_fallbacks'] == 0
    assert successes == 171
    protocol = json.loads((current / 'protocol.json').read_text())
    assert hashlib.sha256((ROOT / 'experiments/value_benchmark.py').read_bytes()).hexdigest() == protocol['source_sha256']
    with zipfile.ZipFile(ROOT / 'paper_results/reproduction_20260914.zip') as archive:
        assert archive.testzip() is None
    print('PASS: 192 current terminal records: 171 successes and 21 retained failures.')
    for directory, expected_count, expected_success, source in [
        ('scale_review', 165, 156, 'scale_benchmark.py'),
        ('scale_ablation', 12, 12, 'scale_ablation.py')]:
        folder = ROOT / 'paper_results' / directory
        rows = json.loads((folder / 'summary.json').read_text())
        audit = json.loads((folder / 'independent_audit.json').read_text())
        assert len(rows) == len(audit) == expected_count
        mapping = {r['file']: r for r in rows}
        assert len(mapping) == expected_count
        assert sum(r['status'] == 'converged' for r in rows) == expected_success
        for item in audit:
            run = json.loads((folder / item['file']).read_text())
            assert run == mapping[item['file']]
            assert run['status'] == item['status']
            assert abs(run['gap'] - item['gap']) <= 1e-10 * max(1.0, abs(run['gap']))
            if run['status'] == 'converged':
                assert item['gap'] <= run['tolerance']
            else:
                assert item['gap'] > run['tolerance']
            assert run['dense_fallbacks'] == 0
            assert run['max_linear_residual'] <= 1e-7
            assert run['max_directional_ratio'] <= .5
        protocol = json.loads((folder / 'protocol.json').read_text())
        assert hashlib.sha256((ROOT / 'experiments' / source).read_bytes()).hexdigest() == protocol['source_sha256']
    print('PASS: 165 scale records (156 successes, 9 failures) and 12 successful ablations.')
    print('Stored-record checks do not rerun solvers; see docs/SCALE_BENCHMARKS.md.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['verify', 'figures', 'smoke', 'expanded', 'supporting', 'all'])
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/latest')
    args = parser.parse_args()
    if args.command == 'verify':
        verify()
        return
    output = args.output.resolve()
    protected = [ROOT / p for p in ['paper_results', 'provenance', 'experiments', 'docs', '.git']]
    if output == ROOT or any(output == p or p in output.parents or output in p.parents for p in protected):
        parser.error('Choose a separate output directory, such as runs/latest.')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MEL_OUTPUT_DIR=str(output), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    def run(script, *options):
        subprocess.run([sys.executable, str(ROOT / 'experiments' / script), *options], env=env, check=True)
    if args.command == 'figures':
        shutil.copytree(ROOT / 'paper_results/expanded', output / 'expanded_results', dirs_exist_ok=True)
        run('expanded_comparison.py', '--plots-only')
    elif args.command == 'smoke':
        run('run_experiments.py', '--smoke')
        # Exercise all four added baselines and their independent certificates.
        code = "import expanded_comparison as e; e.self_check(e.datasets()); print('PASS: four added baseline solvers')"
        subprocess.run([sys.executable, '-c', code], cwd=ROOT / 'experiments', env=env, check=True)
    else:
        if args.command in ['supporting', 'all']:
            run('run_experiments.py')
            run('analyze_results.py')
        if args.command in ['expanded', 'all']:
            run('expanded_comparison.py')
    print('Output:', output)


if __name__ == '__main__':
    main()
