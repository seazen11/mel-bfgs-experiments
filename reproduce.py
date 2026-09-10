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
    print('This checks stored records; use smoke or expanded to rerun numerical solvers.')


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
