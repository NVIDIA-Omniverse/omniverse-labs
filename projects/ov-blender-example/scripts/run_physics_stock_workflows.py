"""Run preserved authoring/Action restoration cases in serial fresh stock Blender."""
import argparse
import json
from pathlib import Path
import sys

from run_physics_workflows import digest, environment, run_process


def main():
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / 'tests'))
    from blender_physics_stock_workflows import CASES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=tuple(k for k in CASES if k != 'bake_restore'),
                        default=[k for k in CASES if k != 'bake_restore'])
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    assert sys.platform == 'win32'
    repo, output = args.repo.resolve(), args.output.resolve()
    output.relative_to(repo)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source / 'addon'))
    blender = repo / '_build/lts-validation-5.2.1/install/blender-5.2.1-windows-x64/blender.exe'
    fixture = source / 'tests/blender_physics_stock_workflows.py'
    def command(case, target, blend=None):
        result = [str(blender), '--background', '--factory-startup', '--threads', '2', '--python-exit-code', '17',
                  '--python', str(fixture), '--', '--case', case, '--output', str(target)]
        if blend is not None:
            result += ['--blend', str(blend)]
        return result
    before = {str(p): digest(p) for p in [*source.rglob('*.py'), blender]}
    receipt = dict(scope='stock authoring only; native physics admission is a separate gate', status='prepared',
                   source_before=before, commands={case: command(case, output / case) for case in args.cases}, cells=[])
    path = output / 'suite.json'
    def persist():
        path.write_text(json.dumps(receipt, indent=2) + '\n')
    persist()
    if args.preflight_only:
        receipt['status'] = 'preflight-only; no Blender process launched'
        persist()
        return 0
    def run(case, blend=None):
        target = output / case
        target.mkdir()
        row = run_process(command(case, target, blend), environment(repo, target, offline=True), target, 180)
        row['case'] = case
        receipt['cells'].append(row)
        persist()
        assert row['exit_code'] == 0 and row['job_active_after_exit'] == 0 and 'watchdog_timeout' not in row, row
        result = json.loads((target / 'stock-results.json').read_text())
        assert result['status'] == 'passed' and result['vendor_sdk_imports'] == [], result
        row['result'] = result
        row['result_sha256'] = digest(target / 'stock-results.json')
        persist()
        print('PASS', case, flush=True)
        return result
    try:
        for case in args.cases:
            result = run(case)
            if case == 'bake_actions':
                blend = Path(result['fixture_artifacts']) / 'shared-active.blend'
                receipt['saved_action_blend'] = dict(path=str(blend), sha256=digest(blend))
                run('bake_restore', blend)
        receipt['status'] = 'passed'
    except BaseException:
        receipt['status'] = 'failed'
        raise
    finally:
        receipt['source_changed'] = [p for p, value in before.items() if digest(Path(p)) != value]
        if receipt['source_changed']:
            receipt['status'] = 'failed: source changed'
        persist()
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
