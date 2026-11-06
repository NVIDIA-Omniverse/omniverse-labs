"""Invoke preserved stock-Blender authoring assertions; never native physics."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

CASES = {
    'kinematic_inputs': ('blender_kinematic_inputs.py', True),
    'bake_conversion': ('blender_bake_conversion.py', False),
    'bake_configuration': ('blender_bake_conversion_configuration.py', False),
    'bake_actions': ('blender_bake_actions.py', False),
    'bake_restore': ('blender_bake_restore.py', True),
    'exact_channels': ('blender_presentation_exact_channels.py', False),
    'action_slots': ('blender_presentation_action_slots.py', True),
    'edit_admission': ('blender_presentation_edit_admission.py', True),
    'playback_errors': ('blender_physics_playback_errors.py', True),
    'playback_anchor': ('blender_physics_playback_anchor.py', False),
    'restore_snapshot': ('blender_physics_restore_snapshot.py', False),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=tuple(CASES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blend', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(source / 'addon'))
    filename, positional_output = CASES[args.case]
    fixture = source / 'tests' / filename
    argv = [str(fixture)]
    if positional_output:
        if args.case == 'bake_restore':
            assert args.blend is not None and args.blend.is_file()
        argv += ['--', str(args.blend if args.case == 'bake_restore' else args.output)]
    sys.argv = argv
    state = runpy.run_path(str(fixture), run_name='__main__')
    forbidden = {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp'}
    imported = sorted(k for k in sys.modules if k.split('.')[0] in forbidden)
    assert not imported, imported
    result = dict(status='passed', scope='stock Blender authoring/restoration only; no physics simulation proof',
                  fixture=str(fixture), fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                  vendor_sdk_imports=imported)
    if 'root' in state:
        result['fixture_artifacts'] = str(state['root'])
    (args.output / 'stock-results.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
