# SPDX-License-Identifier: Apache-2.0
"""CPU-only real RNA regression: physics presentation must preserve visual edits."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
sys.path.insert(0, str(args.source / 'addon'))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example import interactive_operator_state as state
from ovrtx_blender_example.physics_endpoints import initialize_owner
from ovrtx_blender_example.physics_frames import RigidFrame
from ovrtx_blender_example.physics_presentation_targets import PhysicsPresentationTarget

props.register()
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
body = bpy.data.objects.new('Owned Body', None)
visual = bpy.data.objects.new('Ordinary Visual Object', None)
scene.collection.objects.link(body)
scene.collection.objects.link(visual)
initialize_owner(body)
bpy.context.view_layer.update()
lock = state.PhysicsPlaybackLock()
target = PhysicsPresentationTarget('/Body', body.ov.physics.persistent_id, body.session_uid,
    body.ov.physics.persistent_id, '', '', 'DYNAMIC', True, 1., 1., RigidFrame(), RigidFrame())
pending = dict(scene_name=scene.name, presentation_targets=(target,),
               poses_by_path={'/Body': dict(translate=(0, 0, 0), orient=(0, 0, 0, 1))})
assert state.apply_pending_runtime_pose_mirror(bpy, pending, lock)['status'] == 'applied'
sessions._runtimes[int(scene.session_uid)] = SimpleNamespace(playback_lock=lock)
events = []
def observe(scene, depsgraph):
    events.append(dict(suppressed=state.interactive_edit_bridge_suppressed(),
                       updates=[dict(name=u.id.name, transform=u.is_updated_transform) for u in depsgraph.updates]))
bpy.app.handlers.depsgraph_update_post.append(observe)
report = dict(status='running', cases={})
def check(name, operation):
    events.clear()
    visual.location.x += 1
    operation()
    report['cases'][name] = list(events)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    visual_events = [e for e in events if any(u['name'] == visual.name for u in e['updates'])]
    assert visual_events and all(not e['suppressed'] for e in visual_events), (name, visual_events)
check('ordinary_control', bpy.context.view_layer.update)
check('physics_poll', sessions._poll_physics_edits)
pending['poses_by_path']['/Body']['translate'] = (3, 0, 0)
check('physics_mirror', lambda: state.apply_pending_runtime_pose_mirror(bpy, pending, lock))
assert body.location.x == 3
assert any(e['suppressed'] and any(u['name'] == body.name for u in e['updates'])
           for e in report['cases']['physics_mirror'])
assert visual.location.x == 3
report['sdk_modules'] = [k for k in sys.modules if k.split('.')[0] in
                        {'ovrtx', 'ovphysx', 'ovstage', 'ovstream', 'warp', 'ovusd', 'ovmaterial', 'ovwarp'}]
assert not report['sdk_modules']
report['status'] = 'passed'
args.output.write_text(json.dumps(report, indent=2) + '\n')
sessions._runtimes.clear()
print('PENDING_VISUAL_EDITS_PASS')
