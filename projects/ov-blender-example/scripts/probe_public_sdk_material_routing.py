# SPDX-License-Identifier: Apache-2.0
"""Inspect actual Blender material -> accepted renderer shader value routing."""
import argparse
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--blend', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--composed', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--material', default='Cube Orange')
parser.add_argument('--source-material', default='/World/_materials/Cube_Orange')
parser.add_argument('--expected-material', default='/OVRTX_Materials/Cube_Orange')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
import bpy
bpy.ops.wm.open_mainfile(filepath=args.blend)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import material_value_conversion
from ovrtx_blender_example.usd_prim_resolver import UsdPrimResolver
from ovrtx_blender_example.blender_interactive_edit_builders import material_value_edits_from_resolver
from types import SimpleNamespace
material = bpy.data.materials[args.material]
principled = material.node_tree.nodes.get('Principled BSDF')
principled.inputs['Base Color'].default_value = (.025,.08,.65,1)
for socket, value in {'IOR':2.2, 'Coat Weight':1.0, 'Coat Roughness':.05,
                      'Emission Color':(.1,.8,.15,1), 'Emission Strength':.5}.items():
    principled.inputs[socket].default_value = value
resolver = UsdPrimResolver(material_paths_by_session_uid={material.session_uid: args.source_material},
                           material_source_scene_path=args.source)
resolver.scan(SimpleNamespace(input_usd_path=args.composed))
rows = []
for attribute in material_value_conversion.usd_attribute_values(material):
    result = resolver.resolve_material(material, usd_attribute=attribute.name,
                                       property_name=attribute.blender_property_path)
    rows.append(dict(attribute=attribute.name, status=str(result.status),
        resolved=None if result.value is None else result.value.prim_path,
        diagnostics=result.diagnostics_dict()))
from ovrtx_blender_example import scene_generation_sessions as sessions
from ovrtx_blender_example.scene_generation import BlenderId, BlenderPrimPath
from ovrtx_blender_example.blender_interactive_edit_builders import build_interactive_edits_from_depsgraph
from ovrtx_blender_example.interactive_edit_planner import InteractiveEditPlanner
from ovrtx_blender_example.runtime_scheduler import RuntimeScheduler
from ovrtx_blender_example.ovrtx_value_updates import OvrtxSessionUpdatePort, OvrtxValueUpdateResult
scene = bpy.context.scene
mapping = BlenderPrimPath(material.name, 'MATERIAL', args.source_material, args.source_material)
generation = SimpleNamespace(usd_path=args.source, blender_prim_paths={BlenderId('MATERIAL',material.session_uid):mapping})
composition = SimpleNamespace(source_scene_path=args.source, composed_scene_path=args.composed)
runtime = SimpleNamespace(ovrtx=SimpleNamespace(controller=SimpleNamespace(_spec=SimpleNamespace(ovrtx_scene_composition=composition))))
sessions._owners[scene.session_uid] = SimpleNamespace(current_generation=generation)
sessions._runtimes[scene.session_uid] = runtime
resolver, _ = sessions.current_generation_edit_context(scene)
edits = build_interactive_edits_from_depsgraph(SimpleNamespace(updates=[SimpleNamespace(id=material)]),
    usd_prim_resolver=resolver, materials=[material])
accepted = sessions.resolve_current_scene_edit_group(scene, edits, {})
assert accepted and len(accepted) == len(edits), ('owner rejected material edits', edits)
scheduler = RuntimeScheduler()
plans = [InteractiveEditPlanner().plan(edit) for edit in accepted]
submitted = scheduler.submit_edits(tuple(plan.to_intent() for plan in plans))
assert all(result.accepted for result in submitted), submitted
class CapturingRenderer:
    def __init__(self): self.values=[]
    def update_attribute_values(self, identity, values):
        self.values.extend(values)
        return OvrtxValueUpdateResult(len(values), 1)
renderer = CapturingRenderer()
applied = scheduler.apply_pending_view_values(OvrtxSessionUpdatePort(renderer, 'probe'))
assert applied.values_written and any(v.attribute == 'inputs:base_color' for v in renderer.values), applied

expected_inputs = {'inputs:specular_ior', 'inputs:coat_weight', 'inputs:coat_roughness',
                   'inputs:emission_color', 'inputs:emission_luminance'}
assert expected_inputs <= {value.attribute for value in renderer.values}, renderer.values
# Actual Blender node socket wiring must enter the generation topology lane.
image_node = material.node_tree.nodes.new('ShaderNodeTexImage')
image_node.image = bpy.data.images.new('Routing probe texture', width=4, height=4)
material.node_tree.links.new(image_node.outputs['Color'], principled.inputs['Base Color'])
texture_edits = build_interactive_edits_from_depsgraph(SimpleNamespace(updates=[SimpleNamespace(id=material)]),
    usd_prim_resolver=resolver, materials=[material])
assert len(texture_edits) == 1 and str(texture_edits[0].shape.value) == 'topology', texture_edits
assert texture_edits[0].metadata['diverged_texture_inputs'] == ('inputs:base_color',), texture_edits
texture_accepted = sessions.resolve_current_scene_edit_group(scene, texture_edits, {})
assert texture_accepted and len(texture_accepted) == 1, texture_edits

report = dict(texture_connect_owner_accepted=True, activated_default_inputs=sorted(expected_inputs), material=material.name, source=args.source, composed=args.composed,
              owner_accepted_count=len(accepted), typed_dispatch=[dict(path=v.prim_path, attribute=v.attribute, value=v.value) for v in renderer.values],
              scan=resolver._scan_diagnostics, projected_paths=resolver._resolved_material_paths,
              attribute_routes=rows, edits=[dict(path=e.usd_prim_path, attribute=e.usd_attribute,
                                               value=e.value) for e in edits])
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
assert any(e.usd_attribute == 'inputs:base_color' and e.usd_prim_path.startswith(args.expected_material + '/') for e in edits), report
assert not any(e.usd_attribute == 'inputs:diffuseColor' for e in edits), report
print(json.dumps({'status':'passed','edited_shader':args.expected_material,'edit_count':len(edits)}))
