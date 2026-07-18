"""Synthetic pass/fail coverage for audit_lighting_world.py under Blender."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import bpy


script = Path(__file__).resolve().parents[1] / "scripts" / "audit_lighting_world.py"
spec = spec_from_file_location("audit_lighting_world", script)
module = module_from_spec(spec)
spec.loader.exec_module(module)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
world = bpy.data.worlds.new("TestWorld")
scene.world = world
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()
output = nodes.new("ShaderNodeOutputWorld")
background = nodes.new("ShaderNodeBackground")
background.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
background.inputs["Strength"].default_value = 0.25
links.new(background.outputs["Background"], output.inputs["Surface"])

data = bpy.data.lights.new("KeyData", type="AREA")
data.energy = 500.0
data.shape = "DISK"
data.size = 2.0
key = bpy.data.objects.new("Key", data)
scene.collection.objects.link(key)
bpy.context.view_layer.update()

passed = module.audit({"scene": scene.name, "lights": ["Key"], "require_effective_lighting": True})
assert passed["ok"], passed
assert passed["boundary"].startswith("Blender authoring state only")

data.size = 0.0
failed = module.audit({"scene": scene.name, "lights": ["Key"], "require_effective_lighting": True})
assert not failed["ok"], failed
assert "invalid_area_size" in failed["lights"][0]["issues"]

missing = module.audit({"scene": scene.name, "lights": ["Missing"], "require_effective_lighting": True})
assert not missing["ok"], missing
assert missing["missing_lights"] == ["Missing"]

print("lighting audit synthetic tests: pass")
