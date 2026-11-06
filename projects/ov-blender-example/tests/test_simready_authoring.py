"""Composition validation admits only RNA dimensional float roundoff."""
import pytest

pytest.importorskip("pxr")
from ovrtx_blender_example.simready_authoring import equivalent_text

SOURCE = '''#usda 1.0
def Xform "Body" (
    prepend apiSchemas = ["PhysicsMassAPI", "PhysicsCollisionAPI"]
) {
    point3f physics:centerOfMass = (0.1, -0.2, 0.3)
    float physics:mass = 2
    token visibility = "inherited"
    rel material:binding:physics = </Material>
    def Mesh "Geometry" {}
}
'''


def test_only_dimensional_float_roundoff_is_admitted():
    assert equivalent_text(SOURCE, SOURCE.replace("(0.1, -0.2, 0.3)", "(0.099999994, -0.19999999, 0.29999998)"))


@pytest.mark.parametrize("old,new", [
    ("(0.1, -0.2, 0.3)", "(0.1001, -0.2, 0.3)"),
    ("physics:mass = 2", "physics:mass = 2.0001"),
    ('"inherited"', '"invisible"'),
    ("</Material>", "</Other>"),
])
def test_changed_physics_visibility_or_material_is_rejected(old, new):
    assert not equivalent_text(SOURCE, SOURCE.replace(old, new))
