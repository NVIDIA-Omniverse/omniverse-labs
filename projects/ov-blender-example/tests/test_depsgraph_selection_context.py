"""Selection context for the live depsgraph bridge (OMPE-115792 n-02-04).

ViewLayer is not an ID, so bpy hands out a fresh Python wrapper on every
access and ``context.view_layer is depsgraph.view_layer`` never holds. These
tests model that with wrappers that share ``as_pointer()`` but not identity.
"""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))

import ovrtx_blender_example.engine as engine_module  # noqa: E402
from ovrtx_blender_example import interactive_operator_state as operator_state  # noqa: E402
from ovrtx_blender_example import physics_live_kinematics  # noqa: E402
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402
from ovrtx_blender_example.physics_edit_snapshot import PhysicsChanges  # noqa: E402


class _ViewLayerStruct:
    """The C-side ViewLayer; ``wrapper()`` mimics bpy's per-access wrappers."""

    def __init__(self, pointer: int) -> None:
        self.pointer = pointer
        self.objects = SimpleNamespace(active=None)

    def wrapper(self) -> "_ViewLayerWrapper":
        return _ViewLayerWrapper(self)


class _ViewLayerWrapper:
    def __init__(self, struct: _ViewLayerStruct) -> None:
        self._struct = struct

    def as_pointer(self) -> int:
        return self._struct.pointer

    @property
    def objects(self):
        return self._struct.objects


class _Context:
    def __init__(self, scene, layer: _ViewLayerStruct, selected=()) -> None:
        self.scene = scene
        self._layer = layer
        self._selected = list(selected)
        self.selected_objects_reads = 0

    @property
    def view_layer(self):
        return self._layer.wrapper()

    @property
    def selected_objects(self):
        self.selected_objects_reads += 1
        return list(self._selected)


class _Depsgraph:
    def __init__(self, updates, layer: _ViewLayerStruct) -> None:
        self.updates = updates
        self._layer = layer

    @property
    def view_layer(self):
        return self._layer.wrapper()


class _ForbiddenObjects:
    def __iter__(self):
        pytest.fail("camera-only depsgraph update walked scene.objects")

    def __len__(self):
        pytest.fail("camera-only depsgraph update sized scene.objects")


def _camera_update():
    camera = SimpleNamespace(
        name="Camera",
        type="CAMERA",
        session_uid=5,
        bl_rna=SimpleNamespace(identifier="Object"),
    )
    return SimpleNamespace(
        id=camera,
        is_updated_transform=True,
        is_updated_geometry=False,
        is_updated_shading=False,
    )


@pytest.fixture
def isolated_handler(monkeypatch):
    """Keep the handler's non-selection collaborators inert and scene-free."""

    monkeypatch.setattr(physics_live_kinematics, "evaluating", lambda: False)
    monkeypatch.setattr(sessions, "observe_physics_depsgraph", lambda *_a: PhysicsChanges())
    monkeypatch.setattr(sessions, "affected_blender_ids", lambda *_a: set())
    monkeypatch.setattr(sessions, "affected_blender_update_provenance", lambda *_a: {})
    monkeypatch.setattr(sessions, "shader_only_generation_update", lambda *_a: True)
    monkeypatch.setattr(sessions, "record_ignored_scene_update", lambda *_a: None)
    monkeypatch.setattr(sessions, "mark_scene_dirty", lambda *_a, **_k: pytest.fail("camera move marked dirty"))
    monkeypatch.setattr(engine_module, "_capture_mesh_geometry_ids", lambda *_a: set())
    monkeypatch.setattr(engine_module, "_capture_visibility_ids", lambda *_a: set())
    submitted = []

    def submit(depsgraph, *, context=None, scene=None):
        # The adapter's first use of the context is the selection resolver.
        submitted.append((context, operator_state.resolve_blender_selection_to_edit_owners(context)))
        return []

    monkeypatch.setattr(engine_module, "submit_depsgraph_interactive_edits_to_active_viewports", submit)
    errors = []
    monkeypatch.setattr(
        engine_module,
        "_record_interactive_edit_bridge_diagnostics",
        lambda **kwargs: errors.append(kwargs.get("last_error")),
    )
    return submitted, errors


def test_camera_only_update_on_active_layer_never_walks_scene_objects(monkeypatch, isolated_handler):
    submitted, errors = isolated_handler
    scene = SimpleNamespace(session_uid=41, objects=_ForbiddenObjects())
    layer = _ViewLayerStruct(0x5EED)
    context = _Context(scene, layer)
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(context=context))

    for _ in range(3):
        engine_module._live_interactive_edit_depsgraph_handler(scene, _Depsgraph([_camera_update()], layer))

    assert errors == []
    assert [entry[0] for entry in submitted] == [context] * 3
    assert all(entry[1]["selected_object_count"] == 0 for entry in submitted)
    # Blender's own selection list for the active layer is the only source read.
    assert context.selected_objects_reads == 3


def test_active_layer_returns_blender_context_with_its_selection(monkeypatch):
    layer = _ViewLayerStruct(0xA11)
    scene = SimpleNamespace(objects=_ForbiddenObjects())
    selected = SimpleNamespace(name="Cube")
    context = _Context(scene, layer, selected=(selected,))
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(context=context))

    assert context.view_layer is not context.view_layer  # bpy wrapper behaviour
    resolved = engine_module._selection_context_for_scene(scene, _Depsgraph([], layer))

    assert resolved is context
    assert resolved.selected_objects == [selected]


def test_foreign_callback_layer_keeps_explicit_select_get(monkeypatch):
    active = _ViewLayerStruct(0x1)
    foreign = _ViewLayerStruct(0x2)
    queried = []

    class _Object:
        def __init__(self, name, selected_on):
            self.name = name
            self._selected_on = selected_on

        def select_get(self, *, view_layer=None):
            queried.append((self.name, view_layer.as_pointer()))
            return view_layer.as_pointer() in self._selected_on

    first = _Object("A", {0x2})
    second = _Object("B", {0x1})
    scene = SimpleNamespace(objects=(first, second))
    context = _Context(scene, active, selected=(second,))
    monkeypatch.setattr(engine_module, "bpy", SimpleNamespace(context=context))

    resolved = engine_module._selection_context_for_scene(scene, _Depsgraph([], foreign))

    assert resolved is not context
    assert resolved.view_layer.as_pointer() == 0x2
    assert resolved.selected_objects == (first,)
    assert queried == [("A", 0x2), ("B", 0x2)]
    assert context.selected_objects_reads == 0


def test_same_rna_struct_compares_pointers_not_values():
    struct = _ViewLayerStruct(0x77)
    assert engine_module._same_rna_struct(struct.wrapper(), struct.wrapper())
    assert not engine_module._same_rna_struct(struct.wrapper(), _ViewLayerStruct(0x78).wrapper())
    # Plain stand-ins keep identity semantics even when their values compare equal.
    first = SimpleNamespace(objects=SimpleNamespace(active=None))
    second = SimpleNamespace(objects=SimpleNamespace(active=None))
    assert first == second
    assert engine_module._same_rna_struct(first, first)
    assert not engine_module._same_rna_struct(first, second)
    assert not engine_module._same_rna_struct(None, None)
