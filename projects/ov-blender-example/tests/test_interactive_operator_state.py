# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import interactive_operator_state as operator_state  # noqa: E402
from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditShape,
    edit_location,
    InteractiveEdit,
)
from ovrtx_blender_example.shared_stage_composition import BodyPose  # noqa: E402


class _FakeBlenderObject(dict):
    def __init__(self, name: str = "Cube") -> None:
        super().__init__()
        self.name = name
        self.selected = False
        self.lock_location = [False, True, False]
        self.lock_rotation = [False, False, True]
        self.lock_scale = [True, False, False]
        self.matrix_world = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (2.0, 3.0, 4.0, 1.0),
        )

    def select_set(self, selected: bool) -> None:
        self.selected = bool(selected)

    def select_get(self) -> bool:
        return self.selected


class _FakeBlenderObjectCollection(list):
    def get(self, name: str) -> object | None:
        for obj in self:
            if getattr(obj, "name", "") == name:
                return obj
        return None


def test_selection_resolution_selects_edit_owner_for_stock_selection() -> None:
    owner = _FakeBlenderObject("Orange_00")
    child = _FakeBlenderObject("Orange_00_mesh")
    child[usd_paths.SELECTION_OWNER_OBJECT_PROP] = "Orange_00"
    child.select_set(True)
    objects = _FakeBlenderObjectCollection([owner, child])
    view_layer_objects = SimpleNamespace(active=child)
    context = SimpleNamespace(
        selected_objects=[child],
        scene=SimpleNamespace(objects=objects),
        view_layer=SimpleNamespace(objects=view_layer_objects),
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["changed"] is True
    assert result["status"] == "resolved"
    assert result["group_supported"] is True
    assert result["group_rejected"] is False
    assert result["resolved_owner_count"] == 1
    assert result["sources"][0]["source_name"] == "Orange_00_mesh"
    assert result["sources"][0]["owner_name"] == "Orange_00"
    assert result["sources"][0]["ownership_source"] == "usd_selection_owner"
    assert child.select_get() is False
    assert owner.select_get() is True
    assert view_layer_objects.active is owner


def test_selection_resolution_updates_the_callback_view_layer() -> None:
    view_layer_objects = SimpleNamespace(active=None)
    view_layer = SimpleNamespace(objects=view_layer_objects)

    class _LayerSelectedObject(_FakeBlenderObject):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.selected_view_layers = set()

        def select_set(self, selected: bool, *, view_layer=None) -> None:
            if selected:
                self.selected_view_layers.add(id(view_layer))
            else:
                self.selected_view_layers.discard(id(view_layer))

        def select_get(self, *, view_layer=None) -> bool:
            return id(view_layer) in self.selected_view_layers

    owner = _LayerSelectedObject("Orange_00")
    child = _LayerSelectedObject("Orange_00_mesh")
    child[usd_paths.SELECTION_OWNER_OBJECT_PROP] = owner.name
    child.select_set(True, view_layer=view_layer)
    view_layer_objects.active = child
    context = SimpleNamespace(
        selected_objects=[child],
        scene=SimpleNamespace(
            objects=_FakeBlenderObjectCollection([owner, child])
        ),
        view_layer=view_layer,
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["changed"] is True
    assert child.select_get(view_layer=view_layer) is False
    assert owner.select_get(view_layer=view_layer) is True
    assert view_layer_objects.active is owner


def test_selection_resolution_records_unmapped_selection_as_inspection_only() -> None:
    obj = _FakeBlenderObject("LooseCube")
    obj.select_set(True)
    context = SimpleNamespace(
        selected_objects=[obj],
        scene=SimpleNamespace(objects=_FakeBlenderObjectCollection([obj])),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["changed"] is False
    assert result["status"] == "unsupported_selection_group"
    assert result["group_supported"] is False
    assert result["group_rejected"] is True
    assert result["resolved_owner_count"] == 0
    assert result["unresolved_reasons"] == ["unmapped_selection_source"]
    assert result["sources"][0]["status"] == "unresolved"
    assert result["sources"][0]["owner_category"] == "inspection_only"
    assert result["sources"][0]["preview_only"] is True


def test_selection_resolution_does_not_treat_unmapped_light_as_owner() -> None:
    obj = _FakeBlenderObject("LooseLight")
    obj.type = "LIGHT"
    obj.select_set(True)
    context = SimpleNamespace(
        selected_objects=[obj],
        scene=SimpleNamespace(objects=_FakeBlenderObjectCollection([obj])),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["status"] == "unsupported_selection_group"
    assert result["sources"][0]["status"] == "unresolved"
    assert result["sources"][0]["owner_category"] == "inspection_only"


def test_selection_resolution_records_mapped_render_prim_as_preview_only() -> None:
    obj = _FakeBlenderObject("Mesh")
    obj[usd_paths.SOURCE_USD_PATH_PROP] = "/World/Asset/Mesh"
    obj.select_set(True)
    context = SimpleNamespace(
        selected_objects=[obj],
        scene=SimpleNamespace(objects=_FakeBlenderObjectCollection([obj])),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["status"] == "unsupported_selection_group"
    assert result["sources"][0]["status"] == "preview_only"
    assert result["sources"][0]["mapping_basis"] == "source_identity"
    assert result["sources"][0]["owner_category"] == "inspection_only"
    assert result["sources"][0]["source_usd_path"] == "/World/Asset/Mesh"
    assert result["sources"][0]["owner_usd_path"] == ""
    assert result["sources"][0]["unresolved_reason"] == "preview_only_selection_source"


def test_selection_resolution_rejects_mixed_group_without_selecting_owner() -> None:
    owner = _FakeBlenderObject("Orange_00")
    child = _FakeBlenderObject("Orange_00_mesh")
    loose = _FakeBlenderObject("LooseCube")
    child[usd_paths.SELECTION_OWNER_OBJECT_PROP] = "Orange_00"
    child.select_set(True)
    loose.select_set(True)
    objects = _FakeBlenderObjectCollection([owner, child, loose])
    view_layer_objects = SimpleNamespace(active=child)
    context = SimpleNamespace(
        selected_objects=[child, loose],
        scene=SimpleNamespace(objects=objects),
        view_layer=SimpleNamespace(objects=view_layer_objects),
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["changed"] is False
    assert result["status"] == "unsupported_selection_group"
    assert result["group_rejected"] is True
    assert child.select_get() is True
    assert loose.select_get() is True
    assert owner.select_get() is False
    assert view_layer_objects.active is child


def test_selection_resolution_records_direct_tagged_owner_identity() -> None:
    obj = _FakeBlenderObject("Cube")
    obj[usd_paths.USD_PRIM_PATH_PROP] = "/World/Cube"
    obj[usd_paths.DATA_AUTHORITY_PROP] = "view"
    obj[usd_paths.BLENDER_PROPERTY_PATH_PROP] = "matrix_world"
    obj.select_set(True)
    context = SimpleNamespace(
        selected_objects=[obj],
        scene=SimpleNamespace(objects=_FakeBlenderObjectCollection([obj])),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
    )

    result = operator_state.resolve_blender_selection_to_edit_owners(context)

    assert result["status"] == "resolved"
    assert result["group_supported"] is True
    assert result["owner_categories"] == ["view_value_owner"]
    assert result["sources"][0]["edit_target_identity"]["usd_prim_path"] == "/World/Cube"
    assert result["sources"][0]["edit_target_identity"]["usd_layer_id"] == ""
    assert result["sources"][0]["edit_target_identity"]["data_authority"] == "view"


def test_physics_playback_lock_rejects_physics_edit_but_allows_look_edit() -> None:
    lock = operator_state.PhysicsPlaybackLock()
    obj = _FakeBlenderObject()
    lock.lock_object("/World/TestScene/Cube", obj, generation=3)
    attempted = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_layer_id="/layers/scene.usda",
            usd_prim_path="/World/TestScene/Cube",
            blender_property_path="matrix_world",
        ),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )
    look_only = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/TestScene/Cube",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.5, 0.25),
    )

    rejected = lock.reject_edit(attempted)

    assert rejected is not None
    assert rejected.reason == "physics_playback_locked"
    assert lock.reject_edit(look_only) is None
    assert lock.diagnostics()["rejected_edit_count"] == 1


def test_runtime_pose_mirror_prepares_pending_scene_update_without_render_session() -> None:
    obj = _FakeBlenderObject()
    obj["ovrtx.usd_prim_path"] = "/World/PhysicsIsland/DynamicBodies/Cube_00"
    scene = SimpleNamespace(name="Scene", objects=[obj])
    poses = (
        BodyPose(
            "/World/PhysicsIsland/DynamicBodies/Cube_00",
            (1.0, 2.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    pending, result = operator_state.prepare_runtime_pose_mirror(
        poses,
        object(),
        SimpleNamespace(scene=scene),
        owning_generation=9,
        last_applied={"status": "applied"},
    )

    assert result["status"] == "scheduled"
    assert result["matched_pose_count"] == 1
    assert result["interaction_object_count"] == 1
    assert result["interaction_object_paths"] == ["/World/PhysicsIsland/DynamicBodies/Cube_00"]
    assert result["missing_object_paths"] == []
    assert pending["owning_generation"] == 9
    assert pending["poses_by_path"]["/World/PhysicsIsland/DynamicBodies/Cube_00"] == {
        "translate": (1.0, 2.0, 3.0),
        "orient": (0.0, 0.0, 0.0, 1.0),
    }


def test_initial_condition_and_mirror_policy_are_operator_state() -> None:
    request = SimpleNamespace(
        timeline_controls_enabled=True,
        timeline_playing=False,
        timeline_frame=1,
        timeline_start=1,
    )

    assert operator_state.request_at_initial_condition(request)
    assert operator_state.should_mirror_runtime_poses(
        at_initial_condition=True,
        lock_was_active=True,
    )
    assert not operator_state.should_mirror_runtime_poses(
        at_initial_condition=True,
        lock_was_active=False,
    )
