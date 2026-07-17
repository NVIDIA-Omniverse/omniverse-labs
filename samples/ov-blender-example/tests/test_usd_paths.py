# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402


class _FakeObject(dict):
    def __init__(
        self,
        name: str,
        *,
        parent: object | None = None,
        data: object | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.parent = parent
        self.data = data


class _FakePrim:
    def __init__(self, path: str, schemas: object = ()) -> None:
        self._path = path
        self._schemas = schemas

    def GetPath(self) -> str:
        return self._path

    def GetMetadata(self, key: str) -> object:
        if key == "apiSchemas":
            return self._schemas
        return None


class _FakeStage:
    def __init__(self, prims: list[_FakePrim]) -> None:
        self._prims = prims

    def Traverse(self) -> list[_FakePrim]:
        return self._prims


def test_load_usd_path_index_reads_valid_and_rigid_body_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    usd_path = tmp_path / "stage.usda"
    stage = _FakeStage(
        [
            _FakePrim("/World"),
            _FakePrim("/World/PhysicsIsland/DynamicBodies/Body", ("PhysicsRigidBodyAPI",)),
            _FakePrim("/World/PhysicsIsland/DynamicBodies/Body/Geom"),
        ]
    )

    class _FakeStageFactory:
        @staticmethod
        def Open(path: str) -> _FakeStage | None:
            assert path == str(usd_path)
            return stage

    pxr = ModuleType("pxr")
    pxr.Usd = SimpleNamespace(Stage=_FakeStageFactory)
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    path_index = usd_paths.load_usd_path_index(usd_path)

    assert path_index["available"] is True
    assert path_index["reason"] == ""
    assert path_index["valid_paths"] == {
        "/World",
        "/World/PhysicsIsland/DynamicBodies/Body",
        "/World/PhysicsIsland/DynamicBodies/Body/Geom",
    }
    assert path_index["rigid_body_paths"] == {"/World/PhysicsIsland/DynamicBodies/Body"}


def test_camera_path_helpers_preserve_binding_match_rules() -> None:
    world = _FakeObject("World")
    nested_camera = _FakeObject("Camera.001", parent=world)
    root_camera = _FakeObject("OvrtxCamera.001")
    data_camera = _FakeObject(
        "ImportedCamera",
        data={usd_paths.SOURCE_USD_PATH_PROP: "/World/DataCamera"},
    )

    assert usd_paths.camera_match_sources("/World/Camera") == (
        usd_paths.CAMERA_SOURCE_USD_PATH_MATCH,
        usd_paths.CAMERA_HIERARCHY_PATH_MATCH,
    )
    assert (
        usd_paths.camera_match_sources("/OvrtxCamera")
        == usd_paths.CAMERA_MATCH_SOURCE_ORDER
    )
    assert (
        usd_paths.camera_usd_path_for_source(
            data_camera,
            usd_paths.CAMERA_SOURCE_USD_PATH_MATCH,
        )
        == "/World/DataCamera"
    )
    assert (
        usd_paths.camera_usd_path_for_source(
            nested_camera,
            usd_paths.CAMERA_HIERARCHY_PATH_MATCH,
        )
        == "/World/Camera"
    )
    assert (
        usd_paths.camera_usd_path_for_source(
            root_camera,
            usd_paths.CAMERA_ROOT_OBJECT_PATH_MATCH,
        )
        == "/OvrtxCamera"
    )
    assert usd_paths.nested_hierarchy_usd_path(root_camera) == ""


def test_known_usd_path_preserves_normalized_runtime_path_contract() -> None:
    assert usd_paths.clean_usd_path(" /World/Camera ") == "/World/Camera"
    assert usd_paths.known_usd_path("/World/Camera") is True
    assert usd_paths.known_usd_path(" /World/Camera ") is False
    assert usd_paths.known_usd_path("???") is False
    assert usd_paths.known_usd_path("World/Camera") is False


def test_reserve_unique_child_path_sanitizes_and_reserves_without_topology_policy() -> None:
    occupied = {"/World/First_Body"}

    assert usd_paths.reserve_unique_child_path("/World", "First Body", occupied) == "/World/First_Body_2"
    assert usd_paths.reserve_unique_child_path("/World", "23 weird--body", occupied) == "/World/_23_weird_body"
    assert usd_paths.valid_usd_identifier("") == "Prim"

    try:
        usd_paths.reserve_unique_child_path("World", "Body", occupied)
    except ValueError as exc:
        assert "absolute prim path" in str(exc)
    else:
        raise AssertionError("relative topology parent was accepted")


def test_id_property_reads_blender_like_mapping_and_getters() -> None:
    mapping = {usd_paths.USD_PRIM_PATH_PROP: "/World/Mapping"}
    getter = _FakeObject("Getter")
    getter[usd_paths.USD_PRIM_PATH_PROP] = "/World/Getter"

    assert usd_paths.id_property(mapping, usd_paths.USD_PRIM_PATH_PROP, "") == "/World/Mapping"
    assert usd_paths.id_property(getter, usd_paths.USD_PRIM_PATH_PROP, "") == "/World/Getter"
    assert usd_paths.id_property(None, usd_paths.USD_PRIM_PATH_PROP, "default") == "default"


def test_usd_prim_path_from_prim_uses_usd_api_then_path_attribute() -> None:
    assert usd_paths.usd_prim_path_from_prim(_FakePrim(" /World/ApiPath ")) == "/World/ApiPath"
    assert usd_paths.usd_prim_path_from_prim(SimpleNamespace(path="/World/Fallback")) == "/World/Fallback"
    assert usd_paths.usd_prim_path_from_prim(SimpleNamespace(path="???")) == ""


def test_usd_prim_type_name_from_prim_uses_usd_api_then_type_name_attribute() -> None:
    def broken_type_name() -> str:
        raise RuntimeError("broken")

    assert (
        usd_paths.usd_prim_type_name_from_prim(
            SimpleNamespace(GetTypeName=lambda: "Mesh", type_name="Fallback")
        )
        == "Mesh"
    )
    assert usd_paths.usd_prim_type_name_from_prim(SimpleNamespace(type_name="DomeLight")) == "DomeLight"
    assert (
        usd_paths.usd_prim_type_name_from_prim(
            SimpleNamespace(GetTypeName=broken_type_name, type_name="Fallback")
        )
        == ""
    )


def test_normalized_usd_leaf_name_extracts_last_path_part_with_blender_suffix() -> None:
    assert usd_paths.normalized_usd_leaf_name("/World/Props/Cube.001") == "Cube"
    assert usd_paths.normalized_usd_leaf_name("/World/Props/Cube") == "Cube"
    assert usd_paths.normalized_usd_leaf_name("/") == ""


def test_resolved_usd_path_prefers_imported_source_path_then_hierarchy() -> None:
    path_index = {
        "valid_paths": {
            "/World/Body",
            "/World/Body/Geom",
        },
        "rigid_body_paths": set(),
    }
    body = _FakeObject("WrongName")
    body["ovrtx:sourceUsdPath"] = "/World/Body"
    geom = _FakeObject("Geom", parent=_FakeObject("Body", parent=_FakeObject("World")))

    assert usd_paths.resolved_usd_path(body, path_index) == "/World/Body"
    assert usd_paths.resolved_usd_path(geom, path_index) == "/World/Body/Geom"


def test_nearest_dynamic_body_path_walks_to_rigid_body_under_root() -> None:
    path_index = {
        "valid_paths": set(),
        "rigid_body_paths": {
            "/World/PhysicsIsland/DynamicBodies/Body",
            "/World/Other/Body",
        },
    }

    assert (
        usd_paths.nearest_dynamic_body_path(
            "/World/PhysicsIsland/DynamicBodies/Body/Geom",
            path_index,
            "/World/PhysicsIsland/DynamicBodies",
        )
        == "/World/PhysicsIsland/DynamicBodies/Body"
    )


def test_tag_body_identity_sets_edit_owner_and_selection_source_props() -> None:
    owner = _FakeObject("Body")
    visual = _FakeObject("BodyMesh")

    usd_paths.tag_body_edit_owner(owner, "/World/PhysicsIsland/DynamicBodies/Body")
    usd_paths.tag_body_selection_source(
        visual,
        "/World/PhysicsIsland/DynamicBodies/Body",
        owner,
    )

    assert owner["ovrtx.usd_prim_path"] == "/World/PhysicsIsland/DynamicBodies/Body"
    assert owner["ovrtx.data_authority"] == "sim"
    assert visual["ovrtx.selection_owner_object"] == "Body"
    assert (
        visual["ovrtx.selection_owner_usd_prim_path"]
        == "/World/PhysicsIsland/DynamicBodies/Body"
    )


def test_authoring_prim_path_reads_direct_identity_and_original_fallback() -> None:
    from types import SimpleNamespace

    direct = SimpleNamespace(
        ov=SimpleNamespace(usd=SimpleNamespace(prim_path="/World/Materials/Paint"))
    )
    # Evaluated depsgraph copies of some ID types (materials in Blender
    # 5.1) do not carry add-on PointerProperty data: the identity must be
    # readable from the evaluated copy's original datablock (task04-02).
    evaluated_copy = SimpleNamespace(
        ov=SimpleNamespace(usd=SimpleNamespace(prim_path="")),
        original=direct,
    )
    no_identity = SimpleNamespace()

    assert usd_paths.authoring_prim_path(direct) == "/World/Materials/Paint"
    assert usd_paths.authoring_prim_path(evaluated_copy) == "/World/Materials/Paint"
    assert usd_paths.authoring_prim_path(no_identity) == ""
