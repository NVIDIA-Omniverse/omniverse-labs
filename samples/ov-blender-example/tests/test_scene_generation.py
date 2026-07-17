# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import scene_generation  # noqa: E402
from ovrtx_blender_example.ovrtx_value_updates import (  # noqa: E402
    OvrtxAttributeValue,
    OvrtxTransformValue,
)
from ovrtx_blender_example.shared_stage_composition import BodyPose  # noqa: E402
from ovrtx_blender_example.world_dome_conversion import DEFAULT_DOME_OWNER_PATH  # noqa: E402


def _mock_layered_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scene_generation, "_validate_identity_free_base", lambda *_: None)
    monkeypatch.setattr(
        scene_generation,
        "_write_identity_binding",
        lambda path, *_: path.write_bytes(b"ids"),
    )
    monkeypatch.setattr(
        scene_generation,
        "_write_stock_root",
        lambda path, _ids, base: path.write_bytes(base.read_bytes()),
    )
    monkeypatch.setattr(scene_generation, "_validate_composed_generation", lambda *_: None)
    monkeypatch.setattr(
        scene_generation,
        "_write_layered_generation",
        lambda path, _records, _deltas, base: path.write_bytes(base.read_bytes()),
    )


def test_failed_replacement_keeps_last_valid_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    exports = 0

    def export(_scene: object, path: Path) -> None:
        nonlocal exports
        exports += 1
        if exports == 2:
            raise scene_generation.SceneGenerationError("export failed")
        path.write_bytes(b"first generation")

    monkeypatch.setattr(scene_generation, "_stock_export", export)
    monkeypatch.setattr(
        scene_generation,
        "_validated_blender_prim_paths",
        lambda _scene, _path: {
            scene_generation.BlenderId("OBJECT", 1): scene_generation.BlenderPrimPath(
                "Cube",
                "MESH",
                "/Cube",
                "/Cube/Cube",
            )
        },
    )
    monkeypatch.setattr(scene_generation, "_remove_stock_export_identities", lambda *_: None)
    _mock_layered_writes(monkeypatch)
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    first = owner.replace(object())
    assert first is not None

    with pytest.raises(scene_generation.SceneGenerationError, match="export failed"):
        owner.replace(object())

    assert owner.current_generation is first
    assert owner.reuse() is first
    assert Path(first.usd_path).read_bytes() == b"first generation"
    assert list(owner.work_directory.glob(".candidate-*")) == []


def test_close_removes_only_owner_work_directory(monkeypatch, tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("keep", encoding="utf-8")

    def export(_scene: object, path: Path) -> None:
        path.write_bytes(b"generation")

    monkeypatch.setattr(scene_generation, "_stock_export", export)
    monkeypatch.setattr(scene_generation, "_validated_blender_prim_paths", lambda *_: {})
    monkeypatch.setattr(scene_generation, "_remove_stock_export_identities", lambda *_: None)
    _mock_layered_writes(monkeypatch)
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    assert owner.replace(object()) is not None

    owner.close()
    owner.close()

    assert not owner.work_directory.exists()
    assert outside.read_text(encoding="utf-8") == "keep"
    with pytest.raises(scene_generation.SceneGenerationError, match="closed"):
        owner.reuse()


def test_reusable_base_rebinds_new_session_uids_without_exporting_again(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_lookup = scene_generation._source_lookup  # noqa: SLF001
    source = tmp_path / "scene.blend"
    source.write_bytes(b"saved scene")
    blender_data = SimpleNamespace(
        filepath=str(source),
        is_dirty=False,
        libraries=(),
        images=(),
        materials=(),
    )
    monkeypatch.setattr(
        scene_generation,
        "bpy",
        SimpleNamespace(
            data=blender_data,
            app=SimpleNamespace(version=(4, 5, 0), version_cycle="release", build_hash=b"abc"),
            context=SimpleNamespace(view_layer=SimpleNamespace(name="ViewLayer")),
            path=SimpleNamespace(abspath=lambda path, **_kwargs: path),
        ),
    )
    exports = []

    def export(scene: object, path: Path) -> None:
        exports.append(scene)
        path.write_bytes(b"uid-free stock base")

    def mappings(scene: object, _path: Path) -> dict[object, object]:
        obj = scene.objects[0]
        return {
            scene_generation.BlenderId("OBJECT", obj.session_uid): (
                scene_generation.BlenderPrimPath(
                    obj.name_full,
                    "MESH",
                    "/World/Cube",
                    "/World/Cube/Cube",
                    obj.data.session_uid,
                )
            )
        }

    monkeypatch.setattr(scene_generation, "_stock_export", export)
    monkeypatch.setattr(scene_generation, "_validated_blender_prim_paths", mappings)
    monkeypatch.setattr(scene_generation, "_remove_stock_export_identities", lambda *_: None)
    monkeypatch.setattr(scene_generation, "_validate_identity_free_base", lambda *_: None)
    monkeypatch.setattr(scene_generation, "_topology_fingerprints", lambda *_: {})
    _mock_layered_writes(monkeypatch)

    def make_scene(object_uid: int, mesh_uid: int) -> SimpleNamespace:
        return SimpleNamespace(
            name_full="Scene",
            frame_current=1,
            frame_subframe=0.0,
            world=None,
            objects=(
                SimpleNamespace(
                    name_full="Cube",
                    session_uid=object_uid,
                    type="MESH",
                    library=None,
                    data=SimpleNamespace(
                        name_full="Cube",
                        session_uid=mesh_uid,
                        library=None,
                    ),
                ),
            ),
        )

    cache = tmp_path / "cache"
    first = scene_generation.SceneGenerationOwner(tmp_path / "first", cache).replace(
        make_scene(11, 21)
    )
    second = scene_generation.SceneGenerationOwner(tmp_path / "second", cache).replace(
        make_scene(12, 22)
    )

    assert len(exports) == 1
    assert first is not None and first.diagnostics["reusable_base_status"] == "published"
    assert second is not None and second.diagnostics["reusable_base_status"] == "hit"
    assert first.base_digest == second.base_digest
    assert first.digest != second.digest
    assert scene_generation.BlenderId("OBJECT", 12) in second.blender_prim_paths
    assert scene_generation.BlenderId("OBJECT", 11) not in second.blender_prim_paths
    assert (
        second.blender_prim_paths[
            scene_generation.BlenderId("OBJECT", 12)
        ].data_session_uid
        == 22
    )
    second_snapshot = Path(second.usd_path).parent / "stock-base" / "scene.usdc"
    assert second_snapshot.read_bytes() == b"uid-free stock base"

    cache_entry = cache / first.diagnostics["reusable_base_lookup"]
    copytree = scene_generation.shutil.copytree

    def fail_cache_snapshot(source: Path, *args: object, **kwargs: object) -> Path:
        if Path(source) == cache_entry:
            raise FileNotFoundError("concurrent cache replacement")
        return copytree(source, *args, **kwargs)

    monkeypatch.setattr(scene_generation.shutil, "copytree", fail_cache_snapshot)
    fallback = scene_generation.SceneGenerationOwner(
        tmp_path / "snapshot-fallback", cache
    ).replace(make_scene(16, 26))
    monkeypatch.setattr(scene_generation.shutil, "copytree", copytree)
    assert fallback is not None
    assert fallback.diagnostics["reusable_base_status"] == "published"
    assert len(exports) == 2

    manifest_path = cache_entry / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mappings"][0]["object_path"] = "/Bogus"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    cache_entry.with_name(f".{cache_entry.name}.publication-lock").mkdir()

    third = scene_generation.SceneGenerationOwner(tmp_path / "third", cache).replace(
        make_scene(13, 23)
    )
    assert len(exports) == 3
    assert third is not None
    assert third.diagnostics["reusable_base_status"] == "published"
    assert second_snapshot.read_bytes() == b"uid-free stock base"

    lookups = iter((("d" * 64, ""), ("", "dirty_source")))
    monkeypatch.setattr(scene_generation, "_source_lookup", lambda _scene: next(lookups))
    skipped_cache = tmp_path / "skipped-cache"
    skipped = scene_generation.SceneGenerationOwner(
        tmp_path / "skipped", skipped_cache
    ).replace(make_scene(14, 24))
    assert skipped is not None
    assert skipped.diagnostics["reusable_base_status"] == "publication_skipped"
    assert not skipped_cache.exists()
    monkeypatch.setattr(scene_generation, "_source_lookup", source_lookup)

    (cache_entry / "scene.usdc").write_bytes(b"corrupt")
    def fail_publication(*_args: object) -> None:
        raise OSError("read only")

    monkeypatch.setattr(scene_generation, "_publish_reusable_base", fail_publication)
    fourth = scene_generation.SceneGenerationOwner(tmp_path / "fourth", cache).replace(
        make_scene(15, 25)
    )
    assert fourth is not None
    assert fourth.diagnostics["reusable_base_status"] == "publication_failed"


def test_reusable_base_lookup_invalidates_external_input_and_dirty_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "scene.blend"
    image_path = tmp_path / "texture.png"
    source.write_bytes(b"scene")
    image_path.write_bytes(b"first")
    data = SimpleNamespace(
        filepath=str(source),
        is_dirty=False,
        libraries=(),
        images=(
            SimpleNamespace(
                filepath=str(image_path),
                packed_file=None,
                packed_files=(),
                source="FILE",
                library=None,
            ),
        ),
    )
    monkeypatch.setattr(
        scene_generation,
        "bpy",
        SimpleNamespace(
            data=data,
            app=SimpleNamespace(version=(4, 5, 0), version_cycle="release", build_hash=b"abc"),
            context=SimpleNamespace(view_layer=SimpleNamespace(name="ViewLayer")),
            path=SimpleNamespace(abspath=lambda path, **_kwargs: path),
        ),
    )
    scene = SimpleNamespace(name_full="Scene", frame_current=1, frame_subframe=0.0)

    first, reason = scene_generation._source_lookup(scene)  # noqa: SLF001
    image_path.write_bytes(b"second")

    assert first and not reason
    assert scene_generation._source_lookup(scene)[0] != first  # noqa: SLF001
    data.is_dirty = True
    assert scene_generation._source_lookup(scene) == ("", "dirty_source")  # noqa: SLF001
    data.is_dirty = False
    data.images[0].source = "SEQUENCE"
    assert scene_generation._source_lookup(scene) == (  # noqa: SLF001
        "",
        "external_input_unreadable",
    )
    data.images[0].source = "FILE"
    data.objects = (
        SimpleNamespace(modifiers=(SimpleNamespace(type="MESH_CACHE"),)),
    )
    assert scene_generation._source_lookup(scene) == (  # noqa: SLF001
        "",
        "external_input_unreadable",
    )


def test_stock_base_digest_includes_generated_assets() -> None:
    first = scene_generation._stock_base_digest(  # noqa: SLF001
        {"scene.usdc": "a" * 64, "textures/albedo.png": "b" * 64}
    )
    second = scene_generation._stock_base_digest(  # noqa: SLF001
        {"scene.usdc": "a" * 64, "textures/albedo.png": "c" * 64}
    )

    assert first != second


def test_sparse_construction_failure_retains_candidate_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fail_closure(*_args: object) -> object:
        raise scene_generation.SceneGenerationError("closure failed")

    monkeypatch.setattr(
        scene_generation,
        "_sparse_object_closure",
        fail_closure,
    )

    with pytest.raises(scene_generation.SceneGenerationError) as raised:
        scene_generation._reconcile_sparse_generation(  # noqa: SLF001
            tmp_path,
            object(),
            SimpleNamespace(number=0),
            (scene_generation.BlenderId("MESH", 9),),
        )

    retained = list(tmp_path.glob("failed-candidate-000001-*"))
    assert len(retained) == 1
    assert raised.value.diagnostics[-1]["candidate_artifact"] == str(retained[0])


def test_sparse_light_closure_includes_supported_parent() -> None:
    mesh_data = SimpleNamespace(session_uid=31, materials=())
    parent = SimpleNamespace(
        name_full="Parent",
        type="MESH",
        session_uid=11,
        data=mesh_data,
        parent=None,
    )
    light = SimpleNamespace(
        name_full="Point",
        type="LIGHT",
        session_uid=12,
        data=SimpleNamespace(session_uid=32),
        parent=parent,
    )
    parent_id = scene_generation.BlenderId("OBJECT", 11)
    light_id = scene_generation.BlenderId("OBJECT", 12)
    predecessor = SimpleNamespace(
        blender_prim_paths={
            parent_id: scene_generation.BlenderPrimPath(
                "Parent",
                "MESH",
                "/World/Parent",
                "/World/Parent/Parent",
                31,
            )
        }
    )

    objects, removed, materials = scene_generation._sparse_object_closure(  # noqa: SLF001
        SimpleNamespace(objects=(parent, light)),
        predecessor,
        (light_id,),
    )

    assert objects == (parent, light)
    assert removed == set()
    assert materials == set()


def test_sparse_closure_selects_light_owner_from_light_data_identity() -> None:
    light_data = SimpleNamespace(session_uid=22)
    light = SimpleNamespace(
        session_uid=11,
        type="LIGHT",
        data=light_data,
        parent=None,
    )
    predecessor = SimpleNamespace(
        blender_prim_paths={
            scene_generation.BlenderId("OBJECT", 11): scene_generation.BlenderPrimPath(
                "Key",
                "LIGHT",
                "/World/Key",
                "/World/Key/Key",
                data_session_uid=22,
            )
        }
    )

    objects, removed, materials = scene_generation._sparse_object_closure(  # noqa: SLF001
        SimpleNamespace(objects=(light,)),
        predecessor,
        (scene_generation.BlenderId("LIGHT", 22),),
    )

    assert objects == (light,)
    assert removed == set()
    assert materials == set()


def test_sparse_material_closure_ignores_light_data() -> None:
    material = SimpleNamespace(session_uid=41)
    mesh = SimpleNamespace(
        name_full="Mesh",
        type="MESH",
        session_uid=11,
        data=SimpleNamespace(session_uid=31, materials=(material,)),
        parent=None,
    )
    light = SimpleNamespace(
        name_full="Sun",
        type="LIGHT",
        session_uid=12,
        data=SimpleNamespace(session_uid=32),
        parent=None,
    )
    predecessor = SimpleNamespace(blender_prim_paths={})

    objects, _removed, materials = scene_generation._sparse_object_closure(  # noqa: SLF001
        SimpleNamespace(objects=(mesh, light)),
        predecessor,
        (scene_generation.BlenderId("MATERIAL", 41),),
    )

    assert objects == (mesh,)
    assert materials == {scene_generation.BlenderId("MATERIAL", 41)}


def test_export_time_dirty_ids_are_noop_when_topology_is_unchanged() -> None:
    material = SimpleNamespace(session_uid=41, name_full="Hero Blue")
    mesh = SimpleNamespace(
        session_uid=31,
        name_full="Cube",
        vertices=(
            SimpleNamespace(co=(0.0, 0.0, 0.0)),
            SimpleNamespace(co=(1.0, 0.0, 0.0)),
            SimpleNamespace(co=(0.0, 1.0, 0.0)),
        ),
        polygons=(SimpleNamespace(vertices=(0, 1, 2)),),
        materials=(material,),
    )
    light_data = SimpleNamespace(session_uid=32, name_full="Key Light")
    mesh_object = SimpleNamespace(
        session_uid=11,
        name_full="Hero Cube",
        type="MESH",
        data=mesh,
        parent=None,
    )
    light_object = SimpleNamespace(
        session_uid=12,
        name_full="Key Light",
        type="LIGHT",
        data=light_data,
        parent=None,
    )
    scene = SimpleNamespace(objects=(mesh_object, light_object))
    generation = SimpleNamespace(
        blender_prim_paths={
            scene_generation.BlenderId("OBJECT", 11): scene_generation.BlenderPrimPath(
                "Hero Cube", "MESH", "/World/Hero_Cube", "/World/Hero_Cube/Cube", 31
            ),
            scene_generation.BlenderId("OBJECT", 12): scene_generation.BlenderPrimPath(
                "Key Light", "LIGHT", "/World/Key_Light", "/World/Key_Light/Area", 32
            ),
        },
        topology_fingerprints=scene_generation._topology_fingerprints(scene),  # noqa: SLF001
    )
    affected = (
        scene_generation.BlenderId("OBJECT", 11),
        scene_generation.BlenderId("OBJECT", 12),
        scene_generation.BlenderId("MESH", 31),
        scene_generation.BlenderId("LIGHT", 32),
        scene_generation.BlenderId("MATERIAL", 41),
    )
    object_ids = scene_generation._affected_object_ids(  # noqa: SLF001
        scene,
        generation,
        affected,
    )
    current = scene_generation._topology_fingerprints(scene, object_ids)  # noqa: SLF001

    assert scene_generation._is_noop_topology_edit(  # noqa: SLF001
        generation,
        affected,
        current,
    )


def test_light_form_change_is_not_a_noop_topology_edit() -> None:
    light_data = SimpleNamespace(
        session_uid=32,
        name_full="Key Light",
        type="POINT",
        shape="",
    )
    light_object = SimpleNamespace(
        session_uid=12,
        name_full="Key Light",
        type="LIGHT",
        data=light_data,
        parent=None,
    )
    scene = SimpleNamespace(objects=(light_object,))
    generation = SimpleNamespace(
        blender_prim_paths={
            scene_generation.BlenderId("OBJECT", 12): scene_generation.BlenderPrimPath(
                "Key Light", "LIGHT", "/World/Key_Light", "/World/Key_Light/Area", 32
            ),
        },
        topology_fingerprints=scene_generation._topology_fingerprints(scene),  # noqa: SLF001
    )

    light_data.type = "SUN"
    affected = (scene_generation.BlenderId("LIGHT", 32),)
    object_ids = scene_generation._affected_object_ids(  # noqa: SLF001
        scene,
        generation,
        affected,
    )
    current = scene_generation._topology_fingerprints(scene, object_ids)  # noqa: SLF001

    assert not scene_generation._is_noop_topology_edit(  # noqa: SLF001
        generation,
        affected,
        current,
    )


def test_cleanup_failure_retains_generation_state(monkeypatch, tmp_path: Path) -> None:
    def fail_cleanup(_path: Path) -> None:
        raise OSError("busy")

    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    generation = object()
    owner._current = generation  # noqa: SLF001
    monkeypatch.setattr(
        scene_generation.shutil,
        "rmtree",
        fail_cleanup,
    )

    with pytest.raises(scene_generation.SceneGenerationError, match="cleanup failed"):
        owner.close()

    assert owner.current_generation is generation


def test_scene_owner_rebinds_retained_values_by_blender_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    object_id = scene_generation.BlenderId("OBJECT", 10)
    material_id = scene_generation.BlenderId("MATERIAL", 20)
    predecessor = SimpleNamespace(
        blender_prim_paths={
            object_id: scene_generation.BlenderPrimPath(
                "Cube", "MESH", "/World/Cube", "/World/Cube/Cube"
            ),
            material_id: scene_generation.BlenderPrimPath(
                "Material", "MATERIAL", "/World/Materials/Material", "/World/Materials/Material"
            ),
        }
    )
    candidate = SimpleNamespace(
        materialize_usd=lambda: "/tmp/candidate.usdc",
        blender_prim_paths={
            object_id: scene_generation.BlenderPrimPath(
                "Cube", "MESH", "/World/__ovrtx/generation_1/Cube", "/World/__ovrtx/generation_1/Cube/Cube"
            ),
            material_id: scene_generation.BlenderPrimPath(
                "Material", "MATERIAL", "/World/__ovrtx/generation_1/Materials/Material", "/World/__ovrtx/generation_1/Materials/Material"
            ),
        }
    )
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner.retain_transform_values(
        (OvrtxTransformValue("/World/Cube", ((1.0,),)),)
    )
    owner.retain_attribute_values(
        (
            OvrtxAttributeValue(
                "/World/Materials/Material/Shader", "inputs:roughness", 0.4, "Float"
            ),
        )
    )

    class Prim:
        def __init__(self, path: str, attributes: tuple[str, ...] = ()) -> None:
            self.path = path
            self.attributes = attributes

        def GetPath(self) -> str:
            return self.path

        def HasAttribute(self, attribute: str) -> bool:
            return attribute in self.attributes

    stage = SimpleNamespace(
        Traverse=lambda: (
            Prim("/World/__ovrtx/generation_1/Materials/Material"),
            Prim(
                "/World/__ovrtx/generation_1/Materials/Material/RenamedShader",
                ("inputs:roughness",),
            ),
        )
    )
    pxr = SimpleNamespace(Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda _path: stage)))
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    transforms, attributes, _initial_conditions = owner.retained_values_for(candidate)

    assert transforms[0].prim_path == "/World/__ovrtx/generation_1/Cube"
    assert attributes[0].prim_path == "/World/__ovrtx/generation_1/Materials/Material/RenamedShader"

    owner._current = candidate  # noqa: SLF001
    owner.retain_attribute_values(
        (
            OvrtxAttributeValue(
                "/World/__ovrtx/generation_1/Materials/Material/RenamedShader",
                "inputs:roughness",
                0.8,
                "Float",
            ),
        )
    )
    _transforms, updated_attributes, _initial_conditions = owner.retained_values_for(
        candidate
    )

    assert len(updated_attributes) == 1
    assert updated_attributes[0].value == 0.8


def test_rejected_generation_preserves_predecessor_retained_values(tmp_path: Path) -> None:
    object_id = scene_generation.BlenderId("OBJECT", 10)
    mapping = scene_generation.BlenderPrimPath(
        "Cube", "MESH", "/World/Cube", "/World/Cube/Cube"
    )
    predecessor = SimpleNamespace(blender_prim_paths={object_id: mapping})
    candidate = SimpleNamespace(blender_prim_paths={})
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner.retain_transform_values(
        (OvrtxTransformValue("/World/Cube", ((1.0,),)),)
    )
    owner._pending = candidate  # noqa: SLF001

    assert owner.retained_values_for(candidate)[0] == ()
    owner.reject(candidate)

    assert owner.retained_values_for(predecessor)[0][0].prim_path == "/World/Cube"


def test_accept_prunes_removed_identity_values_and_rebinds_initial_conditions(
    tmp_path: Path,
) -> None:
    object_id = scene_generation.BlenderId("OBJECT", 10)
    predecessor = SimpleNamespace(
        blender_prim_paths={
            object_id: scene_generation.BlenderPrimPath(
                "Cube", "MESH", "/World/Cube", "/World/Cube/Cube"
            )
        }
    )
    rebound = SimpleNamespace(
        blender_prim_paths={
            object_id: scene_generation.BlenderPrimPath(
                "Cube", "MESH", "/World/Version/Cube", "/World/Version/Cube/Cube"
            )
        }
    )
    removed = SimpleNamespace(blender_prim_paths={})
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner.retain_transform_values(
        (OvrtxTransformValue("/World/Cube", ((1.0,),)),)
    )
    owner.retain_initial_conditions(
        (
            BodyPose(
                prim_path="/World/Cube",
                translate=(0.0, 0.0, 1.0),
                orient=(0.0, 0.0, 0.0, 1.0),
            ),
        )
    )

    assert owner.retained_values_for(rebound)[2][0].prim_path == "/World/Version/Cube"
    owner._pending = removed  # noqa: SLF001
    owner.accept(removed)

    assert owner.retained_values_for(removed) == ((), (), ())


def test_scene_owner_retains_supported_world_dome_values_across_generations(
    tmp_path: Path,
) -> None:
    world_id = scene_generation.BlenderId("WORLD", 30)
    world_mapping = scene_generation.BlenderPrimPath(
        "World",
        "WORLD",
        DEFAULT_DOME_OWNER_PATH,
        DEFAULT_DOME_OWNER_PATH,
    )
    predecessor = SimpleNamespace(blender_prim_paths={world_id: world_mapping})
    candidate = SimpleNamespace(blender_prim_paths={world_id: world_mapping})
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner.retain_attribute_values(
        (
            OvrtxAttributeValue(
                DEFAULT_DOME_OWNER_PATH,
                "inputs:intensity",
                1133.8,
                "Float",
            ),
        )
    )

    _transforms, attributes, _initial_conditions = owner.retained_values_for(candidate)

    assert attributes == (
        OvrtxAttributeValue(
            DEFAULT_DOME_OWNER_PATH,
            "inputs:intensity",
            1133.8,
            "Float",
        ),
    )


def test_world_reconciliation_builds_a_full_pending_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    predecessor = SimpleNamespace(number=0)
    candidate = SimpleNamespace(number=1)
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    replacements = []

    def replace(scene: object) -> object:
        replacements.append(scene)
        owner._current = candidate  # noqa: SLF001
        return candidate

    monkeypatch.setattr(owner, "replace", replace)
    scene = object()

    assert owner.reconcile(scene, {scene_generation.BlenderId("WORLD", 30)}) is candidate
    assert replacements == [scene]
    assert owner.current_generation is predecessor
    assert owner.pending_generation is candidate


def test_reconciliation_numbers_advance_past_rejected_candidates(
    monkeypatch,
    tmp_path: Path,
) -> None:
    predecessor = SimpleNamespace(number=4)
    rejected = SimpleNamespace(number=5)
    candidate = SimpleNamespace(number=6)
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner._failed.append(rejected)  # noqa: SLF001
    numbers = []
    monkeypatch.setattr(scene_generation, "_affected_object_ids", lambda *_args: set())
    monkeypatch.setattr(scene_generation, "_topology_fingerprints", lambda *_args: {})
    monkeypatch.setattr(scene_generation, "_is_noop_topology_edit", lambda *_args: False)
    monkeypatch.setattr(
        scene_generation,
        "_reconcile_sparse_generation",
        lambda *_args, number, **_kwargs: numbers.append(number) or candidate,
    )

    assert owner.reconcile(
        object(), {scene_generation.BlenderId("MESH", 30)}
    ) is candidate
    assert numbers == [6]


def test_scene_owner_does_not_replay_light_value_after_datablock_replacement(
    tmp_path: Path,
) -> None:
    object_id = scene_generation.BlenderId("OBJECT", 10)
    predecessor = SimpleNamespace(
        blender_prim_paths={
            object_id: scene_generation.BlenderPrimPath(
                "Key",
                "LIGHT",
                "/World/Key",
                "/World/Key/Key",
                data_session_uid=20,
            )
        }
    )
    replacement = SimpleNamespace(
        blender_prim_paths={
            object_id: scene_generation.BlenderPrimPath(
                "Key",
                "LIGHT",
                "/World/Version/Key",
                "/World/Version/Key/Key",
                data_session_uid=21,
            )
        }
    )
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner.retain_attribute_values(
        (
            OvrtxAttributeValue(
                "/World/Key/Key",
                "inputs:intensity",
                500.0,
                "Float",
            ),
        )
    )

    assert owner.retained_values_for(predecessor)[1][0].value == 500.0
    assert owner.retained_values_for(replacement)[1] == ()

    owner._pending = replacement  # noqa: SLF001
    owner.accept(replacement)
    assert owner.retained_values_for(replacement)[1] == ()


def test_scene_owner_rebinds_light_value_by_datablock_identity(
    tmp_path: Path,
) -> None:
    predecessor = SimpleNamespace(
        blender_prim_paths={
            scene_generation.BlenderId("OBJECT", 10): scene_generation.BlenderPrimPath(
                "Key", "LIGHT", "/World/Key", "/World/Key/Key", data_session_uid=20
            )
        }
    )
    replacement_owner = SimpleNamespace(
        blender_prim_paths={
            scene_generation.BlenderId("OBJECT", 11): scene_generation.BlenderPrimPath(
                "MovedKey",
                "LIGHT",
                "/World/MovedKey",
                "/World/MovedKey/Key",
                data_session_uid=20,
            )
        }
    )
    owner = scene_generation.SceneGenerationOwner(tmp_path / "generations")
    owner._current = predecessor  # noqa: SLF001
    owner.retain_attribute_values(
        (
            OvrtxAttributeValue(
                "/World/Key/Key", "inputs:intensity", 500.0, "Float"
            ),
        )
    )

    attributes = owner.retained_values_for(replacement_owner)[1]

    assert attributes == (
        OvrtxAttributeValue(
            "/World/MovedKey/Key", "inputs:intensity", 500.0, "Float"
        ),
    )
