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
        lambda path, _records, _deltas, base, **_kwargs: path.write_bytes(base.read_bytes()),
    )


@pytest.mark.parametrize('kind', ['CURVE', 'CURVES', 'SURFACE', 'FONT', 'META'])
def test_expanded_geometry_data_keeps_complete_export_owner_closure(kind):
    identity = scene_generation.BlenderId(kind, 1)
    assert scene_generation._expanded_visibility_owner_changed(SimpleNamespace(objects=[]), [identity])


@pytest.mark.parametrize('notification,current_particles', [('OBJECT', True), ('OBJECT', False), ('MESH', False)])
def test_particle_owner_replacement_precedes_direct_updates(monkeypatch, tmp_path, notification, current_particles):
    from ovrtx_blender_example import direct_scene_updates
    object_id = scene_generation.BlenderId('OBJECT', 11)
    prior = SimpleNamespace(particle_geometry_owners={} if current_particles else {object_id: 31})
    objects = [SimpleNamespace(type='MESH', session_uid=11, data=SimpleNamespace(session_uid=31),
                              particle_systems=[object()])] if current_particles else []
    owner = scene_generation.SceneGenerationOwner(tmp_path)
    owner._current = prior
    candidate = SimpleNamespace(number=2)
    monkeypatch.setattr(owner, 'replace', lambda _: candidate)
    monkeypatch.setattr(direct_scene_updates, 'plan', lambda *_: pytest.fail('Partial direct edit ran before expanded-owner replacement'))
    changed = scene_generation.BlenderId(notification, 11 if notification == 'OBJECT' else 31)
    assert owner.reconcile(SimpleNamespace(objects=objects), [changed]) is candidate
    assert owner._current is prior and owner._pending is candidate


def test_mesh_shading_controls_avoid_elements_and_detect_binding_visibility_modifier_changes():
    class Elements:
        def __iter__(self):
            pytest.fail("Shading guard read per-element mesh data")
    material = SimpleNamespace(session_uid=3)
    data = SimpleNamespace(name_full="Mesh", session_uid=2, materials=[material],
        vertices=Elements(), corner_normals=Elements(),
        shape_keys=SimpleNamespace(key_blocks=[SimpleNamespace(name="Key", data=Elements(), value=0)]))
    modifier = SimpleNamespace(type="SUBSURF", show_viewport=True, show_render=True)
    obj = SimpleNamespace(name_full="Object", type="MESH", session_uid=1, data=data,
        parent=None, modifiers=[modifier], material_slots=[], hide_render=False)
    original = scene_generation._mesh_shading_inputs(obj)
    assert scene_generation._mesh_shading_inputs(obj) == original
    obj.hide_render = True
    assert scene_generation._mesh_shading_inputs(obj) != original
    obj.hide_render = False
    modifier.show_render = False
    assert scene_generation._mesh_shading_inputs(obj) != original
    modifier.show_render = True
    data.materials = [SimpleNamespace(session_uid=4)]
    assert scene_generation._mesh_shading_inputs(obj) != original


@pytest.mark.parametrize('paths', [('/World/env_light',), ('/World/StudioDome',)])
def test_owned_stock_export_maps_unique_world_dome(monkeypatch, paths):
    prims=[SimpleNamespace(GetPath=lambda path=path:path,GetTypeName=lambda:'DomeLight',GetAttribute=lambda name:None) for path in paths]
    stage=SimpleNamespace(Traverse=lambda:iter(prims),GetPrimAtPath=lambda path:next((p for p in prims if p.GetPath()==path),None))
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda path:stage))))
    monkeypatch.setattr(scene_generation,'_blender_module_provider',lambda:SimpleNamespace(data=SimpleNamespace(materials=[])))
    scene=SimpleNamespace(objects=[],world=SimpleNamespace(session_uid=30,name_full='World'))
    mappings=scene_generation._validated_blender_prim_paths(scene,Path('owned.usda'))
    assert mappings[scene_generation.BlenderId('WORLD',30)].schema_path==paths[0]


def test_owned_stock_export_rejects_ambiguous_world_domes(monkeypatch):
    prims=[SimpleNamespace(GetPath=lambda path=path:path,GetTypeName=lambda:'DomeLight',GetAttribute=lambda name:None) for path in ('/World/env_light','/World/other')]
    stage=SimpleNamespace(Traverse=lambda:iter(prims),GetPrimAtPath=lambda path:None)
    monkeypatch.setitem(sys.modules,'pxr',SimpleNamespace(Usd=SimpleNamespace(Stage=SimpleNamespace(Open=lambda path:stage))))
    monkeypatch.setattr(scene_generation,'_blender_module_provider',lambda:SimpleNamespace(data=SimpleNamespace(materials=[])))
    scene=SimpleNamespace(objects=[],world=SimpleNamespace(session_uid=30,name_full='World'))
    with pytest.raises(scene_generation.SceneGenerationError,match='World dome identity is not unique'):
        scene_generation._validated_blender_prim_paths(scene,Path('owned.usda'))


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


    # An unchanged material ID says nothing about its node connections.
    assert not scene_generation._is_noop_topology_edit(
        generation, (*affected, scene_generation.BlenderId("MATERIAL", 41)), current
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


def _canonical_world_predecessor(monkeypatch):
    from ovrtx_blender_example import world_dome_conversion
    monkeypatch.setattr(world_dome_conversion, 'environment_texture_dome',
                        lambda world: {'texture_file': 'sky.exr'} if world else None)
    world_id = scene_generation.BlenderId('WORLD', 30)
    return SimpleNamespace(number=0, blender_prim_paths={
        world_id: scene_generation.BlenderPrimPath('Morning', 'WORLD',
            DEFAULT_DOME_OWNER_PATH, DEFAULT_DOME_OWNER_PATH)},
        opinion_records=(SimpleNamespace(usd_prim_path=DEFAULT_DOME_OWNER_PATH,
            diagnostics={'source': 'world_environment'}),), topology_fingerprints={})


@pytest.mark.parametrize('mesh_changed', [False, True])
def test_canonical_world_reconciliation_preserves_geometry_and_refreshes_identity(monkeypatch, tmp_path, mesh_changed):
    prior = _canonical_world_predecessor(monkeypatch)
    obj_id = scene_generation.BlenderId('OBJECT', 11)
    obj = SimpleNamespace(name_full='Drawer', type='MESH', session_uid=11,
                          data=SimpleNamespace(session_uid=31, materials=()), parent=None)
    prior.blender_prim_paths[obj_id] = scene_generation.BlenderPrimPath(
        'Drawer', 'MESH', '/World/Drawer', '/World/Drawer/Drawer', 31)
    prior.topology_fingerprints[obj_id] = 'same'
    scene = SimpleNamespace(objects=(obj,), world=SimpleNamespace(name_full='Afternoon', session_uid=40))
    world_id = scene_generation.BlenderId('WORLD', 40)
    owner = scene_generation.SceneGenerationOwner(tmp_path / 'generations')
    owner._current = prior
    for guard in ('_shadow_visibility_changed', '_visibility_requires_full_export', '_expanded_visibility_owner_changed'):
        monkeypatch.setattr(scene_generation, guard, lambda *_: False)
    monkeypatch.setattr(scene_generation, '_affected_object_ids', lambda *_: {obj_id})
    monkeypatch.setattr(scene_generation, '_topology_fingerprints', lambda *_: {obj_id: 'changed' if mesh_changed else 'same'})
    monkeypatch.setattr(owner, 'replace', lambda *_: pytest.fail('Canonical World exported entire scene'))
    calls = []
    candidate = SimpleNamespace(number=1)
    def sparse(directory, current, predecessor, affected, *, number):
        objects, removed, materials = scene_generation._sparse_object_closure(current, predecessor, affected)
        assert objects == ((obj,) if mesh_changed else ())
        assert not removed and not materials
        mappings = scene_generation._refresh_canonical_world_mapping(current, predecessor,
                    predecessor.blender_prim_paths, affected)
        assert scene_generation.BlenderId('WORLD', 30) not in mappings
        assert mappings[world_id].schema_path == DEFAULT_DOME_OWNER_PATH
        assert mappings[world_id].blender_id_name == 'Afternoon'
        calls.append(affected)
        return candidate
    monkeypatch.setattr(scene_generation, '_reconcile_sparse_generation', sparse)
    assert owner.reconcile(scene, {world_id, obj_id}) is candidate
    assert calls == [tuple(sorted({world_id, obj_id} if mesh_changed else {world_id}))]
    assert owner.current_generation is prior and owner.pending_generation is candidate


@pytest.mark.parametrize('unsupported', ['unowned', 'noncanonical', 'unsupported_graph'])
def test_world_sparse_guard_keeps_conservative_full_export(monkeypatch, tmp_path, unsupported):
    prior = _canonical_world_predecessor(monkeypatch)
    scene = SimpleNamespace(world=SimpleNamespace(name_full='Night', session_uid=40))
    if unsupported == 'unowned': prior.opinion_records = ()
    elif unsupported == 'noncanonical':
        identity = scene_generation.BlenderId('WORLD', 30)
        prior.blender_prim_paths[identity] = scene_generation.BlenderPrimPath('Stock', 'WORLD', '/World/env_light', '/World/env_light')
    else:
        from ovrtx_blender_example import world_dome_conversion
        monkeypatch.setattr(world_dome_conversion, 'environment_texture_dome', lambda *_: None)
    owner = scene_generation.SceneGenerationOwner(tmp_path / 'generations')
    owner._current = prior
    candidate = SimpleNamespace(number=1)
    monkeypatch.setattr(owner, 'replace', lambda *_: candidate)
    assert owner.reconcile(scene, {scene_generation.BlenderId('WORLD', 40)}) is candidate


def _asset_opinion(path, value):
    from pxr import Sdf
    layer = Sdf.Layer.CreateAnonymous('.usda')
    prim = Sdf.CreatePrimInLayer(layer, path)
    Sdf.AttributeSpec(prim, 'texture', Sdf.ValueTypeNames.Asset).default = Sdf.AssetPath(value)
    return scene_generation.AddOnUsdOpinionRecord(path, layer)


@pytest.mark.parametrize('meters,axis', [(1.0, 'Z'), (0.01, 'Y'), (0.125, 'Z')])
def test_stock_metadata_survives_generation_wrappers_and_sparse_reuse(tmp_path, meters, axis):
    from pxr import Sdf, Usd, UsdGeom

    base = tmp_path / 'export.usda'
    source = Usd.Stage.CreateNew(str(base))
    world = source.DefinePrim('/World')
    source.SetDefaultPrim(world)
    metadata = dict(metersPerUnit=meters, upAxis=axis, kilogramsPerUnit=0.5,
                    startTimeCode=-2.5, endTimeCode=91.25,
                    timeCodesPerSecond=48.0, framesPerSecond=24.0,
                    defaultPrim='World')
    for key, value in metadata.items():
        source.SetMetadata(key, value)
    source.GetRootLayer().Save()
    original = base.read_bytes()
    binding = tmp_path / 'identity.usda'
    binding.write_text('#usda 1.0\nover "World" {\n custom int identity = 7\n}\n')
    stock = tmp_path / 'stock.usda'
    scene_generation._write_stock_root(stock, binding, base)

    opinion = _asset_opinion('/World/Material', 'texture.png')
    first = tmp_path / 'first'
    first.mkdir()
    composed = first / 'composed.usda'
    scene_generation._write_layered_generation(composed, (opinion,), (), stock)
    # Sparse exports can have their own headers. They must not replace the
    # complete stock base's stage interpretation or timeline.
    delta = tmp_path / 'delta.usda'
    delta.write_text('#usda 1.0\n(metersPerUnit = 7\nupAxis = "Y"\n'
                     'startTimeCode = 999)\nover "World" {\n custom int revision = 2\n}\n')
    deltas = (SimpleNamespace(usd_path=str(delta)),)
    second = tmp_path / 'second'
    second.mkdir()
    sparse_stock = second / 'stock.usda'
    scene_generation._write_layered_generation(sparse_stock, (), deltas, stock)
    sparse = second / 'composed.usda'
    scene_generation._write_layered_generation(sparse, (opinion,), deltas, stock,
        predecessor=SimpleNamespace(usd_path=str(composed), opinion_records=(opinion,)))
    assert not (second / 'opinion-0000.usda').exists()
    for path in (stock, composed, sparse_stock, sparse):
        stage = Usd.Stage.Open(str(path))
        assert UsdGeom.GetStageMetersPerUnit(stage) == meters
        assert UsdGeom.GetStageUpAxis(stage) == axis
        assert stage.GetDefaultPrim().GetPath() == Sdf.Path('/World')
        for key, value in metadata.items():
            assert stage.GetRootLayer().pseudoRoot.HasInfo(key), (path, key)
            assert stage.GetMetadata(key) == value, (path, key)
        assert stage.GetPrimAtPath('/World').GetAttribute('identity').Get() == 7
    assert stage.GetPrimAtPath('/World').GetAttribute('revision').Get() == 2
    assert base.read_bytes() == original


def test_generation_wrappers_do_not_author_absent_stage_metadata(tmp_path):
    from pxr import Sdf, Usd, UsdGeom

    base = tmp_path / 'export.usda'
    base.write_text('#usda 1.0\ndef "World" {}\n')
    binding = tmp_path / 'identity.usda'
    binding.write_text('#usda 1.0\n')
    stock = tmp_path / 'stock.usda'
    scene_generation._write_stock_root(stock, binding, base)
    composed = tmp_path / 'composed.usda'
    stage = scene_generation._write_layered_generation(composed, (), (), stock)
    for path in (stock, composed):
        root = Sdf.Layer.FindOrOpen(str(path))
        assert set(root.pseudoRoot.ListInfoKeys()) == {'subLayers', 'subLayerOffsets'}
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.01
    assert UsdGeom.GetStageUpAxis(stage) == 'Y'
    assert not stage.GetDefaultPrim()


def test_layered_generation_reuses_unchanged_opinions_and_keeps_asset_context(tmp_path):
    from pxr import Sdf, Usd
    base = tmp_path / 'base.usda'
    base.write_text('#usda 1.0\ndef "World" {}\n')
    original = tmp_path / 'original'
    original.mkdir()
    (original / 'textures').mkdir()
    (original / 'textures' / 'oak.png').write_bytes(b'asset')
    oak = _asset_opinion('/World/Oak', 'textures/oak.png')
    old_sky = _asset_opinion('/World/Sky', 'old.exr')
    first = original / 'composed.usda'
    scene_generation._write_layered_generation(first, (oak, old_sky), (), base)
    prior = SimpleNamespace(usd_path=str(first), opinion_records=(oak, old_sky))
    retained = original / 'opinion-0000.usda'
    before = retained.stat()
    changed = _asset_opinion('/World/Sky', 'new.exr')
    next_dir = tmp_path / 'next'
    next_dir.mkdir()
    second = next_dir / 'composed.usda'
    scene_generation._write_layered_generation(second, (oak, changed), (), base, predecessor=prior)
    root = Sdf.Layer.FindOrOpen(str(second))
    assert Path(Sdf.ComputeAssetPathRelativeToLayer(root, root.subLayerPaths[0])) == retained
    assert not (next_dir / 'opinion-0000.usda').exists()
    assert (next_dir / 'opinion-0001.usda').is_file()
    assert retained.stat().st_mtime_ns == before.st_mtime_ns
    stage = Usd.Stage.Open(str(second))
    asset = stage.GetPrimAtPath('/World/Oak').GetAttribute('texture').Get()
    assert Path(asset.resolvedPath) == original / 'textures' / 'oak.png'
    # A reused record no longer lives beside the immediate predecessor root.
    # Follow its actual sublayer path on subsequent generations as well.
    third_dir = tmp_path / 'third'
    third_dir.mkdir()
    third = third_dir / 'composed.usda'
    prior = SimpleNamespace(usd_path=str(second), opinion_records=(oak, changed))
    scene_generation._write_layered_generation(third, (oak, changed), (), base, predecessor=prior)
    assert not list(third_dir.glob('opinion-*.usda'))


@pytest.mark.parametrize('mutation', ['disk', 'memory'])
def test_opinion_reuse_rejects_mutated_accepted_files(tmp_path, mutation):
    from pxr import Sdf
    base = tmp_path / 'base.usda'
    base.write_text('#usda 1.0\ndef "World" {}\n')
    original = tmp_path / 'original'
    original.mkdir()
    record = _asset_opinion('/World/Oak', 'oak.png')
    first = original / 'composed.usda'
    scene_generation._write_layered_generation(first, (record,), (), base)
    opinion = original / 'opinion-0000.usda'
    if mutation == 'disk': opinion.write_text('#usda 1.0\nover "World" {}\n')
    else:
        layer = Sdf.Layer.FindOrOpen(str(opinion))
        layer.GetAttributeAtPath('/World/Oak.texture').default = Sdf.AssetPath('changed.png')
    next_dir = tmp_path / 'next'
    next_dir.mkdir()
    second = next_dir / 'composed.usda'
    scene_generation._write_layered_generation(second, (record,), (), base,
        predecessor=SimpleNamespace(usd_path=str(first), opinion_records=(record,)))
    assert (next_dir / 'opinion-0000.usda').is_file()


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


def test_material_graph_with_unchanged_object_tags_reaches_pending_generation(monkeypatch, tmp_path):
    object_id = scene_generation.BlenderId('OBJECT', 11)
    material_id = scene_generation.BlenderId('MATERIAL', 41)
    predecessor = SimpleNamespace(number=3, blender_prim_paths={},
                                  topology_fingerprints={object_id:'unchanged-mesh'})
    candidate = SimpleNamespace(number=4)
    owner = scene_generation.SceneGenerationOwner(tmp_path / 'generations')
    owner._current = predecessor
    monkeypatch.setattr(scene_generation, '_affected_object_ids', lambda *_: {object_id})
    monkeypatch.setattr(scene_generation, '_topology_fingerprints', lambda *_: {object_id:'unchanged-mesh'})
    calls = []
    def reconcile(directory, scene, prior, affected, *, number):
        calls.append((prior, affected, number))
        return candidate
    monkeypatch.setattr(scene_generation, '_reconcile_sparse_generation', reconcile)
    assert owner.reconcile(object(), {object_id}) is None
    assert calls == []
    assert owner.reconcile(object(), {object_id, material_id}) is candidate
    assert calls == [(predecessor, (material_id,), 4)]
    assert owner.current_generation is predecessor
    assert owner.pending_generation is candidate


@pytest.mark.parametrize("retain_old,add_new", [(True, True), (False, False), (False, True)])
def test_sparse_camera_add_delete_replace_matches_topology_detection(retain_old, add_new):
    old_id = scene_generation.BlenderId("OBJECT", 161)
    new_id = scene_generation.BlenderId("OBJECT", 181)
    def camera(uid):
        return SimpleNamespace(session_uid=uid, type="CAMERA", parent=None,
                               data=SimpleNamespace(session_uid=uid + 1))
    old, new = camera(161), camera(181)
    objects = ([old] if retain_old else []) + ([new] if add_new else [])
    predecessor = SimpleNamespace(blender_prim_paths={
        old_id: scene_generation.BlenderPrimPath("Camera", "CAMERA",
            "/World/Camera", "/World/Camera/Camera", 162)})
    affected = ([old_id] if not retain_old else []) + ([new_id] if add_new else [])
    selected, removed, materials = scene_generation._sparse_object_closure(
        SimpleNamespace(objects=objects), predecessor, tuple(affected))
    assert selected == ((new,) if add_new else ())
    assert removed == (set() if retain_old else {old_id})
    assert materials == set()


def test_sparse_unknown_object_still_rejected():
    with pytest.raises(scene_generation.SceneGenerationError, match="unsupported affected"):
        scene_generation._sparse_object_closure(
            SimpleNamespace(objects=()), SimpleNamespace(blender_prim_paths={}, opinion_records=()),
            (scene_generation.BlenderId("OBJECT", 999),))


def test_retained_revision_changes_only_with_actual_scene_value(tmp_path):
    identity = scene_generation.BlenderId('OBJECT', 10)
    mapping = scene_generation.BlenderPrimPath('Cube', 'MESH', '/World/Cube', '/World/Cube/Cube')
    owner = scene_generation.SceneGenerationOwner(tmp_path / 'generations')
    owner._current = SimpleNamespace(blender_prim_paths={identity: mapping})
    first = OvrtxTransformValue('/World/Cube', ((1.0,),))
    owner.retain_transform_values((first,))
    revision = owner.retained_revision
    assert revision > 0
    owner.retain_transform_values((first,))
    assert owner.retained_revision == revision
    owner.retain_transform_values((OvrtxTransformValue('/World/Cube', ((2.0,),)),))
    assert owner.retained_revision > revision


@pytest.mark.parametrize('source_type,expected', [('GROUP_INPUT', True), ('MESH_CUBE', False)])
def test_geometryless_control_proof_requires_static_identity_geometry(source_type, expected):
    geometry = SimpleNamespace(type='GEOMETRY')
    source = SimpleNamespace(type=source_type)
    output = SimpleNamespace(type='GROUP_OUTPUT', inputs=[SimpleNamespace(
        type='GEOMETRY', is_linked=True, links=[SimpleNamespace(from_node=source, from_socket=geometry)])])
    modifier = SimpleNamespace(type='NODES', node_group=SimpleNamespace(nodes=[source, output]))
    assert scene_generation._has_only_geometryless_node_modifiers(SimpleNamespace(modifiers=[modifier])) is expected


@pytest.mark.parametrize('vertices,instance,expected', [(0,False,True),(1,False,False),(0,True,False)])
def test_empty_control_proof_vetoes_source_geometry_and_instancing_without_evaluation(monkeypatch, vertices, instance, expected):
    obj = SimpleNamespace(type='MESH', session_uid=11,
        data=SimpleNamespace(vertices=[None]*vertices, polygons=[]), modifiers=[object()],
        is_instancer=instance)
    monkeypatch.setattr(scene_generation, '_blender_module_provider', lambda:pytest.fail('Static empty control proof evaluated whole dependency graph'))
    monkeypatch.setattr(scene_generation, '_has_only_geometryless_node_modifiers', lambda _:True)
    monkeypatch.setattr(scene_generation, '_mesh_shading_inputs', lambda _:('binding',('semantic','modifier-inputs','shape')))
    assert bool(scene_generation._empty_evaluated_mesh_proofs(SimpleNamespace(objects=[obj]))) is expected


def test_control_changes_skip_geometry_export_but_preserve_world_update(monkeypatch, tmp_path):
    prior = _canonical_world_predecessor(monkeypatch)
    obj_id, mesh_id = scene_generation.BlenderId('OBJECT',11), scene_generation.BlenderId('MESH',31)
    world_id = scene_generation.BlenderId('WORLD',40)
    obj = SimpleNamespace(type='MESH', session_uid=11, data=SimpleNamespace(session_uid=31))
    prior.blender_prim_paths[obj_id] = scene_generation.BlenderPrimPath('Control','MESH','/World/Control','/World/Control/Control',31)
    prior.topology_fingerprints[obj_id] = 'old-input'
    prior.empty_evaluated_mesh_proofs = {obj_id:'same-empty-output-and-export-guards'}
    scene = SimpleNamespace(objects=[obj], world=SimpleNamespace(name_full='Afternoon', session_uid=40))
    owner = scene_generation.SceneGenerationOwner(tmp_path)
    owner._current = prior
    for guard in ('_shadow_visibility_changed','_visibility_requires_full_export','_expanded_visibility_owner_changed'):
        monkeypatch.setattr(scene_generation,guard,lambda *_:False)
    monkeypatch.setattr(scene_generation,'_topology_fingerprints',lambda *_:{obj_id:'new-input'})
    monkeypatch.setattr(scene_generation,'_empty_evaluated_mesh_proofs',lambda *_:dict(prior.empty_evaluated_mesh_proofs))
    received = []
    candidate = SimpleNamespace(number=1)
    def sparse(*args,**kwargs):
        received.append(args[-1])
        return candidate
    monkeypatch.setattr(scene_generation,'_reconcile_sparse_generation',sparse)
    assert owner.reconcile(scene,{obj_id,mesh_id,world_id}) is candidate
    assert received == [(world_id,)]


@pytest.mark.parametrize('flags,controls_changed,has_modifiers,expected', [
    ((False,False,True),False,False,True),
    ((False,True,False),False,False,True),
    ((False,False,False),False,False,True),
    ((True,False,True),False,False,False),
    ((None,False,True),False,False,False),
    ((False,None,True),False,False,False),
    ((False,False,None),False,False,False),
    ((False,False,True),True,False,False),
    ((False,True,False),False,True,False),
    ((False,False,True),False,True,True),
])
def test_non_geometry_provenance_uses_flags_and_authored_controls(monkeypatch, flags, controls_changed, has_modifiers, expected):
    object_id = scene_generation.BlenderId('OBJECT',11)
    obj = SimpleNamespace(type='MESH',session_uid=11,data=SimpleNamespace(session_uid=31),
                          modifiers=[object()] if has_modifiers else [])
    prior = SimpleNamespace(mesh_shading_inputs={object_id:('accepted',)},topology_fingerprints={object_id:'mesh'})
    monkeypatch.setattr(scene_generation,'_mesh_shading_inputs',lambda _:('changed' if controls_changed else 'accepted',))
    proven = scene_generation._proven_non_geometry_objects(SimpleNamespace(objects=[obj]), prior,
        (object_id,),{object_id},{object_id:scene_generation.SceneUpdateProvenance(*flags)})
    assert proven == ({object_id} if expected else set())


def test_shared_mesh_geometry_and_unknown_notifications_cannot_be_overridden(monkeypatch):
    objects = [SimpleNamespace(type='MESH',session_uid=i,data=SimpleNamespace(session_uid=31),modifiers=[]) for i in (11,12)]
    object_ids = {scene_generation.blender_id(obj,'OBJECT') for obj in objects}
    mesh_id = scene_generation.BlenderId('MESH',31)
    prior = SimpleNamespace(mesh_shading_inputs={i:('accepted',) for i in object_ids},topology_fingerprints={i:'mesh' for i in object_ids})
    monkeypatch.setattr(scene_generation,'_mesh_shading_inputs',lambda _:('accepted',))
    provenance = {i:scene_generation.SceneUpdateProvenance(False,False,True) for i in object_ids}
    for mesh_note in (scene_generation.SceneUpdateProvenance(True,False,False),scene_generation.SceneUpdateProvenance()):
        provenance[mesh_id] = mesh_note
        assert scene_generation._proven_non_geometry_objects(SimpleNamespace(objects=objects),prior,
            tuple(object_ids|{mesh_id}),object_ids,provenance) == set()
    provenance[mesh_id] = scene_generation.SceneUpdateProvenance(False,False,True)
    assert scene_generation._proven_non_geometry_objects(SimpleNamespace(objects=objects),prior,
        tuple(object_ids|{mesh_id}),object_ids,provenance) == object_ids


def test_non_geometry_mixed_world_edit_never_hashes_mesh_elements(monkeypatch,tmp_path):
    prior = _canonical_world_predecessor(monkeypatch)
    obj_id,mesh_id,world_id = scene_generation.BlenderId('OBJECT',11),scene_generation.BlenderId('MESH',31),scene_generation.BlenderId('WORLD',40)
    obj = SimpleNamespace(type='MESH',session_uid=11,data=SimpleNamespace(session_uid=31),modifiers=[])
    prior.blender_prim_paths[obj_id] = scene_generation.BlenderPrimPath('Object','MESH','/World/Object','/World/Object/Object',31)
    prior.topology_fingerprints[obj_id] = 'accepted'
    prior.mesh_shading_inputs = {obj_id:('accepted-controls',)}
    scene = SimpleNamespace(objects=[obj],world=SimpleNamespace(name_full='Afternoon',session_uid=40))
    owner = scene_generation.SceneGenerationOwner(tmp_path)
    owner._current = prior
    for guard in ('_shadow_visibility_changed','_visibility_requires_full_export','_expanded_visibility_owner_changed'):
        monkeypatch.setattr(scene_generation,guard,lambda *_:False)
    monkeypatch.setattr(scene_generation,'_mesh_shading_inputs',lambda _:('accepted-controls',))
    def fingerprints(current,identities):
        assert not identities, 'Proven non-geometry owners must not hash elements'
        return {}
    monkeypatch.setattr(scene_generation,'_topology_fingerprints',fingerprints)
    candidate = SimpleNamespace(number=1)
    def sparse(directory,current,predecessor,affected,**kwargs):
        assert affected == (world_id,)
        return candidate
    monkeypatch.setattr(scene_generation,'_reconcile_sparse_generation',sparse)
    flags = scene_generation.SceneUpdateProvenance(False,False,True)
    assert owner.reconcile(scene,{obj_id,mesh_id,world_id},update_provenance={obj_id:flags,mesh_id:flags}) is candidate


def test_writer_stage_reused_by_validator_without_reopening(tmp_path,monkeypatch):
    from pxr import Usd
    base=tmp_path/'base.usda';base.write_text('#usda 1.0\ndef Xform "Root" {}\n')
    path=tmp_path/'composed.usda'
    stage=scene_generation._write_layered_generation(path,(),(),base)
    monkeypatch.setattr(Usd.Stage,'Open',lambda *_:pytest.fail('Validation reopened already composed Stage'))
    scene_generation._validate_composed_generation(path,{},set(),set(),stage=stage)


def test_shared_validation_retains_relationship_and_path_guards(tmp_path):
    from pxr import Usd
    path=tmp_path/'source.usda';path.write_text('#usda 1.0\ndef Xform "Root" { rel target=</Missing>\n}\n')
    stage=Usd.Stage.Open(str(path))
    with pytest.raises(scene_generation.SceneGenerationError,match='targets inactive'):
        scene_generation._validate_composed_generation(path,{},set(),set(),stage=stage)
    with pytest.raises(scene_generation.SceneGenerationError,match='does not match candidate'):
        scene_generation._validate_composed_generation(tmp_path/'other.usda',{},set(),set(),stage=stage)


def test_opinion_compilation_shares_stock_stage_between_visibility_and_world(tmp_path,monkeypatch):
    from pxr import Usd
    from ovrtx_blender_example import object_visibility,world_scene_layer
    stock=tmp_path/'stock.usda'
    stock.write_text('#usda 1.0\ndef Xform "Owner" { def Mesh "Child" {}\n}\ndef DomeLight "ExporterDome" {}\n')
    obj=SimpleNamespace(type='EMPTY',session_uid=11,name_full='Instance',instance_collection=object())
    scene=SimpleNamespace(objects=[obj],world=object())
    mapping=scene_generation.BlenderPrimPath('Instance','EMPTY','/Owner','/Owner')
    monkeypatch.setattr(object_visibility,'scene_visibility',lambda _:((obj,True),))
    monkeypatch.setattr(world_scene_layer,'environment_texture_dome',lambda _:dict(texture_file=str(tmp_path/'sky.exr'),intensity=1.,color=(1.,1.,1.)))
    original_open=Usd.Stage.Open
    opened=[]
    def tracked_open(source,*args,**kwargs):
        if str(source)==str(stock):opened.append(str(source))
        return original_open(source,*args,**kwargs)
    monkeypatch.setattr(Usd.Stage,'Open',tracked_open)
    records=scene_generation._compile_add_on_opinions(scene,{scene_generation.BlenderId('OBJECT',11):mapping},stock)
    assert opened==[str(stock)]
    assert {'/Owner/Child','/ExporterDome','/World/StudioDome'} <= {record.usd_prim_path for record in records}


_CAMERA_EXPORT = '''#usda 1.0
def Xform "World" {
    def Xform "Ortho" {
        custom int ovrtx:object_session_uid = 41
    }
    def Xform "Persp" {
        custom int ovrtx:object_session_uid = 42
        def Camera "PerspData" {
            token projection = "perspective"
            custom string userProperties:blender:data_name = "PerspData"
        }
    }
}
'''


def _camera(uid, name, data_name, camera_type, frame=((-2.0, -1.0), (2.0, 1.0))):
    corners = [SimpleNamespace(x=x, y=y, z=-1.0)
               for x in (frame[0][0], frame[1][0]) for y in (frame[0][1], frame[1][1])]
    data = SimpleNamespace(name=data_name, type=camera_type, lens=35.0, clip_start=0.25, clip_end=80.0,
                           view_frame=lambda scene=None: corners)
    return SimpleNamespace(type='CAMERA', session_uid=uid, name_full=name, name=name, data=data)


def test_orthographic_camera_gets_the_schema_stock_export_omits(tmp_path, monkeypatch):
    from pxr import Usd, UsdGeom
    path = tmp_path / 'export.usda'
    path.write_text(_CAMERA_EXPORT)
    ortho = _camera(41, 'Ortho', 'Ortho.001', 'ORTHO', frame=((-3.0, -1.0), (2.0, 1.5)))
    persp = _camera(42, 'Persp', 'PerspData', 'PERSP')
    scene = SimpleNamespace(objects=[ortho, persp])
    scene_generation._author_omitted_orthographic_cameras(path, scene)
    stage = Usd.Stage.Open(str(path))
    camera = UsdGeom.Camera(stage.GetPrimAtPath('/World/Ortho/Ortho_001'))
    assert camera and camera.GetProjectionAttr().Get() == 'orthographic'
    assert camera.GetHorizontalApertureAttr().Get() == pytest.approx(50.0)
    assert camera.GetVerticalApertureAttr().Get() == pytest.approx(25.0)
    assert camera.GetHorizontalApertureOffsetAttr().Get() == pytest.approx(-5.0)
    assert camera.GetVerticalApertureOffsetAttr().Get() == pytest.approx(2.5)
    assert tuple(camera.GetClippingRangeAttr().Get()) == pytest.approx((0.25, 80.0))
    assert camera.GetPrim().GetAttribute('userProperties:blender:data_name').Get() == 'Ortho.001'
    # The perspective camera keeps exactly the stock-exported data prim.
    assert [str(c.GetName()) for c in stage.GetPrimAtPath('/World/Persp').GetChildren()] == ['PerspData']
    monkeypatch.setattr(scene_generation, '_blender_module_provider',
                        lambda: SimpleNamespace(data=SimpleNamespace(materials=[])))
    mappings = scene_generation._validated_blender_prim_paths(scene, path)
    assert mappings[scene_generation.BlenderId('OBJECT', 41)].schema_path == '/World/Ortho/Ortho_001'
    assert mappings[scene_generation.BlenderId('OBJECT', 42)].schema_path == '/World/Persp/PerspData'


def test_export_without_orthographic_cameras_is_untouched(tmp_path):
    path = tmp_path / 'export.usda'
    path.write_text(_CAMERA_EXPORT)
    before = path.read_bytes()
    scene_generation._author_omitted_orthographic_cameras(
        path, SimpleNamespace(objects=[_camera(42, 'Persp', 'PerspData', 'PERSP')]))
    assert path.read_bytes() == before


def test_orthographic_camera_repair_rejects_a_conflicting_path(tmp_path):
    path = tmp_path / 'export.usda'
    path.write_text(_CAMERA_EXPORT.replace(
        'custom int ovrtx:object_session_uid = 41\n',
        'custom int ovrtx:object_session_uid = 41\n        def Xform "Ortho_001" {\n        }\n', 1))
    with pytest.raises(scene_generation.SceneGenerationError, match='already used'):
        scene_generation._author_omitted_orthographic_cameras(
            path, SimpleNamespace(objects=[_camera(41, 'Ortho', 'Ortho.001', 'ORTHO')]))


def test_panoramic_camera_is_rejected_by_type(tmp_path, monkeypatch):
    path = tmp_path / 'export.usda'
    path.write_text('#usda 1.0\ndef Xform "World" {\n    def Xform "Pano" {\n'
                    '        custom int ovrtx:object_session_uid = 43\n    }\n}\n')
    monkeypatch.setattr(scene_generation, '_blender_module_provider',
                        lambda: SimpleNamespace(data=SimpleNamespace(materials=[])))
    scene = SimpleNamespace(objects=[_camera(43, 'Pano', 'PanoData', 'PANO')])
    scene_generation._author_omitted_orthographic_cameras(path, scene)
    with pytest.raises(scene_generation.SceneGenerationError, match='pano camera .* is not supported') as caught:
        scene_generation._validated_blender_prim_paths(scene, path)
    assert caught.value.diagnostics[0]['reason'] == 'unsupported_camera_type'


def test_stock_export_contract_names_the_orthographic_camera_repair():
    # Reusable bases written before the repair have no orthographic Camera
    # schema; the contract change makes their lookup key miss.
    assert 'ortho-cameras' in scene_generation._STOCK_EXPORT_CONTRACT
