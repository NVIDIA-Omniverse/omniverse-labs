# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import usd_prim_resolver  # noqa: E402
from ovrtx_blender_example import uv_usd_prim  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditShape,
    InteractiveEdit,
    edit_location,
)


class _FakeAttr:
    def __init__(self, value=None, interpolation="faceVarying") -> None:
        self.value = value
        self.interpolation = interpolation

    def Get(self):
        return self.value

    def GetMetadata(self, name):
        return self.interpolation if name == "interpolation" else None

    def IsValid(self):
        return True


class _FakePrim:
    def __init__(self, path, type_name, attrs=None, info_id="") -> None:
        self.path = path
        self.type_name = type_name
        self.attrs = attrs or {}
        self.info_id = info_id
        self.attributes = tuple(self.attrs)

    def GetAttribute(self, name):
        if name == "info:id" and self.info_id:
            return _FakeAttr(self.info_id)
        return self.attrs.get(name)


def _prims():
    return (
        _FakePrim("/World/Looks/Paint", "Material"),
        _FakePrim(
            "/World/Looks/Paint/Shader",
            "Shader",
            {"inputs:diffuseColor": _FakeAttr((1.0, 1.0, 1.0))},
            "UsdPreviewSurface",
        ),
        _FakePrim("/World/Key/KeyLight", "RectLight"),
        _FakePrim(
            "/World/StudioDome",
            "DomeLight",
            {"inputs:intensity": _FakeAttr(1.0), "inputs:color": _FakeAttr((1.0, 1.0, 1.0))},
        ),
        _FakePrim(
            "/World/Quad",
            "Mesh",
            {"primvars:st": _FakeAttr(((0.0, 0.0), (1.0, 0.0)))},
        ),
    )


def _request(path="/tmp/scene.usda"):
    return SimpleNamespace(input_usd_path=path)


def test_explicit_owned_world_path_does_not_infer_other_stage_domes(monkeypatch):
    prims=tuple(_FakePrim('/World/env_light' if prim.path=='/World/StudioDome' else prim.path,
        prim.type_name,prim.attrs,prim.info_id) for prim in _prims())
    monkeypatch.setattr(usd_prim_resolver,'_open_stage_prims',lambda path:(object(),prims,None))
    default=usd_prim_resolver.UsdPrimResolver();default.scan(_request())
    assert default.resolve_world_dome().value is None
    owned=usd_prim_resolver.UsdPrimResolver(world_dome_path='/World/env_light');owned.scan(_request())
    assert owned.resolve_world_dome().value.prim_path=='/World/env_light'


def test_resolver_scans_once_per_scene_and_resolves_all_domains(monkeypatch) -> None:
    calls = []

    def open_stage(path):
        calls.append(path)
        return object(), _prims(), None

    monkeypatch.setattr(usd_prim_resolver, "_open_stage_prims", open_stage)
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request())
    resolver.scan(_request())

    material = resolver.resolve_material(
        SimpleNamespace(name="Paint"),
        usd_attribute="inputs:diffuseColor",
        property_name="diffuse_color",
    )
    light = resolver.resolve_light(SimpleNamespace(name="Key", name_full="Key", parent=None))
    dome = resolver.resolve_world_dome()
    mesh = {"ovrtx:sourceUsdPath": "/World/Quad"}
    mesh_result = resolver.resolve_uv(mesh)

    assert calls == ["/tmp/scene.usda"]
    assert material.value is not None
    assert light.value is not None
    assert dome.value is not None
    assert mesh_result.value is not None
    assert resolver.diagnostics()["scan"]["prim_count"] == len(_prims())


def test_scene_change_and_reset_invalidate_scan_and_uv_validation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), calls.append(path) or _prims(), None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request("/tmp/a.usda"))
    resolver.record_uv_loop_order_validation(
        InteractiveEdit(
            shape=EditShape.VALUE,
            data_authority=DataAuthority.VIEW,
            **edit_location(
                usd_prim_path="/World/Quad",
                usd_attribute=uv_usd_prim.TARGET_USD_ATTRIBUTE,
            ),
            metadata={
                "loop_order_validation": {
                    "status": uv_usd_prim.RESOLVED,
                    "mesh_prim_path": "/World/Quad",
                }
            },
        )
    )
    assert resolver.uv_loop_order_validation("/World/Quad") is not None

    resolver.scan(_request("/tmp/b.usda"))
    assert resolver.uv_loop_order_validation("/World/Quad") is None
    resolver.reset()

    assert calls == ["/tmp/a.usda", "/tmp/b.usda"]
    assert resolver.diagnostics()["scan"] == {"available": False, "reason": "not_loaded"}


def test_missing_scene_path_fails_closed_without_scan(monkeypatch) -> None:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda _path: (_ for _ in ()).throw(AssertionError("must not scan")),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(None)

    result = resolver.resolve_world_dome()
    assert result.value is None
    assert result.error_reason == "usd_stage_unavailable"
    assert resolver.diagnostics()["scan"]["reason"] == "missing_input_usd_path"


def test_material_indexes_do_not_repeat_geometry_prim_reads(monkeypatch):
    calls = []
    class Mesh(_FakePrim):
        def GetAttribute(self, name):
            calls.append(name)
            return None
    mesh = Mesh("/World/LargeMesh", "Mesh")
    monkeypatch.setattr(usd_prim_resolver, "_open_stage_prims", lambda path: (object(), (*_prims(), mesh), None))
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request())
    assert resolver.diagnostics()["scan"]["available"]
    assert "info:id" not in calls
    assert not any(name.startswith("inputs:") for name in calls)
    assert resolver.resolve_material(SimpleNamespace(name="Paint"),
        usd_attribute="inputs:diffuseColor", property_name="diffuse_color").value is not None


def test_owner_thread_scan_reuse_keeps_identity_and_validation_independent(monkeypatch):
    calls=[]
    monkeypatch.setattr(usd_prim_resolver, "_open_stage_prims", lambda path: (object(),calls.append(path) or _prims(),None))
    source=usd_prim_resolver.UsdPrimResolver()
    source.scan(_request())
    source._uv_loop_order_validations['/World/Quad']={'status':'source-only'}
    target=usd_prim_resolver.UsdPrimResolver(object_paths_by_session_uid={91:'/World/Owned'})
    assert target.reuse_scan_from(source,_request())
    target.scan(_request())
    assert len(calls)==1
    assert target.resolve_blender_object(SimpleNamespace(name='Different',session_uid=91)).value=='/World/Owned'
    assert target.uv_loop_order_validation('/World/Quad') is None
    assert source._blender_object_index['by_session_uid'].get(91) is None
    source.reset()
    assert target.resolve_material(SimpleNamespace(name='Paint'),
        usd_attribute='inputs:diffuseColor',property_name='diffuse_color').value is not None


def test_scan_reuse_rejects_stale_failed_or_different_material_identity(monkeypatch):
    monkeypatch.setattr(usd_prim_resolver, "_open_stage_prims", lambda path: (object(),_prims(),None))
    source=usd_prim_resolver.UsdPrimResolver()
    source.scan(_request())
    assert not usd_prim_resolver.UsdPrimResolver().reuse_scan_from(source,_request('/other.usda'))
    assert not usd_prim_resolver.UsdPrimResolver(material_paths_by_session_uid={91:'/Other'}).reuse_scan_from(source,_request())
    source.reset()
    assert not usd_prim_resolver.UsdPrimResolver().reuse_scan_from(source,_request())


def test_scan_time_index_failure_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda _path: (object(), _prims(), None),
    )
    monkeypatch.setattr(
        usd_prim_resolver.material_usd_prim,
        "_material_prim_index_from_prims",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad prim")),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()

    resolver.scan(_request())

    result = resolver.resolve_light(SimpleNamespace(name="Key"))
    assert result.value is None
    assert result.error_reason == "usd_stage_unavailable"
    assert resolver.diagnostics()["scan"]["reason"] == "RuntimeError: bad prim"


def test_transient_failed_scan_retries_same_path_then_reuses_success(monkeypatch):
    calls = []

    def open_stage(path):
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError('export pending')
        return object(), _prims(), None

    monkeypatch.setattr(usd_prim_resolver, '_open_stage_prims', open_stage)
    resolver = usd_prim_resolver.UsdPrimResolver()
    request = _request()
    resolver.scan(request)
    assert not resolver.diagnostics()['scan']['available']
    resolver.scan(request)
    assert resolver.diagnostics()['scan']['available']
    resolver.scan(request)
    assert len(calls) == 2


def test_resolver_maps_exact_blender_object_name_to_exported_xform(monkeypatch) -> None:
    prims = (
        _FakePrim(
            "/Object_With_Spaces",
            "Xform",
            {"userProperties:blender:object_name": _FakeAttr("Object With Spaces")},
        ),
        _FakePrim(
            "/Object_With_Spaces/Mesh",
            "Mesh",
            {"userProperties:blender:object_name": _FakeAttr("Object With Spaces")},
        ),
    )
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), prims, None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request())

    resolution = resolver.resolve_blender_object(SimpleNamespace(name="Object With Spaces"))

    assert resolution.value == "/Object_With_Spaces"
    assert resolution.diagnostics["match_source"] == "blender_object_name"
    assert resolver.diagnostics()["blender_object"]["candidate_count"] == 1


def test_resolver_prefers_supplied_blender_object_session_uid(monkeypatch) -> None:
    prims = (
        _FakePrim(
            "/Renamed_Object",
            "Xform",
            {
                "userProperties:blender:object_name": _FakeAttr("Old Name"),
            },
        ),
    )
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), prims, None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver(
        object_paths_by_session_uid={91: "/Renamed_Object"}
    )
    resolver.scan(_request())

    resolution = resolver.resolve_blender_object(
        SimpleNamespace(name="Current Name", session_uid=91)
    )

    assert resolution.value == "/Renamed_Object"
    assert resolution.diagnostics["match_source"] == "blender_session_uid"


def test_resolver_prefers_supplied_light_object_session_uid(monkeypatch) -> None:
    prims = (_FakePrim("/World/Renamed_Light/Renamed_Light", "SphereLight"),)
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), prims, None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver(
        light_paths_by_object_session_uid={92: "/World/Renamed_Light/Renamed_Light"}
    )
    resolver.scan(_request())

    resolution = resolver.resolve_light(
        SimpleNamespace(name="Current Light", name_full="Current Light", session_uid=92)
    )

    assert resolution.value.prim_path == "/World/Renamed_Light/Renamed_Light"
    assert resolution.diagnostics["match_source"] == "blender_session_uid"


def _authored_object(name: str, prim_path: str) -> SimpleNamespace:
    """Blender-object fake carrying the ``ov.usd.prim_path`` authoring identity."""

    return SimpleNamespace(
        name=name,
        ov=SimpleNamespace(usd=SimpleNamespace(prim_path=prim_path)),
    )


def _authored_generation_prims() -> tuple[_FakePrim, ...]:
    """Prims shaped like an authored scene generation (task04-01).

    The converters author no exported-name attributes: meshes are an Xform
    object root with a Mesh child, and the light prim itself *is* the
    object root (a UsdLux type, not an Xform).
    """

    return (
        _FakePrim("/World/Cube", "Xform"),
        _FakePrim("/World/Cube/Mesh", "Mesh"),
        _FakePrim("/World/Lights/Key", "RectLight"),
    )


def test_resolver_maps_authoring_prim_path_to_authored_object_roots(monkeypatch) -> None:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), _authored_generation_prims(), None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request("/tmp/generation.usdc"))

    mesh = resolver.resolve_blender_object(_authored_object("Cube", "/World/Cube"))
    light = resolver.resolve_blender_object(_authored_object("Key", "/World/Lights/Key"))

    assert mesh.value == "/World/Cube"
    assert mesh.diagnostics["match_source"] == "authoring_prim_path"
    # The light object root is the UsdLux prim itself — resolution must not
    # require an Xform (its transform op lives on the light prim).
    assert light.value == "/World/Lights/Key"
    assert light.diagnostics["match_source"] == "authoring_prim_path"


def test_resolver_fails_closed_when_authoring_prim_path_is_not_in_scene(monkeypatch) -> None:
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), _authored_generation_prims(), None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request("/tmp/generation.usdc"))

    # A camera (or any object the converters do not emit) has no authored
    # prim; a stale authoring identity must not resolve to anything.
    unconverted = resolver.resolve_blender_object(_authored_object("Camera", ""))
    stale = resolver.resolve_blender_object(_authored_object("Cube", "/World/Removed"))

    assert unconverted.value is None
    assert unconverted.error_reason == "blender_object_name_not_found"
    assert stale.value is None
    assert stale.error_reason == "authoring_prim_path_not_in_scene"
    assert stale.diagnostics["authoring_prim_path"] == "/World/Removed"


def test_resolver_falls_back_to_exported_name_when_authoring_path_is_absent(monkeypatch) -> None:
    prims = (
        _FakePrim(
            "/Exported_Cube",
            "Xform",
            {"userProperties:blender:object_name": _FakeAttr("Cube")},
        ),
    )
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), prims, None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request("/tmp/exported.usda"))

    # Direct-USD stages from the stock exporter: a stale authoring identity
    # that is absent from the stage falls back to the exported-name index.
    resolution = resolver.resolve_blender_object(
        _authored_object("Cube", "/World/StaleAuthoredRoot")
    )

    assert resolution.value == "/Exported_Cube"
    assert resolution.diagnostics["match_source"] == "blender_object_name"




def test_resolver_rejects_missing_or_ambiguous_blender_object_name(monkeypatch) -> None:
    prims = (
        _FakePrim(
            "/A",
            "Xform",
            {"userProperties:blender:object_name": _FakeAttr("Duplicate")},
        ),
        _FakePrim(
            "/B",
            "Xform",
            {"userProperties:blender:object_name": _FakeAttr("Duplicate")},
        ),
    )
    monkeypatch.setattr(
        usd_prim_resolver,
        "_open_stage_prims",
        lambda path: (object(), prims, None),
    )
    resolver = usd_prim_resolver.UsdPrimResolver()
    resolver.scan(_request())

    missing = resolver.resolve_blender_object(SimpleNamespace(name="Missing"))
    ambiguous = resolver.resolve_blender_object(SimpleNamespace(name="Duplicate"))

    assert missing.value is None
    assert missing.error_reason == "blender_object_name_not_found"
    assert ambiguous.value is None
    assert ambiguous.error_reason == "ambiguous_blender_object_name"
    assert ambiguous.diagnostics["candidate_paths"] == ("/A", "/B")


def test_generation_material_identity_survives_sanitized_or_renamed_blender_name(monkeypatch):
    monkeypatch.setattr(usd_prim_resolver, '_open_stage_prims', lambda path: (object(), _prims(), None))
    resolver = usd_prim_resolver.UsdPrimResolver(material_paths_by_session_uid={71: '/World/Looks/Paint'})
    resolver.scan(_request())
    material = SimpleNamespace(session_uid=71, name='Cube Orange renamed', name_full='Cube Orange renamed')
    result = resolver.resolve_material(material, usd_attribute='inputs:diffuseColor', property_name='node_tree')
    assert result.value.prim_path == '/World/Looks/Paint/Shader'
    assert result.diagnostics['match_source'] == 'scene_generation_material_identity'
    material.session_uid = 72
    assert resolver.resolve_material(material, usd_attribute='inputs:diffuseColor', property_name='node_tree').value is None


def test_material_values_follow_exact_accepted_presentation_binding(monkeypatch):
    from ovrtx_blender_example import materialx_openpbr_conversion as conversion
    prims = (*_prims(), _FakePrim('/OVRTX_Materials/GeneratedPaint', 'Material'),
             _FakePrim('/OVRTX_Materials/GeneratedPaint/OpenPBR', 'Shader',
                       {'inputs:base_color': _FakeAttr((.8,.1,0))}, 'ND_open_pbr_surface_surfaceshader'))
    monkeypatch.setattr(usd_prim_resolver, '_open_stage_prims', lambda path: (object(), prims, None))
    monkeypatch.setattr(usd_prim_resolver, '_source_material_binding_rows', lambda path: (('/World/Quad', '/World/Looks/Paint'),))
    monkeypatch.setattr(conversion, '_materialx_binding_identity_from_prims', lambda *a, **kw: {
        'available': True, 'bindings': [{'binding_target':'/World/Quad', 'material_path':'/OVRTX_Materials/GeneratedPaint'}]})
    resolver = usd_prim_resolver.UsdPrimResolver(material_paths_by_session_uid={71:'/World/Looks/Paint'},
                                               material_source_scene_path='/source.usda')
    resolver.scan(_request('/accepted.usda'))
    material = SimpleNamespace(session_uid=71, name='Different Blender Name')
    color = resolver.resolve_material(material, usd_attribute='inputs:base_color', property_name='Base Color')
    preview = resolver.resolve_material(material, usd_attribute='inputs:diffuseColor', property_name='Base Color')
    assert color.value.prim_path == '/OVRTX_Materials/GeneratedPaint/OpenPBR'
    assert preview.value is None

    # Missing converter defaults remain editable only on this verified effective root.
    for attr in ('inputs:specular_ior', 'inputs:coat_weight', 'inputs:emission_luminance'):
        result = resolver.resolve_material(material, usd_attribute=attr, property_name='node_tree')
        assert result.value.prim_path == '/OVRTX_Materials/GeneratedPaint/OpenPBR'
