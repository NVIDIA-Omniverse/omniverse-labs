# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_scene_composition, ovrtx_session
from ovrtx_blender_example.render_requests import (
    CameraProjectionState,
    MaterialPresentationLayer,
    RenderRequest,
)


class _UnreadableDiagnostics(Mapping[str, object]):
    def __getitem__(self, _key: str) -> object:
        raise AssertionError("operational code traversed artifact diagnostics")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("operational code traversed artifact diagnostics")

    def __len__(self) -> int:
        raise AssertionError("operational code traversed artifact diagnostics")


def _request(tmp_path: Path, **changes: object) -> RenderRequest:
    request = RenderRequest(
        input_usd_path=str(tmp_path / "scene.usda"),
        sensor_paths=("/Render/ProductA", "/Render/ProductB"),
        selected_sensor_paths=("/Render/ProductA",),
        width=320,
        height=180,
        camera_prim_path="/World/Camera",
    )
    return replace(request, **changes)


def _material_layer(*, material_count: int) -> MaterialPresentationLayer:
    return MaterialPresentationLayer(
        target_path="/World/Geom",
        layer_body='def Scope "OVRTX_Materials"\n{\n}\n',
        authored_properties=(("/World/Geom", "material:binding"),),
        digest_content={
            "source": "materialx_openpbr",
            "digest": "material-digest",
            "layer_body": 'def Scope "OVRTX_Materials"\n{\n}\n',
        },
        diagnostics={
            "source": "materialx_openpbr",
            "digest": "material-digest",
            "status": "generated",
            "material_count": material_count,
            "materials": [
                {
                    "material_name": f"Material {index}",
                    "node_inventory": [
                        {"name": f"Node {node}", "type": "BSDF_PRINCIPLED"}
                        for node in range(20)
                    ],
                }
                for index in range(material_count)
            ],
        },
    )


def _presentation_text(
    composition: ovrtx_scene_composition.OvrtxSceneComposition,
    source: str = "viewport_camera_projection",
) -> str:
    record = next(
        item for item in composition.presentation_layers if item["source"] == source
    )
    return Path(str(record["path"])).read_text(encoding="utf-8")


def test_prepared_composition_identity_excludes_verbose_material_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed")
    )
    small_layer = _material_layer(material_count=1)
    large_layer = _material_layer(material_count=100)

    small_request = _request(tmp_path, material_scene_layer=small_layer)
    large_request = _request(tmp_path, material_scene_layer=large_layer)
    small = ovrtx_session.build_spec(small_request)
    large = ovrtx_session.build_spec(large_request)

    assert small == large
    operational_record = large.ovrtx_scene_composition.presentation_layers[0]
    assert set(operational_record) == {
        "generated",
        "path",
        "source",
        "target_path",
    }
    assert "materials" not in operational_record
    assert ovrtx_scene_composition.diagnostics(small.ovrtx_scene_composition) == (
        ovrtx_scene_composition.diagnostics(large.ovrtx_scene_composition)
    )
    artifact_diagnostics = ovrtx_scene_composition.diagnostics(
        large.ovrtx_scene_composition,
        request=large_request,
    )
    material_record = artifact_diagnostics["presentation_layers"][0]
    assert material_record["material_count"] == 100
    assert len(material_record["materials"]) == 100


def test_preparation_and_operational_diagnostics_never_read_layer_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    layer = replace(_material_layer(material_count=1), diagnostics=_UnreadableDiagnostics())

    spec = ovrtx_session.build_spec(
        _request(tmp_path, material_scene_layer=layer),
    )
    evidence = ovrtx_scene_composition.diagnostics(spec.ovrtx_scene_composition)

    assert evidence["presentation_layer_count"] == 2
    assert all("materials" not in record for record in evidence["presentation_layers"])


def test_artifact_diagnostics_reject_mismatched_layer_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    request = _request(tmp_path, material_scene_layer=_material_layer(material_count=1))
    composition = ovrtx_session.build_spec(request).ovrtx_scene_composition
    mismatched_layer = replace(
        _material_layer(material_count=1),
        digest_content={
            "source": "materialx_openpbr",
            "digest": "other-material-digest",
            "layer_body": 'def Scope "Other_Materials"\n{\n}\n',
        },
    )

    with pytest.raises(ValueError, match="artifact digest"):
        ovrtx_scene_composition.diagnostics(
            composition,
            request=replace(request, material_scene_layer=mismatched_layer),
        )


def test_build_spec_uses_creation_inputs_only(tmp_path: Path) -> None:
    request = _request(tmp_path)

    spec = ovrtx_session.build_spec(request)
    # Per-frame/runtime-mutable fields do not join session identity. The
    # resolved presentation render var does (task02-02) and is covered
    # separately, so it stays out of this list.
    changed_presentation = ovrtx_session.build_spec(
        replace(
            request,
            selected_sensor_paths=("/Render/ProductB",),
            min_samples=8,
            max_samples=64,
            camera_matrix=((1.0, 0.0, 0.0, 0.0),) * 4,
            timeline_frame=12,
        ),
    )

    assert spec.sensor_paths == ("/Render/ProductA", "/Render/ProductB")
    assert spec.width == 320
    assert spec.height == 180
    assert spec.camera_prim_path == "/World/Camera"
    assert spec.camera_pose_source == ovrtx_session.COMPOSED_SCENE
    assert changed_presentation.camera_pose_source == ovrtx_session.RUNTIME_UPDATE
    assert replace(changed_presentation, camera_pose_source=spec.camera_pose_source) == spec


def test_render_var_joins_session_identity(tmp_path: Path) -> None:
    """task02-02: the resolved presentation render var is session identity.

    Two requests differing only in the classified ``render_var`` build specs
    that differ solely in that field, and ``reuse_decision`` returns a replace
    with ``render_var_changed`` (both directions). An unset/empty mapping keeps
    the LDR default so pre-task requests are unchanged.
    """

    ldr = ovrtx_session.build_spec(_request(tmp_path))
    hdr = ovrtx_session.build_spec(
        _request(tmp_path, color_presentation={"render_var": "HdrColor"}),
    )

    assert ldr.render_var == "LdrColor"
    assert hdr.render_var == "HdrColor"
    # The render var is the only spec difference the mode flip introduces.
    assert replace(hdr, render_var=ldr.render_var) == ldr
    assert ovrtx_session.reuse_decision(ldr, hdr) == ovrtx_session.OvrtxSessionReuseDecision(
        reuse=False,
        reason="render_var_changed",
    )
    assert ovrtx_session.reuse_decision(hdr, ldr).reason == "render_var_changed"
    assert ovrtx_session.reuse_decision(ldr, ldr).reason == "same_session"


def test_provenance_stamp_never_changes_session_identity(tmp_path: Path) -> None:
    """task05-04: authored-generation provenance is artifact metadata only.

    The authored-generation digest/number recorded with render evidence must
    not join the session spec or composition digest — provenance describes the
    input, it is not identity (the input path already is).
    """

    request = _request(tmp_path)

    spec = ovrtx_session.build_spec(request)
    stamped = ovrtx_session.build_spec(
        replace(
            request,
            authored_generation_digest="f" * 64,
            authored_generation=12,
        )
    )

    assert stamped == spec
    assert stamped.ovrtx_scene_composition.digest == spec.ovrtx_scene_composition.digest
    assert ovrtx_session.reuse_decision(spec, stamped).reuse is True


def test_build_spec_normalizes_empty_declared_sensor_set(tmp_path: Path) -> None:
    spec = ovrtx_session.build_spec(_request(tmp_path, sensor_paths=()))

    assert spec.sensor_paths == ("/Render/OmniverseKit/HydraTextures/ViewportTexture0",)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"width": 640}, "output_shape_changed"),
        ({"input_usd_path": "/other/scene.usda"}, "scene_composition_changed"),
        ({"sensor_paths": ("/Render/ProductA",)}, "scene_composition_changed"),
        ({"camera_prim_path": "/World/OtherCamera"}, "camera_prim_changed"),
        (
            {"color_presentation": {"render_var": "HdrColor"}},
            "render_var_changed",
        ),
    ],
)
def test_reuse_policy_rejects_session_creation_changes(
    tmp_path: Path,
    change: dict[str, object],
    reason: str,
) -> None:
    current = ovrtx_session.build_spec(_request(tmp_path))
    desired = ovrtx_session.build_spec(_request(tmp_path, **change))

    assert ovrtx_session.reuse_decision(current, desired) == ovrtx_session.OvrtxSessionReuseDecision(
        reuse=False,
        reason=reason,
    )


@pytest.mark.parametrize(("width", "height"), [(320, 180), (640, 360)])
def test_probe_session_dimensions_are_authored_before_session_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))

    spec = ovrtx_session.build_spec(
        _request(tmp_path, width=width, height=height)
    )
    text = _presentation_text(spec.ovrtx_scene_composition)

    assert spec.camera_prim_path == "/World/Camera"
    assert spec.sensor_paths == ("/Render/ProductA", "/Render/ProductB")
    assert text.count(f"uniform int2 resolution = ({width}, {height})") == 2


def test_probe_output_dimension_change_requires_replacement_session(
    tmp_path: Path,
) -> None:
    live_transform = ovrtx_session.build_spec(
        _request(tmp_path, width=320, height=180)
    )
    shared_stage = ovrtx_session.build_spec(
        _request(tmp_path, width=640, height=360)
    )

    assert ovrtx_session.reuse_decision(
        live_transform, shared_stage
    ) == ovrtx_session.OvrtxSessionReuseDecision(
        reuse=False,
        reason="output_shape_changed",
    )


def test_reuse_policy_allows_adding_but_not_removing_runtime_camera_pose(tmp_path: Path) -> None:
    composed = ovrtx_session.build_spec(_request(tmp_path))
    runtime = replace(composed, camera_pose_source=ovrtx_session.RUNTIME_UPDATE)

    assert ovrtx_session.reuse_decision(composed, runtime).reuse is True
    assert ovrtx_session.reuse_decision(runtime, composed) == ovrtx_session.OvrtxSessionReuseDecision(
        reuse=False,
        reason="camera_pose_override_removed",
    )


def test_equal_specs_reuse(tmp_path: Path) -> None:
    spec = ovrtx_session.build_spec(_request(tmp_path))

    assert ovrtx_session.reuse_decision(spec, spec) == ovrtx_session.OvrtxSessionReuseDecision(
        reuse=True,
        reason="same_session",
    )


def test_reuse_policy_reports_declared_sensor_change_for_equal_scene_evidence(
    tmp_path: Path,
) -> None:
    current = ovrtx_session.build_spec(_request(tmp_path))
    desired = replace(current, sensor_paths=("/Render/ProductA",))

    assert ovrtx_session.reuse_decision(current, desired) == ovrtx_session.OvrtxSessionReuseDecision(
        reuse=False,
        reason="declared_sensors_changed",
    )


def test_scene_composition_authors_every_declared_render_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    request = _request(
        tmp_path,
        camera_projection=CameraProjectionState(
            source="viewport",
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=20.25,
        ),
    )

    spec = ovrtx_session.build_spec(request)
    text = _presentation_text(spec.ovrtx_scene_composition)

    assert 'def RenderProduct "ProductA"' in text
    assert 'def RenderProduct "ProductB"' in text
    assert text.count("rel camera = </World/Camera>") == 2
    assert text.count("uniform int2 resolution = (320, 180)") == 2
    assert "rel orderedVars = [</Render/ProductA/LdrColor>, </Render/ProductA/HdrColor>]" in text
    assert "rel orderedVars = [</Render/ProductB/LdrColor>, </Render/ProductB/HdrColor>]" in text
    assert text.count('def RenderVar "LdrColor"') == 2
    assert text.count('def RenderVar "HdrColor"') == 2


def test_scene_composition_generates_fixed_render_products_for_resolved_camera(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    request = _request(
        tmp_path,
        camera_prim_path="/World/OVRTXCamera",
        camera_projection=CameraProjectionState(
            source="viewport",
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=20.25,
        ),
    )

    spec = ovrtx_session.build_spec(request)
    text = _presentation_text(spec.ovrtx_scene_composition)

    assert 'over "OVRTXCamera"' in text
    assert text.count("def RenderProduct") == 2
    assert "rel camera = </World/OVRTXCamera>" in text
    assert "rel orderedVars = [</Render/ProductA/LdrColor>, </Render/ProductA/HdrColor>]" in text
    assert "rel orderedVars = [</Render/ProductB/LdrColor>, </Render/ProductB/HdrColor>]" in text
    assert text.count('def RenderVar "LdrColor"') == 2
    assert text.count('def RenderVar "HdrColor"') == 2
    assert 'uniform string sourceName = "LdrColor"' in text


def test_final_render_composition_authors_the_scene_camera_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    scene_camera_matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (1.5, -2.0, 3.25, 1.0),
    )
    request = _request(
        tmp_path,
        current_scene_generation=True,
        camera_prim_path="/World/OVRTXCamera",
        scene_camera_matrix=scene_camera_matrix,
    )

    spec = ovrtx_session.build_spec(request)
    text = _presentation_text(spec.ovrtx_scene_composition)

    assert spec.camera_pose_source == ovrtx_session.COMPOSED_SCENE
    assert 'def Camera "OVRTXCamera"' in text
    assert (
        "matrix4d xformOp:transform = "
        "( (1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (1.5, -2, 3.25, 1) )"
    ) in text
    assert 'uniform token[] xformOpOrder = ["xformOp:transform"]' in text
    # Artifact diagnostics recompute contributions from the request; the
    # authored pose must round-trip through the digest validation.
    evidence = ovrtx_scene_composition.diagnostics(
        spec.ovrtx_scene_composition,
        request=request,
    )
    camera_record = next(
        record
        for record in evidence["presentation_layers"]
        if record["source"] == "viewport_camera_projection"
    )
    assert camera_record["scene_camera_pose_authored"] is True


def test_scene_camera_pose_changes_the_composition_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    base = _request(
        tmp_path,
        current_scene_generation=True,
        camera_prim_path="/World/OVRTXCamera",
    )
    posed = replace(
        base,
        scene_camera_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 5.0, 1.0),
        ),
    )
    moved = replace(
        base,
        scene_camera_matrix=(
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 9.0, 1.0),
        ),
    )

    unposed_spec = ovrtx_session.build_spec(base)
    posed_spec = ovrtx_session.build_spec(posed)
    moved_spec = ovrtx_session.build_spec(moved)

    assert posed_spec.ovrtx_scene_composition.digest != unposed_spec.ovrtx_scene_composition.digest
    assert posed_spec.ovrtx_scene_composition.digest != moved_spec.ovrtx_scene_composition.digest
    assert ovrtx_session.reuse_decision(posed_spec, moved_spec).reason == "scene_composition_changed"
    unposed_text = _presentation_text(unposed_spec.ovrtx_scene_composition)
    assert "xformOp" not in unposed_text


def test_direct_usd_composition_ignores_a_scene_camera_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))
    # No generated scene presentation (direct USD input): the stage's own
    # camera is authoritative and the pose contribution must not author.
    request = _request(
        tmp_path,
        scene_camera_matrix=((1.0, 0.0, 0.0, 0.0),) * 3 + ((0.0, 0.0, 5.0, 1.0),),
    )

    spec = ovrtx_session.build_spec(request)
    text = _presentation_text(spec.ovrtx_scene_composition)

    assert "xformOp" not in text
    assert spec == ovrtx_session.build_spec(replace(request, scene_camera_matrix=None))


def test_scene_composition_authors_declared_products_without_camera_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))

    both = ovrtx_session.build_spec(_request(tmp_path))
    one = ovrtx_session.build_spec(
        _request(tmp_path, sensor_paths=("/Render/ProductA",)),
    )
    text = _presentation_text(both.ovrtx_scene_composition)

    assert 'def RenderProduct "ProductA"' in text
    assert 'def RenderProduct "ProductB"' in text
    assert text.count("rel camera = </World/Camera>") == 2
    assert text.count("uniform int2 resolution = (320, 180)") == 2
    assert one.ovrtx_scene_composition.composed_scene_path != both.ovrtx_scene_composition.composed_scene_path
    assert one.ovrtx_scene_composition.digest != both.ovrtx_scene_composition.digest


def test_session_preparation_deduplicates_declared_sensor_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))

    spec = ovrtx_session.build_spec(
        _request(tmp_path, sensor_paths=("/Render/ProductA", "/Render/ProductA")),
    )
    text = _presentation_text(spec.ovrtx_scene_composition)
    direct = ovrtx_scene_composition.compose(
        source_scene_path=str(tmp_path / "scene.usda"),
        camera_prim_path="/World/Camera",
        sensor_paths=("/Render/ProductB", "/Render/ProductB"),
        width=320,
        height=180,
        camera_projection=None,
        material_scene_layer=None,
    )
    direct_text = _presentation_text(direct)

    assert spec.sensor_paths == ("/Render/ProductA",)
    assert text.count('def RenderProduct "ProductA"') == 1
    assert text.count("uniform int2 resolution = (320, 180)") == 1
    assert spec.ovrtx_scene_composition.pass_through is False
    assert spec.ovrtx_scene_composition.conflict_records == ()
    assert direct_text.count('def RenderProduct "ProductB"') == 1
    assert direct.conflict_records == ()

    noisy_spec = replace(
        spec,
        sensor_paths=("", "/Render/ProductA", "/Render/ProductA"),
    )
    assert noisy_spec == spec
    assert ovrtx_session.reuse_decision(spec, noisy_spec).reason == "same_session"

    scalar = ovrtx_scene_composition.compose(
        source_scene_path=str(tmp_path / "scene.usda"),
        camera_prim_path="/World/Camera",
        sensor_paths="/Render/ProductC",
        width=320,
        height=180,
        camera_projection=None,
        material_scene_layer=None,
    )
    scalar_text = _presentation_text(scalar)
    assert scalar_text.count('def RenderProduct "ProductC"') == 1

    with pytest.raises(TypeError, match="sensor paths must be strings"):
        ovrtx_scene_composition.compose(
            source_scene_path=str(tmp_path / "scene.usda"),
            camera_prim_path="/World/Camera",
            sensor_paths=("/Render/ProductD", None),  # type: ignore[arg-type]
            width=320,
            height=180,
            camera_projection=None,
            material_scene_layer=None,
        )


def test_direct_session_spec_defaults_an_empty_declared_sensor_set(tmp_path: Path) -> None:
    spec = ovrtx_session.build_spec(_request(tmp_path))

    empty = replace(spec, sensor_paths=("",))
    unordered = replace(spec, sensor_paths={"/Render/B", "/Render/A"})  # type: ignore[arg-type]

    assert empty.sensor_paths == ("/Render/OmniverseKit/HydraTextures/ViewportTexture0",)
    assert unordered.sensor_paths == ("/Render/A", "/Render/B")


def test_direct_composition_uses_canonical_empty_and_unordered_sensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "composed"))

    empty = ovrtx_scene_composition.compose(
        source_scene_path=str(tmp_path / "scene.usda"),
        camera_prim_path="/World/Camera",
        sensor_paths=(),
        width=320,
        height=180,
        camera_projection=None,
        material_scene_layer=None,
    )
    unordered = ovrtx_scene_composition.compose(
        source_scene_path=str(tmp_path / "scene.usda"),
        camera_prim_path="/World/Camera",
        sensor_paths={"/Render/B", "/Render/A"},
        width=320,
        height=180,
        camera_projection=None,
        material_scene_layer=None,
    )
    empty_evidence = ovrtx_scene_composition.diagnostics(
        empty,
        request=_request(tmp_path, sensor_paths=()),
    )
    unordered_evidence = ovrtx_scene_composition.diagnostics(
        unordered,
        request=_request(
            tmp_path,
            sensor_paths={"/Render/B", "/Render/A"},  # type: ignore[arg-type]
        ),
    )

    assert empty.pass_through is False
    assert empty_evidence["presentation_layers"][0]["sensor_paths"] == [
        "/Render/OmniverseKit/HydraTextures/ViewportTexture0",
    ]
    assert unordered_evidence["presentation_layers"][0]["sensor_paths"] == [
        "/Render/A",
        "/Render/B",
    ]


def test_session_dimensions_normalize_before_composition_identity(tmp_path: Path) -> None:
    zero = ovrtx_session.build_spec(_request(tmp_path, width=0, height=0))
    negative = ovrtx_session.build_spec(_request(tmp_path, width=-1, height=-2))
    direct = replace(zero, width=0, height=-2)

    assert zero == negative == direct
    assert zero.width == 1
    assert zero.height == 1
    assert ovrtx_session.reuse_decision(zero, negative).reason == "same_session"


def test_scene_composition_evidence_is_deeply_immutable() -> None:
    layer_identifiers = ["layer-a"]
    labels = {"b", "a"}
    composition = ovrtx_scene_composition.OvrtxSceneComposition(
        source_scene_path="/scene.usda",
        composed_scene_path="/scene.usda",
        presentation_layers=({"nested": {"values": [1, 2], "labels": labels}},),
        digest="digest",
        pass_through=True,
        session_layer_identifiers=layer_identifiers,  # type: ignore[arg-type]
    )
    sensor_paths = ["/Render/A"]
    spec = ovrtx_session.OvrtxSessionSpec(
        ovrtx_scene_composition=composition,
        sensor_paths=sensor_paths,  # type: ignore[arg-type]
        width=320,
        height=180,
        camera_prim_path="/World/Camera",
        camera_pose_source=ovrtx_session.COMPOSED_SCENE,
    )
    layer_identifiers.append("layer-b")
    sensor_paths.append("/Render/B")
    labels.add("c")

    record = composition.presentation_layers[0]
    with pytest.raises(TypeError):
        record["other"] = 3  # type: ignore[index]
    with pytest.raises(TypeError):
        record["nested"]["other"] = 3  # type: ignore[index]
    with pytest.raises(AttributeError):
        record["nested"]["values"].append(3)
    assert composition.session_layer_identifiers == ("layer-a",)
    assert spec.sensor_paths == ("/Render/A",)
    assert record["nested"]["labels"] == ("a", "b")
    original_hash = hash(composition)
    with pytest.raises(AttributeError):
        record._items = ()  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        del record._items  # type: ignore[attr-defined]
    assert hash(composition)
    assert hash(composition) == original_hash
    assert hash(spec)

    evidence = ovrtx_scene_composition.diagnostics(composition)
    evidence["presentation_layers"][0]["nested"]["values"].append(3)
    assert record["nested"]["values"] == (1, 2)
    json.dumps(evidence)


def test_scene_composition_rejects_unsupported_mutable_evidence() -> None:
    scalar_identifier = ovrtx_scene_composition.OvrtxSceneComposition(
        source_scene_path="/scene.usda",
        composed_scene_path="/scene.usda",
        presentation_layers=(),
        digest="digest",
        pass_through=True,
        session_layer_identifiers="layer-a",  # type: ignore[arg-type]
    )
    assert scalar_identifier.session_layer_identifiers == ("layer-a",)
    assert ovrtx_scene_composition.diagnostics(scalar_identifier)[
        "session_layer_identifiers"
    ] == ["layer-a"]
    assert hash(scalar_identifier)

    unordered_identifiers = replace(
        scalar_identifier,
        session_layer_identifiers={"layer-c", "layer-a", "layer-b"},  # type: ignore[arg-type]
    )
    assert unordered_identifiers.session_layer_identifiers == (
        "layer-a",
        "layer-b",
        "layer-c",
    )

    with pytest.raises(TypeError, match="unsupported composition evidence value"):
        ovrtx_scene_composition.OvrtxSceneComposition(
            source_scene_path="/scene.usda",
            composed_scene_path="/scene.usda",
            presentation_layers=({"payload": bytearray(b"mutable")},),
            digest="digest",
            pass_through=True,
        )

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        ovrtx_scene_composition.OvrtxSceneComposition(
            source_scene_path="/scene.usda",
            composed_scene_path="/scene.usda",
            presentation_layers=({1: "integer", "1": "string"},),  # type: ignore[dict-item]
            digest="digest",
            pass_through=True,
        )

    with pytest.raises(TypeError, match="floats must be finite"):
        ovrtx_scene_composition.OvrtxSceneComposition(
            source_scene_path="/scene.usda",
            composed_scene_path="/scene.usda",
            presentation_layers=({"value": float("nan")},),
            digest="digest",
            pass_through=True,
        )

    with pytest.raises(TypeError, match="session layer identifiers must be strings"):
        ovrtx_scene_composition.OvrtxSceneComposition(
            source_scene_path="/scene.usda",
            composed_scene_path="/scene.usda",
            presentation_layers=(),
            digest="digest",
            pass_through=True,
            session_layer_identifiers=(bytearray(b"mutable"),),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="session layer identifiers must be strings"):
        ovrtx_scene_composition.OvrtxSceneComposition(
            source_scene_path="/scene.usda",
            composed_scene_path="/scene.usda",
            presentation_layers=(),
            digest="digest",
            pass_through=True,
            session_layer_identifiers=({"identifier": "layer-a"},),  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="composition digest must be a string"):
        ovrtx_scene_composition.OvrtxSceneComposition(
            source_scene_path="/scene.usda",
            composed_scene_path="/scene.usda",
            presentation_layers=(),
            digest=bytearray(b"mutable"),  # type: ignore[arg-type]
            pass_through=True,
        )


def test_frozen_mapping_sets_have_stable_cross_process_order() -> None:
    script = f"""
import json
import sys
sys.path.insert(0, {str(ROOT / 'addon')!r})
from ovrtx_blender_example import ovrtx_scene_composition

class HashableDict(dict):
    def __hash__(self):
        return hash(self[\"value\"])

composition = ovrtx_scene_composition.OvrtxSceneComposition(
    source_scene_path=\"/scene.usda\",
    composed_scene_path=\"/scene.usda\",
    presentation_layers=(
        {{\"values\": {{HashableDict(value=value) for value in (\"a\", \"b\", \"c\")}}}},
    ),
    digest=\"digest\",
    pass_through=True,
)
print(json.dumps(ovrtx_scene_composition.diagnostics(composition), sort_keys=True))
"""
    outputs = {
        subprocess.check_output(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        )
        for seed in ("1", "2", "3", "4", "5")
    }

    assert len(outputs) == 1


def test_session_spec_rejects_invalid_scalar_state(tmp_path: Path) -> None:
    spec = ovrtx_session.build_spec(_request(tmp_path))

    with pytest.raises(TypeError, match="OVRTX scene composition must be an OvrtxSceneComposition"):
        replace(spec, ovrtx_scene_composition={})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="camera prim path must be a string"):
        replace(spec, camera_prim_path=["/World/Camera"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="camera pose source"):
        replace(spec, camera_pose_source="invalid")  # type: ignore[arg-type]
