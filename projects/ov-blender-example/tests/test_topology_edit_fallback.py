# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import ovrtx_session  # noqa: E402
from ovrtx_blender_example.render_requests import (  # noqa: E402
    MaterialPresentationLayer,
    RenderRequest,
)
from ovrtx_blender_example.topology_edit_fallback import (  # noqa: E402
    COLLIDER_TOPOLOGY_CHANGED,
    ENVIRONMENT_TEXTURE_CHANGED,
    LIGHT_FAMILY_CHANGED,
    LIGHT_FORM_CHANGED,
    LIGHT_TYPE_CHANGED,
    MATERIAL_BINDING_CHANGED,
    MATERIAL_GRAPH_CHANGED,
    MESH_TOPOLOGY_CHANGED,
    PRIM_CREATE_DELETE,
    SCENE_TOPOLOGY_CHANGED,
    UV_COUNT_MISMATCH,
    WORLD_ASSIGNMENT_CHANGED,
    WORLD_NODE_GRAPH_CHANGED,
    coalesce_topology_reasons,
    topology_reasons_for_edit,
    topology_rekey_diagnostics,
)


def _compose_request(request: RenderRequest):
    return ovrtx_session.build_spec(request).ovrtx_scene_composition


def test_topology_reasons_for_edit_maps_topology_change_kinds() -> None:
    reasons = topology_reasons_for_edit(
        "scene_topology",
        {
            "topology_change_kinds": [
                "mesh_topology",
                "material_graph",
                "material_binding",
                "light_type",
                "light_form",
                "light_family",
                "prim_create_delete",
                "collider_topology",
                "uv_count_mismatch",
                "environment_texture",
                "world_node_graph",
                "world_assignment",
                "unclassified",
            ]
        },
    )

    assert reasons == (
        MESH_TOPOLOGY_CHANGED,
        MATERIAL_GRAPH_CHANGED,
        MATERIAL_BINDING_CHANGED,
        LIGHT_TYPE_CHANGED,
        LIGHT_FORM_CHANGED,
        LIGHT_FAMILY_CHANGED,
        PRIM_CREATE_DELETE,
        COLLIDER_TOPOLOGY_CHANGED,
        UV_COUNT_MISMATCH,
        ENVIRONMENT_TEXTURE_CHANGED,
        WORLD_NODE_GRAPH_CHANGED,
        WORLD_ASSIGNMENT_CHANGED,
    )


def test_coalesce_topology_reasons_deduplicates_in_canonical_order() -> None:
    reasons = coalesce_topology_reasons(
        [
            LIGHT_TYPE_CHANGED,
            " custom_reason ",
            MATERIAL_BINDING_CHANGED,
            MATERIAL_GRAPH_CHANGED,
            LIGHT_TYPE_CHANGED,
            "",
        ]
    )

    assert reasons == (
        MATERIAL_GRAPH_CHANGED,
        MATERIAL_BINDING_CHANGED,
        LIGHT_TYPE_CHANGED,
        "custom_reason",
    )


def test_topology_reasons_for_edit_uses_defaults_and_metadata_override() -> None:
    assert topology_reasons_for_edit("collider_topology") == (COLLIDER_TOPOLOGY_CHANGED,)
    assert topology_reasons_for_edit("collider_structure") == (COLLIDER_TOPOLOGY_CHANGED,)
    assert topology_reasons_for_edit("scene_topology") == (SCENE_TOPOLOGY_CHANGED,)
    assert topology_reasons_for_edit("scene_structure") == (SCENE_TOPOLOGY_CHANGED,)

    assert topology_reasons_for_edit(
        "material_topology",
        {
            "topology_reasons": [
                MATERIAL_BINDING_CHANGED,
                MATERIAL_GRAPH_CHANGED,
                MATERIAL_BINDING_CHANGED,
            ]
        },
    ) == (MATERIAL_GRAPH_CHANGED, MATERIAL_BINDING_CHANGED)
    assert topology_reasons_for_edit(
        "scene_topology",
        {"topology_reason": ENVIRONMENT_TEXTURE_CHANGED},
    ) == (ENVIRONMENT_TEXTURE_CHANGED,)


def test_topology_rekey_diagnostics_records_composition_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))

    first = _compose_request(
        RenderRequest(
            input_usd_path=str(source),
            material_scene_layer=_material_scene_layer(
                digest="material-a",
                layer_body='def Scope "OVRTX_Materials"\n{\n}\n',
            ),
        )
    )
    second = _compose_request(
        RenderRequest(
            input_usd_path=str(source),
            material_scene_layer=_material_scene_layer(
                digest="material-b",
                layer_body='def Scope "OVRTX_MaterialsB"\n{\n}\n',
            ),
        )
    )

    diagnostics = topology_rekey_diagnostics(
        reasons=[MATERIAL_BINDING_CHANGED, MATERIAL_GRAPH_CHANGED, MATERIAL_BINDING_CHANGED],
        old_composition=first,
        new_composition=second,
        requested_write_path=str(tmp_path / "topology-overlay.usda"),
        write_requested=True,
    )

    assert diagnostics["topology_reasons"] == [MATERIAL_GRAPH_CHANGED, MATERIAL_BINDING_CHANGED]
    assert diagnostics["old_composition_identity"]["composition_digest"] == first.digest
    assert diagnostics["new_composition_identity"]["composition_digest"] == second.digest
    assert diagnostics["old_composition_identity"]["source_scene_path"] == str(source)
    assert diagnostics["old_composition_identity"]["composed_scene_path"] == first.composed_scene_path
    assert diagnostics["new_composition_identity"]["source_scene_path"] == str(source)
    assert diagnostics["new_composition_identity"]["composed_scene_path"] == second.composed_scene_path
    assert diagnostics["composition_identity_status"] == "old_and_new_recorded"
    assert diagnostics["old_composition_identity"]["presentation_sources"] == [
        "materialx_openpbr",
    ]
    assert diagnostics["new_composition_identity"]["presentation_sources"] == [
        "materialx_openpbr",
    ]
    assert diagnostics["composition_identity_changed"] is True
    assert diagnostics["write_requested"] is True
    assert diagnostics["session_rekey_status"] == "requested"
    assert diagnostics["session_rekey_requested"] is True
    assert diagnostics["refinement_reset"] is True


def test_topology_rekey_diagnostics_does_not_request_rekey_when_blocked() -> None:
    diagnostics = topology_rekey_diagnostics(
        reasons=[SCENE_TOPOLOGY_CHANGED],
        session_rekey_status="blocked",
    )

    assert diagnostics["session_rekey_status"] == "blocked"
    assert diagnostics["session_rekey_requested"] is False
    assert diagnostics["refinement_reset"] is False


def _material_scene_layer(*, digest: str, layer_body: str) -> MaterialPresentationLayer:
    return MaterialPresentationLayer(
        target_path="",
        layer_body=layer_body,
        authored_properties=(),
        digest_content={"source": "materialx_openpbr", "digest": digest},
        diagnostics={
            "source": "materialx_openpbr",
            "digest": digest,
        },
    )
