# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example.blender_callback_adapters import (  # noqa: E402
    BlenderEditCallbackAdapter,
    BlenderRenderCallbackAdapter,
    ExactStageRenderCallbackAdapter,
)
from ovrtx_blender_example import blender_callback_adapters  # noqa: E402
from ovrtx_blender_example.blender_signal_translation import BlenderSignalTranslationError  # noqa: E402
from ovrtx_blender_example.scene_generation import (  # noqa: E402
    BlenderId,
    BlenderPrimPath,
    SceneGenerationError,
)
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditShape,
    InteractiveEdit,
    edit_location,
)
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderEditSignal,
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
)


def test_render_callback_adapter_materializes_current_scene_before_signal_translation() -> None:
    camera = SimpleNamespace(session_uid=41)
    scene = SimpleNamespace(camera=camera)
    context = object()
    depsgraph = SimpleNamespace(scene=scene)
    request = object()
    signals: list[BlenderRenderSignal] = []
    generations: list[object] = []

    class _Generation:
        blender_prim_paths = {
            BlenderId("OBJECT", 41): BlenderPrimPath("Camera", "CAMERA", "/Camera", "/Camera/Camera")
        }

        @staticmethod
        def materialize_usd() -> str:
            return "/tmp/generated.usdc"

    def generation_for_scene(received_scene: object) -> object:
        generations.append(received_scene)
        return _Generation()

    class _Translator:
        def translate(self, signal: BlenderRenderSignal) -> object:
            signals.append(signal)
            return request

    adapter = BlenderRenderCallbackAdapter(
        generation_for_scene=generation_for_scene,
        translator=_Translator(),
        engine_id="engine-A",
    )

    assert adapter.view_draw(context, depsgraph) is request
    assert signals == [
        BlenderRenderSignal(
            source=BlenderRenderSignalSource.VIEW_DRAW,
            intent=BlenderRenderIntent.VIEWPORT,
            scene=scene,
            input_usd_path="/tmp/generated.usdc",
            camera_prim_path="/Camera/Camera",
            render_product_path="/Render/OmniverseKit/HydraTextures/ViewportTexture0",
            context=context,
            engine_id="engine-A",
            current_scene_generation=True,
        )
    ]
    assert generations == [scene]


def test_render_callback_adapter_translates_scene_generation_errors() -> None:
    def generation_for_scene(_scene: object) -> object:
        raise SceneGenerationError("unsupported Blender object: Grease Pencil")

    adapter = BlenderRenderCallbackAdapter(generation_for_scene=generation_for_scene)

    with pytest.raises(BlenderSignalTranslationError, match="unsupported Blender object"):
        adapter.view_draw_from_scene(object(), object())


def test_render_callback_adapter_defers_only_viewport_generation() -> None:
    scene = SimpleNamespace(camera=SimpleNamespace(session_uid=41))
    generation = SimpleNamespace(
        blender_prim_paths={
            BlenderId("OBJECT", 41): BlenderPrimPath(
                "Camera", "CAMERA", "/Camera", "/Camera/Camera"
            )
        },
        materialize_usd=lambda: "/tmp/generated.usdc",
    )
    calls = []
    adapter = BlenderRenderCallbackAdapter(
        generation_for_scene=lambda _scene: (calls.append("barrier"), generation)[1],
        viewport_generation_for_scene=lambda _scene: (
            calls.append("viewport"),
            generation,
        )[1],
        translator=SimpleNamespace(translate=lambda _signal: "request"),
    )

    assert adapter.view_update_from_scene(scene, object()) == "request"
    assert adapter.final_render_from_scene(scene) == "request"
    assert adapter.translate(
        BlenderRenderSignalSource.VIEW_DRAW,
        BlenderRenderIntent.FINAL_RENDER,
        scene,
        object(),
    ) == "request"
    assert adapter.translate(
        BlenderRenderSignalSource.FINAL_RENDER,
        BlenderRenderIntent.VIEWPORT,
        scene,
        object(),
    ) == "request"
    assert calls == ["viewport", "barrier", "viewport", "barrier"]


def test_render_callback_adapter_requires_active_scene_camera() -> None:
    adapter = BlenderRenderCallbackAdapter(
        generation_for_scene=lambda _scene: SimpleNamespace(
            materialize_usd=lambda: "/tmp/generated.usdc",
            blender_prim_paths={},
        )
    )

    with pytest.raises(BlenderSignalTranslationError, match="active camera"):
        adapter.final_render_from_scene(SimpleNamespace(camera=None))


def test_exact_stage_adapter_uses_explicit_internal_stage_identity() -> None:
    signals: list[BlenderRenderSignal] = []

    class _Translator:
        def translate(self, signal: BlenderRenderSignal) -> object:
            signals.append(signal)
            return "request"

    adapter = ExactStageRenderCallbackAdapter(
        input_usd_path="/fixtures/exact.usda",
        camera_prim_path="/Fixture/Camera",
        render_product_path="/Fixture/Product",
        translator=_Translator(),
        engine_id="fixture-engine",
    )

    assert adapter.final_render_from_scene(object()) == "request"
    assert signals[0].input_usd_path == "/fixtures/exact.usda"
    assert signals[0].camera_prim_path == "/Fixture/Camera"
    assert signals[0].render_product_path == "/Fixture/Product"


def test_edit_callback_adapter_fans_out_depsgraph_edits_to_active_engines() -> None:
    edit = object()
    depsgraph = SimpleNamespace(updates=(SimpleNamespace(id=object()),))
    selection_resolution = {"changed": False, "group_rejected": False}
    diagnostics: list[dict[str, object]] = []
    resolver_calls: list[tuple[object, dict[str, object]]] = []
    submitted: list[object] = []
    observed: list[tuple[object, object]] = []
    scene = object()

    class _Engine:
        def build_interactive_edits_from_depsgraph(self, depsgraph: object, **kwargs: object) -> list[object]:
            resolver_calls.append((depsgraph, dict(kwargs)))
            return [edit]

        def submit_interactive_edit(self, received_edit: object) -> dict[str, object]:
            submitted.append(received_edit)
            return {"accepted": True}

    policies = object()
    translator_factory = object()
    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (_Engine(),),
        selection_resolver=lambda context: selection_resolution,
        bridge_diagnostics_recorder=lambda **kwargs: diagnostics.append(dict(kwargs)),
        value_edit_conversion_policies=policies,
        edit_translator_factory=translator_factory,
        edit_observer=lambda received_scene, received_edit: observed.append(
            (received_scene, received_edit)
        ),
    )

    results_context = object()
    results = adapter.submit_depsgraph_interactive_edits(
        depsgraph,
        context=results_context,
        scene=scene,
    )

    assert results == [{"accepted": True}]
    assert submitted == [edit]
    assert observed == []
    assert resolver_calls == [
        (
            depsgraph,
            {
                "context": results_context,
                "selection_resolution": selection_resolution,
                "value_edit_conversion_policies": policies,
                "edit_translator_factory": translator_factory,
            },
        )
    ]
    assert diagnostics[-1] == {
        "suppressed": False,
        "active_engine_count": 1,
        "submitted_edit_count": 1,
        "result_count": 1,
    }


def test_edit_callback_adapter_does_not_observe_an_edit_rejected_by_active_engine() -> None:
    edit = object()
    observed: list[object] = []

    class _Engine:
        def build_interactive_edits_from_depsgraph(self, _depsgraph: object, **_kwargs: object) -> list[object]:
            return [edit]

        def submit_interactive_edit(self, _edit: object) -> SimpleNamespace:
            return SimpleNamespace(accepted=False, reason="physics_playback_locked")

    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (_Engine(),),
        selection_resolver=lambda _context: {"changed": False, "group_rejected": False},
        edit_observer=lambda _scene, received_edit: observed.append(received_edit),
    )

    results = adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=()),
        scene=object(),
    )

    assert results[0].reason == "physics_playback_locked"
    assert observed == []


def test_edit_callback_adapter_does_not_observe_value_during_selection_change() -> None:
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Cube",
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
        ),
        value=((2.0, 0.0, 0.0, 0.0),) * 4,
    )
    observed: list[InteractiveEdit] = []

    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (object(),),
        selection_resolver=lambda _context: {"changed": True, "group_rejected": False},
        edit_builder=lambda _depsgraph, **_kwargs: [edit],
        edit_observer=lambda _scene, received_edit: observed.append(received_edit),
        edit_observer_context=lambda _scene: (object(), ()),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=()),
        scene=object(),
    ) == []
    assert observed == []


def test_selection_change_submits_topology_to_active_engine_before_scene_observation() -> None:
    edit = InteractiveEdit(
        shape=EditShape.TOPOLOGY,
        data_authority=DataAuthority.VIEW,
        **edit_location(usd_prim_path="/World/Cube", blender_property_path="data"),
    )
    rejected = object()
    submitted: list[InteractiveEdit] = []
    observed: list[InteractiveEdit] = []

    class _Engine:
        def submit_interactive_edit(self, received_edit: InteractiveEdit) -> object:
            submitted.append(received_edit)
            return rejected

    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (_Engine(),),
        selection_resolver=lambda _context: {"changed": True, "group_rejected": False},
        edit_builder=lambda _depsgraph, **_kwargs: [edit],
        edit_observer=lambda _scene, received_edit: observed.append(received_edit),
        edit_observer_context=lambda _scene: (object(), ()),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=()),
        scene=object(),
    ) == [rejected]
    assert submitted == [edit]
    assert observed == []


def test_edit_callback_adapter_builds_signal_for_fallback_depsgraph_edits() -> None:
    edited_id = object()
    edit = object()
    depsgraph = SimpleNamespace(updates=(SimpleNamespace(id=edited_id),))
    selection_resolution = {"changed": False, "group_rejected": False}
    builder_calls: list[tuple[object, dict[str, object]]] = []
    diagnostics: list[dict[str, object]] = []
    submitted: list[tuple[object, object]] = []
    scene = object()
    resolver = object()
    light_objects = (object(),)

    def _builder(depsgraph: object, **kwargs: object) -> list[object]:
        builder_calls.append((depsgraph, dict(kwargs)))
        return [edit]

    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (),
        selection_resolver=lambda context: selection_resolution,
        bridge_diagnostics_recorder=lambda **kwargs: diagnostics.append(dict(kwargs)),
        edit_builder=_builder,
        edit_observer=lambda received_scene, received_edit: submitted.append(
            (received_scene, received_edit)
        ),
        edit_observer_context=lambda _scene: (resolver, light_objects),
    )

    assert adapter.submit_depsgraph_interactive_edits(depsgraph, scene=scene) == []
    assert submitted == [(scene, edit)]
    called_depsgraph, kwargs = builder_calls[0]
    assert tuple(update.id for update in called_depsgraph.updates) == (edited_id,)
    assert kwargs["selection_resolution"] == selection_resolution
    assert kwargs["usd_prim_resolver"] is resolver
    assert kwargs["light_objects"] == light_objects
    assert diagnostics[-1] == {
        "suppressed": False,
        "active_engine_count": 0,
        "submitted_edit_count": 1,
        "result_count": 0,
    }


def test_edit_callback_adapter_uses_injected_translator_factory() -> None:
    edited_id = object()
    edit = object()
    depsgraph = SimpleNamespace(updates=(SimpleNamespace(id=edited_id),))
    factory_kwargs: list[dict[str, object]] = []
    signals: list[BlenderEditSignal] = []

    class _Translator:
        def translate(self, signal: BlenderEditSignal) -> tuple[object, ...]:
            signals.append(signal)
            return (edit,)

    def translator_factory(**kwargs: object) -> _Translator:
        factory_kwargs.append(dict(kwargs))
        return _Translator()

    adapter = BlenderEditCallbackAdapter(edit_translator_factory=translator_factory)

    edits = adapter.translate_depsgraph_edits(depsgraph)

    assert edits == (edit,)
    assert len(factory_kwargs) == 1
    assert signals[0].id_items == (edited_id,)


def test_edit_observer_reports_topology_when_selection_group_is_rejected() -> None:
    edit = object()
    result = object()
    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (),
        selection_resolver=lambda _context: {
            "changed": False,
            "group_rejected": True,
        },
        edit_builder=lambda _depsgraph, **_kwargs: [edit],
        edit_observer=lambda _scene, received: result if received is edit else None,
        edit_observer_context=lambda _scene: (object(), ()),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=(SimpleNamespace(id=object()),)),
        scene=object(),
    ) == [result]


def _mapped_transform(source_name: str, prim_path: str) -> InteractiveEdit:
    return InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path=prim_path,
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
            provenance={
                "selection_resolution": {
                    "source_name": source_name,
                    "status": "unresolved",
                }
            },
        ),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )


def test_authoritative_scene_mapping_admits_native_unowned_selection() -> None:
    edit = _mapped_transform("Hat", "/World/Hat")
    submitted: list[InteractiveEdit] = []
    adapter = BlenderEditCallbackAdapter(
        edit_builder=lambda _depsgraph, **_kwargs: [edit],
        selection_resolver=lambda _context: {
            "changed": False,
            "group_rejected": True,
            "selected_object_count": 1,
            "sources": [{"source_name": "Hat", "status": "unresolved"}],
        },
        edit_group_observer=lambda _scene, edits, _selection: (
            submitted.extend(edits) or tuple(edits)
        ),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=(SimpleNamespace(id=object()),)),
        scene=object(),
    ) == [edit]
    assert submitted == [edit]


def test_authoritative_scene_mapping_waits_for_redirected_selection_event() -> None:
    translations: list[object] = []
    adapter = BlenderEditCallbackAdapter(
        edit_builder=lambda depsgraph, **_kwargs: translations.append(depsgraph),
        selection_resolver=lambda _context: {
            "changed": True,
            "group_rejected": False,
        },
        edit_group_observer=lambda _scene, edits, _selection: tuple(edits),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=(SimpleNamespace(id=object()),)),
        scene=object(),
    ) == []
    assert translations == []


def test_authoritative_scene_mapping_dispatches_engine_rejection_atomically() -> None:
    submitted: list[InteractiveEdit] = []
    adapter = BlenderEditCallbackAdapter(
        edit_builder=lambda _depsgraph, **_kwargs: [],
        selection_resolver=lambda _context: {
            "changed": False,
            "group_rejected": True,
            "selected_object_count": 2,
            "sources": [
                {"source_name": "Hat", "status": "unresolved"},
                {"source_name": "Loose", "status": "unresolved"},
            ],
        },
        edit_group_observer=lambda _scene, edits, _selection: (
            submitted.extend(edits) or tuple(edits)
        ),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=(SimpleNamespace(id=object()),)),
        scene=object(),
    ) == []
    assert submitted == []


def test_authoritative_scene_mapping_dispatches_complete_transform_group() -> None:
    edits = [
        _mapped_transform("Hat", "/World/Hat"),
        _mapped_transform("Box", "/World/Box"),
    ]
    submitted: list[InteractiveEdit] = []
    adapter = BlenderEditCallbackAdapter(
        edit_builder=lambda _depsgraph, **_kwargs: edits,
        selection_resolver=lambda _context: {
            "changed": False,
            "group_rejected": True,
            "selected_object_count": 2,
            "sources": [
                {"source_name": "Hat", "status": "unresolved"},
                {"source_name": "Box", "status": "unresolved"},
            ],
        },
        edit_group_observer=lambda _scene, values, _selection: (
            submitted.extend(values) or tuple(values)
        ),
    )

    assert adapter.submit_depsgraph_interactive_edits(
        SimpleNamespace(updates=(SimpleNamespace(id=object()),)),
        scene=object(),
    ) == edits
    assert submitted == edits


def test_edit_callback_adapter_records_selection_only_changes_without_submitting() -> None:
    selection_resolution = {"changed": True, "group_rejected": False}
    diagnostics: list[dict[str, object]] = []
    selection_records: list[object] = []
    submitted: list[object] = []

    class _Engine:
        def record_interactive_selection_resolution(self, selection_resolution: object) -> None:
            selection_records.append(selection_resolution)

        def submit_interactive_edit(self, edit: object) -> None:
            submitted.append(edit)

    adapter = BlenderEditCallbackAdapter(
        active_engines=lambda: (_Engine(),),
        selection_resolver=lambda context: selection_resolution,
        bridge_diagnostics_recorder=lambda **kwargs: diagnostics.append(dict(kwargs)),
    )

    assert adapter.submit_depsgraph_interactive_edits(SimpleNamespace(updates=())) == []
    assert selection_records == [selection_resolution]
    assert submitted == []
    assert diagnostics[-1] == {
        "suppressed": False,
        "active_engine_count": 1,
        "submitted_edit_count": 0,
        "result_count": 0,
    }
