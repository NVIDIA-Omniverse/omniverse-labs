# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from dataclasses import replace
import importlib
import json
from types import ModuleType, SimpleNamespace
import gc
import sys
import weakref

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

import ovrtx_blender_example.engine as engine_module  # noqa: E402
from ovrtx_blender_example import (  # noqa: E402
    bundled_runtime,
    ovrtx_gpu_lease,
)
from ovrtx_blender_example.engine import (
    build_request_from_scene,
    _rgba16f_to_float_array,
    _track_viewport_engine,
    _untrack_viewport_engine,
    _viewport_draw_geometry,
    _upload_viewport_texture,
    _write_image,
    reconnect_viewport_sessions,
    write_viewport_session_outputs,
    interactive_edit_bridge_diagnostics,
    register_interactive_edit_bridge,
    resolve_blender_selection_to_edit_owners,
    submit_depsgraph_interactive_edits_to_active_viewports,
    submit_interactive_edit_to_active_viewports,
    suppress_interactive_edit_bridge,
    unregister_interactive_edit_bridge,
    viewport_session_statuses,
)
from ovrtx_blender_example import interactive_operator_state as operator_state  # noqa: E402
from ovrtx_blender_example.ovrtx_runtime_client import (  # noqa: E402
    RenderResult,
    _render_worker_startup_diagnostics,
)
from ovrtx_blender_example.render_requests import (  # noqa: E402
    MaterialPresentationLayer,
    RenderRequest,
)
from ovrtx_blender_example.ovrtx_value_updates import OvrtxValueUpdateResult  # noqa: E402
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderRenderIntent,
    BlenderRenderSignalSource,
)
from ovrtx_blender_example import ovrtx_session  # noqa: E402
from ovrtx_blender_example import ovrtx_session_controller as controller_module  # noqa: E402
from ovrtx_blender_example.ovrtx_scene_composition import (  # noqa: E402
    diagnostics as ovrtx_scene_composition_diagnostics,
    _camera_override_layer_text,
    _usda_asset_path as _composition_usda_asset_path,
)
from ovrtx_blender_example import usd_paths as usd_paths  # noqa: E402
from ovrtx_blender_example import color_presentation, viewport_artifact_recorder, viewport_profile, render_requests  # noqa: E402
from ovrtx_blender_example.interactive_edit_planner import (  # noqa: E402
    DataAuthority,
    EditShape,
    edit_location,
    InteractiveEdit,
    InteractiveEditPlanner,
)
from ovrtx_blender_example.scene_generation import (  # noqa: E402
    BlenderId,
    BlenderPrimPath,
    SceneGenerationOwner,
)


@pytest.fixture(autouse=True)
def _current_scene_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    camera_mapping = SimpleNamespace(schema_path="/World/Camera")
    generated = tmp_path / "generated.usdc"
    generated.write_text("#usda 1.0\n", encoding="utf-8")

    def generation_for_scene(scene: object) -> object:
        camera = getattr(scene, "camera", None)
        if camera is None:
            camera = SimpleNamespace(name="Camera")
            scene.camera = camera
        if not hasattr(camera, "session_uid"):
            camera.session_uid = 41
        if not hasattr(camera, "name"):
            camera.name = "Camera"
        return SimpleNamespace(
            blender_prim_paths={BlenderId("OBJECT", 41): camera_mapping},
            materialize_usd=lambda: str(generated),
        )

    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "generation_for_scene",
        generation_for_scene,
    )
    monkeypatch.setattr(engine_module, "_EXACT_STAGE_CONFIGURATION", None)
    engine_module._RENDER_CALLBACK_ADAPTERS.clear()


def _compose_request(request: RenderRequest):
    return ovrtx_session.build_spec(request).ovrtx_scene_composition


def test_internal_exact_stage_configuration_selects_exact_adapter(tmp_path: Path) -> None:
    stage = tmp_path / "exact.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")

    engine_module.configure_exact_stage(
        input_usd_path=str(stage),
        camera_prim_path="/Fixture/Camera",
        render_product_path="/Fixture/Product",
    )
    adapter = engine_module._render_callback_adapter("fixture")

    assert adapter.__class__.__name__ == "ExactStageRenderCallbackAdapter"
    assert adapter._input_usd_path == str(stage)
    assert adapter._translator._include_material_presentation is False


def _presentation_text(composition: object, source: str) -> str:
    record = next(
        item
        for item in composition.presentation_layers
        if item["source"] == source
    )
    return Path(str(record["path"])).read_text(encoding="utf-8")


def _material_scene_layer(
    *,
    digest: str,
    layer_body: str,
    binding_targets: tuple[str, ...] = (),
) -> MaterialPresentationLayer:
    return MaterialPresentationLayer(
        target_path=min(binding_targets) if binding_targets else "",
        layer_body=layer_body.rstrip(),
        authored_properties=tuple(
            (path, "material:binding") for path in binding_targets
        ),
        digest_content={
            "source": "materialx_openpbr",
            "digest": digest,
            "layer_body": layer_body,
        },
        diagnostics={"source": "materialx_openpbr", "digest": digest},
    )


def _timings(**values: float) -> dict[str, float]:
    timings = {phase: 0.0 for phase in viewport_profile.TIMING_PHASES}
    timings.update(values)
    return timings


def _verbose_material_layer(*, digest: str, material_name: str) -> MaterialPresentationLayer:
    layer_body = 'def Scope "OVRTX_Materials"\n{\n}\n'
    return MaterialPresentationLayer(
        target_path="/World/Geom",
        layer_body=layer_body.rstrip(),
        authored_properties=(("/World/Geom", "material:binding"),),
        digest_content={
            "source": "materialx_openpbr",
            "digest": digest,
            "layer_body": layer_body,
        },
        diagnostics={
            "source": "materialx_openpbr",
            "digest": digest,
            "status": "generated",
            "material_count": 1,
            "materials": [
                {
                    "material_name": material_name,
                    "node_inventory": [{"name": "Principled BSDF"}],
                }
            ],
        },
    )


class _ArtifactRuntimeClient:
    def __init__(self, simulation_id: str, *, fail_start: bool = False) -> None:
        self.simulation_id = simulation_id
        self.fail_start = fail_start
        self.startup_diagnostics = {"render_worker": {"status": "ready"}}
        self.last_render_timings: dict[str, object] = {}
        self.shutdown_called = False

    def start_session(self, _spec: object, simulation_id: str | None = None) -> str:
        if self.fail_start:
            self.startup_diagnostics = {"render_worker": {"status": "failed"}}
            raise engine_module.RenderClientError("start failed")
        return simulation_id or self.simulation_id

    def render_result(self, _simulation_id: str, **kwargs: object) -> RenderResult:
        return RenderResult(
            width=1,
            height=1,
            rgba8=b"\x00\x00\x00\xff",
            completed_samples=int(kwargs["additional_samples"]),
            session_completed_samples=1,
            simulation_time_ns=1,
        )

    def delete_simulation(self, _simulation_id: str) -> str:
        return "stopped"

    def shutdown(self) -> None:
        self.shutdown_called = True


def test_viewport_artifact_sources_follow_successful_ensure_not_failed_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        source = tmp_path / "scene.usda"
        source.write_text("#usda 1.0\n", encoding="utf-8")
        installed_entry = _verbose_material_layer(
            digest="installed-material",
            material_name="Installed",
        )
        request = RenderRequest(
            input_usd_path=str(source),
            material_scene_layer=installed_entry,
        )
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR",
            str(tmp_path / "composed"),
        )
        installed_client = _ArtifactRuntimeClient("installed")
        clients = [installed_client, _ArtifactRuntimeClient("recovered")]
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda _request: clients.pop(0),
        )
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine._ensure_viewport_session(request)

        viewport_artifact = render_engine._viewport_artifact()
        viewport_records = viewport_artifact["ovrtx_scene_composition"][
            "presentation_layers"
        ]
        viewport_record = next(
            record
            for record in viewport_records
            if record["source"] == "materialx_openpbr"
        )
        assert viewport_record["materials"][0]["material_name"] == "Installed"

        rejected_request = replace(
            request,
            material_scene_layer=_verbose_material_layer(
                digest="rejected-material",
                material_name="Rejected",
            ),
        )
        installed_client.fail_start = True
        with pytest.raises(module.RenderClientError, match="start failed"):
            render_engine._ensure_viewport_session(rejected_request)

        retained_record = next(
            record
            for record in render_engine._viewport_artifact()[
                "ovrtx_scene_composition"
            ]["presentation_layers"]
            if record["source"] == "materialx_openpbr"
        )
        assert retained_record["digest"] == "installed-material"
        assert retained_record["materials"][0]["material_name"] == "Installed"
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_final_render_callback_restores_verbose_material_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        source = tmp_path / "scene.usda"
        source.write_text("#usda 1.0\n", encoding="utf-8")
        request = RenderRequest(
            input_usd_path=str(source),
            width=1,
            height=1,
            min_samples=1,
            max_samples=1,
            material_scene_layer=_verbose_material_layer(
                digest="final-material",
                material_name="Final",
            ),
        )
        client = _ArtifactRuntimeClient("final")
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda _request: client,
        )
        monkeypatch.setattr(
            module,
            "_render_callback_adapter",
            lambda _engine_id="": SimpleNamespace(final_render=lambda _depsgraph: request),
        )
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR",
            str(tmp_path / "composed"),
        )
        result_path = tmp_path / "render-result.json"
        monkeypatch.setenv("OV_BLENDER_EXAMPLE_RENDER_ARTIFACT", str(result_path))
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine.update_stats = lambda *_args: None
        render_engine._write_blender_result = lambda _result: None
        # The standalone final render polls test_break() while its job runs
        # on the short-lived RPC thread (task05-03).
        render_engine.test_break = lambda: False

        render_engine.render(SimpleNamespace(scene=object()))

        material_record = json.loads(result_path.read_text(encoding="utf-8"))[
            "ovrtx_scene_composition"
        ]["presentation_layers"][0]
        assert material_record["digest"] == "final-material"
        assert material_record["materials"][0]["material_name"] == "Final"
        assert material_record["materials"][0]["node_inventory"] == [
            {"name": "Principled BSDF"}
        ]
        assert client.shutdown_called is True
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_final_render_changes_only_sample_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        source = tmp_path / "scene.usda"
        source.write_text("#usda 1.0\n", encoding="utf-8")
        presentation = color_presentation.presentation_from_scene(
            None,
            requested_mode=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
        )
        translated = RenderRequest(
            input_usd_path=str(source),
            min_samples=1,
            max_samples=8,
            color_presentation=presentation,
        )
        client = _ArtifactRuntimeClient("final")
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda _request: client,
        )
        monkeypatch.setattr(
            module,
            "_render_callback_adapter",
            lambda _engine_id="": SimpleNamespace(
                final_render=lambda _depsgraph: translated
            ),
        )
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR",
            str(tmp_path / "composed"),
        )
        captured: list[RenderRequest] = []
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine.update_stats = lambda *_args: None
        render_engine._write_blender_result = lambda _result: None
        render_engine._write_result_artifact = (
            lambda _result, request, _composition, *, scene: captured.append(request)
        )

        render_engine.render(SimpleNamespace(scene=object()))

        assert len(captured) == 1
        # The final-render job renders chunked, cancellable sample batches
        # to the fixed endpoint: min_samples stays the translated first
        # batch size and only max_samples is clamped. Color presentation is
        # the one final-render-owned replacement (scene-linear HDR policy).
        assert captured[0].min_samples == 1
        assert captured[0].max_samples == 8
        assert captured[0].color_presentation == color_presentation.presentation_from_scene(
            None,
            requested_mode=color_presentation.MODE_SCENE_LINEAR_HDR,
        )
        assert replace(captured[0], color_presentation=presentation) == translated
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_result_artifact_records_env_override_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct (env-override) F12 records env_override with no digest."""

    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        source = tmp_path / "scene.usda"
        source.write_text("#usda 1.0\n", encoding="utf-8")
        request = RenderRequest(
            input_usd_path=str(source),
            width=1,
            height=1,
            min_samples=1,
            max_samples=1,
        )
        client = _ArtifactRuntimeClient("final")
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda _request: client,
        )
        monkeypatch.setattr(
            module,
            "_render_callback_adapter",
            lambda _engine_id="": SimpleNamespace(final_render=lambda _depsgraph: request),
        )
        monkeypatch.setattr(
            module,
            "_final_render_color_presentation_from_scene",
            lambda _scene: {},
        )
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR",
            str(tmp_path / "composed"),
        )
        result_path = tmp_path / "render-result.json"
        monkeypatch.setenv("OV_BLENDER_EXAMPLE_RENDER_ARTIFACT", str(result_path))
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine.update_stats = lambda *_args: None
        render_engine._write_blender_result = lambda _result: None
        render_engine.test_break = lambda: False
        reports: list[tuple[set[str], str]] = []
        render_engine.report = lambda levels, message: reports.append((set(levels), message))

        render_engine.render(SimpleNamespace(scene=object()))

        assert reports == []
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        assert artifact["input_source"] == "env_override"
        assert artifact["authored_generation_digest"] == ""
        assert artifact["authored_generation"] is None
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_viewport_artifact_direct_route_provenance_is_env_override() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine._viewport_request = RenderRequest(
            input_usd_path="/fixtures/render.usda",
        )

        artifact = render_engine._viewport_artifact()

        assert artifact["input_source"] == "env_override"
        assert artifact["authored_generation_digest"] == ""
        assert artifact["authored_generation"] is None
        assert artifact["input_usd_path"] == "/fixtures/render.usda"
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_final_render_ends_active_viewport_before_starting_final_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        source = tmp_path / "scene.usda"
        source.write_text("#usda 1.0\n", encoding="utf-8")
        monkeypatch.setenv(
            "OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR",
            str(tmp_path / "composed"),
        )
        monkeypatch.setenv(
            ovrtx_gpu_lease.LOCK_DIR_ENV,
            str(tmp_path / "gpu-locks"),
        )
        monkeypatch.setenv("OVRTX_ACTIVE_CUDA_GPUS", "0")
        request = RenderRequest(
            input_usd_path=str(source),
            width=1,
            height=1,
            min_samples=1,
            max_samples=1,
        )

        class _LeasedRuntimeClient(_ArtifactRuntimeClient):
            def __init__(self, label: str) -> None:
                super().__init__(label)
                self.label = label
                self.lease: ovrtx_gpu_lease.OvrtxGpuLease | None = None
                self.started = False

            def start_session(self, _spec: object, simulation_id: str | None = None) -> str:
                try:
                    self.lease = ovrtx_gpu_lease.acquire(
                        metadata={"entrypoint": self.label},
                        timeout_s=0,
                    )
                except ovrtx_gpu_lease.OvrtxGpuLeaseBusy as exc:
                    raise module.RenderClientError(str(exc)) from exc
                self.started = True
                return simulation_id or self.label

            def shutdown(self) -> None:
                super().shutdown()
                lease = self.lease
                self.lease = None
                if lease is not None:
                    lease.close()

        viewport_client = _LeasedRuntimeClient("viewport")
        final_client = _LeasedRuntimeClient("final")
        clients = [viewport_client, final_client]
        monkeypatch.setattr(
            controller_module,
            "_runtime_client_from_request",
            lambda _request: clients.pop(0),
        )
        monkeypatch.setattr(
            module,
            "_render_callback_adapter",
            lambda _engine_id="": SimpleNamespace(final_render=lambda _depsgraph: request),
        )
        viewport_engine = module.OvrtxExampleRenderEngine()
        end_reasons: list[object] = []
        viewport_engine._write_viewport_session_outputs = (
            lambda *, end_reason="": end_reasons.append(end_reason)
        )
        viewport_engine._ensure_viewport_session(request)
        assert viewport_client.lease is not None

        final_engine = module.OvrtxExampleRenderEngine()
        reports: list[tuple[set[str], str]] = []
        final_engine.report = lambda levels, message: reports.append((set(levels), message))
        final_engine.update_stats = lambda *_args: None
        final_engine._write_blender_result = lambda _result: None
        final_engine._write_result_artifact = lambda *_args, **_kwargs: None

        final_engine.render(SimpleNamespace(scene=object()))

        assert reports == []
        assert viewport_client.shutdown_called is True
        assert final_client.started is True
        assert final_client.shutdown_called is True
        assert end_reasons == [module.ViewportSessionEndReason.SESSION_REPLACED]
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)



def _camera_projection(
    *,
    focal_length: float = 28.0,
    horizontal_aperture: float = 50.0,
    vertical_aperture: float = 28.0,
    render_size: tuple[int, int] = (1280, 720),
) -> render_requests.CameraProjectionState:
    return render_requests.CameraProjectionState(
        source=render_requests.PERSPECTIVE_USER_VIEW,
        focal_length=focal_length,
        horizontal_aperture=horizontal_aperture,
        vertical_aperture=vertical_aperture,
        render_size=render_size,
    )


def _reload_engine_module_with_fake_bpy() -> tuple[object, bool, object | None]:
    fake_bpy = ModuleType("bpy")
    fake_bpy.types = SimpleNamespace(RenderEngine=object)
    def persistent(function: object) -> object:
        setattr(function, "_bpy_persistent", None)
        return function
    fake_bpy.app = SimpleNamespace(handlers=SimpleNamespace(persistent=persistent))
    had_bpy = "bpy" in sys.modules
    original_bpy = sys.modules.get("bpy")
    sys.modules["bpy"] = fake_bpy
    try:
        module = importlib.reload(engine_module)
        return module, had_bpy, original_bpy
    except Exception:
        _restore_engine_module_bpy(had_bpy, original_bpy)
        raise


def _restore_engine_module_bpy(had_bpy: bool, original_bpy: object | None) -> None:
    if had_bpy:
        sys.modules["bpy"] = original_bpy
    else:
        sys.modules.pop("bpy", None)
    importlib.reload(engine_module)


def test_live_depsgraph_bridge_handler_is_persistent() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        assert hasattr(module._live_interactive_edit_depsgraph_handler, "_bpy_persistent")
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


@pytest.mark.parametrize("callback_name", ["render", "view_update", "view_draw"])
def test_render_callbacks_report_translation_errors_without_replacing_session(
    callback_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        render_engine = module.OvrtxExampleRenderEngine()
        active_request = object()
        render_engine._viewport_request = active_request
        reports: list[tuple[set[str], str]] = []
        session_requests: list[object] = []
        render_engine.report = lambda levels, message: reports.append((set(levels), message))
        render_engine.update_stats = lambda *args: None
        render_engine._ensure_viewport_session = (
            lambda request, _scene=None: session_requests.append(request)
        )
        render_engine._use_native_viewport_fallback = lambda presentation: False
        monkeypatch.setattr(
            module.viewport_presentation,
            "apply_native_fallback_for_context",
            lambda context: {},
        )

        class _FailingAdapter:
            def final_render(self, depsgraph: object) -> object:
                raise module.BlenderSignalTranslationError(
                    BlenderRenderSignalSource.FINAL_RENDER,
                    "material conversion failed",
                )

            def view_update(self, context: object, depsgraph: object) -> object:
                raise module.BlenderSignalTranslationError(
                    BlenderRenderSignalSource.VIEW_UPDATE,
                    "material conversion failed",
                )

            def view_draw(self, context: object, depsgraph: object) -> object:
                raise module.BlenderSignalTranslationError(
                    BlenderRenderSignalSource.VIEW_DRAW,
                    "material conversion failed",
                )

        monkeypatch.setattr(module, "_render_callback_adapter", lambda engine_id="": _FailingAdapter())
        depsgraph = SimpleNamespace(scene=object())
        if callback_name == "render":
            render_engine.render(depsgraph)
        else:
            getattr(render_engine, callback_name)(object(), depsgraph)

        source = {
            "render": "final_render",
            "view_update": "view_update",
            "view_draw": "view_draw",
        }[callback_name]
        assert reports == [({"ERROR"}, f"{source}: material conversion failed")]
        assert session_requests == []
        assert render_engine._viewport_request is active_request
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


@pytest.mark.parametrize("callback_name", ["view_update", "view_draw"])
def test_viewport_callbacks_defer_while_scene_generation_is_authoring(
    callback_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        render_engine = module.OvrtxExampleRenderEngine()
        scene = object()
        translated = []
        monkeypatch.setattr(
            module.scene_generation_sessions,
            "is_authoring",
            lambda received: received is scene,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "_render_callback_adapter",
            lambda engine_id="": translated.append(engine_id),
        )

        getattr(render_engine, callback_name)(object(), SimpleNamespace(scene=scene))

        assert translated == []
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_outer_view_update_hands_off_after_nested_callback_defers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        render_engine = module.OvrtxExampleRenderEngine()
        scene = object()
        depsgraph = SimpleNamespace(scene=scene)
        context = object()
        request = module.RenderRequest()
        authoring = False
        translated = []
        handed_off = []

        class _ReentrantAdapter:
            def view_update(self, received_context: object, received_depsgraph: object) -> object:
                nonlocal authoring
                translated.append((received_context, received_depsgraph))
                authoring = True
                render_engine.view_update(context, depsgraph)
                authoring = False
                return request

        monkeypatch.setattr(
            module.scene_generation_sessions,
            "is_authoring",
            lambda received: received is scene and authoring,
        )
        monkeypatch.setattr(
            module.viewport_presentation,
            "apply_native_fallback_for_context",
            lambda _context: {},
        )
        monkeypatch.setattr(
            module,
            "_render_callback_adapter",
            lambda engine_id="": _ReentrantAdapter(),
        )
        render_engine._use_native_viewport_fallback = lambda _presentation: False
        render_engine._begin_async_viewport_session = (
            lambda value, *_args: handed_off.append(value)
        )
        render_engine._write_viewport_camera_snapshot = lambda *_args: None
        render_engine.update_stats = lambda *_args: None
        render_engine.tag_redraw = lambda: None

        render_engine.view_update(context, depsgraph)

        assert translated == [(context, depsgraph)]
        assert handed_off == [request]
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


class _HealthyNativeClient:
    def check_health(self) -> dict[str, object]:
        return {"serving": True, "endpoint": "127.0.0.1:50051"}


class _UnhealthyNativeClient:
    def check_health(self) -> dict[str, object]:
        return {"serving": False, "endpoint": "127.0.0.1:50051"}


def test_render_worker_startup_diagnostics_records_health() -> None:
    diagnostics = _render_worker_startup_diagnostics(
        _HealthyNativeClient(),
        simulation_id="sim",
        worker_command="worker",
    )

    assert diagnostics["status"] == "running"
    assert diagnostics["simulation_id"] == "sim"
    assert diagnostics["health"] == {"serving": True, "endpoint": "127.0.0.1:50051"}


def test_final_render_policy_disables_blender_postprocess() -> None:
    assert engine_module.FINAL_RENDER_USE_POSTPROCESS is False


def test_render_worker_startup_diagnostics_marks_unhealthy_worker_failed() -> None:
    diagnostics = _render_worker_startup_diagnostics(
        _UnhealthyNativeClient(),
        simulation_id="sim",
        worker_command="worker",
    )

    assert diagnostics["status"] == "failed"
    assert diagnostics["error"] == "render worker health reported serving=false"


def _viewport_request_scene(
    *,
    sync_viewport_camera: bool = True,
    presentation: str = color_presentation.MODE_SCENE_LINEAR_HDR,
) -> SimpleNamespace:
    return SimpleNamespace(
        render=SimpleNamespace(
            resolution_x=1280,
            resolution_y=720,
            resolution_percentage=100,
        ),
        ovrtx_example=SimpleNamespace(
            render_product_path="/Render/Product",
            min_samples=1,
            max_samples=128,
            camera_prim_path="/World/Camera",
            sync_viewport_camera=sync_viewport_camera,
            simulation_reset_token=0,
            color_presentation_mode=presentation,
        ),
        view_settings=SimpleNamespace(
            view_transform="AgX",
            look="Medium High Contrast",
            exposure=0.25,
            gamma=1.1,
        ),
        display_settings=SimpleNamespace(display_device="sRGB"),
        frame_current=1,
        frame_start=1,
        frame_end=1,
    )


def _frame_point(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def _orthographic_camera_data(
    *,
    frame_width: float,
    frame_height: float,
    frame_center_x: float = 0.0,
    frame_center_y: float = 0.0,
    lens: float,
    clip_start: float,
    clip_end: float,
    sensor_fit: str = "AUTO",
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> SimpleNamespace:
    half_width = frame_width * 0.5
    half_height = frame_height * 0.5

    def view_frame(*, scene: object) -> tuple[SimpleNamespace, ...]:
        return (
            _frame_point(frame_center_x - half_width, frame_center_y - half_height),
            _frame_point(frame_center_x + half_width, frame_center_y - half_height),
            _frame_point(frame_center_x + half_width, frame_center_y + half_height),
            _frame_point(frame_center_x - half_width, frame_center_y + half_height),
        )

    return SimpleNamespace(
        type="ORTHO",
        lens=lens,
        sensor_fit=sensor_fit,
        clip_start=clip_start,
        clip_end=clip_end,
        shift_x=shift_x,
        shift_y=shift_y,
        dof=SimpleNamespace(use_dof=False),
        view_frame=view_frame,
    )


def _viewport_request_context(
    *,
    view_perspective: str = "PERSP",
    scene: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        region=SimpleNamespace(width=2766, height=1228),
        region_data=SimpleNamespace(
            view_perspective=view_perspective,
            view_matrix=_FakeViewMatrix(),
            window_matrix=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 2.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
        ),
        screen=SimpleNamespace(is_animation_playing=False),
        scene=scene,
    )


class _FakeBuffer:
    def __init__(self, kind: str, length: int, data: object) -> None:
        self.kind = kind
        self.length = length
        self.data = data


class _FakeTextureBase:
    def __init__(self, size: tuple[int, int], *, format: str, data: _FakeBuffer) -> None:
        self.size = size
        self.format = format
        self.data = data
        self.filtered = False
        self.__class__.created.append(self)

    def filter_mode(self, enabled: bool) -> None:
        self.filtered = enabled


class _FakeTexture(_FakeTextureBase):
    created: list["_FakeTexture"] = []

    def __init__(self, size: tuple[int, int], *, format: str, data: _FakeBuffer) -> None:
        super().__init__(size, format=format, data=data)
        self.updates: list[tuple[_FakeBuffer, str]] = []

    def update(self, buffer: _FakeBuffer, *, format: str) -> None:
        self.updates.append((buffer, format))


class _FakeTextureWithoutUpdate(_FakeTextureBase):
    created: list["_FakeTextureWithoutUpdate"] = []


class _FakeTextureWithFailingUpdate(_FakeTexture):
    created: list["_FakeTextureWithFailingUpdate"] = []

    def update(self, buffer: _FakeBuffer, *, format: str) -> None:
        raise RuntimeError("patched update rejected payload")


class _FakeTextureRejectingUbyte(_FakeTexture):
    created: list["_FakeTextureRejectingUbyte"] = []

    def __init__(self, size: tuple[int, int], *, format: str, data: _FakeBuffer) -> None:
        if data.kind == "UBYTE":
            raise RuntimeError("stock constructor rejected UBYTE payload")
        super().__init__(size, format=format, data=data)


class _FakeGpu:
    def __init__(self, texture_type: type[object]) -> None:
        class Types:
            Buffer = _FakeBuffer
            GPUTexture = texture_type

        self.types = Types


class _FakeTrackedEngine:
    def __init__(self) -> None:
        self.write_count = 0
        self.end_reasons: list[object] = []

    def _write_viewport_session_outputs(self, *, end_reason: object = "") -> None:
        self.write_count += 1
        self.end_reasons.append(end_reason)


class _FakeEditableEngine:
    def __init__(self) -> None:
        self.received_edits: list[InteractiveEdit] = []
        self.selection_records: list[dict[str, object]] = []

    def submit_interactive_edit(self, edit: InteractiveEdit) -> dict[str, object]:
        self.received_edits.append(edit)
        return {"accepted": True, "data_authority": edit.data_authority.value}

    def record_interactive_selection_resolution(
        self,
        selection_resolution: dict[str, object],
    ) -> dict[str, object]:
        record = {"action": "observation", "selection_resolution": dict(selection_resolution)}
        self.selection_records.append(record)
        return record


class _FakeReconnectableEngine:
    def __init__(self, *, had_session: bool = True, stopped: bool = True) -> None:
        self.had_session = had_session
        self.stopped = stopped
        self.reconnect_count = 0

    def _request_viewport_session_reconnect(self) -> tuple[bool, bool]:
        self.reconnect_count += 1
        return self.had_session, self.stopped


class _FakeStatusEngine:
    def __init__(self, status: dict[str, object]) -> None:
        self.status = status

    def _viewport_session_status(self) -> dict[str, object]:
        return dict(self.status)


class _FakeLockingEngine:
    def __init__(self, lock: operator_state.PhysicsPlaybackLock) -> None:
        self.lock = lock
        self.received_edits: list[InteractiveEdit] = []

    def submit_interactive_edit(self, edit: InteractiveEdit) -> object:
        result = self.lock.reject_edit(edit)
        if result is not None:
            return result
        self.received_edits.append(edit)
        return {"accepted": True, "data_authority": edit.data_authority.value}


class _FakeDepsgraphUpdate:
    def __init__(self, id_data: object) -> None:
        self.id = id_data


class _FakeDepsgraph:
    def __init__(self, updates: list[object], *, view_layer: object | None = None) -> None:
        self.updates = updates
        self.view_layer = view_layer


def test_engine_feeds_one_usd_prim_resolver_without_hiding_builder_context() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    original_builder = module.build_interactive_edits_from_depsgraph
    original_render_callback_adapter = module._render_callback_adapter
    original_current_generation_edit_context = (
        module.scene_generation_sessions.current_generation_edit_context
    )
    try:
        light_object = object()
        module.bpy.data = SimpleNamespace(objects=(light_object,))
        builder_calls: list[tuple[object, dict[str, object]]] = []

        def fake_builder(depsgraph: object, **kwargs: object) -> list[str]:
            builder_calls.append((depsgraph, kwargs))
            return ["planned-edit"]

        class _PrimResolver:
            def __init__(self) -> None:
                self.requests: list[object] = []

            def scan(self, request: object) -> None:
                self.requests.append(request)

            def reset(self) -> None:
                pass

        module.build_interactive_edits_from_depsgraph = fake_builder
        render_engine = module.OvrtxExampleRenderEngine()
        prim_resolver = _PrimResolver()
        request = module.RenderRequest(input_usd_path="/base.usda")
        edited_id = object()
        depsgraph = SimpleNamespace(updates=(SimpleNamespace(id=edited_id),))
        render_engine._viewport_request = request
        render_engine._usd_prim_resolver = prim_resolver
        render_engine._ovrtx_scene_composition = SimpleNamespace(
            source_scene_path="/base.usda",
            composed_scene_path="/composed.usda",
            session_layer_identifiers=("anon:session-layer",),
        )

        module._render_callback_adapter = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("active viewport request should be reused")
        )
        policies = object()
        translator_factory = module.blender_signal_translation.InteractiveEditTranslator
        edits = render_engine.build_interactive_edits_from_depsgraph(
            depsgraph,
            selection_resolution={"selection": "resolved"},
            value_edit_conversion_policies=policies,
            edit_translator_factory=translator_factory,
        )

        assert edits == ["planned-edit"]
        assert prim_resolver.requests == [request]
        assert len(builder_calls) == 1
        called_depsgraph, kwargs = builder_calls[0]
        assert tuple(update.id for update in called_depsgraph.updates) == (edited_id,)
        assert kwargs["usd_prim_resolver"] is prim_resolver
        assert kwargs["light_objects"] == (light_object,)
        assert kwargs["selection_resolution"] == {"selection": "resolved"}
        assert kwargs["write_target_input_usd_path"] is None
        assert kwargs["write_target_ignored_layer_identifiers"] == ()
        assert kwargs["value_edit_conversion_policies"] is policies

        generation_resolver = _PrimResolver()
        module.scene_generation_sessions.current_generation_edit_context = (
            lambda received_scene: (
                generation_resolver,
                (light_object,),
            )
        )
        render_engine._viewport_request = None
        depsgraph.scene = object()

        render_engine.build_interactive_edits_from_depsgraph(depsgraph)

        assert prim_resolver.requests == [request]
        assert generation_resolver.requests == []
        assert builder_calls[-1][1]["usd_prim_resolver"] is generation_resolver
        assert builder_calls[-1][1]["light_objects"] == (light_object,)
    finally:
        module.build_interactive_edits_from_depsgraph = original_builder
        module._render_callback_adapter = original_render_callback_adapter
        module.scene_generation_sessions.current_generation_edit_context = (
            original_current_generation_edit_context
        )
        _restore_engine_module_bpy(had_bpy, original_bpy)


class _FakeViewMatrix:
    def inverted(self) -> tuple[tuple[float, ...], ...]:
        return (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )


class _FakeBlenderObject(dict):
    def __init__(self, name: str = "Cube") -> None:
        super().__init__()
        self.name = name
        self.name_full = name
        self.type = "MESH"
        self.bl_rna = SimpleNamespace(identifier="Object")
        self.hide_select = False
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


def _texture_result(width: int = 2, height: int = 2) -> RenderResult:
    return RenderResult(
        width=width,
        height=height,
        rgba8=bytes(range(width * height * 4)),
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=10,
    )


def test_artifact_status_tracks_live_and_closed_sessions() -> None:
    recorder = viewport_artifact_recorder.Recorder(
        profile_factory=lambda: {},
        record_profile=lambda _profile, _record: None,
        profile_summary=lambda _profile, _latency_ms: {"enabled": True},
        enabled=lambda: True,
    )
    request = RenderRequest(max_samples=64)
    partial_result = RenderResult(
        width=1280,
        height=720,
        rgba8=b"",
        completed_samples=32,
        session_completed_samples=32,
        simulation_time_ns=10,
    )
    complete_result = RenderResult(
        width=1280,
        height=720,
        rgba8=b"",
        completed_samples=64,
        session_completed_samples=64,
        simulation_time_ns=20,
    )

    def artifact_status(render_result: RenderResult, *, running: bool) -> str:
        return recorder.artifact(
            viewport_artifact_recorder.State(
                simulation_id=None,
                request=request,
                result=render_result,
                snapshot_index=0,
                render_count=0,
                draw_count=0,
                snapshot_count=0,
                camera_update_count=0,
                camera_controls_mode="usd_camera",
                running=running,
            )
        )["status"]

    assert artifact_status(partial_result, running=True) == "running"
    assert artifact_status(partial_result, running=False) == "stopped"
    assert artifact_status(complete_result, running=True) == "running"
    assert artifact_status(complete_result, running=False) == "complete"


def test_viewport_output_write_tracks_engines_weakly() -> None:
    engine = _FakeTrackedEngine()
    reference = weakref.ref(engine)

    try:
        _track_viewport_engine(engine)
        assert write_viewport_session_outputs() == 1
        assert engine.write_count == 1
        assert engine.end_reasons == [engine_module.ViewportSessionEndReason.OUTPUT_WRITTEN]
    finally:
        _untrack_viewport_engine(engine)
    assert write_viewport_session_outputs() == 0

    _track_viewport_engine(engine)
    del engine
    gc.collect()

    assert reference() is None
    assert write_viewport_session_outputs() == 0


def test_explicit_viewport_profile_write_is_not_repeated_during_session_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        profile_path = tmp_path / "viewport-profile.json"
        monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_PROFILE", str(profile_path))
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine._ovrtx_session_controller = SimpleNamespace(
            diagnostics=lambda: {
                "active": True,
                "lifecycle_events": (),
                "startup": {"render_worker": {"status": "ready"}},
            },
            ensure=lambda _request: SimpleNamespace(
                composition=None,
                session_started=False,
            ),
            shutdown=lambda: None,
        )
        render_engine._write_viewport_artifact = lambda **_kwargs: 0.0
        artifact_calls: list[str] = []
        render_engine._viewport_artifact = lambda *_args, **kwargs: (
            artifact_calls.append(str(kwargs.get("end_reason", "")))
            or {"artifact_id": "ovrtx-viewport-preview"}
        )

        render_engine._write_viewport_session_outputs(
            end_reason=module.ViewportSessionEndReason.OUTPUT_WRITTEN
        )
        monkeypatch.setattr(
            module.session_lifecycle,
            "write_crash_marker",
            lambda **_kwargs: {"marker_active": True},
        )
        monkeypatch.setattr(
            module.session_lifecycle,
            "clear_crash_marker",
            lambda: {"marker_active": False},
        )
        render_engine._ensure_viewport_session(RenderRequest())
        render_engine._end_viewport_session(module.ViewportSessionEndReason.ENGINE_DESTROYED)

        assert artifact_calls == [str(module.ViewportSessionEndReason.OUTPUT_WRITTEN)]
        assert json.loads(profile_path.read_text(encoding="utf-8"))[
            "viewport_artifact"
        ] == {"artifact_id": "ovrtx-viewport-preview"}
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_viewport_profile_write_guard_resets_for_new_draw_and_session_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine._viewport_session_outputs_written = True
        render_engine._viewport_request = RenderRequest(width=1, height=1)
        render_engine._record_profile(
            _texture_result(width=1, height=1),
            _timings(),
            True,
            started_at_ns=1,
            ended_at_ns=2,
            started_monotonic_ns=1,
            rgba_available_monotonic_ns=2,
            ended_monotonic_ns=3,
            span_boundaries={},
        )
        assert render_engine._viewport_session_outputs_written is False

        render_engine._viewport_session_outputs_written = True
        render_engine._ovrtx_session_controller = SimpleNamespace(
            ensure=lambda _request: SimpleNamespace(
                composition=None,
                session_started=True,
            ),
            diagnostics=lambda: {
                "active": True,
                "lifecycle_events": (),
                "startup": {"render_worker": {"status": "ready"}},
            },
            shutdown=lambda: None,
        )
        monkeypatch.setattr(
            module.session_lifecycle,
            "write_crash_marker",
            lambda **_kwargs: {"marker_active": True},
        )
        render_engine._ensure_viewport_session(RenderRequest())
        assert render_engine._viewport_session_outputs_written is False
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_reconnect_viewport_sessions_uses_engine_owned_reconnect_hook() -> None:
    live_engine = _FakeReconnectableEngine(had_session=True)
    idle_engine = _FakeReconnectableEngine(had_session=False)
    try:
        _track_viewport_engine(live_engine)
        _track_viewport_engine(idle_engine)

        diagnostics = reconnect_viewport_sessions()
    finally:
        _untrack_viewport_engine(live_engine)
        _untrack_viewport_engine(idle_engine)

    assert diagnostics["status"] == "requested"
    assert diagnostics["teardown_confirmed"] is True
    assert diagnostics["active_session_count"] == 2
    assert diagnostics["reconnected_session_count"] == 1
    assert diagnostics["end_reason"] == engine_module.ViewportSessionEndReason.RECONNECT_REQUESTED
    assert live_engine.reconnect_count == 1
    assert idle_engine.reconnect_count == 1


def test_reconnect_viewport_sessions_reports_unconfirmed_teardown() -> None:
    engine = _FakeReconnectableEngine(stopped=False)
    try:
        _track_viewport_engine(engine)

        diagnostics = reconnect_viewport_sessions()
    finally:
        _untrack_viewport_engine(engine)

    assert diagnostics["status"] == "teardown_unconfirmed"
    assert diagnostics["teardown_confirmed"] is False
    assert diagnostics["reconnected_session_count"] == 0


def test_final_render_init_reconnects_live_viewport(monkeypatch) -> None:
    scene = SimpleNamespace(render=SimpleNamespace(engine=engine_module.ENGINE_ID))
    request = RenderRequest()
    reconnected: list[bool] = []
    monkeypatch.setattr(engine_module, "_FINAL_RENDER_RESTORE_VIEWPORT", False)
    monkeypatch.setattr(engine_module, "_FINAL_RENDER_REQUEST", None)
    monkeypatch.setattr(
        engine_module,
        "reconnect_viewport_sessions",
        lambda: reconnected.append(True) or {"active_session_count": 1},
    )
    monkeypatch.setattr(
        engine_module,
        "_render_callback_adapter",
        lambda _adapter_id: SimpleNamespace(
            final_render_from_scene=lambda _scene: request
        ),
    )

    engine_module._final_render_init_handler(scene)

    assert reconnected == [True]
    assert engine_module._FINAL_RENDER_RESTORE_VIEWPORT is True
    assert engine_module._FINAL_RENDER_REQUEST is request


def test_restart_ovrtx_workers_terminates_and_respawns_worker(monkeypatch) -> None:
    import ovrtx_blender_example as _pkg
    from ovrtx_blender_example import runtime_services

    restarted: list[Path] = []
    monkeypatch.setattr(
        _pkg,
        "runtime_bundle_status",
        lambda: {"state": "ready", "current_root": "/runtime"},
    )
    monkeypatch.setattr(
        runtime_services.owner,
        "restart_ovrtx",
        lambda root: restarted.append(root),
    )
    monkeypatch.setattr(engine_module.scene_generation_sessions, "pause_preparation", lambda: True)
    monkeypatch.setattr(engine_module.scene_generation_sessions, "resume_preparation", lambda: None)

    diagnostics = engine_module.restart_ovrtx_workers()

    assert restarted == [Path("/runtime")]
    assert diagnostics["status"] == "restarted"
    assert diagnostics["end_reason"] == engine_module.ViewportSessionEndReason.WORKER_RESTART_REQUESTED


def test_restart_stops_all_panes_before_replacing_shared_worker(monkeypatch) -> None:
    import ovrtx_blender_example as _pkg
    from ovrtx_blender_example import runtime_services

    events: list[str] = []
    scene = SimpleNamespace(session_uid=42)

    class Engine:
        _viewport_scene = scene

        def __init__(self, name: str) -> None:
            self.name = name

        def _request_viewport_worker_restart(self) -> tuple[bool, bool]:
            events.append(f"stop:{self.name}")
            return True, True

    first, second = Engine("first"), Engine("second")
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "deactivate_all_ovrtx",
        lambda: events.append("deactivate") or True,
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "pause_preparation",
        lambda: events.append("pause") or True,
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "resume_preparation",
        lambda: events.append("resume"),
    )
    monkeypatch.setattr(
        _pkg,
        "runtime_bundle_status",
        lambda: {"state": "ready", "current_root": "/runtime"},
    )
    monkeypatch.setattr(
        runtime_services.owner,
        "restart_ovrtx",
        lambda _root: events.append("restart"),
    )
    try:
        _track_viewport_engine(first)
        _track_viewport_engine(second)
        diagnostics = engine_module.restart_ovrtx_workers()
    finally:
        _untrack_viewport_engine(first)
        _untrack_viewport_engine(second)

    assert diagnostics["status"] == "restarted"
    assert set(events[:2]) == {"stop:first", "stop:second"}
    assert events[2:] == ["pause", "deactivate", "restart", "resume"]


def test_restart_does_not_replace_worker_after_unconfirmed_pane_stop(monkeypatch) -> None:
    from ovrtx_blender_example import runtime_services

    class Engine:
        _viewport_scene = None

        def _request_viewport_worker_restart(self) -> tuple[bool, bool]:
            return True, False

    engine = Engine()
    monkeypatch.setattr(
        runtime_services.owner,
        "restart_ovrtx",
        lambda _root: pytest.fail("worker must remain alive"),
    )
    try:
        _track_viewport_engine(engine)
        diagnostics = engine_module.restart_ovrtx_workers()
    finally:
        _untrack_viewport_engine(engine)

    assert diagnostics["status"] == "teardown_unconfirmed"
    assert diagnostics["teardown_confirmed"] is False


def test_viewport_session_statuses_reads_engine_owned_status() -> None:
    engine = _FakeStatusEngine({"status": "live", "label": "Live"})
    try:
        _track_viewport_engine(engine)

        diagnostics = viewport_session_statuses()
    finally:
        _untrack_viewport_engine(engine)

    assert diagnostics == {
        "status": "available",
        "active_session_count": 1,
        "sessions": [{"status": "live", "label": "Live"}],
    }


def test_viewport_session_status_exposes_runtime_terminal_failure() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        render_engine = module.OvrtxExampleRenderEngine()

        class _Controller:
            def diagnostics(self) -> dict[str, object]:
                return {"active": True}

            def shutdown(self) -> None:
                pass

        render_engine._ovrtx_session_controller = _Controller()
        render_engine._current_result = _texture_result()
        render_engine._viewport_request = RenderRequest(max_samples=32)
        render_engine._runtime_tick_result = module.RuntimeTickResult(
            status=module.RuntimeTickStatus.FAILED,
            enabled=True,
            skipped_reason="OVPhysX step failed",
        )

        status = render_engine._viewport_session_status()

        assert status["status"] == "live"
        assert status["runtime_status"] == "failed"
        assert status["runtime_failure"] == "OVPhysX step failed"
        assert status["completed_samples"] == 1
        assert status["max_samples"] == 32
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_active_viewport_edit_submission_reaches_tracked_engines() -> None:
    engine = _FakeEditableEngine()
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_layer_id="/layers/scene.usda",
            usd_prim_path="/World/TestScene/Cube",
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
        ),
        value=((1.0, 0.0, 0.0, 0.0),) * 4,
    )

    try:
        _track_viewport_engine(engine)
        results = submit_interactive_edit_to_active_viewports(edit)
    finally:
        _untrack_viewport_engine(engine)

    assert results == [{"accepted": True, "data_authority": "view"}]
    assert engine.received_edits == [edit]


def test_shared_authoring_runtime_receives_one_edit_submission() -> None:
    runtime = object()
    first = _FakeEditableEngine()
    second = _FakeEditableEngine()
    first._viewport_generation_runtime = runtime
    second._viewport_generation_runtime = runtime
    edit = InteractiveEdit(
        shape=EditShape.VALUE,
        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_prim_path="/World/Hat",
            usd_attribute="xformOp:transform",
            blender_property_path="matrix_world",
        ),
        value=((1.0, 0.0, 0.0, 1.0),) * 4,
    )

    try:
        _track_viewport_engine(first)
        _track_viewport_engine(second)
        results = submit_interactive_edit_to_active_viewports(edit)
    finally:
        _untrack_viewport_engine(first)
        _untrack_viewport_engine(second)

    assert len(results) == 1
    assert len(first.received_edits) + len(second.received_edits) == 1


def test_selection_resolution_selects_child_edit_owner() -> None:
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

    diagnostics = resolve_blender_selection_to_edit_owners(context)

    assert diagnostics["changed"] is True
    assert diagnostics["selected_object_count"] == 1
    assert diagnostics["resolved_owner_count"] == 1
    assert child.select_get() is False
    assert owner.select_get() is True
    assert view_layer_objects.active is owner


def test_live_depsgraph_bridge_submits_tagged_edits_to_active_viewports() -> None:
    engine = _FakeEditableEngine()
    obj = _FakeBlenderObject()
    obj.session_uid = 101
    obj.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    obj.select_set(True)
    context = SimpleNamespace(
        selected_objects=[obj],
        scene=SimpleNamespace(objects=_FakeBlenderObjectCollection([obj])),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=obj)),
    )

    try:
        _track_viewport_engine(engine)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            _FakeDepsgraph([_FakeDepsgraphUpdate(obj)]),
            context=context,
        )
    finally:
        _untrack_viewport_engine(engine)

    diagnostics = interactive_edit_bridge_diagnostics()
    assert results == [{"accepted": True, "data_authority": "view"}]
    assert len(engine.received_edits) == 1
    assert diagnostics["suppressed"] is False
    assert diagnostics["last_active_viewport_engine_count"] == 1
    assert diagnostics["last_submitted_edit_count"] == 1
    assert diagnostics["last_result_count"] == 1
    assert diagnostics["last_error"] == ""
    submitted = engine.received_edits[0]
    selection_record = submitted.provenance["selection_resolution"]
    assert selection_record["source_name"] == "Cube"
    assert selection_record["owner_category"] == "view_value_owner"


def test_live_depsgraph_bridge_submits_current_scene_group_once(
    monkeypatch,
) -> None:
    scene = SimpleNamespace(session_uid=12)
    submitted = []
    matching = _FakeEditableEngine()
    foreign = _FakeEditableEngine()
    matching._viewport_request = object()
    foreign._viewport_request = object()
    matching._viewport_generation_runtime = object()
    foreign._viewport_generation_runtime = object()
    obj = _FakeBlenderObject()
    obj.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "resolve_current_scene_edit_group",
        lambda _scene, edits, _selection: tuple(edits),
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "submit_current_scene_edit_group",
        lambda _scene, edits, _selection: submitted.extend(edits) or ({"accepted": True},),
    )

    try:
        _track_viewport_engine(matching)
        _track_viewport_engine(foreign)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            _FakeDepsgraph([_FakeDepsgraphUpdate(obj)]),
            scene=scene,
        )
    finally:
        _untrack_viewport_engine(matching)
        _untrack_viewport_engine(foreign)

    assert len(results) == 1
    assert len(submitted) == 1
    assert matching.received_edits == []
    assert foreign.received_edits == []


def test_live_depsgraph_bridge_uses_callback_scene_selection_across_windows(
    monkeypatch,
    tmp_path,
) -> None:
    callback_scene = SimpleNamespace(
        session_uid=22,
        objects=_FakeBlenderObjectCollection(),
    )
    global_scene = SimpleNamespace(
        session_uid=11,
        objects=_FakeBlenderObjectCollection(),
    )
    global_owner = _FakeBlenderObject("Global Owner")
    global_child = _FakeBlenderObject("Global Child")
    global_child[usd_paths.SELECTION_OWNER_OBJECT_PROP] = global_owner.name
    global_child.select_set(True)
    global_scene.objects.extend((global_owner, global_child))

    edited = _FakeBlenderObject("Callback Cube")
    edited.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/callback.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/Callback/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    edited.session_uid = 220
    edited.select_set(True)
    callback_scene.objects.append(edited)
    global_view_layer = SimpleNamespace(objects=SimpleNamespace(active=global_child))
    callback_view_layer = SimpleNamespace(objects=SimpleNamespace(active=edited))
    monkeypatch.setattr(
        engine_module,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(
                scene=global_scene,
                selected_objects=[global_child],
                view_layer=global_view_layer,
                window_manager=SimpleNamespace(
                    windows=(
                        SimpleNamespace(scene=global_scene, view_layer=global_view_layer),
                        SimpleNamespace(scene=callback_scene, view_layer=callback_view_layer),
                    )
                ),
            )
        ),
    )
    owner = SceneGenerationOwner(tmp_path / "generations")
    generation = SimpleNamespace(
        number=0,
        usd_path=str(tmp_path / "callback.usda"),
        blender_prim_paths={
            BlenderId("OBJECT", edited.session_uid): BlenderPrimPath(
                edited.name,
                edited.type,
                "/World/Callback/Cube",
                "/World/Callback/Cube/Callback_Cube",
            )
        },
    )
    owner._current = generation  # noqa: SLF001
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "_owners",
        {callback_scene.session_uid: owner},
    )

    matching = _FakeEditableEngine()
    matching._viewport_generation_runtime = object()
    foreign = _FakeEditableEngine()
    foreign._viewport_generation_runtime = object()

    try:
        _track_viewport_engine(matching)
        _track_viewport_engine(foreign)
        results = engine_module._live_interactive_edit_depsgraph_handler(
            callback_scene,
            _FakeDepsgraph([_FakeDepsgraphUpdate(edited)]),
        )
    finally:
        _untrack_viewport_engine(matching)
        _untrack_viewport_engine(foreign)

    assert results is None
    runtime = engine_module.scene_generation_sessions.active_runtime_for_scene(
        callback_scene
    )
    assert runtime is not None
    applied = []

    class Port:
        def update_transforms(self, values):
            applied.extend(values)
            return OvrtxValueUpdateResult(len(values), 1)

        def update_attribute_values(self, values):
            return OvrtxValueUpdateResult(len(values), 1)

    application = runtime.scheduler.apply_pending_view_values(Port())

    assert application.values_written is True
    assert foreign.received_edits == []
    assert applied[0].prim_path == "/World/Callback/Cube"
    assert applied[0].matrix[0][3] == 2.0
    assert global_child.select_get() is True
    assert global_owner.select_get() is False

    final_activations = []
    monkeypatch.setattr(engine_module.scene_generation_sessions, "_runtimes", {})

    class _FinalAdapter:
        last_error = ""

        def __init__(self, _controller):
            pass

        def update_request(self, _request):
            pass

        def activate(self, received_generation, **values):
            final_activations.append((received_generation, values))
            return True

        def deactivate(self):
            return "stopped"

    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "OvrtxGenerationAdapter",
        _FinalAdapter,
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "generation_requires_physics",
        lambda _generation: False,
    )
    engine_module.scene_generation_sessions.activate_for_final_render(
        callback_scene,
        RenderRequest(input_usd_path=generation.usd_path),
        controller=object(),
    )

    assert final_activations[0][0] is generation
    assert final_activations[0][1]["transform_values"][0].matrix[0][3] == 2.0


def test_live_depsgraph_bridge_uses_callback_view_layer_selection(monkeypatch) -> None:
    global_view_layer = SimpleNamespace(objects=SimpleNamespace(active=None))
    callback_view_layer = SimpleNamespace(objects=SimpleNamespace(active=None))

    class _LayerSelectedObject(_FakeBlenderObject):
        def __init__(self, name):
            super().__init__(name)
            self.selections = {}

        def select_get(self, *, view_layer=None):
            return self.selections.get(id(view_layer), False)

        def select_set(self, selected, *, view_layer=None):
            self.selections[id(view_layer)] = bool(selected)

    global_owner = _LayerSelectedObject("Global Owner")
    global_child = _LayerSelectedObject("Global Child")
    global_child[usd_paths.SELECTION_OWNER_OBJECT_PROP] = global_owner.name
    global_child.select_set(True, view_layer=global_view_layer)
    global_view_layer.objects.active = global_child
    edited = _LayerSelectedObject("Callback Cube")
    edited.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/callback.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/Callback/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    edited.select_set(True, view_layer=callback_view_layer)
    callback_view_layer.objects.active = edited
    scene = SimpleNamespace(
        session_uid=12,
        objects=_FakeBlenderObjectCollection((global_owner, global_child, edited))
    )
    submitted = []
    monkeypatch.setattr(
        engine_module,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(
                scene=scene,
                selected_objects=[global_child],
                view_layer=global_view_layer,
            )
        ),
    )
    matching = _FakeEditableEngine()
    matching._viewport_generation_runtime = object()
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "resolve_current_scene_edit_group",
        lambda _scene, edits, _selection: tuple(edits),
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "submit_current_scene_edit_group",
        lambda _scene, edits, _selection: submitted.extend(edits) or (),
    )

    try:
        _track_viewport_engine(matching)
        engine_module._live_interactive_edit_depsgraph_handler(
            scene,
            _FakeDepsgraph(
                [_FakeDepsgraphUpdate(edited)],
                view_layer=callback_view_layer,
            ),
        )
    finally:
        _untrack_viewport_engine(matching)

    assert len(submitted) == 1
    assert submitted[0].usd_prim_path == "/World/Callback/Cube"
    assert matching.received_edits == []
    assert global_child.select_get(view_layer=global_view_layer) is True
    assert global_owner.select_get(view_layer=global_view_layer) is False


def test_live_depsgraph_bridge_keeps_exact_stage_viewport_active(
    monkeypatch,
) -> None:
    exact_stage = _FakeEditableEngine()
    exact_stage._viewport_request = object()
    monkeypatch.setattr(
        engine_module,
        "_EXACT_STAGE_CONFIGURATION",
        {"input_usd_path": "/tmp/exact.usda"},
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "owns_request",
        lambda _scene, _request: False,
    )
    obj = _FakeBlenderObject()
    obj.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )

    try:
        _track_viewport_engine(exact_stage)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            _FakeDepsgraph([_FakeDepsgraphUpdate(obj)]),
            scene=object(),
        )
    finally:
        _untrack_viewport_engine(exact_stage)

    assert len(results) == 1
    assert len(exact_stage.received_edits) == 1


def test_live_depsgraph_bridge_resolves_selection_before_submitting_edits() -> None:
    engine = _FakeEditableEngine()
    owner = _FakeBlenderObject("Orange_00")
    owner.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/physics.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/PhysicsIsland/DynamicBodies/Orange_00",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "sim",
        }
    )
    child = _FakeBlenderObject("Orange_00_mesh")
    child[usd_paths.SELECTION_OWNER_OBJECT_PROP] = "Orange_00"
    child.select_set(True)
    objects = _FakeBlenderObjectCollection([owner, child])
    context = SimpleNamespace(
        selected_objects=[child],
        scene=SimpleNamespace(objects=objects),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=child)),
    )

    try:
        _track_viewport_engine(engine)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            _FakeDepsgraph([_FakeDepsgraphUpdate(child), _FakeDepsgraphUpdate(owner)]),
            context=context,
        )
    finally:
        _untrack_viewport_engine(engine)

    diagnostics = interactive_edit_bridge_diagnostics()
    assert results == []
    assert engine.received_edits == []
    assert diagnostics["last_submitted_edit_count"] == 0
    assert diagnostics["selection_resolution"]["changed"] is True
    assert len(engine.selection_records) == 1
    assert engine.selection_records[0]["selection_resolution"]["changed"] is True
    assert owner.select_get() is True
    assert child.select_get() is False


def test_live_depsgraph_bridge_rejects_multi_selection_with_unmapped_source() -> None:
    engine = _FakeEditableEngine()
    owner = _FakeBlenderObject("Cube")
    owner.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    unmapped = _FakeBlenderObject("Loose")
    owner.select_set(True)
    unmapped.select_set(True)
    objects = _FakeBlenderObjectCollection([owner, unmapped])
    context = SimpleNamespace(
        selected_objects=[owner, unmapped],
        scene=SimpleNamespace(objects=objects),
        view_layer=SimpleNamespace(objects=SimpleNamespace(active=owner)),
    )

    try:
        _track_viewport_engine(engine)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            _FakeDepsgraph([_FakeDepsgraphUpdate(owner)]),
            context=context,
        )
    finally:
        _untrack_viewport_engine(engine)

    diagnostics = interactive_edit_bridge_diagnostics()
    assert results == []
    assert engine.received_edits == []
    assert diagnostics["last_submitted_edit_count"] == 0
    assert diagnostics["selection_resolution"]["group_rejected"] is True
    assert diagnostics["selection_resolution"]["unresolved_reasons"] == ["unmapped_selection_source"]
    assert len(engine.selection_records) == 1
    observation = engine.selection_records[0]["selection_resolution"]
    assert observation["group_rejected"] is True
    assert observation["unresolved_reasons"] == ["unmapped_selection_source"]


def test_live_depsgraph_bridge_can_be_suppressed_for_runtime_owned_writes() -> None:
    engine = _FakeEditableEngine()
    obj = _FakeBlenderObject()
    obj.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )

    try:
        _track_viewport_engine(engine)
        with suppress_interactive_edit_bridge():
            results = submit_depsgraph_interactive_edits_to_active_viewports(
                _FakeDepsgraph([_FakeDepsgraphUpdate(obj)])
            )
    finally:
        _untrack_viewport_engine(engine)

    diagnostics = interactive_edit_bridge_diagnostics()
    assert results == []
    assert engine.received_edits == []
    assert diagnostics["suppressed"] is True
    assert diagnostics["last_active_viewport_engine_count"] == 1
    assert diagnostics["last_submitted_edit_count"] == 0
    assert diagnostics["last_result_count"] == 0
    assert diagnostics["suppress_depth"] == 0


def test_unclassified_depsgraph_updates_do_not_dirty_scene_generation(monkeypatch) -> None:
    marked_scenes: list[object] = []
    submitted_depsgraphs: list[object] = []
    scene = object()
    depsgraph = object()
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "mark_scene_dirty",
        lambda value, updated_ids=None: marked_scenes.append(value),
    )
    monkeypatch.setattr(
        engine_module,
        "submit_depsgraph_interactive_edits_to_active_viewports",
        lambda value, *, context=None, scene=None: submitted_depsgraphs.append(
            (value, context, scene)
        ),
    )

    with suppress_interactive_edit_bridge():
        engine_module._live_interactive_edit_depsgraph_handler(scene, depsgraph)

    assert marked_scenes == []
    assert submitted_depsgraphs == [(depsgraph, None, scene)]

    engine_module._live_interactive_edit_depsgraph_handler(scene, depsgraph)

    assert marked_scenes == []
    assert submitted_depsgraphs == [
        (depsgraph, None, scene),
        (depsgraph, None, scene),
    ]


def test_world_assignment_scene_update_queues_only_identity_changes(monkeypatch) -> None:
    world_id = BlenderId("WORLD", 30)
    unrelated_id = BlenderId("OBJECT", 41)
    marked = []
    scene = object()
    depsgraph = SimpleNamespace(
        updates=[
            SimpleNamespace(
                id=SimpleNamespace(
                    bl_rna=SimpleNamespace(identifier="Scene"),
                    name="Scene",
                )
            )
        ]
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "affected_blender_ids",
        lambda _depsgraph: {unrelated_id},
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "topology_identity_changes",
        lambda _scene: {world_id},
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "is_reconciling",
        lambda _scene: False,
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "mark_scene_dirty",
        lambda received_scene, affected, **kwargs: marked.append(
            (received_scene, set(affected), kwargs)
        ),
    )
    monkeypatch.setattr(
        engine_module,
        "submit_depsgraph_interactive_edits_to_active_viewports",
        lambda *_args, **_kwargs: [],
    )
    viewport = _FakeEditableEngine()
    viewport._viewport_scene = scene

    try:
        _track_viewport_engine(viewport)
        engine_module._live_interactive_edit_depsgraph_handler(scene, depsgraph)
    finally:
        _untrack_viewport_engine(viewport)

    assert marked == [
        (scene, {world_id}, {"defer_world_reconciliation": True})
    ]


def test_export_suppression_excludes_classified_topology_noise(monkeypatch) -> None:
    scene = SimpleNamespace(session_uid=12)
    affected = {BlenderId("MESH", 91)}
    marked = []
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "affected_blender_ids",
        lambda _depsgraph: set(affected),
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "topology_identity_changes",
        lambda _scene: set(),
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "is_reconciling",
        lambda _scene: False,
    )
    monkeypatch.setattr(
        engine_module.scene_generation_sessions,
        "mark_scene_dirty",
        lambda received, identities: marked.append((received, identities)),
    )
    monkeypatch.setattr(
        engine_module,
        "submit_depsgraph_interactive_edits_to_active_viewports",
        lambda *_args, **_kwargs: [],
    )

    with suppress_interactive_edit_bridge():
        engine_module._live_interactive_edit_depsgraph_handler(scene, object())

    assert marked == []


def test_physics_playback_lock_preserves_and_restores_transform_locks() -> None:
    lock = operator_state.PhysicsPlaybackLock()
    obj = _FakeBlenderObject()
    previous_locks = {
        "lock_location": tuple(obj.lock_location),
        "lock_rotation": tuple(obj.lock_rotation),
        "lock_scale": tuple(obj.lock_scale),
    }

    lock.lock_object("/World/TestScene/Cube", obj, generation=7)

    assert tuple(obj.lock_location) == (True, True, True)
    assert tuple(obj.lock_rotation) == (True, True, True)
    assert tuple(obj.lock_scale) == (True, True, True)
    assert lock.diagnostics()["active"] is True
    assert lock.diagnostics()["owning_physics_generation"] == 7
    assert lock.diagnostics()["locked_object_paths"] == ["/World/TestScene/Cube"]

    lock.clear(reason="initial_condition_frame", frame1_cleared=True)

    assert tuple(obj.lock_location) == previous_locks["lock_location"]
    assert tuple(obj.lock_rotation) == previous_locks["lock_rotation"]
    assert tuple(obj.lock_scale) == previous_locks["lock_scale"]
    assert lock.diagnostics()["active"] is False
    assert lock.diagnostics()["frame1_cleared"] is True


def test_physics_playback_lock_rejects_locked_transform_without_preserving_attempt() -> None:
    lock = operator_state.PhysicsPlaybackLock()
    obj = _FakeBlenderObject()
    runtime_matrix = obj.matrix_world
    lock.lock_object("/World/TestScene/Cube", obj, generation=3)
    obj.matrix_world = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (99.0, 99.0, 99.0, 1.0),
    )
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_layer_id="/layers/scene.usda",
            usd_prim_path="/World/TestScene/Cube",
            blender_property_path="matrix_world",
        ),
        value=obj.matrix_world,
    )

    result = lock.reject_edit(edit)

    assert result is not None
    assert result.accepted is False
    assert result.reason == "physics_playback_locked"
    assert result.diagnostics["discarded_attempted_value"] is True
    assert obj.matrix_world == runtime_matrix
    assert lock.diagnostics()["rejected_edit_count"] == 1
    assert lock.diagnostics()["last_rejected_data_authority"] == "view"
    assert lock.diagnostics()["last_rejected_edit_path"] == "/World/TestScene/Cube"


def test_physics_playback_lock_does_not_reject_look_only_edit() -> None:
    lock = operator_state.PhysicsPlaybackLock()
    obj = _FakeBlenderObject()
    lock.lock_object("/World/TestScene/Cube", obj, generation=3)
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_layer_id="/layers/look.usda",
            usd_prim_path="/World/TestScene/Cube",
            usd_attribute="inputs:diffuseColor",
            blender_property_path="diffuse_color",
        ),
        value=(1.0, 0.5, 0.25),
    )

    assert lock.reject_edit(edit) is None
    assert lock.diagnostics()["rejected_edit_count"] == 0


def test_physics_playback_lock_ignores_internal_lock_update_noise() -> None:
    lock = operator_state.PhysicsPlaybackLock()
    obj = _FakeBlenderObject()
    obj.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    lock.lock_object("/World/TestScene/Cube", obj, generation=4)
    edit = InteractiveEdit(
        shape=EditShape.VALUE,

        data_authority=DataAuthority.VIEW,
        **edit_location(
            usd_layer_id="/layers/scene.usda",
            usd_prim_path="/World/TestScene/Cube",
            blender_property_path="matrix_world",
        ),
        value=lock._records["/World/TestScene/Cube"]["edit_value"],
    )

    result = lock.reject_edit(edit)

    assert result is not None
    assert result.accepted is False
    assert result.reason == "physics_playback_lock_internal_update"
    assert lock.diagnostics()["rejected_edit_count"] == 0
    assert lock.diagnostics()["ignored_internal_update_count"] == 1
    assert lock.diagnostics()["reason"] == "active_physics_generation"


def test_live_depsgraph_bridge_rejects_locked_physics_edit_before_submission() -> None:
    lock = operator_state.PhysicsPlaybackLock()
    engine = _FakeLockingEngine(lock)
    obj = _FakeBlenderObject()
    obj.update(
        {
            usd_paths.USD_LAYER_ID_PROP: "/layers/scene.usda",
            usd_paths.USD_PRIM_PATH_PROP: "/World/TestScene/Cube",
            usd_paths.BLENDER_PROPERTY_PATH_PROP: "matrix_world",
            usd_paths.DATA_AUTHORITY_PROP: "view",
        }
    )
    lock.lock_object("/World/TestScene/Cube", obj, generation=4)
    obj.matrix_world = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (42.0, 42.0, 42.0, 1.0),
    )

    try:
        _track_viewport_engine(engine)
        results = submit_depsgraph_interactive_edits_to_active_viewports(
            _FakeDepsgraph([_FakeDepsgraphUpdate(obj)])
        )
    finally:
        _untrack_viewport_engine(engine)

    assert len(results) == 1
    assert results[0].accepted is False
    assert results[0].reason == "physics_playback_locked"
    assert engine.received_edits == []
    assert lock.diagnostics()["rejected_edit_count"] == 1
    bridge_diagnostics = interactive_edit_bridge_diagnostics()
    assert bridge_diagnostics["last_submitted_edit_count"] == 1
    assert bridge_diagnostics["last_result_count"] == 1


def test_live_depsgraph_bridge_registration_is_idempotent(monkeypatch: object) -> None:
    handlers = SimpleNamespace(depsgraph_update_post=[])
    fake_bpy = SimpleNamespace(app=SimpleNamespace(handlers=handlers))
    monkeypatch.setattr(engine_module, "bpy", fake_bpy)

    assert register_interactive_edit_bridge() is True
    assert register_interactive_edit_bridge() is False
    assert len(handlers.depsgraph_update_post) == 1
    assert interactive_edit_bridge_diagnostics()["registered"] is True

    assert unregister_interactive_edit_bridge() is True
    assert unregister_interactive_edit_bridge() is False
    assert handlers.depsgraph_update_post == []
    assert interactive_edit_bridge_diagnostics()["registered"] is False


def test_viewport_draw_geometry_records_letterboxed_texture_rect() -> None:
    geometry = _viewport_draw_geometry(
        result_width=1280,
        result_height=720,
        region_width=2766,
        region_height=1228,
    )

    assert geometry["render_result"] == {"width": 1280, "height": 720}
    assert geometry["region"] == {"width": 2766, "height": 1228}
    assert geometry["texture_draw_rect"]["x"] == pytest.approx(291.4444444444)
    assert geometry["texture_draw_rect"]["y"] == 0.0
    assert geometry["texture_draw_rect"]["width"] == pytest.approx(2183.1111111111)
    assert geometry["texture_draw_rect"]["height"] == 1228.0


def test_viewport_draw_geometry_can_target_camera_frame_rect() -> None:
    geometry = _viewport_draw_geometry(
        result_width=1280,
        result_height=720,
        region_width=2766,
        region_height=1228,
        target_rect={
            "x": 691.5,
            "y": 225.0,
            "width": 1383.0,
            "height": 778.0,
        },
        target_name="camera_frame",
    )

    assert geometry["draw_target"] == "camera_frame"
    assert geometry["draw_target_rect"] == {
        "x": 691.5,
        "y": 225.0,
        "width": 1383.0,
        "height": 778.0,
    }
    assert geometry["texture_draw_rect"]["x"] == 691.5
    assert geometry["texture_draw_rect"]["y"] == pytest.approx(225.03125)
    assert geometry["texture_draw_rect"]["width"] == 1383.0
    assert geometry["texture_draw_rect"]["height"] == pytest.approx(777.9375)


def test_synced_perspective_viewport_request_matches_region_aspect() -> None:
    scene = _viewport_request_scene()
    context = _viewport_request_context()

    request = build_request_from_scene(
        scene,
        context,
        source=BlenderRenderSignalSource.VIEW_UPDATE,
        intent=BlenderRenderIntent.VIEWPORT,
    )
    geometry = _viewport_draw_geometry(
        result_width=request.width,
        result_height=request.height,
        region_width=context.region.width,
        region_height=context.region.height,
    )

    assert (request.width, request.height) == (1383, 614)
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.PERSPECTIVE_USER_VIEW
    assert projection.route == render_requests.OVRTX_SCENE_COMPOSITION_ROUTE
    assert projection.runtime_status == render_requests.RUNTIME_PROJECTION_UNPROVEN
    assert projection.usd_attributes()["focalLength"] == 28.0
    assert projection.usd_attributes()["horizontalAperture"] == 56.0
    assert projection.usd_attributes()["verticalAperture"] == 28.0
    assert geometry["texture_draw_rect"]["x"] == 0.0
    assert geometry["texture_draw_rect"]["y"] == 0.0
    assert geometry["texture_draw_rect"]["width"] == 2766.0
    assert geometry["texture_draw_rect"]["height"] == 1228.0


def test_request_records_explicit_ldr_color_presentation() -> None:
    request = build_request_from_scene(
        _viewport_request_scene(
            presentation=color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
        ),
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
    )

    assert request.color_presentation["active_mode"] == color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH
    assert request.color_presentation["status"] == color_presentation.STATUS_CURRENT
    assert request.color_presentation["frame_format"] == color_presentation.FRAME_FORMAT_RGBA8
    assert request.color_presentation["conversion"] == color_presentation.CONVERSION_PASSTHROUGH
    assert request.color_presentation["view_settings"] == {
        "view_transform": "AgX",
        "look": "Medium High Contrast",
        "exposure": 0.25,
        "gamma": 1.1,
        "display_device": "sRGB",
    }


def test_request_records_default_hdr_color_presentation() -> None:
    request = build_request_from_scene(
        _viewport_request_scene(),
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
    )

    assert request.color_presentation["requested_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert request.color_presentation["active_mode"] == color_presentation.MODE_SCENE_LINEAR_HDR
    assert request.color_presentation["status"] == color_presentation.STATUS_CURRENT
    assert request.color_presentation["unavailable_reason"] == ""
    assert request.color_presentation["frame_format"] == color_presentation.FRAME_FORMAT_RGBA16F
    assert request.color_presentation["frame_color_mode"] == color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR
    assert request.color_presentation["render_var"] == color_presentation.RENDER_VAR_HDR_COLOR


def test_rgba16f_to_float_array_forces_opaque_alpha() -> None:
    rgba = list(_rgba16f_to_float_array(bytes([0, 60, 0, 56, 0, 52, 0, 0])))

    assert rgba == [1.0, 0.5, 0.25, 1.0]

def test_synced_camera_view_keeps_scene_render_resolution() -> None:
    scene = _viewport_request_scene()
    scene.camera = SimpleNamespace(
        data=SimpleNamespace(
            type="PERSP",
            lens=50.0,
            sensor_width=36.0,
            sensor_height=24.0,
            sensor_fit="HORIZONTAL",
            clip_start=0.1,
            clip_end=1000.0,
            shift_x=0.0,
            shift_y=0.0,
            dof=SimpleNamespace(use_dof=False),
        )
    )
    context = _viewport_request_context(view_perspective="CAMERA", scene=scene)

    request = build_request_from_scene(
        scene,
        context,
        source=BlenderRenderSignalSource.VIEW_UPDATE,
        intent=BlenderRenderIntent.VIEWPORT,
    )

    assert (request.width, request.height) == (1280, 720)
    assert request.camera_matrix is not None
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.ACTIVE_CAMERA_VIEW
    assert projection.usd_attributes() == {
        "projection": "perspective",
        "focalLength": 50.0,
        "horizontalAperture": 36.0,
        "verticalAperture": 20.25,
        "horizontalApertureOffset": 0.0,
        "verticalApertureOffset": 0.0,
        "fStop": 0.0,
        "clippingRange": (0.1, 1000.0),
    }


def test_synced_orthographic_user_view_creates_viewport_camera_override() -> None:
    scene = _viewport_request_scene()
    context = _viewport_request_context(view_perspective="ORTHO")

    request = build_request_from_scene(
        scene,
        context,
        source=BlenderRenderSignalSource.VIEW_UPDATE,
        intent=BlenderRenderIntent.VIEWPORT,
    )

    assert request.camera_matrix is not None
    assert (request.width, request.height) == (1383, 614)
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.ORTHOGRAPHIC_USER_VIEW
    assert projection.usd_attributes() == {
        "projection": "orthographic",
        "focalLength": 28.0,
        "horizontalAperture": 20.0,
        "verticalAperture": 10.0,
        "horizontalApertureOffset": 0.0,
        "verticalApertureOffset": 0.0,
        "fStop": 0.0,
    }


def test_synced_orthographic_active_camera_view_uses_scene_camera_projection() -> None:
    scene = _viewport_request_scene()
    scene.camera = SimpleNamespace(
        data=_orthographic_camera_data(
            frame_width=6.0,
            frame_height=3.375,
            lens=45.0,
            clip_start=0.25,
            clip_end=400.0,
        )
    )
    context = _viewport_request_context(view_perspective="CAMERA", scene=scene)

    request = build_request_from_scene(
        scene,
        context,
        source=BlenderRenderSignalSource.VIEW_UPDATE,
        intent=BlenderRenderIntent.VIEWPORT,
    )

    assert (request.width, request.height) == (1280, 720)
    assert request.camera_matrix is not None
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.ACTIVE_CAMERA_VIEW
    assert projection.usd_attributes() == {
        "projection": "orthographic",
        "focalLength": 45.0,
        "horizontalAperture": 60.0,
        "verticalAperture": 33.75,
        "horizontalApertureOffset": 0.0,
        "verticalApertureOffset": 0.0,
        "fStop": 0.0,
        "clippingRange": (0.25, 400.0),
    }


def test_final_render_from_orthographic_scene_camera_carries_projection_overlay() -> None:
    scene = _viewport_request_scene()
    scene.camera = SimpleNamespace(
        data=_orthographic_camera_data(
            frame_width=5.0,
            frame_height=2.8125,
            lens=50.0,
            clip_start=0.1,
            clip_end=1000.0,
        )
    )

    request = build_request_from_scene(
        scene,
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
    )

    assert request.camera_matrix is None
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.ACTIVE_CAMERA_VIEW
    assert projection.usd_attributes() == {
        "projection": "orthographic",
        "focalLength": 50.0,
        "horizontalAperture": 50.0,
        "verticalAperture": 28.125,
        "horizontalApertureOffset": 0.0,
        "verticalApertureOffset": 0.0,
        "fStop": 0.0,
        "clippingRange": (0.1, 1000.0),
    }


def test_final_render_from_shifted_orthographic_scene_camera_carries_aperture_offsets() -> None:
    scene = _viewport_request_scene()
    scene.camera = SimpleNamespace(
        data=_orthographic_camera_data(
            frame_width=6.0,
            frame_height=3.375,
            frame_center_x=0.6,
            frame_center_y=-1.2,
            lens=45.0,
            clip_start=0.25,
            clip_end=400.0,
            shift_x=0.1,
            shift_y=-0.2,
        )
    )

    request = build_request_from_scene(
        scene,
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
    )

    assert request.camera_matrix is None
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.ACTIVE_CAMERA_VIEW
    assert projection.lens_shift == (0.1, -0.2)
    assert projection.usd_attributes()["projection"] == "orthographic"
    assert projection.usd_attributes()["horizontalAperture"] == 60.0
    assert projection.usd_attributes()["verticalAperture"] == 33.75
    assert projection.usd_attributes()["horizontalApertureOffset"] == 6.0
    assert projection.usd_attributes()["verticalApertureOffset"] == -12.0


def test_final_render_orthographic_projection_does_not_require_viewport_sync() -> None:
    scene = _viewport_request_scene(sync_viewport_camera=False)
    scene.camera = SimpleNamespace(
        data=_orthographic_camera_data(
            frame_width=6.0,
            frame_height=3.375,
            lens=45.0,
            clip_start=0.25,
            clip_end=400.0,
        )
    )

    request = build_request_from_scene(
        scene,
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
    )

    assert request.camera_matrix is None
    projection = request.camera_projection
    assert projection is not None
    assert projection.usd_attributes()["projection"] == "orthographic"
    assert projection.usd_attributes()["horizontalAperture"] == 60.0
    assert projection.usd_attributes()["verticalAperture"] == 33.75


def test_unsynced_perspective_viewport_does_not_create_projection_override() -> None:
    scene = _viewport_request_scene(sync_viewport_camera=False)
    context = _viewport_request_context()

    request = build_request_from_scene(
        scene,
        context,
        source=BlenderRenderSignalSource.VIEW_UPDATE,
        intent=BlenderRenderIntent.VIEWPORT,
    )

    assert request.camera_matrix is None
    assert request.camera_projection is None
    assert (request.width, request.height) == (1280, 720)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_command", "worker-b"),
        ("native_client_module", "module-b"),
    ],
)
def test_viewport_runtime_binding_change_reconstructs_client(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    original_client_factory = controller_module._runtime_client_from_request
    original_write_crash_marker = module.session_lifecycle.write_crash_marker
    original_clear_crash_marker = module.session_lifecycle.clear_crash_marker
    try:
        source = tmp_path / "scene.usda"
        source.write_text("#usda 1.0\n", encoding="utf-8")
        clients: list[object] = []

        class _Client:
            def __init__(self, *, worker_command: str, native_client_module: str) -> None:
                self.worker_command = worker_command
                self.native_client_module = native_client_module
                self.started_specs: list[object] = []
                self.startup_diagnostics = {"render_worker": {"status": "running"}}
                self.shutdown_called = False
                clients.append(self)

            def start_session(self, spec: object, simulation_id: str | None = None) -> str:
                self.started_specs.append(spec)
                return f"sim-{len(clients)}"

            def delete_simulation(self, _simulation_id: str) -> str:
                return "stopped"

            def shutdown(self) -> None:
                self.shutdown_called = True

        controller_module._runtime_client_from_request = lambda request: _Client(
            worker_command=request.worker_command,
            native_client_module=request.native_client_module,
        )
        written_markers: list[dict[str, object]] = []
        cleared_markers = 0

        def _write_crash_marker(**kwargs: object) -> dict[str, object]:
            written_markers.append(dict(kwargs))
            return {"marker_active": True}

        def _clear_crash_marker() -> dict[str, object]:
            nonlocal cleared_markers
            cleared_markers += 1
            return {"marker_active": False}

        module.session_lifecycle.write_crash_marker = _write_crash_marker
        module.session_lifecycle.clear_crash_marker = _clear_crash_marker
        request = RenderRequest(
            input_usd_path=str(source),
            sensor_paths=("/Render/A", "/Render/B"),
            selected_sensor_paths=("/Render/A",),
            worker_command="worker-a",
            native_client_module="module-a",
        )
        render_engine = module.OvrtxExampleRenderEngine()
        render_engine._write_viewport_session_outputs = lambda **_kwargs: None

        render_engine._ensure_viewport_session(request)
        first_client = clients[0]
        first_composition = render_engine._ovrtx_scene_composition
        render_engine._ensure_viewport_session(
            replace(request, selected_sensor_paths=("/Render/B",))
        )

        assert len(clients) == 1
        diagnostics = render_engine._ovrtx_session_controller.diagnostics()
        assert render_engine._ovrtx_scene_composition == first_composition
        assert diagnostics["session_reuse"] == {
            "reuse": True,
            "reason": "same_session",
        }
        assert len(first_client.started_specs) == 1
        assert cleared_markers == 1
        assert render_engine._viewport_lifecycle_phase == ""

        changed_request = replace(request, **{field: value})
        render_engine._ensure_viewport_session(changed_request)

        assert len(clients) == 2
        assert first_client.shutdown_called is True
        diagnostics = render_engine._ovrtx_session_controller.diagnostics()
        assert len(clients[1].started_specs) == 1
        assert diagnostics["session_reuse"] == {
            "reuse": False,
            "reason": "runtime_binding_changed",
        }
        assert [event["event"] for event in diagnostics["lifecycle_events"]] == [
            "created",
            "stopped",
            "replaced",
        ]
        assert render_engine._viewport_artifact()["ovrtx_lifecycle_events"] == list(
            diagnostics["lifecycle_events"]
        )
        assert len(written_markers) == 3
        assert cleared_markers == 1
    finally:
        controller_module._runtime_client_from_request = original_client_factory
        module.session_lifecycle.write_crash_marker = original_write_crash_marker
        module.session_lifecycle.clear_crash_marker = original_clear_crash_marker
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_shallow_ovrtx_close_preserves_viewport_runtime_and_diagnostics() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        shutdown_calls: list[str] = []
        scheduler = SimpleNamespace(shutdown=lambda: shutdown_calls.append("scheduler"))
        engine = module.OvrtxExampleRenderEngine()
        recorder = engine._viewport_artifact_recorder
        engine._runtime_scheduler = scheduler
        engine._ovrtx_session_controller = SimpleNamespace(
            shutdown=lambda: shutdown_calls.append("ovrtx")
        )
        engine._viewport_snapshot_count = 7
        engine._render_count = 11

        engine._close_ovrtx_runtime()

        assert shutdown_calls == ["ovrtx"]
        assert engine._runtime_scheduler is scheduler
        assert engine._viewport_artifact_recorder is recorder
        assert engine._viewport_snapshot_count == 7
        assert engine._render_count == 11
        assert engine._ovrtx_session_controller is None
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_full_viewport_end_shuts_down_scheduler_and_ovrtx() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        shutdown_calls: list[str] = []
        engine = module.OvrtxExampleRenderEngine()
        engine._runtime_scheduler = SimpleNamespace(
            shutdown=lambda: shutdown_calls.append("scheduler")
        )
        engine._ovrtx_session_controller = SimpleNamespace(
            shutdown=lambda: shutdown_calls.append("ovrtx")
        )
        engine._write_viewport_session_outputs = lambda **_kwargs: None

        engine._end_viewport_session(module.ViewportSessionEndReason.RECONNECT_REQUESTED)

        assert shutdown_calls == ["scheduler", "ovrtx"]
        assert engine._runtime_scheduler is None
        assert engine._ovrtx_session_controller is None
        assert engine._viewport_session_started_ns == 0
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_native_fallback_ends_scheduler_after_ovrtx_child_loss() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        shutdown_calls: list[str] = []
        writes: list[dict[str, object]] = []
        presentations: list[dict[str, object]] = []
        engine = module.OvrtxExampleRenderEngine()
        engine._runtime_scheduler = SimpleNamespace(
            shutdown=lambda: shutdown_calls.append("scheduler")
        )
        engine._viewport_session_started_ns = 123
        engine._ovrtx_lifecycle_events = [{"event": "replaced"}]
        def record_artifact(**kwargs: object) -> float:
            writes.append(kwargs)
            presentations.append(dict(engine._viewport_presentation))
            return 0.0

        engine._write_viewport_artifact = record_artifact
        engine.update_stats = lambda *_args: None

        engine._enter_native_viewport_fallback({
            "presentation_mode": module.viewport_presentation.NATIVE_VIEWPORT_FALLBACK,
            "fallback_reason": "unsupported_view",
            "fallback_owned_by_addon": True,
            "view_perspective": "CAMERA",
            "changed": True,
        })
        engine._enter_native_viewport_fallback({
            "presentation_mode": module.viewport_presentation.NATIVE_VIEWPORT_FALLBACK,
            "fallback_reason": "unsupported_view",
            "fallback_owned_by_addon": True,
            "view_perspective": "CAMERA",
            "changed": False,
        })

        assert shutdown_calls == ["scheduler"]
        assert engine._runtime_scheduler is None
        assert writes == [{
            "running": False,
            "end_reason": module.ViewportSessionEndReason.NATIVE_FALLBACK,
        }]
        assert presentations[0]["presentation_mode"] == (
            module.viewport_presentation.NATIVE_VIEWPORT_FALLBACK
        )
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_full_viewport_end_writes_artifact_after_ovrtx_child_loss() -> None:
    module, had_bpy, original_bpy = _reload_engine_module_with_fake_bpy()
    try:
        writes: list[dict[str, object]] = []
        engine = module.OvrtxExampleRenderEngine()
        engine._runtime_scheduler = SimpleNamespace(shutdown=lambda: None)
        engine._viewport_session_started_ns = 123
        engine._ovrtx_lifecycle_events = [{"event": "replaced"}]
        engine._write_viewport_artifact = lambda **kwargs: writes.append(kwargs) or 0.0

        engine._end_viewport_session(module.ViewportSessionEndReason.NATIVE_FALLBACK)

        assert writes == [{
            "running": False,
            "end_reason": module.ViewportSessionEndReason.NATIVE_FALLBACK,
        }]
        assert engine._runtime_scheduler is None
        assert engine._viewport_session_started_ns == 0
    finally:
        _restore_engine_module_bpy(had_bpy, original_bpy)


def test_camera_override_layer_text_preserves_nested_usda_blocks(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    text = _camera_override_layer_text(
        source,
        "/World/Camera",
        "/Render/ViewportTexture0",
        width=1383,
        height=614,
        focal_length=28.0,
        horizontal_aperture=63.068403909,
        vertical_aperture=28.0,
    )

    assert f"@{_composition_usda_asset_path(str(source))}@" in text
    assert 'over "Camera"' in text
    assert "float focalLength = 28" in text
    assert "float horizontalAperture = 63.0684039" in text
    # Direct-USD override layers author overs only for prims the source
    # scene already defines (camera and render product); a `def` here would
    # re-type the source prim. Only the RenderVars are add-on-defined.
    assert 'over "ViewportTexture0"' in text
    assert "def RenderProduct" not in text
    assert (
        "rel orderedVars = [</Render/ViewportTexture0/LdrColor>, "
        "</Render/ViewportTexture0/HdrColor>]"
    ) in text
    assert "uniform int2 resolution = (1383, 614)" in text
    assert text.count("def RenderVar") == 2


def test_camera_override_layer_text_merges_shared_root_prim(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    text = _camera_override_layer_text(
        source,
        "/SceneRoot/cameras/Camera",
        "/SceneRoot/Render/ViewportTexture0",
        width=1383,
        height=614,
        focal_length=28.0,
        horizontal_aperture=63.068403909,
        vertical_aperture=28.0,
    )

    assert text.count('over "SceneRoot"') == 1
    assert 'over "Camera"' in text
    assert 'over "ViewportTexture0"' in text
    assert "def RenderProduct" not in text


def test_perspective_viewport_simulation_id_uses_camera_projection_override_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(work_dir))

    request = RenderRequest(
        input_usd_path=str(source),
        width=1383,
        height=614,
        camera_prim_path="/World/Camera",
        camera_matrix=_FakeViewMatrix().inverted(),
        camera_projection=_camera_projection(
            horizontal_aperture=63.068403909,
            render_size=(1383, 614),
        ),
    )

    composition = _compose_request(request)
    fixture_path = Path(composition.composed_scene_path)
    root_text = fixture_path.read_text(encoding="utf-8")
    text = _presentation_text(composition, "viewport_camera_projection")
    diagnostics = ovrtx_scene_composition_diagnostics(composition, request=request)
    projection_record = diagnostics["presentation_layers"][0]

    assert composition.source_scene_path == str(source)
    assert composition.pass_through is False
    assert composition.presentation_layers[0]["source"] == "viewport_camera_projection"
    assert projection_record["projection_route"] == render_requests.OVRTX_SCENE_COMPOSITION_ROUTE
    assert projection_record["runtime_write_status"] == render_requests.RUNTIME_PROJECTION_UNPROVEN
    assert projection_record["projection_attributes"] == [
        "projection",
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "fStop",
    ]
    assert fixture_path.parent == work_dir
    assert f"@{_composition_usda_asset_path(str(source.resolve()))}@" in root_text
    assert 'over "World"' in text
    assert 'over "Camera"' in text
    assert 'token projection = "perspective"' in text
    assert "float focalLength = 28" in text
    assert "float horizontalAperture = 63.0684039" in text
    assert "float verticalAperture = 28" in text

    assert diagnostics["source_scene_path"] == str(source)
    assert diagnostics["composed_scene_path"] == str(fixture_path)
    assert diagnostics["presentation_layer_count"] == 1
    assert diagnostics["presentation_sources"] == ["viewport_camera_projection"]
    assert diagnostics["pass_through"] is False


def test_structured_camera_projection_overlay_authors_classified_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))

    request = RenderRequest(
        input_usd_path=str(source),
        width=1280,
        height=720,
        camera_prim_path="/World/Camera",
        sensor_paths=("/Render/ViewportTexture0",),
        selected_sensor_paths=("/Render/ViewportTexture0",),
        camera_projection=render_requests.CameraProjectionState(
            source=render_requests.ACTIVE_CAMERA_VIEW,
            focal_length=50.0,
            horizontal_aperture=36.0,
            vertical_aperture=20.25,
            horizontal_aperture_offset=1.25,
            vertical_aperture_offset=-0.5,
            clipping_range=(0.05, 500.0),
            f_stop=200.0,
            focus_distance=7.5,
            viewport_region=(2766, 1228),
            render_size=(1280, 720),
        ),
    )

    composition = _compose_request(request)
    text = _presentation_text(composition, "viewport_camera_projection")
    projection_record = ovrtx_scene_composition_diagnostics(
        composition,
        request=request,
    )["presentation_layers"][0]

    assert "float focalLength = 50" in text
    assert 'token projection = "perspective"' in text
    assert "float horizontalAperture = 36" in text
    assert "float verticalAperture = 20.25" in text
    assert "float horizontalApertureOffset = 1.25" in text
    assert "float verticalApertureOffset = -0.5" in text
    assert "float2 clippingRange = (0.05, 500)" in text
    assert "float fStop = 200" in text
    assert "float focusDistance = 7.5" in text
    assert projection_record["projection_attributes"] == [
        "projection",
        "focalLength",
        "horizontalAperture",
        "verticalAperture",
        "horizontalApertureOffset",
        "verticalApertureOffset",
        "fStop",
        "clippingRange",
        "focusDistance",
    ]


def test_ovrtx_scene_composition_authors_resolution_without_camera_projection() -> None:
    request = RenderRequest(
        input_usd_path="/fixtures/stage.usda",
        width=1280,
        height=720,
        camera_prim_path="/World/Camera",
        camera_projection=None,
    )

    composition = _compose_request(request)
    diagnostics = ovrtx_scene_composition_diagnostics(composition, request=request)

    assert composition.source_scene_path == "/fixtures/stage.usda"
    assert composition.composed_scene_path != "/fixtures/stage.usda"
    assert composition.pass_through is False
    assert composition.presentation_layers[0]["source"] == "viewport_camera_projection"
    assert diagnostics["presentation_layers"][0]["projection_attributes"] == []
    assert diagnostics["presentation_layer_count"] == 1


def test_ovrtx_scene_composition_includes_materialx_openpbr_presentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(work_dir))

    material_layer_body = "\n".join(
        [
            'def Scope "OVRTX_Materials"',
            "{",
            '    def Material "Paint"',
            "    {",
            "    }",
            "}",
            "",
            'over "World"',
            "{",
            '    over "Geom"',
            "    {",
            "        rel material:binding = </OVRTX_Materials/Paint>",
            "    }",
            "}",
            "",
        ]
    )
    request = RenderRequest(
        input_usd_path=str(source),
        width=1280,
        height=720,
        camera_prim_path="/World/Camera",
        camera_projection=None,
        material_scene_layer=_material_scene_layer(
            digest="material-digest",
            layer_body=material_layer_body,
            binding_targets=("/World/Geom",),
        ),
    )

    composition = _compose_request(request)
    fixture_path = Path(composition.composed_scene_path)
    root_text = fixture_path.read_text(encoding="utf-8")
    text = _presentation_text(composition, "materialx_openpbr")
    diagnostics = ovrtx_scene_composition_diagnostics(composition, request=request)
    material_record = diagnostics["presentation_layers"][0]

    assert composition.pass_through is False
    assert fixture_path.parent == work_dir
    assert f"@{_composition_usda_asset_path(str(source.resolve()))}@" in root_text
    assert 'def Scope "OVRTX_Materials"' in text
    assert "rel material:binding = </OVRTX_Materials/Paint>" in text
    assert composition.presentation_layers[0]["source"] == "materialx_openpbr"
    assert composition.presentation_layers[0]["generated"] is True
    assert material_record["digest"] == "material-digest"

    assert diagnostics["presentation_sources"] == [
        "materialx_openpbr",
        "viewport_camera_projection",
    ]


def test_ovrtx_scene_composition_rekeys_when_material_presentation_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))

    request = RenderRequest(
        input_usd_path=str(source),
        width=1280,
        height=720,
        material_scene_layer=_material_scene_layer(
            digest="material-a",
            layer_body='def Scope "OVRTX_Materials"\n{\n}\n',
        ),
    )
    changed_request = RenderRequest(
        input_usd_path=str(source),
        width=1280,
        height=720,
        material_scene_layer=_material_scene_layer(
            digest="material-b",
            layer_body='def Scope "OVRTX_MaterialsB"\n{\n}\n',
        ),
    )

    first = _compose_request(request)
    second = _compose_request(changed_request)

    assert first.digest != second.digest
    assert first.composed_scene_path != second.composed_scene_path


def test_final_render_preparation_uses_material_presentation_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "stage.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_VIEWPORT_WORK_DIR", str(tmp_path / "work"))
    request = RenderRequest(
        input_usd_path=str(source),
        material_scene_layer=_material_scene_layer(
            digest="material-a",
            layer_body="\n".join(
                [
                    'def Scope "OVRTX_Materials"',
                    "{",
                    '    def Material "Paint"',
                    "    {",
                    "    }",
                    "}",
                    "",
                    'over "World"',
                    "{",
                    '    over "Geom"',
                    "    {",
                    "        rel material:binding = </OVRTX_Materials/Paint>",
                    "    }",
                    "}",
                    "",
                ]
            ),
            binding_targets=("/World/Geom",),
        ),
    )

    spec = ovrtx_session.build_spec(request)
    composition = spec.ovrtx_scene_composition

    assert composition.composed_scene_path != str(source)
    assert composition.pass_through is False
    assert composition.presentation_layers[0]["source"] == "materialx_openpbr"


def test_engine_signal_id_uses_stable_blender_pointer() -> None:
    class EngineWrapper:
        def as_pointer(self) -> int:
            return 42

    assert engine_module._engine_signal_id(EngineWrapper()) == engine_module._engine_signal_id(
        EngineWrapper()
    )


def test_build_request_from_scene_uses_bundled_runtime_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    addon_root = tmp_path / "addon"
    worker = addon_root / "bin" / "ovrtx-bridge-server"
    package_root = addon_root / "runtime" / "ovrtx-bridge-server"
    native = addon_root / "native"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    package_root.mkdir(parents=True)
    native.mkdir()
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_WORKER_COMMAND", raising=False)
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", raising=False)
    monkeypatch.setattr(bundled_runtime, "addon_root", lambda: addon_root)
    monkeypatch.setattr(bundled_runtime, "current_platform_id", lambda: "linux-x64")

    request = build_request_from_scene(
        _viewport_request_scene(),
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
    )

    assert request.worker_command
    assert str(native) in sys.path


def test_image_artifact_writes_current_result(tmp_path: Path) -> None:
    render_result = RenderResult(
        width=1,
        height=1,
        rgba8=bytes((255, 0, 0, 255)),
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=0,
    )
    path = tmp_path / "image.png"

    diagnostics = _write_image(str(path), render_result)

    assert diagnostics["path"] == str(path)
    assert diagnostics["width"] == 1
    assert diagnostics["height"] == 1
    assert diagnostics["size_bytes"] > 0
    assert path.is_file()


def test_image_artifact_flips_gpu_rows_to_png_order(monkeypatch, tmp_path: Path) -> None:
    captured = {}
    monkeypatch.setattr(
        engine_module,
        "write_rgba_png",
        lambda _path, _width, _height, rgba8: captured.update(rgba8=rgba8) or {},
    )
    bottom = bytes((255, 0, 0, 255))
    top = bytes((0, 0, 255, 255))

    _write_image(
        str(tmp_path / "image.png"),
        RenderResult(
            width=1,
            height=2,
            rgba8=bottom + top,
            completed_samples=1,
            session_completed_samples=1,
            simulation_time_ns=0,
        ),
    )

    assert captured["rgba8"] == top + bottom


def test_viewport_texture_helper_reuses_cached_texture_for_same_frame() -> None:
    _FakeTexture.created = []
    render_result = _texture_result()
    first = _upload_viewport_texture(
        _FakeGpu(_FakeTexture),
        render_result,
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    second = _upload_viewport_texture(
        _FakeGpu(_FakeTexture),
        render_result,
        cached_texture=first.texture,
        cached_texture_size=first.texture_size,
        cached_texture_snapshot_index=1,
        snapshot_index=1,
        accepts_rgba8=first.accepts_rgba8,
    )

    assert second.texture is first.texture
    assert second.diagnostics["texture_path"] == "reuse"
    assert second.diagnostics["texture_cache_hit"] is True
    assert second.diagnostics["texture_upload_bytes"] == 0
    assert len(_FakeTexture.created) == 1


def test_viewport_texture_helper_updates_cached_same_size_texture() -> None:
    _FakeTexture.created = []
    render_result = _texture_result()
    first = _upload_viewport_texture(
        _FakeGpu(_FakeTexture),
        render_result,
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    updated = _upload_viewport_texture(
        _FakeGpu(_FakeTexture),
        render_result,
        cached_texture=first.texture,
        cached_texture_size=first.texture_size,
        cached_texture_snapshot_index=1,
        snapshot_index=2,
        accepts_rgba8=first.accepts_rgba8,
    )

    assert updated.texture is first.texture
    assert updated.diagnostics["texture_path"] == "update"
    assert updated.diagnostics["texture_cache_hit"] is True
    assert updated.diagnostics["gpu_texture_update_available"] is True
    assert updated.diagnostics["texture_upload_bytes"] == len(render_result.rgba8)
    assert updated.diagnostics["texture_update_ms"] >= 0.0
    assert first.texture.updates[0][0].kind == "UBYTE"
    assert first.texture.updates[0][1] == "UBYTE"
    assert len(_FakeTexture.created) == 1


def test_viewport_texture_helper_constructs_new_texture_without_update_api() -> None:
    _FakeTextureWithoutUpdate.created = []
    render_result = _texture_result()
    first = _upload_viewport_texture(
        _FakeGpu(_FakeTextureWithoutUpdate),
        render_result,
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    second = _upload_viewport_texture(
        _FakeGpu(_FakeTextureWithoutUpdate),
        render_result,
        cached_texture=first.texture,
        cached_texture_size=first.texture_size,
        cached_texture_snapshot_index=1,
        snapshot_index=2,
        accepts_rgba8=first.accepts_rgba8,
    )

    assert second.texture is not first.texture
    assert second.diagnostics["texture_path"] == "new_texture"
    assert second.diagnostics["texture_cache_hit"] is False
    assert second.diagnostics["gpu_texture_update_available"] is False
    assert second.diagnostics["texture_upload_bytes"] == len(render_result.rgba8)
    assert len(_FakeTextureWithoutUpdate.created) == 2


def test_viewport_texture_helper_falls_back_when_update_fails() -> None:
    _FakeTextureWithFailingUpdate.created = []
    render_result = _texture_result()
    first = _upload_viewport_texture(
        _FakeGpu(_FakeTextureWithFailingUpdate),
        render_result,
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    second = _upload_viewport_texture(
        _FakeGpu(_FakeTextureWithFailingUpdate),
        render_result,
        cached_texture=first.texture,
        cached_texture_size=first.texture_size,
        cached_texture_snapshot_index=1,
        snapshot_index=2,
        accepts_rgba8=first.accepts_rgba8,
    )

    assert second.texture is not first.texture
    assert second.diagnostics["texture_path"] == "new_texture"
    assert second.diagnostics["texture_cache_hit"] is False
    assert second.diagnostics["gpu_texture_update_available"] is True
    assert second.diagnostics["texture_update_failed"] is True
    assert "rejected payload" in second.diagnostics["texture_update_error"]


def test_viewport_texture_helper_records_float_fallback_path() -> None:
    _FakeTextureRejectingUbyte.created = []
    render_result = _texture_result()

    upload = _upload_viewport_texture(
        _FakeGpu(_FakeTextureRejectingUbyte),
        render_result,
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    assert upload.diagnostics["texture_path"] == "fallback_float"
    assert upload.diagnostics["texture_cache_hit"] is False
    assert upload.diagnostics["texture_upload_bytes"] == len(render_result.rgba8) * 4
    assert upload.accepts_rgba8 is False
    assert upload.texture.data.kind == "FLOAT"
    assert all(
        upload.diagnostics[key] >= 0.0
        for key in viewport_profile.TEXTURE_TIMING_FIELDS
    )


def test_viewport_texture_helper_uploads_scene_linear_rgba16f() -> None:
    _FakeTexture.created = []
    render_result = RenderResult(
        width=1,
        height=1,
        rgba8=b"",
        completed_samples=1,
        session_completed_samples=1,
        simulation_time_ns=0,
        linear_rgba16f=b"\x00<\x00<\x00<\x00<",
        frame_format=color_presentation.FRAME_FORMAT_RGBA16F,
        frame_color_mode=color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
        render_var=color_presentation.RENDER_VAR_HDR_COLOR,
    )

    upload = _upload_viewport_texture(
        _FakeGpu(_FakeTexture),
        render_result,
        cached_texture=None,
        cached_texture_size=None,
        cached_texture_snapshot_index=0,
        snapshot_index=1,
        accepts_rgba8=True,
    )

    assert upload.texture_size == (1, 1, color_presentation.FRAME_FORMAT_RGBA16F)
    assert upload.texture.format == "RGBA16F"
    assert upload.texture.data.kind == "FLOAT"
    assert upload.diagnostics["texture_path"] == "scene_linear_float"
    assert upload.diagnostics["texture_upload_bytes"] == 16


def test_viewport_profile_records_native_timing_ms_fields() -> None:
    profile = viewport_profile.new()
    record = {
        "rendered": True,
        "camera_changed": True,
        "snapshot_changed": True,
        "timings_ms": {phase: 0.0 for phase in viewport_profile.TIMING_PHASES},
        "native_timings": {
            "render_result": {
                "total_native_ms": 25.0,
                "read_world_state_ms": 20.0,
                "read_success_world_state_ms": 12.0,
                "read_empty_ok_world_state_ms": 8.0,
                "ldr_wait_ms": 22.0,
                "payload_extract_ms": 0.4,
                "step_count": 1,
                "payload_bytes": 3686400,
            },
            "value_update": {
                "total_native_ms": 1.0,
                "write_world_state_ms": 0.8,
            },
        },
    }

    viewport_profile.record(profile, record)
    summary = viewport_profile.summary(profile)

    render_stats = summary["native_timing_stats"]["render_result"]
    assert render_stats["total_native_ms"]["count"] == 1
    assert render_stats["read_world_state_ms"]["max_ms"] == 20.0
    assert render_stats["read_success_world_state_ms"]["max_ms"] == 12.0
    assert render_stats["read_empty_ok_world_state_ms"]["max_ms"] == 8.0
    assert render_stats["ldr_wait_ms"]["max_ms"] == 22.0
    assert render_stats["payload_extract_ms"]["max_ms"] == 0.4
    assert "step_count" not in render_stats
    assert "payload_bytes" not in render_stats
    assert summary["native_timing_stats"]["value_update"]["write_world_state_ms"]["mean_ms"] == 0.8


def test_viewport_profile_records_texture_upload_fields() -> None:
    profile = viewport_profile.new()
    record = {
        "rendered": True,
        "camera_changed": False,
        "snapshot_changed": False,
        "timings_ms": {phase: 0.0 for phase in viewport_profile.TIMING_PHASES},
        "texture_path": "update",
        "texture_cache_hit": True,
        "gpu_texture_update_available": True,
        "texture_upload_bytes": 16,
        "texture_update_ms": 0.25,
        "texture_convert_ms": 0.05,
        "texture_buffer_ms": 0.06,
        "texture_create_ms": 0.07,
        "texture_filter_ms": 0.08,
    }

    viewport_profile.record(profile, record)
    summary = viewport_profile.summary(profile)

    assert summary["texture_path_counts"] == {"update": 1}
    assert summary["texture_cache_hit_count"] == 1
    assert summary["gpu_texture_update_available_count"] == 1
    assert summary["texture_upload_bytes"] == 16
    assert summary["recent_texture_update_stats_ms"]["max_ms"] == 0.25
    assert summary["texture_timing_stats"]["texture_convert_ms"]["max_ms"] == 0.05
    assert summary["texture_timing_stats"]["texture_buffer_ms"]["max_ms"] == 0.06
    assert summary["texture_timing_stats"]["texture_create_ms"]["max_ms"] == 0.07
    assert summary["texture_timing_stats"]["texture_filter_ms"]["max_ms"] == 0.08
    assert summary["recent_draws"][0]["texture_path"] == "update"
    assert summary["recent_draws"][0]["texture_upload_bytes"] == 16


def test_viewport_profile_uses_callback_timing_names() -> None:
    profile = viewport_profile.new()
    timings = {phase: 0.0 for phase in viewport_profile.TIMING_PHASES}
    timings["viewport_callback_ms"] = 12.5
    timings["viewport_texture_draw_ms"] = 1.25
    record = {
        "rendered": True,
        "camera_changed": False,
        "snapshot_changed": False,
        "timings_ms": timings,
    }

    viewport_profile.record(profile, record)
    summary = viewport_profile.summary(profile)

    assert "viewport_callback_ms" in summary["phase_stats"]
    assert "viewport_texture_draw_ms" in summary["phase_stats"]
    assert "view_draw_ms" not in summary["phase_stats"]
    assert "draw_ms" not in summary["phase_stats"]
    assert summary["phase_stats"]["viewport_callback_ms"]["max_ms"] == 12.5
    assert summary["recent_draws"][0]["timings_ms"]["viewport_texture_draw_ms"] == 1.25
    assert "draw_ms" not in summary["recent_draws"][0]["timings_ms"]
    assert summary["recent_phase_stats"]["viewport_texture_draw_ms"]["max_ms"] == 1.25


def test_viewport_profile_records_draw_reasons_and_interval_decomposition() -> None:
    profile = viewport_profile.new()
    base_ns = 1_000_000_000

    for offset_ms, record in (
        (
            0,
            {
                "rendered": True,
                "camera_changed": True,
                "snapshot_changed": True,
                "requested_additional_samples": 1,
                "completed_samples": 1,
                "max_samples": 8,
                "timings_ms": _timings(render_ms=40.0),
            },
        ),
        (
            60,
            {
                "rendered": False,
                "camera_changed": False,
                "snapshot_changed": False,
                "requested_additional_samples": 0,
                "completed_samples": 8,
                "max_samples": 8,
                "timings_ms": _timings(viewport_texture_draw_ms=1.0),
            },
        ),
        (
            120,
            {
                "rendered": True,
                "camera_changed": False,
                "snapshot_changed": False,
                "requested_additional_samples": 1,
                "completed_samples": 2,
                "max_samples": 8,
                "timings_ms": _timings(
                        request_ms=1.0,
                        ensure_session_ms=2.0,
                        acquire_result_ms=55.0,
                        render_ms=50.0,
                    result_convert_ms=5.0,
                    texture_upload_ms=4.0,
                    viewport_texture_draw_ms=3.0,
                    viewport_callback_ms=999.0,
                ),
            },
        ),
        (
            180,
            {
                "rendered": False,
                "camera_changed": False,
                "snapshot_changed": False,
                "requested_additional_samples": 0,
                "completed_samples": 8,
                "max_samples": 8,
                "timings_ms": _timings(viewport_texture_draw_ms=1.0),
            },
        ),
    ):
        timestamp_ns = base_ns + offset_ms * 1_000_000
        record["started_at_ns"] = timestamp_ns
        record["ended_at_ns"] = timestamp_ns
        record["started_monotonic_ns"] = timestamp_ns
        record["ended_monotonic_ns"] = timestamp_ns
        viewport_profile.record(profile, record)

    summary = viewport_profile.summary(profile)
    first, skipped, second, third = summary["recent_draws"]

    assert first["draw_outcome"] == "rendered"
    assert first["draw_phase"] == "startup_warmup"
    assert first["render_reason"] == "camera_changed"
    assert first["reuse_reason"] is None
    assert first["time_since_previous_draw_ms"] is None
    assert first["time_since_previous_render_ms"] is None
    assert first["render_interval_unaccounted_ms"] is None

    assert skipped["draw_outcome"] == "reused"
    assert skipped["draw_phase"] == "steady"
    assert skipped["render_reason"] is None
    assert skipped["reuse_reason"] == "reached_max_samples"
    assert skipped["time_since_previous_draw_ms"] == 60.0
    assert skipped["time_since_previous_render_ms"] == 60.0
    assert skipped["render_interval_measured_work_ms_by_phase"] is None
    assert skipped["render_interval_measured_work_ms"] is None
    assert skipped["render_interval_unaccounted_ms"] is None

    assert second["draw_outcome"] == "rendered"
    assert second["draw_phase"] == "steady"
    assert second["render_reason"] == "refinement_samples"
    assert second["time_since_previous_draw_ms"] == 60.0
    assert second["time_since_previous_render_ms"] == 120.0
    assert (
        second["render_interval_measured_work_ms_by_phase"]["viewport_texture_draw_ms"]
        == 4.0
    )
    assert second["render_interval_measured_work_ms"] == 66.0
    assert second["render_interval_unaccounted_ms"] == 54.0

    assert third["draw_outcome"] == "reused"
    assert third["draw_phase"] == "steady"
    assert third["render_reason"] is None
    assert third["reuse_reason"] == "reached_max_samples"
    assert third["time_since_previous_draw_ms"] == 60.0
    assert third["time_since_previous_render_ms"] == 60.0
    assert third["render_interval_measured_work_ms"] is None
    assert third["render_interval_unaccounted_ms"] is None

    assert summary["draw_outcome_counts"] == {"rendered": 2, "reused": 2}
    assert summary["draw_phase_counts"] == {"startup_warmup": 1, "steady": 3}
    assert summary["recent_draw_phase_counts"] == {"startup_warmup": 1, "steady": 3}
    assert summary["steady_recent_summary"]["draw_count"] == 3
    assert summary["steady_recent_summary"]["render_count"] == 1
    assert summary["timeline_reset_recent_summary"]["draw_count"] == 0
    assert summary["render_reason_counts"] == {"camera_changed": 1, "refinement_samples": 1}
    assert summary["reuse_reason_counts"] == {"reached_max_samples": 2}
    assert summary["recent_render_interval_stats_ms"]["total_ms"] == 120.0
    assert summary["recent_time_since_previous_draw_stats_ms"]["count"] == 3
    assert summary["recent_time_since_previous_draw_stats_ms"]["total_ms"] == 180.0
    assert summary["recent_time_since_previous_render_stats_ms"]["count"] == 3
    assert summary["recent_time_since_previous_render_stats_ms"]["total_ms"] == 240.0
    assert summary["recent_render_interval_measured_work_stats_ms"]["max_ms"] == 66.0
    assert (
        summary["recent_render_interval_measured_work_phase_stats_ms"]["viewport_texture_draw_ms"][
            "max_ms"
        ]
        == 4.0
    )
    assert summary["recent_render_interval_unaccounted_stats_ms"]["max_ms"] == 54.0


def test_viewport_profile_decomposes_callback_wait_from_unaccounted_interval() -> None:
    profile = viewport_profile.new()
    base_ns = 1_000_000_000

    for start_ms, end_ms, record in (
        (
            0,
            40,
            {
                "rendered": True,
                "camera_changed": True,
                "snapshot_changed": True,
                "requested_additional_samples": 1,
                "completed_samples": 1,
                "max_samples": 8,
                "timings_ms": _timings(render_ms=40.0),
            },
        ),
        (
            80,
            90,
            {
                "rendered": False,
                "camera_changed": False,
                "snapshot_changed": False,
                "requested_additional_samples": 0,
                "completed_samples": 8,
                "max_samples": 8,
                "timings_ms": _timings(viewport_texture_draw_ms=1.0),
            },
        ),
        (
            140,
            210,
            {
                "rendered": True,
                "camera_changed": False,
                "snapshot_changed": False,
                "requested_additional_samples": 1,
                "completed_samples": 2,
                "max_samples": 8,
                "timings_ms": _timings(
                        request_ms=1.0,
                        ensure_session_ms=2.0,
                        acquire_result_ms=55.0,
                        render_ms=50.0,
                    result_convert_ms=5.0,
                    texture_upload_ms=4.0,
                    viewport_texture_draw_ms=3.0,
                    viewport_callback_ms=70.0,
                ),
            },
        ),
    ):
        record["started_at_ns"] = base_ns + start_ms * 1_000_000
        record["ended_at_ns"] = base_ns + end_ms * 1_000_000
        record["started_monotonic_ns"] = base_ns + start_ms * 1_000_000
        record["ended_monotonic_ns"] = base_ns + end_ms * 1_000_000
        viewport_profile.record(profile, record)

    summary = viewport_profile.summary(profile)
    _, skipped, second = summary["recent_draws"]

    assert skipped["callback_wait_since_previous_draw_ms"] == 40.0
    assert skipped["time_since_previous_draw_start_ms"] == 80.0
    assert second["callback_wait_since_previous_draw_ms"] == 50.0
    assert second["time_since_previous_draw_start_ms"] == 60.0
    assert second["time_since_previous_render_ms"] == 170.0
    assert second["render_interval_measured_work_ms"] == 66.0
    assert second["render_interval_callback_wait_ms"] == 90.0
    assert second["render_interval_unaccounted_ms"] == 104.0
    assert second["render_interval_unaccounted_after_callback_wait_ms"] == 14.0

    assert summary["recent_callback_wait_since_previous_draw_stats_ms"]["total_ms"] == 90.0
    assert summary["recent_render_interval_callback_wait_stats_ms"]["max_ms"] == 90.0
    assert (
        summary["recent_render_interval_unaccounted_after_callback_wait_stats_ms"]["max_ms"]
        == 14.0
    )


def test_viewport_profile_classifies_timeline_resets() -> None:
    profile = viewport_profile.new()
    base_ns = 1_000_000_000

    for offset_ms, timeline_reset in ((0, False), (60, True), (120, False)):
        timestamp_ns = base_ns + offset_ms * 1_000_000
        viewport_profile.record(
            profile,
            {
                "started_at_ns": timestamp_ns,
                "ended_at_ns": timestamp_ns,
                "started_monotonic_ns": timestamp_ns,
                "ended_monotonic_ns": timestamp_ns,
                "rendered": True,
                "camera_changed": False,
                "snapshot_changed": False,
                "timeline_reset": timeline_reset,
                "requested_additional_samples": 1,
                "completed_samples": 1,
                "max_samples": 8,
                "timings_ms": _timings(render_ms=20.0),
            },
        )

    summary = viewport_profile.summary(profile)
    first, reset, after_reset = summary["recent_draws"]

    assert first["draw_phase"] == "startup_warmup"
    assert reset["draw_phase"] == "timeline_reset"
    assert after_reset["draw_phase"] == "startup_warmup"
    assert summary["draw_phase_counts"] == {
        "startup_warmup": 2,
        "timeline_reset": 1,
    }
    assert summary["timeline_reset_recent_summary"]["draw_count"] == 1
    assert summary["timeline_reset_recent_summary"]["render_count"] == 1


def test_viewport_profile_clamps_negative_unaccounted_render_interval() -> None:
    profile = viewport_profile.new()
    base_ns = 1_000_000_000

    for offset_ms, render_ms in ((0, 0.0), (20, 40.0)):
        timestamp_ns = base_ns + offset_ms * 1_000_000
        viewport_profile.record(
            profile,
            {
                "started_at_ns": timestamp_ns,
                "ended_at_ns": timestamp_ns,
                "started_monotonic_ns": timestamp_ns,
                "ended_monotonic_ns": timestamp_ns,
                "rendered": True,
                "camera_changed": False,
                "snapshot_changed": False,
                "requested_additional_samples": 1,
                "completed_samples": 1,
                "max_samples": 8,
                    "timings_ms": _timings(
                        acquire_result_ms=render_ms,
                        render_ms=render_ms,
                    ),
            },
        )

    second = viewport_profile.summary(profile)["recent_draws"][1]

    assert second["time_since_previous_render_ms"] == 20.0
    assert second["render_interval_measured_work_ms"] == 40.0
    assert second["render_interval_unaccounted_ms"] == 0.0


def test_viewport_profile_reports_render_cadence_from_profile_window() -> None:
    profile = viewport_profile.new()
    base_ns = 1_000_000_000

    for rendered, offset_ms in (
        (True, 0),
        (True, 100),
        (False, 200),
        (True, 300),
        (True, 600),
    ):
        timestamp_ns = base_ns + offset_ms * 1_000_000
        viewport_profile.record(
            profile,
            {
                "started_at_ns": timestamp_ns,
                "ended_at_ns": timestamp_ns,
                "started_monotonic_ns": timestamp_ns,
                "ended_monotonic_ns": timestamp_ns,
                "rendered": rendered,
                "camera_changed": False,
                "snapshot_changed": False,
                "timings_ms": {phase: 0.0 for phase in viewport_profile.TIMING_PHASES},
            },
        )

    summary = viewport_profile.summary(profile)
    interval_stats = summary["recent_render_interval_stats_ms"]

    assert summary["draw_count"] == 5
    assert summary["render_count"] == 4
    assert summary["reuse_count"] == 1
    assert summary["profile_window_started_at_ns"] == base_ns
    assert summary["profile_window_ended_at_ns"] == base_ns + 600_000_000
    assert summary["profile_window_ms"] == 600.0
    assert abs(summary["render_fps"] - (4 / 0.6)) < 0.000001
    assert summary["render_cadence_ms"] == 150.0
    assert interval_stats["count"] == 3
    assert interval_stats["total_ms"] == 600.0
    assert interval_stats["mean_ms"] == 200.0
    assert interval_stats["min_ms"] == 100.0
    assert interval_stats["p50_ms"] == 200.0
    assert interval_stats["p95_ms"] == 300.0
    assert interval_stats["max_ms"] == 300.0


def test_viewport_profile_cadence_uses_monotonic_timestamps() -> None:
    profile = viewport_profile.new()
    wall_base_ns = 10_000_000_000
    monotonic_base_ns = 1_000_000_000

    for wall_offset_ms, monotonic_offset_ms in ((0, 0), (-500, 100), (-250, 300)):
        viewport_profile.record(
            profile,
            {
                "started_at_ns": wall_base_ns + wall_offset_ms * 1_000_000,
                "ended_at_ns": wall_base_ns + wall_offset_ms * 1_000_000,
                "started_monotonic_ns": monotonic_base_ns + monotonic_offset_ms * 1_000_000,
                "ended_monotonic_ns": monotonic_base_ns + monotonic_offset_ms * 1_000_000,
                "rendered": True,
                "camera_changed": False,
                "snapshot_changed": False,
                "timings_ms": {phase: 0.0 for phase in viewport_profile.TIMING_PHASES},
            },
        )

    summary = viewport_profile.summary(profile)
    interval_stats = summary["recent_render_interval_stats_ms"]

    assert summary["profile_window_started_at_ns"] == wall_base_ns - 500_000_000
    assert summary["profile_window_ended_at_ns"] == wall_base_ns
    assert summary["profile_window_ms"] == 300.0
    assert summary["render_fps"] == 10.0
    assert interval_stats["count"] == 2
    assert interval_stats["p50_ms"] == 150.0
    assert interval_stats["p95_ms"] == 200.0
