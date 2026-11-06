# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "addon"))

from ovrtx_blender_example import (  # noqa: E402
    bundled_runtime,
    color_presentation,
    light_scene_layer,
    materialx_openpbr_conversion,
    ovrtx_session,
    properties,
    render_requests,
)
from ovrtx_blender_example.blender_signal_translation import (  # noqa: E402
    BlenderSignalTranslationError,
    RenderRequestTranslator,
)
from ovrtx_blender_example.blender_signals import (  # noqa: E402
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
)


def test_viewport_dimension_clamp_preserves_engine_limit() -> None:
    assert render_requests.clamp_dimension(0) == 1
    assert render_requests.clamp_dimension(20000) == 16384


class _FakeViewMatrix:
    def inverted(self) -> tuple[tuple[float, ...], ...]:
        return (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )


class _FakeOffsetViewMatrix:
    def inverted(self) -> tuple[tuple[float, ...], ...]:
        return (
            (2.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )


def _frame_point(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y)


def _orthographic_camera_data(
    *,
    frame_width: float,
    frame_height: float,
    frame_center_x: float = 0.0,
    frame_center_y: float = 0.0,
    lens: float = 45.0,
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
        clip_start=0.25,
        clip_end=400.0,
        shift_x=shift_x,
        shift_y=shift_y,
        dof=SimpleNamespace(use_dof=False),
        view_frame=view_frame,
    )


def _viewport_context(view_perspective: str = "PERSP", scene: object | None = None) -> SimpleNamespace:
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
        scene=scene,
    )


def _render_scene(
    presentation: str = color_presentation.MODE_SCENE_LINEAR_HDR,
) -> SimpleNamespace:
    return SimpleNamespace(
        render=SimpleNamespace(
            resolution_x=640,
            resolution_y=360,
            resolution_percentage=50,
        ),
        ovrtx_example=SimpleNamespace(
            render_product_path="/Render/Test/Product",
            min_samples=4,
            max_samples=16,
            camera_prim_path="/World/Camera",
            sync_viewport_camera=False,
            simulation_reset_token=7,
            color_presentation_mode=presentation,
        ),
        frame_current=12,
        frame_start=1,
        frame_end=48,
        view_settings=SimpleNamespace(
            view_transform="AgX",
            look="Medium High Contrast",
            exposure=0.25,
            gamma=1.0,
        ),
        display_settings=SimpleNamespace(display_device="sRGB"),
    )


def test_render_request_translator_records_render_signal_source_and_engine_id() -> None:
    signal = BlenderRenderSignal(
        source=BlenderRenderSignalSource.VIEW_DRAW,
        intent=BlenderRenderIntent.VIEWPORT,
        scene=_render_scene(),
        input_usd_path="/tmp/resolved.usdc",
        camera_prim_path="/Generated/Camera",
        render_product_path="/Generated/Product",
        context=_viewport_context(),
        engine_id="engine-A",
    )

    translator = RenderRequestTranslator()
    request = translator.translate(signal)

    # This is the direct-USD validation route, so
    # the signal-supplied render product applies (the live route instead pins
    # the generated default; see the live-route tests below).
    assert request.sensor_paths == ("/Generated/Product",)
    assert (request.width, request.height) == (320, 180)
    assert request.min_samples == 4
    assert request.max_samples == 16
    assert request.simulation_reset_token == 7
    assert request.input_usd_path == "/tmp/resolved.usdc"

    assert request.camera_prim_path == "/Generated/Camera"
    assert request.blender_signal == {
        "source": "view_draw",
        "intent": "viewport",
        "engine_id": "engine-A",
    }
    timings = translator._timings_snapshot()
    assert set(timings) == {
        "total_ms",
        "scene_inputs_ms",
        "camera_ms",
        "runtime_inputs_ms",
        "runtime_state_ms",
        "runtime_defaults_ms",
        "native_client_preflight_ms",
        "material_ms",
        "request_build_ms",
    }
    assert all(value >= 0.0 for value in timings.values())
    assert timings["total_ms"] >= sum(
        timings[key]
        for key in (
            "scene_inputs_ms",
            "camera_ms",
            "runtime_inputs_ms",
            "material_ms",
            "request_build_ms",
        )
    )


@pytest.mark.parametrize('current', [False, True])
def test_final_translator_preserves_current_camera_pose_without_depsgraph_field(current):
    scene = _render_scene()
    # The final boundary already evaluates the pose. Render signals have no
    # depsgraph field, and fake/current handles need not expose evaluated_get.
    scene.camera = SimpleNamespace(matrix_world=(
        (1., 0., 0., 7.), (0., 1., 0., 8.), (0., 0., 1., 9.), (0., 0., 0., 1.)))
    signal = BlenderRenderSignal(source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER, scene=scene,
        input_usd_path='/tmp/current.usda', camera_prim_path='/World/Camera',
        render_product_path='/Render/Product', current_scene_generation=current)
    request = RenderRequestTranslator(include_material_presentation=False).translate(signal)
    assert request.camera_matrix is None
    assert request.scene_camera_matrix is None
    assert request.native_scene_camera_matrix == (render_requests.matrix_to_usd_rows(scene.camera.matrix_world) if current else None)


@pytest.mark.parametrize('field,value,attribute,expected', [
    ('lens', 75., 'focalLength', 75.),
    ('sensor_width', 20., 'horizontalAperture', 20.),
    ('shift_x', .3, 'horizontalApertureOffset', 10.8),
    ('clip_start', 15., 'clippingRange', (15., 100.)),
    ('dof', SimpleNamespace(use_dof=True, focus_distance=2., aperture_fstop=.5, focus_object=None), 'fStop', 50.),
])
def test_current_scene_unsynced_viewport_carries_authored_camera_values(field, value, attribute, expected):
    scene = _render_scene()
    data = SimpleNamespace(type='PERSP', lens=50., sensor_width=36., sensor_height=24.,
        sensor_fit='AUTO', shift_x=0., shift_y=0., clip_start=.1, clip_end=100.,
        dof=SimpleNamespace(use_dof=False))
    scene.camera = SimpleNamespace(data=data, matrix_world=(
        (1.,0.,0.,7.),(0.,1.,0.,8.),(0.,0.,1.,9.),(0.,0.,0.,1.)))
    context = _viewport_context(scene=scene)
    signal = BlenderRenderSignal(BlenderRenderSignalSource.VIEW_DRAW, BlenderRenderIntent.VIEWPORT,
        scene, '/tmp/current.usda', camera_prim_path='/World/Camera',
        render_product_path='/Render/Product', context=context, current_scene_generation=True)
    translator = RenderRequestTranslator(include_material_presentation=False)
    before = translator.translate(signal)
    setattr(data, field, value)
    after = translator.translate(signal)
    assert after.camera_matrix is None and after.scene_camera_matrix is None
    assert after.native_scene_camera_matrix[3][:3] == (7.,8.,9.)
    assert after.camera_projection != before.camera_projection
    from ovrtx_blender_example import direct_camera_updates
    rows = {row['attribute_name']: row['value'] for row in
            direct_camera_updates.projection_values(after.camera_prim_path, after.camera_projection)}
    assert rows[attribute] == expected
    # Explicit-stage USD authority must not borrow the active Blender camera.
    exact = translator.translate(replace(signal, current_scene_generation=False))
    assert exact.camera_projection is None and exact.native_scene_camera_matrix is None
    # A synchronized user view must continue to use its viewport projection.
    scene.ovrtx_example.sync_viewport_camera = True
    synced = translator.translate(signal)
    assert synced.camera_projection.source == render_requests.PERSPECTIVE_USER_VIEW
    assert synced.camera_projection.focal_length == 28.
    assert synced.native_scene_camera_matrix is None


def test_unsynced_camera_replacement_uses_selected_generation_mapping():
    from ovrtx_blender_example.blender_callback_adapters import BlenderRenderCallbackAdapter
    from ovrtx_blender_example.scene_generation import BlenderId, BlenderPrimPath
    scene = _render_scene()
    cameras = [SimpleNamespace(session_uid=uid, name=f'Camera{uid}',
        data=SimpleNamespace(type='PERSP', lens=lens, sensor_width=36., sensor_height=24.,
            sensor_fit='AUTO', shift_x=0., shift_y=0., clip_start=.1, clip_end=100., dof=None),
        matrix_world=((1.,0.,0.,x),(0.,1.,0.,0.),(0.,0.,1.,0.),(0.,0.,0.,1.)))
        for uid,lens,x in ((41,50.,7.),(42,80.,2.))]
    generation = SimpleNamespace(blender_prim_paths={BlenderId('OBJECT', c.session_uid):
        BlenderPrimPath(c.name,'CAMERA',f'/World/{c.name}',f'/World/{c.name}/Camera') for c in cameras},
        materialize_usd=lambda:'/tmp/current.usda')
    adapter = BlenderRenderCallbackAdapter(generation_for_scene=lambda _:generation,
        translator=RenderRequestTranslator(include_material_presentation=False))
    for camera in cameras + cameras[:1]:
        scene.camera = camera
        request = adapter.view_draw_from_scene(scene,_viewport_context(scene=scene))
        assert request.camera_prim_path == f'/World/{camera.name}/Camera'
        assert request.camera_projection.focal_length == camera.data.lens
        assert request.native_scene_camera_matrix[3][0] == camera.matrix_world[0][3]


@pytest.mark.parametrize("playing", [False, True])
def test_current_scene_viewport_uses_blender_timeline_without_environment(monkeypatch, playing):
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_SHARED_STAGE", raising=False)
    scene = _render_scene()
    context = _viewport_context(scene=scene)
    context.screen = SimpleNamespace(is_animation_playing=playing)
    signal = BlenderRenderSignal(
        source=BlenderRenderSignalSource.VIEW_DRAW,
        intent=BlenderRenderIntent.VIEWPORT,
        scene=scene, context=context, current_scene_generation=True,
        input_usd_path="/tmp/generated.usdc", camera_prim_path="/World/Camera",
        render_product_path="/Render/Product",
    )
    request = RenderRequestTranslator(include_material_presentation=False).translate(signal)
    assert request.timeline_controls_enabled
    assert request.timeline_playing is playing
    assert request.timeline_frame == scene.frame_current


@pytest.mark.parametrize("setting,enabled", [("", False), ("0", False), ("false", False), ("1", True)])
def test_exact_stage_timeline_respects_boolean_environment(monkeypatch, setting, enabled):
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_SHARED_STAGE", setting)
    signal = BlenderRenderSignal(
        source=BlenderRenderSignalSource.VIEW_DRAW,
        intent=BlenderRenderIntent.VIEWPORT,
        scene=_render_scene(), context=_viewport_context(),
        input_usd_path="/tmp/exact.usda", camera_prim_path="/World/Camera",
        render_product_path="/Render/Product",
    )
    assert RenderRequestTranslator(include_material_presentation=False).translate(signal).timeline_controls_enabled is enabled


def test_render_intent_selects_policy_without_rewriting_event_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _render_scene()
    scene.ovrtx_example.sync_viewport_camera = True
    material_modes: list[bool] = []
    translator = RenderRequestTranslator()
    monkeypatch.setattr(
        translator,
        "_material_scene_layer_from_scene",
        lambda *_args, **kwargs: material_modes.append(kwargs["use_materialx"]),
    )
    common = {
        "source": BlenderRenderSignalSource.VIEW_DRAW,
        "scene": scene,
        "input_usd_path": "/tmp/resolved.usdc",
        "camera_prim_path": "/Generated/Camera",
        "render_product_path": "/Generated/Product",
        "context": _viewport_context(),
    }

    viewport = translator.translate(
        BlenderRenderSignal(intent=BlenderRenderIntent.VIEWPORT, **common)
    )
    final = translator.translate(
        BlenderRenderSignal(intent=BlenderRenderIntent.FINAL_RENDER, **common)
    )

    # Both intents use the MaterialX/OpenPBR route (F12 matches the viewport).
    assert material_modes == [True, True]
    assert viewport.rtpt_value_route is True
    assert final.rtpt_value_route is False
    assert viewport.camera_matrix is not None
    assert final.camera_matrix is None
    assert (
        viewport.blender_signal["source"]
        == final.blender_signal["source"]
        == "view_draw"
    )
    assert viewport.blender_signal["intent"] == "viewport"
    assert final.blender_signal["intent"] == "final_render"


def test_viewport_intent_without_context_rejects_with_actual_event_source() -> None:
    with pytest.raises(
        BlenderSignalTranslationError,
        match="view_update: viewport render intent requires viewport context",
    ):
        RenderRequestTranslator().translate(
            BlenderRenderSignal(
                source=BlenderRenderSignalSource.VIEW_UPDATE,
                intent=BlenderRenderIntent.VIEWPORT,
                scene=_render_scene(),
                input_usd_path="/tmp/resolved.usdc",
                camera_prim_path="/Generated/Camera",
                render_product_path="/Generated/Product",
            )
        )


@pytest.mark.parametrize(
    ("mode", "render_var", "frame_format", "frame_color_mode", "display_owner"),
    (
        (
            color_presentation.MODE_LDR_RGBA8_DISPLAY_PASSTHROUGH,
            color_presentation.RENDER_VAR_LDR_COLOR,
            color_presentation.FRAME_FORMAT_RGBA8,
            color_presentation.FRAME_COLOR_MODE_DISPLAY_LDR,
            color_presentation.DISPLAY_TRANSFORM_OWNER_OVRTX,
        ),
        (
            color_presentation.MODE_SCENE_LINEAR_HDR,
            color_presentation.RENDER_VAR_HDR_COLOR,
            color_presentation.FRAME_FORMAT_RGBA16F,
            color_presentation.FRAME_COLOR_MODE_SCENE_LINEAR,
            "consumer",
        ),
    ),
)
@pytest.mark.parametrize("source", tuple(BlenderRenderSignalSource))
def test_render_request_translator_preserves_scene_presentation_for_every_callback(
    mode: str,
    render_var: str,
    frame_format: str,
    frame_color_mode: str,
    display_owner: str,
    source: BlenderRenderSignalSource,
) -> None:
    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            source=source,
            intent=(
                BlenderRenderIntent.FINAL_RENDER
                if source is BlenderRenderSignalSource.FINAL_RENDER
                else BlenderRenderIntent.VIEWPORT
            ),
            scene=_render_scene(mode),
            input_usd_path="/tmp/resolved.usdc",
            camera_prim_path="/Generated/Camera",
            render_product_path="/Generated/Product",
            context=_viewport_context(),
        )
    )

    assert request.color_presentation["active_mode"] == mode
    assert request.color_presentation["render_var"] == render_var
    assert request.color_presentation["frame_format"] == frame_format
    assert request.color_presentation["frame_color_mode"] == frame_color_mode
    assert request.color_presentation["display_transform_owner"] == display_owner


def test_render_request_translator_ignores_legacy_native_path_and_worker_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_client_path = tmp_path / "native"
    native_client_path.mkdir()
    monkeypatch.setenv("OV_BLENDER_EXAMPLE_WORKER_COMMAND", "worker")
    monkeypatch.setenv(
        "OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH",
        str(native_client_path),
    )

    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            source=BlenderRenderSignalSource.VIEW_DRAW,
            intent=BlenderRenderIntent.VIEWPORT,
            scene=_render_scene(),
            input_usd_path="/tmp/resolved.usdc",
            camera_prim_path="/Generated/Camera",
            render_product_path="/Generated/Product",
            context=_viewport_context(),
        )
    )

    assert request.worker_command != "worker"
    assert request.native_client_module == ""
    assert str(native_client_path) not in sys.path


def test_render_request_translator_reuses_bundled_runtime_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_WORKER_COMMAND", raising=False)
    monkeypatch.delenv("OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH", raising=False)
    calls = 0

    def defaults(root: Path | None = None) -> bundled_runtime.BundledRuntimeDefaults:
        nonlocal calls
        calls += 1
        return bundled_runtime.BundledRuntimeDefaults(
            root=tmp_path,
            platform_id="linux-x64",
            worker_command="worker",
        )

    monkeypatch.setattr(bundled_runtime, "defaults", defaults)
    translator = RenderRequestTranslator()
    signal = BlenderRenderSignal(
        source=BlenderRenderSignalSource.VIEW_DRAW,
        intent=BlenderRenderIntent.VIEWPORT,
        scene=_render_scene(),
        input_usd_path="/tmp/resolved.usdc",
        camera_prim_path="/Generated/Camera",
        render_product_path="/Generated/Product",
        context=_viewport_context(),
    )

    first = translator.translate(signal)
    second = translator.translate(signal)

    assert calls == 1
    assert first.worker_command == second.worker_command


def test_viewport_keeps_activation_presentation_during_live_value_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material_calls: list[float] = []
    light_calls: list[float] = []

    class Material:
        name = "Paint"
        roughness = 0.25

        def as_pointer(self) -> int:
            return 42

    class Light:
        type = "LIGHT"
        energy = 100.0

        def as_pointer(self) -> int:
            return 43

    presentation = render_requests.MaterialPresentationLayer(
        target_path="/World/Target",
        layer_body="",
        authored_properties=(),
        digest_content={},
        diagnostics={},
    )

    def convert(materials: tuple[object, ...], input_usd_path: str, **_kwargs: object):
        material_calls.append(float(getattr(materials[0], "roughness")))
        return materialx_openpbr_conversion.MaterialSceneConversionResult(
            materialx_openpbr_conversion.MaterialSceneConversionStatus.OK,
            presentation,
        )

    def convert_lights(objects: tuple[object, ...], input_usd_path: str):
        light_calls.append(float(getattr(objects[0], "energy")))
        return presentation

    material = Material()
    light = Light()
    monkeypatch.setattr(materialx_openpbr_conversion, "scene_layer_from_materials", convert)
    monkeypatch.setattr(light_scene_layer, "scene_layer_from_lights", convert_lights)
    translator = RenderRequestTranslator(
        blender_module_provider=lambda: SimpleNamespace(
            data=SimpleNamespace(materials=(material,), objects=(light,))
        )
    )
    scene = _render_scene()

    signal_args = (
        BlenderRenderIntent.VIEWPORT,
        scene,
        "scene.usda",
        "/Camera/Camera",
        "/Render/Product",
    )
    context = _viewport_context()
    first = translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            *signal_args,
            context=context,
        )
    )
    translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_DRAW,
            *signal_args,
            context=context,
        )
    )
    translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_DRAW,
            *signal_args,
            context=context,
        )
    )
    assert material_calls == [0.25]
    assert light_calls == [100.0]

    material.roughness = 0.75
    light.energy = 250.0
    edited = translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            *signal_args,
            context=context,
        )
    )
    assert material_calls == [0.25]
    assert light_calls == [100.0]
    assert ovrtx_session.reuse_decision(
        ovrtx_session.build_spec(first),
        ovrtx_session.build_spec(edited),
    ).reuse

    translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            "replacement.usda",
            "/Camera/Camera",
            "/Render/Product",
            context=context,
        )
    )
    assert material_calls == [0.25, 0.75]
    assert light_calls == [100.0, 250.0]


def test_current_scene_translation_retains_material_and_light_presentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    material_layer = object()
    light_layer = object()
    translator = RenderRequestTranslator(
        blender_module_provider=lambda: SimpleNamespace(
            data=SimpleNamespace(materials=(object(),), objects=())
        )
    )
    monkeypatch.setattr(
        translator,
        "_material_scene_layer_from_scene",
        lambda *_args, **_kwargs: material_layer,
    )
    monkeypatch.setattr(
        translator,
        "_light_scene_layer_from_scene",
        lambda *_args, **_kwargs: light_layer,
    )

    request = translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            _render_scene(),
            "/tmp/generated.usdc",
            camera_prim_path="/Camera/Camera",
            render_product_path="/Render/OVRTX/Product",
            context=_viewport_context(),
            current_scene_generation=True,
        )
    )

    assert request.input_usd_path == "/tmp/generated.usdc"
    assert request.current_scene_generation
    assert request.material_scene_layer is material_layer
    assert request.light_scene_layer is light_layer


def test_final_render_policy_differs_only_by_current_scene_provenance() -> None:
    translator = RenderRequestTranslator(include_material_presentation=False)
    signal = dict(
        source=BlenderRenderSignalSource.FINAL_RENDER,
        intent=BlenderRenderIntent.FINAL_RENDER,
        scene=_render_scene(),
        input_usd_path="/tmp/scene.usda",
        camera_prim_path="/Camera/Camera",
        render_product_path="/Render/Test/Product",
        context=_viewport_context(),
    )

    exact = translator.translate(BlenderRenderSignal(**signal))
    current = translator.translate(
        BlenderRenderSignal(**signal, current_scene_generation=True)
    )

    assert replace(current, current_scene_generation=False) == exact


def test_live_blender_route_ignores_stale_camera_and_product_settings() -> None:
    """Saved scene settings from the old fixture workflow (e.g.
    ``/Camera/Camera``) must not redirect the generated camera or render
    product on the live route: the engine-supplied signal paths pin the
    camera to the generated presentation camera and the render product to
    the generated default. The settings remain honored on the direct-USD
    validation route, where the engine forwards them through the signal."""

    scene = _render_scene()
    scene.ovrtx_example.camera_prim_path = "/Camera/Camera"
    scene.ovrtx_example.render_product_path = "/Stale/Product"

    live_request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            "",
            camera_prim_path=render_requests.LIVE_AUTHORING_CAMERA_PATH,
            render_product_path="",
            context=_viewport_context(),
            current_scene_generation=True,
        )
    )
    assert live_request.camera_prim_path == render_requests.LIVE_AUTHORING_CAMERA_PATH
    assert live_request.sensor_paths == (properties.DEFAULT_RENDER_PRODUCT_PATH,)
    assert live_request.selected_sensor_paths == (properties.DEFAULT_RENDER_PRODUCT_PATH,)

    direct_request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            "/tmp/fixture.usda",
            camera_prim_path="/Camera/Camera",
            render_product_path="/Stale/Product",
            context=_viewport_context(),
        )
    )
    assert direct_request.camera_prim_path == "/Camera/Camera"
    assert direct_request.sensor_paths == ("/Stale/Product",)


def _perspective_scene_camera(matrix_rows: tuple[tuple[float, ...], ...]) -> SimpleNamespace:
    class _Matrix:
        def __getitem__(self, index: int) -> tuple[float, ...]:
            return matrix_rows[index]

    return SimpleNamespace(
        matrix_world=_Matrix(),
        data=SimpleNamespace(
            type="PERSP",
            lens=42.0,
            sensor_width=36.0,
            sensor_height=24.0,
            sensor_fit="HORIZONTAL",
            clip_start=0.5,
            clip_end=250.0,
            shift_x=0.0,
            shift_y=0.0,
            dof=SimpleNamespace(use_dof=False),
        ),
    )


def test_final_render_direct_usd_keeps_the_stage_camera_authoritative() -> None:
    scene = _render_scene()
    scene.camera = _perspective_scene_camera(
        (
            (1.0, 0.0, 0.0, 4.0),
            (0.0, 1.0, 0.0, 5.0),
            (0.0, 0.0, 1.0, 6.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.FINAL_RENDER,
            BlenderRenderIntent.FINAL_RENDER,
            scene,
            "/fixtures/render.usda",
            camera_prim_path="",
            render_product_path="/Render/Test/Product",
        )
    )

    assert request.scene_camera_matrix is None
    assert request.camera_projection is None


def test_viewport_live_authoring_carries_no_composed_scene_camera_pose() -> None:
    scene = _render_scene()
    scene.camera = _perspective_scene_camera(
        (
            (1.0, 0.0, 0.0, 4.0),
            (0.0, 1.0, 0.0, 5.0),
            (0.0, 0.0, 1.0, 6.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    request = RenderRequestTranslator().translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            "",
            camera_prim_path=render_requests.LIVE_AUTHORING_CAMERA_PATH,
            render_product_path="",
            context=_viewport_context(),
            current_scene_generation=True,
        )
    )

    assert request.scene_camera_matrix is None


def test_scene_camera_world_matrix_transposes_to_usd_rows() -> None:
    scene = SimpleNamespace(
        camera=SimpleNamespace(
            matrix_world=(
                (1.0, 0.0, 0.0, 4.0),
                (0.0, 0.0, -1.0, 5.0),
                (0.0, 1.0, 0.0, 6.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
    )

    assert render_requests.scene_camera_world_matrix(scene) == (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (4.0, 5.0, 6.0, 1.0),
    )
    assert render_requests.scene_camera_world_matrix(SimpleNamespace(camera=None)) is None
    assert render_requests.scene_camera_world_matrix(None) is None


def test_exact_stage_translation_preserves_source_materials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        materialx_openpbr_conversion,
        "scene_layer_from_materials",
        lambda *_args, **_kwargs: pytest.fail(
            "exact-stage translation must not replace source materials"
        ),
    )
    translator = RenderRequestTranslator(
        blender_module_provider=lambda: SimpleNamespace(
            data=SimpleNamespace(materials=(object(),))
        ),
        include_material_presentation=False,
    )

    request = translator.translate(
        BlenderRenderSignal(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            _render_scene(),
            "/fixtures/exact.usda",
            camera_prim_path="/Camera/Camera",
            render_product_path="/Render/OVRTX/Product",
            context=_viewport_context(),
        )
    )

    assert request.material_scene_layer is None


def test_camera_derives_perspective_region_state() -> None:
    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=True,
        context=_viewport_context(),
    )

    assert (request.width, request.height) == (1383, 614)
    assert request.camera_controls_mode == "blender_view"
    assert request.camera_matrix is not None
    projection = request.camera_projection
    assert projection is not None
    assert projection.source == render_requests.PERSPECTIVE_USER_VIEW
    assert projection.route == render_requests.OVRTX_SCENE_COMPOSITION_ROUTE
    assert projection.runtime_status == render_requests.RUNTIME_PROJECTION_UNPROVEN
    diagnostics = projection.to_diagnostics()
    assert diagnostics["composed_usd_only_attributes"] == ["clippingRange"]
    assert "fixture" + "_attributes" not in diagnostics
    assert "fixture" + "_only_attributes" not in diagnostics
    assert projection.viewport_region == (2766, 1228)
    assert projection.render_size == (1383, 614)
    assert projection.usd_attributes() == {
        "projection": "perspective",
        "focalLength": 28.0,
        "horizontalAperture": 56.0,
        "verticalAperture": 28.0,
        "horizontalApertureOffset": 0.0,
        "verticalApertureOffset": 0.0,
        "fStop": 0.0,
    }


def test_camera_can_keep_fixed_resolution_while_syncing_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(render_requests.ENV_FIXED_VIEWPORT_RESOLUTION, "1")

    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=True,
        context=_viewport_context(),
    )

    assert (request.width, request.height) == (1280, 720)
    assert request.camera_controls_mode == "blender_view"


@pytest.mark.parametrize('fit,width,height,expected', [('HORIZONTAL',1280,720,(3.6,-7.2)),('VERTICAL',1280,720,(2.4,-4.8)),('AUTO',720,1280,(3.6,-7.2))])
def test_active_camera_lens_shift_uses_fitted_sensor_units(fit,width,height,expected):
    data=SimpleNamespace(type='PERSP',lens=50,sensor_width=36,sensor_height=24,sensor_fit=fit,shift_x=.1,shift_y=-.2)
    projection=render_requests.camera_projection_from_scene_camera(SimpleNamespace(camera=SimpleNamespace(data=data)),None,width,height)
    assert (projection.horizontal_aperture_offset,projection.vertical_aperture_offset)==expected


@pytest.mark.parametrize("camera_type", ("PANO", "UNKNOWN_PROJECTION"))
@pytest.mark.parametrize("viewport", (False, True))
def test_unsupported_scene_camera_projection_is_reported_without_perspective_fallback(
    camera_type, viewport, monkeypatch,
):
    from ovrtx_blender_example.ovrtx_runtime_client import RenderClientError

    monkeypatch.setenv(render_requests.ENV_FIXED_VIEWPORT_RESOLUTION, "1")
    camera_data = SimpleNamespace(
        type=camera_type, lens=50.0, sensor_width=36.0, sensor_height=24.0,
        sensor_fit="AUTO", shift_x=0.0, shift_y=0.0,
    )
    scene = SimpleNamespace(camera=SimpleNamespace(data=camera_data))
    context = _viewport_context("CAMERA", scene) if viewport else None
    with pytest.raises(RenderClientError, match="unsupported in OVRTX.*perspective or orthographic"):
        render_requests.camera(
            base_width=640, base_height=360, camera_prim_path="/World/Camera",
            sync_viewport_camera=viewport, context=context, scene=scene,
        )
    assert camera_data.type == camera_type


def test_unsupported_active_camera_does_not_block_independent_user_perspective_view():
    scene = SimpleNamespace(camera=SimpleNamespace(data=SimpleNamespace(type="PANO")))
    request = render_requests.camera(
        base_width=640, base_height=360, camera_prim_path="/World/Camera",
        sync_viewport_camera=True, context=_viewport_context("PERSP", scene), scene=scene,
    )
    assert request.camera_projection.projection == "perspective"
    assert request.camera_projection.source == render_requests.PERSPECTIVE_USER_VIEW


def test_camera_derives_active_camera_projection_and_dof_state() -> None:
    camera_data = SimpleNamespace(
        type="PERSP",
        lens=50.0,
        sensor_width=36.0,
        sensor_height=24.0,
        sensor_fit="HORIZONTAL",
        clip_start=0.05,
        clip_end=250.0,
        shift_x=0.1,
        shift_y=-0.2,
        dof=SimpleNamespace(
            use_dof=True,
            aperture_fstop=2.0,
            focus_object=None,
            focus_distance=7.5,
        ),
    )
    scene = SimpleNamespace(camera=SimpleNamespace(data=camera_data))

    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=True,
        context=_viewport_context("CAMERA", scene=scene),
    )

    projection = request.camera_projection
    assert (request.width, request.height) == (1280, 720)
    assert projection is not None
    assert projection.source == render_requests.ACTIVE_CAMERA_VIEW
    assert projection.sensor_fit == "HORIZONTAL"
    assert projection.lens_shift == (0.1, -0.2)
    assert projection.usd_attributes() == {
        "projection": "perspective",
        "focalLength": 50.0,
        "horizontalAperture": 36.0,
        "verticalAperture": 20.25,
        "horizontalApertureOffset": 3.6,
        "verticalApertureOffset": -7.2,
        "fStop": 200.0,
        "clippingRange": (0.05, 250.0),
        "focusDistance": 7.5,
    }


def test_camera_derives_orthographic_user_view_projection_state() -> None:
    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=True,
        context=_viewport_context("ORTHO"),
    )

    projection = request.camera_projection
    assert (request.width, request.height) == (1383, 614)
    assert request.camera_controls_mode == "blender_view"
    assert request.camera_matrix is not None
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


def test_camera_derives_active_orthographic_projection_state_without_viewport_context() -> None:
    scene = SimpleNamespace(
        camera=SimpleNamespace(
            data=_orthographic_camera_data(frame_width=6.0, frame_height=3.375)
        )
    )

    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=True,
        context=None,
        scene=scene,
    )

    projection = request.camera_projection
    assert request.camera_controls_mode == "usd_camera"
    assert request.camera_matrix is None
    assert projection is not None
    assert projection.source == render_requests.ACTIVE_CAMERA_VIEW
    assert projection.sensor_fit == "AUTO"
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


def test_camera_derives_shifted_active_orthographic_projection_offsets() -> None:
    scene = SimpleNamespace(
        camera=SimpleNamespace(
            data=_orthographic_camera_data(
                frame_width=6.0,
                frame_height=3.375,
                frame_center_x=0.6,
                frame_center_y=-1.2,
                shift_x=0.1,
                shift_y=-0.2,
            )
        )
    )

    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=True,
        context=None,
        scene=scene,
    )

    projection = request.camera_projection
    assert projection is not None
    assert projection.lens_shift == (0.1, -0.2)
    assert projection.usd_attributes()["horizontalAperture"] == 60.0
    assert projection.usd_attributes()["verticalAperture"] == 33.75
    assert projection.usd_attributes()["horizontalApertureOffset"] == 6.0
    assert projection.usd_attributes()["verticalApertureOffset"] == -12.0


def test_camera_derives_final_orthographic_projection_when_viewport_sync_is_disabled() -> None:
    scene = SimpleNamespace(
        camera=SimpleNamespace(
            data=_orthographic_camera_data(frame_width=3.375, frame_height=6.0, sensor_fit="VERTICAL")
        )
    )

    request = render_requests.camera(
        base_width=720,
        base_height=1280,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=False,
        context=None,
        scene=scene,
    )

    projection = request.camera_projection
    assert request.camera_controls_mode == "usd_camera"
    assert request.camera_matrix is None
    assert projection is not None
    assert projection.sensor_fit == "VERTICAL"
    assert projection.usd_attributes()["horizontalAperture"] == 33.75
    assert projection.usd_attributes()["verticalAperture"] == 60.0


def test_camera_falls_back_to_usd_camera() -> None:
    request = render_requests.camera(
        base_width=1280,
        base_height=720,
        camera_prim_path="/World/Camera",
        sync_viewport_camera=False,
        context=_viewport_context(),
    )

    assert (request.width, request.height) == (1280, 720)
    assert request.camera_controls_mode == "usd_camera"
    assert request.camera_matrix is None
    assert request.camera_projection is None


def test_scene_camera_pose_delta_reports_seed_match() -> None:
    camera_matrix = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    scene = SimpleNamespace(camera=SimpleNamespace(matrix_world=camera_matrix))

    assert (
        render_requests.scene_camera_pose_delta(
            SimpleNamespace(
                scene=scene,
                region_data=SimpleNamespace(view_matrix=_FakeViewMatrix()),
            )
        )
        == 0.0
    )
    assert (
        render_requests.scene_camera_pose_delta(
            SimpleNamespace(
                scene=scene,
                region_data=SimpleNamespace(view_matrix=_FakeOffsetViewMatrix()),
            )
        )
        == 1.0
    )


def test_tick_omits_blender_objects() -> None:
    request = SimpleNamespace(
        input_usd_path="/fixture.usda",
        timeline_controls_enabled=True,
        timeline_playing=True,
        timeline_frame=12,
        timeline_start=3,
        timeline_end=18,
        simulation_reset_token=4,
    )

    tick = render_requests.tick(
        request,
        now_ns=123,
    )

    assert tick.input_usd_path == "/fixture.usda"
    assert not hasattr(tick, "ovrtx_updates")
    assert tick.now_ns == 123
    assert tick.timeline_controls_enabled is True
    assert tick.timeline_playing is True
    assert tick.timeline_frame == 12
    assert tick.timeline_start == 3
    assert tick.timeline_end == 18
    assert tick.simulation_reset_token == 4
    assert not hasattr(tick, "ovrtx_updates")


def test_reset_reason_prioritizes_runtime_composition() -> None:
    assert (
        render_requests.reset_reason(
            composition_changed=True,
            camera_changed=True,
            snapshot_changed=True,
            value_edit=True,
        )
        == "composition_changed"
    )
    assert render_requests.reset_reason(camera_changed=True) == "camera_changed"
    assert render_requests.reset_reason(snapshot_changed=True) == "snapshot_changed"
    assert render_requests.reset_reason() == ""


def test_reset_reason_value_edit_yields_to_existing_reasons() -> None:
    # task04-06: an applied value-update batch with no other change records
    # value_edit; camera-only (and other) changes keep their existing
    # reasons even when values were written in the same tick.
    assert render_requests.reset_reason(value_edit=True) == "value_edit"
    assert (
        render_requests.reset_reason(camera_changed=True, value_edit=True)
        == "camera_changed"
    )
    assert (
        render_requests.reset_reason(snapshot_changed=True, value_edit=True)
        == "snapshot_changed"
    )
    assert (
        render_requests.reset_reason(composition_changed=True, value_edit=True)
        == "composition_changed"
    )


@pytest.mark.parametrize("generation_path,expected", [
    ("/tmp/generated.usdc", {7: "/World/_materials/Lamp"}),
    ("/tmp/other.usdc", None),
])
def test_current_scene_translation_passes_generation_material_identity(
    monkeypatch: pytest.MonkeyPatch, generation_path: str, expected,
) -> None:
    from ovrtx_blender_example import direct_scene_updates, scene_generation, scene_generation_sessions

    scene = _render_scene()
    generation = SimpleNamespace(
        usd_path=generation_path,
        blender_prim_paths={
            scene_generation.BlenderId("MATERIAL", 7): SimpleNamespace(schema_path="/World/_materials/Lamp"),
            scene_generation.BlenderId("OBJECT", 9): SimpleNamespace(schema_path="/World/Lamp/Mesh"),
        },
    )
    monkeypatch.setitem(scene_generation_sessions._owners, int(getattr(scene, "session_uid", 0)),
                        SimpleNamespace(current_generation=generation))
    monkeypatch.setattr(direct_scene_updates, "presentation", lambda *_: None)
    seen = {}
    translator = RenderRequestTranslator(
        blender_module_provider=lambda: SimpleNamespace(data=SimpleNamespace(materials=(object(),), objects=()))
    )
    monkeypatch.setattr(translator, "_material_scene_layer_from_scene",
                        lambda *_args, **kwargs: seen.update(kwargs) or None)
    monkeypatch.setattr(translator, "_light_scene_layer_from_scene", lambda *_args, **_kwargs: None)

    translator.translate(BlenderRenderSignal(
        BlenderRenderSignalSource.VIEW_UPDATE, BlenderRenderIntent.VIEWPORT, scene, "/tmp/generated.usdc",
        camera_prim_path="/Camera/Camera", render_product_path="/Render/OVRTX/Product",
        context=_viewport_context(), current_scene_generation=True))

    assert seen["known_material_paths"] == expected


def test_material_route_selection_per_intent() -> None:
    from ovrtx_blender_example.blender_signal_translation import material_route_uses_materialx

    assert material_route_uses_materialx(BlenderRenderIntent.VIEWPORT) is True
    assert material_route_uses_materialx(BlenderRenderIntent.FINAL_RENDER) is True


def test_final_render_reuses_nothing_and_viewport_reuses_material_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = _render_scene()
    calls: list[tuple[bool, bool]] = []
    translator = RenderRequestTranslator()
    monkeypatch.setattr(
        translator,
        "_material_scene_layer_from_scene",
        lambda *_args, **kwargs: calls.append((kwargs["reuse"], kwargs["use_materialx"])),
    )
    common = {
        "source": BlenderRenderSignalSource.VIEW_DRAW,
        "scene": scene,
        "input_usd_path": "/tmp/resolved.usdc",
        "camera_prim_path": "/Generated/Camera",
        "render_product_path": "/Generated/Product",
        "context": _viewport_context(),
    }
    translator.translate(BlenderRenderSignal(intent=BlenderRenderIntent.VIEWPORT, **common))
    translator.translate(BlenderRenderSignal(intent=BlenderRenderIntent.FINAL_RENDER, **common))
    assert calls == [(True, True), (False, True)]
