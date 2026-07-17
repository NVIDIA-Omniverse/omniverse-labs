# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete translators from Blender signals to add-on payloads."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from . import blender_interactive_edit_builders
from . import bundled_runtime
from . import color_presentation
from . import interactive_operator_state
from . import light_scene_layer
from . import materialx_openpbr_conversion
from . import render_requests
from . import runtime_bundle_status
from . import runtime_generation
from . import usd_preview_emission_layer
from .blender_signals import (
    BlenderEditSignal,
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
)
from .interactive_edit_planner import InteractiveEdit
from .properties import (
    DEFAULT_RENDER_PRODUCT_PATH,
    RTPT_RENDER_SETTINGS,
)
from .render_requests import MaterialPresentationLayer, RenderRequest
from .usd_prim_resolver import UsdPrimResolver
from .value_edit_conversion import ValueEditConversionPolicies

try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]


def _resolve_materialized_runtime_root() -> Path | None:
    """Resolve the installed runtime's current root so the render path finds the
    worker/native client where Install Runtime placed them (mirrors preflight)."""

    try:
        status = runtime_bundle_status()
    except Exception:
        return None
    if status.get("state") != "ready":
        return None
    current_root = str(status.get("current_root") or "")
    return Path(current_root) if current_root else None


class BlenderSignalTranslationError(RuntimeError):
    """Raised when a Blender render signal cannot produce a render request."""

    def __init__(self, source: BlenderRenderSignalSource, message: str) -> None:
        self.source = source
        super().__init__(f"{source.value}: {message}")


class RenderRequestTranslator:
    """Translate a single Blender render signal into a render request."""

    def __init__(
        self,
        *,
        blender_module_provider: Callable[[], Any] | None = None,
        include_material_presentation: bool = True,
    ) -> None:
        self._blender_module_provider = blender_module_provider or (lambda: bpy)
        self._include_material_presentation = include_material_presentation
        self._material_scene_layer_cache_key: tuple[str, tuple[int, ...]] | None = None
        self._material_scene_layer_cache_value: MaterialPresentationLayer | None = None
        self._light_scene_layer_cache_key: tuple[str, tuple[int, ...]] | None = None
        self._light_scene_layer_cache_value: MaterialPresentationLayer | None = None
        self._bundled_runtime_defaults: bundled_runtime.BundledRuntimeDefaults | None = None
        self._last_timings: dict[str, float] = {}
        self._runtime_generation: int | None = None
        self._runtime_root: Path | None = None

    def _materialized_runtime_root(self) -> Path | None:
        # Recompute only when the runtime changes; translate() runs per viewport draw.
        generation = runtime_generation()
        if generation != self._runtime_generation:
            self._runtime_root = _resolve_materialized_runtime_root()
            self._bundled_runtime_defaults = None
            self._runtime_generation = generation
        return self._runtime_root

    def _bundled_defaults(self) -> bundled_runtime.BundledRuntimeDefaults:
        root = self._materialized_runtime_root()
        if self._bundled_runtime_defaults is None:
            self._bundled_runtime_defaults = bundled_runtime.defaults(root=root)
        return self._bundled_runtime_defaults

    def translate(self, signal: BlenderRenderSignal) -> RenderRequest:
        translation_started_ns = time.perf_counter_ns()
        scene = signal.scene
        context = signal.context if signal.intent is BlenderRenderIntent.VIEWPORT else None
        if signal.intent is BlenderRenderIntent.VIEWPORT and context is None:
            raise BlenderSignalTranslationError(
                signal.source,
                "viewport render intent requires viewport context",
            )
        phase_started_ns = time.perf_counter_ns()
        width, height = render_requests.resolution_from_scene(scene)
        prefs = _addon_preferences(context)

        settings = getattr(scene, "ovrtx_example", None)
        render_product_path = signal.render_product_path or DEFAULT_RENDER_PRODUCT_PATH
        min_samples = max(1, int(getattr(settings, "min_samples", 1)))
        max_samples = max(min_samples, int(getattr(settings, "max_samples", 128)))
        # RTPT quality values, sourced from the single-source-of-truth mapping
        # so names/dtypes/defaults have one definition (task01-01). Authored on
        # every composition regardless of whether they differ from defaults.
        rtpt_quality = {
            name: getattr(settings, name, spec.default)
            for name, spec in RTPT_RENDER_SETTINGS.items()
        }
        # DLSS Super-Resolution toggle (default True = worker default). Honored
        # on the generated RenderProduct at session creation, so a change re-keys
        # the session with no worker restart; also written to the worker config.
        dlss_enabled = bool(getattr(settings, "dlss_enabled", True))
        camera_prim_path = signal.camera_prim_path
        sync_viewport_camera = bool(getattr(settings, "sync_viewport_camera", True))
        requested_color_presentation = str(
            getattr(settings, "color_presentation_mode", color_presentation.DEFAULT_MODE)
        )
        scene_inputs_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        camera = render_requests.camera(
            base_width=width,
            base_height=height,
            camera_prim_path=camera_prim_path,
            sync_viewport_camera=sync_viewport_camera,
            context=context,
            scene=scene if self._include_material_presentation else None,
        )
        camera_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        runtime_inputs_started_ns = phase_started_ns
        # The authored scene generation maps the active camera to a composed
        # camera prim (signal.camera_prim_path), so the camera pose/projection
        # come from render_requests.camera(); no runtime scene-camera override.
        camera_projection = camera.camera_projection
        scene_camera_matrix = None
        screen = getattr(context, "screen", None)
        timeline_controls_enabled = bool(os.environ.get("OV_BLENDER_EXAMPLE_SHARED_STAGE", ""))
        timeline_playing = bool(getattr(screen, "is_animation_playing", False))
        simulation_reset_token = int(getattr(settings, "simulation_reset_token", 0))

        input_usd_path = signal.input_usd_path
        worker_command = os.environ.get(
            "OV_BLENDER_EXAMPLE_WORKER_COMMAND",
            getattr(prefs, "worker_command", ""),
        )
        native_client_module = os.environ.get(
            "OV_BLENDER_EXAMPLE_NATIVE_CLIENT_MODULE",
            getattr(prefs, "native_client_module", ""),
        )
        native_client_path = os.environ.get(
            "OV_BLENDER_EXAMPLE_NATIVE_CLIENT_PATH",
            getattr(prefs, "native_client_path", ""),
        )
        runtime_state_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        if not worker_command or not native_client_path:
            bundle = self._bundled_defaults()
            if not worker_command:
                worker_command = bundle.worker_command
            if not native_client_path:
                native_client_path = bundle.native_client_path
        if not native_client_module:
            native_client_module = bundled_runtime.DEFAULT_OVRTX_NATIVE_CLIENT_MODULE
        runtime_defaults_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        if native_client_path:
            from .preflight import ensure_native_client_path

            ensure_native_client_path(native_client_path)
        native_client_preflight_ms = (
            time.perf_counter_ns() - phase_started_ns
        ) / 1_000_000.0
        runtime_inputs_ms = (
            time.perf_counter_ns() - runtime_inputs_started_ns
        ) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        # Current-scene generation controls camera/product composition; the
        # generated USD still needs Blender's render-ready material and light
        # presentation layers. Viewport layers are activation state: admitted
        # live values stay on the runtime update stream until a new generation
        # path or non-reused final render rebuilds the presentation.
        current_scene_generation = signal.current_scene_generation
        material_scene_layer = (
            self._material_scene_layer_from_scene(
                input_usd_path,
                reuse=signal.intent is BlenderRenderIntent.VIEWPORT,
                use_materialx=signal.intent is BlenderRenderIntent.VIEWPORT,
                source=signal.source,
            )
            if self._include_material_presentation
            else None
        )
        light_scene_layer_value = (
            self._light_scene_layer_from_scene(
                input_usd_path,
                reuse=signal.intent is BlenderRenderIntent.VIEWPORT,
            )
            if self._include_material_presentation
            else None
        )
        material_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0

        phase_started_ns = time.perf_counter_ns()
        request = RenderRequest(
            input_usd_path=input_usd_path,
            current_scene_generation=current_scene_generation,
            sensor_paths=(render_product_path or DEFAULT_RENDER_PRODUCT_PATH,),
            selected_sensor_paths=(render_product_path or DEFAULT_RENDER_PRODUCT_PATH,),
            width=camera.width,
            height=camera.height,
            min_samples=min_samples,
            max_samples=max_samples,
            camera_prim_path=camera.camera_prim_path,
            camera_matrix=camera.camera_matrix,
            camera_projection=camera_projection,
            scene_camera_matrix=scene_camera_matrix,
            worker_command=worker_command,
            native_client_module=native_client_module,
            timeline_controls_enabled=timeline_controls_enabled,
            timeline_playing=timeline_playing,
            timeline_frame=int(getattr(scene, "frame_current", 1)),
            timeline_start=int(getattr(scene, "frame_start", 1)),
            timeline_end=int(getattr(scene, "frame_end", 1)),
            simulation_reset_token=simulation_reset_token,
            material_scene_layer=material_scene_layer,
            light_scene_layer=light_scene_layer_value,
            color_presentation=color_presentation.presentation_from_scene(
                scene,
                requested_mode=requested_color_presentation,
            ),
            rtpt_quality=rtpt_quality,
            # Live viewport routes RTPT quality changes as runtime attribute
            # writes on the render thread (task01-04), so the values are
            # excluded from the composition digest — a change must not replace
            # the running session. F12 keeps them in the digest (it composes a
            # fresh session per job); the value never routes live there.
            rtpt_value_route=signal.intent is BlenderRenderIntent.VIEWPORT,
            dlss_enabled=dlss_enabled,
            blender_signal={
                "source": signal.source.value,
                "intent": signal.intent.value,
                "engine_id": signal.engine_id,
            },
        )
        request_build_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        total_ms = (time.perf_counter_ns() - translation_started_ns) / 1_000_000.0
        self._last_timings = {
            "total_ms": total_ms,
            "scene_inputs_ms": scene_inputs_ms,
            "camera_ms": camera_ms,
            "runtime_inputs_ms": runtime_inputs_ms,
            "runtime_state_ms": runtime_state_ms,
            "runtime_defaults_ms": runtime_defaults_ms,
            "native_client_preflight_ms": native_client_preflight_ms,
            "material_ms": material_ms,
            "request_build_ms": request_build_ms,
        }
        return request

    def _timings_snapshot(self) -> Mapping[str, float]:
        return dict(self._last_timings)

    def _material_scene_layer_from_scene(
        self,
        input_usd_path: str,
        *,
        reuse: bool = False,
        use_materialx: bool = True,
        source: BlenderRenderSignalSource,
    ) -> MaterialPresentationLayer | None:
        bpy_module = self._blender_module_provider()
        if bpy_module is None:
            return None
        data = getattr(bpy_module, "data", None)
        materials = tuple(getattr(data, "materials", ()) or ())
        if not use_materialx:
            return usd_preview_emission_layer.scene_layer_from_materials(
                materials,
                input_usd_path,
            )
        cache_key = (input_usd_path, tuple(_blender_identity(material) for material in materials))
        if reuse and cache_key == self._material_scene_layer_cache_key:
            return self._material_scene_layer_cache_value
        result = materialx_openpbr_conversion.scene_layer_from_materials(
            materials,
            input_usd_path,
            allow_stock_fallback=True,
        )
        if result.status is materialx_openpbr_conversion.MaterialSceneConversionStatus.ERROR:
            raise BlenderSignalTranslationError(source, str(result.error_reason))
        self._material_scene_layer_cache_key = cache_key
        self._material_scene_layer_cache_value = result.value
        return result.value

    def _light_scene_layer_from_scene(
        self,
        input_usd_path: str,
        *,
        reuse: bool = False,
    ) -> MaterialPresentationLayer | None:
        bpy_module = self._blender_module_provider()
        if bpy_module is None:
            return None
        data = getattr(bpy_module, "data", None)
        objects = tuple(getattr(data, "objects", ()) or ())
        light_objects = tuple(obj for obj in objects if str(getattr(obj, "type", "")) == "LIGHT")
        cache_key = (input_usd_path, tuple(_blender_identity(obj) for obj in light_objects))
        if reuse and cache_key == self._light_scene_layer_cache_key:
            return self._light_scene_layer_cache_value
        result = light_scene_layer.scene_layer_from_lights(light_objects, input_usd_path)
        self._light_scene_layer_cache_key = cache_key
        self._light_scene_layer_cache_value = result
        return result


def _blender_identity(value: Any) -> int:
    as_pointer = getattr(value, "as_pointer", None)
    return int(as_pointer()) if callable(as_pointer) else id(value)


class InteractiveEditTranslator:
    """Translate Blender edit signals into interactive edit payloads."""

    def __init__(
        self,
        *,
        edit_builder: Callable[..., list[InteractiveEdit]] | None = None,
        usd_prim_resolver: UsdPrimResolver | None = None,
        light_objects: Any = (),
        worlds: Any = (),
        value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
        selection_resolution: Mapping[str, Any] | None = None,
        selection_resolver: Callable[[Any | None], Mapping[str, Any]] | None = None,
    ) -> None:
        self._edit_builder = (
            edit_builder
            or blender_interactive_edit_builders.build_interactive_edits_from_depsgraph
        )
        self._usd_prim_resolver = usd_prim_resolver
        self._light_objects = light_objects
        self._worlds = worlds
        self._value_edit_conversion_policies = value_edit_conversion_policies
        self._selection_resolution = selection_resolution
        self._selection_resolver = (
            selection_resolver
            or interactive_operator_state.resolve_blender_selection_to_edit_owners
        )
        self.selection_resolution: Mapping[str, Any] | None = None

    def translate(self, signal: BlenderEditSignal) -> tuple[InteractiveEdit, ...]:
        selection_resolution = self._resolved_selection(signal)
        self.selection_resolution = selection_resolution
        if bool(selection_resolution.get("changed", False)) or bool(selection_resolution.get("group_rejected", False)):
            return ()
        depsgraph = _depsgraph_from_items(signal.id_items)
        return tuple(
            self._edit_builder(
                depsgraph,
                value_edit_conversion_policies=self._value_edit_conversion_policies,
                usd_prim_resolver=self._usd_prim_resolver,
                light_objects=self._light_objects,
                worlds=self._worlds,
                selection_resolution=selection_resolution,
                write_target_input_usd_path=signal.input_usd_path,
                write_target_ignored_layer_identifiers=signal.ignored_layer_identifiers,
            )
        )

    def _resolved_selection(self, signal: BlenderEditSignal) -> Mapping[str, Any]:
        if self._selection_resolution is not None:
            return self._selection_resolution
        return self._selection_resolver(signal.context)


@dataclass(frozen=True)
class _SignalUpdate:
    id: Any


@dataclass(frozen=True)
class _SignalDepsgraph:
    updates: tuple[_SignalUpdate, ...]


def _depsgraph_from_items(id_items: Iterable[Any]) -> _SignalDepsgraph:
    return _SignalDepsgraph(tuple(_SignalUpdate(item) for item in id_items))


def _addon_preferences(context: Any | None) -> Any:
    try:
        from . import get_addon_preferences

        return get_addon_preferences(context)
    except Exception:
        return None
