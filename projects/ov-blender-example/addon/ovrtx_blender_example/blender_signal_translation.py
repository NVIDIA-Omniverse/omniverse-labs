# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete translators from Blender signals to add-on payloads."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Any, Callable, Iterable, Mapping

from . import native_exposure
from . import blender_interactive_edit_builders
from . import bundled_runtime
from . import color_presentation
from . import depsgraph_updates
from . import interactive_operator_state
from . import light_scene_layer
from . import materialx_openpbr_conversion
from . import performance_profile
from . import render_requests
from . import runtime_bundle_status
from . import runtime_generation
from . import supersample as supersample_module
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


def material_route_uses_materialx(intent: BlenderRenderIntent) -> bool:
    """Select the material translation route for a render intent.

    Every intent uses the MaterialX/OpenPBR translation, so final renders (F12)
    show the same materials as the viewport. Materials MaterialX rejects keep the
    stock USD export plus UsdPreviewSurface repairs inside the same layer
    (``usd_preview_emission_layer.with_repairs_for_unconverted``). F12 used to be
    held on the preview-only route; the render-init handler in ``engine.py`` is what
    fixed the F12 lock-up, not the route.
    """

    return True


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

    def __init__(self, source: BlenderRenderSignalSource, message: str, *, pending: bool = False) -> None:
        self.source = source
        # A pending translation is waiting on work already scheduled on the
        # main thread (first scene generation); callers show status, not an error.
        self.pending = pending
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

    @performance_profile.timed("signal_translation.render_request")
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
        dlss_quality = str(getattr(settings, "dlss_quality", "INHERIT"))
        # Edge anti-aliasing: render above the output size, F12 averages down
        # (supersample.py). Viewport factor is opt-in; final default is 2.
        supersample = supersample_module.effective_factor(
            width, height,
            getattr(
                settings,
                "viewport_supersample" if signal.intent is BlenderRenderIntent.VIEWPORT else "final_supersample",
                "1" if signal.intent is BlenderRenderIntent.VIEWPORT else "2",
            ),
        )
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
            supersample=supersample,
        )
        camera_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        runtime_inputs_started_ns = phase_started_ns
        # Current-scene requests can update the already-mapped native camera
        # without composing a new USD layer. Preserve its evaluated world pose.
        camera_projection = camera.camera_projection
        scene_camera_matrix = None
        native_scene_camera_matrix = None
        if signal.current_scene_generation and camera.camera_matrix is None:
            # With viewport sync disabled, a current-scene preview still owns
            # the active Blender camera. Carry its authored values in the
            # snapshot so lens/clip/DOF edits invalidate held frames and reach
            # the mapped native camera. Exact-stage requests retain USD authority.
            if signal.intent is BlenderRenderIntent.VIEWPORT:
                camera_projection = render_requests.camera_projection_from_scene_camera(
                    scene, None, camera.width, camera.height)
            native_scene_camera_matrix = render_requests.scene_camera_world_matrix(scene)
        screen = getattr(context, "screen", None)
        timeline_controls_enabled = (
            signal.intent is BlenderRenderIntent.VIEWPORT
            and (
                signal.current_scene_generation
                or os.environ.get("OV_BLENDER_EXAMPLE_SHARED_STAGE", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
        )
        timeline_playing = bool(getattr(screen, "is_animation_playing", False))
        simulation_reset_token = int(getattr(settings, "simulation_reset_token", 0))

        input_usd_path = signal.input_usd_path
        runtime_state_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        phase_started_ns = time.perf_counter_ns()
        worker_command = self._bundled_defaults().worker_command
        native_client_module = ""  # request metadata; no native loader in Blender
        runtime_defaults_ms = (time.perf_counter_ns() - phase_started_ns) / 1_000_000.0
        native_client_preflight_ms = 0.0
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
        native_presentation = None
        known_material_paths = None
        if current_scene_generation:
            from . import scene_generation_sessions, direct_scene_updates
            owner = scene_generation_sessions._owners.get(int(getattr(scene, 'session_uid', 0)))
            generation = getattr(owner, 'current_generation', None)
            if generation is not None and generation.usd_path == input_usd_path:
                native_presentation = direct_scene_updates.presentation(owner, generation)
                # Exact exported material identity; names cannot identify
                # library-linked materials ("Name [lib]" vs prim "Name").
                known_material_paths = {
                    int(identity.session_uid): str(mapping.schema_path)
                    for identity, mapping in getattr(generation, 'blender_prim_paths', {}).items()
                    if getattr(identity, 'kind', '') == 'MATERIAL'
                }
        presentation = (
            usd_preview_emission_layer.presentation_indexes(input_usd_path)
            if (self._include_material_presentation
                and signal.intent is not BlenderRenderIntent.VIEWPORT
                and native_presentation is None
                and self._blender_module_provider() is not None)
            else None
        )
        presentation_index_finished_ns = time.perf_counter_ns()
        material_scene_layer = native_presentation[0] if native_presentation is not None else (
            self._material_scene_layer_from_scene(
                input_usd_path,
                reuse=signal.intent is BlenderRenderIntent.VIEWPORT,
                generation_owned=current_scene_generation,
                use_materialx=material_route_uses_materialx(signal.intent),
                source=signal.source,
                presentation=presentation,
                known_material_paths=known_material_paths,
            )
            if self._include_material_presentation
            else None
        )
        material_finished_ns = time.perf_counter_ns()
        light_scene_layer_value = native_presentation[1] if native_presentation is not None else (
            self._light_scene_layer_from_scene(
                input_usd_path,
                reuse=signal.intent is BlenderRenderIntent.VIEWPORT,
                generation_owned=current_scene_generation,
                presentation=presentation,
            )
            if self._include_material_presentation
            else None
        )
        presentation_finished_ns = time.perf_counter_ns()
        material_ms = (presentation_finished_ns - phase_started_ns) / 1_000_000.0
        presentation_timings = dict(presentation.timings) if presentation is not None else {}
        if presentation is not None:
            presentation_timings.update({
                "presentation_material_conversion_ms": (material_finished_ns - presentation_index_finished_ns) / 1_000_000.0,
                "presentation_light_conversion_ms": (presentation_finished_ns - material_finished_ns) / 1_000_000.0,
            })

        phase_started_ns = time.perf_counter_ns()
        request = RenderRequest(
            input_usd_path=input_usd_path,
            current_scene_generation=current_scene_generation,
            sensor_paths=(render_product_path or DEFAULT_RENDER_PRODUCT_PATH,),
            selected_sensor_paths=(render_product_path or DEFAULT_RENDER_PRODUCT_PATH,),
            width=camera.width,
            height=camera.height,
            supersample=supersample,
            min_samples=min_samples,
            max_samples=max_samples,
            camera_prim_path=camera.camera_prim_path,
            camera_matrix=camera.camera_matrix,
            camera_projection=camera_projection,
            scene_camera_matrix=scene_camera_matrix,
            native_scene_camera_matrix=native_scene_camera_matrix,
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
            semantic_output_path=(str(getattr(getattr(scene, "ovrtx_example", None), "semantic_output_path", ""))
                                  if signal.intent is not BlenderRenderIntent.VIEWPORT else ""),
            native_exposure=native_exposure.from_scene(scene),
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
            dlss_quality=dlss_quality,
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
            **presentation_timings,
            "request_build_ms": request_build_ms,
        }
        return request

    def _timings_snapshot(self) -> Mapping[str, float]:
        return dict(self._last_timings)

    @performance_profile.timed("signal_translation.material_layer")
    def _material_scene_layer_from_scene(
        self,
        input_usd_path: str,
        *,
        reuse: bool = False,
        generation_owned: bool = False,
        use_materialx: bool = True,
        source: BlenderRenderSignalSource,
        presentation: usd_preview_emission_layer.PresentationIndexes | None = None,
        known_material_paths: Mapping[int, str] | None = None,
    ) -> MaterialPresentationLayer | None:
        # Generated paths identify immutable activation state. Supported value
        # edits travel separately; topology replaces the generation/path. Avoid
        # rebuilding a global RNA identity inventory on every navigation draw.
        if (reuse and generation_owned and use_materialx
                and self._material_scene_layer_cache_key is not None
                and self._material_scene_layer_cache_key[0] == input_usd_path):
            return self._material_scene_layer_cache_value
        bpy_module = self._blender_module_provider()
        if bpy_module is None:
            return None
        data = getattr(bpy_module, "data", None)
        materials = tuple(getattr(data, "materials", ()) or ())
        if not use_materialx:
            return usd_preview_emission_layer.scene_layer_from_materials(
                materials,
                input_usd_path,
                **({"presentation": presentation} if presentation is not None else {}),
                **({"known_material_paths": known_material_paths} if known_material_paths else {}),
            )
        cache_key = (input_usd_path, tuple(_blender_identity(material) for material in materials))
        if reuse and cache_key == self._material_scene_layer_cache_key:
            return self._material_scene_layer_cache_value
        result = materialx_openpbr_conversion.scene_layer_from_materials(
            materials,
            input_usd_path,
            allow_stock_fallback=True,
            **({"binding_identity": presentation.binding_identity} if presentation is not None else {}),
            **({"known_material_paths": known_material_paths} if known_material_paths else {}),
        )
        if result.status is materialx_openpbr_conversion.MaterialSceneConversionStatus.ERROR:
            raise BlenderSignalTranslationError(source, str(result.error_reason))
        layer = usd_preview_emission_layer.with_repairs_for_unconverted(
            result,
            materials,
            input_usd_path,
            **({"presentation": presentation} if presentation is not None else {}),
            **({"known_material_paths": known_material_paths} if known_material_paths else {}),
        )
        self._material_scene_layer_cache_key = cache_key
        self._material_scene_layer_cache_value = layer
        return layer

    @performance_profile.timed("signal_translation.light_layer")
    def _light_scene_layer_from_scene(
        self,
        input_usd_path: str,
        *,
        reuse: bool = False,
        generation_owned: bool = False,
        presentation: usd_preview_emission_layer.PresentationIndexes | None = None,
    ) -> MaterialPresentationLayer | None:
        if (reuse and generation_owned
                and self._light_scene_layer_cache_key is not None
                and self._light_scene_layer_cache_key[0] == input_usd_path):
            return self._light_scene_layer_cache_value
        bpy_module = self._blender_module_provider()
        if bpy_module is None:
            return None
        data = getattr(bpy_module, "data", None)
        objects = tuple(getattr(data, "objects", ()) or ())
        light_objects = tuple(obj for obj in objects if str(getattr(obj, "type", "")) == "LIGHT")
        cache_key = (input_usd_path, tuple(_blender_identity(obj) for obj in light_objects))
        if reuse and cache_key == self._light_scene_layer_cache_key:
            return self._light_scene_layer_cache_value
        result = light_scene_layer.scene_layer_from_lights(
            light_objects, input_usd_path,
            **({"light_index": presentation.lights} if presentation is not None else {}),
        )
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
        materials: Any = (),
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
        self._materials = materials
        self._value_edit_conversion_policies = value_edit_conversion_policies
        self._selection_resolution = selection_resolution
        self._selection_resolver = (
            selection_resolver
            or interactive_operator_state.resolve_blender_selection_to_edit_owners
        )
        self.selection_resolution: Mapping[str, Any] | None = None

    @performance_profile.timed("signal_translation.interactive_edit")
    def translate(self, signal: BlenderEditSignal) -> tuple[InteractiveEdit, ...]:
        selection_resolution = self._resolved_selection(signal)
        self.selection_resolution = selection_resolution
        if bool(selection_resolution.get("changed", False)) or bool(selection_resolution.get("group_rejected", False)):
            return ()
        depsgraph = signal.depsgraph if signal.depsgraph is not None else _depsgraph_from_items(signal.id_items)
        updates = signal.updates if signal.depsgraph is not None else None
        return tuple(
            self._edit_builder(
                depsgraph,
                **depsgraph_updates.keyword(self._edit_builder, updates),
                value_edit_conversion_policies=self._value_edit_conversion_policies,
                usd_prim_resolver=self._usd_prim_resolver,
                light_objects=self._light_objects,
                worlds=self._worlds,
                materials=self._materials,
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
