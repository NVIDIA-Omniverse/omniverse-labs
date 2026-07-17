# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concrete adapters from Blender callbacks to add-on-owned signals."""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping

from . import blender_interactive_edit_builders
from . import interactive_operator_state
from .blender_signals import (
    BlenderEditSignal,
    BlenderEditSignalSource,
    BlenderRenderIntent,
    BlenderRenderSignal,
    BlenderRenderSignalSource,
    BlenderSignalTranslator,
)
from .blender_signal_translation import (
    BlenderSignalTranslationError,
    InteractiveEditTranslator,
    RenderRequestTranslator,
)
from .interactive_edit_planner import (
    EditShape,
    InteractiveEdit,
)
from .interactive_edit_workflow import EditWorkflowResult
from .render_requests import RenderRequest
from .properties import DEFAULT_RENDER_PRODUCT_PATH
from .scene_generation import SceneGenerationError, blender_id
from .usd_prim_resolver import UsdPrimResolver
from .value_edit_conversion import ValueEditConversionPolicies


BridgeDiagnosticsRecorder = Callable[..., None]
RenderSignalTranslator = BlenderSignalTranslator[BlenderRenderSignal, RenderRequest]
EditSignalTranslator = BlenderSignalTranslator[BlenderEditSignal, tuple[InteractiveEdit, ...]]
EditTranslatorFactory = Callable[..., EditSignalTranslator]
SceneGenerationProvider = Callable[[Any], Any]
EditObserverContextProvider = Callable[[Any], tuple[UsdPrimResolver | None, Iterable[Any]]]
EditGroupObserver = Callable[
    [Any, Iterable[InteractiveEdit], Mapping[str, Any]],
    Iterable[EditWorkflowResult],
]


class BlenderRenderCallbackAdapter:
    """Build render requests from concrete Blender render callbacks."""

    def __init__(
        self,
        *,
        generation_for_scene: SceneGenerationProvider,
        viewport_generation_for_scene: SceneGenerationProvider | None = None,
        translator: RenderSignalTranslator | None = None,
        engine_id: str = "",
    ) -> None:
        self._generation_for_scene = generation_for_scene
        self._viewport_generation_for_scene = (
            viewport_generation_for_scene or generation_for_scene
        )
        self._translator = translator or RenderRequestTranslator()
        self._engine_id = engine_id

    def final_render(self, depsgraph: Any) -> RenderRequest:
        return self.final_render_from_scene(depsgraph.scene)

    def final_render_from_scene(self, scene: Any) -> RenderRequest:
        return self.translate(
            BlenderRenderSignalSource.FINAL_RENDER,
            BlenderRenderIntent.FINAL_RENDER,
            scene,
        )

    def view_update(self, context: Any, depsgraph: Any) -> RenderRequest:
        return self.view_update_from_scene(depsgraph.scene, context)

    def view_update_from_scene(self, scene: Any, context: Any) -> RenderRequest:
        return self.translate(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            context,
        )

    def view_draw(self, context: Any, depsgraph: Any) -> RenderRequest:
        return self.view_draw_from_scene(depsgraph.scene, context)

    def view_draw_from_scene(self, scene: Any, context: Any) -> RenderRequest:
        return self.translate(
            BlenderRenderSignalSource.VIEW_DRAW,
            BlenderRenderIntent.VIEWPORT,
            scene,
            context,
        )

    def _translation_timings_snapshot(self) -> Mapping[str, float]:
        return self._translator._timings_snapshot()

    def translate(
        self,
        source: BlenderRenderSignalSource,
        intent: BlenderRenderIntent,
        scene: Any,
        context: Any | None = None,
    ) -> RenderRequest:
        return self._translate_scene_generation(source, intent, scene, context)

    def _translate_scene_generation(
        self,
        source: BlenderRenderSignalSource,
        intent: BlenderRenderIntent,
        scene: Any,
        context: Any | None,
    ) -> RenderRequest:
        try:
            provider = (
                self._generation_for_scene
                if source is BlenderRenderSignalSource.FINAL_RENDER
                else self._viewport_generation_for_scene
            )
            generation = provider(scene)
            camera = getattr(scene, "camera", None)
            if camera is None:
                raise SceneGenerationError("Current Blender scene has no active camera.")
            mapping = generation.blender_prim_paths.get(blender_id(camera, "OBJECT"))
            if mapping is None:
                raise SceneGenerationError(
                    f"Scene generation has no mapped prim for active camera {camera.name!r}."
                )
            input_usd_path = generation.materialize_usd()
        except SceneGenerationError as exc:
            raise BlenderSignalTranslationError(source, str(exc)) from exc
        return self._translator.translate(
            BlenderRenderSignal(
                source=source,
                intent=intent,
                scene=scene,
                input_usd_path=input_usd_path,
                camera_prim_path=mapping.schema_path,
                render_product_path=DEFAULT_RENDER_PRODUCT_PATH,
                context=context,
                engine_id=self._engine_id,
                current_scene_generation=True,
            )
        )

class ExactStageRenderCallbackAdapter:
    """Build render requests for explicit internal exact-stage harnesses."""

    def __init__(
        self,
        *,
        input_usd_path: str,
        camera_prim_path: str,
        render_product_path: str,
        translator: RenderSignalTranslator | None = None,
        engine_id: str = "",
    ) -> None:
        self._input_usd_path = str(input_usd_path)
        self._camera_prim_path = str(camera_prim_path)
        self._render_product_path = str(render_product_path)
        self._translator = translator or RenderRequestTranslator()
        self._engine_id = engine_id

    def final_render_from_scene(self, scene: Any) -> RenderRequest:
        return self.translate(
            BlenderRenderSignalSource.FINAL_RENDER,
            BlenderRenderIntent.FINAL_RENDER,
            scene,
        )

    def final_render(self, depsgraph: Any) -> RenderRequest:
        return self.final_render_from_scene(depsgraph.scene)

    def view_update(self, context: Any, depsgraph: Any) -> RenderRequest:
        return self.view_update_from_scene(depsgraph.scene, context)

    def view_draw(self, context: Any, depsgraph: Any) -> RenderRequest:
        return self.view_draw_from_scene(depsgraph.scene, context)

    def view_update_from_scene(self, scene: Any, context: Any) -> RenderRequest:
        return self.translate(
            BlenderRenderSignalSource.VIEW_UPDATE,
            BlenderRenderIntent.VIEWPORT,
            scene,
            context,
        )

    def view_draw_from_scene(self, scene: Any, context: Any) -> RenderRequest:
        return self.translate(
            BlenderRenderSignalSource.VIEW_DRAW,
            BlenderRenderIntent.VIEWPORT,
            scene,
            context,
        )

    def _translation_timings_snapshot(self) -> Mapping[str, float]:
        return self._translator._timings_snapshot()

    def translate(
        self,
        source: BlenderRenderSignalSource,
        intent: BlenderRenderIntent,
        scene: Any,
        context: Any | None = None,
    ) -> RenderRequest:
        return self._translator.translate(
            BlenderRenderSignal(
                source=source,
                intent=intent,
                scene=scene,
                input_usd_path=self._input_usd_path,
                camera_prim_path=self._camera_prim_path,
                render_product_path=self._render_product_path,
                context=context,
                engine_id=self._engine_id,
            )
        )


class BlenderEditCallbackAdapter:
    """Fan out Blender depsgraph callbacks as interactive edit signals."""

    def __init__(
        self,
        *,
        active_engines: Callable[[], Iterable[Any]] | None = None,
        bridge_suppressed: Callable[[], bool] | None = None,
        selection_resolver: Callable[[Any | None], Mapping[str, Any]] | None = None,
        bridge_diagnostics_recorder: BridgeDiagnosticsRecorder | None = None,
        edit_builder: Callable[..., list[InteractiveEdit]] | None = None,
        value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
        edit_translator_factory: EditTranslatorFactory | None = None,
        edit_observer: Callable[[Any, InteractiveEdit], EditWorkflowResult | None]
        | None = None,
        edit_observer_context: EditObserverContextProvider | None = None,
        edit_group_observer: EditGroupObserver | None = None,
    ) -> None:
        self._active_engines = active_engines or (lambda: ())
        self._bridge_suppressed = bridge_suppressed or (lambda: False)
        self._selection_resolver = (
            selection_resolver
            or interactive_operator_state.resolve_blender_selection_to_edit_owners
        )
        self._bridge_diagnostics_recorder = bridge_diagnostics_recorder or (lambda **kwargs: None)
        self._edit_builder = (
            edit_builder
            or blender_interactive_edit_builders.build_interactive_edits_from_depsgraph
        )
        self._value_edit_conversion_policies = value_edit_conversion_policies
        self._edit_translator_factory = edit_translator_factory or InteractiveEditTranslator
        self._edit_observer = edit_observer
        self._edit_observer_context = edit_observer_context
        self._edit_group_observer = edit_group_observer

    def submit_depsgraph_interactive_edits(
        self,
        depsgraph: Any,
        *,
        context: Any | None = None,
        scene: Any | None = None,
    ) -> list[EditWorkflowResult]:
        active_engines = list(self._active_engines())
        active_engine_count = len(active_engines)
        if self._bridge_suppressed():
            self._record_bridge_diagnostics(
                suppressed=True,
                active_engine_count=active_engine_count,
                submitted_edit_count=0,
                result_count=0,
            )
            return []

        selection_resolution = self._selection_resolver(context)
        selection_changed = bool(selection_resolution.get("changed", False))
        selection_rejected = bool(selection_resolution.get("group_rejected", False))
        if self._edit_group_observer is not None:
            if selection_changed:
                self._record_selection_resolution(active_engines, selection_resolution)
                self._record_bridge_diagnostics(
                    suppressed=False,
                    active_engine_count=active_engine_count,
                    submitted_edit_count=0,
                    result_count=0,
                )
                return []
            resolver = None
            light_objects: Any = ()
            if scene is not None and self._edit_observer_context is not None:
                resolver, light_objects = self._edit_observer_context(scene)
            edits = self.translate_depsgraph_edits(
                depsgraph,
                context=context,
                selection_resolution={
                    **selection_resolution,
                    "group_rejected": False,
                },
                usd_prim_resolver=resolver,
                light_objects=light_objects,
            )
            results = self._observe_edits(scene, edits, selection_resolution)
            self._record_bridge_diagnostics(
                suppressed=False,
                active_engine_count=active_engine_count,
                submitted_edit_count=len(edits),
                result_count=len(results),
            )
            return results

        selection_blocks = selection_changed or selection_rejected
        observed_edits: tuple[InteractiveEdit, ...] | None = None
        observed_results: list[EditWorkflowResult] = []
        if (
            scene is not None
            and self._edit_observer_context is not None
            and self._edit_observer is not None
            and (not active_engines or selection_blocks)
        ):
            resolver, light_objects = self._edit_observer_context(scene)
            observed_edits = self.translate_depsgraph_edits(
                depsgraph,
                context=context,
                selection_resolution={
                    **selection_resolution,
                    "changed": False,
                    "group_rejected": False,
                },
                usd_prim_resolver=resolver,
                light_objects=light_objects,
            )
            if (
                not active_engines and selection_blocks
            ):
                observed_results = self._observe_edits(
                    scene, observed_edits, selection_resolution
                )
        if selection_blocks:
            self._record_selection_resolution(active_engines, selection_resolution)
            submitted_edit_count = 0
            if (
                active_engines
                and observed_edits is not None
            ):
                topology_edits = tuple(
                    edit for edit in observed_edits
                    if edit.shape == EditShape.TOPOLOGY
                )
                for engine in active_engines:
                    submitter = getattr(engine, "submit_interactive_edit", None)
                    if not callable(submitter):
                        continue
                    for edit in topology_edits:
                        observed_results.append(submitter(edit))
                        submitted_edit_count += 1
            self._record_bridge_diagnostics(
                suppressed=False,
                active_engine_count=active_engine_count,
                submitted_edit_count=submitted_edit_count,
                result_count=len(observed_results),
            )
            return observed_results

        results: list[EditWorkflowResult] = list(observed_results)
        submitted_edit_count = 0
        if active_engines:
            for engine in active_engines:
                edits = self._edits_for_engine(
                    engine,
                    depsgraph,
                    context=context,
                    selection_resolution=selection_resolution,
                )
                submitted_edit_count += len(edits)
                submitter = getattr(engine, "submit_interactive_edit", None)
                if not callable(submitter):
                    continue
                for edit in edits:
                    results.append(submitter(edit))
        else:
            edits = observed_edits
            if edits is None:
                edits = self.translate_depsgraph_edits(
                    depsgraph,
                    context=context,
                    selection_resolution=selection_resolution,
                )
            results.extend(self._observe_edits(scene, edits, selection_resolution))
            submitted_edit_count = len(edits)

        self._record_bridge_diagnostics(
            suppressed=False,
            active_engine_count=active_engine_count,
            submitted_edit_count=submitted_edit_count,
            result_count=len(results),
        )
        return results

    def _observe_edits(
        self,
        scene: Any | None,
        edits: Iterable[InteractiveEdit],
        selection_resolution: Mapping[str, Any],
    ) -> list[EditWorkflowResult]:
        edits = tuple(edits)
        if scene is None:
            return []
        if self._edit_group_observer is not None:
            return list(
                self._edit_group_observer(scene, edits, selection_resolution)
            )
        if self._edit_observer is None:
            return []
        results = []
        for edit in edits:
            result = self._edit_observer(scene, edit)
            if result is not None:
                results.append(result)
        return results

    def translate_depsgraph_edits(
        self,
        depsgraph: Any,
        *,
        context: Any | None = None,
        selection_resolution: Mapping[str, Any] | None = None,
        input_usd_path: str | None = None,
        ignored_layer_identifiers: Iterable[str] = (),
        usd_prim_resolver: UsdPrimResolver | None = None,
        light_objects: Any = (),
        worlds: Any = (),
        value_edit_conversion_policies: ValueEditConversionPolicies | None = None,
    ) -> tuple[InteractiveEdit, ...]:
        signal = BlenderEditSignal(
            source=BlenderEditSignalSource.DEPSGRAPH,
            id_items=depsgraph_id_items(depsgraph),
            context=context,
            input_usd_path=input_usd_path,
            ignored_layer_identifiers=tuple(ignored_layer_identifiers),
        )
        return self._edit_translator_factory(
            edit_builder=self._edit_builder,
            usd_prim_resolver=usd_prim_resolver,
            light_objects=light_objects,
            worlds=worlds,
            value_edit_conversion_policies=(
                value_edit_conversion_policies
                or self._value_edit_conversion_policies
            ),
            selection_resolution=selection_resolution,
            selection_resolver=self._selection_resolver,
        ).translate(signal)

    def _edits_for_engine(
        self,
        engine: Any,
        depsgraph: Any,
        *,
        context: Any | None,
        selection_resolution: Mapping[str, Any],
    ) -> list[InteractiveEdit]:
        resolver = getattr(engine, "build_interactive_edits_from_depsgraph", None)
        if callable(resolver):
            return list(
                resolver(
                    depsgraph,
                    context=context,
                    selection_resolution=selection_resolution,
                    value_edit_conversion_policies=self._value_edit_conversion_policies,
                    edit_translator_factory=self._edit_translator_factory,
                )
            )
        return list(
            self.translate_depsgraph_edits(
                depsgraph,
                context=context,
                selection_resolution=selection_resolution,
            )
        )

    def _record_selection_resolution(
        self,
        active_engines: Iterable[Any],
        selection_resolution: Mapping[str, Any],
    ) -> None:
        for engine in active_engines:
            recorder = getattr(engine, "record_interactive_selection_resolution", None)
            if not callable(recorder):
                continue
            recorder(selection_resolution)

    def _record_bridge_diagnostics(self, **kwargs: Any) -> None:
        self._bridge_diagnostics_recorder(**kwargs)


def depsgraph_id_items(depsgraph: Any) -> tuple[Any, ...]:
    return tuple(
        getattr(update, "id", update)
        for update in getattr(depsgraph, "updates", ())
    )
