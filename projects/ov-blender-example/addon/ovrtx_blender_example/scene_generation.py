# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private stock-exported USD generations for current Blender scenes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import wraps
from hashlib import sha256
from pathlib import Path
import errno
import glob
import json
import os
import shutil
import tempfile
import threading
import time
from types import MappingProxyType
from typing import Any

import numpy as np

from . import curve_widths, performance_profile
from .usd_opinion_records import (
    AddOnUsdOpinionRecord,
    SceneGenerationError,
    SceneGenerationPending,
    SparseAddOnOpinionChange,
)
from .light_value_conversion import authored_light_form
from .world_dome_conversion import DEFAULT_DOME_OWNER_PATH


try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]


from .object_visibility import GEOMETRY_TYPES, PARTICLE_EMITTER_NAME_ATTRIBUTE
SUPPORTED_OBJECT_TYPES = GEOMETRY_TYPES | {"CAMERA", "LIGHT", "EMPTY"}
MATRIX_OVERRIDE_PROP = "ovrtx.matrix_override"
_OBJECT_UID_ATTRIBUTE = "ovrtx:object_session_uid"
_MESH_UID_ATTRIBUTE = "ovrtx:mesh_session_uid"
_MATERIAL_UID_ATTRIBUTE = "ovrtx:material_session_uid"
_IDENTITY_PROPERTIES = {
    "OBJECT": "ovrtx:object_session_uid",
    "MESH": "ovrtx:mesh_session_uid",
    "MATERIAL": "ovrtx:material_session_uid",
}
_DELTA_COMPACTION_COUNT = 20
_REUSABLE_BASE_SCHEMA = 2
_STOCK_EXPORT_CONTRACT = "blender-usd-v8-retained-mesh-shading-packed-assets-owned-particle-hair-ortho-cameras-ray-visibility"


def _replace_with_retry(source: Path, destination: Path, *, attempts: int = 12, delay: float = 0.5) -> None:
    """Atomically move ``source`` onto ``destination``, retrying transient locks.

    The freshly written candidate generation directory can be briefly held open by
    security software, which makes the directory replace fail with a sharing
    violation (WinError 5) on Windows. Those handles clear on their own, so a
    bounded backoff turns a spurious failure into a short wait.
    """
    for attempt in range(attempts):
        try:
            source.replace(destination)
            return
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay * (attempt + 1))


def _synchronized(function: Any) -> Any:
    @wraps(function)
    def call(self: "SceneGenerationOwner", *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return function(self, *args, **kwargs)

    return call


@dataclass(frozen=True, order=True)
class BlenderId:
    kind: str
    session_uid: int


@dataclass(frozen=True)
class BlenderPrimPath:
    blender_id_name: str
    blender_id_type: str
    object_path: str
    schema_path: str
    data_session_uid: int = 0


@dataclass(frozen=True)
class SceneUpdateProvenance:
    geometry: bool | None = None
    transform: bool | None = None
    shading: bool | None = None
    context: tuple[Any, ...] = ()

    def merged(self, other):
        def union(a, b):
            return True if a is True or b is True else (False if a is False and b is False else None)
        # A context-free observer of visibility carries no evaluated-output
        # change. It is neutral when merged with a real depsgraph notification;
        # unknown flags and conflicting evaluation contexts remain conservative.
        neutral = SceneUpdateProvenance(False, False, False)
        if self == neutral:
            return other
        if other == neutral:
            return self
        if self.context != other.context:
            return SceneUpdateProvenance()
        return SceneUpdateProvenance(union(self.geometry, other.geometry),
            union(self.transform, other.transform), union(self.shading, other.shading), self.context)


@dataclass(frozen=True)
class SceneTopologyDelta:
    usd_path: str
    affected_ids: tuple[BlenderId, ...]
    replaced_prim_paths: tuple[str, ...]
    deleted_prim_paths: tuple[str, ...]
    selected_object_count: int
    digest: str


@dataclass(frozen=True)
class SceneGeneration:
    number: int
    digest: str
    predecessor_number: int | None
    base_digest: str
    usd_path: str
    blender_prim_paths: Mapping[BlenderId, BlenderPrimPath]
    opinion_records: tuple[AddOnUsdOpinionRecord, ...]
    sparse_change: SparseAddOnOpinionChange
    base_usd_path: str
    topology_deltas: tuple[SceneTopologyDelta, ...]
    topology_fingerprints: Mapping[BlenderId, str]
    diagnostics: Mapping[str, Any]
    world_session_uid: int = 0
    unsupported_light_values: Mapping[BlenderId, Mapping[str, Any]] = field(default_factory=dict)
    material_graph_fingerprints: Mapping[BlenderId, str] = field(default_factory=dict)
    mesh_shading_inputs: Mapping[BlenderId, tuple[Any, ...]] = field(default_factory=dict)
    object_visibility: Mapping[BlenderId, bool] = field(default_factory=dict)
    shadow_visibility: Mapping[BlenderId, bool] = field(default_factory=dict)
    scene_object_count: int = -1
    empty_evaluated_mesh_proofs: Mapping[BlenderId, str] = field(default_factory=dict)
    particle_geometry_owners: Mapping[BlenderId, int] = field(default_factory=dict)
    unadmitted_rendering_overrides: tuple[str, ...] = ()

    def materialize_usd(self) -> str:
        return self.usd_path


class SceneGenerationOwner:
    """Create, retain, and clean up stock-exported scene generations."""

    def __init__(
        self,
        work_directory: str | Path,
        reusable_base_directory: str | Path | None = None,
    ) -> None:
        self.work_directory = Path(work_directory).expanduser().resolve()
        self.reusable_base_directory = (
            None
            if reusable_base_directory is None
            else Path(reusable_base_directory).expanduser().resolve()
        )
        self._current: SceneGeneration | None = None
        self._pending: SceneGeneration | None = None
        self._failed: list[SceneGeneration] = []
        self._transform_values: dict[BlenderId, tuple[str, Any]] = {}
        self._attribute_values: dict[tuple[BlenderId, str, str, str], Any] = {}
        self._initial_conditions: dict[BlenderId, tuple[str, Any]] = {}
        self._retained_revision = 0
        self._closed = False
        self._lock = threading.RLock()

    @property
    @_synchronized
    def current_generation(self) -> SceneGeneration | None:
        return self._current

    @property
    @_synchronized
    def retained_revision(self) -> int:
        return self._retained_revision

    @property
    @_synchronized
    def pending_generation(self) -> SceneGeneration | None:
        return self._pending

    @property
    @_synchronized
    def failed_generations(self) -> tuple[SceneGeneration, ...]:
        return tuple(self._failed)

    @performance_profile.timed("scene_generation.replace")
    @_synchronized
    def replace(self, scene: Any, *, force: bool = False) -> SceneGeneration | None:
        if self._closed:
            raise SceneGenerationError("scene generation owner is closed")
        if self._pending is not None:
            raise SceneGenerationError("scene generation candidate is awaiting runtime handoff")
        number = self._next_generation_number()
        self.work_directory.mkdir(parents=True, exist_ok=True)
        candidate = Path(
            tempfile.mkdtemp(
                prefix=f".candidate-{number:06d}-",
                dir=self.work_directory,
            )
        )
        started = time.perf_counter()
        timings: dict[str, float] = {}
        cache_status = "disabled"
        cache_reason = "reusable_base_directory_unavailable"
        lookup_key = ""
        publication: tuple[Path, dict[str, Any], Path, bool] | None = None
        try:
            phase = time.perf_counter()
            lookup_key, cache_reason = _source_lookup(scene)
            timings["fingerprint_ms"] = (time.perf_counter() - phase) * 1000.0
            cache_path = (
                None
                if not lookup_key or self.reusable_base_directory is None
                else self.reusable_base_directory / lookup_key
            )
            phase = time.perf_counter()
            snapshot = candidate / "stock-base"
            cached = None
            rejection_reason = ""
            if cache_path is not None and cache_path.is_dir():
                try:
                    # Copy before validation so the accepted digest
                    # describes session-owned bytes, not mutable shared state.
                    shutil.copytree(cache_path, snapshot)
                    cached, rejection_reason = _load_reusable_base(
                        scene,
                        snapshot,
                        lookup_key,
                    )
                except OSError as exc:
                    rejection_reason = f"snapshot_failed:{type(exc).__name__}"
            timings["lookup_ms"] = (time.perf_counter() - phase) * 1000.0
            if cached is not None:
                phase = time.perf_counter()
                confirmed_lookup, confirmed_reason = _source_lookup(scene)
                timings["fingerprint_ms"] += (
                    time.perf_counter() - phase
                ) * 1000.0
                if confirmed_lookup != lookup_key:
                    cached = None
                    rejection_reason = (
                        confirmed_reason or "source_changed_during_lookup"
                    )
            if cached is None:
                shutil.rmtree(snapshot, ignore_errors=True)
            if not lookup_key:
                cache_status = "ineligible"
            elif self.reusable_base_directory is None:
                cache_status = "disabled"
            elif cached is not None:
                base_path, base_digest, mappings = cached
                cache_status = "hit"
                cache_reason = ""
            else:
                cache_status = "miss"
                cache_reason = (
                    rejection_reason
                    if rejection_reason
                    else "cache_absent"
                )

            if cached is None:
                phase = time.perf_counter()
                stock_directory = candidate / "stock-export"
                stock_directory.mkdir()
                base_path = stock_directory / "scene.usdc"
                _stock_export(scene, base_path)
                mappings = _validated_blender_prim_paths(scene, base_path)
                _remove_stock_export_identities(base_path)
                _validate_identity_free_base(base_path)
                artifact_digests = _artifact_digests(stock_directory)
                base_digest = _stock_base_digest(artifact_digests)
                timings["export_ms"] = (time.perf_counter() - phase) * 1000.0
                if lookup_key and self.reusable_base_directory is not None:
                    manifest = _reusable_base_manifest(
                        scene,
                        lookup_key,
                        mappings,
                        artifact_digests,
                    )
                    publication = (
                        stock_directory,
                        manifest,
                        self.reusable_base_directory / lookup_key,
                        bool(rejection_reason),
                    )
            else:
                timings["export_ms"] = 0.0

            phase = time.perf_counter()
            binding_path = candidate / "identity.usda"
            _write_identity_binding(binding_path, mappings)
            stock_root = candidate / "stock.usda"
            _write_stock_root(stock_root, binding_path, base_path)
            timings["composition_ms"] = (time.perf_counter() - phase) * 1000.0

            phase = time.perf_counter()
            records = _compile_add_on_opinions(scene, mappings, stock_root)
            timings["opinion_compilation_ms"] = (time.perf_counter() - phase) * 1000.0
            digest = _generation_digest(
                base_digest,
                _mapping_identity_digest(mappings),
                records,
            )
            world_session_uid = _world_session_uid(scene)
            if (
                not force and self._current is not None
                and self._current.base_digest == base_digest
                and dict(self._current.blender_prim_paths) == mappings
                and getattr(self._current, "world_session_uid", 0)
                == world_session_uid
                and tuple(record.digest for record in self._current.opinion_records)
                == tuple(record.digest for record in records)
            ):
                shutil.rmtree(candidate, ignore_errors=True)
                return None
            destination = self.work_directory / f"generation-{number:06d}-{digest[:16]}"
            phase = time.perf_counter()
            composed_stage = _write_layered_generation(candidate / "composed.usda", records, (), stock_root)
            timings["composition_ms"] += (time.perf_counter() - phase) * 1000.0
            phase = time.perf_counter()
            _validate_composed_generation(
                candidate / "composed.usda", mappings, set(), set(),
                **({"stage": composed_stage} if hasattr(composed_stage, "GetPrimAtPath") else {})
            )
            composed_stage = None
            timings["validation_ms"] = (time.perf_counter() - phase) * 1000.0
            if publication is not None:
                phase = time.perf_counter()
                current_lookup, current_reason = _source_lookup(scene)
                timings["fingerprint_ms"] += (
                    time.perf_counter() - phase
                ) * 1000.0
                if current_lookup != lookup_key:
                    publication = None
                    cache_status = "publication_skipped"
                    cache_reason = current_reason or "source_changed_during_export"
            if publication is not None:
                try:
                    published = _publish_reusable_base(*publication)
                    reused, rejection_reason = _load_reusable_base(
                        scene, published, lookup_key
                    )
                    if reused is None:
                        cache_status = "rejected_fresh"
                        cache_reason = rejection_reason
                    else:
                        cache_status = "published"
                        cache_reason = ""
                except Exception as exc:
                    # Cache publication is optional; the validated fresh generation wins.
                    cache_status = "publication_failed"
                    cache_reason = f"{type(exc).__name__}: {exc}"
            candidate.replace(destination)
        except Exception:
            shutil.rmtree(candidate, ignore_errors=True)
            raise
        generation = SceneGeneration(
            number=number,
            digest=digest,
            predecessor_number=(
                None if self._current is None else self._current.number
            ),
            base_digest=base_digest,
            usd_path=str(destination / "composed.usda"),
            blender_prim_paths=MappingProxyType(dict(mappings)),
            opinion_records=records,
            sparse_change=_opinion_change(self._current, records),
            base_usd_path=str(destination / "stock.usda"),
            topology_deltas=(),
            topology_fingerprints=MappingProxyType(_topology_fingerprints(scene)),
            empty_evaluated_mesh_proofs=MappingProxyType(_empty_evaluated_mesh_proofs(scene)),
            particle_geometry_owners=MappingProxyType(_particle_geometry_owners(scene)),
            unadmitted_rendering_overrides=_unadmitted_rendering_overrides(scene),
            mesh_shading_inputs=MappingProxyType(_mesh_shading_inputs_for_scene(scene)),
            scene_object_count=len(getattr(scene, "objects", ())),
            unsupported_light_values=MappingProxyType(_unsupported_light_values(scene)),
            material_graph_fingerprints=MappingProxyType(_material_graph_fingerprints(scene, mappings)),
            diagnostics=MappingProxyType(
                {
                    "mode": "complete_export",
                    "physics_initial_frame": float(getattr(scene, "frame_current", 1)) + float(getattr(scene, "frame_subframe", 0)),
                    "physics_initial_fps": float(getattr(getattr(scene, "render", None), "fps", 24)) / float(getattr(getattr(scene, "render", None), "fps_base", 1)),
                    "complete_export": True,
                    "selected_objects_only": False,
                    "reusable_base_status": cache_status,
                    "reusable_base_lookup": lookup_key,
                    "reusable_base_reason": cache_reason,
                    **timings,
                    "total_ms": (time.perf_counter() - started) * 1000.0,
                }
            ),
            world_session_uid=world_session_uid,
            object_visibility=MappingProxyType(_object_visibility(scene)),
            shadow_visibility=MappingProxyType(_mesh_shadow_visibility(scene)),
        )
        self._current = generation
        return generation

    @performance_profile.timed("scene_generation.reconcile")
    @_synchronized
    def reconcile(
        self,
        scene: Any,
        affected_ids: set[BlenderId] | frozenset[BlenderId] | tuple[BlenderId, ...],
        *, update_provenance: Mapping[BlenderId, SceneUpdateProvenance] | None = None,
    ) -> SceneGeneration | None:
        if self._closed:
            raise SceneGenerationError("scene generation owner is closed")
        if self._pending is not None:
            raise SceneGenerationError("scene generation candidate is awaiting runtime handoff")
        if self._current is None:
            return self.replace(scene)
        affected = tuple(sorted(set(affected_ids)))
        if not affected:
            return None
        from . import direct_scene_updates
        particle_owner_changed = _particle_geometry_owner_changed(scene, self._current, affected)
        if not particle_owner_changed and direct_scene_updates.plan(self, scene, self._current, affected, update_provenance or {}):
            return None
        affected = tuple(sorted(direct_scene_updates.fallback_ids(self, affected)))
        if (particle_owner_changed or any(identity.kind == "PHYSICS" for identity in affected)
                or (any(identity.kind == "WORLD" for identity in affected)
                    and not _can_reconcile_canonical_world(scene, self._current))
                or _shadow_visibility_changed(scene, self._current, affected)
                or _visibility_requires_full_export(scene, self._current)
                or _expanded_visibility_owner_changed(scene, affected)):
            predecessor = self._current
            generation = self.replace(scene)
            if generation is not None:
                self._current = predecessor
                self._pending = generation
            return generation
        object_ids = _affected_object_ids(scene, self._current, affected)
        proof_started = time.perf_counter()
        proven_non_geometry = _proven_non_geometry_objects(scene, self._current, affected,
                                                          object_ids, update_provenance or {})
        # Mixed edits keep proven visibility-only owners out of geometry export
        # even when another owner needs population. Visibility opinions still
        # travel with the candidate; unsupported geometry does not erase them.
        direct_state = getattr(self, '_direct_state', None)
        if direct_state is not None:
            proven_non_geometry.update(direct_state.visibility_only_ids & object_ids)
        admission_ms = (time.perf_counter() - proof_started) * 1000.0
        hash_started = time.perf_counter()
        current_fingerprints = _topology_fingerprints(scene, object_ids - proven_non_geometry)
        current_fingerprints.update({identity: self._current.topology_fingerprints[identity]
                                     for identity in proven_non_geometry})
        topology_hash_ms = (time.perf_counter() - hash_started) * 1000.0
        empty_proof_started = time.perf_counter()
        empty_proofs = _empty_evaluated_mesh_proofs(scene, object_ids)
        empty_proof_ms = (time.perf_counter() - empty_proof_started) * 1000.0
        previous_proofs = getattr(self._current, "empty_evaluated_mesh_proofs", {})
        suppressed = []
        for identity, signature in empty_proofs.items():
            if previous_proofs.get(identity) == signature and identity in self._current.topology_fingerprints:
                if current_fingerprints.get(identity) != self._current.topology_fingerprints[identity]:
                    suppressed.append(identity)
                current_fingerprints[identity] = self._current.topology_fingerprints[identity]
        proof_ms = (time.perf_counter() - proof_started) * 1000.0
        if _is_noop_topology_edit(self._current, affected, current_fingerprints):
            if _object_visibility(scene) == dict(getattr(self._current, "object_visibility", {})):
                return None
            generation = _reconcile_visibility_generation(self.work_directory, scene, self._current, number=self._next_generation_number())
            self._pending = generation
            return generation
        # Blender can notify unchanged objects alongside an actual mesh edit.
        # The same fingerprints used by the all-no-op gate prove which object
        # notifications need stock geometry; retain shader and unknown IDs.
        affected = _changed_topology_notifications(self._current, affected, current_fingerprints)
        generation = _reconcile_sparse_generation(
            self.work_directory,
            scene,
            self._current,
            affected,
            number=self._next_generation_number(),
        )
        if isinstance(generation, SceneGeneration):
            generation = replace(generation, diagnostics=MappingProxyType({
                **dict(generation.diagnostics), "reconciliation_fingerprint_ms": proof_ms,
                "reconciliation_object_count": len(object_ids),
                "reconciliation_geometry_admitted_count": len(object_ids - proven_non_geometry),
                "reconciliation_geometry_excluded_count": len(proven_non_geometry),
                "reconciliation_provenance_admission_ms": admission_ms,
                "reconciliation_topology_hash_ms": topology_hash_ms,
                "reconciliation_empty_control_proof_ms": empty_proof_ms,
                "unchanged_empty_output_objects": tuple(identity.session_uid for identity in suppressed),
            }))
        self._pending = generation
        return generation

    @_synchronized
    def reconcile_final_frame(self, scene: Any) -> SceneGeneration:
        """Capture an evaluated animation frame without retiring its predecessor.

        Render Animation does not deliver ordinary viewport dirty notifications.
        A changed frame therefore requires stock export even when topology is
        unchanged. Its authored values supersede retained edits only after ACK.
        """
        frame = float(getattr(scene, "frame_current", 1)) + float(getattr(scene, "frame_subframe", 0))
        previous = self.reuse()
        captured = previous.diagnostics.get("final_render_frame", previous.diagnostics.get("physics_initial_frame", frame))
        if captured == frame:
            return previous
        if self._pending is not None:
            raise SceneGenerationError("Previous final frame is awaiting runtime handoff")
        retained = (dict(self._transform_values), dict(self._attribute_values), dict(self._initial_conditions), self._retained_revision)
        try:
            candidate = self.replace(scene, force=True)
        finally:
            # Export may cause Blender depsgraph callbacks; their values must
            # not modify the accepted predecessor before candidate admission.
            self._transform_values, self._attribute_values, self._initial_conditions, self._retained_revision = retained
            self._current = previous
        assert candidate is not None
        candidate = replace(candidate, diagnostics=MappingProxyType({
            **candidate.diagnostics, "final_render_frame": frame,
            "final_frame_replaces_retained_values": True}))
        self._pending = candidate
        return candidate

    @_synchronized
    def accept(self, generation: SceneGeneration) -> None:
        if generation is not self._pending:
            raise SceneGenerationError("scene generation candidate is not pending")
        self._current = generation
        self._pending = None
        if getattr(generation, "diagnostics", {}).get("final_frame_replaces_retained_values"):
            self._transform_values.clear()
            self._attribute_values.clear()
        current_ids = set(generation.blender_prim_paths)
        self._transform_values = {
            identity: value
            for identity, value in self._transform_values.items()
            if identity in current_ids
        }
        self._attribute_values = {
            key: value
            for key, value in self._attribute_values.items()
            if _retained_attribute_mapping_identities(generation, key[0])
        }
        self._initial_conditions = {
            identity: value
            for identity, value in self._initial_conditions.items()
            if identity in current_ids
        }

    @_synchronized
    def reject(self, generation: SceneGeneration) -> None:
        if generation is not self._pending:
            raise SceneGenerationError("scene generation candidate is not pending")
        self._failed.append(generation)
        self._pending = None

    @_synchronized
    def reuse(self) -> SceneGeneration:
        if self._closed:
            raise SceneGenerationError("scene generation owner is closed")
        if self._pending is not None:
            return self._pending
        if self._current is None:
            raise SceneGenerationError("scene generation is unavailable")
        return self._current

    def _next_generation_number(self) -> int:
        return 1 + max(
            (
                generation.number
                for generation in (*self._failed, self._current)
                if generation is not None
            ),
            default=-1,
        )

    @_synchronized
    def retain_transform_values(self, values: Any) -> None:
        generation = self._require_current_generation()
        for value in values:
            target = _mapped_value_target(generation, value.prim_path)
            if target is None:
                continue
            identity, field, _suffix = target
            previous = self._transform_values.get(identity)
            if not _retained_values_equal(previous, (field, value)):
                self._retained_revision += 1
            self._transform_values[identity] = (field, value)

    @_synchronized
    def retain_attribute_values(self, values: Any) -> None:
        generation = self._require_current_generation()
        for value in values:
            target = _mapped_value_target(
                generation,
                value.prim_path,
                allow_descendant=True,
            )
            if target is None:
                continue
            identity, field, suffix = target
            identity = _retained_attribute_identity(generation, identity)
            stable_suffix = "*" if identity.kind == "MATERIAL" and suffix else suffix
            key = (identity, field, stable_suffix, value.attribute)
            if not _retained_values_equal(self._attribute_values.get(key), value):
                self._retained_revision += 1
            self._attribute_values[key] = value

    @_synchronized
    def retain_initial_conditions(self, values: Any) -> None:
        generation = self._require_current_generation()
        for value in values:
            target = _mapped_value_target(generation, value.prim_path)
            if target is None:
                continue
            identity, field, _suffix = target
            self._initial_conditions[identity] = (field, value)

    @_synchronized
    def retained_values_for(
        self,
        generation: SceneGeneration,
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
        if (generation is self._pending
                and getattr(generation, "diagnostics", {}).get("final_frame_replaces_retained_values")):
            return (), (), ()
        mappings = generation.blender_prim_paths
        transforms = tuple(
            replace(value, prim_path=getattr(mappings[identity], field))
            for identity, (field, value) in sorted(self._transform_values.items())
            if identity in mappings
        )
        material_stage: list[Any] = []
        attributes = []
        for (identity, field, suffix, _attribute), value in sorted(
            self._attribute_values.items()
        ):
            for mapping_identity in _retained_attribute_mapping_identities(
                generation, identity
            ):
                attributes.append(
                    replace(
                        value,
                        prim_path=_retained_attribute_path(
                            generation,
                            mapping_identity,
                            field,
                            suffix,
                            value.attribute,
                            material_stage,
                        ),
                    )
                )
        initial_conditions = tuple(
            replace(value, prim_path=getattr(mappings[identity], field))
            for identity, (field, value) in sorted(self._initial_conditions.items())
            if identity in mappings
        )
        return transforms, tuple(attributes), initial_conditions

    def _require_current_generation(self) -> SceneGeneration:
        if self._closed or self._current is None:
            raise SceneGenerationError("scene generation is unavailable for retained values")
        return self._current

    @_synchronized
    def close(self, *, defer_residue: bool = False) -> Path | None:
        """Remove the work directory and release every retained generation.

        Transient Windows sharing errors are retried with a bounded backoff.
        If they outlast it and ``defer_residue`` is true, the owner still
        closes and the directory is returned for later removal; otherwise any
        failure raises and the owner keeps its state.
        """
        if self._closed:
            return None
        residue = None
        try:
            _remove_tree_with_retry(self.work_directory)
        except OSError as exc:
            if not (defer_residue and _is_transient_cleanup_error(exc)):
                raise SceneGenerationError(
                    f"scene generation cleanup failed: {exc}",
                    ({"work_directory": str(self.work_directory), "reason": "cleanup_failed"},),
                ) from exc
            residue = self.work_directory
        self._current = None
        self._pending = None
        self._failed.clear()
        self._transform_values.clear()
        self._attribute_values.clear()
        self._initial_conditions.clear()
        self._closed = True
        return residue


# Windows reports a directory whose files are still open elsewhere (typically a
# real-time scanner reading a freshly exported texture) as access denied (5),
# sharing violation (32) or directory not empty (145, a delete-pending entry).
# These clear once the other handle closes, so cleanup retries them briefly.
_TRANSIENT_CLEANUP_WINERRORS = frozenset({5, 32, 145})
_TRANSIENT_CLEANUP_ERRNOS = frozenset({errno.ENOTEMPTY, errno.EBUSY})
CLEANUP_RETRY_DELAYS_SECONDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)
_cleanup_sleep = time.sleep


def _is_transient_cleanup_error(exc: BaseException) -> bool:
    winerror = getattr(exc, "winerror", None)
    if winerror is not None:
        return winerror in _TRANSIENT_CLEANUP_WINERRORS
    return isinstance(exc, OSError) and exc.errno in _TRANSIENT_CLEANUP_ERRNOS


def _remove_tree_with_retry(path: Path) -> None:
    """``shutil.rmtree`` with a bounded backoff for transient sharing errors."""
    delays = iter(CLEANUP_RETRY_DELAYS_SECONDS)
    while True:
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            delay = next(delays, None) if _is_transient_cleanup_error(exc) else None
            if delay is None:
                raise
        _cleanup_sleep(delay)


def _mapped_value_target(
    generation: SceneGeneration,
    prim_path: str,
    *,
    allow_descendant: bool = False,
) -> tuple[BlenderId, str, str] | None:
    candidates: list[tuple[BlenderId, str, str]] = []
    for identity, mapping in generation.blender_prim_paths.items():
        if mapping.schema_path == prim_path:
            candidates.append((identity, "schema_path", ""))
        elif mapping.object_path == prim_path:
            candidates.append((identity, "object_path", ""))
        elif allow_descendant and prim_path.startswith(mapping.schema_path + "/"):
            candidates.append(
                (identity, "schema_path", prim_path[len(mapping.schema_path) :])
            )
    if not candidates:
        return None
    shortest_suffix = min(len(candidate[2]) for candidate in candidates)
    candidates = [
        candidate for candidate in candidates if len(candidate[2]) == shortest_suffix
    ]
    if len(candidates) != 1:
        raise SceneGenerationError(
            f"runtime value target is not a unique current Blender identity: {prim_path}"
        )
    return candidates[0]


def _retained_attribute_identity(
    generation: SceneGeneration,
    identity: BlenderId,
) -> BlenderId:
    mapping = generation.blender_prim_paths.get(identity)
    if (
        mapping is not None
        and mapping.blender_id_type == "LIGHT"
        and mapping.data_session_uid > 0
    ):
        return BlenderId("LIGHT", mapping.data_session_uid)
    return identity


def _retained_attribute_mapping_identities(
    generation: SceneGeneration,
    identity: BlenderId,
) -> tuple[BlenderId, ...]:
    if identity.kind != "LIGHT":
        return (identity,) if identity in generation.blender_prim_paths else ()
    return tuple(
        candidate
        for candidate, mapping in sorted(generation.blender_prim_paths.items())
        if mapping.blender_id_type == "LIGHT"
        and mapping.data_session_uid == identity.session_uid
    )


def _retained_attribute_path(
    generation: SceneGeneration,
    identity: BlenderId,
    field: str,
    suffix: str,
    attribute: str,
    material_stage: list[Any],
) -> str:
    root = getattr(generation.blender_prim_paths[identity], field)
    if identity.kind != "MATERIAL" or not suffix:
        return root + suffix
    from pxr import Usd  # type: ignore

    if not material_stage:
        material_stage.append(Usd.Stage.Open(generation.materialize_usd()))
    stage = material_stage[0]
    if stage is None:
        raise SceneGenerationError("retained material value stage is unavailable")
    candidates = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if (
            (str(prim.GetPath()) == root or str(prim.GetPath()).startswith(root + "/"))
            and prim.HasAttribute(attribute)
        )
    ]
    if len(candidates) != 1:
        raise SceneGenerationError(
            f"retained material value target is not unique: {root}.{attribute}"
        )
    return candidates[0]


def _blender_module_provider() -> Any:
    return bpy


def _source_lookup(scene: Any) -> tuple[str, str]:
    module = _blender_module_provider()
    data = getattr(module, "data", None)
    source = Path(str(getattr(data, "filepath", "") or ""))
    if data is None:
        return "", "blender_data_unavailable"
    if bool(getattr(data, "is_dirty", True)):
        return "", "dirty_source"
    if not source.is_file():
        return "", "unsaved_or_missing_source"
    dependencies = _external_source_paths(module)
    if dependencies is None:
        return "", "external_input_unreadable"
    files = [source.resolve(), *dependencies]
    try:
        file_records = [
            {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
            for path in files
        ]
    except OSError:
        return "", "source_unreadable"
    from .object_visibility import scene_view_layer
    app = getattr(module, "app", None)
    contract = {
        "schema": _REUSABLE_BASE_SCHEMA,
        "export_contract": _STOCK_EXPORT_CONTRACT,
        "shadow_visibility": sorted((str(getattr(obj, "name_full", "")), bool(getattr(obj, "visible_shadow", True)), bool(getattr(obj, "visible_camera", True))) for obj in getattr(scene, "objects", ()) if getattr(obj, "type", "") == "MESH"),
        "blender": {
            "version": tuple(getattr(app, "version", ())),
            "version_cycle": str(getattr(app, "version_cycle", "")),
            "build_hash": _text_value(getattr(app, "build_hash", "")),
        },
        "source_scene": str(getattr(scene, "name_full", getattr(scene, "name", ""))),
        "frame": int(getattr(scene, "frame_current", 0)),
        "subframe": float(getattr(scene, "frame_subframe", 0.0)),
        "meters_per_blender_unit": float(getattr(getattr(scene, "unit_settings", None), "scale_length", 1.0)),
        "view_layer": str(getattr(
            scene_view_layer(scene, module),
            "name", "")),
        "files": file_records,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest(), ""


def _external_source_paths(module: Any) -> list[Path] | None:
    data = getattr(module, "data", None)
    paths: set[Path] = set()
    values = [
        (str(getattr(library, "filepath", "") or ""), library)
        for library in getattr(data, "libraries", ())
    ]
    values.extend(
        (str(getattr(image, "filepath", "") or ""), image)
        for image in getattr(data, "images", ())
        if not getattr(image, "packed_file", None)
        and not tuple(getattr(image, "packed_files", ()) or ())
        and str(getattr(image, "source", "")) not in {"GENERATED", "VIEWER"}
        and str(getattr(image, "filepath", "") or "")
    )
    for collection_name in (
        "cache_files",
        "fonts",
        "movieclips",
        "sounds",
        "volumes",
    ):
        values.extend(
            (str(getattr(value, "filepath", "") or ""), value)
            for value in getattr(data, collection_name, ())
            if str(getattr(value, "filepath", "") or "") not in {"", "<builtin>"}
            and not getattr(value, "packed_file", None)
        )
    if any(
        str(getattr(modifier, "type", "")) in {"FLUID", "MESH_CACHE", "OCEAN"}
        for obj in getattr(data, "objects", ())
        for modifier in getattr(obj, "modifiers", ())
    ):
        # These use cache directories/files outside Blender's shared
        # CacheFile datablocks; add exact closure only when reuse is measured there.
        return None
    if any(
        str(getattr(owner, "source", "")) == "SEQUENCE"
        or bool(getattr(owner, "is_sequence", False))
        for _, owner in values
    ):
        # Numbered sequence closure is frame-dependent; enumerate it
        # only when reusable sequence scenes are a measured requirement.
        return None
    abspath = getattr(getattr(module, "path", None), "abspath", None)
    for raw, owner in values:
        if not raw or not callable(abspath):
            return None
        try:
            resolved = str(abspath(raw, library=getattr(owner, "library", None)))
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        pattern = resolved.replace("<UDIM>", "*")
        if "#" in pattern:
            pattern = pattern.replace("#", "*")
        matches = [Path(path).resolve() for path in glob.glob(pattern)]
        if not matches or any(not path.is_file() for path in matches):
            return None
        paths.update(matches)
    return sorted(paths)


def _text_value(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _stable_blender_id(value: Any, kind: str) -> dict[str, str]:
    library = getattr(value, "library", None)
    return {
        "kind": kind,
        "library": str(getattr(library, "filepath", "") or ""),
        "name": str(getattr(value, "name_full", getattr(value, "name", ""))),
    }


def _reusable_base_manifest(
    scene: Any,
    lookup_key: str,
    mappings: Mapping[BlenderId, BlenderPrimPath],
    artifact_digests: Mapping[str, str],
) -> dict[str, Any]:
    module = _blender_module_provider()
    objects = {
        _session_uid(obj, "OBJECT"): obj for obj in getattr(scene, "objects", ())
    }
    materials = {
        _session_uid(material, "MATERIAL"): material
        for material in getattr(getattr(module, "data", None), "materials", ())
    }
    entries = []
    for identity, mapping in sorted(mappings.items()):
        if identity.kind == "OBJECT":
            value = objects[identity.session_uid]
            data = getattr(value, "data", None)
        elif identity.kind == "MATERIAL":
            value = materials[identity.session_uid]
            data = None
        elif identity.kind == "WORLD":
            value = getattr(scene, "world", None)
            data = None
        else:
            raise SceneGenerationError(
                f"unsupported reusable Blender ID kind {identity.kind}"
            )
        entry = {
            "identity": _stable_blender_id(value, identity.kind),
            "blender_id_type": mapping.blender_id_type,
            "object_path": mapping.object_path,
            "schema_path": mapping.schema_path,
        }
        if mapping.data_session_uid:
            entry["data_identity"] = _stable_blender_id(data, mapping.blender_id_type)
        entries.append(entry)
    return {
        "schema": _REUSABLE_BASE_SCHEMA,
        "export_contract": _STOCK_EXPORT_CONTRACT,
        "lookup_key": lookup_key,
        "files": dict(artifact_digests),
        "mappings": entries,
    }


def _artifact_digests(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _stock_base_digest(artifact_digests: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(sorted(artifact_digests.items())),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(b"stock-base-v1\0" + encoded).hexdigest()


def _publish_reusable_base(
    stock_directory: Path,
    manifest: Mapping[str, Any],
    destination: Path,
    replace_rejected: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_dir() and not replace_rejected:
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".reusable-base-", dir=destination.parent))
    rejected: Path | None = None
    try:
        if destination.is_dir() and not replace_rejected:
            return destination
        shutil.copytree(stock_directory, temporary, dirs_exist_ok=True)
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "manifest.sha256").write_text(
            sha256(manifest_bytes).hexdigest(),
            encoding="ascii",
        )
        if destination.is_dir():
            rejected = Path(
                tempfile.mkdtemp(prefix=".rejected-base-", dir=destination.parent)
            )
            rejected.rmdir()
            try:
                destination.replace(rejected)
            except FileNotFoundError:
                rejected = None
        try:
            temporary.replace(destination)
        except OSError:
            if not destination.is_dir():
                raise
        if rejected is not None:
            shutil.rmtree(rejected, ignore_errors=True)
    except Exception:
        if rejected is not None and rejected.is_dir() and not destination.exists():
            rejected.replace(destination)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
        if rejected is not None and rejected.is_dir():
            shutil.rmtree(rejected, ignore_errors=True)
    return destination


def _load_reusable_base(
    scene: Any,
    directory: Path,
    lookup_key: str,
) -> tuple[
    tuple[Path, str, dict[BlenderId, BlenderPrimPath]] | None,
    str,
]:
    try:
        manifest_bytes = (directory / "manifest.json").read_bytes()
        expected_manifest_digest = (directory / "manifest.sha256").read_text(
            encoding="ascii"
        )
        if sha256(manifest_bytes).hexdigest() != expected_manifest_digest:
            raise ValueError("manifest_digest_mismatch")
        manifest = json.loads(manifest_bytes)
        if (
            manifest.get("schema") != _REUSABLE_BASE_SCHEMA
            or manifest.get("export_contract") != _STOCK_EXPORT_CONTRACT
            or manifest.get("lookup_key") != lookup_key
        ):
            raise ValueError("incompatible_manifest")
        expected = manifest.get("files")
        actual = _artifact_digests(directory)
        actual.pop("manifest.json", None)
        actual.pop("manifest.sha256", None)
        if not isinstance(expected, dict) or actual != expected:
            raise ValueError("artifact_digest_mismatch")
        base_path = directory / "scene.usdc"
        base_digest = _stock_base_digest(actual)
        mappings = _resolve_reusable_mappings(scene, manifest["mappings"])
        return (base_path, base_digest, mappings), ""
    except Exception as exc:
        # Cache reuse is optional; malformed USD and provider errors fail closed.
        return None, f"{type(exc).__name__}: {exc}"


def _resolve_reusable_mappings(
    scene: Any,
    entries: Any,
) -> dict[BlenderId, BlenderPrimPath]:
    module = _blender_module_provider()
    candidates: list[tuple[str, Any]] = [
        ("OBJECT", obj) for obj in getattr(scene, "objects", ())
    ]
    candidates.extend(
        ("MATERIAL", material)
        for material in getattr(getattr(module, "data", None), "materials", ())
    )
    world = getattr(scene, "world", None)
    if world is not None:
        candidates.append(("WORLD", world))
    by_stable: dict[str, list[Any]] = {}
    for kind, value in candidates:
        key = json.dumps(_stable_blender_id(value, kind), sort_keys=True)
        by_stable.setdefault(key, []).append(value)
    mappings: dict[BlenderId, BlenderPrimPath] = {}
    for entry in entries:
        key = json.dumps(entry["identity"], sort_keys=True)
        matches = by_stable.get(key, [])
        if len(matches) != 1:
            raise SceneGenerationError("cached Blender identity is not unique")
        value = matches[0]
        kind = str(entry["identity"]["kind"])
        identity = blender_id(value, kind)
        data_uid = 0
        if "data_identity" in entry:
            data = getattr(value, "data", None)
            if (
                _stable_blender_id(data, str(entry["blender_id_type"]))
                != entry["data_identity"]
            ):
                raise SceneGenerationError("cached Blender data identity changed")
            data_uid = _session_uid(data, str(entry["blender_id_type"]))
        mapping = BlenderPrimPath(
            str(getattr(value, "name_full", getattr(value, "name", ""))),
            str(entry["blender_id_type"]),
            str(entry["object_path"]),
            str(entry["schema_path"]),
            data_uid,
        )
        if identity in mappings:
            raise SceneGenerationError("cached Blender identity is duplicated")
        mappings[identity] = mapping
    return mappings



def _unadmitted_rendering_overrides(scene):
    from .object_matte import opinions
    return tuple(str(getattr(obj, 'name', '<object>')) for obj in getattr(scene, 'objects', ())
                 if opinions(obj))


def _particle_geometry_owners(scene):
    return {blender_id(obj, "OBJECT"): int(getattr(getattr(obj, "data", None), "session_uid", 0))
            for obj in getattr(scene, "objects", ()) if getattr(obj, "particle_systems", ())}


def _particle_geometry_owner_changed(scene, generation, affected):
    # Stock particle conversion creates sibling roots, outside the emitter's
    # sparse replacement path. Keep the accepted ownership after the modifier
    # or emitter is removed so its old generated geometry is retired as well.
    owners = dict(getattr(generation, "particle_geometry_owners", {}))
    owners.update(_particle_geometry_owners(scene))
    mesh_ids = {identity.session_uid for identity in affected if identity.kind == "MESH"}
    return any(identity in affected or data_uid in mesh_ids for identity, data_uid in owners.items())


def _expanded_visibility_owner_changed(scene, affected):
    # Expanded stock owners can have several generated schemas. Preserve their
    # complete identity closure rather than enter mesh/light sparse replacement.
    # A datablock edit has the same closure as its object notification. Curves
    # can export both source and evaluated schemas, so a one-root sparse patch
    # would leave duplicate active Blender owner mappings behind.
    if any(identity.kind in GEOMETRY_TYPES - {"MESH"} for identity in affected):
        return True
    return any(
        getattr(obj, "type", "") in GEOMETRY_TYPES - {"MESH"}
        or (getattr(obj, "type", "") == "EMPTY" and getattr(obj, "instance_collection", None) is not None)
        for obj in getattr(scene, "objects", ()) if blender_id(obj, "OBJECT") in affected)


def _visibility_requires_full_export(scene, generation):
    current = {blender_id(obj, "OBJECT") for obj in getattr(scene, "objects", ())}
    mapped = {key: value for key, value in getattr(generation, "blender_prim_paths", {}).items()
              if key.kind == "OBJECT"}
    return any(
        other in current and other != key and value.object_path.startswith(mapping.object_path + "/")
        for key, mapping in mapped.items() if key not in current
        for other, value in mapped.items())


def _view_layer_override(scene, module):
    from .object_visibility import scene_view_layer
    layer = scene_view_layer(scene, module)
    return {"view_layer": layer} if layer is not None else {}


@contextmanager
def _temporary_export_visibility(scene):
    from .object_visibility import export_visibility
    with export_visibility(_blender_module_provider(), scene):
        yield


def _object_visibility(scene):
    from .object_visibility import scene_visibility
    return {blender_id(obj, "OBJECT"): value for obj, value in scene_visibility(scene)}


def _mesh_shadow_visibility(scene):
    # Shadow and camera ray visibility travel together: either change needs a
    # regenerated stock export / opinion set (the camera one is a primvar opinion).
    return {blender_id(obj, "OBJECT"): (bool(getattr(obj, "visible_shadow", True)), bool(getattr(obj, "visible_camera", True)))
            for obj in getattr(scene, "objects", ()) if getattr(obj, "type", "") == "MESH"}


def _shadow_visibility_changed(scene, generation, affected):
    current = _mesh_shadow_visibility(scene)
    previous = getattr(generation, "shadow_visibility", {})
    return any(key in current and key in previous and current[key] != previous[key] for key in affected)


@performance_profile.timed("scene_generation.stock_export")
def _stock_export(
    scene: Any,
    path: Path,
    *,
    selected_objects_only: bool = False,
) -> None:
    module = _blender_module_provider()
    if module is None:
        raise SceneGenerationError("Blender is unavailable for scene generation")
    context = getattr(module, "context", None)
    if context is None:
        raise SceneGenerationError("Blender context is unavailable for scene generation")
    try:
        with (
            _temporary_export_visibility(scene),
            _temporary_export_identities(scene),
            context.temp_override(scene=scene, **_view_layer_override(scene, module)),
            _temporary_particle_hair_curves(
                scene,
                selected_objects_only=selected_objects_only,
            ),
        ):
            result = module.ops.wm.usd_export(
                filepath=str(path),
                selected_objects_only=selected_objects_only,
                export_animation=False,
                export_hair=False,
                evaluation_mode="VIEWPORT",
                root_prim_path="/World",
                export_materials=True,
                generate_preview_surface=True,
                generate_materialx_network=False,
                export_subdivision="TESSELLATE",
                triangulate_meshes=True,
                quad_method="FIXED",
                ngon_method="CLIP",
                export_textures_mode="NEW",
                relative_paths=True,
                overwrite_textures=False,
                export_custom_properties=True,
                author_blender_name=True,
                convert_orientation=False,
                convert_scene_units="METERS",
            )
    except SceneGenerationError:
        raise
    except Exception as exc:
        raise SceneGenerationError(
            f"Blender stock USD export failed: {type(exc).__name__}: {exc}"
        ) from exc
    if set(result) != {"FINISHED"}:
        raise SceneGenerationError(
            f"Blender stock USD export was cancelled: {sorted(result)}"
        )
    if not path.is_file():
        raise SceneGenerationError(
            f"Blender stock USD export did not create {path}"
        )
    _normalize_stock_export_geometry(path, scene=scene)
    _author_omitted_orthographic_cameras(path, scene)
    from .shadow_visibility import author_mesh_shadow_visibility
    author_mesh_shadow_visibility(path, scene)
    from pxr import Usd
    from .physics_frames import PhysicsUnits, normalize_stock_export_units
    stage = Usd.Stage.Open(str(path))
    _repair_packed_texture_assets(stage, path, getattr(getattr(module, 'data', None), 'images', ()))
    normalize_stock_export_units(stage, PhysicsUnits.from_scene(scene))
    # Registered authoring helpers are not an interchange schema. Keep the
    # exporter identity attributes until the normal mapping pass consumes them.
    for prim in stage.Traverse():
        for attr in tuple(prim.GetAttributes()):
            name = attr.GetName()
            if name == "userProperties:ov" or name.startswith("userProperties:ov:"):
                prim.RemoveProperty(name)
    stage.GetRootLayer().Save()


def _hair_path_max_step(settings: Any) -> int:
    """Largest valid ``ParticleSystem.co_hair`` step index (== ``2**render_step``).

    Blender evaluates each hair strand at ``2**render_step`` subdivisions;
    ``co_hair(step=N)`` samples that curve for ``N`` in ``0..2**render_step``.
    ``hair_step`` is the viewport fallback; the exponent is clamped to
    ``[1, 20]`` so a malformed file cannot request ``2**huge`` points.
    """

    render_step = int(getattr(settings, "render_step", 0) or 0)
    if render_step <= 0:
        render_step = int(getattr(settings, "hair_step", 1) or 1)
    render_step = min(max(render_step, 1), 20)
    return 1 << render_step


def _to_object_space(obj_eval: Any, co: Any) -> Any:
    """Convert a world-space ``co_hair`` sample to object-local coordinates.

    ``co_hair`` returns world-space points, but the temporary Curves object
    carries the emitter's ``matrix_world``; without this inverse the transform
    would be applied twice. Falls back to the unmodified coordinate on a missing
    or non-invertible matrix (degenerate transform), preferring slightly wrong
    hair over a crash mid-export.
    """

    matrix_world = getattr(obj_eval, "matrix_world", None)
    if matrix_world is None:
        return co
    try:
        return matrix_world.inverted() @ co
    except (AttributeError, TypeError, ValueError):
        return co


def _hair_widths(vertex_counts: Any, settings: Any) -> np.ndarray:
    root_width, tip_width = curve_widths.particle_width_range(
        getattr(settings, "radius_scale", 1.0),
        getattr(settings, "root_radius", 0.0),
        getattr(settings, "tip_radius", 0.0),
        use_close_tip=bool(getattr(settings, "use_close_tip", False)),
    )
    return curve_widths.fill_hair_widths(
        [int(count) for count in vertex_counts], root_width, tip_width
    )


def _collect_evaluated_particle_hair(psys: Any, obj_eval: Any, settings: Any):
    """Sample parent AND child hair strands from the evaluated particle cache.

    Operator-free replacement for ``curves.convert_from_particle_system``: reads
    the depsgraph-evaluated strand geometry (including interpolated/Simple
    children) through ``co_hair`` so it runs in any context — the render/F12
    context included, where the interactive operator's ``poll()`` fails. Each
    strand is resampled to ``curve_widths.hair_sample_count`` points. Returns
    ``(points, vertex_counts, widths)`` flat numpy arrays, or ``None`` when the
    evaluated cache has no usable child strands (the caller falls back to the
    authored parents).
    """

    if not hasattr(psys, "co_hair"):
        return None
    particles = getattr(psys, "particles", [])
    child_particles = getattr(psys, "child_particles", [])
    num_parents = len(particles)
    num_children = len(child_particles)
    # No children -> the cheaper parent-only fallback loses nothing; defer.
    if num_parents == 0 or num_children == 0:
        return None
    sample_count = curve_widths.hair_sample_count(
        getattr(settings, "render_step", 0), getattr(settings, "hair_step", 1)
    )
    if sample_count < 2:
        return None
    max_step = _hair_path_max_step(settings)
    # Evenly spaced integer step indices into each strand's baked curve.
    steps = np.rint(np.linspace(0, max_step, sample_count)).astype(np.int32)
    total_strands = num_parents + num_children
    points = np.empty(total_strands * sample_count * 3, dtype=np.float32)
    vertex_counts: list[int] = []
    point_idx = 0
    for strand_idx in range(total_strands):
        strand_start = point_idx
        strand_abs_sum = 0.0
        ok = True
        for step in steps:
            try:
                co = psys.co_hair(object=obj_eval, particle_no=strand_idx, step=int(step))
                co = _to_object_space(obj_eval, co)
                x, y, z = float(co[0]), float(co[1]), float(co[2])
            except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
                # co_hair raises for strands not in the cache; abandon the whole
                # strand (a partial strand corrupts the point partitioning).
                ok = False
                break
            base = point_idx * 3
            points[base] = x
            points[base + 1] = y
            points[base + 2] = z
            strand_abs_sum += abs(x) + abs(y) + abs(z)
            point_idx += 1
        # Keep the strand only if every step sampled cleanly and it is not
        # degenerate (all-zero coords mark unborn/failed children); else rewind.
        if ok and strand_abs_sum > 0.0:
            vertex_counts.append(sample_count)
        else:
            point_idx = strand_start
    if not vertex_counts:
        return None
    counts = np.asarray(vertex_counts, dtype=np.int32)
    return points[: point_idx * 3], counts, _hair_widths(counts, settings)


def _collect_parent_particle_hair(psys: Any, settings: Any):
    """Fallback: sample only the authored PARENT strands from ``hair_keys``.

    Used when the evaluated cache has no children. ``hair_keys.co`` is already
    object-local (unlike ``co_hair``), so no transform inversion is needed, and
    the keys are the control points, so the strand keeps its authored resolution.
    """

    particles = getattr(psys, "particles", [])
    if len(particles) == 0:
        return None
    vertex_counts: list[int] = []
    total_keys = 0
    for particle in particles:
        num_keys = len(getattr(particle, "hair_keys", ()))
        if num_keys >= 2:
            vertex_counts.append(num_keys)
            total_keys += num_keys
    if not vertex_counts:
        return None
    points = np.empty(total_keys * 3, dtype=np.float32)
    idx = 0
    for particle in particles:
        keys = particle.hair_keys
        num_keys = len(keys)
        if num_keys < 2:
            continue
        try:
            keys.foreach_get("co", points[idx * 3 : (idx + num_keys) * 3])
        except (AttributeError, RuntimeError):
            for k_idx, key in enumerate(keys):
                co = key.co
                base = (idx + k_idx) * 3
                points[base] = co[0]
                points[base + 1] = co[1]
                points[base + 2] = co[2]
        idx += num_keys
    counts = np.asarray(vertex_counts, dtype=np.int32)
    return points, counts, _hair_widths(counts, settings)


def _build_hair_curves_object(
    module: Any,
    data: Any,
    emitter: Any,
    settings: Any,
    points: np.ndarray,
    vertex_counts: np.ndarray,
    widths: np.ndarray,
) -> Any:
    """Author a temporary Curves object from sampled strands via the data API.

    No ``bpy.ops`` is used, so this runs in the render/F12 context. Native
    ``wm.usd_export`` emits the Curves object as ``BasisCurves`` (per-point
    ``radius`` -> USD ``widths``), which ``_normalize_stock_export_geometry``
    then normalizes to the OVRTX cubic/catmullRom/nonperiodic contract.
    """

    hair_curves = getattr(data, "hair_curves", None)
    objects = getattr(data, "objects", None)
    if hair_curves is None or objects is None:
        return None
    sizes = [int(count) for count in vertex_counts]
    if not sizes:
        return None
    curves_data = hair_curves.new(f"{getattr(emitter, 'name', 'Hair')} OVRTX Hair")
    curves_data.add_curves(sizes)
    curve_points = curves_data.points
    curve_points.foreach_set("position", np.ascontiguousarray(points, dtype=np.float32))
    radii = np.ascontiguousarray(np.asarray(widths, dtype=np.float32) * 0.5)
    try:
        curve_points.foreach_set("radius", radii)
    except (AttributeError, RuntimeError):
        pass
    curves_object = objects.new(curves_data.name, curves_data)
    # Generated roots are siblings. Stable names preserve the visibility owner
    # through cached bases without introducing process-local identity there.
    curves_object[PARTICLE_EMITTER_NAME_ATTRIBUTE] = str(emitter.name_full)
    matrix_world = getattr(emitter, "matrix_world", None)
    if matrix_world is not None:
        try:
            curves_object.matrix_world = matrix_world.copy()
        except (AttributeError, TypeError):
            pass
    _assign_particle_hair_material(curves_object, emitter, settings)
    return curves_object


@contextmanager
def _temporary_particle_hair_curves(
    scene: Any,
    *,
    selected_objects_only: bool = False,
) -> Any:
    """Materialize particle hair as temporary Curves objects for USD export.

    Blender's native ``wm.usd_export`` drops legacy particle hair, so each HAIR
    particle system is sampled from the depsgraph-evaluated cache (``co_hair``
    parent+children, with an authored ``hair_keys`` fallback) and rebuilt as a
    Curves object the native exporter can emit. This is fully operator-free — it
    replaces ``curves.convert_from_particle_system``, whose ``poll()`` requires
    an interactive context and fails during F12/offline renders.
    """

    module = _blender_module_provider()
    context = getattr(module, "context", None)
    if context is None:
        yield
        return
    data = getattr(module, "data", None)
    depsgraph_get = getattr(context, "evaluated_depsgraph_get", None)
    depsgraph = depsgraph_get() if callable(depsgraph_get) else None
    collection_objects = getattr(getattr(scene, "collection", None), "objects", None)
    created_objects: list[Any] = []
    try:
        for emitter in tuple(getattr(scene, "objects", ()) or ()):
            if selected_objects_only and not _object_selected(emitter):
                continue
            particle_systems = tuple(getattr(emitter, "particle_systems", ()) or ())
            if not particle_systems:
                continue
            emitter_eval = (
                emitter.evaluated_get(depsgraph) if depsgraph is not None else emitter
            )
            evaluated_systems = tuple(getattr(emitter_eval, "particle_systems", ()) or ())
            for index, particle_system in enumerate(particle_systems):
                settings = getattr(particle_system, "settings", None)
                if str(getattr(settings, "type", "")) != "HAIR":
                    continue
                source = (
                    evaluated_systems[index]
                    if index < len(evaluated_systems)
                    else particle_system
                )
                sampled = _collect_evaluated_particle_hair(source, emitter_eval, settings)
                if sampled is None:
                    sampled = _collect_parent_particle_hair(source, settings)
                if sampled is None:
                    continue
                points, vertex_counts, widths = sampled
                curves_object = _build_hair_curves_object(
                    module, data, emitter, settings, points, vertex_counts, widths
                )
                if curves_object is None:
                    continue
                if collection_objects is not None:
                    collection_objects.link(curves_object)
                created_objects.append(curves_object)
                if selected_objects_only:
                    select_set = getattr(curves_object, "select_set", None)
                    if callable(select_set):
                        select_set(True)
        view_layer = getattr(context, "view_layer", None)
        if view_layer is not None and callable(getattr(view_layer, "update", None)):
            view_layer.update()
        yield
    finally:
        objects_collection = getattr(data, "objects", None)
        curves_collection = getattr(data, "hair_curves", None)
        for curves_object in reversed(created_objects):
            curves_data = getattr(curves_object, "data", None)
            _remove_blender_id(objects_collection, curves_object)
            _remove_blender_id(curves_collection, curves_data)


def _object_selected(obj: Any) -> bool:
    select_get = getattr(obj, "select_get", None)
    return bool(select_get()) if callable(select_get) else False


def _assign_particle_hair_material(curves: Any, emitter: Any, settings: Any) -> None:
    material_index = max(0, int(getattr(settings, "material", 1) or 1) - 1)
    material_slots = tuple(getattr(emitter, "material_slots", ()) or ())
    if material_index >= len(material_slots):
        return
    material = getattr(material_slots[material_index], "material", None)
    if material is None:
        return
    materials = getattr(getattr(curves, "data", None), "materials", None)
    if materials is not None:
        materials.append(material)


def _remove_blender_id(collection: Any, value: Any) -> None:
    if collection is None or value is None:
        return
    try:
        collection.remove(value, do_unlink=True)
    except TypeError:
        collection.remove(value)


def _repair_packed_texture_assets(stage, path, images):
    """Repair stock's packed filename mismatch using exact exported bytes.

    Newly packed generated images can export under their image name while the
    shader points at the original filepath basename. Only private texture paths
    tied unambiguously to a packed source qualify; ordinary missing assets do not.
    """
    from pxr import Sdf
    missing = []
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath) or value.resolvedPath:
                continue
            if '<UDIM>' in value.path or '<UVTILE>' in value.path:
                continue
            relative = Path(value.path)
            if relative.is_absolute() or relative.parent.as_posix() != 'textures':
                continue
            missing.append((attr, relative.name.casefold()))
    if not missing:
        return
    wanted = {name for _, name in missing}
    packed = {}
    for image in images:
        # Blender's //filename is blend-relative, not a Windows UNC root.
        name = str(getattr(image, 'filepath', '') or '').replace('\\', '/').rsplit('/', 1)[-1].casefold()
        packed_file = getattr(image, 'packed_file', None)
        if name in wanted and packed_file is not None:
            packed.setdefault(name, set()).add(sha256(bytes(packed_file.data or b'')).hexdigest())
    if not packed:
        return
    exported = {}
    for texture in sorted((path.parent / 'textures').glob('*')):
        if texture.is_file():
            digest = sha256()
            with texture.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            exported.setdefault(digest.hexdigest(), texture)
    for attr, name in missing:
        digests = packed.get(name)
        if digests is None:
            continue
        if len(digests) != 1 or next(iter(digests)) not in exported:
            raise SceneGenerationError(f'Packed texture export has no unambiguous preserved asset: {name}')
        texture = exported[next(iter(digests))]
        attr.Set(Sdf.AssetPath('./' + texture.relative_to(path.parent).as_posix()))


def _normalize_stock_export_geometry(path: Path, *, scene=None) -> None:
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt  # type: ignore
    except ModuleNotFoundError as exc:
        raise SceneGenerationError("Blender OpenUSD is unavailable") from exc
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError(
            "stock-exported scene generation could not be opened",
            ({"generated_usd_path": str(path), "reason": "stage_open_failed"},),
        )
    legacy_curves = {str(obj.name): obj.data for obj in getattr(scene, 'objects', ())
                     if str(getattr(obj, 'type', '')) == 'CURVE'}
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        if type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateDoubleSidedAttr(False).Set(False)
            _quantize_mesh_uvs(prim, Gf, Vt)
            # Preserve the retained add-on's current-scene shading contract:
            # the exported scheme/creases are removed, so USD's fallback owns
            # subdivision shading. Forcing `none` changes visible highlights
            # even with identical evaluated points and corner normals. This
            # normalizer only handles our stock export, never supplied USD.
            for property_name in (
                "subdivisionScheme",
                "creaseIndices",
                "creaseLengths",
                "creaseSharpnesses",
            ):
                prim.RemoveProperty(property_name)
        elif type_name == "BasisCurves":
            curves = UsdGeom.BasisCurves(prim)
            owner = prim
            while owner and not owner.IsPseudoRoot():
                name = owner.GetAttribute('userProperties:blender:object_name').Get()
                if name in legacy_curves:
                    data = legacy_curves[name]
                    if getattr(data, 'bevel_mode', 'ROUND') == 'ROUND':
                        # Stock Blender emits 2*point.radius, omitting the
                        # round bevel radius. Scene-unit scaling is applied
                        # to the owning transform later, so stay object-local.
                        widths = curves.GetWidthsAttr().Get()
                        if widths is not None:
                            curves.GetWidthsAttr().Set(Vt.FloatArray([
                                float(width) * float(data.bevel_depth) for width in widths]))
                    break
                owner = owner.GetParent()
            vertex_counts = curves.GetCurveVertexCountsAttr().Get() or []
            # OVRTX renders hair/curve strands only as cubic catmullRom
            # *nonperiodic* BasisCurves (the known-good Junk Shop fixture config).
            # Blender's native catmullRom export authors wrap="pinned", and
            # linear curves also render invisible, so normalize every strand set
            # whose counts are valid for nonperiodic cubic (>= 4 points) to that
            # exact contract. Widths stay vertex-interpolated (length == points),
            # which is valid for nonperiodic cubic.
            if vertex_counts and all(int(count) >= 4 for count in vertex_counts):
                curves.GetTypeAttr().Set(UsdGeom.Tokens.cubic)
                curves.GetBasisAttr().Set(UsdGeom.Tokens.catmullRom)
                curves.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
            _mirror_widths_to_primvar(prim, curves, UsdGeom, Sdf)
    stage.GetRootLayer().Save()


def _author_omitted_orthographic_cameras(path: Path, scene: Any) -> None:
    """Author the Camera schema stock export omits for orthographic cameras.

    Blender's USD camera writer only supports perspective cameras, so an
    orthographic camera exports as a bare transform. Author its data prim the
    way stock names perspective camera data, so identity mapping and the
    composed render camera see a real Camera. Other projections stay omitted
    and are rejected by ``_validated_blender_prim_paths``.
    """

    cameras = {
        _session_uid(obj, "OBJECT"): obj
        for obj in getattr(scene, "objects", ())
        if str(getattr(obj, "type", "")) == "CAMERA"
        and str(getattr(getattr(obj, "data", None), "type", "")) == "ORTHO"
    }
    if not cameras:
        return
    try:
        from pxr import Gf, Sdf, Tf, Usd, UsdGeom  # type: ignore
    except ModuleNotFoundError as exc:
        raise SceneGenerationError("Blender OpenUSD is unavailable") from exc
    from .render_requests import camera_data_clipping_range, orthographic_frame_from_scene_camera

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError(
            "stock-exported scene generation could not be opened",
            ({"generated_usd_path": str(path), "reason": "stage_open_failed"},),
        )
    authored = False
    for prim in stage.Traverse():
        attribute = prim.GetAttribute(_OBJECT_UID_ATTRIBUTE)
        value = attribute.Get() if attribute else None
        obj = cameras.get(int(value)) if value is not None else None
        if obj is None:
            continue
        if any(str(child.GetTypeName()) == "Camera" for child in prim.GetChildren()):
            continue
        data = obj.data
        name = str(getattr(data, "name", ""))
        camera_path = prim.GetPath().AppendChild(Tf.MakeValidIdentifier(name))
        if stage.GetPrimAtPath(camera_path):
            raise SceneGenerationError(
                f"orthographic camera data path is already used for Blender object {obj.name_full!r}",
                ({"blender_id": str(obj.name_full), "candidate_paths": (str(camera_path),),
                  "reason": "orthographic_camera_path_conflict"},),
            )
        frame = orthographic_frame_from_scene_camera(data, scene)
        if frame is None:
            raise SceneGenerationError(
                f"orthographic camera frame is unavailable for Blender object {obj.name_full!r}",
                ({"blender_id": str(obj.name_full), "reason": "orthographic_camera_frame_unavailable"},),
            )
        frame_width, frame_height, frame_center_x, frame_center_y = frame
        camera = UsdGeom.Camera.Define(stage, camera_path)
        # Stock export authors lens values in tenths of a scene unit; the
        # orthographic aperture is the view frame in that same unit.
        camera.CreateProjectionAttr(UsdGeom.Tokens.orthographic)
        camera.CreateFocalLengthAttr(float(getattr(data, "lens", 50.0)) / 100.0)
        camera.CreateHorizontalApertureAttr(frame_width * 10.0)
        camera.CreateVerticalApertureAttr(frame_height * 10.0)
        camera.CreateHorizontalApertureOffsetAttr(frame_center_x * 10.0)
        camera.CreateVerticalApertureOffsetAttr(frame_center_y * 10.0)
        clipping_range = camera_data_clipping_range(data)
        if clipping_range is not None:
            camera.CreateClippingRangeAttr(Gf.Vec2f(*clipping_range))
        camera.GetPrim().CreateAttribute(
            "userProperties:blender:data_name", Sdf.ValueTypeNames.String, custom=True
        ).Set(name)
        authored = True
    if authored:
        stage.GetRootLayer().Save()


def _mirror_widths_to_primvar(prim: Any, curves: Any, usdgeom: Any, sdf: Any) -> None:
    """Mirror the builtin ``widths`` attribute into ``primvars:widths`` (vertex).

    Some RTX/Hydra curve paths read the ``primvars:widths`` primvar and ignore
    the builtin ``UsdGeomCurves.widths`` attribute; authoring both (identical
    per-point values, ``vertex`` interpolation) is belt-and-suspenders so hair
    strands receive their width whichever form the renderer consumes.
    """

    widths = curves.GetWidthsAttr().Get()
    if not widths:
        return
    primvar = usdgeom.PrimvarsAPI(prim).CreatePrimvar(
        "widths",
        sdf.ValueTypeNames.FloatArray,
        usdgeom.Tokens.vertex,
    )
    primvar.Set(widths)


def _remove_stock_export_identities(path: Path) -> None:
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError("stock-exported base could not be opened for sanitizing")
    for prim in stage.Traverse():
        for name in (
            _OBJECT_UID_ATTRIBUTE,
            _MESH_UID_ATTRIBUTE,
            _MATERIAL_UID_ATTRIBUTE,
        ):
            prim.RemoveProperty(name)
    stage.GetRootLayer().Save()


def _validate_identity_free_base(path: Path) -> None:
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError("reusable stock base could not be opened")
    for prim in stage.Traverse():
        if any(
            prim.HasProperty(name)
            for name in (
                _OBJECT_UID_ATTRIBUTE,
                _MESH_UID_ATTRIBUTE,
                _MATERIAL_UID_ATTRIBUTE,
            )
        ):
            raise SceneGenerationError("reusable stock base contains process-local identity")


def _write_identity_binding(
    destination: Path,
    mappings: Mapping[BlenderId, BlenderPrimPath],
) -> None:
    from pxr import Sdf, Usd  # type: ignore

    stage = Usd.Stage.CreateNew(str(destination))
    for identity, mapping in mappings.items():
        if identity.kind == "OBJECT":
            stage.OverridePrim(mapping.object_path).CreateAttribute(
                _OBJECT_UID_ATTRIBUTE, Sdf.ValueTypeNames.Int64
            ).Set(identity.session_uid)
            if mapping.blender_id_type == "MESH" and mapping.data_session_uid:
                stage.OverridePrim(mapping.schema_path).CreateAttribute(
                    _MESH_UID_ATTRIBUTE, Sdf.ValueTypeNames.Int64
                ).Set(mapping.data_session_uid)
        elif identity.kind == "MATERIAL":
            stage.OverridePrim(mapping.object_path).CreateAttribute(
                _MATERIAL_UID_ATTRIBUTE, Sdf.ValueTypeNames.Int64
            ).Set(identity.session_uid)
    stage.GetRootLayer().Save()


def _copy_stock_stage_metadata(root: Any, base: Path) -> None:
    """Keep the complete export's stage interpretation on each owned root."""
    from pxr import Sdf  # type: ignore
    from .usd_stage_metadata import copy_authored_stage_metadata

    source = Sdf.Layer.FindOrOpen(str(base))
    if source is None:
        raise SceneGenerationError(f"stock metadata source could not be opened at {base}")
    copy_authored_stage_metadata(source, root)


def _write_stock_root(destination: Path, binding: Path, base: Path) -> None:
    from pxr import Sdf, Usd  # type: ignore

    root = Sdf.Layer.CreateNew(str(destination))
    _copy_stock_stage_metadata(root, base)
    root.subLayerPaths = [
        _relative_sublayer(destination, binding),
        _relative_sublayer(destination, base),
    ]
    root.Save()
    if Usd.Stage.Open(str(destination)) is None:
        raise SceneGenerationError("session-bound stock base could not be opened")


def _quantize_mesh_uvs(prim: Any, gf: Any, vt: Any) -> None:
    uv_attr = prim.GetAttribute("primvars:st")
    if not uv_attr or not uv_attr.IsValid():
        return
    values = uv_attr.Get()
    if values is None:
        return
    uv_attr.Set(
        vt.Vec2fArray(
            [
                gf.Vec2f(round(float(uv[0]), 3), round(float(uv[1]), 3))
                for uv in values
            ]
        )
    )


def _validated_blender_prim_paths(
    scene: Any,
    path: Path,
) -> dict[BlenderId, BlenderPrimPath]:
    try:
        from pxr import Usd  # type: ignore
    except ModuleNotFoundError as exc:
        raise SceneGenerationError("Blender OpenUSD is unavailable") from exc
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError(
            "stock-exported scene generation could not be opened",
            ({"generated_usd_path": str(path), "reason": "stage_open_failed"},),
        )
    object_prims: dict[int, list[Any]] = {}
    material_prims: dict[int, list[Any]] = {}
    material_paths_by_name: dict[str, list[str]] = {}
    for prim in stage.Traverse():
        for attribute_name, destination in (
            (_OBJECT_UID_ATTRIBUTE, object_prims),
            (_MATERIAL_UID_ATTRIBUTE, material_prims),
        ):
            attribute = prim.GetAttribute(attribute_name)
            value = attribute.Get() if attribute else None
            if value is not None:
                destination.setdefault(int(value), []).append(prim)
        if str(prim.GetTypeName()) == "Material":
            name_attribute = prim.GetAttribute("userProperties:blender:data_name")
            name = name_attribute.Get() if name_attribute else None
            if name:
                material_paths_by_name.setdefault(str(name), []).append(str(prim.GetPath()))

    objects_by_uid = {
        _session_uid(obj, "OBJECT"): obj
        for obj in getattr(scene, "objects", ())
    }
    mappings: dict[BlenderId, BlenderPrimPath] = {}
    for object_uid, candidates in sorted(object_prims.items()):
        obj = objects_by_uid.get(object_uid)
        if obj is None:
            raise SceneGenerationError(
                f"stock-exported object session_uid {object_uid} is not in the Blender scene",
                ({"blender_id_kind": "OBJECT", "session_uid": object_uid, "reason": "unknown_object_identity"},),
            )
        object_type = str(getattr(obj, "type", ""))
        if object_type not in SUPPORTED_OBJECT_TYPES:
            continue
        name = str(obj.name_full)
        if object_type in GEOMETRY_TYPES - {"MESH"}:
            # Stock converts curve/font owners to geometry and copies the same
            # temporary object UID onto their generated child schemas.
            roots = [prim for prim in candidates if not any(
                prim != other and prim.GetPath().HasPrefix(other.GetPath())
                for other in candidates)]
            if len(roots) == 1:
                candidates = roots
        if len(candidates) != 1:
            raise SceneGenerationError(
                f"stock-exported prim identity is not unique for Blender object {name!r}",
                (
                    {
                        "blender_id": name,
                        "blender_id_kind": "OBJECT",
                        "session_uid": object_uid,
                        "candidate_paths": tuple(str(prim.GetPath()) for prim in candidates),
                        "reason": "missing_object_identity" if not candidates else "ambiguous_object_identity",
                    },
                ),
            )
        object_prim = candidates[0]
        data_name = str(getattr(getattr(obj, "data", None), "name", ""))
        schema_paths = [str(object_prim.GetPath())] if object_type == "EMPTY" or object_type in GEOMETRY_TYPES - {"MESH"} else _schema_paths(
            stage, object_prim, object_type, data_name,
            viewport_hidden=False,
        )
        camera_type = str(getattr(getattr(obj, "data", None), "type", ""))
        if object_type == "CAMERA" and not schema_paths and camera_type not in {"PERSP", "ORTHO"}:
            raise SceneGenerationError(
                f"{camera_type.lower()} camera {name!r} is not supported; use a perspective or orthographic camera",
                (
                    {
                        "blender_id": name,
                        "object_path": str(object_prim.GetPath()),
                        "camera_type": camera_type,
                        "reason": "unsupported_camera_type",
                    },
                ),
            )
        if len(schema_paths) != 1:
            raise SceneGenerationError(
                f"stock-exported schema identity is not unique for Blender object {name!r}",
                (
                    {
                        "blender_id": name,
                        "object_path": str(object_prim.GetPath()),
                        "candidate_paths": tuple(schema_paths),
                        "reason": "missing_schema_identity" if not schema_paths else "ambiguous_schema_identity",
                    },
                ),
            )
        data_uid = 0
        if object_type in {"LIGHT", "MESH"}:
            data_uid = _session_uid(obj.data, object_type)
        mappings[BlenderId("OBJECT", object_uid)] = BlenderPrimPath(
            blender_id_name=name,
            blender_id_type=object_type,
            object_path=str(object_prim.GetPath()),
            schema_path=schema_paths[0],
            data_session_uid=data_uid,
        )
    module = _blender_module_provider()
    available_materials = getattr(getattr(module, "data", None), "materials", ())
    materials = {
        _session_uid(material, "MATERIAL"): material
        for material in available_materials
    }
    materials_by_name: dict[str, list[tuple[int, Any]]] = {}
    for pointer, material in materials.items():
        materials_by_name.setdefault(str(material.name_full), []).append((pointer, material))
    mapped_material_paths = set()
    for pointer, candidates in sorted(material_prims.items()):
        material = materials.get(pointer)
        paths = [str(prim.GetPath()) for prim in candidates]
        if material is None or len(paths) != 1:
            raise SceneGenerationError(
                f"stock-exported material session_uid {pointer} is not unique",
                (
                    {
                        "blender_id": "" if material is None else str(material.name_full),
                        "blender_id_type": "MATERIAL",
                        "session_uid": pointer,
                        "candidate_paths": tuple(paths),
                        "reason": "unknown_material_identity" if material is None else "ambiguous_material_identity",
                    },
                ),
            )
        mappings[BlenderId("MATERIAL", pointer)] = BlenderPrimPath(
            blender_id_name=str(material.name_full),
            blender_id_type="MATERIAL",
            object_path=paths[0],
            schema_path=paths[0],
        )
        mapped_material_paths.add(paths[0])
    for name, paths in sorted(material_paths_by_name.items()):
        remaining = [path for path in paths if path not in mapped_material_paths]
        if not remaining:
            continue
        candidates = materials_by_name.get(name, [])
        if not candidates:
            continue
        if len(remaining) != 1 or len(candidates) != 1:
            raise SceneGenerationError(
                f"stock-exported prim identity is not unique for Blender material {name!r}",
                (
                    {
                        "blender_id": name,
                        "blender_id_type": "MATERIAL",
                        "candidate_paths": tuple(remaining),
                        "candidate_session_uids": tuple(pointer for pointer, _material in candidates),
                        "reason": "ambiguous_material_identity",
                    },
                ),
            )
        pointer, material = candidates[0]
        mappings[BlenderId("MATERIAL", pointer)] = BlenderPrimPath(
            blender_id_name=str(material.name_full),
            blender_id_type="MATERIAL",
            object_path=remaining[0],
            schema_path=remaining[0],
        )
    world = getattr(scene, "world", None)
    dome_prim = stage.GetPrimAtPath(DEFAULT_DOME_OWNER_PATH)
    if world is not None and not dome_prim:
        # This stage is the owned stock Blender export. Its World is the one
        # DomeLight outside exported Object ownership; ordinary light objects
        # are already mapped above. Never infer ownership in external stages.
        object_roots = tuple(mapping.object_path for identity, mapping in mappings.items()
                             if identity.kind == 'OBJECT')
        candidates = tuple(prim for prim in stage.Traverse()
            if str(prim.GetTypeName()) == 'DomeLight'
            and not any(str(prim.GetPath()) == root or str(prim.GetPath()).startswith(root + '/')
                        for root in object_roots))
        if len(candidates) > 1:
            raise SceneGenerationError('stock-exported World dome identity is not unique')
        if candidates:
            dome_prim = candidates[0]
    from .world_dome_conversion import environment_texture_dome
    canonical_environment = world is not None and environment_texture_dome(world) is not None
    if world is not None and (canonical_environment or (dome_prim and str(dome_prim.GetTypeName()) == "DomeLight")):
        mappings[blender_id(world, "WORLD")] = BlenderPrimPath(
            blender_id_name=str(world.name_full),
            blender_id_type="WORLD",
            object_path=DEFAULT_DOME_OWNER_PATH if canonical_environment else str(dome_prim.GetPath()),
            schema_path=DEFAULT_DOME_OWNER_PATH if canonical_environment else str(dome_prim.GetPath()),
        )
    return mappings


def _emissive_proxy_records(stock_stage_provider):
    """Always-on light proxies for emissive meshes (GH #44 lane 2)."""
    from .emissive_light_proxy import opinion_records as proxy_records
    module = _blender_module_provider()
    materials = getattr(getattr(module, "data", None), "materials", None)
    by_name: dict = {}
    # Linked (library) materials are not found by ``materials.get(name)``; index
    # every material by name, preferring local ones, as the stock export names them.
    for material in (materials if materials is not None else ()):
        if getattr(material, "library", None) is None or material.name not in by_name:
            by_name[material.name] = material
    try:
        stock_stage = stock_stage_provider()
    except Exception:  # proxies are lighting quality only: never block a generation on them
        return ()
    return proxy_records(stock_stage, by_name.get)


@performance_profile.timed("scene_generation.compile_addon_opinions")
def _compile_add_on_opinions(
    scene: Any,
    mappings: Mapping[BlenderId, BlenderPrimPath],
    stock_usd_path: Path,
    *, stock_stage: Any = None,
) -> tuple[AddOnUsdOpinionRecord, ...]:
    from .object_visibility import opinion_records, merge_records
    from .world_scene_layer import opinion_records as world_opinions
    from .semantic_identity import opinion_records as semantic_opinion_records
    # Request-lifetime reuse only: the stock composition is immutable through
    # this compilation, and discarded before candidate folder promotion.
    shared_stock = [stock_stage]
    def source_stage():
        if shared_stock[0] is None:
            from pxr import Usd
            shared_stock[0] = Usd.Stage.Open(str(stock_usd_path))
            if shared_stock[0] is None:
                raise SceneGenerationError('stock-exported scene generation could not be opened')
        return shared_stock[0]
    visibility_records = opinion_records(scene, mappings, blender_id, stock_usd_path,
                                         stock_stage_provider=source_stage)
    visibility_records += world_opinions(scene, stock_usd_path, stock_stage_provider=source_stage)
    visibility_records += _emissive_proxy_records(source_stage)
    visibility_records += semantic_opinion_records(scene, mappings, blender_id)
    from .object_matte import opinion_records as matte_records
    visibility_records += matte_records(scene, mappings, blender_id)
    if not hasattr(scene, "ov"):
        return merge_records(visibility_records)
    from . import (
        authoring_properties,
        physics_topology_conversion,
        simready_physics_conversion,
    )
    from . import usd_paths
    from .physics_colliders import collider_records
    from .physics_frames import PhysicsUnits
    units = PhysicsUnits.from_scene(scene) if hasattr(scene, "unit_settings") else PhysicsUnits()
    from .physics_owner_compilation import authoring_owners, compile_physics_owners, collider_material

    records = list(visibility_records)
    occupied = _prim_paths(stock_usd_path, stage=source_stage())
    for obj in sorted(scene.objects, key=lambda item: str(item.name_full)):
        values = obj.get(MATRIX_OVERRIDE_PROP)
        if values is None:
            continue
        mapping = mappings.get(BlenderId("OBJECT", _session_uid(obj, "OBJECT")))
        if mapping is None:
            raise SceneGenerationError(
                f"scene generation has no mapped prim for Blender object {obj.name_full!r}"
            )
        records.append(_mapped_transform_override(obj, mapping, values))
    physics_scene = physics_topology_conversion.convert_generation_physics_scene(
        scene
    )
    if physics_scene is not None:
        if physics_scene.usd_prim_path in occupied:
            raise SceneGenerationError(
                "add-on physics scene conflicts with stock-exported topology",
                (
                    {
                        "usd_prim_path": physics_scene.usd_prim_path,
                        "reason": "add_on_root_occupied",
                    },
                ),
            )
        records.append(physics_scene)
        occupied.add(physics_scene.usd_prim_path)
    simready_unibodies = simready_physics_conversion.read_scene_unibodies(scene)
    physics_material_paths: dict[int, str] = {}
    physics_materials = {
        _session_uid(material, "MATERIAL"): material
        for obj, owner, bone_name in authoring_owners(scene)
        for collider in collider_records(owner)
        for material in (collider_material(collider),)
        if material is not None
        and material.ov.physics.schema_opinion != authoring_properties.INHERIT
    }
    physics_materials.update(
        {
            _session_uid(material, "MATERIAL"): material
            for material in simready_physics_conversion.explicit_materials(
                simready_unibodies
            )
        }
    )
    for pointer, material in sorted(
        physics_materials.items(),
        key=lambda item: str(item[1].name_full),
    ):
        mapping = mappings.get(BlenderId("MATERIAL", _session_uid(material, "MATERIAL")))
        path = (
            mapping.object_path
            if mapping is not None
            else usd_paths.reserve_unique_child_path(
                "/World/PhysicsMaterials",
                str(material.name_full),
                occupied,
            )
        )
        preserved = getattr(material.ov.physics, "preserved_prim_path", "")
        if preserved:
            if any(path == preserved or path.startswith(preserved + "/") for path in occupied):
                raise SceneGenerationError("Preserved physics material path overlaps source USD")
            path = preserved
            occupied.add(path)
            mapping = None
        physics_material_paths[pointer] = path
        records.append(
            physics_topology_conversion.convert_generation_physics_material(
                material,
                path,
                mapped=mapping is not None,
                units=units,
            )
        )
    records.extend(compile_physics_owners(scene, mappings, stock_usd_path, physics_material_paths))
    records.extend(
        simready_physics_conversion.convert_scene_unibodies(
            simready_unibodies,
            mappings,
            occupied,
            physics_material_paths,
        )
    )
    from .physics_joint_conversion import compile_scene_joints
    joint_records, replacements = compile_scene_joints(scene, mappings, stock_usd_path, records)
    # Joints may replace a body's owner record to add static contact filters.
    records = [replacements.get(id(record), record) for record in records]
    records.extend(joint_records)
    from .object_visibility import merge_records
    return merge_records(records)


def _stock_body_scale(stock_path, body_path):
    """Composed stock uniform scale of a mapped body (SimReady COM conversion)."""
    from pxr import Usd, UsdGeom
    from .physics_frames import matrix_to_transform
    stage = Usd.Stage.Open(str(stock_path))
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(body_path))
    return matrix_to_transform(tuple(tuple(matrix[column][row] for column in range(4)) for row in range(4))).uniform_scale


def _mapped_transform_override(
    obj: Any,
    mapping: BlenderPrimPath,
    values: Any,
) -> AddOnUsdOpinionRecord:
    import math

    from pxr import Gf, Sdf, Usd, UsdGeom  # type: ignore

    try:
        flattened = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise SceneGenerationError(
            f"invalid matrix override for Blender object {obj.name_full!r}"
        ) from exc
    if len(flattened) != 16 or not all(math.isfinite(value) for value in flattened):
        raise SceneGenerationError(
            f"invalid matrix override for Blender object {obj.name_full!r}"
        )
    layer = Sdf.Layer.CreateAnonymous(".usda")
    stage = Usd.Stage.Open(layer)
    path = Sdf.Path(mapping.object_path)
    ancestors = []
    parent = path.GetParentPath()
    while parent != Sdf.Path.absoluteRootPath:
        ancestors.append(parent)
        parent = parent.GetParentPath()
    for ancestor in reversed(ancestors):
        stage.OverridePrim(ancestor)
    prim = stage.OverridePrim(path)
    xformable = UsdGeom.Xformable(prim)
    op = xformable.AddTransformOp()
    op.Set(Gf.Matrix4d(*(flattened[index : index + 4] for index in range(0, 16, 4))))
    xformable.SetXformOpOrder((op,))
    return AddOnUsdOpinionRecord(
        mapping.object_path,
        layer,
        {
            "source": MATRIX_OVERRIDE_PROP,
            "blender_id": str(obj.name_full),
            "usd_prim_path": mapping.object_path,
        },
    )


def _prim_paths(path: Path, *, stage: Any = None) -> set[str]:
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(path)) if stage is None else stage
    if stage is None:
        raise SceneGenerationError(
            "stock-exported scene generation could not be opened"
        )
    return {str(prim.GetPath()) for prim in stage.TraverseAll()}


def _generation_digest(
    base_digest: str,
    identity_digest: str,
    records: tuple[AddOnUsdOpinionRecord, ...],
) -> str:
    parts = [
        b"scene-generation-v2",
        base_digest.encode("ascii"),
        identity_digest.encode("ascii"),
    ]
    for record in records:
        parts.extend(
            (record.usd_prim_path.encode("utf-8"), record.digest.encode("ascii"))
        )
    encoded = b"".join(len(part).to_bytes(8, "big") + part for part in parts)
    return sha256(encoded).hexdigest()


def _mapping_identity_digest(
    mappings: Mapping[BlenderId, BlenderPrimPath],
) -> str:
    bindings = []
    for identity, mapping in sorted(mappings.items()):
        if identity.kind == "OBJECT":
            bindings.append(
                (
                    _OBJECT_UID_ATTRIBUTE,
                    mapping.object_path,
                    identity.session_uid,
                )
            )
            if mapping.blender_id_type == "MESH" and mapping.data_session_uid:
                bindings.append(
                    (
                        _MESH_UID_ATTRIBUTE,
                        mapping.schema_path,
                        mapping.data_session_uid,
                    )
                )
        elif identity.kind == "MATERIAL":
            bindings.append(
                (
                    _MATERIAL_UID_ATTRIBUTE,
                    mapping.object_path,
                    identity.session_uid,
                )
            )
    return sha256(repr(bindings).encode("utf-8")).hexdigest()


def _opinion_change(
    predecessor: SceneGeneration | None,
    records: tuple[AddOnUsdOpinionRecord, ...],
) -> SparseAddOnOpinionChange:
    if predecessor is None:
        return SparseAddOnOpinionChange()
    previous = {
        record.usd_prim_path: record for record in predecessor.opinion_records
    }
    current = {record.usd_prim_path: record for record in records}
    return SparseAddOnOpinionChange(
        (
            record
            for path, record in current.items()
            if path not in previous or previous[path].digest != record.digest
        ),
        set(previous).difference(current),
    )


def _object_export_hidden(obj: Any) -> bool:
    """Match the viewport evaluation visibility used by stock USD export."""
    if bool(getattr(obj, "hide_viewport", False)):
        return True
    hide_get = getattr(obj, "hide_get", None)
    if callable(hide_get) and hide_get():
        return True
    visible_get = getattr(obj, "visible_get", None)
    return callable(visible_get) and not visible_get()


def _schema_paths(
    stage: Any,
    object_prim: Any,
    object_type: str,
    data_name: str,
    *,
    viewport_hidden: bool = False,
) -> list[str]:
    from pxr import Usd  # type: ignore

    object_path = object_prim.GetPath()
    matches = []
    type_matches = []
    for prim in Usd.PrimRange(object_prim):
        path = prim.GetPath()
        if path != object_path and not path.HasPrefix(object_path):
            continue
        if path != object_path:
            owner_attribute = prim.GetAttribute(_OBJECT_UID_ATTRIBUTE)
            if owner_attribute and owner_attribute.Get() is not None:
                continue
        ancestor = prim.GetParent()
        nested_owner = False
        while ancestor and ancestor.GetPath() != object_path:
            owner_attribute = ancestor.GetAttribute(_OBJECT_UID_ATTRIBUTE)
            if owner_attribute and owner_attribute.Get() is not None:
                nested_owner = True
                break
            ancestor = ancestor.GetParent()
        if nested_owner:
            continue
        type_name = str(prim.GetTypeName())
        if not _is_schema_type_match(object_type, type_name):
            continue
        type_matches.append(str(path))
        attribute = prim.GetAttribute("userProperties:blender:data_name")
        if attribute and str(attribute.Get()) == data_name:
            matches.append(str(path))
    # Stock Blender USD export retains the transform of viewport-hidden objects
    # but intentionally omits their data schema. Keep object/data identity on
    # that transform without inventing geometry or enabling it for rendering.
    # Existing visibility fingerprints require a new export when unhidden.
    # A hidden flag must never excuse ambiguous exported geometry.
    if (viewport_hidden and not type_matches
            and object_type in {"MESH", "LIGHT"}
            and str(object_prim.GetTypeName()) == "Xform"
            and not object_prim.IsInstance()):
        return [str(object_path)]
    if not matches and len(type_matches) == 1:
        return type_matches
    return sorted(matches)


def _is_schema_type_match(object_type: str, type_name: str) -> bool:
    if object_type == "CAMERA":
        return type_name == "Camera"
    if object_type == "LIGHT":
        return type_name.endswith("Light")
    if object_type == "MESH":
        return type_name == "Mesh"
    if object_type in GEOMETRY_TYPES:
        return type_name in {"Xform", "Mesh", "BasisCurves", "NurbsCurves", "Points", "Volume"}
    return False


def blender_id(value: Any, kind: str) -> BlenderId:
    return BlenderId(str(kind), _session_uid(value, str(kind)))


def _session_uid(value: Any, kind: str) -> int:
    uid = int(getattr(value, "session_uid", 0) or 0)
    if uid <= 0:
        name = str(getattr(value, "name_full", getattr(value, "name", "")))
        raise SceneGenerationError(
            f"Blender {kind.lower()} {name!r} has no session_uid",
            ({"blender_id": name, "blender_id_kind": kind, "reason": "missing_session_uid"},),
        )
    return uid


def _world_session_uid(scene: Any) -> int:
    world = getattr(scene, "world", None)
    return 0 if world is None else _session_uid(world, "WORLD")


def effective_object_materials(obj: Any) -> tuple[Any, ...]:
    """Resolve native DATA slots and per-object material overrides in slot order."""
    slots = getattr(obj, "material_slots", None)
    if slots is not None:
        return tuple(slot.material for slot in slots)
    return tuple(getattr(getattr(obj, "data", None), "materials", ()))


def object_export_materials(obj: Any) -> tuple[Any, ...]:
    """Materials reached by an owner, including nested collection instances.

    Instance children remain owned by their scene instancer: never stamp or map
    them as independent scene OBJECT identities.
    """
    materials = {}; visited_objects = set(); visited_collections = set()
    references = []
    def identity(value):
        uid = getattr(value, "session_uid", None)
        if uid is not None:
            return ("session_uid", int(uid))
        pointer = getattr(value, "as_pointer", None)
        if callable(pointer):
            return ("pointer", int(pointer()))
        references.append(value)  # Keep fixture identities alive during recursion.
        return ("python", id(value))
    def visit(owner):
        key = identity(owner)
        if key in visited_objects:
            return
        visited_objects.add(key)
        if str(getattr(owner, "type", "")) in GEOMETRY_TYPES:
            for material in effective_object_materials(owner):
                if material is not None:
                    materials[_session_uid(material, "MATERIAL")] = material
        collection = getattr(owner, "instance_collection", None)
        if collection is not None:
            visit_collection(collection)
    def visit_collection(collection):
        key = identity(collection)
        if key in visited_collections:
            return
        visited_collections.add(key)
        for child in getattr(collection, "objects", ()):
            visit(child)
        for child in getattr(collection, "children", ()):
            visit_collection(child)
    visit(obj)
    return tuple(materials.values())


@contextmanager
def _temporary_export_identities(scene: Any) -> Any:
    values = []
    ids = []
    ids.extend((obj, "OBJECT") for obj in getattr(scene, "objects", ()))
    ids.extend(
        (obj.data, "MESH")
        for obj in getattr(scene, "objects", ())
        if str(getattr(obj, "type", "")) == "MESH"
        and getattr(obj, "data", None) is not None
    )
    ids.extend(
        (material, "MATERIAL")
        for obj in getattr(scene, "objects", ())
        for material in object_export_materials(obj)
    )
    seen = set()
    try:
        for value, kind in ids:
            identity = (kind, _session_uid(value, kind))
            if identity in seen:
                continue
            seen.add(identity)
            key = _IDENTITY_PROPERTIES[kind]
            present = key in value
            previous = value.get(key) if present else None
            values.append((value, key, present, previous))
            value[key] = identity[1]
        yield
    finally:
        for value, key, present, previous in reversed(values):
            if present:
                value[key] = previous
            elif key in value:
                del value[key]


def _geometry_input_value(value: Any) -> Any:
    """Normalize render inputs without hashing RNA addresses or UI state."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, set):
        return tuple(sorted(value))
    if hasattr(value, "session_uid"):
        return (str(getattr(value, "id_type", type(value).__name__)),
                int(value.session_uid))
    try:
        values = tuple(value)
    except TypeError:
        return None
    if all(isinstance(item, (bool, int, float, str)) for item in values):
        return values
    return None


def _geometry_rna_inputs(value: Any) -> tuple[Any, ...]:
    # Only modifier/node scalar settings and ID references are inspected. RNA
    # collections, context, evaluated handles and UI expansion never participate.
    ignored = {"rna_type", "name", "type", "show_expanded", "is_active",
               "execution_time", "persistent_uid", "location", "width", "height",
               "select", "label", "use_custom_color", "color", "dimensions"}
    result = []
    for prop in getattr(getattr(value, "bl_rna", None), "properties", ()):
        name = str(prop.identifier)
        if (name in ignored or bool(getattr(prop, "is_readonly", False))
                or str(prop.type) not in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM", "POINTER"}):
            continue
        normalized = _geometry_input_value(getattr(value, name, None))
        if normalized is not None:
            result.append((name, normalized))
    return tuple(sorted(result))


def _geometry_node_inputs(tree: Any, visited: frozenset[int] = frozenset()) -> Any:
    if tree is None:
        return None
    uid = int(getattr(tree, "session_uid", 0))
    if uid in visited:
        return ("recursive_group", uid)
    visited = visited | {uid}
    nodes = tuple(
        (str(node.name), str(getattr(node, "bl_idname", "")),
         bool(getattr(node, "mute", False)), _geometry_rna_inputs(node),
         tuple((str(getattr(socket, "identifier", socket.name)),
                _geometry_input_value(getattr(socket, "default_value", None)))
               for socket in (*getattr(node, "inputs", ()), *getattr(node, "outputs", ()))),
         _geometry_node_inputs(getattr(node, "node_tree", None), visited))
        for node in getattr(tree, "nodes", ())
    )
    links = tuple(sorted(
        (str(link.from_node.name), str(link.from_socket.identifier),
         str(link.to_node.name), str(link.to_socket.identifier))
        for link in getattr(tree, "links", ())
    ))
    return (uid, nodes, links)


def _mesh_render_inputs(obj: Any, data: Any, *, include_elements: bool = True) -> tuple[Any, ...]:
    from .semantic_identity import fingerprint as semantic_fingerprint
    from .object_matte import opinions as matte_opinions
    modifiers = []
    for modifier in getattr(obj, "modifiers", ()):
        # Only Geometry Nodes modifiers expose IDProperties for interface
        # sockets. Other bpy modifier types have keys() but raise TypeError.
        try:
            keys = modifier.keys() if str(getattr(modifier, "type", "")) == "NODES" else ()
        except TypeError:  # Empty interface has no IDProperty storage yet.
            keys = ()
        custom_inputs = tuple(sorted(
            (str(key), _geometry_input_value(modifier[key]))
            for key in keys
            if str(key) != "_RNA_UI"
        ))
        modifiers.append((str(getattr(modifier, "type", "")),
                          bool(getattr(modifier, "show_viewport", True)),
                          bool(getattr(modifier, "show_render", True)),
                          _geometry_rna_inputs(modifier), custom_inputs,
                          _geometry_node_inputs(getattr(modifier, "node_group", None))))
    keys = getattr(data, "shape_keys", None)
    shape_keys = tuple(
        (str(key.name), float(getattr(key, "value", 0.0)), float(getattr(key, "frame", 0.0)),
         bool(getattr(key, "mute", False)), str(getattr(key, "vertex_group", "")),
         str(getattr(getattr(key, "relative_key", None), "name", "")),
         tuple(tuple(float(v) for v in point.co) for point in getattr(key, "data", ())) if include_elements else ())
        for key in getattr(keys, "key_blocks", ())
    )
    normals = (tuple(tuple(float(v) for v in normal.vector)
                     for normal in getattr(data, "corner_normals", ())) if include_elements else ())
    slots = tuple((str(getattr(slot, "link", "DATA")),
                   _session_uid(slot.material, "MATERIAL") if slot.material else 0)
                  for slot in getattr(obj, "material_slots", ()))
    groups = (tuple(str(group.name) for group in getattr(obj, "vertex_groups", ())),
              tuple(tuple((int(group.group), float(group.weight))
                          for group in getattr(vertex, "groups", ()))
                    for vertex in data.vertices) if include_elements else ())
    only_key = bool(getattr(obj, "show_only_shape_key", False))
    return (semantic_fingerprint(obj), tuple(modifiers), bool(getattr(keys, "use_relative", True)),
            float(getattr(keys, "eval_time", 0.0)), shape_keys, normals, slots,
            groups, only_key, int(getattr(obj, "active_shape_key_index", 0)) if only_key else 0,
            tuple(sorted(matte_opinions(obj).items())))


def _mesh_shading_inputs(obj: Any) -> tuple[Any, ...]:
    """Export controls that may change without Blender's geometry flag.

    A shader edit can propagate shading-only notifications to a large mesh.
    Compare these controls before skipping its per-element topology/UV scan.
    """
    from . import simready_physics_conversion
    data = obj.data
    return (str(obj.name_full), _session_uid(data, "MESH"), str(data.name_full),
            tuple(_session_uid(material, "MATERIAL") if material else 0 for material in getattr(data, "materials", ())),
            simready_physics_conversion.fingerprint_for_object(obj),
            _session_uid(obj.parent, "OBJECT") if getattr(obj, "parent", None) else 0,
            bool(getattr(obj, "hide_render", False)),
            bool(getattr(obj, "hide_viewport", False)),
            _mesh_render_inputs(obj, data, include_elements=False))


def _mesh_shading_inputs_for_scene(scene: Any, identities: set[BlenderId] | None = None) -> dict[BlenderId, tuple[Any, ...]]:
    return {blender_id(obj, "OBJECT"): _mesh_shading_inputs(obj)
            for obj in getattr(scene, "objects", ()) if str(getattr(obj, "type", "")) == "MESH"
            and (identities is None or blender_id(obj, "OBJECT") in identities)}


def _material_graph_fingerprints(scene: Any, mappings: Mapping[BlenderId, BlenderPrimPath]) -> dict[BlenderId, str]:
    from .material_graph_fingerprint import material_graph_signature

    module = _blender_module_provider()
    materials = list(getattr(getattr(module, "data", None), "materials", ()) or ())
    # Fake/headless scene providers and scene-local materials need no bpy data.
    for obj in getattr(scene, "objects", ()) or ():
        materials.extend(object_export_materials(obj))
    result = {}
    for material in materials:
        if material is None:
            continue
        identity = blender_id(material, "MATERIAL")
        if identity in mappings and identity not in result:
            result[identity] = material_graph_signature(material)
    return result


def _unsupported_light_values(scene: Any) -> dict[BlenderId, Mapping[str, Any]]:
    from .light_value_conversion import unsupported_field_values
    return {blender_id(obj, "OBJECT"): unsupported_field_values(obj)
            for obj in getattr(scene, "objects", ()) if str(getattr(obj, "type", "")) == "LIGHT"}


def _topology_fingerprints(
    scene: Any,
    identities: set[BlenderId] | None = None,
) -> dict[BlenderId, str]:
    from . import simready_physics_conversion

    result: dict[BlenderId, str] = {}
    for obj in getattr(scene, "objects", ()):
        object_type = str(getattr(obj, "type", ""))
        if object_type not in {"LIGHT", "MESH"}:
            continue
        identity = blender_id(obj, "OBJECT")
        if identities is not None and identity not in identities:
            continue
        data = obj.data
        if object_type == "MESH":
            payload = (
                object_type,
                str(obj.name_full),
                _session_uid(data, "MESH"),
                str(data.name_full),
                tuple(tuple(float(value) for value in vertex.co) for vertex in data.vertices),
                tuple((tuple(int(value) for value in polygon.vertices),
                       int(getattr(polygon, "material_index", 0)),
                       bool(getattr(polygon, "use_smooth", False)))
                      for polygon in data.polygons),
                tuple(
                    _session_uid(material, "MATERIAL") if material is not None else 0
                    for material in data.materials
                ),
                simready_physics_conversion.fingerprint_for_object(obj),
                _session_uid(obj.parent, "OBJECT") if obj.parent is not None else 0,
                _mesh_render_inputs(obj, data),
            )
        else:
            light_type = str(getattr(data, "type", "") or "").strip().upper()
            light_shape = str(getattr(data, "shape", "") or "").strip().upper()
            payload = (
                object_type,
                str(obj.name_full),
                _session_uid(data, "LIGHT"),
                str(data.name_full),
                light_type,
                light_shape,
                authored_light_form(light_type, light_shape),
                _session_uid(obj.parent, "OBJECT") if obj.parent is not None else 0,
            )
        result[identity] = sha256(repr(payload).encode("utf-8")).hexdigest()
    return result


def _proven_non_geometry_objects(scene, generation, affected, object_ids, provenance):
    """Use explicit notification flags plus accepted authored controls.

    Unknown/geometry notifications and changed bindings, modifiers, visibility,
    namespace/parent or semantic/physics inputs keep the element-hash path.
    Shared Mesh notifications apply to every owner, and cannot be overridden by
    a weaker Object notification. Transform-dependent modifiers stay guarded.
    """
    previous = getattr(generation, "mesh_shading_inputs", {})
    fingerprints = getattr(generation, "topology_fingerprints", {})
    affected_set = set(affected)
    result = set()
    for obj in getattr(scene, "objects", ()):
        if str(getattr(obj, "type", "")) != "MESH":
            continue
        identity = blender_id(obj, "OBJECT")
        if identity not in object_ids or identity not in previous or identity not in fingerprints:
            continue
        sources = [source for source in (identity, blender_id(obj.data, "MESH"))
                   if source in affected_set]
        if not sources:
            continue
        notifications = [provenance.get(source, SceneUpdateProvenance()) for source in sources]
        if any(note.geometry is not False or note.transform is None or note.shading is None
               for note in notifications):
            continue
        # A pose can change evaluated modifier output even with stable inputs.
        if any(note.transform is True for note in notifications) and getattr(obj, "modifiers", ()):
            continue
        if previous[identity] == _mesh_shading_inputs(obj):
            result.add(identity)
    return result


def _has_only_geometryless_node_modifiers(obj):
    modifiers = tuple(getattr(obj, "modifiers", ()))
    if not modifiers:
        return False
    for modifier in modifiers:
        if str(getattr(modifier, "type", "")) != "NODES":
            return False
        tree = getattr(modifier, "node_group", None)
        outputs = [node for node in getattr(tree, "nodes", ())
                   if str(getattr(node, "type", "")) == "GROUP_OUTPUT"]
        # Mesh evaluation alone cannot prove curves/volumes are absent.
        # Admit unlinked geometry or explicit input pass-through only.
        if not outputs:
            return False
        for node in outputs:
            for socket in getattr(node, "inputs", ()):
                if str(getattr(socket, "type", "")) != "GEOMETRY":
                    continue
                seen = set()
                while bool(getattr(socket, "is_linked", False)):
                    links = tuple(getattr(socket, "links", ()))
                    if len(links) != 1 or id(socket) in seen:
                        return False
                    seen.add(id(socket))
                    source = links[0].from_node
                    if str(getattr(source, "type", "")) == "GROUP_INPUT":
                        if str(getattr(links[0].from_socket, "type", "")) != "GEOMETRY":
                            return False
                        break
                    if str(getattr(source, "type", "")) != "REROUTE":
                        return False
                    inputs = tuple(getattr(source, "inputs", ()))
                    if len(inputs) != 1:
                        return False
                    socket = inputs[0]
    return True


def _empty_evaluated_mesh_proofs(scene, identities=None):
    """Bounded proof for empty source meshes with identity GN modifiers.

    Static output wiring proves these modifiers cannot create any geometry
    component. Reject object instancing and particles. There is no reason to
    evaluate the entire dependency graph or enumerate house instances merely
    to confirm an empty mesh passed through unchanged. Keep semantic, binding,
    parent, visibility and physics controls in the proof.
    """
    result = {}
    for obj in getattr(scene, "objects", ()):
        if str(getattr(obj, "type", "")) != "MESH":
            continue
        identity = blender_id(obj, "OBJECT")
        if identities is not None and identity not in identities:
            continue
        if (not getattr(obj, "modifiers", ())
                or not hasattr(obj.data, "vertices") or not hasattr(obj.data, "polygons")
                or len(obj.data.vertices) or len(obj.data.polygons)
                or str(getattr(obj, "instance_type", "NONE")) != "NONE"
                or bool(getattr(obj, "is_instancer", False))
                or bool(getattr(obj, "particle_systems", ()))
                or not _has_only_geometryless_node_modifiers(obj)):
            continue
        guard = list(_mesh_shading_inputs(obj))
        render_inputs = list(guard[-1])
        render_inputs[1] = ()  # Static identity proof replaces modifier inputs.
        guard[-1] = tuple(render_inputs)
        guard.append(bool(getattr(obj, "visible_shadow", True)))
        guard.append(bool(getattr(obj, "visible_camera", True)))
        result[identity] = sha256(repr(tuple(guard)).encode("utf-8")).hexdigest()
    return result


def _affected_object_ids(
    scene: Any,
    generation: SceneGeneration,
    affected: tuple[BlenderId, ...],
) -> set[BlenderId]:
    object_ids = {
        identity for identity in affected if identity.kind == "OBJECT"
    }
    for data_kind in ("LIGHT", "MESH"):
        data_uids = {
            identity.session_uid
            for identity in affected
            if identity.kind == data_kind
        }
        if not data_uids:
            continue
        object_ids.update(
            identity
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "OBJECT"
            and mapping.blender_id_type == data_kind
            and mapping.data_session_uid in data_uids
        )
        object_ids.update(
            blender_id(obj, "OBJECT")
            for obj in getattr(scene, "objects", ())
            if str(getattr(obj, "type", "")) == data_kind
            and _session_uid(obj.data, data_kind) in data_uids
        )
    return object_ids


def _changed_topology_notifications(generation, affected, current):
    previous = getattr(generation, "topology_fingerprints", {})
    def unchanged(identity):
        return identity in previous and identity in current and previous[identity] == current[identity]
    result = []
    for identity in affected:
        if identity.kind == "OBJECT" and unchanged(identity):
            continue
        if identity.kind in {"MESH", "LIGHT"}:
            owners = {owner for owner, mapping in getattr(generation, "blender_prim_paths", {}).items()
                      if owner.kind == "OBJECT" and mapping.blender_id_type == identity.kind
                      and mapping.data_session_uid == identity.session_uid}
            # An unknown/new data ID must still reach normal reconciliation.
            if owners and all(unchanged(owner) for owner in owners):
                continue
        result.append(identity)
    return tuple(result)


def _is_noop_topology_edit(
    generation: SceneGeneration,
    affected: tuple[BlenderId, ...],
    current: Mapping[BlenderId, str],
) -> bool:
    # Mesh fingerprints contain material identities, not their shader graphs.
    # A node connect/disconnect also dirties unchanged objects in Blender;
    # those object fingerprints cannot prove a material edit is a no-op.
    if any(identity.kind in {"MATERIAL", "WORLD"} for identity in affected):
        return False
    object_ids = {
        identity
        for identity in affected
        if identity.kind == "OBJECT"
    }
    for data_kind in ("LIGHT", "MESH"):
        data_uids = {
            identity.session_uid
            for identity in affected
            if identity.kind == data_kind
        }
        if not data_uids:
            continue
        object_ids.update(
            identity
            for identity, mapping in generation.blender_prim_paths.items()
            if identity.kind == "OBJECT"
            and mapping.blender_id_type == data_kind
            and mapping.data_session_uid in data_uids
        )
    if not object_ids:
        return False
    previous = generation.topology_fingerprints
    return all(
        identity in previous
        and identity in current
        and previous[identity] == current[identity]
        for identity in object_ids
    )


def _reconcile_visibility_generation(work_directory, scene, predecessor, *, number):
    """Durable value-only candidate: reuse geometry, paths and topology layers."""
    candidate = Path(tempfile.mkdtemp(prefix=f".candidate-{number:06d}-", dir=work_directory))
    try:
        stock = candidate / "stock.usda"
        stock_stage = _write_layered_generation(stock, (), predecessor.topology_deltas,
                                 Path(predecessor.base_usd_path))
        records = _compile_add_on_opinions(scene, predecessor.blender_prim_paths, stock,
            **({"stock_stage": stock_stage} if hasattr(stock_stage, "GetPrimAtPath") else {}))
        stock_stage = None
        digest = _sparse_generation_digest(predecessor.base_digest,
            predecessor.blender_prim_paths, predecessor.topology_deltas, records)
        composed_stage = _write_layered_generation(candidate / "composed.usda", records,
                                 predecessor.topology_deltas, Path(predecessor.base_usd_path),
                                 predecessor=predecessor)
        _validate_composed_generation(candidate / "composed.usda",
                                     predecessor.blender_prim_paths, set(), set(),
            **({"stage": composed_stage} if hasattr(composed_stage, "GetPrimAtPath") else {}))
        composed_stage = None
        destination = work_directory / f"generation-{number:06d}-{digest[:16]}"
        candidate.replace(destination)
        return replace(predecessor, number=number, digest=digest,
            predecessor_number=predecessor.number, usd_path=str(destination / "composed.usda"),
            opinion_records=records, sparse_change=_opinion_change(predecessor, records),
            object_visibility=MappingProxyType(_object_visibility(scene)),
            unadmitted_rendering_overrides=_unadmitted_rendering_overrides(scene),
            mesh_shading_inputs=MappingProxyType(_mesh_shading_inputs_for_scene(scene)),
            diagnostics=MappingProxyType({**dict(predecessor.diagnostics),
                "mode": "visibility_only", "complete_export": False,
                "selected_objects_only": False}))
    except Exception:
        shutil.rmtree(candidate, ignore_errors=True)
        raise



def _can_reconcile_canonical_world(scene: Any, predecessor: SceneGeneration) -> bool:
    """Only our complete textured dome opinion can replace a World sparsely."""
    from .world_dome_conversion import environment_texture_dome
    world = getattr(scene, "world", None)
    if world is None or environment_texture_dome(world) is None:
        return False
    worlds = [mapping for identity, mapping in getattr(predecessor, "blender_prim_paths", {}).items()
              if identity.kind == "WORLD"]
    return (len(worlds) == 1
            and worlds[0].object_path == DEFAULT_DOME_OWNER_PATH
            and worlds[0].schema_path == DEFAULT_DOME_OWNER_PATH
            and any(record.usd_prim_path == DEFAULT_DOME_OWNER_PATH
                    and record.diagnostics.get("source") == "world_environment"
                    for record in getattr(predecessor, "opinion_records", ())))


def _refresh_canonical_world_mapping(scene, predecessor, mappings, affected):
    if not any(identity.kind == "WORLD" for identity in affected):
        return mappings
    if not _can_reconcile_canonical_world(scene, predecessor):
        raise SceneGenerationError("unsupported sparse World environment")
    world = scene.world
    previous = next(mapping for identity, mapping in predecessor.blender_prim_paths.items()
                    if identity.kind == "WORLD")
    refreshed = {identity: mapping for identity, mapping in mappings.items()
                 if identity.kind != "WORLD"}
    refreshed[blender_id(world, "WORLD")] = replace(previous, blender_id_name=str(world.name_full))
    return refreshed


def _reconcile_sparse_generation(
    work_directory: Path,
    scene: Any,
    predecessor: SceneGeneration,
    affected: tuple[BlenderId, ...],
    *,
    number: int | None = None,
) -> SceneGeneration:
    started = time.perf_counter()
    number = predecessor.number + 1 if number is None else number
    work_directory.mkdir(parents=True, exist_ok=True)
    candidate = Path(
        tempfile.mkdtemp(prefix=f".candidate-{number:06d}-", dir=work_directory)
    )
    timings: dict[str, float] = {}
    try:
        phase = time.perf_counter()
        objects, removed_object_ids, affected_material_ids = _sparse_object_closure(
            scene,
            predecessor,
            affected,
        )
        timings["affected_id_discovery_ms"] = (time.perf_counter() - phase) * 1000.0

        phase = time.perf_counter()
        selected_path = candidate / "selected.usda"
        selected_mappings: dict[BlenderId, BlenderPrimPath] = {}
        if objects:
            _selected_stock_export(scene, objects, selected_path)
            selected_mappings = _validated_blender_prim_paths(scene, selected_path)
        timings["selected_export_ms"] = (time.perf_counter() - phase) * 1000.0

        phase = time.perf_counter()
        delta_path = candidate / "delta.usda"
        rewritten, replaced_paths, deleted_paths = _write_scene_topology_delta(
            selected_path if objects else None,
            delta_path,
            number,
            selected_mappings,
            predecessor,
            removed_object_ids,
            affected_material_ids,
        )
        mappings = _updated_generation_mappings(
            predecessor,
            rewritten,
            removed_object_ids,
            affected_material_ids,
        )
        mappings = _refresh_canonical_world_mapping(scene, predecessor, mappings, affected)
        timings["delta_construction_ms"] = (time.perf_counter() - phase) * 1000.0

        delta_digest = sha256(delta_path.read_bytes()).hexdigest()
        pending_delta = SceneTopologyDelta(
            usd_path=str(delta_path),
            affected_ids=affected,
            replaced_prim_paths=tuple(sorted(replaced_paths)),
            deleted_prim_paths=tuple(sorted(deleted_paths)),
            selected_object_count=len(objects),
            digest=delta_digest,
        )
        topology_deltas = predecessor.topology_deltas + (pending_delta,)
        if len(topology_deltas) >= _DELTA_COMPACTION_COUNT:
            compacted_path = candidate / "delta-compacted.usda"
            _compact_topology_deltas(compacted_path, topology_deltas)
            topology_deltas = (
                SceneTopologyDelta(
                    usd_path=str(compacted_path),
                    affected_ids=tuple(
                        sorted(
                            {
                                identity
                                for delta in topology_deltas
                                for identity in delta.affected_ids
                            }
                        )
                    ),
                    replaced_prim_paths=tuple(
                        sorted(
                            {
                                path
                                for delta in topology_deltas
                                for path in delta.replaced_prim_paths
                            }
                        )
                    ),
                    deleted_prim_paths=tuple(
                        sorted(
                            {
                                path
                                for delta in topology_deltas
                                for path in delta.deleted_prim_paths
                            }
                        )
                    ),
                    selected_object_count=sum(
                        delta.selected_object_count for delta in topology_deltas
                    ),
                    digest=sha256(compacted_path.read_bytes()).hexdigest(),
                ),
            )

        phase = time.perf_counter()
        stock_root = candidate / "stock.usda"
        stock_stage = _write_layered_generation(
            stock_root,
            (),
            topology_deltas,
            Path(predecessor.base_usd_path),
        )
        # The canonical environment is defined by our World opinion, not the
        # stock exporter. Validate stock-owned identities now; the complete
        # mapping, including that additive World, is validated after opinions.
        stock_mappings = {
            identity: mapping for identity, mapping in mappings.items()
            if not (identity.kind == "WORLD"
                    and mapping.object_path == DEFAULT_DOME_OWNER_PATH)
        }
        shared_stock = {"stage": stock_stage} if hasattr(stock_stage, "GetPrimAtPath") else {}
        _validate_composed_generation(stock_root, stock_mappings, replaced_paths, deleted_paths, **shared_stock)
        timings["composition_validation_ms"] = (time.perf_counter() - phase) * 1000.0

        phase = time.perf_counter()
        records = _compile_add_on_opinions(scene, mappings, stock_root,
            **({"stock_stage": stock_stage} if shared_stock else {}))
        shared_stock.clear()
        stock_stage = None
        timings["add_on_opinion_compile_ms"] = (time.perf_counter() - phase) * 1000.0
        materialization_started = time.perf_counter()
        digest = _sparse_generation_digest(
            predecessor.base_digest,
            mappings,
            topology_deltas,
            records,
        )
        destination = work_directory / f"generation-{number:06d}-{digest[:16]}"
        composed_path = candidate / "composed.usda"
        composed_stage = _write_layered_generation(
            composed_path,
            records,
            topology_deltas,
            Path(predecessor.base_usd_path),
            predecessor=predecessor,
        )
        timings["opinion_layer_materialization_ms"] = (time.perf_counter() - materialization_started) * 1000.0
        validation_started = time.perf_counter()
        _validate_composed_generation(composed_path, mappings, replaced_paths, deleted_paths,
            **({"stage": composed_stage} if hasattr(composed_stage, "GetPrimAtPath") else {}))
        composed_stage = None
        timings["opinion_composition_validation_ms"] = (time.perf_counter() - validation_started) * 1000.0
        timings["add_on_opinions_and_materialization_ms"] = (
            time.perf_counter() - phase
        ) * 1000.0
        candidate.replace(destination)
    except Exception as exc:
        failed = work_directory / candidate.name.replace(
            ".candidate-",
            "failed-candidate-",
            1,
        )
        try:
            candidate.replace(failed)
        except OSError:
            failed = candidate
        diagnostics = tuple(getattr(exc, "diagnostics", ())) + (
            {
                "candidate_artifact": str(failed),
                "reason": "sparse_candidate_construction_failed",
            },
        )
        raise SceneGenerationError(str(exc), diagnostics) from exc

    promoted_deltas = tuple(
        SceneTopologyDelta(
            usd_path=(
                str(destination / Path(delta.usd_path).name)
                if Path(delta.usd_path).parent == candidate
                else delta.usd_path
            ),
            affected_ids=delta.affected_ids,
            replaced_prim_paths=delta.replaced_prim_paths,
            deleted_prim_paths=delta.deleted_prim_paths,
            selected_object_count=delta.selected_object_count,
            digest=delta.digest,
        )
        for delta in topology_deltas
    )
    total_ms = (time.perf_counter() - started) * 1000.0
    timings["total_ms"] = total_ms
    timings["residual_ms"] = max(
        0.0,
        total_ms - sum(value for key, value in timings.items()
                       if key not in {"total_ms", "add_on_opinion_compile_ms",
                                      "opinion_layer_materialization_ms", "opinion_composition_validation_ms"}),
    )
    fingerprints = dict(predecessor.topology_fingerprints)
    for identity in removed_object_ids:
        fingerprints.pop(identity, None)
    rewritten_object_ids = {
        identity for identity in rewritten if identity.kind == "OBJECT"
    }
    fingerprints.update(_topology_fingerprints(scene, rewritten_object_ids))
    return SceneGeneration(
        number=number,
        digest=digest,
        predecessor_number=predecessor.number,
        base_digest=predecessor.base_digest,
        usd_path=str(destination / "composed.usda"),
        blender_prim_paths=MappingProxyType(dict(mappings)),
        opinion_records=records,
        sparse_change=_opinion_change(predecessor, records),
        base_usd_path=predecessor.base_usd_path,
        topology_deltas=promoted_deltas,
        topology_fingerprints=MappingProxyType(fingerprints),
        particle_geometry_owners=predecessor.particle_geometry_owners,
        unadmitted_rendering_overrides=_unadmitted_rendering_overrides(scene),
        empty_evaluated_mesh_proofs=MappingProxyType({
            **{identity: signature for identity, signature in predecessor.empty_evaluated_mesh_proofs.items()
               if identity not in removed_object_ids and identity not in rewritten_object_ids},
            **_empty_evaluated_mesh_proofs(scene, rewritten_object_ids),
        }),
        mesh_shading_inputs=MappingProxyType({
            **{identity: values for identity, values in predecessor.mesh_shading_inputs.items()
               if identity not in removed_object_ids},
            **_mesh_shading_inputs_for_scene(scene, rewritten_object_ids),
        }),
        scene_object_count=len(getattr(scene, "objects", ())),
        unsupported_light_values=MappingProxyType(_unsupported_light_values(scene)),
        material_graph_fingerprints=MappingProxyType({
            **{identity: signature for identity, signature in predecessor.material_graph_fingerprints.items()
               if identity in mappings},
            **_material_graph_fingerprints(scene, {
                identity: mapping for identity, mapping in mappings.items()
                if identity in rewritten or identity not in predecessor.blender_prim_paths
            }),
        }),
        diagnostics=MappingProxyType(
            {
                "mode": "sparse_reconciliation",
                "physics_initial_frame": float(getattr(scene, "frame_current", 1)) + float(getattr(scene, "frame_subframe", 0)),
                "physics_initial_fps": float(getattr(getattr(scene, "render", None), "fps", 24)) / float(getattr(getattr(scene, "render", None), "fps_base", 1)),
                "complete_export": False,
                "selected_objects_only": bool(objects),
                "affected_ids": tuple(
                    {"kind": identity.kind, "session_uid": identity.session_uid}
                    for identity in affected
                ),
                **timings,
            }
        ),
        world_session_uid=_world_session_uid(scene),
        object_visibility=MappingProxyType(_object_visibility(scene)),
        shadow_visibility=MappingProxyType(_mesh_shadow_visibility(scene)),
    )


def _sparse_object_closure(
    scene: Any,
    predecessor: SceneGeneration,
    affected: tuple[BlenderId, ...],
) -> tuple[tuple[Any, ...], set[BlenderId], set[BlenderId]]:
    all_current_objects = {
        blender_id(obj, "OBJECT"): obj
        for obj in getattr(scene, "objects", ())
    }
    current_objects = {
        identity: obj
        for identity, obj in all_current_objects.items()
        if str(getattr(obj, "type", "")) in SUPPORTED_OBJECT_TYPES
    }
    previous_objects = {
        identity: mapping
        for identity, mapping in predecessor.blender_prim_paths.items()
        if identity.kind == "OBJECT" and mapping.blender_id_type in SUPPORTED_OBJECT_TYPES
    }
    identities = set(affected)
    selected_ids: set[BlenderId] = set()
    removed_ids: set[BlenderId] = set(previous_objects).difference(current_objects)
    affected_materials = {
        identity for identity in identities if identity.kind == "MATERIAL"
    }
    for identity in identities:
        if identity.kind == "OBJECT":
            obj = current_objects.get(identity)
            previous = previous_objects.get(identity)
            if obj is not None:
                selected_ids.add(identity)
            elif previous is not None:
                removed_ids.add(identity)
            else:
                existing = all_current_objects.get(identity)
                joint = getattr(getattr(getattr(existing, "ov", None), "physics", None), "joint", None)
                if ((joint is not None and getattr(joint, "persistent_id", ""))
                    or (existing is None and any(record.diagnostics.get("joint_session_uid") == str(identity.session_uid)
                                                for record in predecessor.opinion_records))):
                    # Joint-only edits replace complete physics opinions over
                    # retained stock geometry; they never export only the Empty.
                    continue
                reason = (
                    "unsupported_sparse_object_type"
                    if existing is not None
                    else "unsupported_sparse_object"
                )
                raise SceneGenerationError(
                    f"unsupported affected Blender object session_uid {identity.session_uid}",
                    ({"blender_id_kind": identity.kind, "session_uid": identity.session_uid, "reason": reason},),
                )
        elif identity.kind in GEOMETRY_TYPES | {"LIGHT"}:
            data_object_types = ({'CURVE', 'SURFACE', 'FONT'} if identity.kind == 'CURVE' else {identity.kind})
            selected_ids.update(
                object_id
                for object_id, obj in current_objects.items()
                if str(getattr(obj, "type", "")) in data_object_types
                and _session_uid(obj.data, identity.kind) == identity.session_uid
            )
            removed_ids.update(
                object_id
                for object_id, mapping in previous_objects.items()
                if mapping.blender_id_type in data_object_types
                if mapping.data_session_uid == identity.session_uid
                and object_id not in current_objects
            )
        elif identity.kind == "WORLD" and _can_reconcile_canonical_world(scene, predecessor):
            # The complete additive dome opinion owns this supported graph;
            # changing HDR/environment does not dirty object geometry.
            continue
        elif identity.kind == "MATERIAL":
            selected_ids.update(
                object_id
                for object_id, obj in current_objects.items()
                if any(
                    material is not None
                    and _session_uid(material, "MATERIAL") == identity.session_uid
                    for material in object_export_materials(obj)
                )
            )
        else:
            raise SceneGenerationError(
                f"unsupported sparse Blender ID kind {identity.kind!r}",
                ({"blender_id_kind": identity.kind, "session_uid": identity.session_uid, "reason": "unsupported_sparse_id_kind"},),
            )

    binding_users = None
    while True:
        previous_selection = set(selected_ids)
        previous_materials = set(affected_materials)
        if affected_materials and binding_users is None:
            binding_users = _accepted_material_binding_users(predecessor)
        selected_ids.update(owner for material in affected_materials
                            for owner in (binding_users or {}).get(material, ())
                            if owner in current_objects)
        # Materials encountered through a selected object's slots are export
        # dependencies, not dirty IDs. Even a previously unmapped material may
        # be an unchanged linked/coalesced export alias or a hidden asset. New
        # definitions travel with selected stock geometry, without flooding its
        # other users. Only explicitly dirty materials retire accepted paths.
        if affected_materials:
            material_uids = {
                identity.session_uid for identity in affected_materials
            }
            selected_ids.update(
                object_id
                for object_id, obj in current_objects.items()
                if any(
                    material is not None
                    and _session_uid(material, "MATERIAL") in material_uids
                    for material in object_export_materials(obj)
                )
            )

        for identity in tuple(selected_ids):
            parent = current_objects[identity].parent
            while parent is not None:
                parent_id = blender_id(parent, "OBJECT")
                if parent_id not in current_objects:
                    raise SceneGenerationError(
                        f"unsupported parent for affected Blender object session_uid {identity.session_uid}",
                        ({
                            "blender_id_kind": identity.kind,
                            "session_uid": identity.session_uid,
                            "parent_session_uid": parent_id.session_uid,
                            "reason": "unsupported_sparse_parent",
                        },),
                    )
                selected_ids.add(parent_id)
                parent = parent.parent
        selected_roots = {
            _world_child_path(previous_objects[identity].object_path)
            for identity in selected_ids
            if identity in previous_objects
        }
        selected_ids.update(
            identity
            for identity, mapping in previous_objects.items()
            if identity in current_objects
            and _world_child_path(mapping.object_path) in selected_roots
        )
        if (
            selected_ids == previous_selection
            and affected_materials == previous_materials
        ):
            break
    objects = tuple(
        current_objects[identity]
        for identity in sorted(selected_ids)
    )
    return objects, removed_ids, affected_materials


def _accepted_material_binding_users(predecessor: SceneGeneration) -> dict[BlenderId, set[BlenderId]]:
    """Retire material paths only after including all accepted USD binding users.

    Stock instance bindings can refer to shared material paths irrespective of
    current Blender library/slot identity. The accepted graph is authoritative
    for those existing references, including nested instance proxies.
    """
    from pxr import Sdf, Usd
    source = getattr(predecessor, "usd_path", None)
    if source is None:
        return {}
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise SceneGenerationError("Accepted material binding stage is unavailable")
    owners = {Sdf.Path(mapping.object_path): identity
              for identity, mapping in predecessor.blender_prim_paths.items()
              if identity.kind == "OBJECT"}
    materials = {Sdf.Path(mapping.object_path): identity
                 for identity, mapping in predecessor.blender_prim_paths.items()
                 if identity.kind == "MATERIAL"}
    users = {}
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        bindings = [target.GetPrimPath() for relationship in prim.GetRelationships()
                    if str(relationship.GetName()).startswith("material:binding")
                    for target in relationship.GetTargets()]
        if not bindings:
            continue
        owner_ids = {owners[path] for path in prim.GetPath().GetPrefixes() if path in owners}
        for target in bindings:
            for path in target.GetPrefixes():
                material = materials.get(path)
                if material is not None:
                    users.setdefault(material, set()).update(owner_ids)
    return users


def _selected_stock_export(scene: Any, objects: tuple[Any, ...], path: Path) -> None:
    module = _blender_module_provider()
    context = getattr(module, "context", None)
    if context is None:
        raise SceneGenerationError("Blender context is unavailable for sparse scene export")
    selected = tuple(getattr(context, "selected_objects", ()))
    view_layer_objects = getattr(getattr(context, "view_layer", None), "objects", None)
    active = getattr(view_layer_objects, "active", None)
    active_mode = str(getattr(active, "mode", "OBJECT")) if active is not None else "OBJECT"
    # Sparse exports need the same temporary restrictions as full exports
    # before selection, not only once usd_export starts. Restore selection
    # while objects/collections are still selectable, then restore visibility.
    from . import interactive_operator_state
    with interactive_operator_state.suppress_interactive_edit_bridge(), _temporary_export_visibility(scene):
        try:
            if active is not None and active_mode != "OBJECT":
                module.ops.object.mode_set(mode="OBJECT")
            for obj in selected:
                obj.select_set(False)
            for obj in objects:
                if bool(getattr(obj, "hide_render", False)):
                    raise SceneGenerationError(
                        f"affected Blender object {obj.name_full!r} is hidden from render export"
                    )
                obj.select_set(True)
            if view_layer_objects is not None:
                view_layer_objects.active = objects[0]
            _stock_export(scene, path, selected_objects_only=True)
        finally:
            for obj in objects:
                obj.select_set(False)
            for obj in selected:
                obj.select_set(True)
            if view_layer_objects is not None:
                view_layer_objects.active = active
            if active is not None and active_mode != "OBJECT":
                module.ops.object.mode_set(mode=active_mode)


def _write_scene_topology_delta(
    selected_path: Path | None,
    delta_path: Path,
    number: int,
    selected_mappings: Mapping[BlenderId, BlenderPrimPath],
    predecessor: SceneGeneration,
    removed_object_ids: set[BlenderId],
    affected_material_ids: set[BlenderId],
) -> tuple[dict[BlenderId, BlenderPrimPath], set[str], set[str]]:
    from pxr import Sdf, Usd  # type: ignore

    if selected_path is None:
        stage = Usd.Stage.CreateNew(str(delta_path))
    else:
        shutil.copy2(selected_path, delta_path)
        stage = Usd.Stage.Open(str(delta_path))
    if stage is None:
        raise SceneGenerationError("sparse scene topology delta could not be opened")
    generation_root = f"/World/__ovrtx/generation_{number:06d}"
    stage.DefinePrim(generation_root, "Scope")
    move_roots: dict[str, str] = {}
    for identity, mapping in selected_mappings.items():
        if identity.kind != "OBJECT":
            continue
        source_root = _world_child_path(mapping.object_path)
        move_roots[source_root] = generation_root + source_root[len("/World") :]
    for source, destination in sorted(move_roots.items()):
        _move_prim(stage, source, destination)

    # Retarget stock bindings before discarding duplicate unchanged materials.
    # NamespaceEditor also rewrites shader connections and instance bindings.
    material_root = "/World/_materials"
    material_destination = f"{generation_root}/_materials"
    material_moves = {}
    retained_materials = set()
    for identity, mapping in selected_mappings.items():
        if identity.kind != "MATERIAL":
            continue
        accepted = predecessor.blender_prim_paths.get(identity)
        if accepted is not None and identity not in affected_material_ids:
            destination = accepted.object_path
            retained_materials.add(destination)
        else:
            if not mapping.object_path.startswith(material_root + "/"):
                raise SceneGenerationError("stock material is outside /World/_materials")
            destination = material_destination + mapping.object_path[len(material_root):]
        material_moves[identity] = destination
    # Use a private staging namespace to handle swapped stock material names
    # without overwriting another material before its bindings are rewritten.
    for identity, destination in material_moves.items():
        source = selected_mappings[identity].object_path
        temporary = f"{generation_root}/__material_moves/m_{identity.session_uid}"
        _move_prim(stage, source, temporary)
    for identity, destination in material_moves.items():
        _move_prim(stage, f"{generation_root}/__material_moves/m_{identity.session_uid}", destination)
    for path in retained_materials:
        stage.RemovePrim(path)
    stage.RemovePrim(f"{generation_root}/__material_moves")

    rewritten: dict[BlenderId, BlenderPrimPath] = {}
    for identity, mapping in selected_mappings.items():
        if identity.kind == "OBJECT":
            rewritten[identity] = _rewritten_mapping(mapping, move_roots)
        elif identity in material_moves:
            destination = material_moves[identity]
            rewritten[identity] = BlenderPrimPath(
                mapping.blender_id_name, mapping.blender_id_type, destination,
                destination + mapping.schema_path[len(mapping.object_path):],
                mapping.data_session_uid,
            )
        else:
            rewritten[identity] = mapping

    replaced_paths: set[str] = set()
    deleted_paths: set[str] = set()
    for identity, mapping in predecessor.blender_prim_paths.items():
        if identity in removed_object_ids:
            _deactivate_prim(stage, mapping.object_path)
            deleted_paths.add(mapping.object_path)
        elif identity in rewritten and identity.kind == "OBJECT":
            _deactivate_prim(stage, mapping.object_path)
            replaced_paths.add(mapping.object_path)
        elif identity in affected_material_ids:
            _deactivate_prim(stage, mapping.object_path)
            replaced_paths.add(mapping.object_path)
    stage.GetRootLayer().Save()
    return rewritten, replaced_paths, deleted_paths


def _world_child_path(path: str) -> str:
    parts = [part for part in str(path).split("/") if part]
    if len(parts) < 2 or parts[0] != "World":
        raise SceneGenerationError(f"stock-exported path is outside /World: {path}")
    # A sparse generation namespace groups unrelated assemblies. It is not
    # an object hierarchy dependency: preserve the actual stock root beneath
    # it instead of pulling every previously edited assembly into each update.
    if len(parts) >= 4 and parts[1] == "__ovrtx" and parts[2].startswith("generation_"):
        return "/" + "/".join(parts[:4])
    return f"/World/{parts[1]}"


def _deactivate_prim(stage: Any, path: str) -> None:
    from pxr import Sdf  # type: ignore

    prim = Sdf.CreatePrimInLayer(stage.GetRootLayer(), Sdf.Path(path))
    prim.specifier = Sdf.SpecifierOver
    prim.SetInfo("active", False)


def _move_prim(stage: Any, source: str, destination: str) -> None:
    from pxr import Sdf, Usd  # type: ignore

    parent = Sdf.Path(destination).GetParentPath()
    prefixes = parent.GetPrefixes()
    for prefix in prefixes:
        if prefix == Sdf.Path.absoluteRootPath:
            continue
        if not stage.GetPrimAtPath(prefix):
            stage.DefinePrim(prefix, "Scope")
    editor = Usd.NamespaceEditor(stage)
    if not editor.MovePrimAtPath(source, destination):
        raise SceneGenerationError(f"could not queue sparse USD move {source} -> {destination}")
    allowed = editor.CanApplyEdits()
    if not allowed or not editor.ApplyEdits():
        raise SceneGenerationError(f"could not apply sparse USD move {source} -> {destination}: {allowed}")


def _rewritten_mapping(
    mapping: BlenderPrimPath,
    moves: Mapping[str, str],
) -> BlenderPrimPath:
    root = _world_child_path(mapping.object_path)
    destination = moves[root]
    return BlenderPrimPath(
        mapping.blender_id_name,
        mapping.blender_id_type,
        destination + mapping.object_path[len(root) :],
        destination + mapping.schema_path[len(root) :],
        mapping.data_session_uid,
    )


def _updated_generation_mappings(
    predecessor: SceneGeneration,
    rewritten: Mapping[BlenderId, BlenderPrimPath],
    removed_object_ids: set[BlenderId],
    affected_material_ids: set[BlenderId],
) -> dict[BlenderId, BlenderPrimPath]:
    mappings = dict(predecessor.blender_prim_paths)
    for identity in removed_object_ids.union(affected_material_ids):
        mappings.pop(identity, None)
    for identity in rewritten:
        if identity.kind == "OBJECT" or identity in affected_material_ids or identity not in mappings:
            mappings[identity] = rewritten[identity]
    return mappings


def _write_layered_generation(
    destination: Path,
    records: tuple[AddOnUsdOpinionRecord, ...],
    deltas: tuple[SceneTopologyDelta, ...],
    base_path: Path,
    *,
    predecessor: SceneGeneration | None = None,
) -> Any:
    from pxr import Sdf, Usd  # type: ignore

    reusable = _reusable_opinion_layers(predecessor)
    opinion_paths = []
    for index, record in enumerate(records):
        retained = reusable.get((record.usd_prim_path, record.digest))
        if retained is not None:
            # Keep the accepted file at its original location. Relative assets
            # retain their resolver context and snapshots can reuse its fields.
            opinion_paths.append(retained)
            continue
        path = destination.parent / f"opinion-{index:04d}.usda"
        layer = Sdf.Layer.CreateNew(str(path))
        if not layer.ImportFromString(record.layer_text) or not layer.Save():
            raise SceneGenerationError(f"could not persist add-on opinion layer {path}")
        opinion_paths.append(path)
    root = Sdf.Layer.CreateNew(str(destination))
    _copy_stock_stage_metadata(root, base_path)
    root.subLayerPaths = [
        _relative_sublayer(destination, path)
        for path in opinion_paths
    ] + [
        _relative_sublayer(destination, Path(delta.usd_path))
        for delta in reversed(deltas)
    ] + [_relative_sublayer(destination, base_path)]
    root.Save()
    stage = Usd.Stage.Open(str(destination))
    if stage is None:
        raise SceneGenerationError(f"layered scene generation could not be opened at {destination}")
    return stage


def _reusable_opinion_layers(predecessor):
    """Reuse only byte-identical immutable accepted opinion files, never copies."""
    if predecessor is None:
        return {}
    from pxr import Sdf
    root = Sdf.Layer.FindOrOpen(str(predecessor.usd_path))
    records = predecessor.opinion_records
    if root is None or root.dirty or len(root.subLayerPaths) < len(records):
        return {}
    retained = {}
    for record, relative in zip(records, root.subLayerPaths):
        path = Path(Sdf.ComputeAssetPathRelativeToLayer(root, relative))
        try:
            if path.read_bytes() != record.layer_text.encode('utf-8'):
                continue
        except OSError:
            continue
        layer = Sdf.Layer.Find(path.as_posix())
        if layer is not None and layer.dirty:
            continue
        retained[(record.usd_prim_path, record.digest)] = path
    return retained


def _compact_topology_deltas(
    destination: Path,
    deltas: tuple[SceneTopologyDelta, ...],
) -> None:
    from pxr import Sdf, Usd, UsdUtils  # type: ignore

    root = Sdf.Layer.CreateAnonymous(".usda")
    root.subLayerPaths = [delta.usd_path for delta in reversed(deltas)]
    stage = Usd.Stage.Open(root)
    if stage is None:
        raise SceneGenerationError("scene topology deltas could not be opened for compaction")

    def resolve_asset(source_layer: Any, asset_path: str) -> str:
        if not asset_path or os.path.isabs(asset_path):
            return asset_path
        source_path = Path(str(getattr(source_layer, "realPath", "") or ""))
        if not source_path:
            return asset_path
        resolved = (source_path.parent / asset_path).resolve()
        return os.path.relpath(resolved, destination.parent).replace(os.sep, "/")

    flattened = UsdUtils.FlattenLayerStack(stage, resolve_asset)
    if not flattened.Export(str(destination)):
        raise SceneGenerationError(f"scene topology deltas could not be compacted to {destination}")


def _relative_sublayer(root: Path, layer: Path) -> str:
    return os.path.relpath(layer, root.parent).replace(os.sep, "/")


def _validate_composed_generation(
    path: Path,
    mappings: Mapping[BlenderId, BlenderPrimPath],
    replaced_paths: set[str],
    deleted_paths: set[str],
    *, stage: Any = None,
) -> None:
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(path)) if stage is None else stage
    if stage is None:
        raise SceneGenerationError(f"composed sparse scene generation could not be opened: {path}")
    root = stage.GetRootLayer()
    if Path(root.realPath or root.identifier).resolve() != Path(path).resolve():
        raise SceneGenerationError('provided validation stage does not match candidate path')
    active_object_uids: dict[int, list[str]] = {}
    active_material_uids: dict[int, list[str]] = {}
    for prim in stage.Traverse():
        attribute = prim.GetAttribute(_OBJECT_UID_ATTRIBUTE)
        value = attribute.Get() if attribute else None
        if value is not None:
            active_object_uids.setdefault(int(value), []).append(str(prim.GetPath()))
        material_attribute = prim.GetAttribute(_MATERIAL_UID_ATTRIBUTE)
        material_value = material_attribute.Get() if material_attribute else None
        if material_value is not None:
            active_material_uids.setdefault(int(material_value), []).append(
                str(prim.GetPath())
            )
        for relationship in prim.GetRelationships():
            for target in relationship.GetTargets():
                target_prim = stage.GetPrimAtPath(target.GetPrimPath())
                if not target_prim or not target_prim.IsActive():
                    raise SceneGenerationError(
                        f"scene generation relationship {relationship.GetPath()} targets inactive prim {target}"
                    )
    for identity, mapping in mappings.items():
        prim = stage.GetPrimAtPath(mapping.object_path)
        schema = stage.GetPrimAtPath(mapping.schema_path)
        if not prim or not prim.IsActive() or not schema or not schema.IsActive():
            raise SceneGenerationError(
                f"scene generation mapping for {mapping.blender_id_name!r} is inactive"
            )
        if identity.kind == "OBJECT":
            paths = active_object_uids.get(identity.session_uid, [])
            if paths != [mapping.object_path]:
                raise SceneGenerationError(
                    f"scene generation has {len(paths)} active mappings for Blender object {mapping.blender_id_name!r}",
                    ({"blender_id": mapping.blender_id_name, "session_uid": identity.session_uid, "candidate_paths": tuple(paths), "reason": "invalid_active_object_mapping"},),
                )
        elif identity.kind == "MATERIAL" and identity.session_uid in active_material_uids:
            paths = active_material_uids[identity.session_uid]
            if paths != [mapping.object_path]:
                raise SceneGenerationError(
                    f"scene generation has {len(paths)} active mappings for Blender material {mapping.blender_id_name!r}",
                    ({"blender_id": mapping.blender_id_name, "session_uid": identity.session_uid, "candidate_paths": tuple(paths), "reason": "invalid_active_material_mapping"},),
                )
    mapped_material_uids = {
        identity.session_uid for identity in mappings if identity.kind == "MATERIAL"
    }
    unmapped_material_uids = set(active_material_uids).difference(mapped_material_uids)
    if unmapped_material_uids:
        raise SceneGenerationError(
            "scene generation contains stock materials without Blender mappings",
            ({"session_uids": tuple(sorted(unmapped_material_uids)), "reason": "unmapped_active_material"},),
        )
    for stale_path in replaced_paths.union(deleted_paths):
        prim = stage.GetPrimAtPath(stale_path)
        if prim and prim.IsActive():
            raise SceneGenerationError(f"replaced scene prim remained active: {stale_path}")


def _sparse_generation_digest(
    base_digest: str,
    mappings: Mapping[BlenderId, BlenderPrimPath],
    deltas: tuple[SceneTopologyDelta, ...],
    records: tuple[AddOnUsdOpinionRecord, ...],
) -> str:
    parts = [
        b"scene-generation-v3",
        base_digest.encode("ascii"),
        _mapping_identity_digest(mappings).encode("ascii"),
    ]
    parts.extend(delta.digest.encode("ascii") for delta in deltas)
    parts.extend(record.digest.encode("ascii") for record in records)
    encoded = b"".join(len(part).to_bytes(8, "big") + part for part in parts)
    return sha256(encoded).hexdigest()


def _retained_values_equal(previous, current):
    try:
        return bool(previous == current)
    except (ValueError, TypeError):
        # Ambiguous mutable/array inputs conservatively require replay.
        return False


__all__ = [
    "BlenderId",
    "BlenderPrimPath",
    "AddOnUsdOpinionRecord",
    "SceneGeneration",
    "SceneTopologyDelta",
    "SceneGenerationError",
    "SceneGenerationOwner",
    "blender_id",
]
