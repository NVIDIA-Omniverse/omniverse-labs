# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private stock-exported USD generations for current Blender scenes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from hashlib import sha256
from pathlib import Path
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

from . import curve_widths
from .usd_opinion_records import (
    AddOnUsdOpinionRecord,
    SceneGenerationError,
    SparseAddOnOpinionChange,
)
from .light_value_conversion import authored_light_form
from .world_dome_conversion import DEFAULT_DOME_OWNER_PATH


try:
    import bpy  # type: ignore
except ModuleNotFoundError:
    bpy = None  # type: ignore[assignment]


SUPPORTED_OBJECT_TYPES = frozenset({"CAMERA", "LIGHT", "MESH"})
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
_STOCK_EXPORT_CONTRACT = "blender-usd-v1"


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
        self._closed = False
        self._lock = threading.RLock()

    @property
    @_synchronized
    def current_generation(self) -> SceneGeneration | None:
        return self._current

    @property
    @_synchronized
    def pending_generation(self) -> SceneGeneration | None:
        return self._pending

    @property
    @_synchronized
    def failed_generations(self) -> tuple[SceneGeneration, ...]:
        return tuple(self._failed)

    @_synchronized
    def replace(self, scene: Any) -> SceneGeneration | None:
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
                self._current is not None
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
            _write_layered_generation(candidate / "composed.usda", records, (), stock_root)
            timings["composition_ms"] += (time.perf_counter() - phase) * 1000.0
            phase = time.perf_counter()
            _validate_composed_generation(
                candidate / "composed.usda", mappings, set(), set()
            )
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
            diagnostics=MappingProxyType(
                {
                    "mode": "complete_export",
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
        )
        self._current = generation
        return generation

    @_synchronized
    def reconcile(
        self,
        scene: Any,
        affected_ids: set[BlenderId] | frozenset[BlenderId] | tuple[BlenderId, ...],
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
        if any(identity.kind == "WORLD" for identity in affected):
            predecessor = self._current
            generation = self.replace(scene)
            if generation is not None:
                self._current = predecessor
                self._pending = generation
            return generation
        object_ids = _affected_object_ids(scene, self._current, affected)
        current_fingerprints = _topology_fingerprints(scene, object_ids)
        if _is_noop_topology_edit(self._current, affected, current_fingerprints):
            return None
        generation = _reconcile_sparse_generation(
            self.work_directory,
            scene,
            self._current,
            affected,
            number=self._next_generation_number(),
        )
        self._pending = generation
        return generation

    @_synchronized
    def accept(self, generation: SceneGeneration) -> None:
        if generation is not self._pending:
            raise SceneGenerationError("scene generation candidate is not pending")
        self._current = generation
        self._pending = None
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
            self._attribute_values[
                (identity, field, stable_suffix, value.attribute)
            ] = value

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
    def close(self) -> None:
        if self._closed:
            return
        try:
            shutil.rmtree(self.work_directory)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise SceneGenerationError(
                f"scene generation cleanup failed: {exc}",
                ({"work_directory": str(self.work_directory), "reason": "cleanup_failed"},),
            ) from exc
        self._current = None
        self._pending = None
        self._failed.clear()
        self._transform_values.clear()
        self._attribute_values.clear()
        self._initial_conditions.clear()
        self._closed = True


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
    app = getattr(module, "app", None)
    contract = {
        "schema": _REUSABLE_BASE_SCHEMA,
        "export_contract": _STOCK_EXPORT_CONTRACT,
        "blender": {
            "version": tuple(getattr(app, "version", ())),
            "version_cycle": str(getattr(app, "version_cycle", "")),
            "build_hash": _text_value(getattr(app, "build_hash", "")),
        },
        "source_scene": str(getattr(scene, "name_full", getattr(scene, "name", ""))),
        "frame": int(getattr(scene, "frame_current", 0)),
        "subframe": float(getattr(scene, "frame_subframe", 0.0)),
        "view_layer": str(
            getattr(
                getattr(getattr(module, "context", None), "view_layer", None),
                "name",
                "",
            )
        ),
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
            _temporary_export_identities(scene),
            context.temp_override(scene=scene),
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
    _normalize_stock_export_geometry(path)


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


def _normalize_stock_export_geometry(path: Path) -> None:
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
    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName())
        if type_name == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateDoubleSidedAttr(False).Set(False)
            _quantize_mesh_uvs(prim, Gf, Vt)
            for property_name in (
                "subdivisionScheme",
                "creaseIndices",
                "creaseLengths",
                "creaseSharpnesses",
            ):
                prim.RemoveProperty(property_name)
        elif type_name == "BasisCurves":
            curves = UsdGeom.BasisCurves(prim)
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


def _write_stock_root(destination: Path, binding: Path, base: Path) -> None:
    from pxr import Sdf, Usd  # type: ignore

    root = Sdf.Layer.CreateNew(str(destination))
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
        schema_paths = _schema_paths(stage, object_prim, object_type, data_name)
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
    if world is not None and dome_prim and str(dome_prim.GetTypeName()) == "DomeLight":
        mappings[blender_id(world, "WORLD")] = BlenderPrimPath(
            blender_id_name=str(world.name_full),
            blender_id_type="WORLD",
            object_path=DEFAULT_DOME_OWNER_PATH,
            schema_path=DEFAULT_DOME_OWNER_PATH,
        )
    return mappings


def _compile_add_on_opinions(
    scene: Any,
    mappings: Mapping[BlenderId, BlenderPrimPath],
    stock_usd_path: Path,
) -> tuple[AddOnUsdOpinionRecord, ...]:
    if not hasattr(scene, "ov"):
        return ()
    from . import (
        authoring_properties,
        physics_topology_conversion,
        simready_physics_conversion,
    )
    from . import usd_paths

    records = []
    occupied = _prim_paths(stock_usd_path)
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
        for obj in scene.objects
        if str(getattr(obj, "type", "")) == "MESH"
        for material in (getattr(obj.ov.physics.collision, "physics_material", None),)
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
        physics_material_paths[pointer] = path
        records.append(
            physics_topology_conversion.convert_generation_physics_material(
                material,
                path,
                mapped=mapping is not None,
            )
        )
    for obj in sorted(scene.objects, key=lambda item: str(item.name_full)):
        if str(getattr(obj, "type", "")) != "MESH":
            continue
        if not physics_topology_conversion.object_has_physics_opinion(obj):
            continue
        mapping = mappings.get(BlenderId("OBJECT", _session_uid(obj, "OBJECT")))
        if mapping is None:
            raise SceneGenerationError(
                f"scene generation has no mapped prim for Blender object {obj.name_full!r}"
            )
        material = getattr(obj.ov.physics.collision, "physics_material", None)
        material_path = (
            physics_material_paths.get(_session_uid(material, "MATERIAL"), "")
            if material is not None
            else ""
        )
        records.append(
            physics_topology_conversion.convert_mapped_object_physics(
                obj,
                mapping,
                material_path,
            )
        )
    records.extend(
        simready_physics_conversion.convert_scene_unibodies(
            simready_unibodies,
            mappings,
            occupied,
            physics_material_paths,
        )
    )
    return tuple(sorted(records, key=lambda record: record.usd_prim_path))


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


def _prim_paths(path: Path) -> set[str]:
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError(
            "stock-exported scene generation could not be opened"
        )
    return {str(prim.GetPath()) for prim in stage.Traverse()}


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


def _schema_paths(
    stage: Any,
    object_prim: Any,
    object_type: str,
    data_name: str,
) -> list[str]:
    object_path = object_prim.GetPath()
    matches = []
    type_matches = []
    for prim in stage.Traverse():
        path = prim.GetPath()
        if path != object_path and not path.HasPrefix(object_path):
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
        if str(getattr(obj, "type", "")) == "MESH"
        for material in getattr(getattr(obj, "data", None), "materials", ())
        if material is not None
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
                tuple(tuple(int(value) for value in polygon.vertices) for polygon in data.polygons),
                tuple(
                    _session_uid(material, "MATERIAL") if material is not None else 0
                    for material in data.materials
                ),
                simready_physics_conversion.fingerprint_for_object(obj),
                _session_uid(obj.parent, "OBJECT") if obj.parent is not None else 0,
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


def _is_noop_topology_edit(
    generation: SceneGeneration,
    affected: tuple[BlenderId, ...],
    current: Mapping[BlenderId, str],
) -> bool:
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
        _write_layered_generation(
            stock_root,
            (),
            topology_deltas,
            Path(predecessor.base_usd_path),
        )
        _validate_composed_generation(stock_root, mappings, replaced_paths, deleted_paths)
        timings["composition_validation_ms"] = (time.perf_counter() - phase) * 1000.0

        phase = time.perf_counter()
        records = _compile_add_on_opinions(scene, mappings, stock_root)
        digest = _sparse_generation_digest(
            predecessor.base_digest,
            mappings,
            topology_deltas,
            records,
        )
        destination = work_directory / f"generation-{number:06d}-{digest[:16]}"
        composed_path = candidate / "composed.usda"
        _write_layered_generation(
            composed_path,
            records,
            topology_deltas,
            Path(predecessor.base_usd_path),
        )
        _validate_composed_generation(composed_path, mappings, replaced_paths, deleted_paths)
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
        total_ms - sum(value for key, value in timings.items() if key != "total_ms"),
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
        diagnostics=MappingProxyType(
            {
                "mode": "sparse_reconciliation",
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
        if str(getattr(obj, "type", "")) in {"LIGHT", "MESH"}
    }
    previous_objects = {
        identity: mapping
        for identity, mapping in predecessor.blender_prim_paths.items()
        if identity.kind == "OBJECT" and mapping.blender_id_type in {"LIGHT", "MESH"}
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
                reason = (
                    "unsupported_sparse_object_type"
                    if existing is not None
                    else "unsupported_sparse_object"
                )
                raise SceneGenerationError(
                    f"unsupported affected Blender object session_uid {identity.session_uid}",
                    ({"blender_id_kind": identity.kind, "session_uid": identity.session_uid, "reason": reason},),
                )
        elif identity.kind in {"LIGHT", "MESH"}:
            selected_ids.update(
                object_id
                for object_id, obj in current_objects.items()
                if str(getattr(obj, "type", "")) == identity.kind
                and _session_uid(obj.data, identity.kind) == identity.session_uid
            )
            removed_ids.update(
                object_id
                for object_id, mapping in previous_objects.items()
                if mapping.blender_id_type == identity.kind
                if mapping.data_session_uid == identity.session_uid
                and object_id not in current_objects
            )
        elif identity.kind == "MATERIAL":
            selected_ids.update(
                object_id
                for object_id, obj in current_objects.items()
                if any(
                    material is not None
                    and _session_uid(material, "MATERIAL") == identity.session_uid
                    for material in (
                        obj.data.materials
                        if str(getattr(obj, "type", "")) == "MESH"
                        else ()
                    )
                )
            )
        else:
            raise SceneGenerationError(
                f"unsupported sparse Blender ID kind {identity.kind!r}",
                ({"blender_id_kind": identity.kind, "session_uid": identity.session_uid, "reason": "unsupported_sparse_id_kind"},),
            )

    while True:
        previous_selection = set(selected_ids)
        previous_materials = set(affected_materials)
        affected_materials.update(
            BlenderId("MATERIAL", _session_uid(material, "MATERIAL"))
            for identity in selected_ids
            if str(getattr(current_objects[identity], "type", "")) == "MESH"
            for material in current_objects[identity].data.materials
            if material is not None
        )
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
                    for material in (
                        obj.data.materials
                        if str(getattr(obj, "type", "")) == "MESH"
                        else ()
                    )
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


def _selected_stock_export(scene: Any, objects: tuple[Any, ...], path: Path) -> None:
    module = _blender_module_provider()
    context = getattr(module, "context", None)
    if context is None:
        raise SceneGenerationError("Blender context is unavailable for sparse scene export")
    selected = tuple(getattr(context, "selected_objects", ()))
    view_layer_objects = getattr(getattr(context, "view_layer", None), "objects", None)
    active = getattr(view_layer_objects, "active", None)
    active_mode = str(getattr(active, "mode", "OBJECT")) if active is not None else "OBJECT"
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

    move_materials = bool(affected_material_ids)
    material_root = "/World/_materials"
    material_destination = f"{generation_root}/_materials"
    if move_materials and stage.GetPrimAtPath(material_root):
        _move_prim(stage, material_root, material_destination)

    rewritten: dict[BlenderId, BlenderPrimPath] = {}
    for identity, mapping in selected_mappings.items():
        if identity.kind == "OBJECT":
            rewritten[identity] = _rewritten_mapping(mapping, move_roots)
        elif move_materials and mapping.object_path.startswith(material_root):
            rewritten[identity] = BlenderPrimPath(
                mapping.blender_id_name,
                mapping.blender_id_type,
                material_destination + mapping.object_path[len(material_root) :],
                material_destination + mapping.schema_path[len(material_root) :],
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
) -> None:
    from pxr import Sdf, Usd  # type: ignore

    opinion_paths = []
    for index, record in enumerate(records):
        path = destination.parent / f"opinion-{index:04d}.usda"
        layer = Sdf.Layer.CreateNew(str(path))
        if not layer.ImportFromString(record.layer_text) or not layer.Save():
            raise SceneGenerationError(f"could not persist add-on opinion layer {path}")
        opinion_paths.append(path)
    root = Sdf.Layer.CreateNew(str(destination))
    root.subLayerPaths = [
        _relative_sublayer(destination, path)
        for path in opinion_paths
    ] + [
        _relative_sublayer(destination, Path(delta.usd_path))
        for delta in reversed(deltas)
    ] + [_relative_sublayer(destination, base_path)]
    root.Save()
    if Usd.Stage.Open(str(destination)) is None:
        raise SceneGenerationError(f"layered scene generation could not be opened at {destination}")


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
) -> None:
    from pxr import Usd  # type: ignore

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise SceneGenerationError(f"composed sparse scene generation could not be opened: {path}")
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
