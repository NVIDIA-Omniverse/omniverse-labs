# SPDX-License-Identifier: Apache-2.0
"""External-process-only adapters for the public NVIDIA SDKs.

No module in this directory is imported by Blender. Scene ownership belongs to
OVStage; the worker owns all borrowed SDK resources and publishes owned results.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import time
import uuid
import time


def _retained_overlay_values(values, changed_paths):
    """Keep the same untouched certificates as the owning client.

    Missing compiler proof stays conservative. Paths nominate whole subtrees,
    including parent transforms, material graphs and removed/created owners.
    """
    if changed_paths is None:
        return {}
    if not isinstance(changed_paths, (list, tuple)) or any(
            not isinstance(path, str) or not path.startswith('/') or '//' in path
            or any(part in {'.', '..'} for part in path.split('/'))
            for path in changed_paths):
        raise ValueError('Invalid effective population change paths')
    return {key: value for key, value in values.items()
            if not any(key[0] == path or key[0].startswith(path.rstrip('/') + '/')
                       for path in changed_paths)}


@contextmanager
def _timed(timings, stage):
    """Record one unload stage's seconds; the worker logs them (exit-unload)."""
    started = time.monotonic()
    try:
        yield
    finally:
        timings[stage] = time.monotonic() - started


def _ovrtx_version():
    return importlib.metadata.version("ovrtx")


def derived_data_cache(output_root, cache_root, version=_ovrtx_version):
    """UJITSO's DerivedDataCache (built geometry and textures, 0.3-1.6 GB per scene).

    With the add-on's persistent OVRTX cache root it lives beside that root, one
    folder per OVRTX version, so later workers reuse it. The SDK's local data
    store is multi-process: the first process locks the folder, a concurrent one
    forks into a sub folder that is merged back later, and a disk budget bounds it.
    Without a cache root it stays in this worker's own output.
    """
    if cache_root:
        return (Path(cache_root).resolve().parent / ("ujitso-" + version())).resolve()
    return (Path(output_root) / "cache" / "DerivedDataCache").resolve()


def _versions():
    return {name: importlib.metadata.version(name) for name in
            ("ovrtx", "ovphysx", "ovstage", "ovstream", "warp-lang", "numpy", "packaging",
             "physx-usd-schemas")}


def cuda_device_identity(active_cuda_gpus):
    """The CUDA device OVRTX renders on and the OVStream publisher maps from.

    That is the first index of ``active_cuda_gpus``, read from Warp as the
    publisher reads it (``ovstream_frames``). The worker runs with
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` unless the user chose an order.
    """
    try:
        index = int(str(active_cuda_gpus).split(",")[0].strip())
        import warp as wp
        device = wp.get_device(f"cuda:{index}")
        if not device.is_cuda:
            raise ValueError(f"{device} is not a CUDA device")
        return {"status": "known", "index": device.ordinal, "name": device.name,
                "uuid": getattr(device, "uuid", None), "pci_bus_id": getattr(device, "pci_bus_id", None),
                "device_order": os.environ.get("CUDA_DEVICE_ORDER") or "FASTEST_FIRST"}
    except Exception as exc:
        return {"status": "unavailable", "reason": f"Warp could not read CUDA device {active_cuda_gpus!r}: "
                                                   f"{type(exc).__name__}: {exc}"}


def vulkan_device_identity(renderer):
    """The Vulkan physical device OVRTX selected, when OVRTX exposes it.

    OVRTX 0.6 does not: its public API has no device query, and GPU Foundation
    maps the ``active_cuda_gpus`` CUDA indices to Vulkan devices internally.
    """
    return {"status": "unavailable",
            "reason": f"OVRTX {_ovrtx_version()} exposes no selected Vulkan physical device; "
                      "it maps active_cuda_gpus to its Vulkan device internally"}


def _uuid_key(value):
    text = str(value or "").strip().lower()
    text = text[4:] if text.startswith("gpu-") else text
    return text.replace("-", "") or None


def check_gpu_identity(cuda, vulkan):
    """Return the identity record; raise when both devices are known and differ.

    Interop between a CUDA context on one GPU and Vulkan work on another fails
    later ("invalid CUDA interop stream"), so name both GPUs at session start.
    Devices are compared by UUID (Vulkan deviceUUID equals the CUDA UUID on
    NVIDIA), or by LUID when both report one.
    """
    record = {"cuda": cuda, "vulkan": vulkan}
    if cuda.get("status") != "known" or vulkan.get("status") != "known":
        missing = [(name, side) for name, side in (("CUDA", cuda), ("Vulkan", vulkan)) if side.get("status") != "known"]
        return {**record, "check": "unavailable",
                "reason": "; ".join(f"{name}: {side.get('reason', 'unknown')}" for name, side in missing)}
    pairs = [(_uuid_key(cuda.get("uuid")), _uuid_key(vulkan.get("uuid"))),
             (cuda.get("luid"), vulkan.get("luid"))]
    compared = [(left, right) for left, right in pairs if left and right]
    if not compared:
        return {**record, "check": "unavailable", "reason": "CUDA and Vulkan devices share no UUID or LUID"}
    if any(left != right for left, right in compared):
        raise RuntimeError(
            f"GPU mismatch: CUDA device {cuda.get('index')} is {cuda.get('name')!r} ({cuda.get('uuid')}) "
            f"but OVRTX selected Vulkan device {vulkan.get('name')!r} "
            f"({vulkan.get('uuid') or vulkan.get('luid')}). CUDA/Vulkan interop needs one GPU; "
            "choose it with OVRTX_ACTIVE_CUDA_GPUS (CUDA index in PCI bus order, as nvidia-smi lists it)")
    return {**record, "check": "match"}


def _validate_output_selection(sensors, params):
    """Keep unsupported MVP outputs from becoming a partial successful frame."""
    if len(sensors) != 1:
        raise ValueError("Windows MVP requires exactly one RenderProduct; multiple products are future work")
    if params.get("render_var", "LdrColor") not in ("LdrColor", "HdrColor"):
        raise ValueError("Windows MVP supports only LdrColor or HdrColor; additional AOVs are future work")
    if params.get("semantic_outputs", False) is not False:
        raise ValueError("Semantic/additional AOV output is future work for the Windows MVP")
    for key in ("render_vars", "additional_aovs", "aovs"):
        if params.get(key):
            raise ValueError(f"{key} selection is unsupported; additional AOVs are future work")


class RendererBackend:
    def __init__(self, output_root):
        from runtime_identity import verified_identity
        verified_identity()
        import numpy as np
        import ovrtx
        import ovstage
        self.np, self.rtx, self.ovstage = np, ovrtx, ovstage
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.renderer = self.stage = self.publisher = None
        self.ordinal = 0
        self.sensors = []
        self.epoch = uuid.uuid4().hex
        self.sample_count = 0
        self.pending = {}
        self.population_owner = None
        self.pose_population = None
        self.native_pose_fanout = None
        self.native_stage_id = None
        self.scene_config = None
        self.scene_failure = None
        self.desired_values = {}
        self.simulation_time_ns = 0
        self.active_cuda_gpus = os.environ.get("OVRTX_ACTIVE_CUDA_GPUS", os.environ.get("OV_BLENDER_PUBLIC_SDK_GPU", "0"))
        self.gpu_identity = None

    def dispatch(self, op, params):
        if getattr(self, 'scene_failure', None) and op not in {'load', 'unload', 'health'}:
            raise RuntimeError('Scene restoration was not confirmed; reload required: ' + self.scene_failure)
        if getattr(self, 'experimental_structure_uncertain', False) and op not in {'load', 'unload', 'health', 'structural_inspect'}:
            raise RuntimeError('Experimental structural mutation failed; reload the scene before further updates or rendering')
        if op == "health":
            return {"loaded": self.stage is not None, "pid": os.getpid(), "versions": _versions(), "epoch": self.epoch, "native_stage_id": self.native_stage_id,
                    "scene_failure": getattr(self, 'scene_failure', None),
                    "gpu_identity": getattr(self, 'gpu_identity', None),
                    "frame_resources": self.publisher.diagnostics() if self.publisher else None}
        if op == "load":
            _validate_output_selection(list(params.get("sensor_paths") or ()), params)
            self.unload()
            self.scene_failure = None
            self.experimental_structure_uncertain = False
            # OVRTX 0.6 defaults its writable cache root to the user profile.
            # The add-on passes a persistent per-version directory beside the
            # sidecars (a cold shader cache costs minutes per start); fall back
            # to this worker's output. The worker owns crash handling, so the
            # SDK crash reporter stays off.
            configured = os.environ.get('OV_BLENDER_OVRTX_CACHE_ROOT', '').strip()
            # The typed public cache option keeps mutable UJITSO output outside
            # the locked sidecar. Its explicit environment override takes higher
            # precedence, so bind both to the same directory.
            cache = derived_data_cache(self.output_root, configured, _ovrtx_version)
            cache.mkdir(parents=True, exist_ok=True)
            cache_uri = 'local://' + cache.as_posix()
            os.environ['UJITSO_DATASTORE_CONFIG'] = cache_uri
            cache_root = Path(configured) if configured else self.output_root / 'cache' / 'ovrtx'
            cache_root = cache_root.resolve()
            cache_root.mkdir(parents=True, exist_ok=True)
            self.renderer = self.rtx.Renderer(self.rtx.RendererConfig(
                active_cuda_gpus=self.active_cuda_gpus, log_level="warn", log_file_path=str(self.output_root / "ovrtx.log"), keep_system_alive=True,
                datastore_cache=cache_uri, cache_root_path=str(cache_root), enable_crash_reporter=False))
            # The renderer has selected its devices; fail here, naming both
            # GPUs, rather than at the first CUDA/Vulkan interop.
            try:
                self.gpu_identity = check_gpu_identity(cuda_device_identity(self.active_cuda_gpus),
                                                       vulkan_device_identity(self.renderer))
            except RuntimeError:
                self.gpu_identity = None
                self.unload()
                raise
            self.stage = self.ovstage.Stage("blender-public-renderer")
            self.renderer.attach_ovstage(self.stage)
            try:
                from .ovstage_scene_updates import SceneGenerationPopulation
            except ImportError:
                from ovstage_scene_updates import SceneGenerationPopulation
            self.native_stage_id = uuid.uuid4().hex
            self.population_owner = SceneGenerationPopulation(self.stage)
            self.population_owner.initialize(params["scene_path"], stage_metadata=params.get("stage_metadata"))
            try:
                from .ovstage_pose_layers import PoseLayerPopulation
            except ImportError:
                from ovstage_pose_layers import PoseLayerPopulation
            self.pose_population = PoseLayerPopulation(self.population_owner, self.output_root)
            self.ordinal = self.population_owner.ordinal
            self.stage.advance_write_floor(self.ordinal, self.ovstage.Scope.ALL).wait()
            self.renderer.update_from_stage(self.ordinal)
            self.desired_values = {}
            self._apply_scene_config(params)
            self.scene_config = dict(params)
            return self.dispatch("health", {})
        if op == "update_scene":
            return self.update_scene(params)
        if op == "material_graph_update":
            try:
                from . import inline_material_population
            except ImportError:
                import inline_material_population
            return inline_material_population.apply(self, params)
        if op == "instance_graph_update":
            try:
                from . import instance_graph_updates
            except ImportError:
                import instance_graph_updates
            return instance_graph_updates.apply(self, params)
        if op in {'structural_update', 'structural_inspect'}:
            if self.stage is None:
                raise RuntimeError('No scene loaded')
            try:
                from . import direct_native_structure
            except ImportError:
                import direct_native_structure
            return (direct_native_structure.apply(self, params) if op == 'structural_update'
                    else direct_native_structure.inspect(self, params))
        if op == "retarget_camera":
            return self.retarget_camera(params)
        if op == "next_raw_chunk":
            return self.publisher.next_raw_chunk(params["copy_token"])
        if op == "cancel_raw_transfer":
            self.publisher.cancel_raw_transfer(params["copy_token"])
            return {"cancelled": True}
        if op in ("update", "update_attributes", "update_transforms"):
            return self.update(params.get("values", []))
        if op == "resize":
            self.width, self.height = int(params["width"]), int(params["height"])
            if not (0 < self.width <= 16384 and 0 < self.height <= 16384):
                raise ValueError("Invalid render dimensions")
            return self.update([{"prim_path": path, "attribute_name": "resolution", "value_type": "Int2", "value": [self.width, self.height]} for path in self.sensors])
        if op == "render":
            return self.render(params)
        if op == "begin_render":
            return self.begin_render(params)
        if op in ("poll_render", "wait_render"):
            return self.poll_render(params, publish=op == "poll_render")
        if op == "cancel_render":
            entry = self.pending.get(params["ticket"])
            if entry is not None:
                entry["cancelled"] = True
            self._reap_cancelled(wait_token=params["ticket"],
                                 timeout_ns=min(8_000_000, max(0, int(params.get("timeout_ns", 0)))))
            return {"cancelled": True, "native_complete": params["ticket"] not in self.pending}
        if op in ("release", "release_frame"):
            if self.publisher:
                self.publisher.release(params["copy_token"])
            return {"released": True}
        if op == "unload":
            self.unload()
            return {"unloaded": True}
        raise ValueError(f"Unsupported renderer operation: {op}")

    def _apply_scene_config(self, params):
        self.sensors = list(params["sensor_paths"])
        self.dispatch("resize", params)
        settings = []
        for key, value in params.get("settings", {}).items():
            if not key.startswith("/rtx/"):
                raise ValueError(f"Unsupported public renderer setting: {key}")
            for path in self.sensors:
                settings.append({"prim_path": path, "attribute_name": "omni:" + key.strip("/").replace("/", ":"), "value": value})
        if settings:
            self.update(settings)

    def update_scene(self, params):
        _validate_output_selection(list(params.get("sensor_paths") or ()), params)
        started = time.perf_counter()
        timings = {}
        if getattr(self, 'scene_failure', None):
            raise RuntimeError('Scene restoration was not confirmed; reload required')
        if self.population_owner is None or self.scene_config is None:
            raise RuntimeError("No accepted scene generation")
        old_config, old_values = self.scene_config, dict(self.desired_values)
        old_sources = self.population_owner.snapshot_inline_sources()
        changed_paths = params.get('population_changed_prim_paths')
        retained_values = _retained_overlay_values(old_values, changed_paths)
        # Visibility is part of the newly accepted Blender projection. Its
        # current inline source is removed by generation replacement below.
        retained_values = {key: row for key, row in retained_values.items() if key[1] != 'visibility'}
        if params.get("stage_metadata", {}) != old_config.get("stage_metadata", {}):
            raise ValueError("Stage metadata changed; an explicit stage replacement is required")
        for entry in self.pending.values():
            entry["cancelled"] = True
        self._reap_cancelled()
        self._require_idle_native_render()
        try:
            phase_started = time.perf_counter()
            delta = params.get("population_delta")
            if delta is None or len(self.population_owner.overlays) >= 32:
                population_result = self.population_owner.replace(params["scene_path"], ordinal=self.ordinal + 1)
            else:
                if not isinstance(delta, dict) or set(delta) != {"path", "predecessor_path"}:
                    raise ValueError("Invalid typed population delta")
                population_result = self.population_owner.apply_delta(delta["path"],
                    predecessor_path=delta["predecessor_path"],desired_path=params["scene_path"],
                    ordinal=self.ordinal+1,expected_predecessor_path=old_config["scene_path"])
            timings['population_apply_ms'] = (time.perf_counter()-phase_started)*1000
            phase_started = time.perf_counter()
            self.ordinal = self.population_owner.ordinal
            self.stage.advance_write_floor(self.ordinal).wait()
            timings['write_floor_wait_ms'] = (time.perf_counter()-phase_started)*1000
            phase_started = time.perf_counter()
            self.renderer.update_from_stage(self.ordinal)
            timings['renderer_update_from_stage_ms'] = (time.perf_counter()-phase_started)*1000
            phase_started = time.perf_counter()
            self.renderer.reset(time=self.simulation_time_ns / 1e9)
            # Hierarchy and instance identities belong to a population
            # generation, not merely a Stage ID or source filename. A reload
            # may replace subtrees at the same immutable owner path.
            timings['renderer_reset_ms'] = (time.perf_counter()-phase_started)*1000
            phase_started = time.perf_counter()
            scoped = population_result.get('population_update') == 'scoped_overlay'
            # A scoped overlay keeps the native stage and our synthetic world
            # inputs outside the changed subtrees; a replacement drops both.
            try:
                from .native_pose_fanout import successor_fanout
            except ImportError:
                from native_pose_fanout import successor_fanout
            self.native_pose_fanout = successor_fanout(self.native_pose_fanout,
                                                       changed_paths if scoped else None)
            self.desired_values = retained_values if scoped else {}
            self._apply_scene_config(params)
            timings['scene_configuration_ms'] = (time.perf_counter()-phase_started)*1000
            phase_started = time.perf_counter()
            self.pose_population.accepted_generation_replaced(
                retain_sources=population_result.get("population_update") == "scoped_overlay")
            timings['pose_population_accept_ms'] = (time.perf_counter()-phase_started)*1000
        except Exception as error:
            try:
                if getattr(self.population_owner, 'failed', False):
                    raise RuntimeError('Population owner could not restore its predecessor')
                # A matching filename does not prove unchanged contents or
                # native source ownership. Rebuild before replaying values.
                self.population_owner.restore(old_config["scene_path"], ordinal=max(self.ordinal, self.population_owner.ordinal) + 1)
                self.population_owner.restore_inline_sources(old_sources)
                self.ordinal = max(self.ordinal, self.population_owner.ordinal)
                self.stage.advance_write_floor(self.ordinal).wait()
                self.renderer.update_from_stage(self.ordinal)
                self.renderer.reset(time=self.simulation_time_ns / 1e9)
                self.native_pose_fanout = None
                self.pose_population.accepted_generation_replaced()
                self.desired_values = {}
                self._apply_scene_config(old_config)
                if old_values:
                    self.update(list(old_values.values()))
            except Exception as rollback_error:
                self.scene_failure = f'{error}; predecessor restoration failed: {rollback_error}'
                return {"scene_updated": False, "rollback_succeeded": False,
                        "error": f"Scene update failed ({error}); predecessor restoration failed ({rollback_error})"}
            return {"scene_updated": False, "rollback_succeeded": True,
                    "error": f"Scene update rejected; predecessor restored: {error}"}
        self.scene_config = dict(params)
        self.sample_count = 0
        timings['total_ms'] = (time.perf_counter()-started)*1000
        return {"native_update_timings": timings, "scene_updated": True, "native_stage_id": self.native_stage_id, "ordinal": self.ordinal,
                "population_update":population_result.get("population_update","replacement"),
                "overlay_count":population_result.get("overlay_count",0)}

    def retarget_camera(self, params):
        try:
            from .renderer_values import prepare_value
        except ImportError:
            from renderer_values import prepare_value
        values = params.get('values')
        if not isinstance(values, list) or not 1 <= len(values) <= 128:
            raise ValueError('Camera retarget requires a bounded value batch')
        self._reap_cancelled()
        if self.pending:
            raise ValueError('Camera retarget requires completed native samples')
        kinds = {}
        relationships = {}
        camera_attributes = {'omni:xform', 'omni:resetXformStack', 'projection', 'focalLength',
            'horizontalAperture', 'verticalAperture', 'horizontalApertureOffset',
            'verticalApertureOffset', 'clippingRange', 'fStop', 'focusDistance',
            'omni:rtx:autoExposure:enabled', 'omni:rtx:autoExposure:whitePointScale'}
        for row in values:
            prepare_value(row)
            path, name = row['prim_path'], row['attribute_name']
            if path in self.sensors:
                if name == 'camera' and row['value_type'].lower() == 'relationship':
                    if len(row['value']) != 1:
                        raise ValueError('Camera retarget requires exactly one camera target per product')
                    relationships[path] = row['value'][0]
                elif name not in {'omni:rtx:autoExposure:enabled', 'omni:rtx:autoExposure:whitePointScale'}:
                    raise ValueError('Camera retarget cannot change other product state')
                kinds[path] = 'RenderProduct'
            else:
                if name not in camera_attributes:
                    raise ValueError('Camera retarget cannot change non-camera values')
                kinds[path] = 'Camera'
        if set(relationships) != set(self.sensors) or any(kinds.get(p) != 'Camera' for p in relationships.values()):
            raise ValueError('Camera retarget requires one existing camera for every product')
        desired_source = params.get('source_scene_path')
        if desired_source is not None:
            desired_source = str(self.population_owner._source(desired_source))
        result = self.update(values, required_types=kinds)
        if desired_source is not None:
            self.scene_config = {**self.scene_config,'scene_path':desired_source}
        return {**result, 'camera_retargeted': True, 'native_stage_id': self.native_stage_id}

    def _write_phase(self, paths, rows):
        """Keep queries/payloads alive and drain every submitted operation.

        Duplicate columns retain sequential semantics. SDKs without the batch
        entry point use the existing synchronous compatibility path.
        """
        if not rows:
            return
        grouped = {}
        for path, name, data, is_array, semantic in rows:
            grouped.setdefault(path, []).append((name, data, is_array, semantic))
        batch = getattr(self.stage, "write_attributes", None)
        desc = getattr(self.ovstage, "WriteDesc", None)
        with ExitStack() as owners:
            pending = []
            failure = None
            try:
                for path, items in grouped.items():
                    plist = owners.enter_context(paths.create_path_list_from_strings([path]))
                    query = owners.enter_context(self.stage.query_from_path_list(plist))
                    if batch is None or desc is None or len({item[0] for item in items}) != len(items):
                        for name, data, is_array, semantic in items:
                            self.stage.write_attribute(query, name, self.ordinal, data,
                                is_array=is_array, semantic=semantic).wait()
                    else:
                        writes = [desc(attribute=name, tensors=data, is_array=is_array, semantic=semantic)
                                  for name, data, is_array, semantic in items]
                        pending.append(batch(query, writes, self.ordinal))
            except Exception as error:
                failure = error
            # A failed enqueue/wait does not make earlier operations disappear.
            # Drain before releasing any query or returning an uncertain edit.
            for operation in pending:
                try:
                    operation.wait()
                except Exception as error:
                    if failure is None:
                        failure = error
            if failure is not None:
                raise failure

    def update(self, values, *, required_types=None):
        """Bounded mutation counters; call times include any nested replay."""
        started = time.perf_counter()
        result = None
        try:
            result = self._update_values(values, required_types=required_types)
            return result
        finally:
            stats = getattr(self, 'update_diagnostics', None)
            if stats is None:
                stats = self.update_diagnostics = dict(calls=0, input_rows=0,
                    unchanged_poses=0, usd_reprojections=0, failures=0,
                    renderer_resets=0, native_pose_updates=0,
                    total_call_ms=0.0, max_call_ms=0.0)
            elapsed = (time.perf_counter() - started) * 1000
            stats['calls'] += 1
            stats['input_rows'] += len(values)
            stats['total_call_ms'] += elapsed
            stats['max_call_ms'] = max(stats['max_call_ms'], elapsed)
            stats['last_call_ms'] = elapsed
            stats['failures'] += result is None
            if result is not None:
                stats['unchanged_poses'] += result.get('native_unchanged_poses', 0)
                stats['usd_reprojections'] += result.get('native_pose_route') == 'usd_fallback'
                stats['renderer_resets'] += result.get('native_renderer_reset', False)
                stats['native_pose_updates'] += bool(result.get('native_transform_auto_invalidation'))
                stats['last_route'] = result.get('native_pose_route', 'empty')

    def _update_values(self, values, *, required_types=None):
        started = time.perf_counter()
        timings = {}
        if self.stage is None:
            raise RuntimeError("No scene loaded")
        self._require_idle_native_render()
        try:
            from .renderer_values import prepare_value, tensor_value, tensor_width
        except ImportError:
            from renderer_values import prepare_value, tensor_value, tensor_width
        # Validate the entire batch first; an invalid suffix cannot commit a
        # valid prefix. Native failures remain explicitly uncertain upstream.
        prepared = [prepare_value(value) for value in values]
        if not prepared:
            return {"updated_count": 0, "ordinal": self.ordinal,
                    "native_renderer_reset": False, "native_population_replacements": 0}
        visibility_owner = None
        visibility_opinion = None
        if any(item[1] == 'visibility' for item in prepared):
            try:
                from .inline_visibility_population import InlineVisibilityPopulation
            except ImportError:
                from inline_visibility_population import InlineVisibilityPopulation
            population_owner = getattr(self, 'population_owner', None)
            if population_owner is None:
                raise RuntimeError('Visibility updates require populated scene ownership')
            visibility_owner = getattr(population_owner, 'visibility', None)
            if visibility_owner is None:
                visibility_owner = InlineVisibilityPopulation(population_owner)
                population_owner.visibility = visibility_owner
            visibility_opinion = visibility_owner.prepare(prepared)
            # Replacing the current opinion can re-sample properties on its
            # named prims. Retain only accepted values on that bounded set;
            # incoming values always win and no unrelated scene scan occurs.
            incoming = {(item[0], item[1]) for item in prepared}
            prepared.extend(prepare_value(row) for target, row in self.desired_values.items()
                if target[0] in visibility_opinion.values and target[1] != 'visibility'
                and target not in incoming)
        timings['logical_validation_ms'] = (time.perf_counter() - started) * 1000
        phase_started = time.perf_counter()
        np, ovs = self.np, self.ovstage
        pose_inputs = {item[0]: np.asarray(item[5]).reshape(4, 4)
                       for item in prepared if item[1] == "omni:xform"}
        try:
            from .native_pose_fanout import FanoutPlan, NativePoseFanout
        except ImportError:
            from native_pose_fanout import FanoutPlan, NativePoseFanout
        pose_plan = FanoutPlan()
        fanout = getattr(self, "native_pose_fanout", None)
        # Value/mesh/binding writes do not need hierarchy discovery. A reset-only
        # edit must still update a previously accepted logical reset boundary.
        reset_only = any(item[1] == "omni:resetXformStack" for item in prepared)
        if pose_inputs or (reset_only and fanout is not None):
            if fanout is None:
                self.native_pose_fanout = fanout = NativePoseFanout(self.stage, ovs, render_only=True)
            pose_owner = getattr(self, "pose_population", None)
            population_path = getattr(getattr(self, "population_owner", None), "path", None)
            base_path = (getattr(pose_owner, "base", None)
                         if population_path == getattr(pose_owner, "wrapper", None) else population_path)
            pose_plan = fanout.prepare(pose_inputs, values, ordinal=self.ordinal,
                context=(getattr(self, "native_stage_id", id(self.stage)), str(base_path)))
        # Expansion is read-only. Every original and generated row passes the
        # same schema validation before the first native write is enqueued.
        original_paths = {item[0] for item in prepared}
        prepared.extend(prepare_value(row) for row in pose_plan.extra_rows)
        timings['pose_expansion_ms'] = (time.perf_counter() - phase_started) * 1000
        encoded = []
        prim_types = {}
        phase_started = time.perf_counter()
        schema_read_count = 0
        schema_attribute_count = 0
        with ovs.PathDictionary(self.stage) as paths:
            # Read and validate every target schema before the first write.
            # No borrowed group, NumPy view, or query survives this phase.
            by_path = {}
            for item in prepared:
                by_path.setdefault(item[0], []).append(item)
            for path, items in by_path.items():
                type_token = paths.intern_token("usd-prim-type")
                attributes = {paths.intern_token(item[1]): item[1] for item in items}
                schemas = {}
                found_prim = False
                with paths.create_path_list_from_strings([path]) as plist:
                    with self.stage.query_from_path_list(plist) as query:
                        schema_read_count += 1
                        schema_attribute_count += len(set([type_token, *attributes]))
                        with self.stage.read_attributes(query, list(dict.fromkeys([type_token, *attributes])), ovs.OrdinalRange.latest(self.ordinal)) as read:
                            read.wait()
                            for group in read.groups():
                                try:
                                    if group.is_delete:
                                        continue
                                    if group.attribute == type_token:
                                        found_prim = group.prim_count == 1
                                        if found_prim and hasattr(paths, "token_to_string"):
                                            prim_types[path] = paths.token_to_string(int(np.asarray(group.array(0)).reshape(-1)[0]))
                                    else:
                                        name = attributes.get(group.attribute)
                                        if name is None or group.prim_count != 1:
                                            raise ValueError(f"Ambiguous native schema for {path}")
                                        current = (np.asarray(group.array(0)).dtype,
                                                   int(group.raw.semantic), bool(group.is_array),
                                                   tensor_width(group.tensor(0)))
                                        if name in schemas and schemas[name] != current:
                                            raise ValueError(f"Inconsistent native schema for {path}.{name}")
                                        schemas[name] = current
                                finally:
                                    self.stage.release_group(group)
                structural_fanout = (path in pose_plan.structural_paths
                    and path not in original_paths
                    and {item[1] for item in items} == {"omni:xform", "omni:resetXformStack"})
                if not found_prim and not structural_fanout:
                    raise ValueError(f"Update target is not an existing typed USD prim: {path}")
                if required_types and prim_types.get(path) != required_types[path]:
                    raise ValueError(f'Camera retarget target has the wrong native type: {path}')
                for item in items:
                    name = item[1]
                    schema = schemas.get(name)
                    data, is_array, semantic = tensor_value(item, ovs, paths,
                        None if schema is None else schema[0],
                        existing_semantic=None if schema is None else schema[1],
                        existing_is_array=None if schema is None else schema[2],
                        existing_width=None if schema is None else schema[3])
                    encoded.append((path, name, data, is_array, semantic))
            timings['schema_read_encode_ms'] = (time.perf_counter() - phase_started) * 1000
            phase_started = time.perf_counter()
            try:
                from .subset_dependency_updates import subset_dependency_rows
            except ImportError:
                from subset_dependency_updates import subset_dependency_rows
            subset_rows = subset_dependency_rows(self.stage, ovs, paths, prepared,
                                                  prim_types, ordinal=self.ordinal)
            encoded.extend(subset_rows)
            timings['subset_dependency_ms'] = (time.perf_counter() - phase_started) * 1000
            timings['subset_dependency_count'] = len(subset_rows)
            phase_started = time.perf_counter()
            encoded = [item for item in encoded if not (
                item[0] in pose_plan.unchanged_poses
                and item[1] in {"omni:xform", "omni:resetXformStack"})]
            if not encoded:
                self.desired_values.update({(value["prim_path"], value.get("attribute_name", value.get("attribute"))): dict(value) for value in values})
                self.native_pose_fanout.accepted(pose_plan)
                return {"updated_count": len(values), "ordinal": self.ordinal,
                        "native_pose_route": "unchanged", "native_pose_fanout_rows": 0,
                        "native_unchanged_poses": len(pose_plan.unchanged_poses),
                        "native_transform_auto_invalidation": True,
                        "native_renderer_reset": False, "native_population_replacements": 0,
                        "native_camera_history_preserved": pose_plan.camera_only,
                        "native_pose_fallback_reason": ""}
            self.ordinal += 1
            if visibility_opinion is not None:
                visibility_started = time.perf_counter()
                try:
                    visibility_owner.apply(visibility_opinion, ordinal=self.ordinal)
                except Exception:
                    self.experimental_structure_uncertain = True
                    raise
                timings['visibility_population_ms'] = (time.perf_counter() - visibility_started) * 1000
                phase_started = time.perf_counter()
            # Native hierarchy recomputes on matrix writes, whereas changing
            # resetStack alone need not dirty the derived world matrix. Set
            # every boundary before any world input to prevent the first
            # fanout from inheriting its parent transform a second time.
            writable = [item for item in encoded if item[1] != 'visibility' and not (
                pose_plan.fallback_reason and prim_types.get(item[0]) == "Xform"
                and item[1] in {"omni:xform", "omni:resetXformStack"})]
            # Descriptors do not establish execution order. Complete all reset
            # boundaries before enqueuing any matrix or other attribute write.
            for boundary in (True, False):
                phase = [item for item in writable
                         if (item[1] == "omni:resetXformStack") == boundary]
                self._write_phase(paths, phase)
        timings['native_write_ms'] = (time.perf_counter() - phase_started) * 1000
        phase_started = time.perf_counter()
        self.stage.advance_write_floor(self.ordinal, ovs.Scope.ALL).wait()
        timings['write_floor_ms'] = (time.perf_counter() - phase_started) * 1000
        phase_started = time.perf_counter()
        parent_poses = {path: np.asarray(data).reshape(4, 4).tolist()
                        for path, name, data, is_array, semantic in encoded
                        if name == "omni:xform" and prim_types.get(path) == "Xform"
                        and pose_plan.fallback_reason}
        if parent_poses:
            # Attached RTX retains descendant world values on generic parent
            # input writes. Native USD population projects the authored hierarchy
            # correctly while reusing the same stage and accepted Blender export.
            self.ordinal += 1
            try:
                # Retain already accepted fast parent opinions if an unsafe
                # hierarchy requires conservative USD projection later.
                self.pose_population.apply({**self.native_pose_fanout.accepted_parents,
                                             **parent_poses}, ordinal=self.ordinal)
            except Exception:
                self.ordinal = max(self.ordinal, self.population_owner.ordinal)
                self.stage.advance_write_floor(self.ordinal, ovs.Scope.ALL).wait()
                self.renderer.update_from_stage(self.ordinal)
                self.renderer.reset(time=self.simulation_time_ns / 1e9)
                self.sample_count = 0
                raise
            self.stage.advance_write_floor(self.ordinal, ovs.Scope.ALL).wait()
            retained = dict(self.desired_values)
            retained.update({(value["prim_path"], value.get("attribute_name", value.get("attribute"))): dict(value) for value in values})
            owned_parents = self.pose_population.poses
            replay = [row for (path, name), row in retained.items()
                      if not (path in owned_parents and name in {"omni:xform", "omni:resetXformStack"})]
            if replay:
                # Population may restore authored base values. Revalidate and
                # restore accepted native camera/material/light/settings rows;
                # incoming values take precedence over their cached predecessor.
                self.update(replay)
        timings['pose_population_replay_ms'] = (time.perf_counter() - phase_started) * 1000
        phase_started = time.perf_counter()
        self.renderer.update_from_stage(self.ordinal)
        timings['renderer_pull_ms'] = (time.perf_counter() - phase_started) * 1000
        native_pose_only = (pose_plan.auto_invalidation_safe and not pose_plan.fallback_reason
                           and all(item[1] in {"omni:xform", "omni:resetXformStack"} for item in prepared))
        camera_only = pose_plan.camera_only and native_pose_only
        # Native camera history and accumulation invalidation handle validated
        # pose deltas. A global reset adds expensive next-frame initialization.
        # Other scene writes can invalidate radiance too: mesh arrays, material
        # bindings, graph resources, product outputs and new supported columns.
        # Only the proven transform-only route relies on native invalidation.
        phase_started = time.perf_counter()
        if not native_pose_only:
            self.renderer.reset(time=self.simulation_time_ns / 1e9)
        timings['renderer_reset_ms'] = (time.perf_counter() - phase_started) * 1000
        self.desired_values.update({(value["prim_path"], value.get("attribute_name", value.get("attribute"))): dict(value) for value in values})
        # If a previous explicit descendant pose moved through inheritance,
        # recovery must replay its new accepted world value, not its old value.
        accepted_derived = [dict(row) for row in pose_plan.extra_rows
                            if (row["prim_path"], row["attribute_name"]) in self.desired_values]
        self.desired_values.update({(row["prim_path"], row["attribute_name"]): row
                                   for row in accepted_derived})
        if fanout is not None and (pose_inputs or reset_only):
            fanout.accepted(pose_plan)
        self.sample_count = 0
        if os.environ.get("OV_BLENDER_PUBLIC_SDK_TRACE_UPDATES") == "1":
            receipts = []
            with ovs.PathDictionary(self.stage) as paths:
                for item in prepared[:64]:
                    path, name = item[:2]
                    columns = []
                    with paths.create_path_list_from_strings([path]) as plist:
                        with self.stage.query_from_path_list(plist) as query:
                            with self.stage.read_attributes(query, [paths.intern_token(name)], ovs.OrdinalRange.latest(self.ordinal)) as read:
                                read.wait()
                                for group in read.groups():
                                    try:
                                        if not group.is_delete:
                                            array = np.asarray(group.array(0))
                                            columns.append({"shape": list(array.shape), "dtype": str(array.dtype),
                                                            "semantic": int(group.raw.semantic),
                                                            "values": array.reshape(-1)[:32].tolist()})
                                    finally:
                                        self.stage.release_group(group)
                    receipts.append({"prim_path": path, "attribute_name": name,
                                     "semantic": item[4], "readback": columns})
            print("public_sdk update_receipt " + json.dumps({"ordinal": self.ordinal,
                "native_stage_id": self.native_stage_id, "rows": receipts}), flush=True)
        timings['total_ms'] = (time.perf_counter() - started) * 1000
        return {"updated_count": len(values), "ordinal": self.ordinal,
                "native_unchanged_poses": len(pose_plan.unchanged_poses),
                "native_update_timings": timings,
                "native_schema_read_count": schema_read_count,
                "native_schema_attribute_count": schema_attribute_count,
                "accepted_derived_values": accepted_derived,
                "native_pose_route": "usd_fallback" if parent_poses else "native_inputs",
                "native_pose_fanout_rows": len(pose_plan.extra_rows),
                "native_renderer_reset": not native_pose_only,
                "native_population_replacements": int(bool(parent_poses)),
                "native_camera_history_preserved": camera_only,
                "native_transform_auto_invalidation": native_pose_only,
                "native_pose_fallback_reason": pose_plan.fallback_reason}

    def _reap_cancelled(self, *, wait_token=None, timeout_ns=0):
        for token, entry in tuple(self.pending.items()):
            if not entry["cancelled"]:
                continue
            if entry.get("products") is not None:
                del self.pending[token]
                continue
            fetch = entry["operation"].wait(timeout_ns=timeout_ns if token == wait_token else 0)
            if fetch is not None and fetch.fetch(timeout_ns=0) is not None:
                del self.pending[token]

    def _require_idle_native_render(self):
        # Cancellation marks a result obsolete; it does not fence the native
        # asynchronous render. Never publish mutations while step_async still
        # reads this stage. Admission can be retried after normal completion.
        if getattr(self, "pending", None):
            self._reap_cancelled()
            if self.pending:
                raise ValueError("Stage mutation requires completed native samples")

    def begin_render(self, params):
        _validate_output_selection(list(params.get("sensor_paths") or self.sensors), params)
        if self.renderer is None:
            raise RuntimeError("No renderer loaded")
        self._reap_cancelled()
        if len(self.pending) >= 16:
            raise RuntimeError("Too many pending renderer samples")
        sensors = list(params.get("sensor_paths") or self.sensors)
        if not sensors or any(path not in self.sensors for path in sensors):
            raise ValueError("Render products must be declared in this scene")
        token = uuid.uuid4().hex
        operation = self.renderer.step_async(set(sensors), delta_time=1.0/60.0, ordinal=self.ordinal)
        self.sample_count += 1
        self.pending[token] = {"operation": operation, "params": dict(params),
                               "sensors": sensors, "samples": self.sample_count, "cancelled": False,
                               "frame_metadata": {"width": self.width, "height": self.height,
                                   "ordinal": self.ordinal, "native_stage_id": self.native_stage_id}}
        return {"ticket": token, "completed_samples": self.sample_count,
                "simulation_time_ns": int(params.get("simulation_time_ns", 0))}

    def poll_render(self, params, *, publish=True):
        token = params["ticket"]
        entry = self.pending.get(token)
        if entry is None or entry["cancelled"]:
            raise ValueError("Renderer ticket is cancelled or belongs to another session")
        timeout_ns = 0 if publish else min(50_000_000, max(0, int(params.get("timeout_ns", 0))))
        products = entry.get("products")
        if products is None:
            fetch = entry["operation"].wait(timeout_ns=timeout_ns)
            if fetch is None:
                return {"ready": False}
            products = fetch.fetch(timeout_ns=0)
            if products is None:
                return {"ready": False}
            entry["products"] = products
        if not publish:
            return {"ready": True}
        try:
            return {"ready": True, **self._publish_products(products, entry["params"], entry["sensors"], entry["samples"],
                                                           frame_metadata=entry["frame_metadata"])}
        finally:
            del self.pending[token]

    def render(self, params):
        _validate_output_selection(list(params.get("sensor_paths") or self.sensors), params)
        self._reap_cancelled()
        self._require_idle_native_render()
        sensors = list(params.get("sensor_paths") or self.sensors)
        if not sensors or any(path not in self.sensors for path in sensors):
            raise ValueError("Render products must be declared in this scene")
        samples = int(params.get("samples", 1))
        if not 1 <= samples <= 4096:
            raise ValueError("Render samples must be in [1,4096]")
        for _ in range(samples):
            products = self.renderer.step(set(sensors), delta_time=1.0/60.0, ordinal=self.ordinal)
            self.sample_count += 1
        return self._publish_products(products, params, sensors, self.sample_count)

    def _publish_products(self, products, params, sensors, sample_count, *, frame_metadata=None):
        _validate_output_selection(sensors, params)
        import warp as wp
        try:
            from .ovstream_frames import ShmPublisher
        except ImportError:
            from ovstream_frames import ShmPublisher
        def ensure_publisher(array):
            if self.publisher is None:
                self.publisher = ShmPublisher(device=str(array.device), epoch=self.epoch,
                    registry_path=self.output_root / 'ovstream-resources.json')
                if getattr(self, 'gpu_identity', None):
                    self.gpu_identity["publisher_device"] = str(array.device)
        for sensor in sensors:
            if sensor not in products or not products[sensor].frames:
                raise RuntimeError(f"No frame returned for requested RenderProduct {sensor}")
        frame = products[sensors[0]].frames[-1]
        desired = params.get("render_var", "LdrColor")
        def find(name):
            found = [value for key, value in frame.render_vars.items() if key.rsplit("/", 1)[-1] == name]
            if len(found) != 1:
                raise RuntimeError(f"Requested {name} absent from {list(frame.render_vars)}")
            return found[0]
        metadata = frame_metadata or {"width": self.width, "height": self.height,
            "ordinal": self.ordinal, "native_stage_id": self.native_stage_id}
        result = {"epoch": self.epoch, "completed_samples": sample_count, "sensor_path": sensors[0],
                  "simulation_time_ns": int(params.get("simulation_time_ns", 0)),
                  "render_var": desired, **metadata}
        self.simulation_time_ns = max(self.simulation_time_ns, int(params.get("simulation_time_ns", self.simulation_time_ns)))
        semantic_outputs = params.get("semantic_outputs", False)
        if type(semantic_outputs) is not bool:
            raise ValueError("Semantic output selection must be boolean")
        if semantic_outputs:
            return self._publish_semantic_products(frame, find, result, desired, ensure_publisher,
                camera_prim_path=str(params.get("camera_prim_path", "")))
        if desired == "HdrColor":
            hdr_map = find("HdrColor").map(device=self.rtx.Device.CUDA)
            hdr_array = None
            try:
                hdr_array = wp.from_dlpack(hdr_map)
                ensure_publisher(hdr_array)
                result.update(self.publisher.publish_raw_planes_cuda([
                    ("linear_rgba16f", "rgba16f", hdr_array),
                ], presentation_rgba32f=params.get("presentation_rgba32f", False), product_id=sensors[0], aov=desired))
                result.update(transport="ovstream_shm_raw", origin="top_left")
            finally:
                hdr_array = None
                hdr_map.unmap()
            return result
        elif desired != "LdrColor":
            raise NotImplementedError(f"Exact {desired} output not implemented in spike")
        mapped = find("LdrColor").map(device=self.rtx.Device.CUDA)
        try:
            rgba = wp.from_dlpack(mapped)
            ensure_publisher(rgba)
            # Preserve RGBA bytes at every resolution. The ordinary BGRA image
            # route needs both a GPU channel swap and a CPU swap/copy on receipt;
            # the typed raw carrier already preserves the same owned-frame/ACK
            # contract without either conversion.
            result.update(self.publisher.publish_raw_planes_cuda([("rgba8", "rgba8", rgba)], product_id=sensors[0], aov=desired))
            result["transport"] = "ovstream_shm_raw"
            del rgba
        finally:
            mapped.unmap()
        return result

    def _publish_semantic_products(self, frame, find, result, desired, ensure_publisher, *, camera_prim_path):
        """Pack color and uint32 IDs from one SDK frame into one owned hold."""
        import base64
        import warp as wp
        if desired not in {"HdrColor", "LdrColor"}:
            raise ValueError("Semantic capture requires a supported color product")
        # CPU ID-map bytes are small owned metadata. Integer image pixels use
        # the existing lossless raw-plane carrier, never color upload/tonemapping.
        with ExitStack() as stack:
            color_map = stack.enter_context(find(desired).map(device=self.rtx.Device.CUDA))
            ids_map = stack.enter_context(find("SemanticSegmentation").map(device=self.rtx.Device.CUDA))
            labels_map = stack.enter_context(find("SemanticIdMap").map(device=self.rtx.Device.CPU))
            color = wp.from_dlpack(color_map)
            ids = wp.from_dlpack(ids_map)
            labels = self.np.from_dlpack(labels_map)
            width, height = result["width"], result["height"]
            if ids.dtype != wp.uint32 or ids.shape not in {(height, width), (height, width, 1)}:
                raise ValueError("Semantic segmentation must be an exact uint32 image matching color")
            if labels.dtype != self.np.uint8 or labels.ndim != 1 or not 4 <= labels.nbytes <= 4 * 1024 * 1024:
                raise ValueError("Semantic ID map must be a bounded uint8 byte tensor")
            id_map = labels.tobytes()
            ensure_publisher(color)
            color_name, color_format = (("linear_rgba16f", "rgba16f")
                                        if desired == "HdrColor" else ("rgba8", "rgba8"))
            # Optional derived HDR upload optimization only supports one plane.
            # Capture preserves exact HDR and uint32 data and uses CPU display preparation.
            result.update(self.publisher.publish_raw_planes_cuda([
                (color_name, color_format, color), ("semantic_ids", "r32u", ids)]))
            result.update(transport="ovstream_shm_raw", origin="top_left",
                          frame_start_time=frame.start_time, frame_end_time=frame.end_time,
                          camera_prim_path=camera_prim_path)
            result["semantic"] = {key: result[key] for key in (
                "width", "height", "sensor_path", "epoch", "ordinal",
                "simulation_time_ns", "frame_start_time", "frame_end_time", "origin", "camera_prim_path")}
            result["semantic"].update(id_dtype="uint32", id_map_dtype="uint8",
                id_map_base64=base64.b64encode(id_map).decode("ascii"))
            # Borrowed SDK tensor wrappers must not survive unmap.
            del labels, ids, color
        return result

    def unload(self):
        # SDK operations retain native output handles until fetched. Drain them
        # before detaching/destroying the renderer; owner process shutdown has
        # its own bounded terminate/kill path if native completion is wedged.
        # Only stages that ran are timed: an unload of an unloaded backend logs none.
        timings = self.unload_timings = {}
        if self.pending:
            with _timed(timings, "drain"):
                for entry in self.pending.values():
                    if entry.get("products") is None:
                        entry["operation"].wait().fetch()
                self.pending.clear()
        if self.publisher is not None:
            with _timed(timings, "publisher_close"):
                self.publisher.close()
                self.publisher = None
        # A failed detach leaves an OVRTX 0.6 renderer destroy-only; still
        # release the stage and renderer so the next load starts clean.
        try:
            if self.renderer is not None and self.stage is not None:
                with _timed(timings, "detach"):
                    self.renderer.detach_ovstage()
        finally:
            try:
                if self.stage is not None:
                    stage, self.stage = self.stage, None
                    self.population_owner = None
                    self.pose_population = None
                    self.native_pose_fanout = None
                    self.native_stage_id = None
                    with _timed(timings, "stage_destroy"):
                        stage.destroy()
            finally:
                if self.renderer is not None:
                    renderer, self.renderer = self.renderer, None
                    with _timed(timings, "renderer_destroy"):
                        renderer.destroy()



def _physx_schema_paths():
    """PhysX schemas ship in physx-usd-schemas from OVPhysX 0.7.

    Use package metadata only: importing physx_usd_schemas registers with a
    stock USD runtime, which the SDK interpreter does not have.
    """
    from importlib import metadata
    package = metadata.distribution("physx-usd-schemas")
    paths = [Path(package.locate_file(f"physx_usd_schemas/{module}/resources"))
             for module in ("PhysxSchema", "OmniUsdPhysicsDeformableSchema")]
    for path in paths:
        for name in ("plugInfo.json", "generatedSchema.usda"):
            if not (path / name).is_file():
                raise FileNotFoundError(f"Incomplete physx-usd-schemas: missing {path / name}")
    return paths

class PhysicsBackend:
    def __init__(self, output_root):
        from runtime_identity import verified_identity
        verified_identity()
        import numpy as np
        import ovstage
        import ovphysx
        self.np, self.ovstage, self.ovphysx = np, ovstage, ovphysx
        ovstage.population.register_usd_schemas([str(path) for path in _physx_schema_paths()])
        self.physx = self.stage = None
        self.time_ns = 0
        self.body_prims = None
        self.ordinal = 1
        self._runtime_identity = None
        self._has_stepped = False

    def runtime_identity(self):
        """Exact admission identity without hashing any file at worker start.

        Native hashes come from the install-time integrity receipt (validated
        against the lock by ``verified_identity``); the adapter identity comes
        from the record written at package build. Only the native roster and
        file existence are checked here.
        """
        if self._runtime_identity is None:
            from pathlib import PurePosixPath
            import platform
            import sys
            from physics_adapter_identity import recorded_identity
            if sys.platform not in {"win32", "linux"}:
                raise RuntimeError(f"Unvalidated public physics platform: {sys.platform}")
            adapter, adapter_components = recorded_identity()
            suffixes = {".dll", ".pyd"} if sys.platform == "win32" else {".so"}
            receipt = Path(sys.prefix) / "public-sdk-integrity.json"
            try:
                packages = json.loads(receipt.read_text(encoding="utf-8"))["packages"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise RuntimeError(f"Runtime integrity receipt unavailable: {receipt}") from error
            binaries = {}
            for package in (self.ovphysx, self.ovstage):
                root = Path(package.__file__).parent
                prefix = package.__name__ + "/"
                recorded = {key: value for key, value in (packages.get(package.__name__) or {}).items()
                            if key.startswith(prefix) and PurePosixPath(key).suffix.lower() in suffixes}
                found = {f"{prefix}{path.relative_to(root).as_posix()}" for path in root.rglob("*")
                         if path.is_file() and path.suffix.lower() in suffixes}
                if not found:
                    raise RuntimeError(f"No native binaries found for {package.__name__}")
                if found != set(recorded):
                    raise RuntimeError(f"Native binaries for {package.__name__} differ from the installed runtime receipt; reinstall")
                for key in sorted(recorded):
                    binaries[key] = recorded[key]
            try:
                from .runtime_identity import verified_identity
            except ImportError:
                from runtime_identity import verified_identity
            self._runtime_identity = {
                "backend": "public_sdk", "platform": sys.platform, "architecture": platform.machine(),
                "versions": _versions(), "runtime_lock": verified_identity(),
                "native_sha256": binaries, "adapter_sha256": adapter,
                "adapter_components": adapter_components,
            }
        return self._runtime_identity

    def dispatch(self, op, params):
        if op == "health":
            return {"loaded": self.stage is not None, "pid": os.getpid(), "versions": _versions(), "runtime_identity": self.runtime_identity(),
                    "stage_identity": id(self.stage) if self.stage is not None else None,
                    "physics_identity": id(self.physx) if self.physx is not None else None,
                    "cpu_only_policy": self.ovphysx.PhysX.get_cpu_mode() if self.physx is not None else None}
        if op == "load":
            self.unload()
            self.step_calls = self.state_read_calls = 0
            self.state_read_ms = 0.0
            self.native_state_read_sessions = 0
            self.body_prims = set(params["body_prims"]) if "body_prims" in params else None
            device = params.get("device", "gpu")
            if device not in ("cpu", "gpu"):
                raise ValueError(f"Unsupported physics device: {device}")
            self.ovphysx.PhysX.set_cpu_mode(device == "cpu")
            # A successful False setter does not override OVPHYSX_DISABLE_GPU.
            # Query effective native policy before creating a simulation; the
            # authored GPU scene flag and CUDA contexts cannot prove this mode.
            cpu_only = self.ovphysx.PhysX.get_cpu_mode()
            if cpu_only != (device == "cpu"):
                effective = "CPU-only" if cpu_only else "GPU-enabled"
                raise RuntimeError(f"Requested {device.upper()} physics conflicts with effective {effective} SDK policy; "
                                   "check OVPHYSX_DISABLE_GPU and process-wide sticky CPU mode")
            self.physx = self.ovphysx.PhysX()
            self.stage = self.ovstage.Stage("blender-public-physics")
            self.ovstage.population.open_usd(self.stage, str(Path(params["scene_path"]).resolve()), ordinal=1, domains=self.ovstage.PopulationDomain.ALL)
            self.stage.advance_write_floor(1).wait()
            self.physx.wait_task(self.physx.attach_ovstage(self.stage, read_ordinal=1))
            # CPU initial state is readable after attachment. A microscopic
            # warmup step corrupts the next solver step for rotated joint
            # chains on the measured SDK; only DirectGPU needs this bootstrap.
            self._has_stepped = device == "gpu"
            if self._has_stepped:
                self.physx.warmup()
                self.physx.wait_all()
            self.time_ns = 0
            self.ordinal = 1
            return self.dispatch("health", {})
        if op == "step":
            dt_ns = int(params["dt_ns"])
            if not 0 < dt_ns <= 1_000_000_000:
                raise ValueError("Invalid physics timestep")
            read_states = params.get("read_states", True)
            if not isinstance(read_states, bool):
                raise ValueError("read_states must be boolean")
            started = time.perf_counter()
            self.physx.step_sync(dt_ns/1e9)
            step_ms = (time.perf_counter() - started) * 1000
            self.step_calls = getattr(self, 'step_calls', 0) + 1
            self._has_stepped = True
            self.time_ns = int(params.get("time_ns", self.time_ns + dt_ns))
            result = {"simulation_time_ns": self.time_ns, "step_ms": step_ms,
                      "step_calls": self.step_calls}
            if read_states:
                result["states"] = self.read()
            result['state_read_calls'] = getattr(self, 'state_read_calls', 0)
            result['native_state_read_sessions'] = getattr(self, 'native_state_read_sessions', 0)
            result['state_read_ms'] = getattr(self, 'state_read_ms', 0.0) if read_states else 0.0
            return result
        if op == "read":
            states = self.read(params.get("body_prims", self.body_prims))
            return {"states": states, "simulation_time_ns": self.time_ns,
                    "state_read_calls": self.state_read_calls, "state_read_ms": self.state_read_ms,
                    "native_state_read_sessions": self.native_state_read_sessions}
        if op in ("write_poses", "write_velocities"):
            return self.write(op, params)
        if op == "write_masses":
            return self.write_masses(params)
        if op == "unload":
            self.unload()
            return {"unloaded": True}
        raise ValueError(f"Unsupported physics operation: {op}")

    def _columns(self, attribute):
        from ovphysx.types import SimObjectType, ObjectScope
        values = {}
        with self.ovstage.PathDictionary(self.stage) as paths:
            with self.physx.read_sync(SimObjectType.RIGID_BODY, [attribute], scope=ObjectScope.ALL) as result:
                for group in result.groups:
                    if group.is_delete or group.is_array:
                        continue
                    if group.index_map is not None or group.prim_index_map is not None:
                        raise NotImplementedError("Sparse physics read groups need explicit mapping support")
                    names = paths.get_path_strings(group.prim_list)
                    if group.tensors:
                        array = group.tensors[0].numpy()
                        values.update({name: array[i].tolist() for i, name in enumerate(names)})
        return values

    def read(self, body_prims=None):
        started = time.perf_counter()
        self.state_read_calls = getattr(self, 'state_read_calls', 0) + 1
        columns = self._state_columns()
        result = []
        requested = self.body_prims if body_prims is None else body_prims
        for path in columns["translate"]:
            if requested is not None and path not in requested:
                continue
            row = {"prim_path": path, "simulation_time_ns": self.time_ns}
            for key, values in columns.items():
                vector = values[path]
                row[key] = {"found": True, **dict(zip(("i", "j", "k", "r") if key == "orient" else ("x", "y", "z"), vector))}
            result.append(row)
        self.state_read_ms = (time.perf_counter() - started) * 1000
        return result

    def _state_columns(self):
        """Extract all publication columns from one native snapshot/session."""
        from ovphysx.types import SimObjectType, ObjectScope
        attributes = {"position": "translate", "orientation": "orient",
                      "linearVelocity": "linear_velocity", "angularVelocity": "angular_velocity"}
        columns = {key: {} for key in attributes.values()}
        with self.ovstage.PathDictionary(self.stage) as paths:
            self.native_state_read_sessions = getattr(self, 'native_state_read_sessions', 0) + 1
            with self.physx.read_sync(SimObjectType.RIGID_BODY, list(attributes), scope=ObjectScope.ALL) as result:
                for group in result.groups:
                    if group.is_delete or group.is_array:
                        raise NotImplementedError("Array/deleted physics state groups are unsupported")
                    if group.index_map is not None or group.prim_index_map is not None:
                        raise NotImplementedError("Sparse physics read groups need explicit mapping support")
                    attribute = paths.token_to_string(group.attribute)
                    if attribute not in attributes:
                        raise NotImplementedError(f"Unknown emitted physics state attribute: {attribute}")
                    names = paths.get_path_strings(group.prim_list)
                    if len(group.tensors) != 1:
                        raise ValueError("Physics state column must contain one tensor")
                    # Warp numpy() waits for CUDA copies before the context closes.
                    array = group.tensors[0].numpy()
                    values = columns[attributes[attribute]]
                    if len(array) != len(names) or len(set(names)) != len(names) or set(names) & values.keys():
                        raise ValueError("Physics state has duplicate or mismatched body rows")
                    values.update({name: array[i].tolist() for i, name in enumerate(names)})
        roster = set(columns['translate'])
        if any(set(values) != roster for values in columns.values()):
            raise ValueError("Physics state columns have inconsistent body rosters")
        return columns

    def write_masses(self, params):
        from ovphysx.types import SimObjectType
        rows = params.get("values", [])
        updates = {row["prim_path"]: float(row["mass"]) for row in rows}
        if not rows or len(updates) != len(rows):
            raise ValueError("Mass write requires unique body paths")
        masses = self._columns("mass")
        if set(updates) - set(masses):
            raise ValueError("Mass write names unknown physics bodies")
        inertias = self._columns("inertia")
        target_inertias = {}
        for path, mass in updates.items():
            if not self.np.isfinite(mass) or not 0 < mass <= self.np.finfo(self.np.float32).max or self.np.float32(mass) <= 0:
                raise ValueError("Mass must be positive finite float32")
            if self._kinematic(path):
                raise ValueError("Live mass edits require a dynamic body")
            authored = self._stage_value(path, "physics:diagonalInertia")
            diagonal = self.np.zeros(3) if authored is None else self.np.asarray(authored).reshape(-1)
            if diagonal.shape != (3,) or not self.np.isfinite(diagonal).all():
                raise ValueError("Unsupported authored inertia")
            automatic = bool((diagonal == 0).all())
            if not automatic and not (diagonal > 0).all():
                raise ValueError("Partial automatic inertia is not measured for live mass edits")
            tensor = self.np.asarray(inertias[path], dtype=self.np.float32)
            if automatic:
                tensor = (tensor.astype(self.np.float64) * float(self.np.float32(mass)) / masses[path]).astype(self.np.float32)
            if not self.np.isfinite(tensor).all() or not (self.np.diag(tensor.reshape(3, 3)) > 0).all():
                raise ValueError("Mass change produced invalid float32 inertia")
            target_inertias[path] = tensor.tolist()
        before = {name: self._columns(name) for name in ("position", "orientation", "linearVelocity", "angularVelocity")}
        # OVStage remains the authored source; this SDK does not forward its
        # mass edits to native actors, so explicitly update public native mass
        # columns too, then verify the effective values and unchanged motion.
        self.ordinal += 1
        with self.ovstage.PathDictionary(self.stage) as paths:
            for path, mass in updates.items():
                with paths.create_path_list_from_strings([path]) as plist:
                    with self.stage.query_from_path_list(plist) as query:
                        self.stage.write_attribute(query, "physics:mass", self.ordinal,
                            self.np.asarray([mass], dtype=self.np.float32), is_array=False).wait()
        self.stage.advance_write_floor(self.ordinal).wait()
        self.physx.update_from_ovstage(self.ordinal, self.ordinal)
        self.physx.wait_all()
        with self.ovstage.PathDictionary(self.stage) as paths:
            for attribute, changes, previous in (("mass", updates, masses), ("inertia", target_inertias, inertias)):
                with self.physx.write_sync(SimObjectType.RIGID_BODY, attribute) as session:
                    for group in session.groups:
                        names = paths.get_path_strings(group.prim_list)
                        if len(group.tensors) != 1 or group.tensors[0].device.is_cuda:
                            raise RuntimeError("Expected public SDK host body property column")
                        group.tensors[0].assign(self.np.asarray([changes.get(name, previous[name]) for name in names], dtype=self.np.float32))
                        # Commits are asynchronous from 0.7; fail before the next column.
                        session.commit_sync(group)
        self.physx.wait_all()
        actual = self._columns("mass")
        actual_inertias = self._columns("inertia")
        for path, mass in updates.items():
            if actual[path] != float(self.np.float32(mass)):
                raise RuntimeError(f"Native mass readback differs for {path}")
            if not self.np.allclose(actual_inertias[path], target_inertias[path], rtol=1e-6, atol=0):
                raise RuntimeError(f"Native inertia readback differs for {path}")
        if any(self._columns(name) != values for name, values in before.items()):
            raise RuntimeError("Live mass edit changed native body motion")
        return {"status": "ok", "body_count": len(updates), "masses": {path: actual[path] for path in updates},
                "inertias": {path: actual_inertias[path] for path in updates},
                "simulation_time_ns": self.time_ns, "ordinal": self.ordinal}

    def write(self, op, params):
        from ovphysx.types import SimObjectType
        import warp as wp
        fields = (("translate", "position"), ("orient", "orientation")) if op == "write_poses" else (("linear_velocity", "linearVelocity"), ("angular_velocity", "angularVelocity"))
        rows = params.get("values", params.get("poses", params.get("velocities", [])))
        updates = {v["prim_path"]: v for v in rows}
        requested_count = len(rows)
        if len(updates) != requested_count:
            raise ValueError("Physics write paths must be unique")
        if op == "write_velocities":
            return self._write_velocities(updates)
        if set(updates) - set(self._columns("position")):
            raise ValueError("Pose write names unknown physics bodies")
        for row in updates.values():
            for key, count in (("translate", 3), ("orient", 4)):
                value = self.np.asarray(row[key], dtype=self.np.float64)
                if value.shape != (count,) or not self.np.isfinite(value).all():
                    raise ValueError(f"Invalid body {key}")
            if float(self.np.linalg.norm(row["orient"])) < 1e-12:
                raise ValueError("Invalid body orientation")
        if op == "write_poses" and not self._has_stepped:
            self._write_stage_poses(updates)
            return {"status": "ok", "body_count": requested_count}
        if op == "write_poses" and not params.get("reset", False):
            kinematics = {path: value for path, value in updates.items() if self._kinematic(path)}
            if kinematics:
                self._write_stage_poses(kinematics)
                updates = {path: value for path, value in updates.items() if path not in kinematics}
                if not updates:
                    return {"status": "ok", "body_count": requested_count}
        for key, attribute in fields:
            current = self._columns(attribute)
            missing = set(updates) - set(current)
            if missing:
                raise ValueError(f"Unknown physics bodies: {sorted(missing)}")
            with self.ovstage.PathDictionary(self.stage) as paths:
                with self.physx.write_sync(SimObjectType.RIGID_BODY, attribute) as session:
                    for group in session.groups:
                        names = paths.get_path_strings(group.prim_list)
                        rows = []
                        for name in names:
                            update = updates.get(name, {})
                            value = update.get(key, update.get({"linear_velocity": "linear", "angular_velocity": "angular"}.get(key, key), current[name]))
                            if isinstance(value, dict):
                                axes = (("i", "j", "k", "r") if "i" in value else ("x", "y", "z", "w")) if key == "orient" else ("x", "y", "z")
                                value = [value[k] for k in axes]
                            rows.append(value)
                        if len(group.tensors) != 1:
                            raise NotImplementedError("Expected fixed-width rigid-body write group")
                        tensor = group.tensors[0]
                        tensor.assign(self.np.asarray(rows, dtype=self.np.float32))
                        if tensor.device.is_cuda:
                            session.commit_sync(group, cuda_stream=int(wp.get_stream(tensor.device).cuda_stream or 1))
                        else:
                            session.commit_sync(group)
            self.physx.wait_all()
        return {"status": "ok", "body_count": requested_count}

    def _write_velocities(self, updates):
        # Public Python session writes select all rigid bodies. Publishing an
        # untouched kinematic row through that API still attempts forbidden
        # setLinearVelocity calls. OVStage queries select exactly the caller's
        # bodies and are the documented control-edit path.
        ovs, np = self.ovstage, self.np
        current = self._columns("position")
        if set(updates) - set(current):
            raise ValueError("Velocity write names unknown physics bodies")
        prepared = []
        for path, update in updates.items():
            if self._kinematic(path):
                raise ValueError("Kinematic motion requires a pose target")
            for key, alias, attribute in (("linear_velocity", "linear", "physics:velocity"), ("angular_velocity", "angular", "physics:angularVelocity")):
                if key not in update and alias not in update:
                    continue
                value = update.get(key, update.get(alias))
                if isinstance(value, dict):
                    value = [value[k] for k in ("x", "y", "z")]
                data = np.asarray(value, dtype=np.float32).reshape(1, 3)
                if key == "angular_velocity":
                    data = np.rad2deg(data)
                if not np.isfinite(data).all():
                    raise ValueError("Nonfinite velocity")
                prepared.append((path, attribute, data))
        self.ordinal += 1
        with ovs.PathDictionary(self.stage) as paths:
            for path, attribute, data in prepared:
                with paths.create_path_list_from_strings([path]) as plist:
                    with self.stage.query_from_path_list(plist) as query:
                        self.stage.write_attribute(query, attribute, self.ordinal, data, is_array=False).wait()
        self.stage.advance_write_floor(self.ordinal).wait()
        self.physx.update_from_ovstage(self.ordinal, self.ordinal)
        self.physx.wait_all()
        return {"status": "ok", "body_count": len(updates)}

    def _stage_value(self, path, attribute):
        ovs = self.ovstage
        with ovs.PathDictionary(self.stage) as paths:
            with paths.create_path_list_from_strings([path]) as plist:
                with self.stage.query_from_path_list(plist) as query:
                    with self.stage.read_attributes(query, [paths.intern_token(attribute)], ovs.OrdinalRange.latest(self.ordinal)) as result:
                        result.wait()
                        for group in result.groups():
                            try:
                                if not group.is_delete:
                                    return self.np.asarray(group.array(0)).copy()
                            finally:
                                self.stage.release_group(group)
        return None

    def _kinematic(self, path):
        value = self._stage_value(path, "physics:kinematicEnabled")
        return bool(value.reshape(-1)[0]) if value is not None else False

    def _parent_world(self, path):
        """Row-vector world matrix that ovphysx composes a body's ``omni:xform`` with.

        OVPhysX 0.7 composes control-edited ``omni:xform`` up the ancestor
        chain and stops at ``omni:resetXformStack``. Ancestors without a
        transform row (pseudo-root, Scope) contribute identity.
        """
        np = self.np
        world = np.eye(4, dtype=np.float64)

        def resets(prim):
            value = self._stage_value(prim, "omni:resetXformStack")
            return value is not None and bool(np.asarray(value).reshape(-1)[0])

        if resets(path):
            return world
        parts = path.split("/")[1:-1]
        for depth in range(len(parts), 0, -1):
            ancestor = "/" + "/".join(parts[:depth])
            local = self._stage_value(ancestor, "omni:xform")
            if local is not None:
                world = world @ np.asarray(local, dtype=np.float64).reshape(4, 4)
            if resets(ancestor):
                break
        return world

    def _write_stage_poses(self, updates):
        ovs, np = self.ovstage, self.np
        # Pose edits carry world poses; omni:xform is parent-local. Resolve the
        # (static) parent chain at the sealed ordinal before writing.
        parents = {path: self._parent_world(path) for path in updates}
        self.ordinal += 1
        with ovs.PathDictionary(self.stage) as paths:
            for path, pose in updates.items():
                x, y, z, w = pose["orient"]
                length = float(np.linalg.norm([x, y, z, w]))
                if not np.isfinite(length) or length < 1e-12:
                    raise ValueError("Invalid kinematic orientation")
                x, y, z, w = (v / length for v in (x, y, z, w))
                matrix = np.eye(4, dtype=np.float64)
                matrix[:3, :3] = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                    [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                    [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
                ]).T
                matrix[3, :3] = pose["translate"]
                with paths.create_path_list_from_strings([path]) as plist:
                    with self.stage.query_from_path_list(plist) as query:
                        # Preserve body scale, as the authored pose API contains
                        # only rotation and translation.
                        with self.stage.read_attributes(query, [paths.intern_token("omni:fabric:worldMatrix")], ovs.OrdinalRange.latest(self.ordinal - 1)) as result:
                            result.wait()
                            for group in result.groups():
                                try:
                                    if not group.is_delete:
                                        previous = np.asarray(group.array(0)).reshape(4, 4)
                                        matrix[:3, :3] *= np.linalg.norm(previous[:3, :3], axis=1)[:, None]
                                finally:
                                    self.stage.release_group(group)
                        local = matrix @ np.linalg.inv(parents[path])
                        for attribute, value in (("omni:xform", local), ("omni:fabric:worldMatrix", matrix)):
                            self.stage.write_attribute(query, attribute, self.ordinal, value[None], is_array=False, semantic=ovs.AttributeSemantic.MATRIX).wait()
        self.stage.advance_write_floor(self.ordinal).wait()
        self.physx.update_from_ovstage(self.ordinal, self.ordinal)
        self.physx.wait_all()

    def unload(self):
        # Detach is a task from 0.7; the Stage must outlive a successful detach.
        timings = self.unload_timings = {}
        try:
            if self.physx is not None:
                try:
                    if self.stage is not None:
                        with _timed(timings, "detach"):
                            self.physx.wait_task(self.physx.detach_ovstage())
                finally:
                    physx, self.physx = self.physx, None
                    with _timed(timings, "physx_destroy"):
                        physx.destroy()
        finally:
            if self.stage is not None:
                stage, self.stage = self.stage, None
                with _timed(timings, "stage_destroy"):
                    stage.destroy()
