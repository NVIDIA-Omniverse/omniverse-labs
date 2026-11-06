# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-step bake capture and immutable, disk-backed sample interchange.

Capture owns a separate generation, but uses the existing OVPhysX adapter.  The
viewport's publication queue is deliberately not an input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Iterator


def _matrix(value):
    return tuple(tuple(float(x) for x in row) for row in value)


def _main_thread():
    import threading
    if threading.current_thread() is not threading.main_thread():
        raise BakeCaptureError("bake RNA operations require Blender's main thread")


def _source_stamp(scene):
    """A copied stamp taken on the main thread, including bound animation edits."""
    from .physics_edit_snapshot import capture
    from .physics_pose_presentation import _curves
    def rna_values(group, *, nested=False):
        result = [("__type__", group.bl_rna.identifier)]
        for prop in group.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            if prop.type == "COLLECTION":
                if nested:
                    result.append((prop.identifier, tuple(rna_values(item, nested=True)
                        for item in getattr(group, prop.identifier))))
                continue
            if prop.is_readonly:
                continue
            value = getattr(group, prop.identifier)
            if prop.type == "POINTER":
                value = (getattr(value, "session_uid", 0), getattr(value, "name", ""))
            elif getattr(prop, "is_array", False):
                value = tuple(value)
            result.append((prop.identifier, value))
        return tuple(result)

    values = [capture(scene).leaves, scene.frame_current, scene.frame_subframe,
              scene.render.fps, scene.render.fps_base]
    for obj in scene.objects:
        animation = obj.animation_data
        curves = []
        if animation:
            for curve in _curves(animation):
                curves.append((rna_values(curve),
                    tuple(rna_values(point) for point in curve.keyframe_points),
                    tuple(rna_values(point) for point in curve.sampled_points),
                    tuple(rna_values(modifier, nested=True) for modifier in curve.modifiers)))
            curves.append(tuple((track.mute, track.is_solo, tuple(
                (rna_values(strip, nested=True), getattr(getattr(strip, "action_slot", None), "identifier", ""))
                for strip in track.strips)) for track in animation.nla_tracks))
            curves.append(tuple((driver.data_path, driver.array_index, rna_values(driver.driver, nested=True))
                for driver in animation.drivers))
        geometry = hashlib.sha256()
        if obj.type == "MESH":
            for vertex in obj.data.vertices:
                geometry.update(repr(tuple(vertex.co)).encode())
            for edge in obj.data.edges:
                geometry.update(repr(tuple(edge.vertices)).encode())
            for polygon in obj.data.polygons:
                geometry.update(repr((tuple(polygon.vertices), polygon.material_index)).encode())
            if obj.data.shape_keys:
                for key in obj.data.shape_keys.key_blocks:
                    geometry.update(repr((key.name, key.value, key.mute)).encode())
                    for point in key.data:
                        geometry.update(repr(tuple(point.co)).encode())
        values.append((int(obj.session_uid), _matrix(obj.matrix_basis),
                       int(obj.parent.session_uid) if obj.parent else 0,
                       int(animation.action.session_uid) if animation and animation.action else 0,
                       getattr(getattr(animation, "action_slot", None), "identifier", ""), tuple(curves),
                       obj.rotation_mode, tuple(obj.lock_location), tuple(obj.lock_rotation), tuple(obj.lock_scale),
                       tuple((p.name, _matrix(p.matrix_basis), p.rotation_mode,
                              tuple(p.lock_location), tuple(p.lock_rotation), tuple(p.lock_scale),
                              tuple(rna_values(c) for c in p.constraints))
                             for p in getattr(obj.pose, "bones", ())),
                       geometry.hexdigest(), tuple(rna_values(c) for c in obj.constraints),
                       tuple(rna_values(m) for m in obj.modifiers)))
    return repr(values)


def _conversion_context(scene, targets):
    from .physics_pose_presentation import resolve
    from .physics_bake_configuration import channel_configuration, TRANSFORMS, DELTAS
    result = {}
    for target in targets:
        obj, channel = resolve(scene, target)
        item = {"object_world": _matrix(obj.matrix_world),
                "parent_world": _matrix(obj.parent.matrix_world) if obj.parent else None,
                "parent_object_uid": int(obj.parent.session_uid) if obj.parent else 0,
                "parent_type": obj.parent_type, "parent_bone": obj.parent_bone,
                "parent_inverse": _matrix(obj.matrix_parent_inverse),
                "rotation_mode": channel.rotation_mode,
                "basis": _matrix(channel.matrix_basis),
                "delta_location": tuple(obj.delta_location),
                "delta_rotation_euler": tuple(obj.delta_rotation_euler),
                "delta_rotation_quaternion": tuple(obj.delta_rotation_quaternion),
                "delta_scale": tuple(obj.delta_scale)}
        item["delta_configuration"] = channel_configuration(obj, obj, DELTAS, allow_drivers=True)
        if obj.parent and obj.parent_type == "BONE" and obj.parent.pose:
            parent_bone = obj.parent.pose.bones.get(obj.parent_bone)
            if parent_bone:
                item["object_parent_bone_pose"] = _matrix(parent_bone.matrix)
                item["object_parent_bone_length"] = parent_bone.bone.length
        if target.kind == "BONE":
            item.update(rest=_matrix(channel.bone.matrix_local),
                        parent_pose=_matrix(channel.parent.matrix) if channel.parent else None,
                        parent_rest=_matrix(channel.parent.bone.matrix_local) if channel.parent else None,
                        parent_bone=channel.parent.name if channel.parent else "",
                        inherit_scale=channel.bone.inherit_scale,
                        use_local_location=channel.bone.use_local_location,
                        use_inherit_rotation=channel.bone.use_inherit_rotation)
        # Retain unselected intermediate parents too: a selected grandparent
        # changes their evaluated world even when their channels remain intact.
        objects, bones = {}, {}
        def capture_bone(armature, pose_bone):
            key = (int(armature.session_uid), pose_bone.name)
            if key in bones:
                return
            bones[key] = dict(object_uid=key[0], name=key[1], pose=_matrix(pose_bone.matrix),
                identity=pose_bone.bone.ov.persistent_id,
                configuration=channel_configuration(armature, pose_bone, TRANSFORMS, allow_drivers=True),
                basis=_matrix(pose_bone.matrix_basis), rest=_matrix(pose_bone.bone.matrix_local),
                parent=pose_bone.parent.name if pose_bone.parent else "",
                inherit_scale=pose_bone.bone.inherit_scale,
                use_local_location=pose_bone.bone.use_local_location,
                use_inherit_rotation=pose_bone.bone.use_inherit_rotation,
                use_relative_parent=pose_bone.bone.use_relative_parent,
                constrained=any(not c.mute for c in pose_bone.constraints))
            if pose_bone.parent:
                capture_bone(armature, pose_bone.parent)
        def capture_object(value):
            uid = int(value.session_uid)
            if uid in objects:
                return
            objects[uid] = dict(uid=uid, world=_matrix(value.matrix_world),
                identity=value.ov.physics.persistent_id,
                parent_inverse=_matrix(value.matrix_parent_inverse),
                delta_configuration=channel_configuration(value, value, DELTAS, allow_drivers=True),
                configuration=channel_configuration(value, value, TRANSFORMS + DELTAS, allow_drivers=True),
                rotation_mode=value.rotation_mode,
                parent_uid=int(value.parent.session_uid) if value.parent else 0,
                parent_type=value.parent_type, parent_bone=value.parent_bone,
                constrained=any(not c.mute for c in value.constraints))
            if value.parent:
                capture_object(value.parent)
                if value.parent_type == "BONE" and value.parent.pose:
                    parent_bone = value.parent.pose.bones.get(value.parent_bone)
                    if parent_bone:
                        capture_bone(value.parent, parent_bone)
        capture_object(obj)
        if target.kind == "BONE":
            capture_bone(obj, channel)
        item["hierarchy_objects"] = list(objects.values())
        item["hierarchy_bones"] = list(bones.values())
        result[target.body_path] = item
    return result


def _capture_targets(scene, generation):
    """Include sparse legacy Mesh outputs without migrating or assigning IDs."""
    import bpy
    from pxr import Usd, UsdGeom
    from .physics_body_prims import discover_dynamic_body_prims
    from .physics_presentation_targets import presentation_targets, PhysicsPresentationTarget
    from .physics_frames import PhysicsUnits, evaluated_owner_transform, matrix_to_frame
    targets = [t for t in presentation_targets(generation) if t.enabled and t.role == "DYNAMIC"]
    missing = set(discover_dynamic_body_prims(generation.usd_path, root="")) - {t.body_path for t in targets}
    if missing:
        stage = Usd.Stage.Open(generation.usd_path)
        cache = UsdGeom.XformCache()
        units = PhysicsUnits.from_scene(scene)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        objects = {int(o.session_uid): o for o in scene.objects}
        for identity, mapping in generation.blender_prim_paths.items():
            if identity.kind != "OBJECT" or identity.session_uid not in objects:
                continue
            obj = objects[identity.session_uid]
            for path in (mapping.object_path, mapping.schema_path):
                if path not in missing:
                    continue
                owner = evaluated_owner_transform(obj, depsgraph, units=units)
                matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
                body = matrix_to_frame(tuple(tuple(matrix[j][i] for j in range(4)) for i in range(4)))
                targets.append(PhysicsPresentationTarget(path, f"legacy:{identity.session_uid}",
                    identity.session_uid, obj.ov.physics.persistent_id, "", "", "DYNAMIC", True,
                    units.meters_per_blender_unit, owner.uniform_scale,
                    owner.frame.inverse().compose(body), body, owner.axis_scale))
                missing.remove(path)
    if missing:
        raise BakeCaptureError("dynamic body has no Blender bake target: " + ", ".join(sorted(missing)))
    return tuple(targets)


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_configuration(scene):
    from .physics_edit_snapshot import capture
    return tuple(row for row in capture(scene).leaves if row[3] not in {
        "body_pose", "body_basis", "handle_pose", "collider_source_pose"})


class BakeCaptureSession:
    """Main-thread transaction advanced by a modal operator or a headless caller.

    ``poll`` performs at most one completed step and schedules the next.  Native
    calls execute on one worker. ``cancel`` is cooperative at a step boundary;
    ``close`` waits for teardown and restores the original Blender view state.
    """

    def __init__(self, scene, settings, directory, *, output_paths=None, controller_factory=None):
        _main_thread()
        from concurrent.futures import ThreadPoolExecutor
        from . import scene_generation_sessions as sessions
        from .scene_generation import SceneGenerationOwner
        from .physics_authoring import require_scene_ready
        from .physics_presentation_targets import presentation_targets
        from .interactive_operator_state import suppress_interactive_edit_bridge
        from dataclasses import asdict
        require_scene_ready(scene)
        self.scene, self.settings = scene, settings
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.artifact = None
        self.error = None
        self.cancelled = False
        self.closed = False
        self.step = 0
        self.progress = 0.0
        self.future = None
        self.worker = None
        self.executor = None
        self.owner = SceneGenerationOwner(self.directory / "generation")
        self.saved_frame = (scene.frame_current, scene.frame_subframe)
        self.saved_channels = [(channel, {name: tuple(getattr(channel, name)) for name in
            ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")})
            for obj in scene.objects for channel in (obj, *getattr(obj.pose, "bones", ()))]
        self.saved_stopped = set(sessions._presentation_stopped)
        self.runtimes = ()
        import bpy
        self.was_playing = bool(bpy.context.screen and bpy.context.screen.is_animation_playing)
        self.saved_generation = sessions._owners.get(int(scene.session_uid))
        self.saved_generation = self.saved_generation.current_generation if self.saved_generation else None
        try:
            self.runtimes = sessions.acquire_bake_capture(self)
            if self.was_playing:
                bpy.ops.screen.animation_cancel(restore_frame=False)
            with suppress_interactive_edit_bridge():
                # Restore authored bases before evaluating the explicit initial frame.
                from mathutils import Matrix
                retained = sessions._physics_initial_bases.get(int(scene.session_uid), {})
                for obj in scene.objects:
                    for name, channel in [("", obj), *((p.name, p) for p in getattr(obj.pose, "bones", ()))]:
                        basis = retained.get((int(obj.session_uid), name))
                        if basis is not None:
                            channel.matrix_basis = Matrix(basis)
                for runtime in self.runtimes:
                    for record in runtime.playback_lock._presentation_records.values():
                        record["channel"].matrix_basis = record["initial_basis"]
                scene.frame_set(settings.initial_frame)
                for layer in scene.view_layers:
                    layer.update()
                self.generation = self.owner.replace(scene)
            self.targets = _capture_targets(scene, self.generation)
            if not self.targets:
                raise BakeCaptureError("generation has no mapped dynamic bake targets")
            self.output_paths = tuple(sorted(output_paths if output_paths is not None else
                                            (t.body_path for t in self.targets)))
            if not self.output_paths or not set(self.output_paths) <= {t.body_path for t in self.targets}:
                raise BakeCaptureError("bake selection contains an unmapped or disabled target")
            self.manifest = {"timing": settings.timing(), "generation_digest": self.generation.digest,
                             "targets": [asdict(t) for t in self.targets],
                             "output_paths": self.output_paths}
            from pxr import Usd
            stage = Usd.Stage.Open(self.generation.usd_path)
            self.manifest["generation_layers"] = {layer.realPath: _file_digest(layer.realPath)
                for layer in stage.GetUsedLayers() if layer.realPath}
            self.stamp = _source_stamp(scene)
            from .physics_kinematic_inputs import kinematic_targets, capture_kinematic_inputs
            capture_kinematic_inputs(scene, self.generation)  # Reject unsupported dependencies before native activation.
            self.manifest["kinematic_input_paths"] = [t.body_path for t in kinematic_targets(self.generation)]
            self.frozen_configuration = _frozen_configuration(scene)
            self.expected_channels = {id(channel): {name: tuple(getattr(channel, name)) for name in values}
                                      for channel, values in self.saved_channels}
            self.worker = _CaptureWorker(self.generation, settings, self.directory,
                                         self.runtimes, controller_factory)
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ovphysx-bake")
            self.future = self.executor.submit(self.worker.begin)
            self.phase = "begin"
        except BaseException:
            self.close()
            raise

    def cancel(self):
        _main_thread()
        self.cancelled = True

    def poll(self):
        _main_thread()
        if self.closed:
            return True
        from . import scene_generation_sessions as sessions
        from .physics_authoring import require_scene_ready
        from .interactive_operator_state import suppress_interactive_edit_bridge
        try:
            require_scene_ready(self.scene)
            if self.cancelled:
                raise BakeCaptureCancelled("bake capture cancelled")
            if _source_stamp(self.scene) != self.stamp:
                raise BakeCaptureError("authoring changed during bake capture")
            owner = sessions._owners.get(int(self.scene.session_uid))
            if (owner.current_generation if owner else None) is not self.saved_generation:
                raise BakeCaptureError("accepted generation changed during bake capture")
            if not self.future.done():
                return False
            result = self.future.result()
            if self.phase == "finish":
                self.artifact = result
                self.progress = 1.0
                self.close()
                return True
            if self.phase == "begin":
                if not {t.body_path for t in self.targets} <= set(result):
                    raise BakeCaptureError("compiled target missing from native output roster")
                self.manifest["body_paths"] = result
                self.manifest["runtime"] = self.worker.provenance
            else:
                self.step += 1
            if self.step > self.settings.total_steps:
                self.future = self.executor.submit(self.worker.finish, self.manifest)
                self.phase = "finish"
                return False
            frame, substep = divmod(self.step, self.settings.substeps)
            frame += self.settings.initial_frame
            with suppress_interactive_edit_bridge():
                self.scene.frame_set(frame, subframe=substep / self.settings.substeps)
                for layer in self.scene.view_layers:
                    layer.update()
            context = _conversion_context(self.scene, self.targets) if self.settings.sample_step(self.step) else {}
            from .physics_kinematic_inputs import capture_kinematic_inputs
            inputs = capture_kinematic_inputs(self.scene, self.generation)
            if _frozen_configuration(self.scene) != self.frozen_configuration:
                raise BakeCaptureError("animated physics configuration differs from the frozen initial conditions")
            self.stamp = _source_stamp(self.scene)
            self.expected_channels = {id(channel): {name: tuple(getattr(channel, name)) for name in values}
                                      for channel, values in self.saved_channels}
            self.future = self.executor.submit(self.worker.step, self.step, frame, context,
                                                self.settings.sample_step(self.step), inputs)
            self.phase = "step"
            self.progress = self.step / max(1, self.settings.total_steps)
            return False
        except BaseException as exc:
            self.error = exc
            self.close()
            raise

    def close(self):
        _main_thread()
        if self.closed:
            return
        from . import scene_generation_sessions as sessions
        from .interactive_operator_state import suppress_interactive_edit_bridge
        authored_changes = {}
        for channel, values in self.saved_channels:
            try:
                current = {name: tuple(getattr(channel, name)) for name in values}
                expected = getattr(self, "expected_channels", {}).get(id(channel), current)
                # Only values still equal to our last evaluated sample belong
                # to capture. Preserve concurrent edits on every cleanup route,
                # including direct close, cancellation, save and disable.
                authored_changes[id(channel)] = {
                    name: tuple(new if new != prior else saved
                                for new, prior, saved in zip(current[name], expected[name], original))
                    for name, original in values.items()}
            except ReferenceError:
                pass
        try:
            if self.executor:
                try:
                    self.executor.submit(self.worker.close).result()
                finally:
                    self.executor.shutdown(wait=True)
        finally:
            try:
                with suppress_interactive_edit_bridge():
                    self.scene.frame_set(self.saved_frame[0], subframe=self.saved_frame[1])
                    for channel, values in self.saved_channels:
                        try:
                            for name, value in authored_changes.get(id(channel), values).items():
                                setattr(channel, name, value)
                        except ReferenceError:
                            pass
                    for layer in self.scene.view_layers:
                        layer.update()
                    sessions._presentation_stopped.clear()
                    sessions._presentation_stopped.update(self.saved_stopped)
            finally:
                sessions.release_bake_capture(self)
                self.closed = True
                if self.artifact is None:
                    import shutil
                    # This transaction created the directory exclusively above.
                    shutil.rmtree(self.directory)
                if self.was_playing:
                    import bpy
                    if bpy.context.screen and not bpy.context.screen.is_animation_playing:
                        bpy.ops.screen.animation_play()


class BakeCaptureError(RuntimeError):
    pass


class BakeCaptureCancelled(BakeCaptureError):
    pass


@dataclass(frozen=True)
class BakeCaptureSettings:
    initial_frame: int
    start_frame: int
    end_frame: int
    fps: int
    fps_base: float = 1.0
    interval: int = 1
    substeps: int = 1

    def __post_init__(self):
        for name in ("initial_frame", "start_frame", "end_frame", "fps", "interval", "substeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.start_frame < self.initial_frame or self.end_frame < self.start_frame:
            raise ValueError("bake range must start at or after the initial-condition frame")
        if min(self.fps, self.interval, self.substeps) < 1:
            raise ValueError("fps, interval and substeps must be positive")
        if not math.isfinite(self.fps_base) or self.fps_base <= 0:
            raise ValueError("fps_base must be positive and finite")
        if self.step_rate < 1 or self.timestep_ns < 1:
            raise ValueError("fixed-step frequency is outside the native scheduler range")

    @property
    def step_rate(self) -> float:
        return self.fps * self.substeps / self.fps_base

    @property
    def timestep_ns(self) -> int:
        return int(1_000_000_000 / self.step_rate)

    @property
    def total_steps(self) -> int:
        return (self.end_frame - self.initial_frame) * self.substeps

    @property
    def sample_frames(self) -> range:
        # An interval never creates an off-grid final sample.
        return range(self.start_frame, self.end_frame + 1, self.interval)

    def sample_step(self, step: int) -> bool:
        frame, remainder = divmod(step, self.substeps)
        frame += self.initial_frame
        return not remainder and frame in self.sample_frames

    def timing(self) -> dict:
        exact = Fraction(str(self.fps_base)) / (self.fps * self.substeps)
        return {"fps": self.fps, "fps_base": self.fps_base,
                "substeps": self.substeps, "interval": self.interval,
                "initial_frame": self.initial_frame, "start_frame": self.start_frame,
                "end_frame": self.end_frame, "total_steps": self.total_steps,
                "timestep_ns": self.timestep_ns,
                "duration_ns": self.total_steps * self.timestep_ns,
                "requested_step_seconds": [exact.numerator, exact.denominator]}


@dataclass(frozen=True)
class BakeCaptureArtifact:
    """A completed capture. Consumers stream samples; no RNA objects are retained."""

    directory: Path
    manifest_digest: str

    def manifest(self) -> dict:
        raw = (self.directory / "manifest.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest_digest:
            raise BakeCaptureError("bake manifest changed after capture")
        return json.loads(raw)

    def samples(self) -> Iterator[dict]:
        manifest = self.manifest()
        path = self.directory / "samples.jsonl"
        digest = hashlib.sha256()
        count = 0
        with path.open("rb") as source:
            for line in source:
                digest.update(line)
                count += 1
                yield json.loads(line)
        if count != manifest["sample_count"] or digest.hexdigest() != manifest["samples_sha256"]:
            raise BakeCaptureError("bake samples changed after capture")


class _SampleWriter:
    def __init__(self, directory: Path, paths: tuple[str, ...]):
        self.directory = directory
        self.paths = paths
        self.count = 0
        self.digest = hashlib.sha256()
        self.stream = (directory / "samples.partial").open("xb")

    def append(self, frame: int, step: int, poses, context: dict, timestep_ns: int):
        values = {}
        for pose in poses:
            if pose.prim_path in values:
                raise BakeCaptureError("duplicate body in bake sample")
            position, rotation = tuple(pose.translate), tuple(pose.orient)
            if len(position) != 3 or len(rotation) != 4 or not all(
                math.isfinite(x) for x in (*position, *rotation)
            ) or abs(sum(x*x for x in rotation) - 1) > 1e-3:
                raise BakeCaptureError("invalid body pose in bake sample")
            values[pose.prim_path] = {"position": position, "rotation_xyzw": rotation}
        if set(values) != set(self.paths):
            raise BakeCaptureError("incomplete native bake sample; required body roster changed")
        raw = (json.dumps({"frame": frame, "step": step,
                           "simulation_time_ns": step * timestep_ns,
                           "poses": values, "context": context}, sort_keys=True,
                          allow_nan=False, separators=(",", ":")) + "\n").encode()
        self.stream.write(raw)
        self.digest.update(raw)
        self.count += 1

    def finish(self, manifest: dict) -> BakeCaptureArtifact:
        self.stream.close()
        (self.directory / "samples.partial").replace(self.directory / "samples.jsonl")
        manifest = dict(manifest, sample_count=self.count,
                        samples_sha256=self.digest.hexdigest(), format_version=1)
        raw = json.dumps(manifest, sort_keys=True, allow_nan=False, indent=2).encode()
        (self.directory / "manifest.json").write_bytes(raw)
        return BakeCaptureArtifact(self.directory, hashlib.sha256(raw).hexdigest())

    def abort(self):
        self.stream.close()
        (self.directory / "samples.partial").unlink(missing_ok=True)


class _CaptureWorker:
    """All methods run on the same task-owned executor, without accessing RNA."""

    def __init__(self, generation, settings, directory, runtimes, controller_factory=None):
        self.generation = generation
        self.settings = settings
        self.directory = directory
        self.runtimes = runtimes
        self.controller_factory = controller_factory
        self.held = []
        self.predecessors = []
        self.adapter = None
        self.writer = None

    def begin(self):
        from .generation_runtime_adapters import OvphysxGenerationAdapter
        from .ovphysx_stage import OvphysxStageController
        try:
            for runtime in self.runtimes:
                for lock in (runtime._preparation_lock, runtime.scheduler._tick_lock):
                    lock.acquire()
                    self.held.append(lock)
                adapter = runtime.ovphysx
                controller = adapter.controller
                if controller is not None:
                    controller.stop_async_playback(wait=True, invalidate=True)
                    controller.adopt_async_producer_position()
                    from .physics_kinematic_inputs import kinematic_targets
                    input_history = dict(getattr(controller, "_kinematic_input_history", {}))
                    if kinematic_targets(adapter._active_scene_generation) and set(input_history) != set(range(1, controller.step_count+1)):
                        raise BakeCaptureError("cannot restore prior kinematic playback without complete input history")
                    predecessor = (adapter, adapter._active_scene_generation,
                                              adapter._initial_conditions, controller.step_count,
                                              adapter.physics_generation, runtime.scheduler,
                                              runtime.scheduler._controller is controller,
                                              controller.completed, controller.max_steps_limit, input_history)
                    if adapter.deactivate() not in {"stopped", "not_found"}:
                        runtime.reuse_blocked = True
                        raise BakeCaptureError("cannot acquire exclusive physics runtime")
                    self.predecessors.append(predecessor)

            def factory(config):
                configured = replace(config, physics_fps=self.settings.step_rate,
                                     update_fps=self.settings.step_rate,
                                     max_steps=max(1, self.settings.total_steps))
                controller = (self.controller_factory or OvphysxStageController)(configured)
                if (controller.config.timestep_ns != self.settings.timestep_ns or
                        controller.config.steps_per_update != 1):
                    raise BakeCaptureError("controller factory changed the bake timing contract")
                return controller

            self.adapter = OvphysxGenerationAdapter(factory)
            if not self.adapter.activate(self.generation):
                raise BakeCaptureError(self.adapter.last_error or "bake activation failed")
            controller = self.adapter.controller
            self.provenance = {"device": controller.config.device,
                               "server": controller.config.server,
                               "native_client_module": controller.config.ovphysx_native_client_module,
                               "timestep_ns": controller.config.timestep_ns,
                               "steps_per_update": controller.config.steps_per_update}
            if controller.step_count != 0:
                raise BakeCaptureError("bake activation did not retain the initial-condition sample")
            self.initial_time_ns = controller.current_simulation_time_ns
            paths = tuple(sorted(self.adapter.dynamic_body_paths))
            if not paths:
                raise BakeCaptureError("bake has no enabled dynamic output bodies")
            self.writer = _SampleWriter(self.directory, paths)
            return paths
        except BaseException:
            self.close()
            raise

    def step(self, expected_step, frame, context, sample, inputs=None):
        from .ovphysx_stage import OvphysxStageStatus
        controller = self.adapter.controller
        if expected_step:
            if controller.step_count != expected_step - 1:
                raise BakeCaptureError("nonsequential bake step")
            if inputs is not None:
                from .physics_kinematic_inputs import apply_kinematic_inputs
                expected_frame = self.settings.initial_frame + expected_step / self.settings.substeps
                if abs(inputs.frame - expected_frame) > 1e-6:
                    raise BakeCaptureError("kinematic input time differs from the scheduled step")
                apply_kinematic_inputs(controller, inputs, self.generation)
            result = controller.tick(max_steps=expected_step)
            if result.status != OvphysxStageStatus.OK or controller.step_count != expected_step:
                raise BakeCaptureError(controller.last_error or "native bake step failed")
        elif controller.step_count:
            raise BakeCaptureError("initial bake sample was overwritten")
        if controller.current_simulation_time_ns - self.initial_time_ns != expected_step * self.settings.timestep_ns:
            raise BakeCaptureError("native simulation time differs from the fixed-step schedule")
        poses = controller.physics_pose_set(controller.current_simulation_time_ns)
        # Validate every step, including pre-roll and unsampled interval frames.
        if {pose.prim_path for pose in poses} != set(self.writer.paths):
            raise BakeCaptureError("native worker omitted a required body during capture")
        if sample:
            self.writer.append(frame, expected_step, poses, context, self.settings.timestep_ns)
        return expected_step

    def finish(self, manifest):
        if any(_file_digest(path) != digest for path, digest in manifest["generation_layers"].items()):
            raise BakeCaptureError("frozen generation changed during capture")
        manifest = dict(manifest, native_initial_time_ns=self.initial_time_ns,
                        native_final_time_ns=self.adapter.controller.current_simulation_time_ns)
        artifact = self.writer.finish(manifest)
        self.writer = None
        self.close()
        return artifact

    def close(self):
        errors = []
        if self.writer is not None:
            self.writer.abort()
            self.writer = None
        try:
            if self.adapter is not None:
                if self.adapter.deactivate() not in {"stopped", "not_found"}:
                    errors.append("capture worker teardown failed")
                self.adapter = None
            if errors:
                for runtime in self.runtimes:
                    runtime.reuse_blocked = True
                self.predecessors.clear()
            for adapter, generation, initial, steps, physics_generation, scheduler, borrowed, completed, limit, input_history in self.predecessors:
                try:
                    if not adapter.activate(generation, initial_conditions=initial):
                        raise BakeCaptureError(adapter.last_error)
                    while adapter.controller.step_count < steps:
                        before = adapter.controller.step_count
                        if input_history:
                            from .physics_kinematic_inputs import apply_kinematic_inputs
                            apply_kinematic_inputs(adapter.controller, input_history[before+1], generation)
                        adapter.controller.tick(max_steps=before+1 if input_history else steps)
                        if adapter.controller.step_count <= before:
                            raise BakeCaptureError("prior playback replay failed")
                    adapter.controller._kinematic_input_history = input_history
                    adapter.physics_generation = physics_generation
                    adapter.controller.completed = completed
                    adapter.controller.max_steps_limit = limit
                    adapter.controller.sync_async_producer_position()
                    if borrowed:
                        scheduler._controller = adapter.controller
                        adapter.controller.set_pose_publication_wake_hook(
                            scheduler._wake_for_presentation if scheduler._wake_hook is not None else None)
                except Exception as exc:
                    for runtime in self.runtimes:
                        if runtime.ovphysx is adapter:
                            runtime.reuse_blocked = True
                    errors.append(str(exc))
            self.predecessors.clear()
        finally:
            for lock in reversed(self.held):
                lock.release()
            self.held.clear()
        if errors:
            raise BakeCaptureError("; ".join(errors))
