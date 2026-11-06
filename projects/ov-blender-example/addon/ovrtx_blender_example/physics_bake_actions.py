# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private Action construction and deliberate reversible baked playback."""
import json
import uuid

from .physics_bake_capture import BakeCaptureError, _main_thread
from .physics_bake_conversion import _target


def _slot(action, identifier):
    return next((s for s in getattr(action, "slots", ()) if s.identifier == identifier), None)


def _curves(action, slot):
    if getattr(action, "is_action_legacy", False) and hasattr(action, "fcurves"):
        yield action.fcurves
    else:
        for layer in action.layers:
            for strip in layer.strips:
                bag = strip.channelbag(slot) if slot else None
                if bag:
                    yield bag.fcurves


def _paths(obj, channel, names):
    prefix = "" if channel is obj else channel.path_from_id() + "."
    return {name: prefix + name for name in names}


def _validate_owner(obj, outputs):
    if obj.library or obj.override_library or (obj.data and (obj.data.library or obj.data.override_library)):
        raise BakeCaptureError("bake destination must be local and editable")
    if any(not c.mute for _, channel, _ in outputs for c in channel.constraints):
        raise BakeCaptureError("destination constraints conflict; choose a clean output")
    paths = {path for _, channel, names in outputs for path in _paths(obj, channel, names).values()}
    animation = obj.animation_data
    if not animation:
        return
    if any(not c.mute and c.data_path in paths for c in animation.drivers):
        raise BakeCaptureError("destination drivers conflict; choose a clean output")
    if animation.action and (animation.action_blend_type != "REPLACE" or abs(animation.action_influence - 1) > 1e-6):
        raise BakeCaptureError("active Action blending conflicts; choose a clean output")
    solo = any(t.is_solo for t in animation.nla_tracks)
    def strips(items):
        for item in items:
            if item.mute:
                continue
            yield item
            yield from strips(getattr(item, "strips", ()))
    for track in animation.nla_tracks:
        if track.mute or (solo and not track.is_solo):
            continue
        for strip in strips(track.strips):
            if strip.mute or not strip.action:
                continue
            if any(c.data_path in paths and not c.mute
                   for curves in _curves(strip.action, getattr(strip, "action_slot", None)) for c in curves):
                raise BakeCaptureError("NLA affects baked channels; choose a clean output")


def _configuration(obj, outputs):
    """Persistable guard against edits between preparation and activation."""
    from .physics_bake_configuration import channel_configuration, TRANSFORMS, DELTAS
    rows = []
    parent = obj
    while parent:
        rows.append((parent.name_full, parent.parent.name_full if parent.parent else "",
                     parent.parent_type, parent.parent_bone, parent.rotation_mode,
                     tuple(tuple(row) for row in parent.matrix_parent_inverse),
                     channel_configuration(parent, parent, TRANSFORMS + DELTAS)))
        parent = parent.parent
    seen = set()
    for _, channel, _ in outputs:
        while channel is not obj and channel and channel.name not in seen:
            seen.add(channel.name)
            bone = channel.bone
            rows.append((channel.name, bone.ov.persistent_id, channel.rotation_mode,
                         tuple(tuple(row) for row in bone.matrix_local), bone.inherit_scale,
                         bone.use_local_location, bone.use_inherit_rotation, bone.use_relative_parent,
                         channel_configuration(obj, channel, TRANSFORMS)))
            channel = channel.parent
    return json.dumps(rows, sort_keys=True)


class ActionBuilder:
    """One-frame-per-step staging; cancellation removes only its private Actions."""
    def __init__(self, scene, converted, *, name="OVPhysX Bake"):
        _main_thread()
        import bpy
        from .physics_pose_presentation import resolve
        self.scene, self.converted, self.manifest = scene, converted, converted.manifest()
        from pathlib import Path
        from .physics_bake_capture import BakeCaptureArtifact
        from .physics_bake_conversion import validate_captured_configuration
        self.capture = BakeCaptureArtifact(Path(self.manifest["capture_directory"]), self.manifest["capture_digest"])
        validate_captured_configuration(scene, self.capture)
        self.created, self.records, self.done, self.count = [], [], False, 0
        self.identifier = uuid.uuid4().hex
        self.bindings = {}
        self.curves = {}
        # Validate the complete file before creating any Blender datablock.
        names = {}
        scale_required = set()
        from mathutils import Matrix
        target_rows = {t["body_path"]: t for t in self.manifest["targets"]}
        for row, captured in zip(converted.samples(), self.capture.samples(), strict=True):
            if row["frame"] != captured["frame"]:
                raise BakeCaptureError("converted and captured frames differ")
            for path, values in row["channels"].items():
                names.setdefault(path, set()).update(values)
                context = captured["context"][path]
                original_scale = Matrix(context["basis"]).to_scale()
                delta = context["delta_scale"] if not target_rows[path]["bone_identity"] else (1, 1, 1)
                if any(abs(original_scale[i] / delta[i] - values["scale"][i]) > 1e-6 for i in range(3)):
                    scale_required.add(path)
        for path in names:
            if path not in scale_required:
                names[path].discard("scale")
        if set(names) != set(self.manifest["output_paths"]):
            raise BakeCaptureError("converted output roster is incomplete")
        targets = {_target(t).body_path: _target(t) for t in self.manifest["targets"]}
        grouped = {}
        for path in self.manifest["output_paths"]:
            obj, channel = resolve(scene, targets[path])
            if channel.rotation_mode != self.manifest["rotation_modes"][path]:
                raise BakeCaptureError("destination rotation mode changed before Action commit")
            grouped.setdefault(obj, []).append((path, channel, names[path]))
        for obj, outputs in grouped.items():
            _validate_owner(obj, outputs)
        self.configuration = {obj: _configuration(obj, outputs) for obj, outputs in grouped.items()}
        try:
            for obj, outputs in grouped.items():
                animation = obj.animation_data
                original = animation.action if animation else None
                original_slot = getattr(animation, "action_slot", None)
                slot_identifier = original_slot.identifier if original_slot else ""
                action = original.copy() if original else bpy.data.actions.new(name + " • " + obj.name)
                self.created.append(action)
                action.name = name + " • " + obj.name
                action.use_fake_user = True
                if getattr(action, "is_action_legacy", False) and hasattr(action, "fcurves"):
                    slot, curves = None, action.fcurves
                else:
                    from bpy_extras.anim_utils import action_ensure_channelbag_for_slot
                    slot = _slot(action, slot_identifier) if original else None
                    if slot is None:
                        if original and slot_identifier:
                            raise BakeCaptureError("copied Action lost its installed-version slot")
                        slot = action.slots.new(id_type="OBJECT", name=obj.name)
                    curves = action_ensure_channelbag_for_slot(action, slot).fcurves
                replace_paths = {p for _, c, props in outputs for p in _paths(obj, c, props).values()}
                for collection in _curves(action, slot):
                    for curve in tuple(collection):
                        if curve.data_path in replace_paths:
                            collection.remove(curve)
                for path, channel, props in outputs:
                    self.bindings[path] = (obj, channel, _paths(obj, channel, props), curves)
                self.records.append((obj, action, slot.identifier if slot else "", original, slot_identifier, outputs))
            self.iterator = iter(converted.samples())
        except BaseException:
            self.cancel()
            raise

    def step(self):
        _main_thread()
        if self.done:
            return True
        try:
            row = next(self.iterator)
            for path, values in row["channels"].items():
                _, channel, paths, collection = self.bindings[path]
                for name, vector in values.items():
                    if name not in paths:
                        continue
                    for index, value in enumerate(vector):
                        key = path, name, index
                        curve = self.curves.get(key)
                        if curve is None:
                            curve = collection.new(data_path=paths[name], index=index)
                            self.curves[key] = curve
                        point = curve.keyframe_points.insert(row["frame"], value, options={"FAST"})
                        point.interpolation = "LINEAR"
            self.count += 1
            return False
        except StopIteration:
            try:
                self._finish()
                self.done = True
                return True
            except BaseException:
                self.cancel()
                raise
        except BaseException:
            self.cancel()
            raise

    def _finish(self):
        from .physics_bake_capture import BakeCaptureArtifact, _file_digest
        from pathlib import Path
        from .physics_bake_conversion import validate_captured_configuration
        validate_captured_configuration(self.scene, self.capture)
        for curve in self.curves.values():
            curve.update()
        capture_path = Path(self.manifest["capture_directory"])
        capture = BakeCaptureArtifact(capture_path, self.manifest["capture_digest"]).manifest()
        provenance = {"capture": self.manifest["capture_digest"], "conversion": self.converted.manifest_digest,
            "generation": capture.get("generation_digest"), "timing": capture["timing"],
            "runtime": capture.get("runtime"), "output_paths": self.manifest["output_paths"],
            "mapping": [{"body_path": t["body_path"], "body_identity": t["body_identity"],
                         "object_identity": t["object_identity"], "bone_identity": t["bone_identity"],
                         "bone_name": t["bone_name"], "body_offset": t["body_offset"]}
                        for t in self.manifest["targets"] if t["body_path"] in self.manifest["output_paths"]],
            "tolerances": [self.manifest["position_tolerance_m"], self.manifest["rotation_tolerance_rad"]]}
        settings = self.scene.ov_physics_bake
        initial_count = len(settings.records)
        try:
            for obj, action, slot, original, original_slot, outputs in self.records:
                animation = obj.animation_data
                if ((animation.action if animation else None) != original or
                    getattr(getattr(animation, "action_slot", None), "identifier", "") != original_slot):
                    raise BakeCaptureError("destination Action changed while building bake")
                _validate_owner(obj, outputs)
                if _configuration(obj, outputs) != self.configuration[obj]:
                    raise BakeCaptureError("destination configuration changed while building bake")
                action["ovphysx_bake_provenance"] = json.dumps(provenance, sort_keys=True)
                action["ovphysx_bake_outputs"] = json.dumps([
                    (path, "" if channel is obj else channel.path_from_id(), sorted(names),
                     "" if channel is obj else channel.bone.ov.persistent_id)
                    for path, channel, names in outputs])
                action["ovphysx_bake_configuration"] = _configuration(obj, outputs)
                record = settings.records.add()
                record.batch = self.identifier
                record.owner = obj
                record.action, record.slot = action, slot
                record.previous_action, record.previous_slot = original, original_slot
            settings.latest_batch = self.identifier
        except BaseException:
            while len(settings.records) > initial_count:
                settings.records.remove(len(settings.records) - 1)
            raise

    def cancel(self):
        import bpy
        if self.done:
            return
        iterator = getattr(self, "iterator", None)
        if iterator is not None:
            iterator.close()
        for action in self.created:
            if action.name in bpy.data.actions:
                bpy.data.actions.remove(action)
        self.created.clear()


def build_actions(scene, converted, *, name="OVPhysX Bake", cancelled=lambda: False):
    builder = ActionBuilder(scene, converted, name=name)
    while True:
        if cancelled():
            builder.cancel()
            raise BakeCaptureError("Action build cancelled")
        if builder.step():
            return builder.identifier


def _activation_records(scene, batch=None):
    _main_thread()
    settings = scene.ov_physics_bake
    if settings.active:
        raise BakeCaptureError("deactivate the current bake before activating another")
    records = [r for r in settings.records if r.batch == (batch or settings.latest_batch)]
    if not records:
        raise BakeCaptureError("no completed bake to activate")
    for record in records:
        if not record.owner or not record.action:
            raise BakeCaptureError("bake destination or Action was removed")
        animation = record.owner.animation_data
        if ((animation.action if animation else None) != record.previous_action or
            getattr(getattr(animation, "action_slot", None), "identifier", "") != record.previous_slot):
            raise BakeCaptureError("destination assignment changed; rebuild or restore it before activation")
        _validate_owner(record.owner, _record_outputs(record))
        if _configuration(record.owner, _record_outputs(record)) != record.action["ovphysx_bake_configuration"]:
            raise BakeCaptureError("destination configuration changed; rebuild before activation")
    return records


def _record_outputs(record):
    return [(path, _resolve_channel(record.owner, channel, identity), names)
            for path, channel, names, identity in json.loads(record.action["ovphysx_bake_outputs"])]


def _resolve_channel(obj, path, identity, *, missing_ok=False):
    if not path:
        return obj
    matches = [b for b in getattr(obj.pose, "bones", ()) if b.bone.ov.persistent_id == identity]
    if len(matches) == 1:
        return matches[0]
    if missing_ok:
        return None
    raise BakeCaptureError("bake bone destination is missing or ambiguous")


class Activation:
    """Modal drain-before-assignment transaction; never block Blender's thread."""
    def __init__(self, scene, batch=None):
        from . import scene_generation_sessions as sessions
        self.scene = scene
        self.records = _activation_records(scene, batch)
        self.done = False
        self.cancelled = False
        scene.ov_physics_bake.active = True
        scene.ov_physics_bake.active_batch = self.records[0].batch
        try:
            self.future = sessions.set_baked_playback(scene, True)
        except BaseException:
            scene.ov_physics_bake.active = False
            scene.ov_physics_bake.active_batch = ""
            raise

    def step(self):
        from . import scene_generation_sessions as sessions
        from .interactive_operator_state import suppress_interactive_edit_bridge
        if self.done:
            return True
        if not self.future.done():
            return False
        try:
            sessions.finish_baked_playback_pause(self.scene, self.future)
            with suppress_interactive_edit_bridge():
                for record in self.records:
                    animation = record.owner.animation_data_create()
                    if (animation.action != record.previous_action or
                        getattr(animation.action_slot, "identifier", "") != record.previous_slot):
                        raise BakeCaptureError("assignment changed during physics drain")
                    _validate_owner(record.owner, _record_outputs(record))
                    if _configuration(record.owner, _record_outputs(record)) != record.action["ovphysx_bake_configuration"]:
                        raise BakeCaptureError("destination configuration changed during physics drain")
                for record in self.records:
                    animation = record.owner.animation_data_create()
                    record.previous_values = json.dumps([
                        ("" if channel is record.owner else channel.path_from_id(),
                         "" if channel is record.owner else channel.bone.ov.persistent_id,
                         {name: tuple(getattr(channel, name)) for name in names})
                        for _, channel, names in _record_outputs(record)])
                    animation.action = record.action
                    if record.slot:
                        animation.action_slot = _slot(record.action, record.slot)
                self.scene.frame_set(self.scene.frame_current, subframe=self.scene.frame_subframe)
            self.done = True
            return True
        except BaseException:
            if any(r.owner.animation_data and r.owner.animation_data.action == r.action for r in self.records):
                deactivate(self.scene)
            else:
                self.scene.ov_physics_bake.active = False
                self.scene.ov_physics_bake.active_batch = ""
                sessions.set_baked_playback(self.scene, False)
            raise

    def cancel(self):
        if self.done or self.cancelled:
            return
        self.cancelled = True
        from . import scene_generation_sessions as sessions
        self.scene.ov_physics_bake.active = False
        self.scene.ov_physics_bake.active_batch = ""
        sessions.set_baked_playback(self.scene, False)


def activate(scene, batch=None):
    """Begin activation; caller must poll .step() while processing Blender events."""
    return Activation(scene, batch)


def deactivate(scene):
    _main_thread()
    from .interactive_operator_state import suppress_interactive_edit_bridge
    settings = scene.ov_physics_bake
    if not settings.active:
        return
    with suppress_interactive_edit_bridge():
        for record in settings.records:
            if record.batch != settings.active_batch or not record.owner:
                continue
            animation = record.owner.animation_data_create()
            animation.action = record.previous_action
            if record.previous_action and record.previous_slot:
                animation.action_slot = _slot(record.previous_action, record.previous_slot)
            for path, identity, values in json.loads(record.previous_values or "[]"):
                channel = _resolve_channel(record.owner, path, identity, missing_ok=True)
                if channel is None:
                    continue
                for name, value in values.items():
                    setattr(channel, name, value)
        settings.active = False
        settings.active_batch = ""
        scene.frame_set(scene.frame_current, subframe=scene.frame_subframe)
    from . import scene_generation_sessions as sessions
    sessions.set_baked_playback(scene, False)
