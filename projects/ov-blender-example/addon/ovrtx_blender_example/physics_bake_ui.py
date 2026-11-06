# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scene-owned bake workflow, explicit mapping and reversible Action activation."""
import tempfile
import shutil
from pathlib import Path
import bpy
from bpy.props import BoolProperty, CollectionProperty, EnumProperty, IntProperty, PointerProperty, StringProperty
from .physics_bake_capture import BakeCaptureError, BakeCaptureSession, BakeCaptureSettings

_jobs = set()  # UI resources only; physics ownership remains in existing sessions.


@bpy.app.handlers.persistent
def _cancel_jobs(*_args):
    for job in tuple(_jobs):
        try:
            job._cleanup(cancel=True)
        except Exception:
            import traceback
            traceback.print_exc()
            _jobs.discard(job)


class OVPhysicsBakeMapping(bpy.types.PropertyGroup):
    source: PointerProperty(type=bpy.types.Object)
    source_bone: StringProperty(name="Source Bone")
    destination: PointerProperty(type=bpy.types.Object)
    destination_bone: StringProperty(name="Destination Bone")


class OVPhysicsBakeRecord(bpy.types.PropertyGroup):
    batch: StringProperty()
    owner: PointerProperty(type=bpy.types.Object)
    action: PointerProperty(type=bpy.types.Action)
    slot: StringProperty()
    previous_action: PointerProperty(type=bpy.types.Action)
    previous_slot: StringProperty()
    previous_values: StringProperty()


class OVPhysicsBakeSettings(bpy.types.PropertyGroup):
    initial_frame: IntProperty(name="Initial Conditions", default=1)
    start_frame: IntProperty(name="Bake Start", default=1)
    end_frame: IntProperty(name="Bake End", default=250)
    interval: IntProperty(name="Sample Interval", default=1, min=1)
    substeps: IntProperty(name="Simulation Substeps", default=1, min=1)
    scope: EnumProperty(name="Outputs", items=(("ALL", "All Targets", "Simulate all and bake all outputs"),
        ("SELECTED", "Selected Targets", "Simulate all; key selected Objects or bones only")))
    destination: EnumProperty(name="Destination", items=(("SOURCE", "Original Owners", "New private Actions on original owners"),
        ("MAPPED", "Explicit Clean Outputs", "Use the destination mappings below; original rig is unchanged")))
    action_name: StringProperty(name="Action Name", default="OVPhysX Bake")
    mappings: CollectionProperty(type=OVPhysicsBakeMapping)
    records: CollectionProperty(type=OVPhysicsBakeRecord)
    latest_batch: StringProperty()
    active_batch: StringProperty()
    active: BoolProperty(default=False)
    message: StringProperty(default="Bake creates private Actions. Activate deliberately when ready.")


class OVPHYSICS_OT_bake_mapping_add(bpy.types.Operator):
    bl_idname = "ov.physics_bake_mapping_add"
    bl_label = "Add Output Mapping"
    bl_description = "Map a simulated source to an explicit clean output for baking"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        row = context.scene.ov_physics_bake.mappings.add()
        row.source = context.object
        row.source_bone = context.active_pose_bone.name if context.active_pose_bone else ""
        return {"FINISHED"}


class OVPHYSICS_OT_bake_mapping_remove(bpy.types.Operator):
    bl_idname = "ov.physics_bake_mapping_remove"
    bl_label = "Remove Mapping"
    bl_description = "Remove this bake output mapping"
    bl_options = {"REGISTER", "UNDO"}
    index: IntProperty()
    def execute(self, context):
        context.scene.ov_physics_bake.mappings.remove(self.index)
        return {"FINISHED"}


class OVPHYSICS_OT_bake_animation(bpy.types.Operator):
    bl_idname = "ov.physics_bake_animation"
    bl_label = "Bake OVPhysX Animation"
    bl_description = "Simulate the complete physics scene and create private editable Actions"
    bl_options = {"REGISTER", "UNDO"}
    _session = None
    _builder = None
    _timer = None
    _mapped_capture = None

    def invoke(self, context, event):
        from .physics_pose_presentation import resolve
        self._session = self._builder = self._timer = self._mapped_capture = None
        self._scene, self._wm = context.scene, context.window_manager
        self._window = context.window
        settings = self._scene.ov_physics_bake
        try:
            if settings.active:
                raise BakeCaptureError("deactivate baked playback before baking again")
            for name, value in (("initial_frame", context.scene.frame_start),
                                ("start_frame", context.scene.frame_start), ("end_frame", context.scene.frame_end)):
                if not settings.is_property_set(name):
                    setattr(settings, name, value)
            self._directory = Path(tempfile.mkdtemp(prefix="ovphysx-bake-"))
            self._session = BakeCaptureSession(context.scene,
                BakeCaptureSettings(settings.initial_frame, settings.start_frame, settings.end_frame,
                    context.scene.render.fps, context.scene.render.fps_base, settings.interval, settings.substeps),
                self._directory / "capture")
            if settings.scope == "SELECTED":
                paths = []
                for target in self._session.targets:
                    obj, channel = resolve(context.scene, target)
                    if (channel is obj and obj.select_get()) or (channel is not obj and channel.bone.select):
                        paths.append(target.body_path)
                if not paths:
                    raise BakeCaptureError("no selected dynamic outputs")
                self._session.output_paths = tuple(paths)
                self._session.manifest["output_paths"] = tuple(paths)
            self._timer = context.window_manager.event_timer_add(.02, window=context.window)
            context.window_manager.modal_handler_add(self)
            _jobs.add(self)
            settings.message = "Capturing complete physics scene; Escape cancels."
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self._close(context, cancel=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def modal(self, context, event):
        if self not in _jobs:
            return {"CANCELLED"}
        if context.scene != self._scene or context.window != self._window:
            self._cleanup(cancel=True)
            return {"CANCELLED"}
        if event.type == "ESC":
            self._close(context, cancel=True)
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        settings = context.scene.ov_physics_bake
        try:
            if self._builder:
                if not self._builder.step():
                    settings.message = f"Writing sample {self._builder.count}; Escape cancels."
                    return {"RUNNING_MODAL"}
                settings.message = "Bake complete. Activate Baked Playback when ready."
                self._close(context)
                return {"FINISHED"}
            if not self._session.poll():
                settings.message = f"Capturing physics: {self._session.progress:.0%}; Escape cancels."
                return {"RUNNING_MODAL"}
            from .physics_bake_conversion import convert_capture
            from .physics_bake_actions import ActionBuilder
            capture = self._session.artifact
            if settings.destination == "MAPPED":
                from .physics_bake_destinations import remap_capture
                capture = remap_capture(context.scene, capture, settings.mappings, self._directory / "mapped")
                self._mapped_capture = capture
            converted = convert_capture(context.scene, capture, self._directory / "converted")
            self._builder = ActionBuilder(context.scene, converted, name=settings.action_name)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self._close(context, cancel=True)
            settings.message = str(exc)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

    def _close(self, context, cancel=False):
        self._cleanup(cancel=cancel)

    def _cleanup(self, cancel=False):
        try:
            if self._session:
                if cancel:
                    self._session.cancel()
                self._session.close()
        finally:
            try:
                if cancel and self._builder and not self._builder.done:
                    self._builder.cancel()
                if cancel and self._mapped_capture and not (self._builder and self._builder.done):
                    from .physics_bake_destinations import rollback_mapping_identities
                    rollback_mapping_identities(self._scene, self._mapped_capture)
                if cancel and hasattr(self, "_directory") and self._directory.exists():
                    shutil.rmtree(self._directory)
            finally:
                if self._timer:
                    self._wm.event_timer_remove(self._timer)
                    self._timer = None
                _jobs.discard(self)

    def cancel(self, context):
        self._close(context, cancel=True)


class OVPHYSICS_OT_bake_activate(bpy.types.Operator):
    bl_idname = "ov.physics_bake_activate"
    bl_label = "Activate Baked Playback"
    bl_description = "Assign the latest baked Actions to their output Objects or bones"
    bl_options = {"REGISTER", "UNDO"}
    def invoke(self, context, event):
        from .physics_bake_actions import Activation
        self._timer = None
        self._activation = None
        self._scene, self._wm, self._window = context.scene, context.window_manager, context.window
        try:
            self._activation = Activation(context.scene)
            self._timer = context.window_manager.event_timer_add(.02, window=context.window)
            context.window_manager.modal_handler_add(self)
            _jobs.add(self)
            return {"RUNNING_MODAL"}
        except Exception as exc:
            self._cleanup(cancel=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
    def modal(self, context, event):
        if self not in _jobs:
            return {"CANCELLED"}
        if event.type == "ESC" or context.scene != self._scene or context.window != self._window:
            self._cleanup(cancel=True)
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            if not self._activation.step():
                return {"RUNNING_MODAL"}
            self._cleanup()
            return {"FINISHED"}
        except Exception as exc:
            self._cleanup(cancel=True)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
    def _cleanup(self, cancel=False):
        if cancel and self._activation:
            self._activation.cancel()
        if self._timer:
            self._wm.event_timer_remove(self._timer)
            self._timer = None
        _jobs.discard(self)
    def cancel(self, context):
        self._cleanup(cancel=True)


class OVPHYSICS_OT_bake_deactivate(bpy.types.Operator):
    bl_idname = "ov.physics_bake_deactivate"
    bl_label = "Restore Previous Animation"
    bl_description = "Restore the Actions and values that were active before baked playback"
    bl_options = {"REGISTER", "UNDO"}
    def execute(self, context):
        from .physics_bake_actions import deactivate
        try:
            deactivate(context.scene)
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class OVPHYSICS_PT_bake(bpy.types.Panel):
    bl_label = "OVPhysX Animation Bake"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"
    def draw(self, context):
        layout, settings = self.layout, context.scene.ov_physics_bake
        for name in ("initial_frame", "start_frame", "end_frame", "interval", "substeps", "scope", "destination", "action_name"):
            layout.prop(settings, name)
        if settings.destination == "MAPPED":
            for index, mapping in enumerate(settings.mappings):
                box = layout.box()
                box.prop(mapping, "source")
                box.prop(mapping, "source_bone")
                box.prop(mapping, "destination")
                box.prop(mapping, "destination_bone")
                box.operator("ov.physics_bake_mapping_remove").index = index
            layout.operator("ov.physics_bake_mapping_add")
        layout.label(text="Selection limits output; all bodies still simulate.")
        layout.operator("ov.physics_bake_animation")
        layout.operator("ov.physics_bake_deactivate" if settings.active else "ov.physics_bake_activate")
        layout.label(text=settings.message)


CLASSES = (OVPhysicsBakeMapping, OVPhysicsBakeRecord, OVPhysicsBakeSettings,
           OVPHYSICS_OT_bake_mapping_add, OVPHYSICS_OT_bake_mapping_remove,
           OVPHYSICS_OT_bake_animation, OVPHYSICS_OT_bake_activate, OVPHYSICS_OT_bake_deactivate, OVPHYSICS_PT_bake)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ov_physics_bake = PointerProperty(type=OVPhysicsBakeSettings)
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.save_pre):
        if _cancel_jobs not in handlers:
            handlers.append(_cancel_jobs)


def unregister():
    # Completed Actions deliberately survive add-on disable and reload.
    _cancel_jobs()
    for handlers in (bpy.app.handlers.load_pre, bpy.app.handlers.undo_pre, bpy.app.handlers.save_pre):
        if _cancel_jobs in handlers:
            handlers.remove(_cancel_jobs)
    del bpy.types.Scene.ov_physics_bake
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
