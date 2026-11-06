# SPDX-License-Identifier: Apache-2.0
"""Collider collection records and edits; the collection is the only collider model."""
import uuid

from .physics_endpoints import require_writable


class ColliderError(ValueError):
    pass


def collider_records(owner):
    return tuple(getattr(owner.ov.physics, "colliders", ()))


def edit_colliders(owner, scene, *, remove_id=None, move_id=None, direction=1):
    """Add, remove or reorder colliders; stable names survive removal."""
    require_writable(owner)
    from .physics_authoring import require_stopped, repair_collection_animation, mark_dirty
    require_stopped(scene)
    selected_id = remove_id or move_id
    if remove_id and move_id:
        raise ColliderError("Choose one collider mutation")
    if selected_id is not None and not any(
        getattr(record, "persistent_id", "") == selected_id for record in collider_records(owner)
    ):
        raise ColliderError("Collider no longer exists")
    repair_collection_animation(owner, owner.ov.physics.colliders, remove_id)
    from .physics_endpoints import initialize_owner
    armature = getattr(owner, "id_data", owner)
    initialize_owner(armature, owner.bone) if hasattr(owner, "bone") else initialize_owner(owner)
    records = owner.ov.physics.colliders
    if move_id is not None:
        index = next(i for i, c in enumerate(records) if c.persistent_id == move_id)
        target = max(0, min(len(records)-1, index + (1 if direction > 0 else -1)))
        records.move(index, target)
        mark_dirty(scene, owner)
        return records[target]
    if remove_id is not None:
        index = next((i for i, c in enumerate(records) if c.persistent_id == remove_id), None)
        if index is None:
            raise ColliderError("Collider no longer exists")
        # Named F-curve paths do not retarget another collider on removal.
        records.remove(index)
        mark_dirty(scene, owner)
        return None
    record = records.add()
    record.name = record.persistent_id = uuid.uuid4().hex
    record.schema_opinion = "APPLY"
    mark_dirty(scene, owner)
    return record


try:
    import bpy
except ModuleNotFoundError:
    bpy = None

if bpy is not None:
    class OVPHYSICS_OT_edit_colliders(bpy.types.Operator):
        bl_idname = "ov.physics_edit_colliders"
        bl_label = "Edit Collider Collection"
        bl_description = "Remove or reorder a collider in the selected body's collection"
        bl_options = {"REGISTER", "UNDO"}
        remove_id: bpy.props.StringProperty(default="")
        move_id: bpy.props.StringProperty(default="")
        direction: bpy.props.IntProperty(default=1)

        def execute(self, context):
            try:
                edit_colliders(context.active_pose_bone or context.object, context.scene,
                               remove_id=self.remove_id or None, move_id=self.move_id or None,
                               direction=self.direction)
            except Exception as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            return {"FINISHED"}


def register():
    if bpy is not None:
        for cls in (OVPHYSICS_OT_edit_colliders,):
            if not getattr(cls, "is_registered", False):
                bpy.utils.register_class(cls)


def unregister():
    if bpy is not None:
        for cls in (OVPHYSICS_OT_edit_colliders,):
            if getattr(cls, "is_registered", False):
                bpy.utils.unregister_class(cls)
