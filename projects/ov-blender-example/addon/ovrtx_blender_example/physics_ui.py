# SPDX-License-Identifier: Apache-2.0
"""Blender panels and undoable OV physics authoring operators."""
from . import physics_authoring as authoring
from .physics_endpoints import initialize_owner, initialize_joint, bind_bone_endpoint
from .physics_frames import RigidFrame

try:
    import bpy
except ModuleNotFoundError:
    bpy = None


def active_owner(context):
    return context.active_pose_bone or context.object


def require_joint_body(obj, bone=None):
    """Check the selected endpoint before creating a joint handle."""
    target = obj.pose.bones[bone.name] if bone else obj
    body = target.ov.physics.rigid_body
    if body.schema_opinion != "APPLY" or not body.enabled:
        name = f"bone '{bone.name}'" if bone else f"Object '{obj.name}'"
        raise authoring.AuthoringError(f"{name} has no enabled OV body; use Assign OV Body first")


if bpy is not None:
    class PhysicsOperator:
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            try:
                authoring.require_stopped(context.scene)
                self.run(context)
            except (ValueError, AttributeError, RuntimeError) as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}
            return {"FINISHED"}

    class OVPHYSICS_OT_add_body(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_add_body"
        bl_label = "Assign OV Body"
        bl_description = "Assign an OV physics body to the selected Object or active pose bone"

        def run(self, context):
            bone = context.active_pose_bone
            target = initialize_owner(context.object, bone.bone if bone else None)
            target.ov.physics.rigid_body.schema_opinion = "APPLY"
            physics = target.ov.physics
            if not bone and context.object.type == "MESH" and not physics.colliders:
                # A new mesh body collides with its own convex hull by default.
                import uuid
                collider = physics.colliders.add()
                collider.name = collider.persistent_id = uuid.uuid4().hex
                collider.schema_opinion = "APPLY"
                collider.shape = "CONVEX_HULL"
            authoring.mark_dirty(context.scene, target)

    class OVPHYSICS_OT_create_joint(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_create_joint"
        bl_label = "Create OV Joint at Cursor"
        bl_description = "Connect one assigned OV body to the world or two assigned bodies at the 3D cursor"

        def run(self, context):
            from mathutils import Matrix
            body = context.object
            if body is None:
                raise authoring.AuthoringError("Select an OV body first")
            bone = context.active_pose_bone
            others = [o for o in context.selected_objects if o != body]
            if len(others) > 1:
                raise authoring.AuthoringError("Select one body for world attachment or two bodies")
            require_joint_body(body, bone.bone if bone else None)
            if others:
                require_joint_body(others[0])
            # Creating a joint is an explicit authoring action. Give older
            # authored bodies missing IDs after validating both selections.
            initialize_owner(body, bone.bone if bone else None)
            if others:
                initialize_owner(others[0])
            handle = bpy.data.objects.new("OV Joint", None)
            context.scene.collection.objects.link(handle)
            try:
                handle.empty_display_type = "ARROWS"
                handle.empty_display_size = .25
                handle.matrix_world = Matrix.Translation(context.scene.cursor.location)
                joint = initialize_joint(handle)
                joint.endpoint_a.object = body
                if bone:
                    bind_bone_endpoint(joint.endpoint_a, body, bone.bone)
                if others:
                    joint.endpoint_b.object = others[0]
                else:
                    joint.endpoint_b.kind = "WORLD"
                context.view_layer.update()
                authoring.apply_anchor(handle, context.scene, context.evaluated_depsgraph_get())
                joint.enabled = True
            except Exception:
                bpy.data.objects.remove(handle, do_unlink=True)
                raise
            if context.mode == "POSE":
                bpy.ops.object.mode_set(mode="OBJECT")
            for obj in context.selected_objects:
                obj.select_set(False)
            handle.select_set(True)
            context.view_layer.objects.active = handle

    class OVPHYSICS_OT_anchor(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_anchor"
        bl_label = "Apply / Cancel Joint Anchor"
        bl_description = "Commit the joint handle to both frames, repairing referenced duplicate identities, or restore its committed anchor"
        operation: bpy.props.EnumProperty(items=(("APPLY", "Apply Anchor to Both Frames", ""), ("CANCEL", "Cancel Anchor Edit", "")))

        def run(self, context):
            context.view_layer.update()
            if self.operation == "APPLY":
                repaired = authoring.apply_anchor(
                    context.object, context.scene, context.evaluated_depsgraph_get(),
                    repair_duplicate_objects=True,
                )
                if repaired:
                    self.report({"INFO"}, "Reidentified duplicate physics owner: " + ", ".join(repaired))
            else:
                authoring.cancel_anchor(context.object, context.scene)

    class OVPHYSICS_OT_edit_frame(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_edit_frame"
        bl_label = "Edit Local Joint Frame"
        bl_description = "Edit one endpoint's local attachment frame"
        side: bpy.props.EnumProperty(items=(("A", "Frame A", ""), ("B", "Frame B", "")))
        translation: bpy.props.FloatVectorProperty(size=3, subtype="TRANSLATION")
        rotation: bpy.props.FloatVectorProperty(size=4, subtype="QUATERNION", default=(1, 0, 0, 0))

        def invoke(self, context, event):
            record = getattr(context.object.ov.physics.joint, "frame_" + self.side.lower())
            self.translation, self.rotation = record.translation, record.rotation
            return context.window_manager.invoke_props_dialog(self)

        def run(self, context):
            authoring.edit_frame(context.object, context.scene, self.side, RigidFrame(self.translation, self.rotation))

    class OVPHYSICS_OT_joint_type(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_joint_type"
        bl_label = "Change Joint Type / Axis"
        bl_description = "Choose the joint type and primary axis"
        joint_type: bpy.props.EnumProperty(items=tuple((k, k.title(), "") for k in ("FIXED", "REVOLUTE", "PRISMATIC", "D6", "SPHERICAL", "DISTANCE")))
        axis: bpy.props.EnumProperty(items=tuple((k, k, "") for k in "XYZ"))

        def invoke(self, context, event):
            joint = context.object.ov.physics.joint
            self.joint_type, self.axis = joint.type, joint.axis
            return context.window_manager.invoke_props_dialog(self)

        def run(self, context):
            errors = authoring.set_joint_type(context.object, context.scene, self.joint_type, self.axis)
            if errors:
                self.report({"WARNING"}, errors[0])

    class OVPHYSICS_OT_bind_bone(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_bind_bone"
        bl_label = "Bind Selected Bone Name"
        bl_description = "Bind this joint endpoint to the named bone on its armature Object"
        side: bpy.props.EnumProperty(items=(("A", "A", ""), ("B", "B", "")))

        def run(self, context):
            endpoint = getattr(context.object.ov.physics.joint, "endpoint_" + self.side.lower())
            owner = endpoint.object
            if owner is None or owner.type != "ARMATURE":
                raise authoring.AuthoringError("Choose an armature Object")
            bone = owner.data.bones.get(endpoint.bone_name)
            if bone is None:
                raise authoring.AuthoringError("Choose an existing bone")
            bind_bone_endpoint(endpoint, owner, bone)
            authoring.mark_dirty(context.scene, context.object)

    class OVPHYSICS_OT_scene_opinion(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_scene_opinion"
        bl_label = "Set Physics Scene Opinion"
        bl_description = "Apply OV physics scene settings or inherit them from the source"
        opinion: bpy.props.EnumProperty(items=(("INHERIT", "Use Source", ""), ("APPLY", "Apply", "")))

        def run(self, context):
            context.scene.ov.physics_scene.schema_opinion = self.opinion

    class OVPHYSICS_OT_collider_shape(PhysicsOperator, bpy.types.Operator):
        bl_idname = "ov.physics_collider_shape"
        bl_label = "Change Collider Shape"
        bl_description = "Change the selected OV collider's shape"
        collider_id: bpy.props.StringProperty()
        shape: bpy.props.EnumProperty(items=tuple((name, label, "") for name, label in (("CONVEX_HULL", "Convex Hull"), ("ANALYTIC_SPHERE", "Sphere"), ("BOX", "Box"), ("CAPSULE", "Capsule"))))

        def invoke(self, context, event):
            return context.window_manager.invoke_props_dialog(self)

        def run(self, context):
            owner = active_owner(context)
            record = next((c for c in owner.ov.physics.colliders if c.persistent_id == self.collider_id), None)
            if record is None:
                raise authoring.AuthoringError("Collider no longer exists")
            record.shape = self.shape
            authoring.mark_dirty(context.scene, owner)

    class PhysicsPanel:
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"

    class OVPHYSICS_PT_body(PhysicsPanel, bpy.types.Panel):
        bl_label = "OV Physics Body / Bone"
        bl_context = "physics"

        @classmethod
        def poll(cls, context):
            return context.object is not None and not context.object.ov.physics.joint.persistent_id

        def draw(self, context):
            layout = self.layout
            owner = active_owner(context)
            body = owner.ov.physics.rigid_body
            layout.label(text="Bone: " + owner.name if context.active_pose_bone else "Object: " + owner.name)
            layout.enabled = not authoring.simulation_owns_scene(context.scene)
            layout.operator("ov.physics_add_body")
            if not context.active_pose_bone and context.object.type == "MESH":
                if owner.ov.physics.simready_version == 0:
                    layout.operator("ov.physics_convert_simready")
                else:
                    layout.label(text="Converted SimReady: OV records are authoritative")
            layout.prop(body, "schema_opinion")
            layout.prop(body, "role")
            if body.role == "KINEMATIC":
                from .physics_capabilities import kinematic_capability
                layout.label(text=kinematic_capability().reason)
                layout.label(text="Animate owner pose; scale changes require restart")
                layout.label(text="Input constraints/drivers and dynamic parents unavailable")
            layout.prop(body, "enabled")
            layout.prop(body, "mass_mode")
            if body.mass_mode == "MASS":
                layout.prop(body, "mass_kg")
            elif body.mass_mode == "DENSITY":
                layout.prop(body, "density", text="Density (kg/BU³)")
            layout.prop(body, "center_of_mass", text="Center of Mass (BU)")
            layout.prop(body, "diagonal_inertia", text="Inertia (kg BU²; zero = automatic)")
            layout.prop(body, "principal_axes")
            layout.prop(body, "initial_linear_velocity", text="Initial Velocity (BU/s)")
            layout.prop(body, "initial_angular_velocity", text="Initial Angular Velocity (rad/s)")
            layout.prop(body, "linear_damping")
            layout.prop(body, "angular_damping")
            layout.label(text="Apply physics changes by explicit restart")
            from .physics_capabilities import body_control_capability
            layout.label(text=body_control_capability("mass").reason)
            layout.label(text="CCD, sleep and solver tuning unavailable")
            layout.prop(body.offset, "translation", text="Body Offset")
            layout.prop(body.offset, "rotation", text="Offset Rotation")
            layout.operator("ov.physics_create_joint")
            layout.operator("ov.physics_repair_identity")

    class OVPHYSICS_PT_colliders(PhysicsPanel, bpy.types.Panel):
        bl_label = "OV Colliders"
        bl_context = "physics"

        @classmethod
        def poll(cls, context):
            return context.object is not None and not context.object.ov.physics.joint.persistent_id

        def draw(self, context):
            layout = self.layout
            physics = active_owner(context).ov.physics
            layout.enabled = not authoring.simulation_owns_scene(context.scene)
            for record in physics.colliders:
                box = layout.box()
                box.label(text=record.name[:12])
                for field in ("enabled", "schema_opinion"):
                    box.prop(record, field)
                box.label(text="Shape: " + record.shape.replace("_", " ").title())
                box.operator("ov.physics_collider_shape").collider_id = record.persistent_id
                if record.shape == "SIMREADY_CONVEX_DECOMPOSITION":
                    box.label(text="Imported single-mesh cooking only")
                    box.label(text="Compound, role and offset edits unavailable")
                if record.shape == "BOX":
                    box.prop(record, "dimensions", text="Box Dimensions (BU)")
                else:
                    box.prop(record, "radius_m")
                if record.shape == "CAPSULE":
                    box.prop(record, "height_m", text="Cylinder Height (m), Z Axis")
                box.prop(record.offset, "translation", text="Local Offset")
                box.prop(record.offset, "rotation", text="Local Rotation")
                box.prop(record, "physics_material")
                box.prop(record, "mesh_source")
                box.prop(record, "hide_from_render", text="Hide Render Geometry")
                box.operator("ov.physics_edit_colliders", text="Remove Collider").remove_id = record.persistent_id
                row = box.row()
                up = row.operator("ov.physics_edit_colliders", text="Move Up")
                up.move_id, up.direction = record.persistent_id, -1
                down = row.operator("ov.physics_edit_colliders", text="Move Down")
                down.move_id, down.direction = record.persistent_id, 1
            layout.operator("ov.physics_edit_colliders", text="Add Collider")

    class OVPHYSICS_PT_joint(PhysicsPanel, bpy.types.Panel):
        bl_label = "OV Joint"
        bl_context = "physics"

        @classmethod
        def poll(cls, context):
            return context.object is not None and context.object.type == "EMPTY" and bool(context.object.ov.physics.joint.persistent_id)

        def draw(self, context):
            layout = self.layout
            obj, joint = context.object, context.object.ov.physics.joint
            layout.prop(joint, "show_indicator", text="Show Viewport Indicator")
            layout.label(text="Authoring settings")
            from .physics_capabilities import joint_capability
            layout.label(text=joint_capability(joint.type).reason)
            layout.label(text="Live motor streaming is unavailable")
            column = layout.column()
            column.enabled = not authoring.simulation_owns_scene(context.scene)
            column.prop(joint, "enabled")
            column.label(text=f"{joint.type}, axis {joint.axis}")
            column.operator("ov.physics_joint_type")
            for side in ("A", "B"):
                endpoint = getattr(joint, "endpoint_" + side.lower())
                box = column.box()
                box.label(text="Endpoint " + side)
                box.prop(endpoint, "kind")
                if endpoint.kind != "WORLD":
                    box.prop(endpoint, "object")
                if endpoint.kind == "BONE":
                    if endpoint.object and endpoint.object.type == "ARMATURE":
                        box.prop_search(endpoint, "bone_name", endpoint.object.data, "bones")
                    else:
                        box.prop(endpoint, "bone_name")
                    box.operator("ov.physics_bind_bone").side = side
            if authoring.anchor_pending(obj):
                column.label(text="Pending anchor: apply or cancel", icon="ERROR")
                column.label(text="Playback applies this anchor; baking needs Apply")
            row = column.row()
            row.operator("ov.physics_anchor", text="Apply Anchor to Both Frames").operation = "APPLY"
            row.operator("ov.physics_anchor", text="Cancel Anchor Edit").operation = "CANCEL"
            row = column.row()
            row.enabled = not authoring.anchor_pending(obj)
            row.operator("ov.physics_edit_frame", text="Edit Frame A").side = "A"
            row.operator("ov.physics_edit_frame", text="Edit Frame B").side = "B"
            column.label(text="Handle marks last common anchor")
            column.label(text="Separately edited frames may differ")
            for key in authoring.active_axis_keys(joint):
                axis = joint.axes.get(key)
                if axis is None:
                    continue
                box = column.box()
                box.label(text=key)
                box.prop(axis, "state")
                if axis.state == "LIMITED":
                    box.prop(axis, "lower")
                    box.prop(axis, "upper")
                box.prop(axis.drive, "enabled", text="Drive")
                if axis.drive.enabled:
                    box.prop(axis.drive, "mode")
                    if axis.drive.mode == "ACCELERATION":
                        box.label(text="Acceleration drive validation required", icon="ERROR")
                    angular = key.startswith("rot")
                    box.prop(axis.drive, "target_position", text="Target Angle (rad)" if angular else "Target Position (BU)")
                    box.prop(axis.drive, "target_velocity", text="Velocity (rad/s)" if angular else "Velocity (BU/s)")
                    for field in ("stiffness", "damping", "max_effort"):
                        box.prop(axis.drive, field)
            for error in authoring.inactive_animation(obj):
                column.label(text=error, icon="ERROR")
            column.prop(joint, "break_force")
            column.prop(joint, "break_torque")
            column.prop(joint, "collision_enabled")

    class OVPHYSICS_PT_material(PhysicsPanel, bpy.types.Panel):
        bl_label = "OV Physics Material"
        bl_context = "material"

        @classmethod
        def poll(cls, context):
            return context.material is not None

        def draw(self, context):
            for field in ("schema_opinion", "static_friction", "dynamic_friction", "restitution"):
                self.layout.prop(context.material.ov.physics, field)
            self.layout.prop(context.material.ov.physics, "density", text="Density (kg/BU³)")

    class OVPHYSICS_PT_scene(PhysicsPanel, bpy.types.Panel):
        bl_label = "OV Physics Scene"
        bl_context = "scene"

        def draw(self, context):
            layout = self.layout
            physics = context.scene.ov.physics_scene
            layout.label(text="Scene opinion: " + physics.schema_opinion)
            if physics.schema_opinion == "REMOVE":
                layout.label(text="Typed scene removal is unsupported", icon="ERROR")
            row = layout.row()
            row.operator("ov.physics_scene_opinion", text="Use Source").opinion = "INHERIT"
            row.operator("ov.physics_scene_opinion", text="Apply Scene").opinion = "APPLY"
            layout.prop(physics, "gravity_direction")
            layout.prop(physics, "gravity_magnitude")
            layout.prop(context.scene.ov.usd, "physics_scene_path")
            layout.prop(context.scene.ov, "sync_playback_to_ovrtx_frames")
            box = layout.box()
            box.prop(context.scene.ov, "indicators_enabled")
            box.prop(context.scene.ov, "indicators_selected_only")
            box.prop(context.scene.ov, "indicators_colliders")
            box.prop(context.scene.ov, "indicators_size")
            box.label(text="Rendered views use x-ray indicators")

    CLASSES = (OVPHYSICS_OT_add_body, OVPHYSICS_OT_create_joint, OVPHYSICS_OT_anchor,
               OVPHYSICS_OT_edit_frame, OVPHYSICS_OT_joint_type, OVPHYSICS_OT_bind_bone,
               OVPHYSICS_OT_scene_opinion, OVPHYSICS_OT_collider_shape, OVPHYSICS_PT_body, OVPHYSICS_PT_colliders,
               OVPHYSICS_PT_joint, OVPHYSICS_PT_material, OVPHYSICS_PT_scene)
else:
    CLASSES = ()


def register():
    for cls in CLASSES:
        if not getattr(cls, "is_registered", False):
            bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        if getattr(cls, "is_registered", False):
            bpy.utils.unregister_class(cls)
