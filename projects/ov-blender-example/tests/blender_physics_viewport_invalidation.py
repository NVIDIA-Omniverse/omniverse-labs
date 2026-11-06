# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-Blender invalidation matrix for the physics-free viewport reuse path.

Every case starts from an accepted generation of a scene without physics
participants, proves that ``generation_for_viewport`` is on its fast reuse path
(no capture), makes one change through a production entry point, and then
drives the production observers in UI order: depsgraph evaluation (the add-on's
registered ``depsgraph_update_post`` handler), the next viewport lookup, and,
when that lookup only schedules work, the main-thread timer it scheduled.
A case passes when the retained physics snapshot then equals a fresh
``physics_edit_snapshot.capture(scene)``, a restart-worthy change queued
restart intent, and the viewport admission gate no longer trusts a stale
physics-free observation. Cases that miss record how many 0.2 s
``_poll_physics_edits`` ticks discovery needed.

Headless Blender runs no timers, so the scheduled timer bodies are emulated
with the same calls they make (``generation_for_scene`` under suppression, and
``_poll_physics_edits`` called directly). Results go to ``results.json``; the
pytest wrapper asserts each case and marks known gaps.

``--perf N`` instead measures R2 discovery latency and R6 per-callback cost on
an N-object scene.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys
import threading
import time
import traceback
from types import SimpleNamespace
import uuid

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
import ovrtx_blender_example as addon  # noqa: E402

ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = Path(ARGS[0]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
WORK = OUT / "generations"
PERF = int(ARGS[ARGS.index("--perf") + 1]) if "--perf" in ARGS else 0

addon.register()
from ovrtx_blender_example import scene_generation_sessions as sessions  # noqa: E402
from ovrtx_blender_example import physics_edit_snapshot as snapshots  # noqa: E402
from ovrtx_blender_example import physics_authoring  # noqa: E402
from ovrtx_blender_example import physics_live_kinematics  # noqa: E402
from ovrtx_blender_example import interactive_operator_state as suppression  # noqa: E402
from ovrtx_blender_example import engine  # noqa: E402
from ovrtx_blender_example.physics_endpoints import initialize_joint  # noqa: E402
from ovrtx_blender_example.scene_generation import SceneGenerationError  # noqa: E402

# Counters on the observation entry points; ``fresh`` is the uncounted original.
COUNT = {"observe": 0, "capture": 0, "require_scene_ready": 0}
fresh = snapshots.capture
_observe = sessions.observe_physics_edits
_ready = physics_authoring.require_scene_ready


def _counted_observe(*a, **k):
    COUNT["observe"] += 1
    return _observe(*a, **k)


def _counted_capture(*a, **k):
    COUNT["capture"] += 1
    return fresh(*a, **k)


def _counted_ready(*a, **k):
    COUNT["require_scene_ready"] += 1
    return _ready(*a, **k)


sessions.observe_physics_edits = _counted_observe
snapshots.capture = _counted_capture
physics_authoring.require_scene_ready = _counted_ready


def reset_counts():
    for key in COUNT:
        COUNT[key] = 0


# Spy appended after the add-on's handler: records what Blender dispatched.
DISPATCHED: list[list[tuple]] = []


@bpy.app.handlers.persistent
def _spy(_scene, depsgraph):
    DISPATCHED.append([
        (str(u.id.original.bl_rna.identifier), u.id.original.name, bool(u.is_updated_geometry),
         bool(u.is_updated_transform), bool(u.is_updated_shading)) for u in depsgraph.updates])


bpy.app.handlers.depsgraph_update_post.append(_spy)
EMPTY_DISPATCHES = [0]


def S():
    return bpy.context.scene


def uid(scene=None):
    return int((scene or S()).session_uid)


def expected_observation(snapshot):
    return (snapshots.PhysicsSnapshot(tuple(r for r in snapshot.leaves if r[0] == "SCENE")),
            any(r[0] != "SCENE" for r in snapshot.leaves))


def consistent(scene):
    """Retained snapshot and view observation equal a fresh full capture."""
    current = fresh(scene)
    u = uid(scene)
    return (sessions._physics_snapshots.get(u) == current
            and sessions._physics_view_observations.get(u) == expected_observation(current)), current


def accept_pending(scene):
    owner = sessions._owners.get(uid(scene))
    if owner is not None and owner.pending_generation is not None:
        owner.accept(owner.pending_generation)
        sessions._pending_affected.pop(uid(scene), None)


def pump(scene):
    """Run the main-thread timers the viewport lookup scheduled (headless: none run)."""
    ran = []
    u = uid(scene)
    wait_closed()
    for _ in range(4):
        accept_pending(scene)  # runtime handoff acceptance of a pending candidate
        owner = sessions._owners.get(u)
        if owner is None or owner.current_generation is None:
            # _schedule_viewport_generation.construct
            sessions._viewport_generation_timers.discard(u)
            sessions._viewport_generation_errors.pop(u, None)
            sessions.repair_duplicate_identities(scene)
            sessions.generation_for_scene(scene, work_root=WORK)
            ran.append("viewport-construct-timer")
        elif u in sessions._dirty:
            # _schedule_world_reconciliation.reconcile
            sessions._reconciliation_timers.discard(u)
            with suppression.suppress_interactive_edit_bridge():
                try:
                    sessions.generation_for_scene(scene, work_root=WORK)
                finally:
                    for layer in scene.view_layers:
                        layer.update()
            accept_pending(scene)
            if u in sessions._dirty and u in sessions._physics_restart_pending:
                ran.append("reconcile-timer (held by physics restart barrier)")
                break
            ran.append("reconcile-timer")
        else:
            break
    accept_pending(scene)
    return ran


def lookup(scene):
    """generation_for_viewport, as view_update/view_draw's provider calls it."""
    reset_counts()
    try:
        sessions.generation_for_viewport(scene)
        error = None
    except SceneGenerationError as exc:
        error = str(exc)
    return {"captures": COUNT["capture"], "observes": COUNT["observe"], "error": error}


def reuse_active(scene):
    u = uid(scene)
    result = lookup(scene)
    clear = sessions.viewport_joint_admission_clear(scene)
    return (result["captures"] == 0 and result["error"] is None and clear
            and u not in sessions._dirty and u not in sessions._physics_authoring_dirty
            and sessions._physics_view_observations.get(u, (None, True))[1] is False), dict(
                result, admission_clear=clear, dirty=sorted(str(i) for i in sessions._dirty.get(u, ())),
                latched=u in sessions._physics_authoring_dirty,
                observation=None if u not in sessions._physics_view_observations
                else sessions._physics_view_observations[u][1],
                pending=getattr(sessions._owners.get(u), "pending_generation", None) is not None,
                blocked=str(sessions._blocked_reconciliations.get(u)))


def poll_until_consistent(scene, max_ticks):
    for tick in range(1, max_ticks + 1):
        sessions._poll_physics_edits()
        if consistent(scene)[0]:
            return tick
    return None


def participants(snapshot):
    return any(r[0] != "SCENE" for r in snapshot.leaves)


def strip_physics():
    """Remove every stored ``ov`` opinion (raw, no notification) except fixtures."""
    for obj in bpy.data.objects:
        if obj.library is not None:
            continue
        if obj.animation_data is not None:
            obj.animation_data_clear()
        if obj.override_library is None:
            obj.property_unset("ov")
        if obj.pose is not None:
            for bone in obj.pose.bones:
                bone.property_unset("ov")
        for slot in obj.material_slots:
            slot.link = "DATA"
    for armature in bpy.data.armatures:
        if armature.library is None:
            for bone in armature.bones:
                bone.property_unset("ov")
    for material in bpy.data.materials:
        if material.library is None and material.name != "PhysMat":
            material.property_unset("ov")
    for scene in bpy.data.scenes:
        scene.property_unset("ov")
        if scene.unit_settings.scale_length != 1:
            scene.unit_settings.scale_length = 1


def settle(scene=None):
    """Accept the current authoring as the physics-free baseline and prove reuse."""
    scene = scene or S()
    for layer in scene.view_layers:
        layer.update()
    for _ in range(2):
        # Accept physics intent (the explicit-restart barrier) so world
        # reconciliation proceeds, as a completed restart would.
        sessions.observe_physics_edits(scene)
        sessions._physics_restart_pending.pop(uid(scene), None)
        pump(scene)
    sessions.observe_physics_edits(scene)
    sessions._physics_restart_pending.pop(uid(scene), None)
    sessions._viewport_generation_errors.pop(uid(scene), None)
    sessions._poll_physics_edits()  # creates RenderOnlyDiscovery as the 0.2 s timer does
    accept_pending(scene)
    return reuse_active(scene)


def mesh_object(name, material=None, location=(0, 0, 0)):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], [],
                     [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)])
    if material is not None:
        mesh.materials.append(material)
    obj = bpy.data.objects.new(name, mesh)
    obj.location = location
    return obj


def obj(name):
    return bpy.data.objects[name]


# --------------------------------------------------------------------------- case runner
RESULTS: list[dict] = []


def record(cid, ok, **observed):
    RESULTS.append({"id": cid, "ok": bool(ok), **observed})
    print(f"[invalidation] {cid}: {'ok' if ok else 'MISS'} {json.dumps(observed, default=str)[:400]}", flush=True)


def run_case(cid, body, *, base=True):
    try:
        if base:
            strip_physics()
            ready, info = settle()
            if not ready:
                record(cid, False, error="precondition: reuse path not active", precondition=info)
                return
        body(cid)
    except Exception as exc:  # noqa: BLE001 - a failing case must not stop the matrix
        record(cid, False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-1500:])


def discovery_bound_ticks(*scenes):
    """Hard worst case of idle discovery: one pass at MIN_BATCH Objects per tick."""
    batch = snapshots.RenderOnlyDiscovery.MIN_BATCH
    return max(max(1, -(-len(sc.objects) // batch)) for sc in scenes)


def standard(cid, change, *, scene=None, depsgraph=True, max_ticks=8, extra=None, bounded=False):
    """Change -> depsgraph handler -> next viewport lookup -> scheduled timer.

    ``bounded`` rows have no Blender hook at all (raw Python collection edits):
    they pass when the 0.2 s discovery timer (``_poll_physics_edits``, called
    directly because headless Blender runs no timers) detects the change within
    its worst-case bound, after which every pass rule must hold.
    """
    context_scene = scene is None
    scene = scene or S()
    u = uid(scene)
    before = sessions._physics_snapshots[u]
    before_close = sessions._CLOSE_TASK
    DISPATCHED.clear()
    change()
    if context_scene:
        scene = S()  # undo/redo and file reads replace the Scene wrapper
        u = uid(scene)
    latched = u in sessions._physics_authoring_dirty
    during_change = list(DISPATCHED)
    session_closed = sessions._CLOSE_TASK is not before_close
    DISPATCHED.clear()
    if depsgraph:
        bpy.context.view_layer.update()
    dispatched = list(DISPATCHED)
    after_handler = consistent(scene)[0]
    view = lookup(scene)
    after_lookup = consistent(scene)[0]
    timers = pump(scene) if (u in sessions._dirty or view["error"]) else []
    redraw = lookup(scene) if timers else None  # the redraw that presents the timer's result
    ok, current = consistent(scene)
    change_class = snapshots.classify(before, current)
    restart_queued = u in sessions._physics_restart_pending
    clear = sessions.viewport_joint_admission_clear(scene)
    reasons = []
    if not ok:
        reasons.append("retained snapshot differs from fresh capture")
    rebuilt = "viewport-construct-timer" in timers  # a new initial generation needs no restart
    if change_class.restart and not restart_queued and not rebuilt:
        reasons.append("restart-worthy change queued no restart intent")
    if clear and participants(current):
        reasons.append("admission gate still trusts physics-free observation")
    ticks = None
    bound = None
    immediate_reasons = list(reasons)
    if reasons and bounded:
        bound = discovery_bound_ticks(*{S(), scene})
        start = time.perf_counter()
        ticks = poll_until_consistent(scene, bound)
        detection_cpu_s = time.perf_counter() - start
        ok, current = consistent(scene)
        change_class = snapshots.classify(before, current)
        restart_queued = u in sessions._physics_restart_pending
        clear = sessions.viewport_joint_admission_clear(scene)
        reasons = []
        if ticks is None or not ok:
            reasons.append(f"idle discovery missed the change within {bound} tick(s)")
        if change_class.restart and not restart_queued:
            reasons.append("restart-worthy change queued no restart intent")
        if clear and participants(current):
            reasons.append("admission gate still trusts physics-free observation")
        extra = dict(extra or {}, discovery_bound_ticks=bound, immediate_reasons=immediate_reasons,
                     latency_bound_s=bound * .2 + detection_cpu_s)
    elif reasons:
        ticks = poll_until_consistent(scene, max_ticks)
    caught_by = ("session close + rebuild" if session_closed and ok
                 else f"idle discovery within {ticks} tick(s) (bound {bound})" if bound and not reasons
                 else "depsgraph handler" if after_handler and before != current
                 else "latch + viewport lookup" if latched and after_lookup and view["captures"]
                 else "viewport lookup scene-control/participant check" if after_lookup and view["captures"]
                 else "scheduled timer" if timers and ok
                 else "no change visible" if before == current
                 else "missed")
    record(cid, not reasons, caught_by=caught_by, latched_by_change=latched,
           session_closed_by_change=session_closed,
           handler_dispatches_during_change=during_change, handler_dispatches=dispatched, lookup=view, timers=timers,
           redraw_lookup=redraw,
           physics_visible_change=before != current, participants_after=participants(current),
           restart=change_class.restart, structural=change_class.structural, redraw=change_class.redraw,
           restart_queued=restart_queued, admission_clear=clear, reasons=reasons,
           poll_ticks_to_detection=ticks, **(extra or {}))


# --------------------------------------------------------------------------- scene setup
def write_libraries():
    """Library files written in-process; nothing here is part of the main file."""
    paths = {}
    for version, authored in (("v1", False), ("v2", True)):
        lib_obj = mesh_object("LibPlain")
        if authored:
            lib_obj.ov.physics.rigid_body.mass_kg = 6
        path = OUT / f"lib_{version}.blend"
        bpy.data.libraries.write(str(path), {lib_obj}, fake_user=True)
        mesh = lib_obj.data
        bpy.data.objects.remove(lib_obj)
        bpy.data.meshes.remove(mesh)
        paths[version] = path
    return paths


def build_base():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = S()
    mat_a = bpy.data.materials.new("MatA")
    bpy.data.materials.new("MatUnused").use_fake_user = True
    phys = bpy.data.materials.new("PhysMat")
    phys.use_fake_user = True
    phys.ov.physics.static_friction = .9
    for name, x in (("Plain", 0), ("Plain2", 3)):
        scene.collection.objects.link(mesh_object(name, mat_a, (x, 0, 0)))
    scene.collection.objects.link(bpy.data.objects.new("Handle", None))
    bpy.ops.object.armature_add(location=(0, 4, 0))
    bpy.context.object.name = "Rig"
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.location = (0, -10, 3)
    scene.collection.objects.link(camera)
    scene.camera = camera
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.update()
    sessions.generation_for_scene(scene, work_root=WORK)
    pump(scene)
    return scene


# --------------------------------------------------------------------------- matrix
def matrix():
    libs = write_libraries()
    build_base()

    # E: scalar / vector / pointer edits (RNA update callbacks) and operators.
    edits = {
        "E01-object-scalar-setattr": lambda: setattr(obj("Plain").ov.physics.rigid_body, "mass_kg", 3),
        "E02-object-vector-setattr": lambda: setattr(obj("Plain").ov.physics.rigid_body, "center_of_mass", (0, 0, .1)),
        "E03-object-vector-element": lambda: obj("Plain").ov.physics.rigid_body.center_of_mass.__setitem__(1, .2),
        "E04-object-pointer-setattr": lambda: setattr(obj("Handle").ov.physics.joint.endpoint_a, "object", obj("Plain")),
        "E05-object-enum-setattr": lambda: setattr(obj("Plain").ov.physics.rigid_body, "schema_opinion", "APPLY"),
        "E06-material-scalar-setattr": lambda: setattr(bpy.data.materials["MatA"].ov.physics, "static_friction", .7),
        "E07-unused-material-scalar": lambda: setattr(bpy.data.materials["MatUnused"].ov.physics, "restitution", .3),
        "E08-posebone-scalar-setattr": lambda: setattr(obj("Rig").pose.bones[0].ov.physics.rigid_body, "mass_kg", 4),
        "E09-posebone-vector-setattr": lambda: setattr(obj("Rig").pose.bones[0].ov.physics.rigid_body, "initial_linear_velocity", (1, 0, 0)),
        "E10-posebone-pointer-setattr": lambda: setattr(obj("Rig").pose.bones[0].ov.physics.joint.endpoint_a, "object", obj("Plain")),
        "E11-bone-identity-setattr": lambda: setattr(obj("Rig").data.bones[0].ov, "persistent_id", "bone-id"),
        "E12-scene-control-scalar": lambda: setattr(S().ov.physics_scene, "gravity_magnitude", 3),
        "E13-scene-control-vector": lambda: setattr(S().ov.physics_scene, "gravity_direction", (0, 0, 1)),
        "E14-scene-indicator-redraw": lambda: setattr(S().ov, "indicators_size", .5),
        "E15-scene-namespace-string": lambda: setattr(S().ov, "physics_namespace", "/World/Other"),
        "E16-unit-scale": lambda: setattr(S().unit_settings, "scale_length", .01),
    }
    for cid, change in edits.items():
        run_case(cid, lambda cid, change=change: standard(cid, change))
    # The same edits with no depsgraph evaluation in between: the RNA update
    # latch alone must make the next viewport lookup observe.
    for cid, source in (("E20-latch-only-object-setattr", "E01-object-scalar-setattr"),
                        ("E21-latch-only-material-setattr", "E06-material-scalar-setattr"),
                        ("E22-latch-only-posebone-setattr", "E08-posebone-scalar-setattr")):
        run_case(cid, lambda cid, change=edits[source]: standard(cid, change, depsgraph=False))

    def op_context_set(cid):
        def change():
            with bpy.context.temp_override(object=obj("Plain"), active_object=obj("Plain")):
                result = bpy.ops.wm.context_set_float(data_path="object.ov.physics.rigid_body.mass_kg", value=2.5)
            assert result == {"FINISHED"}, result
        standard(cid, change)
    run_case("E17-operator-wm-context-set", op_context_set)

    def op_add_body(cid):
        def change():
            plain = obj("Plain")
            with bpy.context.temp_override(object=plain, active_object=plain, selected_objects=[plain]):
                result = bpy.ops.ov.physics_add_body()
            assert result == {"FINISHED"}, result
        standard(cid, change)
    run_case("E18-operator-add-body", op_add_body)

    def op_scene_opinion(cid):
        def change():
            assert bpy.ops.ov.physics_scene_opinion(opinion="APPLY") == {"FINISHED"}
        standard(cid, change)
    run_case("E19-operator-scene-opinion", op_scene_opinion)

    # R: reset / unset.
    def unset_participant(cid):
        obj("Plain").ov.physics.rigid_body.mass_kg = 3
        settle_participant()
        standard(cid, lambda: obj("Plain").ov.physics.rigid_body.property_unset("mass_kg"),
                 extra={"start": "participant (conservative path)"})
    run_case("R01-unset-last-participant", unset_participant)

    def unset_scene_control(cid):
        S().ov.physics_scene.gravity_magnitude = 3
        ready, info = settle()
        assert ready, info
        standard(cid, lambda: S().ov.physics_scene.property_unset("gravity_magnitude"))
    run_case("R02-unset-scene-control", unset_scene_control)

    def unset_material(cid):
        bpy.data.materials["MatA"].ov.physics.static_friction = .7
        settle_participant()
        standard(cid, lambda: bpy.data.materials["MatA"].ov.physics.property_unset("static_friction"),
                 extra={"start": "material participant (conservative path)"})
    run_case("R03-unset-material", unset_material)

    def reset_to_default_value(cid):
        group = obj("Plain").ov.physics.rigid_body
        standard(cid, lambda: setattr(group, "mass_kg", type(group).bl_rna.properties["mass_kg"].default),
                 extra={"note": "RNA_property_reset path: explicit default value + update"})
    run_case("R04-reset-to-default-value", reset_to_default_value)

    def reset_default_operator(cid):
        poll = bpy.ops.ui.reset_default_button.poll()
        record(cid, True, not_applicable=True, operator_poll=poll,
               note="ui.reset_default_button needs an active UI button; not reachable headless")
    run_case("R05-reset-default-button-operator", reset_default_operator)

    # C: collection add / remove / clear.
    # No RNA update and no depsgraph update exists for these: latency-bounded.
    run_case("C01-raw-collider-add-nonparticipant",
             lambda cid: standard(cid, lambda: obj("Plain").ov.physics.colliders.add(), bounded=True))
    run_case("C02-raw-joint-axes-add-nonparticipant",
             lambda cid: standard(cid, lambda: obj("Handle").ov.physics.joint.axes.add(), bounded=True))
    run_case("C03-raw-posebone-collider-add",
             lambda cid: standard(cid, lambda: obj("Rig").pose.bones[0].ov.physics.colliders.add(),
                                  bounded=True))

    def raw_add_configure(cid):
        def change():
            record_ = obj("Plain").ov.physics.colliders.add()
            record_.shape = "BOX"
        standard(cid, change)
    run_case("C04-raw-add-then-configure", raw_add_configure)

    def raw_remove(cid):
        obj("Plain").ov.physics.colliders.add().name = "c0"
        settle_participant()
        standard(cid, lambda: obj("Plain").ov.physics.colliders.remove(0),
                 extra={"start": "participant (conservative path)"})
    run_case("C05-raw-remove-participant", raw_remove)

    def raw_clear(cid):
        for _ in range(2):
            obj("Plain").ov.physics.colliders.add()
        settle_participant()
        standard(cid, lambda: obj("Plain").ov.physics.colliders.clear(),
                 extra={"start": "participant (conservative path)"})
    run_case("C06-raw-clear-participant", raw_clear)

    def op_remove_collider(cid):
        plain = obj("Plain")
        with bpy.context.temp_override(object=plain, active_object=plain, selected_objects=[plain]):
            assert bpy.ops.ov.physics_add_body() == {"FINISHED"}
        settle_participant()
        collider_id = plain.ov.physics.colliders[0].persistent_id

        def change():
            with bpy.context.temp_override(object=plain, active_object=plain, selected_objects=[plain]):
                result = bpy.ops.ov.physics_edit_colliders(remove_id=collider_id)
            assert result == {"FINISHED"}, result
        standard(cid, change, extra={"start": "participant (conservative path)"})
    run_case("C07-operator-collider-remove", op_remove_collider)

    def mark_dirty_only(cid):
        # The collection contract: raw add/remove plus an explicit owner
        # notification (physics_authoring.mark_dirty), with no depsgraph update.
        def change():
            plain = obj("Plain")
            plain.ov.physics.colliders.add()
            physics_authoring.mark_dirty(S(), plain)
        standard(cid, change)
    run_case("C09-raw-add-with-mark-dirty-notification", mark_dirty_only)

    run_case("C08-initialize-joint-notifies",
             lambda cid: standard(cid, lambda: initialize_joint(obj("Handle"))))

    # R1: physics material reaching a non-participant through a material slot.
    def slot_data(cid):
        standard(cid, lambda: obj("Plain").data.materials.__setitem__(0, bpy.data.materials["PhysMat"]))
        obj("Plain").data.materials[0] = bpy.data.materials["MatA"]
    run_case("R1a-material-slot-assign-data", slot_data)

    def slot_object(cid):
        plain = obj("Plain")

        def change():
            plain.material_slots[0].link = "OBJECT"
            plain.material_slots[0].material = bpy.data.materials["PhysMat"]
        standard(cid, change)
        RESULTS[-1]["capture_sees_object_linked_material"] = any(
            r[0] == "MATERIAL" and r[1] == int(bpy.data.materials["PhysMat"].session_uid)
            for r in fresh(S()).leaves)
    run_case("R1b-material-slot-assign-object-link", slot_object)

    def slot_operator(cid):
        plain = obj("Plain")

        def change():
            with bpy.context.temp_override(object=plain, active_object=plain, selected_objects=[plain]):
                assert bpy.ops.object.material_slot_add() == {"FINISHED"}
            plain.material_slots[-1].material = bpy.data.materials["PhysMat"]
        standard(cid, change)
        with bpy.context.temp_override(object=plain, active_object=plain, selected_objects=[plain]):
            plain.active_material_index = len(plain.material_slots) - 1
            bpy.ops.object.material_slot_remove()
    run_case("R1c-material-slot-add-operator", slot_operator)

    # R3: object linked while kinematic sampling makes the depsgraph handler return early.
    def link_during_kinematics(cid, evaluating=True):
        late = mesh_object("Late", location=(6, 0, 0))

        def change():
            if evaluating:
                physics_live_kinematics._evaluating[0] += 1
            try:
                S().collection.objects.link(late)
                late.ov.physics.colliders.add()
                bpy.context.view_layer.update()
            finally:
                if evaluating:
                    physics_live_kinematics._evaluating[0] -= 1
        try:
            standard(cid, change, max_ticks=40)
        finally:
            bpy.data.objects.remove(late)
            pump(S())
    run_case("R3a-link-raw-participant-during-kinematics", link_during_kinematics)
    run_case("R3b-link-raw-participant-normal", lambda cid: link_during_kinematics(cid, False))

    # R5: empty depsgraph update list.
    def empty_updates(cid):
        scene = S()
        obj("Plain").ov.physics.colliders.add()
        reset_counts()
        changes = sessions.observe_physics_depsgraph(scene, SimpleNamespace(updates=()))
        skipped_observe = COUNT["observe"] == 0
        ok_after, _ = consistent(scene)
        reset_counts()
        sessions.observe_depsgraph_physics_edits(scene, SimpleNamespace(updates=()))
        record(cid, ok_after, camera_skip_returned=str(changes), camera_skip_observed=not skipped_observe,
               fast_path_observes_on_empty=COUNT["observe"] == 1,
               reasons=[] if ok_after else ["observe_physics_depsgraph treated empty updates as camera-only"])
    run_case("R5-empty-depsgraph-updates", empty_updates)

    # S: data shared by several objects / scenes.
    def shared_scene(cid, raw=False):
        main = S()
        other = bpy.data.scenes.new("SceneB")
        other.collection.objects.link(obj("Plain"))
        other.collection.objects.link(obj("Camera"))
        other.camera = obj("Camera")
        try:
            sessions.generation_for_scene(other, work_root=WORK)
            ready, info = settle(other)
            assert ready, info
            ready, info = settle(main)
            assert ready, info
            change = ((lambda: obj("Plain").ov.physics.colliders.add()) if raw
                      else (lambda: setattr(obj("Plain").ov.physics.rigid_body, "mass_kg", 3)))
            standard(cid, change, scene=other, depsgraph=True, bounded=raw)
        finally:
            bpy.data.scenes.remove(other)
    run_case("S01-shared-object-other-scene-setattr", shared_scene)
    run_case("S02-shared-object-other-scene-raw-add", lambda cid: shared_scene(cid, raw=True))

    def shared_material(cid):
        standard(cid, lambda: setattr(bpy.data.materials["MatA"].ov.physics, "dynamic_friction", .4),
                 extra={"users": [o.name for o in bpy.data.objects if o.type == "MESH" and
                                  bpy.data.materials["MatA"] in tuple(o.data.materials)]})
    run_case("S03-shared-material-two-objects", shared_material)

    # A: keyframed / driven physics properties.
    def fcurve(cid):
        plain = obj("Plain")

        def change():
            plain.animation_data_create()
            action = bpy.data.actions.new("PhysAnim")
            plain.animation_data.action = action
            curve = action.fcurve_ensure_for_datablock(plain, "ov.physics.rigid_body.mass_kg")
            curve.keyframe_points.insert(1, 2.0)
            curve.keyframe_points.insert(10, 7.0)
            S().frame_set(5)
        standard(cid, change, extra={"mass_after": None})
        RESULTS[-1]["mass_after"] = plain.ov.physics.rigid_body.mass_kg
        RESULTS[-1]["mass_is_set"] = plain.ov.physics.rigid_body.is_property_set("mass_kg")
        S().frame_set(1)
    run_case("A01-fcurve-on-nonparticipant", fcurve)

    def driver(cid):
        plain = obj("Plain")

        def change():
            curve = plain.driver_add("ov.physics.rigid_body.mass_kg")
            curve.driver.type = "SCRIPTED"
            curve.driver.expression = "frame * 0.5"
            S().frame_set(6)
        standard(cid, change)
        RESULTS[-1]["mass_after"] = plain.ov.physics.rigid_body.mass_kg
        RESULTS[-1]["mass_is_set"] = plain.ov.physics.rigid_body.is_property_set("mass_kg")
        S().frame_set(1)
    run_case("A02-driver-on-nonparticipant", driver)

    def frame_change_participant(cid):
        plain = obj("Plain")
        plain.animation_data_create()
        action = bpy.data.actions.new("PhysAnim2")
        plain.animation_data.action = action
        curve = action.fcurve_ensure_for_datablock(plain, "ov.physics.rigid_body.mass_kg")
        curve.keyframe_points.insert(1, 2.0)
        curve.keyframe_points.insert(10, 7.0)
        S().frame_set(1)
        settle_participant()
        standard(cid, lambda: S().frame_set(8), extra={"start": "animated participant (conservative path)"})
        S().frame_set(1)
    run_case("A03-frame-change-animated-participant", frame_change_participant)

    # L: linked and library-override data.
    def link_library(cid):
        def change():
            with bpy.data.libraries.load(str(libs["v2"]), link=True) as (_src, dst):
                dst.objects = ["LibPlain"]
            S().collection.objects.link(dst.objects[0])
        standard(cid, change)
        cleanup_libraries()
    run_case("L01-link-authored-library-object", link_library)

    def reload_library(cid):
        with bpy.data.libraries.load(str(libs["v1"]), link=True) as (_src, dst):
            dst.objects = ["LibPlain"]
        S().collection.objects.link(dst.objects[0])
        ready, info = settle()
        assert ready, info
        library = dst.objects[0].library

        def change():
            library.filepath = str(libs["v2"])
            library.reload()
        standard(cid, change)
        cleanup_libraries()
    run_case("L02-library-reload-adds-physics", reload_library)

    def override(cid):
        with bpy.data.libraries.load(str(libs["v1"]), link=True) as (_src, dst):
            dst.objects = ["LibPlain"]
        linked = dst.objects[0]
        S().collection.objects.link(linked)
        local = linked.override_create(remap_local_usages=True)
        ready, info = settle()
        assert ready, info
        standard(cid, lambda: setattr(local.ov.physics.rigid_body, "mass_kg", 5))
        bpy.data.objects.remove(local)
        cleanup_libraries()
    run_case("L03-library-override-setattr", override)

    # X / FR: explicit restart and final-render admission after an undetected raw edit.
    def explicit_restart(cid):
        scene = S()
        obj("Plain").ov.physics.colliders.add()
        stale = not consistent(scene)[0]
        reset_counts()
        try:
            sessions.generation_for_scene(scene, explicit_physics_restart=True)
            error = None
        except Exception as exc:  # noqa: BLE001 - authoring validity is not under test
            error = f"{type(exc).__name__}: {exc}"
        ok, current = consistent(scene)
        record(cid, ok, stale_before=stale, captures=COUNT["capture"], restart_error=error,
               participants_after=participants(current))
        accept_pending(scene)
    run_case("X01-explicit-restart-after-raw-edit", explicit_restart)

    def final_render(cid):
        scene = S()
        obj("Plain").ov.physics.colliders.add()
        stale = not consistent(scene)[0]
        reset_counts()
        try:
            engine._render_callback_adapter().final_render_from_scene(scene)
            error = None
        except Exception as exc:  # noqa: BLE001 - only admission/observation is under test
            error = f"{type(exc).__name__}: {exc}"
        ok, current = consistent(scene)
        record(cid, ok and COUNT["require_scene_ready"] > 0, stale_before=stale, captures=COUNT["capture"],
               require_scene_ready_calls=COUNT["require_scene_ready"], render_error=error,
               participants_after=participants(current))
        accept_pending(scene)
    run_case("FR01-final-render-admission-after-raw-edit", final_render)

    # U / R7: undo and redo. presentation_undo_pre (undo_pre/redo_pre) closes the
    # session on every history step, so the accepted generation is dropped
    # before the undo_post latch runs.
    bpy.context.preferences.edit.use_global_undo = True
    scheduled = []
    original_schedule = sessions.schedule_initial_generation

    def capturing_schedule(blender=None):
        """Run the real function; capture the timer it registers (headless runs none)."""
        timers = []
        current = sessions._owners.get(uid())
        at_call = {"close_status": sessions.close_status()["status"],
                   "accepted_generation": current is not None and current.current_generation is not None}
        proxy = SimpleNamespace(context=bpy.context, app=SimpleNamespace(timers=SimpleNamespace(
            register=lambda fn, first_interval=0.0: timers.append(fn))))
        value = original_schedule(proxy)
        scheduled.append({"returned": value, "timers": timers, "at_call": at_call})
        return value
    sessions.schedule_initial_generation = capturing_schedule

    def run_scheduled():
        """Emulate the captured timers; report captures and whether a build happened."""
        wait_closed()
        reset_counts()
        owner = sessions._owners.get(uid())
        before = None if owner is None else owner.current_generation
        for entry in scheduled:
            for timer in entry["timers"]:
                for _ in range(100):
                    if timer() is None:
                        break
                    wait_closed()
        accept_pending(S())
        owner = sessions._owners.get(uid())
        after = None if owner is None else owner.current_generation
        return {"timers": sum(len(e["timers"]) for e in scheduled), "timer_captures": COUNT["capture"],
                "built_generation": after is not None and after is not before,
                "returned": [e["returned"] for e in scheduled], "at_call": [e["at_call"] for e in scheduled]}

    def undo_first_participant(cid):
        bpy.ops.ed.undo_push(message="physics-free")
        obj("Plain").ov.physics.rigid_body.mass_kg = 3
        bpy.ops.ed.undo_push(message="first participant")
        strip_physics()
        bpy.ops.ed.undo_push(message="participant removed")
        ready, info = settle()
        assert ready, info
        scene_uid = uid()
        scheduled.clear()

        def change():
            assert bpy.ops.ed.undo() == {"FINISHED"}
        standard(cid, change)
        RESULTS[-1]["scene_uid_preserved"] = uid() == scene_uid
        RESULTS[-1]["restored_mass_set"] = obj("Plain").ov.physics.rigid_body.is_property_set("mass_kg")
        RESULTS[-1]["history_schedule"] = [{"returned": e["returned"], "at_call": e["at_call"]} for e in scheduled]
        scheduled.clear()

        def redo():
            assert bpy.ops.ed.redo() == {"FINISHED"}
        standard("U02-redo-removes-participant", redo)
        RESULTS[-1]["history_schedule"] = [{"returned": e["returned"], "at_call": e["at_call"]} for e in scheduled]
        # R7 on a real history step: undo_pre already dropped the generation.
        ready, info = settle()
        assert ready, info
        scheduled.clear()
        assert bpy.ops.ed.undo() == {"FINISHED"}
        real = run_scheduled()
        record("R7a-undo-schedule-after-session-close",
               bool(real["at_call"]) and not real["at_call"][0]["accepted_generation"] and real["built_generation"],
               **real, note="undo_pre closed the session; the scheduled build replaces the dropped generation")
    run_case("U01-undo-restores-first-participant", undo_first_participant)

    def history_handler_on_accepted(cid):
        # R7 premise: the undo/redo handler runs while a generation is accepted.
        # A viewport-path build never marks initial-generation diagnostics ready.
        scheduled.clear()
        engine._physics_authoring_history_restored()
        result = run_scheduled()
        ok = not any(result["returned"]) and result["timers"] == 0 and not result["built_generation"]
        scheduled.clear()
        engine._physics_authoring_history_restored()
        second = run_scheduled()
        record(cid, ok, **result, second_call=second, reasons=[] if ok else
               ["history handler scheduled a generation timer while a generation was accepted"])
    run_case("R7b-history-handler-noop-when-accepted", history_handler_on_accepted)

    def history_handler_viewport_built(cid):
        # Same premise, but the accepted generation came from the viewport path
        # after a close (initial-generation diagnostics "unavailable"), which is
        # every generation built after a file load or history step.
        sessions._invalidate_initial_generation()
        history_handler_on_accepted(cid)
    run_case("R7c-history-handler-noop-when-viewport-built", history_handler_viewport_built)

    def undo_physics_free(cid):
        bpy.ops.ed.undo_push(message="before move")
        obj("Plain2").location.x += 1
        bpy.ops.ed.undo_push(message="after move")
        ready, info = settle()
        assert ready, info

        def change():
            assert bpy.ops.ed.undo() == {"FINISHED"}
        standard(cid, change)
    run_case("U03-undo-physics-free-edit", undo_physics_free)
    sessions.schedule_initial_generation = original_schedule

    # F / R4: file load, revert, new file, add-on reload.
    file_cases(libs)


def settle_participant():
    """Accept a state with participants (conservative path), without proving reuse."""
    scene = S()
    bpy.context.view_layer.update()
    sessions.observe_physics_edits(scene)
    sessions._physics_restart_pending.pop(uid(scene), None)
    pump(scene)
    sessions.observe_physics_edits(scene)
    sessions._physics_restart_pending.pop(uid(scene), None)
    assert sessions._physics_view_observations[uid(scene)][1], "expected participants"


def cleanup_libraries():
    for o in tuple(bpy.data.objects):
        if o.library is not None or o.override_library is not None:
            bpy.data.objects.remove(o)
    for library in tuple(bpy.data.libraries):
        bpy.data.libraries.remove(library)
    pump(S())


CLOSE_FAILURES: list[dict] = []


def wait_closed(timeout=60):
    deadline = time.monotonic() + timeout
    while sessions.close_status()["status"] == "closing" and time.monotonic() < deadline:
        time.sleep(.05)
    status = sessions.close_status()
    if status["status"] == "failed" and status not in CLOSE_FAILURES:
        CLOSE_FAILURES.append(status)
        print(f"[invalidation] scene close failed: {status}", flush=True)
    return status["status"]


def after_load(cid, old_uid, **extra):
    """After a load the new scene must not reuse anything; rebuild must equal capture."""
    status = wait_closed()
    scene = S()
    stale_keys = sorted(set(sessions._physics_view_observations) | set(sessions._physics_snapshots)
                        | set(sessions._physics_authoring_dirty))
    view = lookup(scene)
    no_reuse = view["error"] is not None and view["captures"] == 0
    pump(scene)
    ok, current = consistent(scene)
    record(cid, ok and no_reuse and uid(scene) != old_uid if extra.get("expect_new_uid", True) else ok and no_reuse,
           close_status=status, new_scene_uid=uid(scene), old_scene_uid=old_uid,
           stale_latch_keys_after_load=stale_keys, first_lookup=view,
           participants_after=participants(current), **extra)


def file_cases(libs):
    authored = OUT / "authored.blend"

    def open_file(cid):
        obj("Plain").ov.physics.rigid_body.mass_kg = 5
        bpy.ops.wm.save_as_mainfile(filepath=str(authored), copy=True)
        strip_physics()
        ready, info = settle()
        assert ready, info
        old = uid()
        assert bpy.ops.wm.open_mainfile(filepath=str(authored)) == {"FINISHED"}
        after_load(cid, old)
    run_case("F01-open-file", open_file)

    def revert(cid):
        strip_physics()
        ready, info = settle()
        assert ready, info
        old = uid()
        assert bpy.ops.wm.revert_mainfile() == {"FINISHED"}
        after_load(cid, old, reverted_mass_set=obj("Plain").ov.physics.rigid_body.is_property_set("mass_kg"))
    run_case("F02-revert-file", revert, base=False)

    def new_file(cid):
        strip_physics()
        ready, info = settle()
        assert ready, info
        old = uid()
        assert bpy.ops.wm.read_homefile(use_factory_startup=True) == {"FINISHED"}
        after_load(cid, old)
    run_case("F03-read-homefile", new_file, base=False)

    def addon_reload(cid):
        bpy.ops.wm.open_mainfile(filepath=str(authored))
        wait_closed()
        strip_physics()
        ready, info = settle()
        assert ready, info
        old = uid()
        obj("Plain").ov.physics.colliders.add()  # raw, undetected before the reload
        addon.unregister()
        addon.register()
        after_load(cid, old, expect_new_uid=False)
    run_case("F04-addon-reload-with-raw-edit", addon_reload, base=False)

    def pending_close(cid):
        bpy.ops.wm.open_mainfile(filepath=str(authored))
        wait_closed()
        strip_physics()
        ready, info = settle()
        assert ready, info
        old = uid()
        release = threading.Event()
        sessions.close(defer_native=True, after_native=lambda: release.wait(30))
        sessions._poll_physics_edits()  # the 0.2 s timer still observes while closing
        latched_during_close = old in sessions._physics_view_observations
        bpy.ops.wm.open_mainfile(filepath=str(authored))
        survived = old in sessions._physics_view_observations
        blocked = sessions.close_status()["status"]
        release.set()
        after_load(cid, old, latched_during_close=latched_during_close,
                   stale_old_uid_survived_load=survived, status_at_load=blocked)
    run_case("F05-R4-load-during-pending-close", pending_close, base=False)


# --------------------------------------------------------------------------- perf (R2 / R6)
def stats(samples):
    ordered = sorted(samples)
    return {"n": len(samples), "median_ms": statistics.median(samples),
            "p95_ms": ordered[min(len(ordered) - 1, int(round(.95 * len(ordered))) - 1)],
            "max_ms": ordered[-1]}


def timed(fn, samples=20, warmup=3):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        fn()
        out.append((time.perf_counter_ns() - start) / 1e6)
    return stats(out)


def perf(count):
    for existing in tuple(bpy.data.objects):
        bpy.data.objects.remove(existing)
    scene = S()
    mesh = mesh_object("perf_template", bpy.data.materials.new("perf_material")).data
    side = max(1, int(count ** .5 + .999))
    for i in range(count):
        o = bpy.data.objects.new(f"perf_{i:05d}", mesh)
        o.location = (i % side, i // side, 0)
        scene.collection.objects.link(o)
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    bpy.context.view_layer.update()
    result = {"objects": count, "blender": bpy.app.version_string}
    start = time.perf_counter()
    sessions.generation_for_scene(scene, work_root=WORK)
    pump(scene)
    result["build_s"] = time.perf_counter() - start
    objects = list(scene.objects)

    # R6: one authored participant; every callback keeps a full capture.
    participant = objects[0]
    participant.ov.physics.rigid_body.mass_kg = 2
    bpy.context.view_layer.update()
    pump(scene)
    sessions._physics_restart_pending.clear()
    assert sessions._physics_view_observations[uid(scene)][1]
    adapter_gate = lambda: (sessions.viewport_joint_admission_clear(scene)  # noqa: E731
                            or physics_authoring.require_scene_ready(scene, allow_pending_anchors=True))
    reset_counts()
    sessions.generation_for_viewport(scene)
    r6 = {"lookup_captures": COUNT["capture"],
          "generation_for_viewport": timed(lambda: sessions.generation_for_viewport(scene)),
          "viewport_admission_gate": timed(adapter_gate),
          "poll_tick": timed(sessions._poll_physics_edits),
          "capture": timed(lambda: fresh(scene))}
    samples = []

    def handler_probe(scene_, depsgraph):
        start = time.perf_counter_ns()
        sessions.observe_physics_depsgraph(scene_, depsgraph)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    bpy.app.handlers.depsgraph_update_post.append(handler_probe)
    for i in range(23):
        camera.location.x = .001 * (i % 2)
        bpy.context.view_layer.update()
    bpy.app.handlers.depsgraph_update_post.remove(handler_probe)
    r6["depsgraph_camera_move_physics_observation"] = stats(samples[3:]) if len(samples) > 3 else samples
    result["R6_one_participant"] = r6

    # R2: worst-case raw-edit discovery latency on the physics-free scene.
    participant.ov.physics.rigid_body.property_unset("mass_kg")
    participant.property_unset("ov")
    sessions.observe_physics_edits(scene)
    sessions._physics_restart_pending.clear()
    ready, info = reuse_active(scene)
    assert ready, info
    # Idle cost: one full pass of ticks with nothing to find.
    sessions._physics_idle_discovery.pop(uid(scene), None)
    sessions._poll_physics_edits()  # creates discovery and probes the first batch
    discovery = sessions._physics_idle_discovery[uid(scene)]
    idle_ms, idle_probed, passes = [], [], discovery.passes
    while discovery.passes < passes + 2:
        start = time.perf_counter_ns()
        sessions._poll_physics_edits()
        idle_ms.append((time.perf_counter_ns() - start) / 1e6)
        idle_probed.append(discovery.last_probed)
    target = discovery.objects[(discovery.index - 1) % len(discovery.objects)]  # just probed
    target.ov.physics.colliders.add()  # raw: no update callback, no depsgraph update
    view = lookup(scene)
    stale_after_lookup = not consistent(scene)[0]
    stale_admission = sessions.viewport_joint_admission_clear(scene)
    ticks, tick_ms, probed = 0, [], []
    while True:
        ticks += 1
        current = sessions._physics_idle_discovery.get(uid(scene))
        start = time.perf_counter_ns()
        sessions._poll_physics_edits()
        tick_ms.append((time.perf_counter_ns() - start) / 1e6)
        if current is not None:
            probed.append(current.last_probed)
        if consistent(scene)[0] or ticks > len(objects):
            break
    batch = snapshots.RenderOnlyDiscovery.MIN_BATCH
    result["R2_worst_case"] = {
        "scene_objects": len(scene.objects), "min_batch_per_tick": batch,
        "time_budget_s": snapshots.RenderOnlyDiscovery.TIME_BUDGET_S, "timer_interval_s": .2,
        "hard_bound_ticks": -(-len(scene.objects) // batch),
        "lookup_after_raw_edit": view, "stale_after_lookup": stale_after_lookup,
        "admission_clear_while_stale": stale_admission,
        "ticks_to_detection": ticks, "latency_s_at_0.2s_interval": (ticks - 1) * .2 + sum(tick_ms) / 1e3,
        "objects_probed_per_tick": probed, "tick_cpu_ms": stats(tick_ms),
        "idle_tick_cpu_ms": stats(idle_ms), "idle_objects_per_tick": stats(idle_probed),
        "idle_ticks_per_pass": len(idle_ms) / 2,
        "idle_ticks_ms_probed": [[round(ms, 3), n] for ms, n in zip(idle_ms, idle_probed)],
        "restart_queued": uid(scene) in sessions._physics_restart_pending,
        "detected": consistent(scene)[0]}

    # R3: an Object linked with a raw collider that no depsgraph update observed
    # (the kinematics latch is the immediate path; this is the roster backstop).
    target.property_unset("ov")  # the detecting poll's identity repair also authored it
    sessions.observe_physics_edits(scene)
    sessions._physics_restart_pending.clear()
    ready, info = reuse_active(scene)
    assert ready, info
    sessions._poll_physics_edits()
    late = bpy.data.objects.new("perf_late", mesh)
    scene.collection.objects.link(late)
    late.ov.physics.colliders.add()
    r3_ticks, r3_ms = 0, []
    while True:
        r3_ticks += 1
        start = time.perf_counter_ns()
        sessions._poll_physics_edits()
        r3_ms.append((time.perf_counter_ns() - start) / 1e6)
        if consistent(scene)[0] or r3_ticks > len(objects):
            break
    result["R3_unobserved_link"] = {
        "ticks_to_detection": r3_ticks, "latency_s_at_0.2s_interval": (r3_ticks - 1) * .2 + sum(r3_ms) / 1e3,
        "tick_cpu_ms": stats(r3_ms), "detected": consistent(scene)[0]}
    bpy.data.objects.remove(late)
    sessions.observe_physics_edits(scene)
    sessions._physics_restart_pending.clear()
    ready, info = reuse_active(scene)
    assert ready, info

    # frame_change_post on the physics-free scene (no animation; then all animated).
    frame_samples = []

    def frame_probe(scene_, depsgraph):
        start = time.perf_counter_ns()
        sessions.physics_frame_changed(scene_, depsgraph)
        frame_samples.append((time.perf_counter_ns() - start) / 1e6)
    bpy.app.handlers.frame_change_post.append(frame_probe)
    for i in range(23):
        scene.frame_set(1 + i % 2)
    static = stats(frame_samples[3:])
    for o in objects[:-1]:
        if o.type == "MESH":
            o.keyframe_insert("location", frame=1)
            o.location.z += 1
            o.keyframe_insert("location", frame=10)
    scene.frame_set(3)  # relations rebuild frame: every Object is listed once
    frame_samples.clear()
    for i in range(23):
        scene.frame_set(2 + i % 2)
    animated = stats(frame_samples[3:])
    bpy.app.handlers.frame_change_post.remove(frame_probe)
    result["frame_change_handler"] = {"static_scene": static, "all_objects_animated": animated,
                                      "latched": uid(scene) in sessions._physics_authoring_dirty}
    (OUT / f"perf-{count}.json").write_text(json.dumps(result, indent=2, default=str))
    print("PHYSICS_VIEWPORT_INVALIDATION_PERF " + json.dumps(
        {"objects": count, "R2_ticks": ticks, "R2_latency_s": result["R2_worst_case"]["latency_s_at_0.2s_interval"],
         "R3_ticks": r3_ticks, "R6_lookup_ms": r6["generation_for_viewport"]["median_ms"]}), flush=True)


if PERF:
    perf(PERF)
else:
    matrix()
    (OUT / "results.json").write_text(json.dumps({"blender": bpy.app.version_string, "cases": RESULTS,
                                                  "close_failures": CLOSE_FAILURES},
                                                 indent=2, default=str))
    print(f"PHYSICS_VIEWPORT_INVALIDATION_DONE {len(RESULTS)} cases", flush=True)
