# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-RNA check that physics observation refuses worker threads (OMPE-115792 n-02-03).

Run: blender --background --factory-startup --python-exit-code 17 --python <this file>
"""
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "addon"))
from ovrtx_blender_example import authoring_properties as props
from ovrtx_blender_example import physics_edit_snapshot as snapshots
from ovrtx_blender_example import scene_generation_sessions as sessions

props.register()
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
for index in range(64):
    obj = bpy.data.objects.new(f"plain-{index}", bpy.data.meshes.new(f"plain-{index}"))
    scene.collection.objects.link(obj)
camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
scene.collection.objects.link(camera)
bpy.context.view_layer.update()
uid = int(scene.session_uid)
checks = []


def check(label, condition):
    assert condition, label
    checks.append(label)


def is_main():
    return threading.current_thread() is threading.main_thread()


def off_main(function):
    outcome = {}

    def work():
        try:
            outcome["value"] = function()
        except BaseException as exc:  # noqa: BLE001 - asserted below
            outcome["error"] = exc
    thread = threading.Thread(target=work, name="n-02-03-worker")
    thread.start()
    thread.join(10)
    assert not thread.is_alive(), "worker did not finish"
    return outcome.get("error")


# Spy on the helpers that have no guard of their own.
unguarded = []
real_authored, real_discovery = snapshots._authored, snapshots.RenderOnlyDiscovery


def authored(group):
    unguarded.append(("_authored", is_main()))
    return real_authored(group)


class Discovery(real_discovery):
    def __init__(self, owner):
        unguarded.append(("RenderOnlyDiscovery", is_main()))
        super().__init__(owner)


snapshots._authored, snapshots.RenderOnlyDiscovery = authored, Discovery

# Main thread: a full observation of real RNA is admitted and physics-free.
generation = SimpleNamespace(number=1)
sessions._owners[uid] = SimpleNamespace(current_generation=generation, pending_generation=None,
                                       reuse=lambda: generation)
sessions._remember_physics_snapshot(uid, snapshots.capture(scene))
check("main thread capture is physics-free", sessions._physics_view_observations[uid][1] is False)
check("main thread reuse needs no capture", sessions.generation_for_viewport(scene) is generation)
check("main thread scene observation", sessions._render_only_physics_current(scene))
before = list(unguarded)
check("main thread capture used _authored", ("_authored", True) in before)

# Worker threads: every observation entry fails closed before reading Objects.
depsgraph = bpy.context.evaluated_depsgraph_get()
entries = {
    "capture": lambda: snapshots.capture(scene),
    "scene_authoring_observation": lambda: snapshots.scene_authoring_observation(scene),
    "generation_for_viewport": lambda: sessions.generation_for_viewport(scene),
    "_render_only_physics_current": lambda: sessions._render_only_physics_current(scene),
    "observe_physics_depsgraph": lambda: sessions.observe_physics_depsgraph(scene, depsgraph),
    "observe_depsgraph_physics_edits": lambda: sessions.observe_depsgraph_physics_edits(scene, depsgraph),
    "observe_physics_edits": lambda: sessions.observe_physics_edits(scene),
    "_poll_physics_edits": sessions._poll_physics_edits,
}
for name, entry in entries.items():
    error = off_main(entry)
    check(f"worker {name} refused", isinstance(error, RuntimeError) and "main thread" in str(error))
check("no unguarded helper ran on a worker", unguarded == before)
check("worker poll created no discovery", uid not in sessions._physics_idle_discovery)

# The sanctioned render-init admission is thread-local and scoped.
def render_init_job():
    with snapshots.render_init_capture():
        return snapshots.capture(scene)
mark = len(unguarded)
check("render_init_capture admits its own job thread", off_main(render_init_job) is None)
sanctioned = unguarded[mark:]
check("only the sanctioned job thread ran _authored off main",
      sanctioned and all(name == "_authored" and not main for name, main in sanctioned))
del unguarded[mark:]
check("render_init_capture does not leak", isinstance(off_main(lambda: snapshots.capture(scene)), RuntimeError))

# Blender delivers the add-on's callbacks on the main thread.
threads = {}
real_note = sessions.note_physics_authoring_changed


def note(owner=None):
    threads.setdefault("rna_update", []).append(is_main())
    return real_note(owner)


sessions.note_physics_authoring_changed = note
target = bpy.data.objects["plain-0"]
target.ov.physics.joint.show_indicator = False
check("RNA update callback latched on the main thread", threads.get("rna_update") == [True])
check("RNA update callback latched the scene", uid in sessions._physics_authoring_dirty)
sessions.note_physics_authoring_changed = real_note


def depsgraph_handler(_scene, _depsgraph):
    threads.setdefault("depsgraph_update_post", []).append(is_main())


bpy.app.handlers.depsgraph_update_post.append(depsgraph_handler)
camera.location.x += 1
bpy.context.view_layer.update()
bpy.app.handlers.depsgraph_update_post.remove(depsgraph_handler)
check("depsgraph_update_post runs on the main thread", threads.get("depsgraph_update_post", [False])[0]
      and all(threads["depsgraph_update_post"]))

# Main-thread poll after a latch observes and records RNA on the main thread.
check("main thread poll ticks", sessions._poll_physics_edits() == 0.2)
check("poll cleared latch on main thread", uid not in sessions._physics_authoring_dirty)
check("every unguarded helper call ran on the main thread", all(main for _name, main in unguarded))
snapshots._authored, snapshots.RenderOnlyDiscovery = real_authored, real_discovery
print("\n".join(checks))
print("PHYSICS_OBSERVATION_MAIN_THREAD_OK")
