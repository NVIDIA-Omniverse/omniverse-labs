# SPDX-License-Identifier: Apache-2.0
"""Main-thread evaluation for generation-owned, single-step live inputs."""
import math

from .physics_kinematic_inputs import capture_kinematic_inputs

#: Upper bound of steps evaluated per main-thread request.
BATCH_STEPS = 30
#: Main-thread time budget per request; the UI gets control back between batches.
BATCH_BUDGET_S = 0.012
#: Prefetch time per render tick. A long catch-up continues on later ticks so
#: the render thread picks up newer timeline requests instead of finishing a
#: stale target.
TICK_BUDGET_S = 0.1

# (scene uid, generation number, digest) -> pending main-thread requests and
# finished samples. Sources are rebuilt every tick, so request state lives here.
_state = {}
_evaluating = [0]
_animation_epoch = [0]


def evaluating():
    """True while the main thread steps transient frames for kinematic sampling."""
    return _evaluating[0] > 0


def note_animation_edit():
    """Animation data changed; samples evaluated earlier are stale."""
    _animation_epoch[0] += 1


def forget_scene(scene_uid):
    for key in [k for k in _state if k[0] == scene_uid]:
        del _state[key]


class KinematicSampleNotReady(RuntimeError):
    """The requested kinematic input has not been prefetched yet."""


class LiveKinematicSource:
    """Kinematic inputs prefetched on the main thread before runtime locks.

    The render thread calls ``prefetch_*`` while holding no runtime lock (it
    blocks until the main thread has evaluated), then the locked tick only
    reads the cache through ``frames_per_second``/``sample``. Main-thread edit
    handlers and activation therefore never wait on a thread that waits on
    them.
    """

    def __init__(self, scene_uid, generation):
        self.scene_uid = scene_uid
        self.generation = generation
        key = (scene_uid, getattr(generation, "number", None), getattr(generation, "digest", id(generation)))
        for stale in [k for k in _state if k[0] == scene_uid and k != key]:
            del _state[stale]
        self._cache = _state.setdefault(key, {"fps": {}, "samples": {}, "epoch": None, "key": None, "error": None})

    def record_error(self, exc):
        """A prefetch failure is raised by the next locked lookup."""
        self._cache["error"] = exc

    def _raise_recorded(self):
        error, self._cache["error"] = self._cache["error"], None
        if error is not None:
            raise error

    def _guarded(self, callback):
        def invoke():
            import bpy
            from . import scene_generation_sessions as sessions
            runtime = sessions._runtimes.get(self.scene_uid)
            if (runtime is None or runtime.lifecycle_status != "open" or runtime.reuse_blocked
                    or getattr(runtime.scheduler, "baked_playback_paused", False)
                    or runtime.ovphysx._active_scene_generation is not self.generation
                    or sessions._bake_capture_owner is not None):
                raise ValueError("Kinematic input generation is no longer active")
            scene = next((s for s in bpy.data.scenes if int(s.session_uid) == self.scene_uid), None)
            if scene is None:
                raise ValueError("Kinematic input scene was removed")
            return callback(scene)
        return invoke

    def _on_main(self, callback):
        # Callers must hold no runtime lock (see class docstring).
        from .engine import _run_on_main_thread
        return _run_on_main_thread(self._guarded(callback))

    def prefetch_fps(self, initial_frame):
        if initial_frame in self._cache["fps"]:
            return self._cache["fps"][initial_frame]
        def timing(scene):
            fps = scene.render.fps / scene.render.fps_base
            if (self.generation.diagnostics.get("physics_initial_frame") != initial_frame
                    or not math.isclose(self.generation.diagnostics.get("physics_initial_fps", 0), fps, rel_tol=1e-12)):
                raise ValueError("Kinematic initial frame/rate differs from the accepted generation; rebuild at the timeline start")
            return fps
        fps = self._on_main(timing)
        self._cache["fps"][initial_frame] = fps
        return fps

    def prefetch(self, steps, timestep_ns, initial_frame, fps):
        """Evaluate ``steps`` (within the tick budget) and cache them."""
        import time
        deadline = time.perf_counter() + TICK_BUDGET_S
        steps = [step for step in steps if step not in self._cache["samples"]]
        key = (timestep_ns, initial_frame, fps)
        if self._cache["key"] != key or self._cache["epoch"] != _animation_epoch[0]:
            self._cache.update(samples={}, key=key, epoch=_animation_epoch[0])
        # Budgeted main-thread slices; the UI runs between them while this
        # (lock-free) caller waits.
        while steps and time.perf_counter() < deadline:
            samples, epoch = self._on_main(lambda scene: self._evaluate_steps(scene, steps[:BATCH_STEPS], *key))
            if epoch != _animation_epoch[0] or epoch != self._cache["epoch"] or self._cache["key"] != key:
                # Keys changed since earlier slices were cached; the next tick
                # prefetches afresh.
                self._cache.update(samples={}, key=None, epoch=None)
                return
            if not samples:
                return
            self._cache["samples"].update(samples)
            steps = [step for step in steps if step not in samples]

    def frames_per_second(self, initial_frame):
        """Cached frame rate; never waits for the main thread."""
        self._raise_recorded()
        if initial_frame not in self._cache["fps"]:
            raise KinematicSampleNotReady("Kinematic frame rate has not been prefetched")
        return self._cache["fps"][initial_frame]

    def sample(self, step, timestep_ns, initial_frame, fps, through_step=None):
        """Cached input for ``step``; never waits for the main thread."""
        self._raise_recorded()
        cache = self._cache
        if cache["key"] != (timestep_ns, initial_frame, fps) or cache["epoch"] != _animation_epoch[0]:
            cache.update(samples={}, key=None, epoch=None)  # Keys were edited since evaluation.
        for old in [s for s in cache["samples"] if s < step]:
            del cache["samples"][old]
        if step not in cache["samples"]:
            raise KinematicSampleNotReady("Kinematic input has not been prefetched")
        return cache["samples"].pop(step)

    def _evaluate_steps(self, scene, steps, timestep_ns, initial_frame, fps):
        """Evaluate consecutive steps within a time budget, restoring the scene once."""
        import time
        import bpy
        from . import scene_generation_sessions as sessions
        from .interactive_operator_state import suppress_interactive_edit_bridge
        from .physics_authoring import require_scene_ready
        require_scene_ready(scene)
        if not math.isclose(scene.render.fps / scene.render.fps_base, fps, rel_tol=1e-12):
            raise ValueError("Scene frame rate changed during kinematic stepping")
        epoch = _animation_epoch[0]
        started = time.perf_counter()
        saved_frame = scene.frame_current, scene.frame_subframe
        was_stopped = self.scene_uid in sessions._presentation_stopped
        names = ("location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale")
        channels = [(channel, {name: tuple(getattr(channel, name)) for name in names})
                    for obj in scene.objects for channel in (obj, *getattr(obj.pose, "bones", ()))]
        samples = {}
        _evaluating[0] += 1
        try:
            with suppress_interactive_edit_bridge():
                try:
                    for step in steps:
                        target_frame = initial_frame + step * timestep_ns / 1e9 * fps
                        whole = math.floor(target_frame)
                        scene.frame_set(whole, subframe=target_frame - whole)
                        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
                            samples[step] = capture_kinematic_inputs(scene, self.generation,
                                                                     bpy.context.evaluated_depsgraph_get())
                        if time.perf_counter() - started > BATCH_BUDGET_S:
                            break
                finally:
                    scene.frame_set(saved_frame[0], subframe=saved_frame[1])
                    for channel, values in channels:
                        for name, value in values.items():
                            setattr(channel, name, value)
                    for layer in scene.view_layers:
                        layer.update()
                    if was_stopped:
                        sessions._presentation_stopped.add(self.scene_uid)
                    else:
                        sessions._presentation_stopped.discard(self.scene_uid)
        finally:
            _evaluating[0] -= 1
        return samples, epoch
