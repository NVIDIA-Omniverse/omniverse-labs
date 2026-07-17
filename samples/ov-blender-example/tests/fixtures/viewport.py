# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ViewportSession — tracks viewport state for interactive rendering.

This module is the bookkeeping layer behind the interactive viewport
(view_update/view_draw in engine.py). The viewport renders a *live* USD
scene held by ovrtx; the goal here is to push the *smallest possible*
delta to that scene each time Blender's depsgraph ticks, so that orbiting
the camera or dragging an object does not trigger a full re-export of the
entire scene (which would stutter the viewport).

It provides three concerns, all stateless except ViewportSession:

  - Matrix conversion between Blender (column-major) and USD (row-major).
    Every transform we hand to ovrtx must be transposed; the helpers below
    are the single chokepoint for that convention so call sites never have
    to think about it.

  - Camera change detection for live viewport orbit/pan/zoom. The viewport
    "camera" is derived from Blender's view matrix, not a Camera object, so
    it is tracked separately from scene objects.

  - Incremental transform value changes via depsgraph.updates, avoiding full
    scene re-export when only object positions change. This is the hot path
    during animation playback and interactive manipulation.

Design note: the module deliberately tolerates "mock" matrix-like objects
(plain lists of lists, e.g. in tests) as well as real mathutils.Matrix
instances, which is why the conversion helpers index with [i][j] by hand
instead of relying on Matrix-specific methods.
"""

import numpy as np


def matrix_to_row_major_flat(blender_matrix):
    """Convert a Blender 4x4 Matrix to a flat list of 16 doubles in USD row-major order.

    Why this exists: Blender and USD disagree on matrix storage/convention,
    so a raw flatten of a Blender matrix would feed ovrtx a transposed (and
    therefore wrong) transform. This is the canonical single-matrix fixer.

    Blender stores column-major (M*v convention — matrix on the left).
    USD stores row-major (v*M convention — row vector on the left).
    The two are transposes of each other, so we read the matrix into a numpy
    array, transpose it, and flatten in C order to get the 16 doubles USD
    expects (m00,m01,m02,m03, m10,...).

    Returns a plain Python list (not a numpy array) because the ctypes
    bridge / USD attribute setters in engine.py consume native lists.
    """
    m = blender_matrix
    # Build a 4x4 float64 array by hand, one element at a time. We index
    # m[i][j] rather than calling Matrix methods so this also accepts
    # list-of-lists "mock" matrices used in tests. float64 because USD
    # transforms are doubles and we don't want to lose precision for
    # objects placed at large world coordinates.
    arr = np.empty((4, 4), dtype=np.float64)
    for i in range(4):
        row = m[i]  # one Blender row (also valid for a nested list)
        for j in range(4):
            arr[i, j] = row[j]
    # .T performs the Blender->USD transpose; .ravel() flattens row-major.
    return arr.T.ravel().tolist()


def fill_usd_transform_buffer(objects, depsgraph, buf):
    """Batch equivalent of matrix_to_row_major_flat for many moving objects.

    Used on the hottest path: pushing transforms for a large set of objects
    (e.g. a crowd or a physics sim) to ovrtx in one shot. Instead of building
    N small Python lists, it writes every evaluated matrix into a single
    pre-allocated (N, 4, 4) numpy buffer and transposes them all at once.

    Args:
        objects:   iterable of Blender objects, in the same order the caller
                   will associate with the flattened output.
        depsgraph: the evaluated depsgraph; we fetch the evaluated (not the
                   original) object so modifiers/constraints are baked in.
        buf:       caller-owned (N, 4, 4) float buffer, reused across frames
                   to avoid per-frame allocation. We overwrite it in place.

    The returned buffer is USD row-major row-vector order, matching
    matrix_to_row_major_flat exactly, but avoids per-object Python list building.
    """
    for i, obj in enumerate(objects):
        # evaluated_get applies the depsgraph evaluation; matrix_world is the
        # final world transform after parenting/constraints/modifiers.
        buf[i] = obj.evaluated_get(depsgraph).matrix_world
    # transpose(0, 2, 1) transposes each 4x4 in the stack (Blender->USD) while
    # leaving the object axis (0) alone; ascontiguousarray forces a C-contiguous
    # copy so the trailing reshape to one flat (N*16,) vector is zero-copy-safe.
    return np.ascontiguousarray(buf.transpose(0, 2, 1)).reshape(-1)


def matrices_batch_to_row_major(blender_matrices):
    """Convert a batch of Blender 4x4 matrices to row-major flat lists.

    Like fill_usd_transform_buffer, but for callers that already hold the
    matrices (rather than objects) and want a list-of-lists result instead of
    a single flat vector — e.g. when each matrix must stay paired with its own
    prim path. The numpy transpose is done once over the whole stack, so this
    is dramatically faster than looping matrix_to_row_major_flat per matrix.

    Returns list of 16-element lists. Much faster than calling
    matrix_to_row_major_flat in a loop due to batched numpy ops.
    """
    n = len(blender_matrices)
    if n == 0:
        return []  # nothing to do; avoid an empty np.empty/transpose
    arr = np.empty((n, 4, 4), dtype=np.float64)
    # Hand-copy each matrix (works for Matrix and list-of-lists mocks alike).
    for i, m in enumerate(blender_matrices):
        for r in range(4):
            row = m[r]
            for c in range(4):
                arr[i, r, c] = row[c]
    # Transpose each matrix and flatten — transpose(0, 2, 1) swaps the row/col
    # axes per matrix (Blender col-major -> USD row-major) and reshape(n, 16)
    # lays each transposed 4x4 out as 16 consecutive doubles.
    arr = arr.transpose(0, 2, 1).reshape(n, 16)
    return arr.tolist()


def matrices_close(a, b, atol=1e-6, rtol=1e-5):
    """Compare two 4x4 Blender matrices element-wise within tolerance.

    The viewport polls the camera every redraw; without a tolerance, harmless
    float jitter in the view matrix would report "camera moved" on every frame
    and force a needless transform push to ovrtx. This is the gate that lets an
    idle viewport stay idle.

    Why both tolerances: a pure absolute tolerance fails for cameras at large
    coordinates (e.g. 10000, 10000, 10000) where the float ULP is itself larger
    than atol, producing false "moved" positives; a pure relative tolerance
    fails near zero. The combined threshold (mirroring numpy.allclose's formula)
    handles both the tiny-near-origin and the large-coordinate cases.
    """
    for i in range(4):
        for j in range(4):
            ai, bi = a[i][j], b[i][j]
            # Scale the allowed error by the larger of the two magnitudes.
            threshold = atol + rtol * max(abs(ai), abs(bi))
            if abs(ai - bi) > threshold:
                return False  # any element differing => matrices differ
    return True


class ViewportSession:
    """Tracks viewport state for incremental updates.

    One instance lives for the lifetime of an interactive viewport render.
    It remembers what was already exported to ovrtx so subsequent depsgraph
    ticks can be answered with a minimal delta (transform-only updates) rather
    than a full re-export. The engine consults it in this order each tick:
    needs_full_sync() -> (if False) get_transform_updates() + check_camera_changed().

    The maps below are the persistent state; reset() clears everything when the
    scene is torn down or a full re-sync is forced.
    """

    def __init__(self):
        # False until the first full export has been pushed to ovrtx. While
        # False, every needs_full_sync() returns True (we can't do a delta
        # against a scene that was never loaded).
        self.scene_loaded = False
        self.tracked_objects = {}  # obj_name -> sentinel (True): set of names we have exported
        self.prim_paths = {}  # obj_name -> USD prim path string in the live ovrtx stage
        self.instance_map = {}  # base_obj_name -> list of instance tracked names (encounter order)
        self.camera_prim_path = None  # USD prim path for active camera (reserved; camera uses view matrix)
        self.last_view_matrix = None  # last seen context.region_data.view_matrix, for change detection
        # Live-lookdev maps: which light objects / materials were exported and where
        # their prims live, so a light/material edit can be pushed as a tiny attribute
        # write to the live stage instead of forcing a full scene re-export.
        self.light_prims = {}  # light obj.name -> USD prim path (/World/<name>)
        self.material_prims = {}  # material.name -> UsdPreviewSurface shader prim path
        # Last value pushed per (prim, attr) so diff_attrs() emits only real changes
        # and we never reset the path-tracer for an unchanged attribute.
        self._attr_cache = {}  # (prim, attr) -> tuple(floats)

    def needs_full_sync(self, depsgraph):
        """Check if a full scene re-export is needed.

        This is the coarse gate the engine checks first each depsgraph tick.
        Returning True is expensive (the whole scene is rebuilt and re-pushed),
        so we are deliberately conservative about what counts as "topology":
        only actual geometry edits and genuinely new renderable objects qualify.
        Everything else (an object merely *moving*) is handled cheaply by
        get_transform_updates() instead.

        The hasattr/type filtering matters because depsgraph.updates is noisy:
        during physics playback or normal interaction it also emits Scene,
        World, Collection, RigidBodyWorld, and constraint datablocks. Those have
        no .type attribute (or aren't renderable objects), and treating them as
        geometry changes would force a full re-export every single frame of a
        sim — exactly the stutter this class exists to prevent.
        """
        if not self.scene_loaded:
            return True  # no live scene yet — first tick must do a full export

        for update in depsgraph.updates:
            id_data = update.id
            # Only consider geometry updates on actual objects (meshes, curves, etc.)
            # Not Scene, World, Collection, or physics constraint objects.
            # hasattr(id_data, "type") is the cheap test that excludes non-Object
            # datablocks before we inspect what kind of object it is.
            if update.is_updated_geometry and hasattr(id_data, "type"):
                # id_data.type is only present on bpy.types.Object instances.
                # These are the renderable geometry types ovrtx tessellates;
                # a real change to their topology/points means our cached USD
                # mesh is stale and only a full re-export can fix it.
                if id_data.type in ("MESH", "CURVE", "CURVES", "SURFACE", "META", "FONT"):
                    return True
            # New renderable objects require full sync — but only if they're
            # actual objects with a type, not Scene/World/RigidBodyWorld entries.
            # "Not in tracked_objects" is how we detect a never-before-seen
            # object (added to the scene, or revealed by a collection toggle).
            # We can't delta-update something we never exported, so rebuild.
            if (hasattr(id_data, "type") and hasattr(id_data, "name")
                    and id_data.name not in self.tracked_objects
                    and id_data.type in ("MESH", "CAMERA", "LIGHT",
                                         "CURVE", "CURVES", "SURFACE", "META", "FONT")):
                return True

        return False  # only transforms / non-renderable noise — delta path suffices

    def get_transform_updates(self, depsgraph):
        """Get objects with only transform changes (no geometry change).

        This is the cheap delta path, run when needs_full_sync() returned False.
        It scans depsgraph.updates for transform-only changes and translates
        each into a (USD prim path, transposed world matrix) the engine can
        poke straight into the live ovrtx stage without rebuilding geometry.

        Returns list of (prim_path, matrix_world_flat_doubles) tuples.
        Handles instances: when a base object's transform is updated,
        all tracked instances of that object are also updated by reading
        their current transforms from depsgraph.object_instances.

        The two-phase structure (direct objects first, then instances) exists
        because Blender reports a transform change once on the *base* object,
        but instances (from particle systems, geometry-node instancing,
        dupli-collections, array setups, etc.) are not their own datablocks —
        their world matrices only exist transiently in depsgraph.object_instances
        and must be re-read there. We avoid that expensive iteration unless a
        base that actually has instances was touched this tick.
        """
        updates = []
        bases_with_instances = set()  # base names whose instances need re-reading

        for update in depsgraph.updates:
            if not update.is_updated_transform:
                continue  # geometry/material/etc. changes are not our job here
            id_data = update.id
            if not hasattr(id_data, "name"):
                continue  # skip non-Object noise (Scene/World/etc.)
            name = id_data.name
            # Direct match for the base object: it was exported under this name,
            # so we just read its fresh evaluated world matrix and emit it.
            if name in self.prim_paths:
                obj = depsgraph.id_eval_get(id_data)  # evaluated copy (modifiers applied)
                if obj and hasattr(obj, "matrix_world"):
                    flat = matrix_to_row_major_flat(obj.matrix_world)
                    updates.append((self.prim_paths[name], flat))
            # Check if this base object has tracked instances. We don't resolve
            # them here (instance matrices live in object_instances, below); we
            # just note that a pass over object_instances is now warranted.
            if name in self.instance_map:
                bases_with_instances.add(name)

        # Resolve instance transforms via depsgraph.object_instances.
        # Only iterate when we know at least one base with instances was updated.
        # Instances are matched in order: depsgraph.object_instances yields
        # instances in the same order as the original export, so the Nth
        # instance of a given base object corresponds to instance_map[base][N].
        #
        # This positional matching is the load-bearing assumption: there is no
        # stable per-instance ID in Blender, so the export and this update path
        # both rely on object_instances being deterministically ordered. If that
        # order ever changes between export and update (it shouldn't, absent a
        # topology change — which would have tripped needs_full_sync), instance
        # transforms would be assigned to the wrong prims.
        if bases_with_instances:
            # Per-base counter to match instances by encounter order. Starts at 0
            # for each base and advances as we encounter that base's instances.
            inst_counters = {name: 0 for name in bases_with_instances}
            for instance in depsgraph.object_instances:
                if not instance.is_instance:
                    continue  # real (non-instanced) objects handled in phase 1
                obj = instance.object
                base_name = obj.name
                if base_name not in inst_counters:
                    continue  # this base wasn't updated this tick — ignore
                inst_list = self.instance_map[base_name]
                idx = inst_counters[base_name]  # which tracked instance this is
                if idx < len(inst_list):
                    # Guard against more live instances than we exported (e.g. a
                    # particle count grew). Extra instances are silently skipped
                    # here; a count change is topology and handled elsewhere.
                    inst_name = inst_list[idx]
                    inst_counters[base_name] = idx + 1  # advance for the next encounter
                    if inst_name in self.prim_paths:
                        # instance.matrix_world is the per-instance world transform
                        # (already composed with the instancer); transpose to USD.
                        flat = matrix_to_row_major_flat(instance.matrix_world)
                        updates.append((self.prim_paths[inst_name], flat))

        return updates

    def check_camera_changed(self, region_data):
        """Check if the viewport camera has moved since last check.

        The interactive viewport camera is *not* a Blender Camera object — it is
        whatever the user's view is currently pointing at (orbit/pan/zoom). Its
        pose comes from region_data.view_matrix, which we poll each redraw and
        diff against the last value so an idle view emits no camera transform
        value updates.

        Returns the new camera world matrix as a flat 16-double list if changed,
        or None if unchanged.
        """
        if region_data is None:
            return None  # no 3D region (e.g. headless / non-VIEW_3D context)

        view_matrix = region_data.view_matrix

        # Unchanged within tolerance -> nothing to push (see matrices_close for
        # why a plain == comparison would spuriously fire every frame).
        if self.last_view_matrix is not None and matrices_close(view_matrix, self.last_view_matrix):
            return None

        # Store a *copy*: view_matrix is a live Blender object that mutates in
        # place as the user navigates, so keeping a reference would compare a
        # value against itself next tick and never detect motion.
        self.last_view_matrix = view_matrix.copy()
        # The view matrix maps world->camera; ovrtx wants the camera's world
        # transform (camera->world), which is its inverse.
        camera_world = view_matrix.inverted()
        return matrix_to_row_major_flat(camera_world)

    def register_object(self, obj_name, prim_path, base_name=None):
        """Track an exported object for incremental updates.

        Called by the engine once per prim during a full export so this session
        knows the name<->prim mapping it will later need for delta updates.
        Crucially, instances must be registered in the *same order* they are
        yielded by depsgraph.object_instances, because get_transform_updates
        matches them back positionally (see its docstring).

        Args:
            obj_name: The tracked name (e.g. "Cube" or "Cube_inst_5").
            prim_path: USD prim path (e.g. "/World/Cube_inst_5").
            base_name: For instances, the Blender object name of the base
                       object (e.g. "Cube"). When set, transform value changes
                       for the base object will also update this instance.
        """
        self.tracked_objects[obj_name] = True  # sentinel: "we have exported this name"
        self.prim_paths[obj_name] = prim_path
        if base_name is not None:
            # Append preserves encounter order; the list index == the instance's
            # position in object_instances, which is how it is re-matched later.
            self.instance_map.setdefault(base_name, []).append(obj_name)

    def register_light(self, obj_name, prim_path):
        """Track an exported LIGHT object so a later intensity/colour edit can be
        pushed as a live attribute write instead of a full re-export."""
        self.light_prims[obj_name] = prim_path

    def register_material(self, mat_name, shader_prim_path):
        """Track an exported material's UsdPreviewSurface shader prim so a later
        diffuseColor/roughness/metallic edit can be pushed live."""
        self.material_prims[mat_name] = shader_prim_path

    def diff_attrs(self, current):
        """Diff freshly-read light/material attribute values against the last push.

        ``current`` is a dict {(prim, attr): tuple(floats)} of every tracked light/
        material attribute's CURRENT value. Returns the subset that actually changed
        since the previous call as a list of {"p": prim, "a": attr, "v": [floats]},
        and updates the cache. Emitting only real deltas means an object-move tick
        (no lookdev change) pushes zero attrs and avoids a needless path-trace reset.
        """
        out = []
        for key, vals in current.items():
            prev = self._attr_cache.get(key)
            if prev is None or len(prev) != len(vals) or any(
                    abs(a - b) > 1e-6 for a, b in zip(prev, vals)):
                self._attr_cache[key] = vals
                out.append({"p": key[0], "a": key[1], "v": list(vals)})
        return out

    def mark_synced(self):
        # Flip the gate: a full scene now exists in ovrtx, so future ticks may
        # take the cheap delta path instead of re-exporting.
        self.scene_loaded = True

    def reset(self):
        # Drop all tracking state. Called on teardown or when a full re-sync is
        # forced, so the next tick starts from scratch (scene_loaded False forces
        # a full export, and the maps are repopulated by register_object).
        self.scene_loaded = False
        self.tracked_objects.clear()
        self.prim_paths.clear()
        self.instance_map.clear()
        self.camera_prim_path = None
        self.light_prims.clear()
        self.material_prims.clear()
        self._attr_cache.clear()
        self.last_view_matrix = None
