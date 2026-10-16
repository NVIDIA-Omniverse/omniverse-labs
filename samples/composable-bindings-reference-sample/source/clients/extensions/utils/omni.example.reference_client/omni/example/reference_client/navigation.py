# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Viewport sizing and explicit camera actions, independent of telemetry."""

from pxr import Gf, UsdGeom
from omni.kit.actions.core import get_action_registry
import omni.kit.commands
from omni.kit.hotkeys.core import HotkeyFilter, get_hotkey_registry


class ViewportNavigation:
    ACTION = "frame_selection"

    def __init__(self, viewport_window, ext_id, visible_paths, overlay_points):
        self.viewport = viewport_window.viewport_api
        self.visible_paths = visible_paths
        self.overlay_points = overlay_points
        self.owner = ext_id + ".navigation"
        self.previous_fill = self.viewport.fill_frame
        # A fixed 16:9 texture otherwise sits inside the wide docked canvas.
        # SceneView labels can extend into its side margins, unlike tube meshes.
        self.viewport.fill_frame = True
        get_action_registry().register_action(
            self.owner, self.ACTION, self.frame_selection,
            display_name="Frame selection", description="Frame selected prims; frame visible machines when empty")
        self.hotkey = get_hotkey_registry().register_hotkey(
            self.owner, "F", self.owner, self.ACTION,
            filter=HotkeyFilter(windows=[viewport_window.name]))

    def frame_selection(self):
        # Read Kit's current selection at invocation, including multi-selection.
        # Kit handles world bounds, camera aspect and undo. With no selection,
        # frame visible asset geometry, excluding camera/light helper bounds.
        paths = self.viewport.usd_context.get_selection().get_selected_prim_paths()
        return self._frame(paths or self.visible_paths())

    def frame_visible(self):
        return self._frame(self.visible_paths())

    def _frame(self, paths):
        if not paths or not self.viewport.stage or not self.viewport.camera_path:
            return False
        width, height = self.viewport.resolution
        if not width or not height:
            return False
        stage = self.viewport.stage
        cache = UsdGeom.BBoxCache(self.viewport.time, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        points = []
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim: continue
            bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not bounds.IsEmpty(): points.extend(bounds.GetCorner(i) for i in range(8))
        points.extend(self.overlay_points(paths))
        if not points: return False
        camera = stage.GetPrimAtPath(self.viewport.camera_path)
        if UsdGeom.Camera(camera).GetProjectionAttr().Get() == UsdGeom.Tokens.orthographic:
            success, _ = omni.kit.commands.execute('FramePrimsCommand', prim_to_move=self.viewport.camera_path,
                prims_to_frame=paths, usd_context_name=self.viewport.usd_context_name, aspect_ratio=width/height, zoom=1.25)
            return success
        # Include the camera-facing gauge cards in transient bounds, never USD.
        # Fit against the actual conformed projection, including docked aspect ratio.
        view = self.viewport.view
        bounds = Gf.Range3d()
        for point in points: bounds.UnionWith(view.Transform(point))
        center = bounds.GetMidpoint()
        proj = self.viewport.projection
        distance = max(max(abs((p-center)[0])*abs(proj[0][0]),
                           abs((p-center)[1])*abs(proj[1][1])) + (p-center)[2]
                       for p in (view.Transform(point) for point in points)) * 1.18
        world = view.GetInverse()
        target = world.Transform(center)
        eye = target + world.TransformDir(Gf.Vec3d(0,0,1)).GetNormalized() * max(distance, .1)
        world.SetTranslateOnly(eye)
        parent = UsdGeom.XformCache(self.viewport.time).GetParentToWorldTransform(camera)
        success, _ = omni.kit.commands.execute('TransformPrimCommand', path=self.viewport.camera_path,
            new_transform_matrix=world * parent.GetInverse(), time_code=self.viewport.time,
            usd_context_name=self.viewport.usd_context_name)
        return success

    def close(self):
        get_hotkey_registry().deregister_hotkey(self.hotkey)
        get_action_registry().deregister_action(self.owner, self.ACTION)
        self.viewport.fill_frame = self.previous_fill
