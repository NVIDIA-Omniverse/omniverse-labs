# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable render visibility, independent of Blender object/geometry lifetime."""
from contextlib import contextmanager


GEOMETRY_TYPES = frozenset({'MESH', 'CURVE', 'SURFACE', 'FONT', 'META', 'CURVES', 'POINTCLOUD', 'VOLUME'})
PARTICLE_EMITTER_NAME_ATTRIBUTE = 'ovrtx:particle_emitter_name'
HIDE_FOR_CAMERA_PRIMVAR = 'hideForCamera'


def camera_hidden(obj):
    """True for geometry whose Blender camera ray visibility is off."""
    return (str(getattr(obj, 'type', '')) in GEOMETRY_TYPES
            and getattr(obj, 'visible_camera', True) is False)


class VisibilityWatch:
    """Coalesce visibility notifications; unchanged polls never visit objects.

    RNA covers restrictions that need not tag a depsgraph ID; the depsgraph
    covers hide_set, membership and evaluated geometry changes. Layer identity
    is checked cheaply at consumption. Older/incomplete hosts keep polling.
    """
    def __init__(self, scene, module):
        self.scene, self.module = scene, module
        self.dirty, self.active, self.layer = True, False, None
        self.owner = object()
        # Blender's RNA bus requires a Python function, not a bound method.
        def notify():
            self.invalidate()
        def handler(scene, depsgraph):
            self._depsgraph_changed(scene, depsgraph)
        self.notify, self.handler = notify, handler
        self.handlers = getattr(getattr(getattr(module, 'app', None), 'handlers', None),
                                'depsgraph_update_post', None)
        bus = getattr(module, 'msgbus', None)
        if not callable(getattr(bus, 'subscribe_rna', None)) or self.handlers is None:
            return
        try:
            for type_name, names in (
                ('Object', ('hide_render', 'hide_viewport', 'visible_shadow', 'visible_camera')),
                ('Collection', ('hide_render', 'hide_viewport')),
                ('LayerCollection', ('exclude', 'hide_viewport')),
            ):
                rna_type = getattr(module.types, type_name)
                for name in names:
                    bus.subscribe_rna(key=(rna_type, name), owner=self.owner,
                                      args=(), notify=self.notify)
            self.handlers.append(self.handler)
            self.active = True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            self.close()

    def invalidate(self):
        self.dirty = True

    def _depsgraph_changed(self, scene, depsgraph):
        if scene != self.scene:
            return
        for update in getattr(depsgraph, 'updates', ()):
            value = getattr(update, 'id', update)
            value = getattr(value, 'original', value)
            kind = str(getattr(value, 'type', ''))
            identifier = str(getattr(getattr(value, 'bl_rna', None), 'identifier', ''))
            # A camera move/projection edit has no visibility consequence.
            # Dependent meshes/collections still arrive as their own updates.
            if kind == 'CAMERA' or identifier == 'Camera':
                continue
            self.invalidate()
            break

    def pending(self):
        layer = scene_view_layer(self.scene, self.module)
        if layer != self.layer:
            self.layer = layer
            self.invalidate()
        return self.dirty or not self.active

    def close(self):
        bus = getattr(self.module, 'msgbus', None)
        clear = getattr(bus, 'clear_by_owner', None)
        if callable(clear):
            clear(self.owner)
        if self.handlers is not None and self.handler in self.handlers:
            self.handlers.remove(self.handler)
        self.active = False


def _visibility_owner(obj):
    kind = str(getattr(obj, 'type', ''))
    return kind in GEOMETRY_TYPES | {'LIGHT'} or (
        kind == 'EMPTY' and getattr(obj, 'instance_collection', None) is not None)


def scene_view_layer(scene, module=None, view_layer=None):
    """Resolve an owning active layer, never an arbitrary layer of another scene."""
    if view_layer is not None:
        layers = tuple(getattr(scene, "view_layers", ()))
        if layers and not any(layer == view_layer for layer in layers):
            raise ValueError("The requested view layer does not belong to this scene")
        return view_layer
    if module is None:
        try:
            import bpy as module
        except ModuleNotFoundError:
            pass
    context = getattr(module, 'context', None)
    if getattr(context, 'scene', None) == scene:
        return getattr(context, 'view_layer', None)
    windows = getattr(getattr(context, 'window_manager', None), 'windows', ())
    matches = [window.view_layer for window in windows if window.scene == scene]
    if matches and all(layer == matches[0] for layer in matches):
        return matches[0]
    layers = tuple(getattr(scene, 'view_layers', ()))
    if len(layers) > 1:
        raise ValueError('An explicit owning view layer is required for this scene')
    return next(iter(layers), None)


def visible(obj, view_layer=None):
    if not _visibility_owner(obj):
        return True
    if getattr(obj, 'hide_render', False) or getattr(obj, 'hide_viewport', False):
        return False
    getter = getattr(obj, 'visible_get', None)
    if not callable(getter):
        return True
    return bool(getter(view_layer=view_layer)) if view_layer is not None else bool(getter())


def scene_visibility(scene, view_layer=None):
    # A collection restriction is inherited along each membership path; an
    # object linked through any render-enabled path remains render-enabled.
    render_enabled = set()
    def visit(collection, hidden=False):
        hidden = hidden or bool(getattr(collection, 'hide_render', False))
        if not hidden:
            render_enabled.update(int(obj.session_uid) for obj in collection.objects)
        for child in collection.children:
            visit(child, hidden)
    root = getattr(scene, 'collection', None)
    if root is not None:
        visit(root)
    view_layer = scene_view_layer(scene, view_layer=view_layer)
    return tuple((obj, visible(obj, view_layer) and (root is None or int(obj.session_uid) in render_enabled))
                 for obj in getattr(scene, 'objects', ()) if _visibility_owner(obj))


@contextmanager
def export_visibility(module, scene, view_layer=None):
    """Evaluate hidden geometry with stock Blender, then restore every restriction.

    Only generation-preparation timers may enter this context, never drawing.
    No USD visibility is derived from this temporary state.
    """
    view_layer = scene_view_layer(scene, module, view_layer)
    changes, eyes, drivers = [], [], []
    def clear(value, name):
        original = bool(getattr(value, name, False))
        visibility_drivers = tuple(curve for curve in getattr(
            getattr(value, 'animation_data', None), 'drivers', ())
            if name in {'hide_render', 'hide_viewport', 'hide_select'}
            and getattr(curve, 'data_path', None) == name)
        # Dependency evaluation must not immediately reapply a visibility
        # driver while stock export evaluates the temporarily revealed mesh.
        # Save even previously muted curves so nested contexts are reversible.
        for curve in visibility_drivers:
            drivers.append((curve, curve.mute))
            curve.mute = True
        if original or visibility_drivers:
            changes.append((value, name, original))
            setattr(value, name, False)
    def collections(collection):
        clear(collection, 'hide_viewport')
        clear(collection, 'hide_render')
        clear(collection, 'hide_select')
        for child in getattr(collection, 'children', ()):
            collections(child)
    def layers(layer):
        clear(layer, 'exclude')
        clear(layer, 'hide_viewport')
        for child in getattr(layer, 'children', ()):
            layers(child)
    try:
        collections(getattr(scene, 'collection', None))
        layers(getattr(view_layer, 'layer_collection', None))
        for obj in getattr(scene, 'objects', ()):
            clear(obj, 'hide_viewport')
            clear(obj, 'hide_render')
            clear(obj, 'hide_select')
            getter, setter = getattr(obj, 'hide_get', None), getattr(obj, 'hide_set', None)
            hidden = getter(view_layer=view_layer) if callable(getter) and view_layer is not None else (getter() if callable(getter) else False)
            if callable(setter) and hidden:
                eyes.append(obj)
                setter(False, view_layer=view_layer) if view_layer is not None else setter(False)
        if (changes or eyes) and callable(getattr(view_layer, 'update', None)):
            view_layer.update()
        yield
    finally:
        for obj in reversed(eyes):
            obj.hide_set(True, view_layer=view_layer) if view_layer is not None else obj.hide_set(True)
        for value, name, original in reversed(changes):
            setattr(value, name, original)
        for curve, muted in reversed(drivers):
            curve.mute = muted
        if (changes or eyes) and callable(getattr(view_layer, 'update', None)):
            view_layer.update()


def opinion_records(scene, mappings, blender_id, stock_usd_path=None, *, stock_stage_provider=None):
    from pxr import Sdf, Usd, UsdGeom
    from .usd_opinion_records import AddOnUsdOpinionRecord
    records = []
    stock = None
    particle_paths = None
    independent = {str(value.object_path) for value in mappings.values()}
    for obj, is_visible in scene_visibility(scene):
        mapping = mappings.get(blender_id(obj, 'OBJECT'))
        if mapping is None:
            continue
        paths = [mapping.schema_path]
        if getattr(obj, 'particle_systems', ()):
            if stock is None:
                stock = (stock_stage_provider() if stock_stage_provider is not None
                         else (Usd.Stage.Open(str(stock_usd_path)) if stock_usd_path else None))
            if stock is None:
                raise ValueError('Particle visibility requires exported stage identity')
            if particle_paths is None:
                particle_paths = {}
                for prim in stock.Traverse():
                    attribute = prim.GetAttribute(PARTICLE_EMITTER_NAME_ATTRIBUTE)
                    owner_name = attribute.Get() if attribute else None
                    if owner_name is not None:
                        particle_paths.setdefault(str(owner_name), []).append(str(prim.GetPath()))
            paths.extend(particle_paths.get(str(obj.name_full), ()))
        if str(getattr(obj, "type", "")) == "EMPTY" or mapping.schema_path == mapping.object_path and str(getattr(obj, "type", "")) in GEOMETRY_TYPES:
            if stock is None:
                stock = (stock_stage_provider() if stock_stage_provider is not None
                         else (Usd.Stage.Open(str(stock_usd_path)) if stock_usd_path else None))
            if stock is None:
                raise ValueError("Collection instance visibility requires exported stage identity")
            owner = stock.GetPrimAtPath(mapping.object_path)
            paths = [str(child.GetPath()) for child in owner.GetChildren()
                     if str(child.GetPath()) not in independent]
        for path in paths:
            layer = Sdf.Layer.CreateAnonymous('.usda')
            stage = Usd.Stage.Open(layer)
            prim = stage.OverridePrim(path)
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(
                UsdGeom.Tokens.inherited if is_visible else UsdGeom.Tokens.invisible)
            if camera_hidden(obj):
                # Cycles "Ray Visibility > Camera" off: the object still lights and
                # shadows the scene but is not seen by primary rays.
                UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                    HIDE_FOR_CAMERA_PRIMVAR, Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
            records.append(AddOnUsdOpinionRecord(path, layer,
                {'source': 'blender_object_visibility', 'blender_id': str(obj.name_full),
                 'usd_prim_path': path}))
    return tuple(records)


def merge_records(records):
    """Keep one opinion identity per prim, including matte + visibility edits."""
    from pxr import Sdf, Usd
    from .usd_opinion_records import AddOnUsdOpinionRecord
    grouped = {}
    for record in records:
        grouped.setdefault(record.usd_prim_path, []).append(record)
    result = []
    for path, group in sorted(grouped.items()):
        if len(group) == 1:
            result.append(group[0])
            continue
        layers = []
        for record in group:
            layer = Sdf.Layer.CreateAnonymous('.usda')
            layer.ImportFromString(record.layer_text)
            layers.append(layer)
        root = Sdf.Layer.CreateAnonymous('.usda')
        root.subLayerPaths = [layer.identifier for layer in layers]
        result.append(AddOnUsdOpinionRecord(path, Usd.Stage.Open(root).Flatten(),
            {'source': 'combined_addon_opinions', 'sources': tuple(dict(r.diagnostics) for r in group)}))
    return tuple(result)
