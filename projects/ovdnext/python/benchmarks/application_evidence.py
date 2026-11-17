# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only whole-application evidence.

Serialization runs outside timing. GLX/context and viewport observers also run
inside the instrumented draw; their query overhead remains in measured latency.
"""
from collections import Counter
import ctypes
import ctypes.util
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import omni.ui as ui

from ovd_ui.design_tokens import rgba as design_rgba
from ovd_ui.render_settings import DisplayMode, FaceCulling, PlaneMode


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_libraries():
    """Actual Linux mappings, rather than assumptions from linker search paths."""
    paths = set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/') and '.so' in Path(fields[5]).name:
            path = Path(fields[5])
            if not path.is_file():
                raise RuntimeError(f'Mapped library is unavailable for provenance: {path}')
            paths.add(path.resolve())
    return {str(path): sha256(path) for path in sorted(paths)}


def display_environment():
    # Deliberate allowlist: never dump credentials or the process environment.
    names = ('DISPLAY', '__GLX_VENDOR_LIBRARY_NAME', '__GL_SYNC_TO_VBLANK',
             'vblank_mode', 'LIBGL_ALWAYS_SOFTWARE', 'GALLIUM_DRIVER',
             'QT_SCALE_FACTOR', 'QT_AUTO_SCREEN_SCALE_FACTOR')
    return {name: os.environ[name] for name in names if name in os.environ}


class GlxWitness:
    def __init__(self):
        self.library = ctypes.CDLL(ctypes.util.find_library('GL'))
        self.library.glXGetCurrentContext.restype = ctypes.c_void_p
        self.library.glXGetCurrentDisplay.restype = ctypes.c_void_p
        self.library.glXGetCurrentDrawable.restype = ctypes.c_ulong
        self.library.glXQueryDrawable.argtypes = (ctypes.c_void_p, ctypes.c_ulong,
                                                 ctypes.c_int, ctypes.POINTER(ctypes.c_uint))
        self.library.glXQueryDrawable.restype = None
        self.library.glGetIntegerv.argtypes = (ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        self.library.glGetIntegerv.restype = None
        self.library.glBindFramebuffer.argtypes = (ctypes.c_uint, ctypes.c_uint)
        self.library.glBindFramebuffer.restype = None
        self.library.glGetFramebufferAttachmentParameteriv.argtypes = (
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        self.library.glGetFramebufferAttachmentParameteriv.restype = None
        self.library.glBindTexture.argtypes = (ctypes.c_uint, ctypes.c_uint)
        self.library.glBindTexture.restype = None
        self.library.glGetTexLevelParameteriv.argtypes = (
            ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(ctypes.c_int))
        self.library.glGetTexLevelParameteriv.restype = None

    def integer(self, name):
        value = ctypes.c_int()
        self.library.glGetIntegerv(name, ctypes.byref(value))
        return value.value

    def viewport(self):
        values = (ctypes.c_int * 4)()
        self.library.glGetIntegerv(0x0BA2, values)  # GL_VIEWPORT
        return list(values)

    def geometry_attachment(self, renderer):
        # Query the actual color attachment and texture storage, restoring the
        # previous bindings. ModernGL's size/viewport properties are cached.
        gl = self.library
        previous_fbo = self.integer(0x8CA6)  # GL_DRAW_FRAMEBUFFER_BINDING
        previous_texture = self.integer(0x8069)  # GL_TEXTURE_BINDING_2D
        try:
            gl.glBindFramebuffer(0x8CA9, renderer.framebuffer.glo)  # DRAW_FRAMEBUFFER
            actual = []
            for name in (0x8CD0, 0x8CD1):  # OBJECT_TYPE, OBJECT_NAME
                value = ctypes.c_int()
                gl.glGetFramebufferAttachmentParameteriv(0x8CA9, 0x8CE0, name, ctypes.byref(value))
                actual.append(value.value)
            if actual != [0x1702, renderer.texture.glo]:  # GL_TEXTURE
                raise RuntimeError('Geometry framebuffer does not use its published color texture')
            gl.glBindTexture(0x0DE1, actual[1])  # TEXTURE_2D
            dimensions = []
            for name in (0x1000, 0x1001):  # TEXTURE_WIDTH, TEXTURE_HEIGHT
                value = ctypes.c_int()
                gl.glGetTexLevelParameteriv(0x0DE1, 0, name, ctypes.byref(value))
                dimensions.append(value.value)
            if min(dimensions) <= 0:
                raise RuntimeError('Geometry texture has no positive allocated extent')
            return dimensions
        finally:
            gl.glBindTexture(0x0DE1, previous_texture)
            gl.glBindFramebuffer(0x8CA9, previous_fbo)

    def context(self):
        context = self.library.glXGetCurrentContext()
        if not context:
            raise RuntimeError('No current GLX context; composition completion is unverified')
        return int(context)

    def drawable_size(self):
        display = self.library.glXGetCurrentDisplay()
        drawable = self.library.glXGetCurrentDrawable()
        if not display or not drawable:
            raise RuntimeError('No current GLX drawable')
        dimensions = []
        for attribute in (0x801D, 0x801E):  # GLX_WIDTH, GLX_HEIGHT
            value = ctypes.c_uint()
            self.library.glXQueryDrawable(display, drawable, attribute, ctypes.byref(value))
            dimensions.append(value.value)
        if min(dimensions) <= 0:
            raise RuntimeError('GLX drawable has no positive physical extent')
        return dimensions


def surface(app, witness, condition, observed_viewport):
    logical = [float(ui.Workspace.get_main_window_width()), float(ui.Workspace.get_main_window_height())]
    pixels = witness.drawable_size()
    if min(logical) <= 0:
        raise RuntimeError('Main UI window has no positive logical extent')
    scale = [pixel / point for pixel, point in zip(pixels, logical)]
    if not math.isclose(*scale, rel_tol=1e-6):
        raise RuntimeError(f'Nonuniform window pixel scaling is unsupported: pixels={pixels}, UI points={logical}')
    if list(ui.standalone.get_window_size()) != pixels:
        raise RuntimeError('Standalone framebuffer and GLX drawable sizes differ')
    # The global GL viewport now belongs to composition. The geometry observer
    # sampled its actual GL viewport while the renderer's target was bound.
    attachment = witness.geometry_attachment(app.renderer)
    if attachment != list(app.renderer.size) or attachment != list(app.renderer.framebuffer.size):
        raise RuntimeError('Rendered extent and geometry framebuffer differ')
    if observed_viewport != [0, 0, *attachment]:
        raise RuntimeError('Observed geometry draw viewport differs from its attachment')
    return dict(condition=condition, window_logical=logical, window_pixels=pixels,
                geometry_fbo_pixels=attachment, gl_viewport=observed_viewport,
                composition_gl_viewport=witness.viewport(),
                device_pixel_ratio=scale[0])


def effective_settings(app):
    options = app.render_settings
    if app.overlay_mask:
        raise RuntimeError('Cross-frontend evidence currently supports disabled overlays only')
    samples = app.renderer.msaa_color.samples if app.renderer.msaa_color is not None else 0
    return dict(face_culling=FaceCulling(options.face_culling).name.lower(),
                planes=PlaneMode(options.planes).name.lower(),
                background_rgba=list(options.background(design_rgba('background', app.theme))),
                geometry_scale=options.geometry_scale, plane_scale=options.plane_scale,
                particle_size=options.particle_size, show_particles=options.show_particles,
                sleeping_objects=DisplayMode(options.sleeping_objects).name.lower(),
                trigger_shapes=DisplayMode(options.trigger_shapes).name.lower(),
                wireframe=DisplayMode(options.wireframe).name.lower(),
                transparency=app.interaction.transparency_mode.name.lower(),
                transparency_alpha=min(.999, 1 - app.interaction.transparency_amount),
                actual_msaa_samples=samples,
                overlays={**dict.fromkeys(('bounds', 'contacts', 'joints', 'scene_queries',
                                           'coordinate_system', 'velocity'), 'none'),
                          'orbit_center': False})


def matrices(app):
    camera = app.navigation.camera
    return dict(view_matrix=camera.view_matrix().ravel().tolist(),
                projection_matrix=camera.projection(*app.renderer.size).ravel().tolist(),
                effective_view_projection_matrix=app.renderer._view_projection().ravel().tolist())


def _floats(values, *, nullable=False):
    result = np.asarray(values).ravel().tolist()
    if not nullable and not np.isfinite(result).all():
        raise RuntimeError('Native geometry transform or dimensions are not finite')
    return [value if math.isfinite(value) else None for value in result]


def source_geometry(snapshot, scene_id):
    """Same canonical GeometryExtractor bytes used by the Qt diagnostic object.

    Scene ownership is the only filter. Hidden/plane rows remain source evidence;
    submitted visibility is recorded independently and cannot be inferred here.
    """
    resources, lookup = {}, {}
    for mesh in snapshot.meshes:
        vertices = np.asarray(mesh.vertices, dtype='<f4')
        indices = np.asarray(mesh.indices, dtype='<u4')
        vertex_hash = hashlib.sha256(vertices.tobytes()).hexdigest()
        index_hash = hashlib.sha256(indices.tobytes()).hexdigest()
        key = f'{mesh.source_id}:{vertex_hash}:{index_hash}'
        lookup[int(mesh.id)] = key
        resources[key] = dict(source_id=int(mesh.source_id), vertex_count=len(vertices),
                              index_count=len(indices), vertices_sha256=vertex_hash,
                              indices_sha256=index_hash)
    records = []
    overrides = snapshot.override_colors
    for row in np.flatnonzero((snapshot.scene_ids == scene_id) | (snapshot.scene_ids <= 0)):
        resource_id = int(snapshot.resource_ids[row])
        resource_key = lookup.get(resource_id)
        if resource_id and resource_key is None:
            raise RuntimeError('Native geometry references an absent owned resource')
        record = dict(object_id=int(snapshot.object_ids[row]), shape_id=int(snapshot.shape_ids[row]),
                      geometry_id=int(snapshot.geometry_ids[row]), scene_id=int(snapshot.scene_ids[row]),
                      kind=int(snapshot.kinds[row]), actor_class=int(snapshot.actor_classes[row]),
                      shape_flags=int(snapshot.shape_flags[row]), actor_flags=int(snapshot.actor_flags[row]),
                      sleeping=bool(snapshot.sleeping[row]), double_sided=bool(snapshot.double_sided[row]),
                      particle_radius_mode=int(snapshot.particle_radius_modes[row]),
                      transform=_floats(snapshot.transforms[row]), dimensions=_floats(snapshot.dimensions[row]),
                      bounds=_floats(snapshot.bounds[row], nullable=True), resource_key=resource_key,
                      override_color=_floats(overrides[row]) if len(overrides) else None)
        records.append(record)
    key_fields = ('object_id', 'shape_id', 'geometry_id', 'kind')
    records.sort(key=lambda row: tuple(row[field] for field in key_fields))
    indices = Counter()
    for record in records:
        key = tuple(record[field] for field in key_fields)
        record['element_index'] = indices[key]
        indices[key] += 1
    used = {record['resource_key'] for record in records if record['resource_key'] is not None}
    diagnostics = list(snapshot.diagnostics)
    return dict(schema_version=1, complete=not diagnostics, diagnostics=diagnostics,
                records=records, resources=[dict(key=key, **resources[key]) for key in sorted(used)])


def submitted_geometry(renderer):
    snapshot = renderer.snapshot
    rows = renderer._visible_rows
    groups = renderer._groups
    memberships = renderer._group_rows
    if len(groups) != len(memberships):
        raise RuntimeError('Native group membership differs from actual submitted groups')
    resources = {int(mesh.id): mesh for mesh in snapshot.meshes}
    uploaded_rows = []
    for (kind, mesh), (expected_kind, resource_id, flags, members) in zip(groups, memberships):
        resource = resources.get(resource_id)
        cache_key = (expected_kind, int(resource.topology_id) if resource is not None else resource_id, flags)
        if (kind != expected_kind or renderer.meshes.get(cache_key) is not mesh
                or mesh.count != len(members)):
            raise RuntimeError('Submitted GPU object does not match its native source-row group')
        if resource is not None and mesh.revision != resource_id:
            raise RuntimeError('Submitted GPU object retains a different native vertex revision')
        uploaded_rows.extend(int(row) for row in members)
    if sorted(uploaded_rows) != sorted(int(row) for row in rows):
        raise RuntimeError('Native rows were omitted or duplicated in submitted draw groups')
    shapes = sorted([int(snapshot.object_ids[row]), int(snapshot.shape_ids[row]),
                     int(snapshot.geometry_ids[row]), int(snapshot.kinds[row])] for row in uploaded_rows)
    groups = [dict(kind=kind, instances=mesh.count, vertex_count=mesh.vertices.size // 24,
                   index_count=mesh.indices.size // 4 if mesh.indices is not None else 0,
                   transparent=mesh.transparent, double_sided=mesh.double_sided,
                   wireframe=mesh.wireframe) for kind, mesh in renderer._groups]
    return dict(submitted_native_ids=sorted({shape[0] for shape in shapes}),
                submitted_shape_ids=sorted({shape[1] for shape in shapes if shape[1] > 0}),
                submitted_geometry=dict(schema_version=1, complete=True,
                    mapping_supported=True, diagnostics=[],
                    method='Actual submitted GPU groups map to native source-row memberships through kind, cache object identity, instance count and vertex revision; flattened membership equals visible rows including multiplicity.'),
                visible_shape_records=shapes, geometry_records=groups)
