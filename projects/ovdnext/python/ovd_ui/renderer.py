# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Instanced ModernGL viewport sharing OVUI's desktop GL texture.

Native snapshots own geometry and poses; the renderer uploads packed instance
arrays and caches mesh buffers. Picking renders IDs only on a click, reading one
pixel. Normal viewport presentation never reads pixels back to the CPU.
"""
import ctypes
from dataclasses import dataclass, replace
import math
import sys

import moderngl
import numpy as np
import omni.ui as ui
from .overlays import OverlayRenderer
from .camera import Camera
from .orbit_center import orbit_sphere, orbit_transform
from .design_tokens import STAGE_GRADIENT
from .render_settings import DisplayMode, FaceCulling, PlaneMode, RenderingSettings, display_mask
from .geometry_visibility import geometry_visibility


VERTEX = '''#version 330
uniform mat4 view_projection;
uniform int kind;
in vec3 position;
in vec3 normal;
in mat4 instance_matrix;
in vec3 dimensions;
in vec4 instance_color;
in uint instance_id;
out vec3 world_normal;
out vec3 world_position;
out vec4 tint;
out vec2 plane_coordinate;
flat out uint object_id;
void main() {
    vec3 p = position;
    if (kind == 1) p *= dimensions.x;
    else if (kind == 2 || kind == 4) p *= dimensions;
    else if (kind == 3) {
        p *= dimensions.x;
        p.x += sign(position.x) * dimensions.y;
    }
    world_normal = mat3(transpose(inverse(instance_matrix))) * normal;
    world_position = (instance_matrix * vec4(p, 1.0)).xyz;
    plane_coordinate = p.yz;
    tint = instance_color;
    object_id = instance_id;
    gl_Position = view_projection * instance_matrix * vec4(p, 1.0);
    gl_Position.y *= -1.0;
}
'''
FRAGMENT = '''#version 330
uniform bool picking;
uniform bool transparent_pass;
uniform int kind;
uniform float grid_scale;
uniform vec3 ground_tint;
uniform float grid_contrast;
uniform vec4 grid_rgba;
uniform bool subdued_grid;
uniform vec3 mesh_light_direction;
in vec3 world_normal;
in vec3 world_position;
in vec4 tint;
in vec2 plane_coordinate;
flat in uint object_id;
layout(location=0) out vec4 color;
layout(location=1) out float reveal_log;
void main() {
    reveal_log = 0.0;
    if (tint.a <= 0.0) discard;
    if (picking) {
        uint id = object_id + 1u;
        color = vec4(float(id & 255u), float((id >> 8u) & 255u),
                     float((id >> 16u) & 255u), float((id >> 24u) & 255u)) / 255.0;
    } else if (kind == 4) {
        vec2 coordinate = plane_coordinate / grid_scale;
        vec2 grid = abs(fract(coordinate - 0.5) - 0.5) / max(fwidth(coordinate), vec2(0.00001));
        float line = 1.0 - min(min(grid.x, grid.y), 1.0);
        if (subdued_grid) {
            // Fade each family separately when its spacing approaches a pixel.
            // A scalar min/max fade leaves the other family aliasing at grazing angles.
            vec2 footprint = fwidth(coordinate);
            vec2 visibility = 1.0 - smoothstep(vec2(0.18), vec2(0.8), footprint);
            vec2 lines = (1.0 - min(grid, vec2(1.0))) * visibility;
            line = max(lines.x, lines.y);
        }
        vec3 ground_color = ground_tint + line * grid_contrast;
        if (grid_rgba.a > 0.0)
            ground_color = mix(ground_tint, grid_rgba.rgb, line * grid_rgba.a);
        if (any(greaterThan(tint.rgb, vec3(0.0))))
            ground_color = mix(ground_color, clamp(tint.rgb, 0.0, 1.0), 0.45);
        color = vec4(ground_color, tint.a);
    } else {
        vec3 n = normalize(world_normal);
        if (kind == 5) {
            // Recorded triangles retain their actual facets, including shared
            // mesh/heightfield vertices. The Qt renderer uses the same surface
            // derivative; analytic primitives keep their interpolated normals.
            vec3 face_normal = cross(dFdx(world_position), dFdy(world_position));
            float face_length_squared = dot(face_normal, face_normal);
            if (face_length_squared > 0.0)
                n = face_normal * inversesqrt(face_length_squared);
        }
        float light = kind == 5 ? sqrt(abs(dot(n, normalize(mesh_light_direction))))
                               : 0.34 + 0.62 * abs(dot(n, normalize(vec3(0.4, 0.8, 0.5))));
        float rim = kind == 5 ? 0.0 : pow(1.0 - abs(n.z), 3.0) * 0.08;
        color = vec4(tint.rgb * light + rim, tint.a);
    }
    if (!picking && transparent_pass) {
        // Same McGuire-Bavoil weight as the native reference viewer.
        float weight = clamp(pow(min(1.0, color.a * 10.0) + 0.01, 3.0) * 1e3 *
                             pow(1.0 - gl_FragCoord.z * 0.9, 3.0), 1e-2, 3e3);
        // Additive optical depth is equivalent to multiplicative revealage.
        // It permits one MRT pass with ModernGL's global ONE/ONE blend state.
        reveal_log = -log(max(1.0 - color.a, 1e-6));
        color = vec4(color.rgb * color.a, color.a) * weight;
    }
}
'''

COMPOSITE_VERTEX = """#version 330
out vec2 uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""
BACKGROUND_FRAGMENT = """#version 330
uniform vec3 inner_color;
uniform vec3 outer_color;
uniform vec2 center;
uniform vec2 radii;
uniform float outer_stop;
in vec2 uv;
out vec4 color;
void main() {
    float distance = length((uv - center) / radii);
    color = vec4(mix(inner_color, outer_color, clamp(distance / outer_stop, 0.0, 1.0)), 1.0);
}
"""

COMPOSITE_FRAGMENT = """#version 330
uniform sampler2D accumulation;
uniform sampler2D optical_depth;
in vec2 uv;
out vec4 color;
void main() {
    vec4 sum = texture(accumulation, uv);
    float coverage = 1.0 - exp(-texture(optical_depth, uv).r);
    if (coverage < 1e-6) discard;
    color = vec4(sum.rgb / max(sum.a, 1e-5), coverage);
}
"""

PARTICLE_VERTEX = """#version 330
in vec3 position;
in mat4 instance_matrix;
in vec3 dimensions;
in vec4 instance_color;
in uint instance_id;
out vec4 particle_color;
out float particle_radius;
flat out uint particle_id;
void main() {
    gl_Position = instance_matrix * vec4(position, 1.0);
    particle_radius = dimensions.x * length(instance_matrix[0].xyz);
    particle_color = instance_color;
    particle_id = instance_id;
}
"""
PARTICLE_GEOMETRY = """#version 330
layout(points) in;
layout(triangle_strip, max_vertices=4) out;
uniform mat4 view_projection;
uniform vec3 eye;
uniform vec3 up;
in vec4 particle_color[];
in float particle_radius[];
flat in uint particle_id[];
out vec2 coordinate;
out vec4 tint;
flat out uint object_id;
void corner(vec3 center, vec3 offset, vec2 uv) {
    gl_Position = view_projection * vec4(center + offset * particle_radius[0], 1.0);
    gl_Position.y *= -1.0;
    coordinate = uv; tint = particle_color[0]; object_id = particle_id[0];
    EmitVertex();
}
void main() {
    if (particle_radius[0] <= 0.0) return;
    vec3 center = gl_in[0].gl_Position.xyz;
    vec3 vertical = normalize(up);
    vec3 right = normalize(cross(normalize(center - eye), vertical));
    corner(center, vertical, vec2(1,0));
    corner(center, right, vec2(0,1));
    corner(center, -right, vec2(0,-1));
    corner(center, -vertical, vec2(-1,0));
    EndPrimitive();
}
"""
PARTICLE_FRAGMENT = """#version 330
uniform bool picking;
uniform bool transparent_pass;
in vec2 coordinate;
in vec4 tint;
flat in uint object_id;
layout(location=0) out vec4 color;
layout(location=1) out float reveal_log;
void main() {
    float d = dot(coordinate, coordinate);
    if (d > 0.5 || tint.a <= 0.0) discard;
    reveal_log = 0.0;
    if (picking) {
        uint id = object_id + 1u;
        color = vec4(float(id & 255u),float((id>>8u)&255u),float((id>>16u)&255u),float((id>>24u)&255u))/255.0;
    } else {
        color = vec4(tint.rgb, tint.a * mix(0.5,1.0,0.5-d));
        if (transparent_pass) {
            float weight = clamp(pow(min(1.0,color.a*10.0)+0.01,3.0)*1e3*
                pow(1.0-gl_FragCoord.z*0.9,3.0),1e-2,3e3);
            reveal_log = -log(max(1.0-color.a,1e-6));
            color = vec4(color.rgb*color.a,color.a)*weight;
        }
    }
}
"""

# Matrix input consumes four attribute locations. Struct layout is packed to
# match ModernGL's instanced vertex format, including the integer picking ID.
INSTANCE_DTYPE = np.dtype([("matrix", "f4", (4, 4)), ("dimensions", "f4", 3),
                           ("color", "f4", 4), ("id", "u4")])


def unit_sphere():
    vertices = []
    for lat in range(24):
        for lon in range(40):
            points = []
            for a, b in ((lat, lon), (lat+1, lon), (lat+1, lon+1), (lat, lon+1)):
                theta, phi = math.pi*a/24, 2*math.pi*b/40
                # X axis hemispheres also support PhysX capsules.
                points.append((math.cos(theta), math.sin(theta)*math.cos(phi), math.sin(theta)*math.sin(phi)))
            for i in (0, 1, 2, 0, 2, 3):
                vertices.extend((*points[i], *points[i]))
    return np.asarray(vertices, dtype="f4").reshape(-1, 6)


def unit_box():
    vertices = []
    for axis in range(3):
        for sign in (-1, 1):
            points=[]
            normal=[0., 0., 0.]; normal[axis]=sign
            for a, b in ((-1,-1), (1,-1), (1,1), (-1,1)):
                p=[0., 0., 0.]; p[axis]=sign; p[(axis+1)%3]=a; p[(axis+2)%3]=b
                points.append(p)
            for i in ((0, 1, 2, 0, 2, 3) if sign > 0 else (0, 2, 1, 0, 3, 2)):
                vertices.extend((*points[i], *normal))
    return np.asarray(vertices, dtype="f4").reshape(-1, 6)


def unit_plane():
    return np.array([[0,y,z,1,0,0] for y,z in
                     ((-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1))], dtype="f4")


def mesh_data(vertices, indices, kind):
    vertices = np.asarray(vertices, dtype="f4")
    indices = np.asarray(indices, dtype="u4")
    normals = np.zeros_like(vertices)
    if kind == 5 and len(indices):
        triangles = indices.reshape(-1, 3)
        points = vertices[triangles]
        face_normals = np.cross(points[:,1]-points[:,0], points[:,2]-points[:,0])
        for column in range(3):
            np.add.at(normals, triangles[:,column], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:,None], 1e-20)
    else:
        normals[:,1] = 1
    return np.ascontiguousarray(np.concatenate((vertices, normals), axis=1)), indices


def mesh_data_batch(entries):
    """Run the same normal calculation over disjoint, bounded small meshes."""
    offsets = np.cumsum([0] + [len(vertices) for _, vertices, _ in entries])
    vertices = np.concatenate([vertices for _, vertices, _ in entries])
    indices = np.concatenate([indices + np.uint32(offset)
                              for (_, _, indices), offset in zip(entries, offsets)])
    data, _ = mesh_data(vertices, indices, 5)
    return {key: (data[start:stop], original_indices)
            for (key, _, original_indices), start, stop in zip(entries, offsets, offsets[1:])}


def look_at(eye, target, up=(0.,1.,0.)):
    forward = target-eye
    forward /= max(np.linalg.norm(forward), 1e-20)
    side = np.cross(forward, up)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(forward, [0.,0.,1.])
    side /= max(np.linalg.norm(side), 1e-20)
    up = np.cross(side, forward)
    view = np.eye(4, dtype="f4")
    view[:3,:3] = [side,up,-forward]
    view[:3,3] = -view[:3,:3] @ eye
    return view


@dataclass
class GpuMesh:
    vertices: object
    indices: object
    instances: object
    vao: object
    count: int = 0
    revision: int = 0
    transparent: bool = False
    double_sided: bool = False
    wireframe: bool = False

    def close(self):
        for resource in (self.vao,self.instances,self.indices,self.vertices):
            if resource is not None:
                resource.release()


class _LinuxOpenGLLoader:
    """Attach to OvUI's current context using runtime, not development libraries."""

    def __init__(self):
        failures = []
        for soname, prefix in (("libEGL.so.1", "egl"), ("libGL.so.1", "glX")):
            try:
                library = ctypes.CDLL(soname)
            except OSError as error:
                failures.append(f"{soname}: {error}")
                continue
            current = getattr(library, prefix + "GetCurrentContext")
            current.argtypes = []
            current.restype = ctypes.c_void_p
            if not current():
                failures.append(f"{soname}: no current {prefix} context")
                continue
            self.library = library
            self.proc = getattr(library, prefix + "GetProcAddress")
            self.proc.argtypes = [ctypes.c_char_p]
            self.proc.restype = ctypes.c_void_p
            return
        raise RuntimeError("Cannot attach to the current OpenGL context: " + "; ".join(failures))

    def load_opengl_function(self, name):
        return self.proc(name.encode()) or ctypes.cast(getattr(self.library, name, 0), ctypes.c_void_p).value or 0

    # OvUI owns the native context; ModernGL must not release or switch it.
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def release(self):
        pass


class ViewportRenderer:
    def __init__(self, context=None, provider=None):
        if context is None and sys.platform == "linux":
            context = moderngl.create_context(require=330, context=_LinuxOpenGLLoader())
        self.context = context or moderngl.create_context(require=330)
        self.program = self.context.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
        self.particle_program = self.context.program(vertex_shader=PARTICLE_VERTEX,
            geometry_shader=PARTICLE_GEOMETRY, fragment_shader=PARTICLE_FRAGMENT)
        self.composite = self.context.program(vertex_shader=COMPOSITE_VERTEX,fragment_shader=COMPOSITE_FRAGMENT)
        self.composite_vao = self.context.vertex_array(self.composite,[])
        self.background_program = self.background_vao = None
        self.composite["accumulation"].value=0
        self.composite["optical_depth"].value=1
        self.accumulation=self.optical_depth=self.transparent_buffer=self.transparent_clear=None
        self.meshes = {}
        self.overlays=OverlayRenderer(self.context)
        self.query_renderer=None
        self.provider = provider if provider is not None else ui.ImageProvider()
        self.framebuffer = self.texture = self.depth = None
        self.pick_buffer = self.pick_texture = None
        self.size = (0,0)
        self.snapshot = None
        self.camera = Camera()
        self.camera.reset()
        self.wireframe = False
        self.selected = set()
        self.hidden = set()
        self.scene_id = None
        self.show_static = self.show_dynamic = True
        self.opacity = 1.
        self.all_transparent = False
        self.alpha_by_id = {}
        self.instance_alpha = None
        self.device = self.context.info["GL_RENDERER"]
        self.draw_count = 0
        self.instance_count = 0
        self._groups = []
        self._group_signature = None
        self._group_rows = []
        self._appearance = None
        self.light = False
        self.palette = None
        self.actor_class_names = {}
        self.ground_color = None
        self.grid_contrast = .06
        self.grid_rgba = (0., 0., 0., 0.)
        self.settings = None  # Old benchmark path until the owner applies settings.
        self._geometry_settings = None
        self._orbit_mesh = None
        self._orbit_payload = None
        self.msaa_buffer = self.msaa_color = self.msaa_depth = None
        self.samples = 0
        self._attachment_samples = 0

    def set_settings(self, settings):
        if not isinstance(settings, RenderingSettings):
            raise TypeError("Expected validated RenderingSettings")
        had_particles = self.settings is not None
        self.settings = settings
        geometry_settings = replace(settings, draw_orbit_center=False)
        self.samples = min(8, int(self.context.max_samples)) if settings.anti_aliasing else 0
        if self.samples < 2:
            self.samples = 0
        if not had_particles:
            # Point VAOs change program when physical billboard rendering starts.
            for key in [key for key in self.meshes if key[0] == 7]:
                self.meshes.pop(key).close()
        if geometry_settings != self._geometry_settings:
            self._appearance = None
        self._geometry_settings = geometry_settings

    def sample_info(self):
        if self.samples and self._attachment_samples != self.samples:
            return f"MSAA: {self.samples} samples requested; attachments update on the next frame."
        return (f"MSAA: {self.msaa_color.samples} samples (hardware maximum {self.context.max_samples})."
                if self.msaa_color is not None else f"MSAA off (hardware maximum {self.context.max_samples}).")

    def set_palette(self, palette, classes=()):
        """Bind the independent palette and this capture's native class names."""
        self.palette = palette
        self.actor_class_names = dict(classes) if isinstance(classes, dict) else {int(cls.id): cls.name for cls in classes}
        self.overlays.palette = palette
        self._appearance = None

    def _mesh(self, key, data, indices=None):
        if key not in self.meshes:
            vb = self.context.buffer(data.tobytes())
            ib = self.context.buffer(indices.tobytes()) if indices is not None and len(indices) else None
            instances = self.context.buffer(reserve=INSTANCE_DTYPE.itemsize, dynamic=True)
            particles = key[0] == 7 and self.settings is not None
            program = self.particle_program if particles else self.program
            vao = self.context.vertex_array(program,
                [(vb,"3f 12x","position") if particles else (vb,"3f 3f","position","normal"),
                 (instances,"16f 3f 4f 1u /i","instance_matrix","dimensions","instance_color","instance_id")], ib)
            self.meshes[key] = GpuMesh(vb, ib, instances, vao)
        return self.meshes[key]

    def clear(self):
        self.snapshot = None
        self.overlays.set_snapshot(None)
        if self.query_renderer:
            self.query_renderer.clear()
        self._groups = []
        self._group_signature = None
        self._group_rows = []
        self.selected.clear()
        self.hidden.clear()
        self.instance_count = 0
        self.instance_alpha=None
        for mesh in self.meshes.values():
            mesh.close()
        self.meshes.clear()

    def set_overlays(self,snapshot):
        self.overlays.set_snapshot(snapshot)
        query=snapshot.query_geometry if snapshot is not None and snapshot.ready else None
        if query is not None and len(query.object_ids):
            if self.query_renderer is None:
                self.query_renderer=ViewportRenderer(context=self.context)
            self.query_renderer.set_snapshot(query)
            self.query_renderer.wireframe=True
        elif self.query_renderer is not None:
            self.query_renderer.clear()

    def set_snapshot(self, snapshot, reset_camera=False):
        self.snapshot = snapshot
        self._appearance = None
        self._prepare()
        if reset_camera:
            self.frame_all()

    def _prepare_mesh_batch(self, start, resources):
        # Keep transient vertex/triangle arrays bounded. A large individual
        # mesh uses the existing one-mesh path, preserving its memory behavior.
        entries = []
        vertex_count = index_count = 0
        for group_index in range(start, len(self._group_rows)):
            kind, resource, transparent, _ = self._group_rows[group_index]
            if kind != 5:
                continue
            source = resources.get(resource)
            if source is None:
                continue
            mesh = self.meshes.get((kind, int(source.topology_id), transparent))
            if mesh is not None and mesh.revision == resource:
                continue
            vertices, indices = source.vertices, source.indices
            if not len(vertices) or not len(indices):
                continue
            if vertex_count + len(vertices) > 65536 or index_count + len(indices) > 196608:
                break
            entries.append(((kind, resource, transparent), vertices, indices))
            vertex_count += len(vertices)
            index_count += len(indices)
        return mesh_data_batch(entries) if len(entries) > 1 else {}

    def _prepare(self):
        snapshot = self.snapshot
        if snapshot is None:
            self._groups = []
            return
        if self.palette is not None:
            self.palette.poll()
        palette_revision = (id(self.palette), self.palette.revision) if self.palette is not None else None
        plane_distance = self.distance if self.settings is not None and np.any(snapshot.kinds == 4) else None
        appearance = (frozenset(self.selected),frozenset(self.hidden),self.scene_id,self.opacity,tuple(sorted(self.alpha_by_id.items())),id(self.instance_alpha),palette_revision,self._geometry_settings,plane_distance,self.all_transparent)
        if appearance == self._appearance:
            return
        self._appearance = appearance
        count = len(snapshot.object_ids)
        mask, selected_mask = geometry_visibility(snapshot, self.settings, scene_id=self.scene_id,
            hidden=self.hidden, selected=self.selected, opacity=self.opacity, all_transparent=self.all_transparent)
        records = np.zeros(count, dtype=INSTANCE_DTYPE)
        records["matrix"] = snapshot.transforms.transpose(0,2,1)
        records["dimensions"] = snapshot.dimensions
        records["color"] = (.39,.68,.83,self.opacity)
        if self.palette is not None:
            records["color"] = self.palette.geometry_colors(snapshot, self.actor_class_names)
            records["color"][:, 3] = self.opacity
        overrides=getattr(snapshot,"override_colors",())
        if len(overrides)==count:
            records["color"]=overrides
        if self.palette is None and hasattr(snapshot,"sleeping"):
            records["color"][snapshot.sleeping.astype(bool)] = (.42,.49,.57,self.opacity)
        if self.settings is not None:
            options = self.settings
            points = snapshot.kinds == 7
            radius_modes = np.asarray(getattr(snapshot, "particle_radius_modes", np.ones(count)), dtype="u1")
            records["dimensions"][points & (radius_modes != 2), 0] = options.particle_size
            if options.particle_size <= .001:
                records["dimensions"][points & (radius_modes == 2), 0] *= options.particle_size
        rows = np.flatnonzero(mask)
        self._visible_rows = rows
        self.instance_count = len(rows)
        if self.palette is None:
            records["color"][selected_mask] = (.80,.94,.49,self.opacity)
        else:
            selection_tint = self.palette.gizmo("selectionTint")[:3]
            records["color"][selected_mask, :3] += selection_tint
            # Blend planes with the tint itself: adding a bright actor color
            # clamps to white and loses contrast against the light ground.
            records["color"][(snapshot.kinds == 4) & selected_mask, :3] = selection_tint
        # Unselected planes keep the ground material.
        records["color"][(snapshot.kinds == 4) & ~selected_mask, :3] = 0.
        for identity,alpha in self.alpha_by_id.items():
            records["color"][(snapshot.object_ids==identity)|(snapshot.shape_ids==identity)|(snapshot.geometry_ids==identity),3]=alpha
        if self.instance_alpha is not None and len(self.instance_alpha)==count:
            records["color"][:,3]=self.instance_alpha
        transparent=records["color"][:,3]<.9999
        states = transparent.astype("u1")
        states |= np.asarray(getattr(snapshot, "double_sided", np.zeros(count)), dtype="u1") << 1
        if self.settings is not None:
            states |= (snapshot.kinds == 7).astype("u1")
            states |= np.asarray(np.broadcast_to(display_mask(self.settings.wireframe, selected_mask), (count,)), dtype="u1") << 2
        records["id"] = snapshot.object_ids
        # Infinite planes have no finite bound. A view-sized patch gives them a
        # useful visual footprint without blowing up frame-all bounds.
        plane_rows = snapshot.kinds == 4
        world_scale = self.settings.geometry_scale if self.settings else 1.
        records["dimensions"][plane_rows] = max(self.distance / world_scale * 2, 100) * (self.settings.plane_scale if self.settings else 1.)
        resources = {int(mesh.id):mesh for mesh in snapshot.meshes}
        # Exact array comparisons permit pose/color updates to reuse membership;
        # hidden objects, resource revisions and opacity boundaries invalidate it.
        # Keep copies because callers can supply mutable array-backed snapshots.
        signature = (rows, snapshot.kinds[rows], snapshot.resource_ids[rows], states[rows])
        if self._group_signature is None or not all(
                np.array_equal(current, previous)
                for current, previous in zip(signature, self._group_signature)):
            _, kinds, resource_ids, alpha_groups = signature
            order = np.lexsort((alpha_groups, resource_ids, kinds))
            ordered_kinds, ordered_resources, ordered_alpha = kinds[order], resource_ids[order], alpha_groups[order]
            boundaries = np.flatnonzero((ordered_kinds[1:] != ordered_kinds[:-1]) |
                (ordered_resources[1:] != ordered_resources[:-1]) |
                (ordered_alpha[1:] != ordered_alpha[:-1])) + 1
            starts = np.r_[0, boundaries] if len(rows) else ()
            stops = np.r_[boundaries, len(rows)] if len(rows) else ()
            self._group_rows = [
                (int(ordered_kinds[start]), int(ordered_resources[start]), int(ordered_alpha[start]), rows[order[start:stop]])
                for start, stop in zip(starts, stops)]
            self._group_signature = tuple(value.copy() for value in signature)
        groups=[]
        active=set()
        prepared = {}
        for group_index, (kind, resource, is_transparent, selected_rows) in enumerate(self._group_rows):
            source=resources.get(resource)
            cache_key=(kind,int(source.topology_id) if source is not None else resource,is_transparent)
            active.add(cache_key)
            mesh=self.meshes.get(cache_key)
            data_key = (kind, resource, is_transparent)
            if kind == 5 and source is not None and (mesh is None or mesh.revision != resource) and not prepared:
                prepared = self._prepare_mesh_batch(group_index, resources)
            if mesh is None:
                if kind in (1,3): data,indices=unit_sphere(),None
                elif kind == 2: data,indices=unit_box(),None
                elif kind == 4: data,indices=unit_plane(),None
                else:
                    if source is None or not len(source.vertices): continue
                    if data_key in prepared:
                        data, indices = prepared.pop(data_key)
                    else:
                        data,indices=mesh_data(source.vertices,source.indices,kind)
                    if kind == 7:
                        indices = None  # Cloth point draws share vertices, not fabric edge indices.
                mesh=self._mesh(cache_key,data,indices)
                mesh.revision=resource
            elif source is not None and mesh.revision!=resource:
                if data_key in prepared:
                    data, _ = prepared.pop(data_key)
                else:
                    data,_=mesh_data(source.vertices,source.indices,kind)
                payload=data.tobytes()
                if len(payload)!=mesh.vertices.size:
                    mesh.vertices.orphan(len(payload))
                mesh.vertices.write(payload)
                mesh.revision=resource
            payload=records[selected_rows].tobytes()
            if len(payload)>mesh.instances.size:
                mesh.instances.orphan(len(payload))
            mesh.instances.write(payload)
            mesh.count=len(selected_rows)
            mesh.transparent=bool(is_transparent & 1)
            mesh.double_sided=bool(is_transparent & 2)
            mesh.wireframe=bool(is_transparent & 4)
            groups.append((kind,mesh))
        self._groups=groups
        # Bound GPU memory for animated vertex revisions and capture replacement.
        for key in list(self.meshes):
            if key not in active:
                self.meshes.pop(key).close()

    def _frame_bounds(self, selected=False):
        if self.snapshot is None or not len(self.snapshot.object_ids):
            return
        mask=self.snapshot.kinds!=4
        if self.settings is not None:
            self._prepare()
            visible = np.zeros(len(mask), dtype=bool)
            visible[self._visible_rows] = True
            mask &= visible
        if self.scene_id is not None:
            mask &= (self.snapshot.scene_ids==self.scene_id)|(self.snapshot.scene_ids<=0)
        if selected and self.selected:
            selected_ids=list(self.selected)
            mask &= np.isin(self.snapshot.object_ids,selected_ids)|np.isin(self.snapshot.shape_ids,selected_ids)|np.isin(self.snapshot.geometry_ids,selected_ids)
        if self.hidden:
            mask &= ~np.isin(self.snapshot.object_ids,list(self.hidden))
        bounds=self.snapshot.bounds[mask]
        bounds=bounds[np.isfinite(bounds).all(axis=(1,2))]
        if len(bounds):
            low,high=bounds[:,0].min(0),bounds[:,1].max(0)
            scale = self.settings.geometry_scale if self.settings else 1.
            return np.array((low * scale,high * scale))

    def selection_pivot(self):
        bounds = self._frame_bounds(selected=True) if self.selected else None
        return self.camera.mirror(bounds.mean(axis=0)) if bounds is not None else None

    def frame_all(self, selected=False, center_only=False):
        bounds = self._frame_bounds(selected)
        if bounds is not None:
            self.camera.frame_bounds(bounds,center_only=center_only)

    @property
    def target(self): return self.camera.target
    @target.setter
    def target(self,value): self.camera.center_at(value)
    @property
    def distance(self): return self.camera.distance
    @distance.setter
    def distance(self,value):
        target=self.camera.target
        self.camera.set_distance(value)
        self.camera.position=target-self.camera.basis[2]*self.camera.distance
    @property
    def yaw(self): return math.radians(self.camera.yaw)
    @yaw.setter
    def yaw(self,value): self.camera.turn(math.degrees(value)-self.camera.yaw,0,orbit=True)
    @property
    def pitch(self): return -math.radians(self.camera.pitch)
    @pitch.setter
    def pitch(self,value): self.camera.turn(0,-math.degrees(value)-self.camera.pitch,orbit=True)

    def pan(self, dx, dy):
        self.camera.pan(dx,dy,self.size[1])

    def eye(self):
        return self.camera.position.copy()

    def _view_projection(self):
        result = self.camera.projection(*self.size)@self.camera.view_matrix()
        if self.settings is not None and self.settings.geometry_scale != 1.:
            result = result @ np.diag([self.settings.geometry_scale] * 3 + [1.])
        return result

    def _resize(self, size):
        if size == self.size and self._attachment_samples == self.samples:
            return
        self.provider.destroy()
        for resource in (self.framebuffer,self.pick_buffer,self.transparent_buffer,self.transparent_clear,self.accumulation,self.optical_depth,self.texture,self.pick_texture,self.depth,self.msaa_buffer,self.msaa_color,self.msaa_depth):
            if resource is not None: resource.release()
        self.msaa_buffer=self.msaa_color=self.msaa_depth=None
        self.texture=self.context.texture(size,4)
        self.pick_texture=self.context.texture(size,4)
        self.depth=self.context.depth_renderbuffer(size)
        self.framebuffer=self.context.framebuffer([self.texture],self.depth)
        self.pick_buffer=self.context.framebuffer([self.pick_texture],self.depth)
        self.accumulation=self.context.texture(size,4,dtype="f2")
        self.optical_depth=self.context.texture(size,1,dtype="f2")
        self.transparent_buffer=self.context.framebuffer([self.accumulation,self.optical_depth],self.depth)
        # Clear only color attachments, preserving the opaque depth buffer.
        self.transparent_clear=self.context.framebuffer([self.accumulation,self.optical_depth])
        self.transparent_buffer.depth_mask=False
        if self.samples:
            self.msaa_color=self.context.renderbuffer(size,4,samples=self.samples)
            self.msaa_depth=self.context.depth_renderbuffer(size,samples=self.samples)
            self.msaa_buffer=self.context.framebuffer([self.msaa_color],self.msaa_depth)
        self._attachment_samples=self.samples
        self.size=size
        capsule_new=ctypes.pythonapi.PyCapsule_New
        capsule_new.restype=ctypes.py_object
        capsule_new.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_void_p]
        self.provider.set_image_data(capsule_new(self.texture.glo,None,None),*size,ui.TextureFormat.RGBA8_UNORM)

    def _draw(self, picking=False, transparent=False, prepared=False):
        if not prepared:
            self._prepare()
        projection=self._view_projection().T.astype("f4").tobytes()
        self.program["view_projection"].write(projection)
        self.program["picking"].value=picking
        self.program["transparent_pass"].value=transparent
        self.program["grid_scale"].value=10**math.floor(math.log10(max(self.distance,1e-4)))/10
        self.program["ground_tint"].value=(self.ground_color if self.ground_color is not None else
            (.72,.76,.79) if self.light else (.105,.125,.15))
        self.program["grid_contrast"].value=self.grid_contrast
        self.program["grid_rgba"].value=self.grid_rgba
        self.program["subdued_grid"].value=self.ground_color is not None
        if self.settings is not None and any(kind == 7 for kind, _ in self._groups):
            self.particle_program["view_projection"].write(projection)
            self.particle_program["picking"].value=picking
            self.particle_program["transparent_pass"].value=transparent
            self.particle_program["eye"].value=tuple(self.camera.mirror(self.eye()) / self.settings.geometry_scale)
            self.particle_program["up"].value=tuple(self.camera.mirror(self.camera.basis[1]))
        if self.settings is not None:
            self.context.front_face="ccw" if self.camera.left_handed else "cw"  # OVUI texture flips clip Y.
        self.context.point_size=4.
        last_kind=last_wire=last_cull=last_heightfield=None
        heightfield_light=None
        for kind,mesh in self._groups:
            if not picking and mesh.transparent!=transparent:
                continue
            self.draw_count+=1
            if kind != last_kind:
                self.program["kind"].value=kind
                last_kind=kind
            if kind == 5 and mesh.double_sided != last_heightfield:
                # Native double_sided marks PxHeightFieldGeometry exclusively.
                # Qt's HF shader lights view-space faces; ordinary meshes use
                # world-space faces. Transform the light, not the captured mesh.
                if mesh.double_sided and heightfield_light is None:
                    heightfield_light=self.camera.view_matrix()[:3,:3].T @ np.array([1.,2.,3.],dtype="f4")
                self.program["mesh_light_direction"].value=tuple(heightfield_light) if mesh.double_sided else (1.,2.,3.)
                last_heightfield=mesh.double_sided
            wire=(self.wireframe or mesh.wireframe) and not picking and kind != 7
            if wire != last_wire:
                self.context.wireframe=wire
                last_wire=wire
            cull=self.settings.face_culling if self.settings else FaceCulling.NONE
            if kind in (6,7) or mesh.double_sided:
                cull=FaceCulling.NONE
            if cull != last_cull:
                if cull:
                    self.context.enable(moderngl.CULL_FACE)
                    self.context.cull_face="back" if cull == FaceCulling.NORMAL else "front"
                else:
                    self.context.disable(moderngl.CULL_FACE)
                last_cull=cull
            mode=moderngl.LINES if kind==6 else moderngl.POINTS if kind==7 else moderngl.TRIANGLES
            mesh.vao.render(mode,instances=mesh.count)
        self.context.wireframe=False
        self.context.disable(moderngl.CULL_FACE)

    def _draw_queries(self,picking=False):
        renderer=self.query_renderer
        if renderer is None or renderer.snapshot is None:
            return
        renderer.size=self.size
        renderer.camera=self.camera
        renderer.selected,renderer.hidden=self.selected,self.hidden
        renderer.scene_id=self.scene_id
        if self.settings is not None and renderer.settings != self.settings:
            renderer.set_settings(self.settings)
        renderer.draw_count=0
        renderer._draw(picking=picking)

    def _draw_orbit_center(self, scene_length, gizmo_scale):
        if self.settings is None or not self.settings.draw_orbit_center or self.snapshot is None:
            return
        transform = orbit_transform(self.camera, scene_length, gizmo_scale, self.settings.geometry_scale,
                                    pivot=self.selection_pivot())
        if self._orbit_mesh is None:
            points, indices = orbit_sphere()
            vb = self.context.buffer(np.column_stack((points, points)).astype("f4").tobytes())
            ib = self.context.buffer(indices.tobytes())
            instances = self.context.buffer(reserve=INSTANCE_DTYPE.itemsize, dynamic=True)
            vao = self.context.vertex_array(self.program,
                [(vb,"3f 3f","position","normal"),
                 (instances,"16f 3f 4f 1u /i","instance_matrix","dimensions","instance_color","instance_id")], ib)
            self._orbit_mesh = GpuMesh(vb, ib, instances, vao)
        record = np.zeros(1, dtype=INSTANCE_DTYPE)
        record["matrix"] = transform.T
        record["dimensions"] = 1.
        record["color"] = 1.  # Original white material; never palette/selection tinted.
        payload = record.tobytes()
        if payload != self._orbit_payload:
            self._orbit_mesh.instances.write(payload)
            self._orbit_payload = payload
        self.program["kind"].value = 1
        self.context.disable(moderngl.CULL_FACE)
        self.context.wireframe = True
        try:
            self._orbit_mesh.vao.render(moderngl.TRIANGLES, instances=1)
            self.draw_count += 1
        finally:
            self.context.wireframe = False

    def _draw_background(self, colors):
        if self.background_program is None:
            self.background_program = self.context.program(vertex_shader=COMPOSITE_VERTEX,
                fragment_shader=BACKGROUND_FRAGMENT)
            self.background_vao = self.context.vertex_array(self.background_program, [])
            for name, value in STAGE_GRADIENT.items():
                self.background_program[name].value = value
        self.background_program["inner_color"].value = tuple(colors[0][:3])
        self.background_program["outer_color"].value = tuple(colors[1][:3])
        self.background_vao.render(moderngl.TRIANGLES, vertices=3)

    def render(self, width, height, light=False,background=None,ground_color=None,grid_contrast=None,
               scene_length=0.,gizmo_scale=1.,background_gradient=None,*,grid_rgba=None):
        self.light=light
        self.ground_color=tuple(ground_color[:3]) if ground_color is not None else None
        self.grid_contrast=float(grid_contrast) if grid_contrast is not None else (.025 if ground_color is not None else .06)
        # The designer material is opt-in. Explicit backgrounds and grid-off
        # retain their existing plane treatment, including standalone callers.
        self.grid_rgba=(tuple(grid_rgba) if grid_rgba is not None and self.grid_contrast != 0.
                        and (self.settings is None or self.settings.background_mode == "workspace")
                        else (0., 0., 0., 0.))
        self.draw_count=0
        self._resize((max(16,int(width)),max(16,int(height))))
        target=self.msaa_buffer or self.framebuffer
        if self.settings is not None:
            background=self.settings.background(background)
        with self.context.scope(framebuffer=target, enable_only=moderngl.DEPTH_TEST):
            self.context.viewport=(0,0,*self.size)
            target.clear(*(background if background is not None else
                                     (.88,.91,.94,1) if light else (.075,.092,.12,1)))
            if background_gradient is not None and (self.settings is None or self.settings.background_mode == "workspace"):
                # A decoration pass before scene depth exists. It never writes
                # depth or IDs, and explicit flat backgrounds retain their color.
                with self.context.scope(framebuffer=target, enable_only=0):
                    self._draw_background(background_gradient)
            self._draw()
            self._draw_orbit_center(scene_length, gizmo_scale)
        if any(mesh.transparent for _,mesh in self._groups):
            if self.msaa_buffer is not None:
                # Resolve both opaque color and depth for the single-sample OIT pass.
                self.context.copy_framebuffer(self.framebuffer,self.msaa_buffer)
            self.transparent_clear.clear(0,0,0,0)
            with self.context.scope(framebuffer=self.transparent_buffer,enable_only=moderngl.DEPTH_TEST|moderngl.BLEND):
                self.context.blend_func=(moderngl.ONE,moderngl.ONE)
                self._draw(transparent=True,prepared=True)
            with self.context.scope(framebuffer=target,enable_only=moderngl.BLEND):
                self.context.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
                self.accumulation.use(0)
                self.optical_depth.use(1)
                self.composite_vao.render(moderngl.TRIANGLES,vertices=3)

        if self.overlays.snapshot is not None:
            with self.context.scope(framebuffer=target,enable_only=moderngl.DEPTH_TEST|moderngl.BLEND):
                self.context.blend_func=(moderngl.SRC_ALPHA,moderngl.ONE_MINUS_SRC_ALPHA)
                self._draw_queries()
                self.overlays.draw(self._view_projection(),self.eye(),self.distance,self.scene_id,self.hidden,
                                   selected=self.selected,framebuffer=target)
                self.draw_count+=self.overlays.draw_count
                if self.query_renderer is not None:
                    self.draw_count+=self.query_renderer.draw_count
        if self.msaa_buffer is not None:
            self.context.copy_framebuffer(self.framebuffer,self.msaa_buffer)

    def pick(self, x, y):
        """Viewport-local pixel coordinates, same top-down convention as OVUI."""
        if self.pick_buffer is None or not (0<=x<self.size[0] and 0<=y<self.size[1]):
            return None
        with self.context.scope(framebuffer=self.pick_buffer, enable_only=moderngl.DEPTH_TEST):
            self.context.viewport=(0,0,*self.size)
            self.pick_buffer.clear(0,0,0,0)
            self._draw(picking=True)
            self._draw_queries(picking=True)
            self.overlays.draw(self._view_projection(),self.eye(),self.distance,self.scene_id,self.hidden,
                               selected=self.selected,picking=True)
            raw=self.pick_buffer.read(viewport=(int(x),int(y),1,1),components=4,alignment=1)
        encoded=int.from_bytes(raw,"little")
        if encoded==0: return None
        value=encoded-1
        return value-2**32 if value>=2**31 else value

    def close(self):
        if self.query_renderer:
            self.query_renderer.close()
        self.overlays.close()
        if self._orbit_mesh is not None:
            self._orbit_mesh.close()
            self._orbit_mesh = None
        self.provider.destroy()
        for mesh in self.meshes.values(): mesh.close()
        for resource in (self.framebuffer,self.pick_buffer,self.transparent_buffer,self.transparent_clear,self.accumulation,self.optical_depth,self.texture,self.pick_texture,self.depth,self.msaa_buffer,self.msaa_color,self.msaa_depth,self.composite_vao,self.composite,self.program,self.particle_program,self.background_vao,self.background_program):
            if resource is not None: resource.release()
