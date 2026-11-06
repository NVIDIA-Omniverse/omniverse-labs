# SPDX-License-Identifier: Apache-2.0
"""Evaluated Blender mesh -> typed OVStage writes, without an intermediate USD.

The initial population establishes a small schema contract. Extraction visits only
nominated evaluated owners and immediately copies RNA data into owned arrays.
The caller owns evaluation, shared-data user closure, population invalidation,
transport, atomic write/seal/ack, and advancement of accepted projection state.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
import numpy as np


class DirectMeshUnsupported(ValueError):
    """A concrete reason to retain scoped population for this owner."""


@dataclass(frozen=True)
class PrimvarContract:
    usd_name: str
    source_name: str
    value_type: str
    interpolation: str
    indexed: bool = False
    source_kind: str = "attribute"


@dataclass(frozen=True)
class SubsetContract:
    path: str
    material_slots: tuple[int, ...]


@dataclass(frozen=True)
class MeshContract:
    path: str
    normals_interpolation: str = "faceVarying"
    primvars: tuple[PrimvarContract, ...] = ()
    subsets: tuple[SubsetContract, ...] = ()
    material_slots: tuple[int, ...] = ()
    unit_scale: float = 1.0
    has_extent: bool = True


@dataclass(frozen=True)
class MeshPatch:
    values: tuple[dict, ...] = ()
    reason: str | None = None
    vertex_count: int = 0
    triangle_count: int = 0

    @property
    def supported(self):
        return self.reason is None


def _read(collection, attribute, width, dtype=np.float32):
    values = np.empty(len(collection) * width, dtype=dtype)
    collection.foreach_get(attribute, values)
    return values.reshape((-1, width)) if width > 1 else values


def material_slot_ids(mesh):
    # Evaluated GeometrySet materials can have session_uid=0. Identity belongs
    # to the original ID; evaluated objects only supply the current values.
    return tuple(int(getattr(getattr(mat, "original", None) or mat, "session_uid", 0))
                 if mat else 0 for mat in mesh.materials)


def _mesh_geometry_owner(evaluated):
    inspect = getattr(evaluated, "evaluated_geometry", None)
    if not callable(inspect):
        # A mesh object with GN can still emit curves/volumes/instances.
        if any(m.type == "NODES" for m in getattr(evaluated, "modifiers", ())):
            raise DirectMeshUnsupported("geometry_set_api_unavailable_for_nodes")
        if getattr(evaluated, "is_instancer", False):
            raise DirectMeshUnsupported("instance_output_requires_population")
        return None, None
    geometry = inspect()
    for name in ("curves", "pointcloud", "volume", "grease_pencil", "instances"):
        if getattr(geometry, name, None) is not None:
            raise DirectMeshUnsupported(f"evaluated_{name}_component_requires_population")
    # GeometrySet exposes instances through an accessor, not an .instances
    # RNA property. Mixed mesh + instances must never lose its instance half.
    instances = getattr(geometry, "instances_pointcloud", None)
    if callable(instances) and instances() is not None:
        raise DirectMeshUnsupported("evaluated_instances_component_requires_population")
    if geometry.mesh is None:
        raise DirectMeshUnsupported("evaluated_mesh_absent")
    return geometry.mesh, geometry


@contextmanager
def evaluated_mesh(obj, depsgraph):
    """Keep the GeometrySet owning borrowed Mesh RNA alive through extraction."""
    evaluated = obj.evaluated_get(depsgraph)
    temporary = False
    geometry_owner = None
    try:
        if getattr(evaluated, "type", "") != "MESH":
            raise DirectMeshUnsupported("non_mesh_owner_requires_population")
        mesh, geometry_owner = _mesh_geometry_owner(evaluated)
        if mesh is None:
            mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
            temporary = True
        yield mesh
    finally:
        if temporary:
            evaluated.to_mesh_clear()
        # GeometrySet owns its Mesh; release it only after the caller copied RNA.
        geometry_owner = None


def extract_object_patch(obj, depsgraph, contract: MeshContract):
    """Copy final evaluated geometry; always release owned temporary meshes."""
    try:
        with evaluated_mesh(obj, depsgraph) as mesh:
            return build_mesh_patch(mesh, contract)
    except DirectMeshUnsupported as error:
        return MeshPatch(reason=str(error))


def build_mesh_patch(mesh, contract: MeshContract):
    """Produce all mutually dependent mesh columns in one atomic update batch.

    Blender's evaluated loop-triangle output preserves corner UVs, split/custom
    normals and polygon material indices. Geometry changes may change array sizes.
    Existing material subsets update together with topology plus their parent's
    native geomSubsets dependency. New subset schemas, newly needed UV schemas or
    mixed geometry components still require population.
    """
    try:
        if material_slot_ids(mesh) != contract.material_slots:
            raise DirectMeshUnsupported("evaluated_material_slots_changed")
        mesh.calc_loop_triangles()
        points = _read(mesh.vertices, "co", 3) * contract.unit_scale
        loops = _read(mesh.loop_triangles, "loops", 3, np.int32).reshape(-1)
        polygons = _read(mesh.loop_triangles, "polygon_index", 1, np.int32)
        vertices = _read(mesh.loops, "vertex_index", 1, np.int32)
        indices = vertices[loops]
        counts = np.full(len(polygons), 3, dtype=np.int32)
        if not len(points) or not len(polygons):
            raise DirectMeshUnsupported("empty_mesh_requires_population")
        values = []

        def emit(path, attribute, kind, value):
            arr = np.asarray(value)
            if arr.dtype.kind in "fi" and not np.isfinite(arr).all():
                raise DirectMeshUnsupported(f"nonfinite_mesh_column:{attribute}")
            # Keep owned NumPy buffers until the caller chooses wire encoding.
            values.append({"prim_path": path, "attribute_name": attribute,
                           "value_type": kind, "value": np.ascontiguousarray(arr)})

        emit(contract.path, "points", "Point3fArray", points)
        emit(contract.path, "faceVertexCounts", "IntArray", counts)
        emit(contract.path, "faceVertexIndices", "IntArray", indices)
        if contract.has_extent:
            emit(contract.path, "extent", "Float3Array", [points.min(0), points.max(0)])
        interpolation = contract.normals_interpolation
        if interpolation == "faceVarying":
            normals = _read(mesh.corner_normals, "vector", 3)[loops]
        elif interpolation in ("vertex", "varying"):
            normals = _read(mesh.vertex_normals, "vector", 3)
        elif interpolation == "uniform":
            normals = _read(mesh.polygon_normals, "vector", 3)[polygons]
        else:
            raise DirectMeshUnsupported(f"unsupported_normal_interpolation:{interpolation}")
        emit(contract.path, "normals", "Normal3fArray", normals)
        for pv in contract.primvars:
            data, domain = _attribute_values(mesh, pv)
            source_indices = {"POINT": indices, "CORNER": loops, "FACE": polygons}
            if pv.interpolation == "faceVarying":
                if domain not in source_indices:
                    raise DirectMeshUnsupported(f"unsupported_primvar_domain:{pv.source_name}:{domain}")
                # FACE data expands one value per triangle to all three corners.
                data = data[source_indices[domain]]
                if domain == "FACE":
                    data = np.repeat(data, 3, axis=0)
            elif pv.interpolation in ("vertex", "varying") and domain == "POINT":
                pass
            elif pv.interpolation == "uniform" and domain == "FACE":
                data = data[polygons]
            else:
                raise DirectMeshUnsupported(f"primvar_domain_mismatch:{pv.source_name}")
            # Match active export normalization, without quantizing other sets.
            if pv.usd_name == "primvars:st":
                data = np.round(data, 3)
            emit(contract.path, pv.usd_name, pv.value_type, data)
            if pv.indexed:
                emit(contract.path, pv.usd_name + ":indices", "IntArray", np.arange(len(data), dtype=np.int32))
        face_slots = _read(mesh.polygons, "material_index", 1, np.int32)[polygons]
        if contract.subsets:
            represented = set()
            for subset in contract.subsets:
                represented.update(subset.material_slots)
                emit(subset.path, "indices", "IntArray",
                     np.flatnonzero(np.isin(face_slots, subset.material_slots)).astype(np.int32))
            if set(face_slots.tolist()) - represented:
                raise DirectMeshUnsupported("new_material_subset_required")
        elif len(set(face_slots.tolist())) > 1:
            raise DirectMeshUnsupported("new_material_subset_required")
        return MeshPatch(tuple(values), vertex_count=len(points), triangle_count=len(polygons))
    except DirectMeshUnsupported as error:
        return MeshPatch(reason=str(error))


def _attribute_values(mesh, pv):
    if pv.source_kind == "uv":
        layer = mesh.uv_layers.get(pv.source_name)
        if layer is None:
            raise DirectMeshUnsupported(f"missing_uv_layer:{pv.source_name}")
        return _read(layer.data, "uv", 2), "CORNER"
    attribute = mesh.attributes.get(pv.source_name)
    if attribute is None:
        raise DirectMeshUnsupported(f"missing_mesh_attribute:{pv.source_name}")
    layouts = {"FLOAT": ("value", 1, np.float32), "INT": ("value", 1, np.int32),
               "BOOLEAN": ("value", 1, np.bool_), "FLOAT_VECTOR": ("vector", 3, np.float32),
               "FLOAT2": ("vector", 2, np.float32), "FLOAT_COLOR": ("color", 4, np.float32),
               "BYTE_COLOR": ("color", 4, np.float32)}
    if attribute.data_type not in layouts:
        raise DirectMeshUnsupported(f"unsupported_mesh_attribute_type:{attribute.data_type}")
    member, width, dtype = layouts[attribute.data_type]
    data = _read(attribute.data, member, width, dtype)
    # Blender USD color primvars are RGB while the source color attribute is RGBA.
    if width == 4 and "3" in pv.value_type:
        data = data[:, :3]
    return data, attribute.domain


def contract_from_usd(prim, mesh, *, material_paths=(), unit_scale=1.0):
    """Read one already-open initial USD Mesh; no scene traversal or file I/O.

    material_paths is the effective evaluated slot -> USD material path mapping.
    Resolve it once from the initial export mapping; never infer binding by name.
    The caller invalidates this contract whenever scoped population touches it.
    """
    from pxr import Tf, UsdGeom
    if prim.GetTypeName() != "Mesh" or prim.IsInstanceProxy() or prim.IsInstance():
        raise DirectMeshUnsupported("non_owned_mesh_prim")
    usd_mesh = UsdGeom.Mesh(prim)
    if str(usd_mesh.GetSubdivisionSchemeAttr().Get()) != "none":
        raise DirectMeshUnsupported("non_tessellated_mesh")
    if not usd_mesh.GetNormalsAttr().HasAuthoredValue():
        raise DirectMeshUnsupported("missing_authored_normals")
    if list(usd_mesh.GetHoleIndicesAttr().Get() or []):
        raise DirectMeshUnsupported("mesh_holes_require_population")
    uv_names = {Tf.MakeValidIdentifier(layer.name): layer.name for layer in mesh.uv_layers}
    # Blender exports the render-active UV map as st; inactive maps retain names.
    render_uv = next((layer for layer in mesh.uv_layers if layer.active_render), None)
    if render_uv is not None:
        uv_names["st"] = render_uv.name
    attribute_names = {Tf.MakeValidIdentifier(a.name): a.name for a in mesh.attributes}
    primvars = []
    supported_types = {"float[]", "float2[]", "float3[]", "float4[]", "color3f[]", "color4f[]",
                       "texCoord2f[]", "texCoord3f[]", "normal3f[]", "vector3f[]", "point3f[]", "int[]", "bool[]"}
    for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
        if not pv.GetAttr().HasAuthoredValue():
            continue
        name, kind = str(pv.GetPrimvarName()), str(pv.GetTypeName())
        # This scalar render control is guarded by the coordinator's
        # visible_shadow signature. Unknown constants may be authored by GN;
        # retain their explicit fallback rather than leaving a stale column.
        if (name == "doNotCastShadows" and kind == "bool"
                and str(pv.GetInterpolation()) == "constant"):
            continue
        if name.startswith("skel:"):
            raise DirectMeshUnsupported("skinning_requires_population")
        if kind not in supported_types:
            raise DirectMeshUnsupported(f"unsupported_primvar_type:{name}:{kind}")
        if name in uv_names:
            source_name, source_kind = uv_names[name], "uv"
        elif name in attribute_names:
            source_name, source_kind = attribute_names[name], "attribute"
        else:
            raise DirectMeshUnsupported(f"unmapped_primvar:{name}")
        primvars.append(PrimvarContract(str(pv.GetAttr().GetName()), source_name, kind,
                                        str(pv.GetInterpolation()), pv.IsIndexed(), source_kind))
    subsets = []
    for child in prim.GetChildren():
        if child.GetTypeName() != "GeomSubset":
            raise DirectMeshUnsupported("mesh_child_schema_requires_population")
        subset = UsdGeom.Subset(child)
        if str(subset.GetElementTypeAttr().Get()) != "face" or str(subset.GetFamilyNameAttr().Get()) != "materialBind":
            raise DirectMeshUnsupported("non_material_subset_requires_population")
        targets = child.GetRelationship("material:binding").GetTargets()
        slots = tuple(i for i, p in enumerate(material_paths) if targets and str(p) == str(targets[0]))
        if not slots:
            raise DirectMeshUnsupported("unmapped_material_subset")
        subsets.append(SubsetContract(str(child.GetPath()), slots))
    return MeshContract(str(prim.GetPath()), str(usd_mesh.GetNormalsInterpolation()),
                        tuple(primvars), tuple(subsets), material_slot_ids(mesh), unit_scale,
                        usd_mesh.GetExtentAttr().HasAuthoredValue())


def wire_values(patch: MeshPatch):
    """JSON boundary only: no nested Python tuples while extracting/hash-checking."""
    if not patch.supported:
        raise DirectMeshUnsupported(patch.reason)
    return [{**row, "value": row["value"].tolist()} for row in patch.values]
