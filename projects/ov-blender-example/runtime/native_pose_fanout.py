# SPDX-License-Identifier: Apache-2.0
"""Owned, generation-scoped native world-pose updates for ordinary hierarchies.

OVStage's hierarchy may preserve descendant world poses when a parent input is
changed. Explicit descendant inputs avoid that compensation without repopulating
USD. Only copied identities and arrays survive native read scopes.

World inputs, explicit or synthetic descendant ones, set
``omni:resetXformStack`` to True on the native stage. They outlive this object whenever the stage survives a population or
structure change, so the logical identity read before the first synthetic write
is retained per stage (see ``successor_fanout``); a fresh native read would
mistake the synthetic boundary for an authored one and stop fanning out.
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np


class UnsafeHierarchy(RuntimeError):
    pass


@dataclass
class FanoutPlan:
    extra_rows: list = field(default_factory=list)
    parent_poses: dict = field(default_factory=dict)
    fallback_reason: str = ""
    logical_resets: dict = field(default_factory=dict)
    camera_only: bool = False
    auto_invalidation_safe: bool = False
    unchanged_poses: set = field(default_factory=set)
    structural_paths: set = field(default_factory=set)
    # Pre-write identities of explicit world-input (reset=True) pose targets.
    pinned_identities: dict = field(default_factory=dict)


_TRANSFORM_TYPES = {"Xform", "Mesh", "Cube", "Sphere", "Capsule", "Cylinder", "Cone",
                    "Camera", "SphereLight", "RectLight",
                    "DiskLight", "CylinderLight", "DistantLight", "DomeLight"}
_IDENTITY_ATTRIBUTES = ("usd-prim-type", "usd-schemas", "omni:resetXformStack",
                        "physics:rigidBodyEnabled", "physics:collisionEnabled", "physics:mass")
_RENDER_PHYSICS_SCHEMAS = frozenset({"PhysicsRigidBodyAPI", "PhysicsCollisionAPI",
    "PhysicsMeshCollisionAPI", "PhysicsMassAPI", "PhysicsArticulationRootAPI",
    "PhysxRigidBodyAPI", "PhysxCollisionAPI", "PhysxArticulationAPI"})


class NativePoseFanout:
    def __init__(self, stage, ovs, *, max_prims=512, max_depth=64, render_only=False):
        self.stage, self.ovs = stage, ovs
        # Only RendererBackend owns this contract: its stage has no attached
        # solver. PhysicsBackend owns a different stage in a different worker.
        self.render_only = render_only
        self.rigid_paths = set()
        self.max_prims, self.max_depth = max_prims, max_depth
        self.context = None
        self.identities = {}
        self.children = {}
        self.instance_paths = None
        self.accepted_parents = {}
        self.structural_writes = set()
        # Logical identities of prims holding native world inputs we wrote
        # (explicit or synthetic), and the untyped synthetic ones among them. A successor for the same native stage
        # starts its first context with its predecessor's surviving records.
        self.synthetic_identities = {}
        self.synthetic_structural = set()
        self.synthetic_rigid = set()

    def _reset_context(self, context):
        if context != self.context:
            inherited = self.context is None
            self.context = context
            self.identities.clear()
            self.children.clear()
            self.instance_paths = None
            self.accepted_parents.clear()
            self.rigid_paths.clear()
            self.structural_writes.clear()
            if inherited:
                # Seeded identities skip the native read that classifies rigid
                # bodies; carry that classification with them.
                self.identities.update(self.synthetic_identities)
                self.structural_writes.update(self.synthetic_structural)
                self.rigid_paths.update(self.synthetic_rigid)
            else:
                self.synthetic_identities.clear()
                self.synthetic_structural.clear()
                self.synthetic_rigid.clear()

    def _read(self, path, names, ordinal, paths):
        result = {}
        tokens = {paths.intern_token(name): name for name in names}
        with paths.create_path_list_from_strings([path]) as plist:
            with self.stage.query_from_path_list(plist) as query:
                with self.stage.read_attributes(query, list(tokens), self.ovs.OrdinalRange.latest(ordinal)) as read:
                    read.wait()
                    for group in read.groups():
                        try:
                            if group.is_delete:
                                continue
                            if group.prim_count != 1 or group.attribute not in tokens:
                                raise UnsafeHierarchy("Ambiguous native hierarchy column")
                            name = tokens[group.attribute]
                            if name in result:
                                raise UnsafeHierarchy("Duplicate native hierarchy column")
                            result[name] = np.asarray(group.array(0)).copy()
                        finally:
                            self.stage.release_group(group)
        return result

    def _identity(self, path, ordinal, paths):
        if path not in self.identities or self.identities[path][0] == "":
            columns = self._read(path, _IDENTITY_ATTRIBUTES, ordinal, paths)
            kinds = columns.get("usd-prim-type")
            if path in self.structural_writes and kinds is not None:
                raise UnsafeHierarchy("Structural node identity changed: " + path)
            if path in self.structural_writes:
                resets = columns.get("omni:resetXformStack")
                if resets is None or resets.size != 1 or not bool(resets.reshape(-1)[0]):
                    raise UnsafeHierarchy("Structural node reset changed: " + path)
                columns.pop("omni:resetXformStack")
            # OVStage exposes implicit, untyped USD ancestors in its hierarchy
            # without an identity column. This is only a candidate structural
            # node: each plan must also prove its inherited world/input below.
            if not columns and self.render_only:
                self.identities[path] = ("", False)
                return self.identities[path]
            if kinds is None or kinds.size != 1:
                raise UnsafeHierarchy("Missing native prim identity: " + path)
            kind = paths.token_to_string(int(kinds.reshape(-1)[0]))
            schemas = tuple(paths.token_to_string(int(value))
                            for value in columns.get("usd-schemas", np.empty(0)).reshape(-1))
            physics = {schema for schema in schemas
                       if "physics" in schema.lower() or "physx" in schema.lower()}
            if physics and (not self.render_only or physics - _RENDER_PHYSICS_SCHEMAS):
                raise UnsafeHierarchy("Unsupported physics hierarchy: " + path)
            physics_columns = any(name in columns for name in _IDENTITY_ATTRIBUTES[3:])
            if physics_columns and (not self.render_only or not physics):
                raise UnsafeHierarchy("Unclassified physics columns: " + path)
            if "PhysicsRigidBodyAPI" in physics:
                self.rigid_paths.add(path)
            resets = columns.get("omni:resetXformStack")
            if resets is not None and resets.size != 1:
                raise UnsafeHierarchy("Ambiguous reset boundary: " + path)
            self.identities[path] = (kind, bool(resets.reshape(-1)[0]) if resets is not None else False)
        return self.identities[path]

    def _worlds(self, targets, ordinal, paths):
        """Read only affected poses in one query, copying before groups release."""
        if not targets:
            return {}
        if not callable(getattr(paths, 'get_path_strings', None)):
            return {path: self._read(path, ('omni:fabric:worldMatrix',), ordinal, paths)
                    .get('omni:fabric:worldMatrix') for path in targets}
        worlds = {}
        prim_lists = {}
        token = paths.intern_token('omni:fabric:worldMatrix')
        with paths.create_path_list_from_strings(list(targets)) as plist:
            with self.stage.query_from_path_list(plist) as query:
                with self.stage.read_attributes(query, [token], self.ovs.OrdinalRange.latest(ordinal)) as read:
                    read.wait()
                    for group in read.groups():
                        try:
                            if group.is_delete:
                                continue
                            if group.attribute != token or group.is_array:
                                raise UnsafeHierarchy('Unsupported world matrix mapping')
                            # Groups can share one large prim list with offsets.
                            # Resolving that entire list per group is quadratic.
                            key = group.prim_list
                            if isinstance(key, list):
                                key = tuple(key)
                            if key not in prim_lists:
                                prim_lists[key] = paths.get_path_strings(group.prim_list)
                            names = prim_lists[key]
                            values = np.asarray(group.array(0))
                            if values.size % 16 or group.data_count != group.prim_count:
                                raise UnsafeHierarchy('Ambiguous world matrix group')
                            matrices = values.reshape(-1, 4, 4)
                            for index in range(group.prim_count):
                                path = names[group.prim_index(index)]
                                value = matrices[group.data_row_index(index)]
                                if path not in targets or path in worlds:
                                    raise UnsafeHierarchy('Ambiguous world matrix identity')
                                worlds[path] = value.copy()
                        finally:
                            self.stage.release_group(group)
        return worlds

    def _check_instances(self, path):
        if self.instance_paths is None:
            instancing = getattr(self.ovs, "instancing", None)
            if instancing is None or not instancing.available():
                raise UnsafeHierarchy("Native instance identity unavailable")
            prototypes = tuple(instancing.get_prototype_roots(self.stage))
            instances = tuple(root for prototype in prototypes
                              for root in instancing.get_instance_roots(self.stage, prototype))
            self.instance_paths = frozenset((*prototypes, *instances))
        # Camera/controller work must not scan every instance in the scene.
        # Membership depends only on the target's ancestors, not scene size.
        ancestor = path
        while ancestor:
            if ancestor in self.instance_paths:
                raise UnsafeHierarchy("Instance/prototype hierarchy: " + path)
            ancestor = ancestor.rpartition("/")[0]

    def _descendants(self, root, ordinal, paths, explicit_roots=()):
        # Cache only exact native path strings. Logical reset boundaries remain
        # those read before our synthetic reset=True world-input writes.
        queue = deque([(root, 0)])
        seen = set()
        result = []
        while queue:
            path, depth = queue.popleft()
            if path in seen or depth > self.max_depth or len(seen) >= self.max_prims:
                raise UnsafeHierarchy("Hierarchy is cyclic or exceeds fanout budget")
            seen.add(path)
            # Each explicitly moved parent owns its own subtree. Rewalking it
            # from every ancestor is quadratic on deeply nested scene replays.
            if path != root and path in explicit_roots:
                continue
            self._check_instances(path)
            kind, reset = self._identity(path, ordinal, paths)
            if path != root and path in self.rigid_paths and path not in explicit_roots:
                raise UnsafeHierarchy("Nested rigid body requires explicit world pose: " + path)
            if kind not in _TRANSFORM_TYPES and kind not in {"GeomSubset", ""}:
                raise UnsafeHierarchy("Unsupported native hierarchy type: " + kind)
            if path != root and reset:
                continue
            # Material face subsets have no transform of their own. Traverse
            # through them without requesting/writing a nonexistent matrix;
            # physics/instancing admission still applies to all visited prims.
            if kind != "GeomSubset":
                result.append(path)
            if path not in self.children:
                with paths.create_path_list_from_strings([path]) as plist:
                    hierarchy = self.stage.get_hierarchy(plist, ordinal, self.ovs.HierarchyRelation.CHILDREN)
                if hierarchy.input_count != 1 or not hierarchy.items[0].ok:
                    raise UnsafeHierarchy("Native hierarchy lookup failed")
                children = []
                for raw in hierarchy.items[0].paths:
                    child = raw if raw.startswith("/") else paths.path_to_string(int(raw))
                    if not child.startswith(path + "/") or "/" in child[len(path) + 1:]:
                        raise UnsafeHierarchy("Child escapes its native parent")
                    children.append(child)
                if len(set(children)) != len(children):
                    raise UnsafeHierarchy("Duplicate native children")
                self.children[path] = tuple(children)
            if kind == "" and not self.children[path]:
                raise UnsafeHierarchy("Untyped hierarchy node has no children: " + path)
            queue.extend((child, depth + 1) for child in self.children[path])
        return tuple(result)

    def prepare(self, poses, values, *, ordinal, context):
        self._reset_context(context)
        plan = FanoutPlan()
        resets = {row["prim_path"]: bool(row["value"]) for row in values
                  if row.get("attribute_name", row.get("attribute")) == "omni:resetXformStack"}
        plan.logical_resets = {path: reset for path, reset in resets.items() if path not in poses}
        if not poses:
            return plan
        try:
            with self.ovs.PathDictionary(self.stage) as paths:
                pose_kinds = {}
                finite_invertible_poses = all(np.isfinite(matrix).all()
                    and np.linalg.cond(matrix) <= 1e12 for matrix in poses.values())
                for path, matrix in poses.items():
                    kind, _reset = self._identity(path, ordinal, paths)
                    if kind not in _TRANSFORM_TYPES:
                        raise UnsafeHierarchy("Unsupported native pose owner: " + path)
                    self._check_instances(path)
                    pose_kinds[path] = kind
                    if resets.get(path) is True:
                        plan.pinned_identities[path] = self.identities[path]
                    if kind == "Xform":
                        plan.parent_poses[path] = np.asarray(matrix, dtype=np.float64).reshape(4, 4).copy()
                # Broad Blender notifications and generation replay can resend
                # identical world poses. Compare native state, not a stale
                # desired-value cache that physics or another owner can change.
                # One camera/leaf update already has bounded work; avoid adding
                # a read there solely to optimize an occasional duplicate.
                if (plan.parent_poses and finite_invertible_poses and not plan.logical_resets
                        and all(resets.get(path) is True for path in poses)):
                    current = self._worlds(poses, ordinal, paths)
                    changed = {path for path, matrix in poses.items()
                               if not np.array_equal(current.get(path), matrix)}
                    for path in poses.keys() - changed:
                        ancestor = path.rpartition('/')[0]
                        while ancestor and ancestor not in changed:
                            ancestor = ancestor.rpartition('/')[0]
                        # An explicit unchanged child must still be pinned if
                        # an ancestor moves in the same batch.
                        if not ancestor:
                            plan.unchanged_poses.add(path)
                if not plan.parent_poses:
                    plan.camera_only = all(kind == "Camera" for kind in pose_kinds.values())
                    plan.auto_invalidation_safe = (finite_invertible_poses and not plan.logical_resets
                        and all(resets.get(path) is True for path in poses))
                    return plan
                if plan.logical_resets or any(resets.get(path) is not True for path in poses):
                    raise UnsafeHierarchy("Pose batch does not specify world-input semantics")
                descendants = {}
                active_parents = plan.parent_poses.keys() - plan.unchanged_poses
                for root in active_parents:
                    for path in self._descendants(root, ordinal, paths, plan.parent_poses):
                        descendants.setdefault(path, []).append(root)
                # Camera history and scene accumulation invalidation are
                # separate: validated scene poses invalidate RTX accumulation
                # internally without an expensive global renderer reset.
                plan.camera_only = (all(kind in {"Xform", "Camera"} for kind in pose_kinds.values())
                    and all(self.identities[path][0] in {"Xform", "Camera"} for path in descendants)
                    and all(any(self.identities[path][0] == "Camera" and root in owners
                                for path, owners in descendants.items()) for root in plan.parent_poses))
                # Read all world poses before any write. Relative transforms use
                # the closest explicit parent; explicit child poses win outright.
                worlds = self._worlds(descendants, ordinal, paths)
                for path in descendants:
                    value = worlds.get(path)
                    if value is None or value.size != 16 or not np.isfinite(value).all():
                        raise UnsafeHierarchy("Missing finite world matrix: " + path)
                    worlds[path] = value.reshape(4, 4)
                for path in descendants:
                    if self.identities[path][0] != "":
                        continue
                    parent = path.rpartition("/")[0]
                    columns = self._read(path, ("omni:xform", "omni:localMatrix"), ordinal, paths)
                    native_input = columns.get("omni:xform")
                    # Before our reset=True write, OVStage's implicit ancestor
                    # carries an identity LOCAL input, even below a transformed
                    # parent. Only accepted synthetic writes switch it to world.
                    expected_input = worlds[parent] if path in self.structural_writes and parent in worlds else np.eye(4)
                    if (parent not in worlds or native_input is None or native_input.size != 16
                            or "omni:localMatrix" in columns
                            or not np.array_equal(worlds[path], worlds[parent])
                            or not np.array_equal(native_input.reshape(4, 4), expected_input)):
                        raise UnsafeHierarchy("Untyped node does not inherit its parent pose: " + path)
                    plan.structural_paths.add(path)
                inverses = {}
                for root in active_parents:
                    matrix = worlds[root]
                    if np.linalg.cond(matrix) > 1e12:
                        raise UnsafeHierarchy("Singular/ill-conditioned parent world pose")
                    inverses[root] = np.linalg.inv(matrix)
                projected = {path: np.asarray(matrix).copy() for path, matrix in poses.items()}
                for path, roots in descendants.items():
                    if path in poses:
                        continue
                    root = max(roots, key=len)
                    matrix = worlds[path] @ inverses[root] @ plan.parent_poses[root]
                    parent = path.rpartition("/")[0]
                    if path in plan.structural_paths:
                        # Breadth-first traversal places parents before their
                        # children. Copy their projected pose exactly: inverse
                        # roundoff must not invent a structural local transform.
                        if parent not in projected:
                            raise UnsafeHierarchy("Structural parent projection unavailable: " + path)
                        matrix = projected[parent].copy()
                    projected[path] = matrix
                    if not np.isfinite(matrix).all():
                        raise UnsafeHierarchy("Fanout pose is not finite")
                    plan.extra_rows.extend((dict(prim_path=path, attribute_name="omni:xform",
                        value_type="Matrix4d", value=matrix.tolist()), dict(prim_path=path,
                        attribute_name="omni:resetXformStack", value_type="Bool", value=True)))
                plan.auto_invalidation_safe = finite_invertible_poses
        except (UnsafeHierarchy, ValueError, RuntimeError, AttributeError, IndexError, np.linalg.LinAlgError) as error:
            plan.extra_rows.clear()
            plan.unchanged_poses.clear()
            plan.structural_paths.clear()
            plan.fallback_reason = str(error)
            # The fallback projects Xform poses through USD, not native inputs.
            plan.pinned_identities = {path: identity for path, identity in plan.pinned_identities.items()
                                      if identity[0] != "Xform"}
            plan.camera_only = False
            plan.auto_invalidation_safe = False
        return plan

    def accepted(self, plan):
        self.accepted_parents.update({path: matrix.copy() for path, matrix in plan.parent_poses.items()})
        for path, identity in plan.pinned_identities.items():
            self.synthetic_identities.setdefault(path, identity)
            if path in self.rigid_paths:
                self.synthetic_rigid.add(path)
        for row in plan.extra_rows:
            if row["attribute_name"] == "omni:xform" and row["prim_path"] in self.identities:
                self.synthetic_identities.setdefault(row["prim_path"], self.identities[row["prim_path"]])
            if row["attribute_name"] == "omni:xform" and self.identities.get(row["prim_path"], (None,))[0] == "":
                self.structural_writes.add(row["prim_path"])
                self.synthetic_structural.add(row["prim_path"])
            if row["attribute_name"] == "omni:xform" and row["prim_path"] in self.accepted_parents:
                self.accepted_parents[row["prim_path"]] = np.asarray(row["value"], dtype=np.float64).copy()
        for path, reset in plan.logical_resets.items():
            if path in self.identities:
                self.identities[path] = (self.identities[path][0], reset)
            if path in self.synthetic_identities:
                self.synthetic_identities[path] = (self.synthetic_identities[path][0], reset)


def _under(path, roots):
    return any(path == root or path.startswith(root.rstrip("/") + "/") for root in roots)


def successor_fanout(previous, changed_paths):
    """Fanout for the same native stage after a change scoped to ``changed_paths``.

    Synthetic inputs outside the changed subtrees remain on the stage, so their
    logical identities carry over. Prims inside them are rediscovered natively.
    ``None`` for an unscoped change (full replacement) or nothing to carry; the
    backend then creates a fresh fanout on its next pose update.
    """
    if changed_paths is None or not getattr(previous, "synthetic_identities", None):
        return None
    roots = tuple(changed_paths)
    identities = {path: identity for path, identity in previous.synthetic_identities.items()
                  if not _under(path, roots)}
    if not identities:
        return None
    successor = NativePoseFanout(previous.stage, previous.ovs, max_prims=previous.max_prims,
                                 max_depth=previous.max_depth, render_only=previous.render_only)
    successor.synthetic_identities = identities
    successor.synthetic_structural = {path for path in previous.synthetic_structural
                                      if path in identities}
    successor.synthetic_rigid = {path for path in previous.synthetic_rigid if path in identities}
    return successor
