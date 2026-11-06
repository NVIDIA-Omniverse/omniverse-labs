# SPDX-License-Identifier: Apache-2.0
"""World-input fanout, owned SDK reads and conservative hierarchy admission."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "runtime"))
from native_pose_fanout import NativePoseFanout, successor_fanout


def matrix(x=0):
    value = np.eye(4)
    value[3, 0] = x
    return value


def rows(poses):
    return [row for path, value in poses.items() for row in (
        dict(prim_path=path, attribute_name="omni:xform", value_type="Matrix4d", value=value.tolist()),
        dict(prim_path=path, attribute_name="omni:resetXformStack", value_type="Bool", value=True))]


class Native:
    def __init__(self):
        self.nodes = {}
        self.tokens = {}
        self.hierarchy_queries = 0
        self.identity_reads = 0
        self.released = 0
        self.borrowed = 0
        self.prototype_roots = []
        self.instance_roots = []
        self.numeric_children = False
        self.world_queries = 0
        self.ovs = SimpleNamespace(PathDictionary=lambda stage: nullcontext(self),
            OrdinalRange=SimpleNamespace(latest=lambda ordinal: ordinal),
            HierarchyRelation=SimpleNamespace(CHILDREN=1),
            instancing=SimpleNamespace(available=lambda: True,
                get_prototype_roots=lambda stage: self.prototype_roots,
                get_instance_roots=lambda stage, root: self.instance_roots))

    def add(self, path, kind="Xform", x=0, reset=False, schemas=()):
        self.nodes[path] = {"usd-prim-type": np.array([self.intern_token(kind)]),
            "usd-schemas": np.array([self.intern_token(v) for v in schemas], dtype=np.uint64),
            "omni:resetXformStack": np.array([reset]), "omni:fabric:worldMatrix": matrix(x).reshape(1,16)}

    def intern_token(self, token):
        return self.tokens.setdefault(token, len(self.tokens) + 1)

    def token_to_string(self, token):
        return next(name for name, value in self.tokens.items() if value == token)

    def path_to_string(self, token):
        return self.token_to_string(token)

    def create_path_list_from_strings(self, paths):
        return nullcontext(paths)

    def query_from_path_list(self, paths):
        return nullcontext(paths)

    def get_path_strings(self, paths):
        return list(paths)

    def read_attributes(self, query, tokens, ordinal):
        names = [self.token_to_string(token) for token in tokens]
        self.world_queries += names == ['omni:fabric:worldMatrix']
        self.identity_reads += "usd-prim-type" in names
        groups = []
        for path in query:
            for name in names:
                value = self.nodes.get(path, {}).get(name)
                if value is None:
                    continue
                borrowed = value.copy()
                self.borrowed += 1
                groups.append(SimpleNamespace(is_delete=False, prim_count=1,
                    prim_list=[path], index_map=None, prim_index_map=None,
                    data_count=1, prim_index=lambda i: i, data_row_index=lambda i: i,
                    attribute=self.intern_token(name), array=lambda index, value=borrowed: value,
                    storage=borrowed, raw=SimpleNamespace(semantic=5 if value.size == 16 else 0),
                    is_array=False, tensor=lambda index, size=value.size: SimpleNamespace(dtype=SimpleNamespace(lanes=size))))
        return nullcontext(SimpleNamespace(wait=lambda: None, groups=lambda: groups))

    def release_group(self, group):
        # Poison the borrowed storage immediately: fanout must have copied it.
        group.storage.fill(0)
        self.released += 1

    def get_hierarchy(self, query, ordinal, relation):
        self.hierarchy_queries += 1
        children = [path for path in self.nodes if path.startswith(query[0] + "/")
                    and "/" not in path[len(query[0]) + 1:]]
        if self.numeric_children:
            children = [str(self.intern_token(path)) for path in children]
        return SimpleNamespace(input_count=1, items=[SimpleNamespace(ok=True, paths=tuple(children))])


def helper():
    native = Native()
    native.add("/World/Parent", x=1)
    native.add("/World/Parent/Leaf", kind="Mesh", x=3)
    return native, NativePoseFanout(native, native.ovs)


def prepare(fanout, poses, *, values=None, context=("stage", "accepted.usda")):
    return fanout.prepare(poses, rows(poses) if values is None else values,
                          ordinal=4, context=context)


def extras(plan):
    return {row["prim_path"]: np.array(row["value"]) for row in plan.extra_rows
            if row["attribute_name"] == "omni:xform"}


def test_parent_world_fanout_uses_owned_native_values_and_row_matrix_convention():
    native, fanout = helper()
    transform = matrix(10)
    transform[:2, :2] = ((0, 2), (-2, 0))
    plan = prepare(fanout, {"/World/Parent": transform})
    assert not plan.fallback_reason
    np.testing.assert_allclose(extras(plan)["/World/Parent/Leaf"], matrix(3) @ np.linalg.inv(matrix(1)) @ transform)
    assert native.borrowed == native.released
    assert all(row["value"] is True for row in plan.extra_rows if row["attribute_name"] == "omni:resetXformStack")


def test_nested_explicit_parent_and_leaf_poses_override_ancestor_fanout():
    native, fanout = helper()
    native.add("/World/Parent/Nested", x=5)
    native.add("/World/Parent/Nested/Leaf", kind="Mesh", x=8)
    native.add("/Outside", kind="Mesh", x=99)
    poses = {"/World/Parent": matrix(10), "/World/Parent/Nested": matrix(20),
             "/World/Parent/Leaf": matrix(100)}
    plan = prepare(fanout, poses)
    assert not plan.fallback_reason
    output = extras(plan)
    assert set(output) == {"/World/Parent/Nested/Leaf"}
    np.testing.assert_allclose(output["/World/Parent/Nested/Leaf"], matrix(23))


def test_synthetic_reset_inputs_do_not_break_subsequent_parent_inheritance():
    native, fanout = helper()
    first = prepare(fanout, {"/World/Parent": matrix(10)})
    fanout.accepted(first)
    native.nodes["/World/Parent"]["omni:fabric:worldMatrix"] = matrix(10).reshape(1,16)
    native.nodes["/World/Parent/Leaf"]["omni:fabric:worldMatrix"] = matrix(12).reshape(1,16)
    native.nodes["/World/Parent/Leaf"]["omni:resetXformStack"] = np.array([True])
    second = prepare(fanout, {"/World/Parent": matrix(1)})
    np.testing.assert_allclose(extras(second)["/World/Parent/Leaf"], matrix(3))
    assert native.identity_reads == 2 and native.hierarchy_queries == 2


def test_authored_reset_boundary_is_independent_but_explicit_nested_root_can_move():
    native, fanout = helper()
    native.add("/World/Parent/Independent", reset=True, x=40)
    native.add("/World/Parent/Independent/Leaf", kind="Mesh", x=41)
    first = prepare(fanout, {"/World/Parent": matrix(10)})
    assert set(extras(first)) == {"/World/Parent/Leaf"}
    second = prepare(fanout, {"/World/Parent": matrix(10), "/World/Parent/Independent": matrix(50)})
    np.testing.assert_allclose(extras(second)["/World/Parent/Independent/Leaf"], matrix(51))


def test_numeric_hierarchy_tokens_are_resolved_and_generation_invalidates_identity_cache():
    native, fanout = helper()
    native.numeric_children = True
    first = prepare(fanout, {"/World/Parent": matrix(10)})
    assert not first.fallback_reason
    native.add("/World/Parent/NewLeaf", kind="Mesh", x=4)
    second = prepare(fanout, {"/World/Parent": matrix(10)}, context=("stage", "new.usda"))
    assert set(extras(second)) == {"/World/Parent/Leaf", "/World/Parent/NewLeaf"}


@pytest.mark.parametrize("unsafe", ["physics_schema", "physics_column", "instance", "unknown_type", "missing_world", "singular", "budget"])
def test_unsafe_hierarchy_returns_no_partial_expansion(unsafe):
    native, fanout = helper()
    leaf = native.nodes["/World/Parent/Leaf"]
    if unsafe == "physics_schema": leaf["usd-schemas"] = np.array([native.intern_token("PhysicsRigidBodyAPI")])
    if unsafe == "physics_column": leaf["physics:mass"] = np.array([1.0])
    if unsafe == "instance": native.prototype_roots = ["/Proto"]; native.instance_roots = ["/World/Parent/Leaf"]
    if unsafe == "unknown_type": leaf["usd-prim-type"] = np.array([native.intern_token("PointInstancer")])
    if unsafe == "missing_world": leaf.pop("omni:fabric:worldMatrix")
    if unsafe == "singular": native.nodes["/World/Parent"]["omni:fabric:worldMatrix"] = np.zeros((1,16))
    if unsafe == "budget": fanout.max_prims = 1
    plan = prepare(fanout, {"/World/Parent": matrix(10)})
    assert plan.fallback_reason and not plan.extra_rows
    assert not fanout.accepted_parents


def test_explicit_reset_without_pose_changes_logical_boundary_only_after_acceptance():
    native, fanout = helper()
    prepare(fanout, {"/World/Parent": matrix(10)})
    value = dict(prim_path="/World/Parent/Leaf", attribute_name="omni:resetXformStack", value=True)
    reset = prepare(fanout, {}, values=[value])
    assert fanout.identities["/World/Parent/Leaf"][1] is False
    fanout.accepted(reset)
    assert not extras(prepare(fanout, {"/World/Parent": matrix(10)}))


def test_direct_mesh_pose_does_not_discover_or_modify_other_hierarchies():
    native, fanout = helper()
    result = prepare(fanout, {"/World/Parent/Leaf": matrix(40)})
    assert not result.extra_rows and not result.parent_poses and not result.fallback_reason
    assert native.hierarchy_queries == 0


def test_material_subset_does_not_force_reprojection_or_receive_matrix_writes():
    native, fanout = helper()
    subset = '/World/Parent/Leaf/MaterialFaces'
    native.add(subset, kind='GeomSubset')
    native.nodes[subset].pop('omni:fabric:worldMatrix')
    plan = prepare(fanout, {'/World/Parent': matrix(10)})
    assert not plan.fallback_reason
    assert set(extras(plan)) == {'/World/Parent/Leaf'}
    np.testing.assert_allclose(extras(plan)['/World/Parent/Leaf'], matrix(12))
    native.add(subset + '/Unsupported', kind='PointInstancer')
    assert prepare(fanout, {'/World/Parent': matrix(11)}, context=('stage', 'new')).fallback_reason


def test_explicit_nested_parents_each_own_a_bounded_subtree():
    native = Native()
    poses = {}
    path = ''
    for index in range(30):
        path += '/Nested'
        native.add(path, x=index)
        native.add(path + '/Mesh', kind='Mesh', x=index + .5)
        poses[path] = matrix(index * 2 + 1)
    fanout = NativePoseFanout(native, native.ovs, max_prims=4, max_depth=4)
    plan = prepare(fanout, poses)
    assert not plan.fallback_reason
    assert len(extras(plan)) == 30
    assert native.hierarchy_queries == 60
    assert native.world_queries == 2
    np.testing.assert_allclose(extras(plan)[path + '/Mesh'], matrix(59.5))


def test_warm_camera_admission_does_not_iterate_scene_instance_inventory():
    native, fanout = helper()
    native.add('/View/Camera', kind='Camera')
    prepare(fanout, {'/View/Camera': matrix(1)})
    class MembershipOnly(frozenset):
        checks = 0
        def __iter__(self):
            pytest.fail('Camera update scanned the instance inventory')
        def __contains__(self, value):
            self.checks += 1
            return super().__contains__(value)
    fanout.instance_paths = MembershipOnly('/Instances/I' + str(n) for n in range(10000))
    for index in range(10):
        assert not prepare(fanout, {'/View/Camera': matrix(index)}).fallback_reason
    assert fanout.instance_paths.checks == 20
    fanout.instance_paths = MembershipOnly(['/View'])
    assert prepare(fanout, {'/View/Camera': matrix(1)}).fallback_reason


def backend_fixture():
    from public_sdk_backend import RendererBackend
    native, fanout = helper()
    events = []
    ready = SimpleNamespace(wait=lambda: None)
    native.write_attribute = lambda query, name, ordinal, value, **kw: events.append(
        ("write", query[0], name, ordinal, np.asarray(value).copy())) or ready
    native.advance_write_floor = lambda ordinal, scope: events.append(("floor", ordinal)) or ready
    native.ovs.Scope = SimpleNamespace(ALL=0)
    native.ovs.AttributeSemantic = SimpleNamespace(NONE=0, MATRIX=5)
    backend = RendererBackend.__new__(RendererBackend)
    backend.stage = native
    backend.ovstage = native.ovs
    backend.np = np
    backend.ordinal = 4
    backend.native_stage_id = "stage"
    backend.native_pose_fanout = fanout
    backend.population_owner = SimpleNamespace(path="accepted.usda")
    backend.pose_population = SimpleNamespace(base="accepted.usda", wrapper=None, poses={},
        apply=lambda *a, **k: pytest.fail("Safe hierarchy must not repopulate USD"))
    backend.desired_values = {}
    backend.simulation_time_ns = 0
    backend.sample_count = 32
    backend.renderer = SimpleNamespace(update_from_stage=lambda ordinal: events.append(("update", ordinal)),
        reset=lambda **kw: events.append(("reset",)))
    return backend, native, events


def test_backend_commits_root_and_owned_descendants_at_one_ordinal_without_population():
    backend, native, events = backend_fixture()
    result = backend.update(rows({"/World/Parent": matrix(10)}))
    writes = [event for event in events if event[0] == "write"]
    assert len(writes) == 4 and {event[3] for event in writes} == {5}
    assert [event[2] for event in writes] == ["omni:resetXformStack", "omni:resetXformStack",
                                             "omni:xform", "omni:xform"]
    assert {event[1] for event in writes} == {"/World/Parent", "/World/Parent/Leaf"}
    assert events[-2:] == [("floor", 5), ("update", 5)]
    assert ("reset",) not in events and result["native_transform_auto_invalidation"]
    assert result["updated_count"] == 2
    assert result["native_pose_route"] == "native_inputs" and result["native_pose_fanout_rows"] == 2
    assert backend.sample_count == 0


def test_unchanged_scene_pose_replay_does_not_write_or_reset():
    backend, native, events = backend_fixture()
    result = backend.update(rows({'/World/Parent': matrix(1)}))
    assert result['updated_count'] == 2 and result['native_pose_route'] == 'unchanged'
    assert result['native_unchanged_poses'] == 1
    assert not events and backend.ordinal == 4 and backend.sample_count == 32
    assert native.hierarchy_queries == 0
    assert '/World/Parent' in backend.native_pose_fanout.accepted_parents
    # Native state may change independently; the cached desired value is never
    # sufficient to discard a new request after that change.
    native.nodes['/World/Parent']['omni:fabric:worldMatrix'] = matrix(2).reshape(1, 16)
    result = backend.update(rows({'/World/Parent': matrix(1)}))
    assert result['native_pose_route'] == 'native_inputs' and events


def test_unchanged_child_still_pins_world_pose_when_its_ancestor_moves():
    native, fanout = helper()
    native.add('/World/Parent/Nested', x=5)
    native.add('/World/Parent/Nested/Leaf', kind='Mesh', x=8)
    plan = prepare(fanout, {'/World/Parent': matrix(10), '/World/Parent/Nested': matrix(5)})
    assert not plan.fallback_reason and not plan.unchanged_poses
    np.testing.assert_allclose(extras(plan)['/World/Parent/Nested/Leaf'], matrix(8))


def test_unchanged_parent_does_not_expand_its_unrelated_subtree():
    native, fanout = helper()
    native.add('/Other', x=5)
    native.add('/Other/Mesh', kind='Mesh', x=8)
    plan = prepare(fanout, {'/World/Parent': matrix(1), '/Other': matrix(10)})
    assert plan.unchanged_poses == {'/World/Parent'}
    assert set(extras(plan)) == {'/Other/Mesh'}
    assert native.hierarchy_queries == 2


def test_batched_world_reads_resolve_shared_prim_lists_and_sparse_rows_once():
    native, fanout = helper()
    names = ['/World/Parent', '/World/Parent/Leaf', '/Unrelated']
    token = native.intern_token('omni:fabric:worldMatrix')
    groups = []
    for prim, data in ((1, 0), (0, 1)):
        storage = np.stack([matrix(3), matrix(1)])
        groups.append(SimpleNamespace(is_delete=False, is_array=False, attribute=token,
            prim_list=names, prim_count=1, data_count=1,
            prim_index=lambda i, prim=prim: prim, data_row_index=lambda i, data=data: data,
            array=lambda i, storage=storage: storage, storage=storage))
    native.read_attributes = lambda *a: nullcontext(SimpleNamespace(wait=lambda:None, groups=lambda:groups))
    resolutions = []
    native.get_path_strings = lambda paths: resolutions.append(paths) or paths
    worlds = fanout._worlds(set(names[:2]), 4, native)
    assert len(resolutions) == 1 and native.released == 2
    np.testing.assert_allclose(worlds['/World/Parent'], matrix(1))
    np.testing.assert_allclose(worlds['/World/Parent/Leaf'], matrix(3))


def test_expanded_descendant_schema_failure_prevents_any_root_write_or_ordinal_commit():
    backend, native, events = backend_fixture()
    # An incompatible descendant column is discovered only after expansion;
    # the valid parent prefix must still remain completely uncommitted.
    native.nodes["/World/Parent/Leaf"]["omni:xform"] = np.ones((1,16), dtype=np.int32)
    with pytest.raises(ValueError, match="floating value cannot replace"):
        backend.update(rows({"/World/Parent": matrix(10)}))
    assert not events and backend.ordinal == 4
    assert not backend.desired_values and not backend.native_pose_fanout.accepted_parents


def test_invalid_original_suffix_prevents_discovery_and_mutation():
    backend, native, events = backend_fixture()
    invalid = dict(prim_path="/World/Parent", attribute_name="inputs:exposure", value_type="Float", value=float("nan"))
    with pytest.raises(ValueError, match="finite numeric"):
        backend.update(rows({"/World/Parent": matrix(10)}) + [invalid])
    assert not events and native.hierarchy_queries == 0


@pytest.mark.parametrize("leaf_type", ["Camera", "Mesh", "RectLight"])
def test_validated_pose_hierarchies_use_native_invalidation_and_camera_history_is_separate(leaf_type):
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Leaf", kind=leaf_type, x=3)
    result = backend.update(rows({"/World/Parent": matrix(10)}))
    assert result["native_camera_history_preserved"] is (leaf_type == "Camera")
    assert ("reset",) not in events
    assert result["native_transform_auto_invalidation"]


def test_direct_camera_leaf_preserves_history_but_mixed_light_edit_resets():
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Leaf", kind="Camera", x=3)
    result = backend.update(rows({"/World/Parent/Leaf": matrix(10)}))
    assert result["native_camera_history_preserved"] and ("reset",) not in events
    events.clear()
    light = dict(prim_path="/World/Parent", attribute_name="inputs:exposure", value_type="Float", value=2.)
    result = backend.update(rows({"/World/Parent/Leaf": matrix(10)}) + [light])
    assert not result["native_camera_history_preserved"] and ("reset",) in events
    assert not result["native_transform_auto_invalidation"]


@pytest.mark.parametrize("kind", ["Mesh", "RectLight", "SphereLight", "Camera"])
def test_validated_direct_leaf_pose_uses_automatic_invalidation(kind):
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Leaf", kind=kind, x=3)
    result = backend.update(rows({"/World/Parent/Leaf": matrix(10)}))
    assert result["native_transform_auto_invalidation"] and ("reset",) not in events
    assert native.hierarchy_queries == 0


@pytest.mark.parametrize("reason", ["unknown_type", "physics", "instance", "missing_instance_api"])
def test_direct_leaf_unsafe_authority_keeps_global_reset(reason):
    backend, native, events = backend_fixture()
    path = "/World/Parent/Leaf"
    if reason == "unknown_type":
        native.add(path, kind="CustomPrim", x=3)
    elif reason == "physics":
        native.add(path, kind="Mesh", x=3, schemas=("PhysicsRigidBodyAPI",))
    elif reason == "instance":
        native.prototype_roots = [path]
    else:
        native.ovs.instancing.available = lambda: False
    result = backend.update(rows({path: matrix(10)}))
    assert not result["native_transform_auto_invalidation"] and ("reset",) in events
    assert result["native_pose_fallback_reason"]


def test_reset_only_without_known_pose_keeps_global_reset():
    backend, native, events = backend_fixture()
    result = backend.update([dict(prim_path="/World/Parent/Leaf",
        attribute_name="omni:resetXformStack", value_type="Bool", value=True)])
    assert not result["native_transform_auto_invalidation"] and ("reset",) in events


@pytest.mark.parametrize("kind", ["Mesh", "Camera"])
def test_direct_pose_without_explicit_world_boundary_keeps_global_reset(kind):
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Leaf", kind=kind, x=3)
    result = backend.update(rows({"/World/Parent/Leaf": matrix(10)})[:1])
    assert not result["native_transform_auto_invalidation"] and ("reset",) in events
    assert not result["native_camera_history_preserved"]


@pytest.mark.parametrize("path", ["/World/Parent", "/World/Parent/Leaf"])
@pytest.mark.parametrize("kind", ["Mesh", "Camera"])
def test_singular_requested_pose_keeps_global_reset(path, kind):
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Leaf", kind=kind, x=3)
    singular = matrix(10)
    singular[0, 0] = 0
    result = backend.update(rows({path: singular}))
    assert not result["native_transform_auto_invalidation"] and ("reset",) in events
    assert not result["native_camera_history_preserved"]


def test_mixed_camera_and_geometry_pose_hierarchy_uses_native_scene_invalidation():
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Camera", kind="Camera", x=3)
    result = backend.update(rows({"/World/Parent": matrix(10)}))
    assert not result["native_camera_history_preserved"] and ("reset",) not in events
    assert result["native_transform_auto_invalidation"]


def test_inherited_previously_explicit_nested_root_updates_fallback_pose_cache():
    native, fanout = helper()
    native.add("/World/Parent/Nested", x=5)
    native.add("/World/Parent/Nested/Leaf", kind="Mesh", x=8)
    first = prepare(fanout, {"/World/Parent/Nested": matrix(20)})
    fanout.accepted(first)
    native.nodes["/World/Parent/Nested"]["omni:fabric:worldMatrix"] = matrix(20).reshape(1,16)
    native.nodes["/World/Parent/Nested/Leaf"]["omni:fabric:worldMatrix"] = matrix(23).reshape(1,16)
    native.nodes["/World/Parent/Nested"]["omni:resetXformStack"] = np.array([True])
    second = prepare(fanout, {"/World/Parent": matrix(10)})
    fanout.accepted(second)
    np.testing.assert_allclose(fanout.accepted_parents["/World/Parent/Nested"], matrix(29))


@pytest.mark.parametrize("leaf_type", ["Camera", "RectLight"])
def test_first_world_input_does_not_double_inherit_camera_or_light_parent(leaf_type):
    backend, native, events = backend_fixture()
    native.add("/World/Parent/Leaf", kind=leaf_type, x=3)
    native.nodes["/World/Parent"]["omni:xform"] = matrix(1).reshape(1,16)
    native.nodes["/World/Parent/Leaf"]["omni:xform"] = matrix(2).reshape(1,16)
    record = native.write_attribute
    def apply(query, name, ordinal, value, **kwargs):
        result = record(query, name, ordinal, value, **kwargs)
        path = query[0]
        node = native.nodes[path]
        node[name] = np.asarray(value).copy()
        # Match the observed native contract: a reset flag alone does not
        # dirty worldMatrix, but a matrix input recomputes it immediately.
        if name == "omni:xform":
            local = node[name].reshape(4,4)
            parent = native.nodes.get(path.rpartition("/")[0])
            if not node["omni:resetXformStack"].reshape(-1)[0] and parent is not None:
                local = local @ parent["omni:fabric:worldMatrix"].reshape(4,4)
            node["omni:fabric:worldMatrix"] = local.reshape(1,16)
        return result
    native.write_attribute = apply
    backend.update(rows({"/World/Parent": matrix(10)}))
    np.testing.assert_allclose(native.nodes["/World/Parent"]["omni:fabric:worldMatrix"].reshape(4,4), matrix(10))
    np.testing.assert_allclose(native.nodes["/World/Parent/Leaf"]["omni:fabric:worldMatrix"].reshape(4,4), matrix(12))


def test_parent_update_receipt_exposes_only_previously_retained_descendant_values():
    backend, native, events = backend_fixture()
    child = '/World/Parent/Leaf'
    explicit = rows({child: matrix(3)})
    backend.desired_values.update({(row['prim_path'], row['attribute_name']): row for row in explicit})
    result = backend.update(rows({'/World/Parent': matrix(10)}))
    derived = result['accepted_derived_values']
    assert result['updated_count'] == 2
    assert {row['prim_path'] for row in derived} == {child}
    assert {row['attribute_name'] for row in derived} == {'omni:xform', 'omni:resetXformStack'}
    pose = next(row for row in derived if row['attribute_name'] == 'omni:xform')
    np.testing.assert_allclose(pose['value'], matrix(12))
    assert backend.desired_values[(child, 'omni:xform')] == pose


def test_parent_update_does_not_publish_unretained_fanout_as_recovery_values():
    backend, native, events = backend_fixture()
    result = backend.update(rows({'/World/Parent': matrix(10)}))
    assert result['native_pose_fanout_rows'] > 0
    assert result['accepted_derived_values'] == []


def test_render_only_rigid_projection_keeps_scaled_visuals_and_rest():
    backend, native, events = backend_fixture()
    native.nodes['/World/Parent']['usd-schemas'] = np.array([native.intern_token('PhysicsRigidBodyAPI')])
    native.nodes['/World/Parent/Leaf']['usd-schemas'] = np.array([native.intern_token('PhysicsCollisionAPI')])
    scaled = matrix(3); scaled[0, 0] = 2
    native.nodes['/World/Parent/Leaf']['omni:fabric:worldMatrix'] = scaled.reshape(1, 16)
    backend.native_pose_fanout = None  # construct through render-stage owner
    first = backend.update(rows({'/World/Parent': matrix(10)}))
    assert first['native_pose_route'] == 'native_inputs'
    assert first['native_transform_auto_invalidation']
    assert ('reset',) not in events
    writes = {event[1]: event[4].reshape(4, 4) for event in events
              if event[0] == 'write' and event[2] == 'omni:xform'}
    expected = scaled.copy(); expected[3, 0] = 12
    np.testing.assert_allclose(writes['/World/Parent/Leaf'], expected)
    for path, world in writes.items():
        native.nodes[path]['omni:fabric:worldMatrix'] = world.reshape(1, 16)
    events.clear()
    backend.update(rows({'/World/Parent': matrix(1)}))
    restored = next(event[4] for event in events if event[0] == 'write'
                    and event[1:3] == ('/World/Parent/Leaf', 'omni:xform'))
    np.testing.assert_allclose(restored.reshape(4, 4), scaled)
    assert ('reset',) not in events
    assert backend.update_diagnostics['usd_reprojections'] == 0


def test_render_only_nested_rigid_requires_explicit_projection_and_invalidates():
    native, _ = helper()
    native.add('/World/Parent/Nested', x=4, schemas=('PhysicsRigidBodyAPI',))
    native.add('/World/Parent/Nested/Visual', kind='Mesh', x=5)
    fanout = NativePoseFanout(native, native.ovs, render_only=True)
    assert 'explicit world pose' in prepare(fanout, {'/World/Parent': matrix(10)}).fallback_reason
    plan = prepare(fanout, {'/World/Parent': matrix(10), '/World/Parent/Nested': matrix(20)})
    assert not plan.fallback_reason
    np.testing.assert_allclose(extras(plan)['/World/Parent/Nested/Visual'], matrix(21))
    native.nodes['/World/Parent/Nested']['usd-schemas'] = np.array([native.intern_token('PhysxDeformableAPI')])
    plan = prepare(fanout, {'/World/Parent/Nested': matrix(30)}, context=('stage', 'replaced'))
    assert 'Unsupported physics' in plan.fallback_reason and not plan.extra_rows


def test_render_only_unknown_physics_has_explicit_population_fallback():
    backend, native, events = backend_fixture()
    native.nodes['/World/Parent']['usd-schemas'] = np.array([native.intern_token('PhysxDeformableAPI')])
    backend.native_pose_fanout = None
    def project(poses, ordinal):
        events.append(('population', poses, ordinal))
        backend.pose_population.poses.update(poses)
    backend.pose_population.apply = project
    result = backend.update(rows({'/World/Parent': matrix(10)}))
    assert result['native_pose_route'] == 'usd_fallback'
    assert 'Unsupported physics hierarchy' in result['native_pose_fallback_reason']
    assert ('reset',) in events and any(row[0] == 'population' for row in events)
    assert backend.update_diagnostics['usd_reprojections'] == 1
    assert backend.update_diagnostics['renderer_resets'] == 1


def structural_fixture():
    native = Native()
    native.add('/Body', schemas=('PhysicsRigidBodyAPI',))
    native.nodes['/Body/Colliders'] = {
        'omni:fabric:worldMatrix': matrix().reshape(1, 16),
        'omni:xform': matrix().reshape(1, 16)}
    native.add('/Body/Colliders/Box', kind='Cube', x=2, schemas=('PhysicsCollisionAPI',))
    native.add('/Body/Visual', kind='Mesh', x=3)
    native.nodes['/Body/Colliders/Box']['omni:fabric:worldMatrix'][0, 0] = 1.5
    return native, NativePoseFanout(native, native.ovs, render_only=True)


def test_untyped_container_traverses_hidden_primitive_and_visible_mesh_preserving_scale():
    native, fanout = structural_fixture()
    plan = prepare(fanout, {'/Body': matrix(10)})
    assert not plan.fallback_reason
    assert set(extras(plan)) == {'/Body/Colliders', '/Body/Colliders/Box', '/Body/Visual'}
    expected = matrix(12); expected[0, 0] = 1.5
    np.testing.assert_array_equal(extras(plan)['/Body/Colliders/Box'], expected)
    np.testing.assert_array_equal(extras(plan)['/Body/Visual'], matrix(13))
    assert native.borrowed == native.released


@pytest.mark.parametrize('invalid', ['world_missing', 'input_missing', 'world_nonfinite',
    'world_shape', 'input_shape', 'local', 'parent_mismatch', 'input_mismatch', 'schema', 'reset', 'leaf'])
def test_untyped_structural_admission_is_conservative(invalid):
    native, fanout = structural_fixture()
    node = native.nodes['/Body/Colliders']
    if invalid == 'world_missing': node.pop('omni:fabric:worldMatrix')
    if invalid == 'input_missing': node.pop('omni:xform')
    if invalid == 'world_nonfinite': node['omni:fabric:worldMatrix'][0, 0] = np.nan
    if invalid == 'world_shape': node['omni:fabric:worldMatrix'] = np.zeros(15)
    if invalid == 'input_shape': node['omni:xform'] = np.zeros(15)
    if invalid == 'local': node['omni:localMatrix'] = matrix(2)
    if invalid == 'parent_mismatch': node['omni:fabric:worldMatrix'] = matrix(2)
    if invalid == 'input_mismatch': node['omni:xform'] = matrix(2)
    if invalid == 'schema': node['usd-schemas'] = np.array([native.intern_token('UnknownAPI')])
    if invalid == 'reset': node['omni:resetXformStack'] = np.array([False])
    if invalid == 'leaf': native.nodes.pop('/Body/Colliders/Box')
    plan = prepare(fanout, {'/Body': matrix(10)})
    assert plan.fallback_reason and not plan.extra_rows


def test_untyped_world_and_identity_are_revalidated_on_every_changed_pose():
    native, fanout = structural_fixture()
    assert not prepare(fanout, {'/Body': matrix(10)}).fallback_reason
    native.nodes['/Body/Colliders']['omni:xform'] = matrix(3)
    assert prepare(fanout, {'/Body': matrix(20)}).fallback_reason
    native.nodes['/Body/Colliders']['omni:xform'] = matrix()
    native.nodes['/Body/Colliders']['usd-schemas'] = np.array([native.intern_token('UnknownAPI')])
    assert prepare(fanout, {'/Body': matrix(30)}).fallback_reason


def test_explicit_untyped_owner_and_non_render_stage_rejected():
    native, fanout = structural_fixture()
    assert prepare(fanout, {'/Body/Colliders': matrix(10)}).fallback_reason
    ordinary = NativePoseFanout(native, native.ovs)
    assert prepare(ordinary, {'/Body': matrix(10)}).fallback_reason



def test_structural_synthetic_reset_is_owned_only_after_accept_and_world_rechecked():
    native, fanout = structural_fixture()
    first = prepare(fanout, {'/Body': matrix(10)})
    assert not first.fallback_reason
    node = native.nodes['/Body/Colliders']
    node['omni:resetXformStack'] = np.array([True])
    assert prepare(fanout, {'/Body': matrix(20)}).fallback_reason
    node.pop('omni:resetXformStack')
    first = prepare(fanout, {'/Body': matrix(10)})
    fanout.accepted(first)
    node['omni:resetXformStack'] = np.array([True])
    node['omni:xform'] = matrix(10)
    node['omni:fabric:worldMatrix'] = matrix(10)
    native.nodes['/Body']['omni:fabric:worldMatrix'] = matrix(10)
    assert not prepare(fanout, {'/Body': matrix(20)}).fallback_reason
    node['omni:resetXformStack'] = np.array([False])
    assert prepare(fanout, {'/Body': matrix(20)}).fallback_reason
    node['omni:resetXformStack'] = np.array([True])
    assert prepare(fanout, {'/Body': matrix(20)}, context=('stage', 'replaced.usda')).fallback_reason


@pytest.mark.parametrize('kind', ['Cube', 'Sphere', 'Capsule', 'Cylinder', 'Cone'])
def test_standard_geometric_primitive_transform_fanout(kind):
    native, fanout = structural_fixture()
    native.nodes['/Body/Colliders/Box']['usd-prim-type'] = np.array([native.intern_token(kind)])
    transform = matrix(4); transform[:2, :2] = ((0, 2), (-3, 0))
    plan = prepare(fanout, {'/Body': transform})
    assert not plan.fallback_reason
    local = matrix(2); local[0, 0] = 1.5
    np.testing.assert_array_equal(extras(plan)['/Body/Colliders/Box'], local @ transform)


@pytest.mark.parametrize('alteration', ['remove_reset', 'typed', 'schema'])
def test_accepted_structural_identity_cannot_change(alteration):
    native, fanout = structural_fixture()
    plan = prepare(fanout, {'/Body': matrix(10)}); fanout.accepted(plan)
    node = native.nodes['/Body/Colliders']
    node['omni:resetXformStack'] = np.array([True])
    if alteration == 'remove_reset': node.pop('omni:resetXformStack')
    if alteration == 'typed': node['usd-prim-type'] = np.array([native.intern_token('Xform')])
    if alteration == 'schema': node['usd-schemas'] = np.array([native.intern_token('UnknownAPI')])
    assert prepare(fanout, {'/Body': matrix(20)}).fallback_reason


def test_untyped_input_is_local_identity_until_accepted_world_reset_under_nonidentity_parent():
    native, fanout = structural_fixture()
    initial = matrix(3); initial[0, 0] = 1.2; initial[1, 1] = .8
    node = native.nodes['/Body/Colliders']
    native.nodes['/Body']['omni:fabric:worldMatrix'] = initial.copy()
    node['omni:fabric:worldMatrix'] = initial.copy()
    for path in ('/Body/Colliders/Box', '/Body/Visual'):
        native.nodes[path]['omni:fabric:worldMatrix'] = native.nodes[path]['omni:fabric:worldMatrix'].reshape(4, 4) @ initial
    target = matrix(10)
    first = prepare(fanout, {'/Body': target})
    assert not first.fallback_reason
    expected = matrix(2); expected[0, 0] = 1.5
    np.testing.assert_allclose(extras(first)['/Body/Colliders/Box'], expected @ target, atol=1e-12)
    # An unowned world-looking input is not evidence of local inheritance.
    node['omni:xform'] = initial.copy()
    assert prepare(fanout, {'/Body': target}).fallback_reason
    node['omni:xform'] = matrix()
    first = prepare(fanout, {'/Body': target}); fanout.accepted(first)
    node['omni:resetXformStack'] = np.array([True])
    node['omni:xform'] = target.copy(); node['omni:fabric:worldMatrix'] = target.copy()
    native.nodes['/Body']['omni:fabric:worldMatrix'] = target.copy()
    assert not prepare(fanout, {'/Body': initial}).fallback_reason
    node['omni:xform'] = matrix()
    assert prepare(fanout, {'/Body': initial}).fallback_reason


def test_nested_untyped_inheritance_remains_exact_across_nonidentity_updates():
    native, fanout = structural_fixture()
    initial = matrix(3); initial[0, 0] = 1.2; initial[1, 1] = .8
    native.nodes['/Body']['omni:fabric:worldMatrix'] = initial.copy()
    native.nodes['/Body/Colliders']['omni:fabric:worldMatrix'] = initial.copy()
    native.nodes['/Body/Colliders/Nested'] = {
        'omni:fabric:worldMatrix': initial.copy(), 'omni:xform': matrix()}
    native.nodes['/Body/Colliders/Nested/Box'] = native.nodes.pop('/Body/Colliders/Box')
    native.nodes['/Body/Colliders/Nested/Box']['omni:fabric:worldMatrix'] = matrix(2) @ initial
    for target in (matrix(10), initial, matrix(-4), initial):
        plan = prepare(fanout, {'/Body': target})
        assert not plan.fallback_reason
        projected = extras(plan)
        np.testing.assert_array_equal(projected['/Body/Colliders'], target)
        np.testing.assert_array_equal(projected['/Body/Colliders/Nested'], target)
        fanout.accepted(plan)
        native.nodes['/Body']['omni:fabric:worldMatrix'] = target.copy()
        for path, world in projected.items():
            native.nodes[path]['omni:fabric:worldMatrix'] = world.copy()
            native.nodes[path]['omni:xform'] = world.copy()
            native.nodes[path]['omni:resetXformStack'] = np.array([True])


def pinned_after_first_move():
    """Accept a parent move and leave the native stage as the writes left it."""
    native, fanout = helper()
    fanout.accepted(prepare(fanout, {"/World/Parent": matrix(10)}))
    native.nodes["/World/Parent"]["omni:fabric:worldMatrix"] = matrix(10).reshape(1, 16)
    native.nodes["/World/Parent/Leaf"]["omni:fabric:worldMatrix"] = matrix(12).reshape(1, 16)
    native.nodes["/World/Parent/Leaf"]["omni:resetXformStack"] = np.array([True])
    return native, fanout


def test_scoped_overlay_successor_still_fans_out_to_synthetically_pinned_child():
    native, fanout = pinned_after_first_move()
    # An unrelated object add arrives as a scoped overlay; the stage survives.
    native.add("/World/__ovrtx/generation_1/Plane", kind="Mesh", x=7)
    successor = successor_fanout(fanout, ["/World/__ovrtx"])
    plan = prepare(successor, {"/World/Parent": matrix(1)}, context=("stage", "overlay.usda"))
    assert not plan.fallback_reason
    np.testing.assert_allclose(extras(plan)["/World/Parent/Leaf"], matrix(3))


def test_fresh_fanout_mistakes_synthetic_reset_for_authored_boundary():
    # The defect a successor avoids: native state alone reads as a boundary.
    native, _fanout = pinned_after_first_move()
    plan = prepare(NativePoseFanout(native, native.ovs), {"/World/Parent": matrix(1)},
                   context=("stage", "overlay.usda"))
    assert "/World/Parent/Leaf" not in extras(plan)


@pytest.mark.parametrize("changed", [["/World/Parent/Leaf"], ["/World/Parent"], ["/World"]])
def test_successor_rediscovers_prims_inside_changed_subtrees(changed):
    _native, fanout = pinned_after_first_move()
    successor = successor_fanout(fanout, changed)
    kept = {} if successor is None else successor.synthetic_identities
    root = changed[0]
    assert not any(path == root or path.startswith(root + "/") for path in kept)
    assert "/World/Parent/Leaf" not in kept


def test_replacement_or_unowned_predecessor_has_no_successor():
    _native, fanout = pinned_after_first_move()
    assert successor_fanout(fanout, None) is None
    assert successor_fanout(None, ["/World/__ovrtx"]) is None
    native, unused = helper()
    assert successor_fanout(unused, ["/World/__ovrtx"]) is None


def test_successor_records_survive_only_its_first_context():
    native, fanout = pinned_after_first_move()
    successor = successor_fanout(fanout, ["/World/__ovrtx"])
    successor.accepted(prepare(successor, {"/World/Parent": matrix(1)}, context=("stage", "overlay.usda")))
    assert successor.synthetic_identities == {"/World/Parent": ("Xform", False),
                                              "/World/Parent/Leaf": ("Mesh", False)}
    prepare(successor, {"/World/Parent": matrix(2)}, context=("stage", "replaced.usda"))
    assert successor.synthetic_identities == {}


def test_scoped_overlay_successor_keeps_explicitly_pinned_child_following_its_parent():
    # Physics presentation writes a body's visual child as an explicit world
    # input; an unrelated add must not turn that pin into an authored boundary.
    native, fanout = helper()
    fanout.accepted(prepare(fanout, {"/World/Parent/Leaf": matrix(3)}))
    native.nodes["/World/Parent/Leaf"]["omni:resetXformStack"] = np.array([True])
    assert fanout.synthetic_identities == {"/World/Parent/Leaf": ("Mesh", False)}
    successor = successor_fanout(fanout, ["/World/__ovrtx"])
    plan = prepare(successor, {"/World/Parent": matrix(11)}, context=("stage", "overlay.usda"))
    assert not plan.fallback_reason
    np.testing.assert_allclose(extras(plan)["/World/Parent/Leaf"], matrix(13))


def test_usd_fallback_does_not_claim_native_pins_on_xform_poses():
    native, fanout = helper()
    native.add("/World/Parent/Nested", x=5)
    native.add("/World/Parent/Nested/Leaf", kind="Mesh", x=8)
    native.prototype_roots = ["/World/Parent/Nested"]
    plan = prepare(fanout, {"/World/Parent": matrix(10), "/World/Parent/Leaf": matrix(3)})
    assert plan.fallback_reason
    assert plan.pinned_identities == {"/World/Parent/Leaf": ("Mesh", False)}


def test_successor_keeps_nested_rigid_body_admission_for_explicit_pins():
    native, _unused = helper()
    native.add("/World/Parent/Nested", x=4, schemas=("PhysicsRigidBodyAPI",))
    native.add("/World/Parent/Nested/Visual", kind="Mesh", x=5)
    fanout = NativePoseFanout(native, native.ovs, render_only=True)
    fanout.accepted(prepare(fanout, {"/World/Parent/Nested": matrix(6)}))
    native.nodes["/World/Parent/Nested"]["omni:resetXformStack"] = np.array([True])
    successor = successor_fanout(fanout, ["/World/__ovrtx"])
    plan = prepare(successor, {"/World/Parent": matrix(10)}, context=("stage", "overlay.usda"))
    assert "Nested rigid body requires explicit world pose" in plan.fallback_reason
    assert not plan.extra_rows
