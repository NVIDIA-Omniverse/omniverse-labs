# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / "runtime" / "ovstage_scene_updates.py"
SPEC = importlib.util.spec_from_file_location("scene_updates_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Population:
    def __init__(self):
        self.calls = []
        self.next = 0
        self.fail = None

    def open_usd_from_string(self, stage, text, **kwargs):
        self.calls.append(("open", stage, text, kwargs))

    def add_usd_reference(self, stage, path, target):
        if self.fail == "add":
            raise RuntimeError("parse failed")
        self.next += 1
        self.calls.append(("add", stage, path, target, self.next))
        return self.next

    def add_usd_reference_from_string(self, stage, text, target):
        return self.add_usd_reference(stage, text, target)

    def remove_usd(self, stage, handle):
        self.calls.append(("remove", stage, handle))

    def apply_usd_changes(self, stage, **kwargs):
        self.calls.append(("apply", stage, kwargs))
        if self.fail == "apply":
            self.fail = None
            raise RuntimeError("apply failed")
        if self.fail == "always_apply":
            raise RuntimeError("apply always fails")

    def reset_usd(self, stage):
        self.calls.append(("reset", stage))


class Stage:
    def advance_write_floor(self, ordinal):
        self.floor = ordinal
        return self

    def wait(self):
        pass


def initialized(tmp_path):
    first, second = tmp_path / "one.usda", tmp_path / "two.usda"
    first.write_text("#usda 1.0\n")
    second.write_text("#usda 1.0\n")
    pop, stage = Population(), Stage()
    owner = MODULE.SceneGenerationPopulation(stage, population=pop)
    owner.initialize(first, domains=7)
    return owner, pop, second


def test_inline_sources_survive_reset_with_new_handles_and_same_opinions(tmp_path):
    import sys
    sys.path.insert(0, str(PATH.parent))
    owner, pop, second = initialized(tmp_path)
    snapshot = dict(structure=('#usda 1.0\nover "Owned" {}\n', ('/Owned',)),
        materials={'/Material': '#usda 1.0\nover "Material" {}\n'},
        visibility=({'/World': 'invisible'}, '#usda 1.0\nover "World" { token visibility = "invisible" }\n'),
        instances=[dict(owner_id=7, projection_path='/World/_DirectGeometry/Object_7',
            hidden_source_path='/World/Source', layer_body='#usda 1.0\nover "World" {}\n')])
    owner.restore_inline_sources(snapshot)
    accepted = owner.snapshot_inline_sources()
    handles = (owner.structure.handle, owner.visibility.handle,
               owner.materials.handles['/Material'], owner.instance_graphs.entries[7].handle)
    owner.restore(owner.path, ordinal=owner.ordinal + 1)
    owner.restore_inline_sources(accepted)
    assert owner.snapshot_inline_sources() == snapshot
    assert not set(handles) & {owner.structure.handle, owner.visibility.handle,
        owner.materials.handles['/Material'], owner.instance_graphs.entries[7].handle}


def test_source_replacement_projects_once_to_same_stage(tmp_path):
    owner, pop, second = initialized(tmp_path)
    stage, old = owner.stage, owner.handle
    pop.calls.clear()
    result = owner.replace(second, ordinal=10)
    assert [call[0] for call in pop.calls] == ["add", "remove", "apply"]
    assert all(call[1] is stage for call in pop.calls)
    assert pop.calls[1][2] == old
    assert result["ordinal"] == 10 and owner.path == second


def test_same_source_replacement_rebuilds_instead_of_duplicate_reference_removal(tmp_path):
    owner, pop, second = initialized(tmp_path)
    previous = owner.handle
    pop.calls.clear()
    result = owner.replace(owner.path, ordinal=owner.ordinal + 1)
    assert [call[0] for call in pop.calls] == ['reset', 'open', 'add', 'apply']
    assert owner.handle != previous and result['ordinal'] == owner.ordinal == owner.stage.floor


def test_parse_failure_keeps_previous_generation(tmp_path):
    owner, pop, second = initialized(tmp_path)
    old = (owner.handle, owner.path, owner.ordinal)
    pop.fail = "add"
    with pytest.raises(RuntimeError, match="parse"):
        owner.replace(second, ordinal=3)
    assert (owner.handle, owner.path, owner.ordinal) == old
    assert not owner.failed


def test_apply_failure_restores_predecessor_before_raising(tmp_path):
    owner, pop, second = initialized(tmp_path)
    old = owner.path
    pop.fail = "apply"
    with pytest.raises(MODULE.SceneGenerationUpdateError, match="predecessor restored") as failure:
        owner.replace(second, ordinal=3)
    assert failure.value.rollback_succeeded
    assert not owner.failed and owner.path == old
    assert owner.ordinal == owner.stage.floor == 5
    assert [call[0] for call in pop.calls][-4:] == ["reset", "open", "add", "apply"]
    owner.replace(second, ordinal=6)
    assert owner.path == second


def test_failed_restoration_requires_explicit_recovery(tmp_path):
    owner, pop, second = initialized(tmp_path)
    pop.fail = "always_apply"
    with pytest.raises(MODULE.SceneGenerationUpdateError) as failure:
        owner.replace(second, ordinal=3)
    assert not failure.value.rollback_succeeded and owner.failed
    with pytest.raises(RuntimeError, match="recovery"):
        owner.replace(second, ordinal=5)


def test_reject_stale_ordinal_before_source_mutation(tmp_path):
    owner, pop, second = initialized(tmp_path)
    count = len(pop.calls)
    with pytest.raises(ValueError, match="increase"):
        owner.replace(second, ordinal=owner.ordinal)
    assert len(pop.calls) == count


def test_native_replacement_removes_omitted_attribute(tmp_path):
    ovs = pytest.importorskip("ovstage")
    np = pytest.importorskip("numpy")
    template = '''#usda 1.0
def Mesh "Mesh" {
    point3f[] points = [(0,0,0),(1,0,0),(0,1,0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0,1,2]
    ATTR
}
'''
    first, second = tmp_path/"first.usda", tmp_path/"second.usda"
    first.write_text(template.replace("ATTR", "color3f[] primvars:displayColor = [(0.2,0.3,0.4)]"))
    second.write_text(template.replace("ATTR", ""))
    stage = ovs.Stage("scene-generation-attribute-test")
    owner = MODULE.SceneGenerationPopulation(stage)
    results = []
    try:
        for index, source in enumerate((first, second)):
            owner.initialize(source) if index == 0 else owner.replace(source, ordinal=3)
            stage.advance_write_floor(owner.ordinal).wait()
            found = []
            with ovs.PathDictionary(stage) as paths:
                with paths.create_path_list_from_strings(["/Mesh"]) as plist:
                    with stage.query_from_path_list(plist) as query:
                        with stage.read_attributes(query, [paths.intern_token("primvars:displayColor")],
                                                   ovs.OrdinalRange.latest(owner.ordinal)) as read:
                            read.wait()
                            for group in read.groups():
                                try:
                                    if not group.is_delete:
                                        found.append(np.asarray(group.array(0)).copy().tolist())
                                finally:
                                    stage.release_group(group)
            results.append(found)
        assert np.allclose(results[0][0], [0.2,0.3,0.4])
        assert results[1] == []
    finally:
        stage.destroy()


def test_sparse_overlay_keeps_base_handle_and_acknowledges_desired_source(tmp_path):
    owner,pop,second=initialized(tmp_path)
    patch=tmp_path/'patch.usda';patch.write_text('#usda 1.0\n')
    previous=owner.path;base=owner.handle;pop.calls.clear()
    result=owner.apply_delta(patch,predecessor_path=previous,desired_path=second,ordinal=3)
    assert [c[0] for c in pop.calls]==['add','apply']
    assert owner.handle==base and owner.path==second and len(owner.overlays)==1
    assert result['population_update']=='scoped_overlay'
    owner.replace(previous,ordinal=4)
    assert not owner.overlays
    assert [c[0] for c in pop.calls][-4:]==['reset','open','add','apply']
    assert owner.base_path == previous and not owner.overlay_paths


def test_repeated_overlay_replaces_complete_desired_scene_after_undo(tmp_path):
    owner, pop, second = initialized(tmp_path)
    first = owner.path
    patch_a, patch_b = tmp_path / 'a.usda', tmp_path / 'b.usda'
    for path in (patch_a, patch_b):
        path.write_text('#usda 1.0\n')
    owner.apply_delta(patch_a, predecessor_path=first, desired_path=second, ordinal=3)
    owner.apply_delta(patch_b, predecessor_path=second, desired_path=first, ordinal=4)
    pop.calls.clear()
    result = owner.apply_delta(patch_a, predecessor_path=first, desired_path=second, ordinal=5)
    assert result['population_update'] == 'replacement'
    assert result['population_fallback_reason'] == 'reused_overlay_source'
    assert all(call[2] != str(patch_a) for call in pop.calls if call[0] == 'add')
    assert owner.path == owner.base_path == second
    assert not owner.overlays and not owner.overlay_paths


def test_repeated_overlay_failed_replacement_restores_complete_predecessor(tmp_path):
    owner, pop, second = initialized(tmp_path)
    first = owner.path
    patch = tmp_path / 'a.usda'
    patch.write_text('#usda 1.0\n')
    owner.apply_delta(patch, predecessor_path=first, desired_path=second, ordinal=3)
    third = tmp_path / 'three.usda'
    third.write_text('#usda 1.0\n')
    pop.fail = 'apply'
    with pytest.raises(MODULE.SceneGenerationUpdateError) as failure:
        owner.apply_delta(patch, predecessor_path=second, desired_path=third, ordinal=4)
    assert failure.value.rollback_succeeded and owner.path == owner.base_path == second
    assert not owner.overlays and not owner.overlay_paths


def test_sparse_overlay_rejects_stale_base_before_mutation(tmp_path):
    owner,pop,second=initialized(tmp_path);before=len(pop.calls)
    with pytest.raises(ValueError,match='stale'):
        owner.apply_delta(second,predecessor_path=second,desired_path=second,ordinal=3)
    assert len(pop.calls)==before


def test_sparse_overlay_failure_restores_complete_predecessor(tmp_path):
    owner,pop,second=initialized(tmp_path);previous=owner.path
    pop.fail='apply'
    with pytest.raises(MODULE.SceneGenerationUpdateError) as failure:
        owner.apply_delta(second,predecessor_path=previous,desired_path=second,ordinal=3)
    assert failure.value.rollback_succeeded and owner.path==previous and not owner.overlays


def test_delta_after_camera_retarget_uses_accepted_logical_source(tmp_path):
    owner,pop,second=initialized(tmp_path)
    desired=tmp_path/'desired.usda';desired.write_text('#usda 1.0\n')
    patch=tmp_path/'patch.usda';patch.write_text('#usda 1.0\n')
    backing=owner.path
    owner.apply_delta(patch,predecessor_path=second,expected_predecessor_path=second,
                      desired_path=desired,ordinal=3)
    assert owner.path==desired and owner.handle==1 and backing!=second


@pytest.mark.parametrize('helper', ['visibility', 'structure', 'instance_graphs', 'materials'])
@pytest.mark.parametrize('delta', [False, True])
def test_reset_invalidates_inline_handles_even_if_clear_failed(tmp_path, helper, delta):
    from types import SimpleNamespace
    owner, pop, second = initialized(tmp_path)
    previous = owner.path
    def reject_clear():
        raise RuntimeError('inline source removal failed')
    setattr(owner, helper, SimpleNamespace(clear=reject_clear))
    with pytest.raises(MODULE.SceneGenerationUpdateError) as failure:
        if delta:
            owner.apply_delta(second, predecessor_path=previous, desired_path=second, ordinal=3)
        else:
            owner.replace(second, ordinal=3)
    assert failure.value.rollback_succeeded
    assert getattr(owner, helper) is None
    owner.replace(second, ordinal=owner.ordinal + 1)
    assert owner.path == second


def test_predecessor_restoration_resets_unknown_candidate_references(tmp_path):
    owner, pop, second = initialized(tmp_path)
    accepted = owner.path
    unknown = pop.add_usd_reference(owner.stage, str(second), '/')
    pop.calls.clear()
    owner.restore(accepted, ordinal=owner.ordinal + 1)
    assert [call[0] for call in pop.calls] == ['reset', 'open', 'add', 'apply']
    assert owner.path == accepted and owner.handle != unknown
    assert owner.ordinal == owner.stage.floor and not owner.overlays


def test_failed_explicit_restore_quarantines_population(tmp_path):
    owner, pop, second = initialized(tmp_path)
    pop.fail = 'always_apply'
    with pytest.raises(RuntimeError):
        owner.restore(owner.path, ordinal=owner.ordinal + 1)
    assert owner.failed
    with pytest.raises(RuntimeError, match='recovery'):
        owner.replace(second, ordinal=owner.ordinal + 3)
