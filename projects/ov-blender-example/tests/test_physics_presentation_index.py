"""Identity lookup is linear per writable transaction, never per body."""
from types import SimpleNamespace as NS
import pytest
from ovrtx_blender_example.physics_pose_presentation import ResolutionIndex


class Counted:
    def __init__(self, items): self.items, self.scans = items, 0
    def __iter__(self): self.scans += 1; return iter(self.items)


def test_object_resolution_is_one_scene_pass_and_invalidates_each_transaction():
    objects = Counted([NS(session_uid=i, ov=NS(physics=NS(persistent_id=str(i)))) for i in range(1000)])
    scene = NS(objects=objects)
    index = ResolutionIndex(scene)
    for i in range(1000):
        target = NS(object_session_uid=i, object_identity=str(i), bone_identity='')
        assert index.resolve(target)[0] is objects.items[i]
    assert objects.scans == 1
    objects.items[-1].ov.physics.persistent_id = 'replaced'
    with pytest.raises(ValueError, match='stale'):
        ResolutionIndex(scene).resolve(target)
    objects.items.pop()
    with pytest.raises(ValueError, match='stale'):
        ResolutionIndex(scene).resolve(target)


def test_bone_lookup_scans_once_and_new_transaction_follows_rename_or_rejects_ambiguity():
    bones = Counted([NS(name=str(i), ov=NS(persistent_id=str(i))) for i in range(100)])
    channels = {b.name: object() for b in bones.items}
    obj = NS(session_uid=42, ov=NS(physics=NS(persistent_id='rig')), data=NS(bones=bones), pose=NS(bones=channels))
    scene = NS(objects=[obj]); index = ResolutionIndex(scene)
    for i in range(100):
        target = NS(object_session_uid=42, object_identity='rig', bone_identity=str(i))
        assert index.resolve(target)[1] is channels[str(i)]
    assert bones.scans == 1
    bones.items[-1].name = 'renamed'; channels['renamed'] = object()
    assert ResolutionIndex(scene).resolve(target)[1] is channels['renamed']
    bones.items.append(NS(name='duplicate',ov=NS(persistent_id='99')))
    with pytest.raises(ValueError, match='ambiguous'):
        ResolutionIndex(scene).resolve(target)
