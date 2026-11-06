"""A later anchor error rolls back scene-wide identities and repair notices."""
from copy import deepcopy
from types import SimpleNamespace as N

import pytest

from ovrtx_blender_example import physics_authoring as authoring
from ovrtx_blender_example import scene_generation_sessions as sessions, user_messages


def owner(name, identity, *, joint=False):
    frame = lambda: N(translation=(0., 0., 0.), rotation=(1., 0., 0., 0.))
    return N(name=name, ov=N(physics=N(persistent_id=identity,
        rigid_body=N(persistent_id=identity), joint=N(persistent_id='joint-' + identity,
        enabled=joint, frame_a=frame(), frame_b=frame(), committed_handle=frame(), frames_initialized=True))))


@pytest.mark.parametrize('history_exists', [False, True])
@pytest.mark.parametrize('fail_second', [False, True])
def test_pending_batch_identity_history_frames_and_notice_are_atomic(monkeypatch, history_exists, fail_second):
    original, duplicate = owner('original', 'shared'), owner('copy', 'shared')
    first, second = owner('first', 'a', joint=True), owner('second', 'b', joint=True)
    scene = N(session_uid=5, objects=[original, duplicate, first, second])
    history = {9: {'unrelated': 100}}
    if history_exists:
        history[5] = {'shared': 1}
    saved = deepcopy(history)
    monkeypatch.setattr(sessions, '_identity_owners', history)
    monkeypatch.setattr(authoring, 'require_scene_ready', lambda *a, **k: None)
    monkeypatch.setattr(authoring, 'anchor_pending', lambda obj: True)
    dirty, notices = [], []
    monkeypatch.setattr(authoring, 'mark_dirty', lambda scene, obj: dirty.append(obj.name))
    monkeypatch.setattr(user_messages, 'report_info', lambda message, **k: notices.append(message))
    def apply(obj, scene, depsgraph, **kwargs):
        assert kwargs['report_repair'] is False and kwargs['mark_scene'] is False
        if obj is second and fail_second:
            raise ValueError('later endpoint rejects')
        duplicate.ov.physics.persistent_id = 'repaired'
        duplicate.ov.physics.rigid_body.persistent_id = 'repaired-body'
        duplicate.ov.physics.joint.persistent_id = 'repaired-joint'
        sessions._identity_owners[5] = {'shared': 1, 'repaired': 2}
        obj.ov.physics.joint.frame_a.translation = (1., 2., 3.)
        obj.ov.physics.joint.frames_initialized = False
        return ('copy',)
    monkeypatch.setattr(authoring, 'apply_anchor', apply)
    if fail_second:
        with pytest.raises(ValueError, match='later endpoint'):
            authoring.apply_pending_anchors_for_playback(scene)
        assert (duplicate.ov.physics.persistent_id, duplicate.ov.physics.rigid_body.persistent_id,
                duplicate.ov.physics.joint.persistent_id) == ('shared', 'shared', 'joint-shared')
        assert sessions._identity_owners == saved
        assert first.ov.physics.joint.frame_a.translation == (0., 0., 0.)
        assert first.ov.physics.joint.frames_initialized is True
        assert dirty == notices == []
    else:
        assert authoring.apply_pending_anchors_for_playback(scene) == ('first', 'second')
        assert duplicate.ov.physics.persistent_id == 'repaired'
        assert dirty == ['first', 'second']
        assert len(notices) == 1 and notices[0].endswith('copy')
        assert notices[0].count('copy') == 1
    assert original.ov.physics.persistent_id == 'shared'
    assert sessions._identity_owners[9] == saved[9]
