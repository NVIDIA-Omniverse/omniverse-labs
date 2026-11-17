# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atomic evidence Go uses real native views; these sinks are not GUI QA."""
from dataclasses import replace
import json
import pytest

from test_application_presentation import app, Value
from test_evidence_binding import prepared, settle
from ovd_ui.evidence_binding import EvidenceController


@pytest.fixture(params=["left", "right"], ids=["left-displayed", "right-displayed"])
def displayed_evidence(app, tmp_path, request):
    loaded, private, document = prepared(tmp_path)
    app.evidence_controller = EvidenceController()
    app.pending_evidence = None
    app.workflow.display_session = loaded.session
    app.workflow.display_path = loaded.source_path
    app.workflow.display_data_path = loaded.data_path
    app.workflow.verified_payload = loaded.verified_payload
    app._display(loaded.source_path, loaded.session)
    path = tmp_path / "evidence.json";path.write_text(json.dumps(document))
    app.evidence_controller.import_file(path, app._evidence_context())
    settle(app.evidence_controller)
    selected = next(m for m in app.evidence_controller.markers if m.role == request.param)
    index = next(i for i,s in enumerate(app.scenes) if s.id == selected.scene_id)
    app._select_scene(Value(index), None)
    app.seek(1, snap=False)
    try:
        yield app
    finally:
        app.evidence_controller.close();loaded.close()


def test_marker_selection_does_not_seek_and_go_adopts_exact_frame_object(displayed_evidence):
    app = displayed_evidence
    marker = next(m for m in app.evidence_controller.markers if m.scene_id == app.scene_id)
    before = app.frame
    app._select_evidence_marker(marker)
    assert app.frame == before == 1
    assert app._go_evidence()
    assert app.frame == app.playback.frame == app.presentation.frame == 2
    assert app.selection.get_snapshot().paths() == [f"/objects/{marker.instance_id}"]
    assert app.presentation.evidence_sample_stamp(marker.reads[0].instance_id, marker.reads[0].property_id).frame == 2


def test_both_role_scene_clip_and_stale_action_prechecks_preserve_playback(displayed_evidence):
    app = displayed_evidence
    app.playback.direction = 1
    app.evidence_controller.select("pair")
    assert not app._go_evidence() and app.playback.direction == 1
    current = next(m for m in app.evidence_controller.markers if m.scene_id == app.scene_id)
    other = next(m for m in app.evidence_controller.markers if m.role != current.role)
    app.evidence_controller.select("pair", other.role)
    assert not app._go_evidence() and "scene" in app.status_label.text.lower()
    assert app.frame == 1 and app.playback.direction == 1
    app.evidence_controller.select("pair", current.role)
    key = app._evidence_action_key()
    count = len(app.playback.clips)
    app.playback.clip.stop = 2
    assert not app._go_evidence(key)
    assert not app._go_evidence()
    assert len(app.playback.clips) == count and app.frame == 1 and app.playback.direction == 1


def test_failed_frozen_identity_check_never_publishes_selection(displayed_evidence):
    app = displayed_evidence
    controller = app.evidence_controller
    current = next(m for m in controller.markers if m.scene_id == app.scene_id)
    controller.select("pair", current.role)
    stored = controller.sets[controller.selected_set]
    marker = controller.selected_marker
    bad = replace(marker, identity=dict(marker.identity, object_uid=999999))
    stored.binding = replace(stored.binding, markers=tuple(bad if m == marker else m for m in stored.binding.markers))
    before = tuple(app.selection.get_snapshot().paths())
    assert not app._go_evidence()
    assert "identity changed" in app.status_label.text.lower()
    assert app.frame == app.playback.frame == 1
    assert tuple(app.selection.get_snapshot().paths()) == before
    assert next(s for s in app.session.scenes() if s.id == app.scene_id).current_frame == 1
    assert app.pending_evidence is None


@pytest.mark.parametrize("stale", [False, True])
def test_deferred_publication_holds_selection_until_exact_apply(displayed_evidence, stale):
    app = displayed_evidence
    native, previous = app.session, app.presentation
    class Deferred:
        request_applied = False
        def __getattr__(self, name):return getattr(previous, name)
    class Source:
        defer = True
        def __getattr__(self, name):return getattr(native, name)
        def presentation(self, *args, **kwargs):
            if self.defer:
                self.defer = False
                return Deferred()
            return native.presentation(*args, **kwargs)
    app.session = app.workflow.display_session = Source()
    current = next(m for m in app.evidence_controller.markers if m.scene_id == app.scene_id)
    app.evidence_controller.select("pair", current.role)
    before = tuple(app.selection.get_snapshot().paths())
    assert not app._go_evidence()
    assert app.pending_evidence is not None and app.frame == 1
    assert tuple(app.selection.get_snapshot().paths()) == before
    if stale:app.playback.clip.stop = 2
    assert app._retry_evidence_go() is (not stale)
    assert app.frame == (1 if stale else 2)
    assert app.pending_evidence is None
