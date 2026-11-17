# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject false source coverage even when total submitted instance counts agree."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as State

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'benchmarks/application_evidence.py'
spec = importlib.util.spec_from_file_location('application_evidence', SOURCE)
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


@pytest.fixture
def renderer():
    mesh = State(count=2, revision=7, vertices=State(size=72), indices=State(size=12),
                 transparent=False, double_sided=False, wireframe=False)
    snapshot = State(object_ids=np.array([22, 33]), shape_ids=np.array([23, 35]),
                     geometry_ids=np.array([24, 34]), kinds=np.array([5, 5]),
                     meshes=[State(id=7, topology_id=42)])
    return State(snapshot=snapshot, _visible_rows=np.array([0, 1]), _groups=[(5, mesh)],
                 _group_rows=[(5, 7, 0, np.array([0, 1]))], meshes={(5, 42, 0): mesh})


def test_shared_mesh_keeps_both_native_shape_identities(renderer):
    result = evidence.submitted_geometry(renderer)
    assert result['submitted_geometry']['complete']
    assert result['submitted_shape_ids'] == [23, 35]
    assert result['visible_shape_records'] == [[22, 23, 24, 5], [33, 35, 34, 5]]
    assert len(result['geometry_records']) == 1  # One real instanced draw group.


def test_equal_count_with_one_omitted_and_one_duplicated_source_row_rejects(renderer):
    renderer._group_rows[0] = (5, 7, 0, np.array([0, 0]))
    with pytest.raises(RuntimeError, match='omitted or duplicated'):
        evidence.submitted_geometry(renderer)


def test_same_count_from_a_different_gpu_cache_object_rejects(renderer):
    renderer.meshes[(5, 42, 0)] = State(**vars(renderer._groups[0][1]))
    with pytest.raises(RuntimeError, match='GPU object does not match'):
        evidence.submitted_geometry(renderer)


def test_stale_uploaded_vertex_revision_rejects(renderer):
    renderer._groups[0][1].revision = 6
    with pytest.raises(RuntimeError, match='different native vertex revision'):
        evidence.submitted_geometry(renderer)


def test_missing_draw_group_rejects(renderer):
    renderer._groups.clear()
    with pytest.raises(RuntimeError, match='differs from actual submitted groups'):
        evidence.submitted_geometry(renderer)
