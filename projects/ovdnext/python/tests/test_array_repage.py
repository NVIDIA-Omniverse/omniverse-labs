# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit re-paging retains source identities and rejects obsolete containers."""
from pathlib import Path

from _pvd_native import Session
from ovd_ui.adapters import PvdDataAdapter
from ovd_ui.array_settings import ArrayDisplaySettings
from ovd_ui.models import ObjectTreeModel

ROOT=Path(__file__).resolve().parents[2]


def test_actual_ue4_repage_preserves_explicit_rows_and_all_native_references(tmp_path):
    source=Session(str(ROOT/'test/data/ue4_shooter.pxd2'),str(tmp_path))
    scene=next(s for s in source.scenes() if s.stream_id==686661696)
    adapter=PvdDataAdapter(source.presentation(scene.id,30))
    model=ObjectTreeModel();model.populate(adapter)
    keys=[next(key for key,node in model._specs.items() if node.kind==kind)
          for kind in ('object','reference','group','page')]
    identities=model.row_identities(keys)
    native=model._item(keys[0]);native_before={key for key,node in model._specs.items() if node.kind=='object'}
    before_cache=len(model.cache)
    # max=0 suppresses property expansion only, never hierarchy ownership refs.
    restored=model.set_array_settings(ArrayDisplaySettings(20,0),selected_rows=identities)
    assert [item.key for item in restored]==keys[:3]
    assert model._item(keys[0]) is native
    assert {key for key,node in model._specs.items() if node.kind=='object'}==native_before
    assert len(model.cache)<before_cache+20
    assert model.restore_rows([identities[-1]])==()
    model.set_array_settings(ArrayDisplaySettings(10,1000))
    assert model.restore_rows([identities[-1]])==()  # No resurrection of an old partition.
    revision=model.revision
    model.set_array_settings(ArrayDisplaySettings(10,99999),selected_rows=identities[:1])
    assert model.revision==revision


def test_actual_query_element_identity_survives_repage_and_capture_boundary_rejects(tmp_path):
    source=Session(str(ROOT/'test/data/cloth_physx34.pxd2'),str(tmp_path))
    adapter=PvdDataAdapter(source.presentation(frame=1))
    model=ObjectTreeModel();model.populate(adapter)
    record=model.canonical_item(-6);row=model.row_identity(record)
    assert row.query_id==-6 and row.kind=='array_element'
    restored=model.set_array_settings(ArrayDisplaySettings(9999,0),selected_rows=[row])
    assert len(restored)==1 and restored[0] is record
    assert model.row_identity(restored[0]).token==row.token
    other=Session(str(ROOT/'test/data/cloth_physx34.pxd2'),str(tmp_path/'other'))
    model.populate(PvdDataAdapter(other.presentation(frame=1)))
    assert model.restore_rows([row])==()


def test_large_logical_groups_keep_model_expansion_physically_bounded(tmp_path):
    source=Session(str(ROOT/'test/data/ue4_shooter.pxd2'),str(tmp_path))
    scene=next(s for s in source.scenes() if s.stream_id==686661696)
    adapter=PvdDataAdapter(source.presentation(scene.id,30))
    model=ObjectTreeModel(page_size=9999);model.populate(adapter)
    assert model.page_size==9999
    assert all(len(node.children)<=64 for node in model._specs.values())
    page=next(node for node in model._specs.values() if node.kind=='page' and len(node.children)==64)
    before=len(model.cache)
    children=model.get_item_children(model._item(page.key))
    assert len(children)==64 and len(model.cache)-before<=70
