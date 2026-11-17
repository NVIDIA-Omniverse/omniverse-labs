# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native synthetic query IDs and native structure field layouts stay authoritative."""
from pathlib import Path
import struct
import numpy as np
import pytest
from _pvd_native import Session
from native_fixtures import executable
ROOT=Path(__file__).resolve().parents[2]


def test_query_records_match_pick_ids_and_owned_fields(tmp_path):
    s=Session(str(ROOT/'test/data/cloth_physx34.pxd2'),str(tmp_path))
    s.seek(s.scenes()[0].id,1)
    records=s.query_records()
    o=s.overlay_snapshot(64)
    assert [r.query_id for r in records]==[-6,-22,-38,-54,-70,-5,-21,-37,-53,-69]
    assert {r.query_id for r in records}==set(o.object_ids)
    assert [(r.query_id,r.property_id,r.element_index) for r in records]==[(r.query_id,r.property_id,r.element_index) for r in o.query_records]
    ray=records[0]
    assert ray.element_size==64
    record=s.array_record(ray.scene_instance_id,ray.property_id,ray.element_index)
    fields={f.name:f for f in record.fields}
    assert record.class_id==ray.class_id
    p=np.frombuffer(fields['origin'].data,dtype='<f4')
    v=np.frombuffer(fields['unitDir'].data,dtype='<f4')
    distance=struct.unpack('<f',fields['distance'].data)[0]
    arrow=o.positions[o.object_ids==ray.query_id][0]
    np.testing.assert_allclose(arrow[0],p)
    np.testing.assert_allclose(arrow[1],p+v*distance)
    assert fields['hits_arrayName'].string_value=='SceneQueries.Hits'
    s.seek(s.scenes()[0].id,30)
    np.testing.assert_allclose(np.frombuffer(fields['origin'].data,dtype='<f4'),p)
    with pytest.raises(IndexError):s.array_record(ray.scene_instance_id,ray.property_id,999999)
    with pytest.raises(ValueError):s.array_record(ray.scene_instance_id,0,0)


def test_ovd_does_not_invent_legacy_query_structures(tmp_path):
    s=Session(str(ROOT/'test/data/test_scene.ovd'),str(tmp_path))
    assert s.query_records()==[]
    with pytest.raises(ValueError):s.array_record(1,1,0)


def test_hierarchy_revision_detects_name_and_reference_changes_with_same_alive_ids(tmp_path):
    import subprocess
    binary=executable('pvd_geometry_fixture','PVD_GEOMETRY_FIXTURE')
    source=tmp_path/'hierarchy.ovd'
    subprocess.run([str(binary),str(source),'--hierarchy'],check=True)
    s=Session(str(source),str(tmp_path));scene=s.scenes()[0]
    revisions=[];alive=[]
    for frame in range(1,5):
        s.seek(scene.id,frame)
        revisions.append(s.hierarchy_revision())
        assert s.hierarchy_revision()==revisions[-1]
        alive.append([i.id for i in s.instances() if i.alive])
    assert alive==[alive[0]]*4
    assert revisions[0]==revisions[1] # numeric value alone changes
    assert revisions[2]>revisions[1] # name changes without births/deaths
    assert revisions[3]>revisions[2] # reference changes at equal byte size


def test_hierarchy_delta_names_exact_ue4_query_geometry_changes(tmp_path):
    s=Session(str(ROOT/'test/data/ue4_shooter.pxd2'),str(tmp_path));scene=s.scenes()[0]
    s.seek(scene.id,30);previous=s.hierarchy_revision();s.seek(scene.id,31)
    delta=s.hierarchy_changes(previous)
    assert delta.revision>previous and not delta.schema_changed and not delta.query_records_changed
    assert delta.changed_ids==[1143,4404] and delta.removed_ids==[4403]
    same=s.hierarchy_changes(delta.revision)
    assert same.changed_ids==[] and same.removed_ids==[]
    s.seek(scene.id,32);s.hierarchy_revision()
    assert s.hierarchy_changes(previous).schema_changed # skipped delta requires full rebuild


@pytest.mark.parametrize('capture,property_name',[('test_scene.ovd','rigidBodyFlags'),('cloth_physx34.pxd2','RigidBodyFlags')])
def test_named_values_are_owned_native_enum_and_flag_metadata(tmp_path,capture,property_name):
    s=Session(str(ROOT/'test/data'/capture),str(tmp_path));s.seek(s.scenes()[0].id,1)
    cls=next(c for c in s.classes() if c.name=='PxRigidDynamic')
    prop=next(p for p in s.properties(cls.id) if p.name==property_name)
    names=dict(s.named_values(prop.id));assert names['eKINEMATIC']==1 and names['eENABLE_CCD']==4
    with pytest.raises(IndexError):s.named_values(-1)
    with pytest.raises(IndexError):s.named_values(2**31-1)
    del s
    assert names['eKINEMATIC']==1
