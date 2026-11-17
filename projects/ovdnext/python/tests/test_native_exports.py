# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exports use native writers, preserve source state, and publish complete files."""
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from _pvd_native import Session
from native_fixtures import executable
ROOT=Path(__file__).resolve().parents[2]


def test_save_rejects_hardlink_alias_without_changing_capture(tmp_path):
    import os
    original = tmp_path / "source.ovd"
    original.write_bytes((ROOT / "test/data/test_scene.ovd").read_bytes())
    alias = tmp_path / "alias.ovd"
    os.link(original, alias)
    source = Session(str(original), str(tmp_path))
    before = original.read_bytes()
    with pytest.raises(ValueError):
        source.save(str(alias), True)
    assert original.read_bytes() == alias.read_bytes() == before
    assert os.path.samefile(original, alias)


def test_save_rejects_directory_symlink_parent_alias_without_changing_capture(tmp_path):
    import os
    nested = tmp_path / "subdir"
    deep = nested / "deep"
    deep.mkdir(parents=True)
    original = nested / "source.ovd"
    original.write_bytes((ROOT / "test/data/test_scene.ovd").read_bytes())
    link = tmp_path / "link"
    try:
        link.symlink_to(nested if sys.platform == "win32" else deep, target_is_directory=True)
    except OSError as error:
        if sys.platform == "win32" and getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows host lacks symlink creation privilege")
        raise
    alias = link / "source.ovd" if sys.platform == "win32" else link / ".." / "source.ovd"
    assert os.path.samefile(original, alias)
    source = Session(str(original), str(tmp_path))
    before = original.read_bytes()
    identity = (original.stat().st_dev, original.stat().st_ino)
    with pytest.raises(ValueError):
        source.save(str(alias), True)
    assert original.read_bytes() == alias.read_bytes() == before
    assert (original.stat().st_dev, original.stat().st_ino) == identity
    assert os.path.samefile(original, alias)


def capture(tmp_path, name='cloth_physx34.pxd2', frame=5):
    s=Session(str(ROOT/'test/data'/name),str(tmp_path));s.seek(s.scenes()[0].id,frame);return s


def compare_frame(left,right):
    assert [(i.id,i.class_id,i.alive) for i in left.instances()]==[(i.id,i.class_id,i.alive) for i in right.instances()]
    a,b=left.geometry_snapshot(),right.geometry_snapshot()
    for field in ('object_ids','shape_ids','geometry_ids','scene_ids','kinds','transforms','dimensions'):
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
    am={m.source_id:m for m in a.meshes};bm={m.source_id:m for m in b.meshes}
    assert am.keys()==bm.keys()
    for key in am:
        np.testing.assert_array_equal(am[key].vertices,bm[key].vertices)
        np.testing.assert_array_equal(am[key].indices,bm[key].indices)
    # Captured cloth state, named handles, and query arrays survive native transfer.
    for obj in left.instances():
        if not obj.alive:continue
        for prop in left.properties(obj.class_id):
            if prop.type=='StringHandle':assert left.string_value(obj.id,prop.id)==right.string_value(obj.id,prop.id)
            else:assert left.read(obj.id,prop.id,0,8*1024*1024)==right.read(obj.id,prop.id,0,8*1024*1024),(obj.id,prop.name)


def test_native_save_reopens_full_capture_without_seeking_source(tmp_path):
    source=capture(tmp_path)
    cap=source.export_capabilities();assert cap.format=='PXD2' and cap.save and cap.clip and cap.repx
    path=tmp_path/'all.pxd2';size=source.save(str(path));assert size==path.stat().st_size
    assert int.from_bytes(path.read_bytes()[:4],'little')==907025328
    saved=Session(str(path),str(tmp_path));assert saved.scenes()[0].frame_count==source.scenes()[0].frame_count
    assert source.scenes()[0].current_frame==5
    saved.seek(saved.scenes()[0].id,5);compare_frame(source,saved)


@pytest.mark.parametrize('start,stop',[(0,3),(5,8),(5,6)])
def test_native_crop_exact_range_snapshot_and_frame_values(tmp_path,start,stop):
    source=capture(tmp_path);scene=source.scenes()[0];before=source.geometry_snapshot().transforms.copy()
    path=tmp_path/'clip.pxd2';assert source.save_clip(str(path),scene.id,start,stop)==path.stat().st_size
    assert source.scenes()[0].current_frame==5
    np.testing.assert_array_equal(source.geometry_snapshot().transforms,before)
    clip=Session(str(path),str(tmp_path));target=clip.scenes()[0]
    assert target.frame_count==stop-start and target.current_frame==0
    for local in range(stop-start):
        source.seek(scene.id,start+local);clip.seek(target.id,local);compare_frame(source,clip)
    assert not list(tmp_path.glob('*.ovui-*'))


def test_ovd_save_copies_original_bytes_and_reports_native_export_limits(tmp_path):
    source=capture(tmp_path,'test_scene.ovd',1);cap=source.export_capabilities()
    assert cap.format=='OVD' and cap.save and not cap.clip and not cap.repx
    assert cap.clip_reason and cap.repx_reason
    path=tmp_path/'copy.ovd';source.save(str(path));assert path.read_bytes()==(ROOT/'test/data/test_scene.ovd').read_bytes()
    assert Session(str(path),str(tmp_path)).info().ovd
    with pytest.raises(ValueError):source.save_clip(str(tmp_path/'bad.pxd2'),source.scenes()[0].id,0,1)
    with pytest.raises(ValueError):source.export_repx(str(tmp_path/'bad.repx'))
    assert not (tmp_path/'bad.pxd2').exists() and not (tmp_path/'bad.repx').exists()


def test_ovd_save_rejects_a_symlink_alias_of_the_source(tmp_path):
    source=capture(tmp_path,'test_scene.ovd',1)
    alias=tmp_path/'alias.ovd'
    try:
        alias.symlink_to(ROOT/'test/data/test_scene.ovd')
    except OSError as error:
        if sys.platform=='win32' and getattr(error,'winerror',None)==1314:
            pytest.skip('Windows host lacks symlink creation privilege')
        raise
    with pytest.raises(ValueError):source.save(str(alias),True)
    assert alias.is_symlink()


def test_repx_original_all_and_selected_output_valid_xml(tmp_path):
    source=capture(tmp_path)
    all_path=tmp_path/'all.repx';source.export_repx(str(all_path));root=ET.parse(all_path).getroot()
    assert root.tag=='PhysX30Collection' and root.attrib['version']=='3.4.0'
    assert len(root.findall('PxCloth'))==5
    classes={c.id:c.name for c in source.classes()}
    actors=[i.id for i in source.instances() if i.alive and classes[i.class_id]=='PxRigidDynamic']
    assert actors
    path=tmp_path/'selected.repx';source.export_repx(str(path),[actors[0]])
    selected=ET.parse(path).getroot()
    assert [int(x.findtext('Id')) for x in selected.findall('PxRigidDynamic')]==[actors[0]]
    assert not selected.findall('PxCloth')
    assert source.scenes()[0].current_frame==5
    with pytest.raises(ValueError):source.export_repx(str(tmp_path/'empty.repx'),[])
    with pytest.raises(IndexError):source.export_repx(str(tmp_path/'unknown.repx'),[-6])
    assert not (tmp_path/'empty.repx').exists() and not (tmp_path/'unknown.repx').exists()


def test_invalid_ranges_and_output_failures_preserve_existing_files(tmp_path):
    s=capture(tmp_path);scene=s.scenes()[0];path=tmp_path/'existing.pxd2';path.write_bytes(b'existing')
    for start,stop in [(5,5),(6,5),(0,scene.frame_count+1)]:
        with pytest.raises(IndexError):s.save_clip(str(path),scene.id,start,stop,True)
        assert path.read_bytes()==b'existing'
    with pytest.raises(RuntimeError):s.save(str(path))
    assert path.read_bytes()==b'existing'
    s.save(str(path),True);assert Session(str(path),str(tmp_path)).scenes()[0].frame_count==scene.frame_count
    with pytest.raises(RuntimeError):s.save(str(tmp_path/'missing'/'file.pxd2'))
    with pytest.raises(ValueError):s.save(str(ROOT/'test/data/cloth_physx34.pxd2'),True)
    assert not list(tmp_path.glob('*.ovui-*'))


def test_crop_replays_native_origin_shift_events(tmp_path):
    binary=executable('pvd_geometry_fixture','PVD_GEOMETRY_FIXTURE')
    source_path=tmp_path/'shift.pxd2'
    subprocess.run([str(binary),str(source_path),'--legacy-shift'],check=True)
    source=Session(str(source_path),str(tmp_path));scene=source.scenes()[0]
    source.seek(scene.id,1)
    assert not source.export_capabilities().repx # custom minimal schema is not an exportable PhysX scene
    prop=next(p for p in source.properties(source.instances()[1].class_id) if p.name=='GlobalPose')
    path=tmp_path/'clip.pxd2';source.save_clip(str(path),scene.id,1,4)
    clip=Session(str(path),str(tmp_path));target=clip.scenes()[0]
    assert target.frame_count==3
    for local in range(3):
        source.seek(scene.id,local+1);clip.seek(target.id,local)
        a=np.frombuffer(source.read(2,prop.id),dtype='<f4');b=np.frombuffer(clip.read(2,prop.id),dtype='<f4')
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(b[4:],[100-10*local,200-20*local,300-30*local])


def test_crop_uses_selected_scene_timeline_and_preserves_other_scenes(tmp_path):
    source=capture(tmp_path,'ue4_shooter.pxd2',30);scene=source.scenes()[1]
    source.seek(scene.id,30)
    path=tmp_path/'multi.pxd2';source.save_clip(str(path),scene.id,30,32)
    clip=Session(str(path),str(tmp_path));target=next(s for s in clip.scenes() if s.id==scene.id)
    assert {s.id for s in clip.scenes()}=={s.id for s in source.scenes()}
    assert target.frame_count==2
    for local in range(2):
        source.seek(scene.id,30+local);clip.seek(target.id,local);compare_frame(source,clip)
