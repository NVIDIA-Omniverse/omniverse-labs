# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from _pvd_native import Session
from ovd_ui.exports import ExportRequest,export_capture

ROOT=Path(__file__).resolve().parents[2]


def test_private_native_clip_and_repx_export_publish_complete_files(tmp_path):
    source=ROOT/'test/data/cloth_physx34.pxd2'
    displayed=Session(str(source),str(tmp_path))
    scene=displayed.scenes()[0]
    displayed.seek(scene.id,9)
    target=tmp_path/'clip.pxd2'
    request=ExportRequest(source,target,'clip',scene.id,9,5,8)
    path,size=export_capture(request)
    assert path==target and size==target.stat().st_size
    saved=Session(str(target),str(tmp_path))
    assert saved.scenes()[0].frame_count==3
    assert displayed.scenes()[0].current_frame==9
    before=target.read_bytes()
    with pytest.raises(FileExistsError):export_capture(request)
    assert target.read_bytes()==before
    xml=tmp_path/'scene.repx'
    export_capture(ExportRequest(source,xml,'repx',scene.id,9))
    assert ET.parse(xml).getroot().tag=='PhysX30Collection'
    assert displayed.scenes()[0].current_frame==9


def test_export_capability_suffix_and_link_failures_preserve_destination(tmp_path):
    source=ROOT/'test/data/test_scene.ovd'
    with pytest.raises(ValueError):export_capture(ExportRequest(source,tmp_path/'invalid.ovd','clip',1,1,0,1))
    with pytest.raises(ValueError,match='PXD2'):export_capture(ExportRequest(source,tmp_path/'invalid.pxd2','clip',1,1,0,1))
    assert not (tmp_path/'invalid.pxd2').exists()


def test_export_link_failure_preserves_destination(tmp_path):
    from filesystem_capabilities import symlink
    original=tmp_path/'original.repx';original.write_text('preserve')
    link=tmp_path/'link.repx';symlink(link, original)
    legacy=ROOT/'test/data/cloth_physx34.pxd2'
    session=Session(str(legacy),str(tmp_path))
    with pytest.raises(ValueError,match='regular file'):
        export_capture(ExportRequest(legacy,link,'repx',session.scenes()[0].id,5,overwrite=True))
    assert original.read_text()=='preserve' and link.is_symlink()
