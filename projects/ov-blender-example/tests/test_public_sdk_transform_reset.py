# SPDX-License-Identifier: Apache-2.0
"""Native transform and visibility commits invalidate accumulated radiance."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "runtime"))
from public_sdk_backend import RendererBackend


@pytest.mark.parametrize("attribute,kind,value", [
    ("omni:xform", "Matrix4d", np.eye(4).tolist()),
    ("omni:resetXformStack", "Bool", True),
    ("xformOp:transform", "Matrix4d", np.eye(4).tolist()),
    ("visibility", "Token", "invisible"),
    ("visibility", "Token", "inherited"),
    ("omni:rtx:autoExposure:enabled", "Bool", True),
    ("omni:rtx:autoExposure:enabled", "Bool", False),
    ("omni:rtx:autoExposure:whitePointScale", "Float", 4.0),
])
def test_transform_or_visibility_commit_updates_stage_then_resets_native_history(attribute, kind, value):
    events = []
    ready = SimpleNamespace(wait=lambda: None)
    group = SimpleNamespace(is_delete=False, attribute="usd-prim-type", prim_count=1)
    read = SimpleNamespace(wait=lambda: None, groups=lambda: (group,))
    stage = SimpleNamespace(
        query_from_path_list=lambda paths: nullcontext("query"),
        read_attributes=lambda *args: nullcontext(read),
        write_attribute=lambda query,name,ordinal,data,**kwargs: events.append(("write",name)) or ready,
        advance_write_floor=lambda ordinal,scope: events.append(("floor",ordinal)) or ready,
        release_group=lambda group: None)
    paths = SimpleNamespace(intern_token=lambda token: {"invisible": 1, "inherited": 2}.get(token, token),
                            create_path_list_from_strings=lambda values: nullcontext(values))
    backend = RendererBackend.__new__(RendererBackend)
    backend.stage=stage;backend.np=np;backend.ordinal=4;backend.desired_values={}
    backend.simulation_time_ns=500000000;backend.sample_count=32
    backend.ovstage=SimpleNamespace(PathDictionary=lambda stage: nullcontext(paths),
        OrdinalRange=SimpleNamespace(latest=lambda ordinal: ordinal), Scope=SimpleNamespace(ALL=0),
        AttributeSemantic=SimpleNamespace(NONE=0,MATRIX=5,TOKEN_ID=2))
    backend.renderer=SimpleNamespace(update_from_stage=lambda ordinal: events.append(("update",ordinal)),
        reset=lambda **kwargs: events.append(("reset",kwargs["time"])))
    if attribute == 'visibility':
        opinion = SimpleNamespace(values={'/World/Parent': value})
        visibility = SimpleNamespace(prepare=lambda rows: opinion,
            apply=lambda candidate, *, ordinal: events.append(('population', attribute)))
        backend.population_owner = SimpleNamespace(visibility=visibility)
    result=backend.update([{"prim_path":"/World/Parent","attribute_name":attribute,
                           "value_type":kind,"value":value}])
    assert result["updated_count"]==1
    assert events==[("population" if attribute == "visibility" else "write",attribute),
                    ("floor",5),("update",5),("reset",.5)]
    assert backend.sample_count==0


def test_parent_pose_projection_replays_camera_and_incoming_light_values_without_recursive_projection():
    events = []
    ready = SimpleNamespace(wait=lambda: None)
    def read(query, attributes, ordinal):
        group = SimpleNamespace(is_delete=False, attribute="usd-prim-type", prim_count=1,
            array=lambda index: np.array([1 if query=="/World/Parent" else 2],dtype=np.uint64))
        return nullcontext(SimpleNamespace(wait=lambda: None,groups=lambda: (group,)))
    stage = SimpleNamespace(query_from_path_list=lambda paths: nullcontext(paths[0]),
        read_attributes=read, release_group=lambda group: None,
        write_attribute=lambda query,name,ordinal,data,**kw: events.append(("write",query,name,np.asarray(data).copy())) or ready,
        advance_write_floor=lambda ordinal,scope: ready)
    paths = SimpleNamespace(intern_token=lambda token: token,
        token_to_string=lambda token: "Xform" if token==1 else "Camera",
        create_path_list_from_strings=lambda values: nullcontext(values))
    backend = RendererBackend.__new__(RendererBackend)
    backend.stage=stage;backend.np=np;backend.ordinal=4;backend.simulation_time_ns=0;backend.sample_count=32
    backend.ovstage=SimpleNamespace(PathDictionary=lambda stage: nullcontext(paths),
        OrdinalRange=SimpleNamespace(latest=lambda ordinal: ordinal),Scope=SimpleNamespace(ALL=0),
        AttributeSemantic=SimpleNamespace(NONE=0,MATRIX=5))
    backend.renderer=SimpleNamespace(update_from_stage=lambda ordinal: None,reset=lambda **kw: None)
    camera={"prim_path":"/World/Camera","attribute_name":"omni:xform","value_type":"Matrix4d","value":np.eye(4).tolist()}
    light={"prim_path":"/World/Light","attribute_name":"inputs:exposure","value_type":"Float","value":2}
    backend.desired_values={(row["prim_path"],row["attribute_name"]):row for row in (camera,light)}
    backend.population_owner=SimpleNamespace(ordinal=4)
    def project(updates, *, ordinal):
        events.append(("project",updates))
        backend.pose_population.poses.update(updates)
    backend.pose_population=SimpleNamespace(poses={},apply=project)
    incoming={**light,"value":3}
    backend.update([{"prim_path":"/World/Parent","attribute_name":"omni:xform","value_type":"Matrix4d","value":np.eye(4).tolist()}, incoming])
    assert len([row for row in events if row[0]=="project"])==1
    projected=next(i for i,row in enumerate(events) if row[0]=="project")
    replay=events[projected+1:]
    assert any(row[0]=="write" and row[1:3]==("/World/Camera","omni:xform") for row in replay)
    assert any(row[0]=="write" and row[2]=="inputs:exposure" and row[3][0]==3 for row in replay)
    assert not any(row[0]=="write" and row[1]=="/World/Parent" for row in events)
