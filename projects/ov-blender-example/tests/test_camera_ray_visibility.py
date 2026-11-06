# SPDX-License-Identifier: Apache-2.0
"""Blender camera ray visibility becomes an RTX hideForCamera primvar (GH #25 portals)."""
from types import SimpleNamespace as N

import pytest
from pxr import Sdf

from ovrtx_blender_example import object_visibility as visibility


class _Obj(N):
    def visible_get(self, view_layer=None):
        return True


def _scene(*objects):
    return N(objects=list(objects), collection=None, view_layers=[N()])


def _identity(obj, kind):
    return (kind, obj.name)


def _records(*objects):
    mappings = {_identity(o, "OBJECT"): N(schema_path=f"/World/{o.name}/Mesh", object_path=f"/World/{o.name}")
                for o in objects}
    return visibility.opinion_records(_scene(*objects), mappings, _identity)


def _layer(record):
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(record.layer_text)
    return layer


def test_camera_invisible_mesh_gets_hide_for_camera_and_stays_otherwise_visible():
    hidden = _Obj(name="Portal", name_full="Portal", type="MESH", hide_render=False, hide_viewport=False, visible_camera=False,
                  session_uid=1, particle_systems=())
    seen = _Obj(name="Wall", name_full="Wall", type="MESH", hide_render=False, hide_viewport=False, visible_camera=True,
                session_uid=2, particle_systems=())
    by_path = {record.usd_prim_path: _layer(record) for record in _records(hidden, seen)}
    portal = by_path["/World/Portal/Mesh"]
    assert portal.GetAttributeAtPath("/World/Portal/Mesh.primvars:hideForCamera").default is True
    assert portal.GetAttributeAtPath("/World/Portal/Mesh.visibility").default == "inherited"
    assert by_path["/World/Wall/Mesh"].GetAttributeAtPath("/World/Wall/Mesh.primvars:hideForCamera") is None


def test_camera_hidden_is_geometry_only():
    assert visibility.camera_hidden(N(type="MESH", visible_camera=False))
    assert not visibility.camera_hidden(N(type="MESH", visible_camera=True))
    assert not visibility.camera_hidden(N(type="LIGHT", visible_camera=False))
