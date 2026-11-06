"""Hidden stock-export omissions retain identity and reject ambiguous geometry."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))
from ovrtx_blender_example import scene_generation as generation


class PrimPath(str):
    def AppendChild(self, name):
        return PrimPath(self + '/' + name)

    def HasPrefix(self, other):
        return self == other or self.startswith(other + '/')


class Prim:
    def __init__(self, path, kind='Xform', parent=None, **attributes):
        self.path, self.kind, self.parent, self.attributes = PrimPath(path), kind, parent, attributes

    def GetPath(self):
        return self.path

    def GetTypeName(self):
        return self.kind

    def GetParent(self):
        return self.parent

    def IsInstance(self):
        return False

    def GetAttribute(self, name):
        return SimpleNamespace(Get=lambda: self.attributes[name]) if name in self.attributes else None


def test_schema_lookup_is_scoped_and_excludes_nested_object_geometry(monkeypatch):
    owner = Prim('/Object', **{generation._OBJECT_UID_ATTRIBUTE: 1})
    shape = Prim('/Object/Shape', 'Mesh', owner, **{'userProperties:blender:data_name': 'shared'})
    child = Prim('/Object/Child', 'Mesh', owner, **{generation._OBJECT_UID_ATTRIBUTE: 2, 'userProperties:blender:data_name': 'shared'})
    grandchild = Prim('/Object/Child/Shape', 'Mesh', child, **{'userProperties:blender:data_name': 'shared'})
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(Usd=SimpleNamespace(PrimRange=lambda root: [owner, shape, child, grandchild])))
    stage = SimpleNamespace(Traverse=lambda: pytest.fail('schema lookup scanned the whole stage'))
    assert generation._schema_paths(stage, owner, 'MESH', 'shared') == ['/Object/Shape']


@pytest.mark.parametrize("kind", ["MESH", "LIGHT"])
def test_hidden_schema_omission_keeps_transform_identity(monkeypatch, kind):
    owner = Prim('/Object', **{generation._OBJECT_UID_ATTRIBUTE: 1})
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(Usd=SimpleNamespace(PrimRange=lambda root: [owner])))
    assert generation._schema_paths(None, owner, kind, 'data', viewport_hidden=True) == ['/Object']
    assert generation._schema_paths(None, owner, kind, 'data') == []


def test_hidden_flag_does_not_excuse_ambiguous_schemas(monkeypatch):
    owner = Prim('/Object')
    children = [Prim('/Object/A', 'Mesh', owner), Prim('/Object/B', 'Mesh', owner)]
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(Usd=SimpleNamespace(PrimRange=lambda root: [owner, *children])))
    assert generation._schema_paths(None, owner, 'MESH', 'unknown', viewport_hidden=True) == []


def test_hidden_instance_missing_schema_still_fails(monkeypatch):
    owner = Prim('/Object')
    owner.IsInstance = lambda: True
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(Usd=SimpleNamespace(PrimRange=lambda root: [owner])))
    assert generation._schema_paths(None, owner, 'MESH', 'data', viewport_hidden=True) == []


@pytest.mark.parametrize("properties,hidden", [
    ({"hide_render": True}, False),
    ({"hide_viewport": True}, True),
    ({"hide_get": lambda: True}, True),
    ({"visible_get": lambda: False}, True),
    ({"hide_get": lambda: False, "visible_get": lambda: True}, False),
])
def test_export_visibility_matches_viewport_evaluation(properties, hidden):
    assert generation._object_export_hidden(SimpleNamespace(**properties)) is hidden
