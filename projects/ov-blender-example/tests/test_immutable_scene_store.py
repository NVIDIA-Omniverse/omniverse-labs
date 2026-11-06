# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import os
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'addon'))
from ovrtx_blender_example.immutable_scene_store import ImmutableSceneStore
Sdf = pytest.importorskip('pxr.Sdf')
Usd = pytest.importorskip('pxr.Usd')


def fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'texture.bin').write_bytes(b'accepted-texture')
    layer = Sdf.Layer.CreateNew(str(source / 'child.usda'))
    prim = Sdf.CreatePrimInLayer(layer, '/World')
    prim.specifier = Sdf.SpecifierDef
    prim.typeName = 'Xform'
    attr = Sdf.AttributeSpec(prim, 'texture', Sdf.ValueTypeNames.Asset)
    attr.default = Sdf.AssetPath('./texture.bin')
    Sdf.AttributeSpec(prim, 'size', Sdf.ValueTypeNames.Double).default = 1.0
    layer.Save()
    root = Sdf.Layer.CreateNew(str(source / 'root.usda'))
    root.subLayerPaths = ['./child.usda']
    root.Save()
    return source, root, layer, ImmutableSceneStore(tmp_path / 'owned')


def values(snapshot):
    stage = Usd.Stage.Open(snapshot.path)
    prim = stage.GetPrimAtPath('/World')
    return prim.GetAttribute('size').Get(), Path(prim.GetAttribute('texture').Get().path).read_bytes()


def test_same_stat_texture_edit_cannot_reuse_accepted_bytes(tmp_path, monkeypatch):
    source, root, child, store = fixture(tmp_path)
    texture = source / 'texture.bin'
    texture.write_bytes(b'first')
    before = texture.stat()
    real_stat = Path.stat
    monkeypatch.setattr(Path, 'stat', lambda path, *a, **k:
                        before if path == texture else real_stat(path, *a, **k))
    first = store.capture(root.realPath)
    texture.write_bytes(b'other')
    second = store.capture(root.realPath)
    assert values(first)[1] == b'first'
    assert values(second)[1] == b'other'


def test_external_layer_and_texture_updates_preserve_previous_version(tmp_path):
    source, root, child, store = fixture(tmp_path)
    old = store.capture(root.realPath)
    child.GetAttributeAtPath('/World.size').default = 7.0
    child.Save()
    (source / 'texture.bin').write_bytes(b'candidate-texture')
    candidate = store.capture(root.realPath)
    assert values(old) == (1.0, b'accepted-texture')
    assert values(candidate) == (7.0, b'candidate-texture')
    store.retain([old.path])  # rejected candidate
    assert Path(old.path).exists() and not Path(candidate.path).exists()
    assert values(old) == (1.0, b'accepted-texture')
    candidate = store.capture(root.realPath)
    store.retain([candidate.path])  # accepted candidate
    assert values(candidate) == (7.0, b'candidate-texture')
    assert not Path(old.path).exists()
    store.retain([])
    assert not tuple(store.root.iterdir())


def test_asset_arrays_timesamples_udim_and_relative_reference(tmp_path):
    source, root, child, store = fixture(tmp_path)
    for name in ('tile.1001.bin', 'tile.1002.bin', 'other.bin'):
        (source / name).write_bytes(name.encode())
    prim = child.GetPrimAtPath('/World')
    assets = Sdf.AttributeSpec(prim, 'textures', Sdf.ValueTypeNames.AssetArray)
    assets.default = [Sdf.AssetPath('./texture.bin'), Sdf.AssetPath('./tile.<UDIM>.bin')]
    child.SetTimeSample('/World.texture', 1, Sdf.AssetPath('./other.bin'))
    child.Save()
    root.subLayerPaths = []
    ref = Sdf.CreatePrimInLayer(root, '/World')
    ref.specifier = Sdf.SpecifierDef
    ref.referenceList.prependedItems = [Sdf.Reference('./child.usda', '/World')]
    root.Save()
    accepted = store.capture(root.realPath)
    stage = Usd.Stage.Open(accepted.path)
    prim = stage.GetPrimAtPath('/World')
    paths = prim.GetAttribute('textures').Get()
    assert Path(paths[0].path).read_bytes() == b'accepted-texture'
    assert '<UDIM>' in paths[1].path
    assert Path(paths[1].path.replace('<UDIM>', '1001')).read_bytes() == b'tile.1001.bin'
    assert Path(paths[1].path.replace('<UDIM>', '1002')).read_bytes() == b'tile.1002.bin'
    assert Path(prim.GetAttribute('texture').Get(1).path).read_bytes() == b'other.bin'
    assert all(str(store.root) in str(Path(asset.path)) for asset in paths)


def test_one_hundred_accepted_mutations_have_bounded_disk_ownership(tmp_path):
    source, root, child, store = fixture(tmp_path)
    for revision in range(100):
        (source / 'texture.bin').write_bytes(str(revision).encode())
        current = store.capture(root.realPath)
        store.retain([current.path])
        assert len(store.snapshots) == 1
        assert len(list(store.root.glob('scene-*'))) == 1
        assert len(list(store.root.glob('assets-*'))) == 1
        assert values(current)[1] == str(revision).encode()


def test_sparse_overlay_keeps_over_specs_and_versions_inline_assets(tmp_path):
    source, root, child, store = fixture(tmp_path)
    body = '#usda 1.0\nover "World" {\n asset texture = @' + (source / 'texture.bin').as_posix() + '@\n}\n'
    overlay = source / 'overlay.usda'
    overlay.write_text(body)
    snapshot = store.capture(overlay, flatten=False)
    assert Sdf.Layer.FindOrOpen(snapshot.path).GetPrimAtPath('/World').specifier == Sdf.SpecifierOver
    inline, rewritten = store.capture_inline(body)
    layer = Sdf.Layer.CreateAnonymous()
    assert layer.ImportFromString(rewritten)
    path = layer.GetAttributeAtPath('/World.texture').default.path
    (source / 'texture.bin').write_bytes(b'changed')
    assert Path(path).read_bytes() == b'accepted-texture'


def test_missing_local_asset_stays_missing_after_original_is_created(tmp_path):
    store = ImmutableSceneStore(tmp_path / 'owned')
    source = tmp_path / 'missing.png'
    path, row = store.capture_value(dict(prim_path='/Texture', attribute_name='inputs:file',
        value_type='Asset', value=str(source)))
    source.write_bytes(b'new file')
    assert not Path(row['value']).exists()
    _, candidate = store.capture_value({**row, 'value': str(source)})
    assert Path(candidate['value']).read_bytes() == b'new file'


def test_repeated_failed_serialization_keeps_only_accepted_files(tmp_path, monkeypatch):
    source, root, child, store = fixture(tmp_path)
    accepted = store.capture(root.realPath)
    original_export = Sdf.Layer.Export

    def fail_after_write(layer, path, *args, **kwargs):
        original_export(layer, path, *args, **kwargs)
        raise OSError('injected serialization failure after partial output')

    monkeypatch.setattr(Sdf.Layer, 'Export', fail_after_write)
    for revision in range(10):
        (source / 'texture.bin').write_bytes(str(revision).encode())
        with pytest.raises(OSError, match='serialization failure'):
            store.capture(root.realPath)
        store.retain([accepted.path])
        assert len(list(store.root.glob('scene-*'))) == 1
        assert len(list(store.root.glob('assets-*'))) == 1
        assert values(accepted) == (1.0, b'accepted-texture')


def test_texture_names_keep_full_content_identity_and_nontexture_module_name(tmp_path):
    store = ImmutableSceneStore(tmp_path / 'owned')
    results = []
    for name, payload in [('first.exr', b'one'), ('second.exr', b'one'),
                          ('first.exr', b'two'), ('example_module.mdl', b'mdl')]:
        source = tmp_path / name
        source.write_bytes(payload)
        snapshot, row = store.capture_value(dict(value_type='Asset', value=str(source)))
        captured = Path(row['value'])
        assert captured.read_bytes() == payload
        assert len(captured.parent.name.removeprefix('assets-')) == 64
        if name.endswith('.mdl'):
            assert captured.name == name  # MDL import identity must not be renamed.
        results.append((snapshot, captured))
    assert len({str(path) for _, path in results}) == 4
    store.retain([results[0][0]])
    assert results[0][1].read_bytes() == b'one'
    assert all(not path.exists() for _, path in results[1:])
    store.retain([])
    assert not tuple(store.root.iterdir())


@pytest.mark.parametrize('token,tiles', [('<UDIM>', ('1001', '1002')),
                                        ('<UVTILE>', ('u1_v1', 'u2_v1'))])
def test_compact_texture_tile_patterns_preserve_members(tmp_path, token, tiles):
    store = ImmutableSceneStore(tmp_path / 'owned')
    for tile in tiles:
        (tmp_path / ('long-material-name.' + tile + '.exr')).write_bytes(tile.encode())
    _, row = store.capture_value(dict(value_type='Asset',
        value=str(tmp_path / ('long-material-name.' + token + '.exr'))))
    assert token in row['value']
    for tile in tiles:
        assert Path(row['value'].replace(token, tile)).read_bytes() == tile.encode()
    store.retain([])
    assert not tuple(store.root.iterdir())


@pytest.mark.parametrize('token,tile', [('<UDIM>', '1001'), ('<UVTILE>', 'u1_v1')])
def test_canonical_target_identity_does_not_replace_authored_tile_name(tmp_path, monkeypatch, token, tile):
    from ovrtx_blender_example import immutable_scene_store as module
    member = tmp_path / ('material.' + tile + '.exr')
    target = tmp_path / 'unrelated-target-name.exr'
    member.write_bytes(b'linked bytes')
    target.write_bytes(b'linked bytes')
    canonical = module._canonical_path
    # Model the canonical spelling supplied by a filesystem symlink. Creating
    # actual Windows file symlinks requires SeCreateSymbolicLinkPrivilege.
    monkeypatch.setattr(module, '_canonical_path',
                        lambda value: target if module._ordinary_path(value) == member else canonical(value))
    store = ImmutableSceneStore(tmp_path / 'owned')
    _, row = store.capture_value(dict(value_type='Asset', value=str(tmp_path / ('material.' + token + '.exr'))))
    assert token in row['value']
    assert Path(row['value'].replace(token, tile)).read_bytes() == b'linked bytes'
    assert 'unrelated-target-name' not in row['value']


def test_long_texture_source_and_destination_preserve_alias_cache_and_rollback(tmp_path):
    # Match the observed 160-character owned root and 266-character source.
    root = tmp_path / ('r' * (160 - len(str(tmp_path)) - 1))
    source_parent = tmp_path / 'source'
    source = source_parent / ('s' * (266 - len(str(source_parent)) - 70)) / ('a' * 64 + '.exr')
    assert len(str(root)) == 160 and len(str(source)) == 266
    source_io = Path('\\\\?\\' + str(source)) if os.name == 'nt' else source
    source_io.parent.mkdir(parents=True)
    source_io.write_bytes(b'accepted texture')
    store = ImmutableSceneStore(root)
    row = dict(value_type='Asset', value=str(source))
    old_path, old_row = store.capture_value(row)
    owned = Path(old_row['value'])
    assert len(str(owned)) < 260 and not str(owned).startswith('\\\\?\\')
    assert owned.read_bytes() == b'accepted texture'
    aliased_path, aliased_row = store.capture_value({**row, 'value': str(source_io)})
    assert (aliased_path, aliased_row) == (old_path, old_row)
    assert len(store.asset_cache) == 1
    source_io.write_bytes(b'candidate texture')
    new_path, new_row = store.capture_value(row)
    assert new_path != old_path and Path(new_row['value']).read_bytes() == b'candidate texture'
    store.retain([old_path])
    assert owned.read_bytes() == b'accepted texture' and not Path(new_path).exists()
    assert not Path(new_row['value']).exists()
    store.retain([])
    assert not tuple(root.iterdir())
