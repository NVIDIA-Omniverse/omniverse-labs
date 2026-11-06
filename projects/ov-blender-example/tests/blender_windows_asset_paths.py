"""Stock Blender CPU capture/resolution/decode proof; no renderer or vendor SDK."""
from pathlib import Path
import hashlib
import json
import sys
import traceback

import bpy
from pxr import Ar, Sdf, Usd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import immutable_scene_store

out, payload_path = map(Path, sys.argv[sys.argv.index('--') + 1:])
out = out.resolve()
out.mkdir(parents=True, exist_ok=True)
report = {'status': 'running', 'scope': 'stock Blender CPU; no native renderer',
          'blender_version': bpy.app.version_string, 'cases': []}
report['source_sha256'] = hashlib.sha256(Path(immutable_scene_store.__file__).read_bytes()).hexdigest()


def extended(path):
    # Independent fixture spelling, not the product path helper under test.
    return Path('\\\\?\\' + str(path))


def decode(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = list(image.pixels[:])  # Image loading is lazy until pixel access.
        assert image.has_data and pixels
        return {'size': list(image.size), 'pixels': pixels}
    finally:
        bpy.data.images.remove(image)


def sized_parent(base, name, length):
    count = length - len(str(base)) - len(name) - 2
    assert 1 <= count <= 255
    return base / ('p' * count) / name


try:
    payload = extended(payload_path).read_bytes()
    report['retained_payload_sha256'] = hashlib.sha256(payload).hexdigest()
    assert len(payload) == 465
    assert report['retained_payload_sha256'] == 'a17a0a25c46ed4e7b5826d1b19e832da0560500d8a2c1a58a45b0fc6336165a9'
    candidate_file = out / 'candidate.exr'
    image = bpy.data.images.new('Candidate', width=1, height=1, float_buffer=True)
    image.pixels[:] = [1.0, 0.0, 0.0, 1.0]
    image.file_format = 'OPEN_EXR'
    image.filepath_raw = str(candidate_file)
    image.save()
    bpy.data.images.remove(image)
    candidate_bytes = candidate_file.read_bytes()
    for name, source_length, basename in (
            ('long-basename', 189, 'a' * 64 + '.exr'),
            ('generated-source', 266, 'color_0C0C0C.exr')):
        case_root = out / name
        if name == 'generated-source':
            export = sized_parent(case_root, 'stock-export', source_length - len('/textures/' + basename))
            source = export / 'textures' / basename
            layer_path = export / 'stock.usda'
            authored = './textures/' + basename
        else:
            source = sized_parent(case_root, basename, source_length)
            layer_path = case_root / 'stock.usda'
            authored = source.as_posix()
        assert len(str(source)) == source_length
        extended(source.parent).mkdir(parents=True, exist_ok=True)
        extended(source).write_bytes(payload)
        ordinary_error = None
        try:
            assert source.read_bytes() == payload
        except OSError as exc:
            ordinary_error = repr(exc)
        if source_length == 266:
            assert ordinary_error, 'This proof requires the observed ordinary-path refusal'
        layer = Sdf.Layer.CreateNew(str(layer_path))
        prim = Sdf.CreatePrimInLayer(layer, '/Texture')
        prim.specifier = Sdf.SpecifierDef
        prim.typeName = 'Shader'
        Sdf.AttributeSpec(prim, 'inputs:file', Sdf.ValueTypeNames.Asset).default = Sdf.AssetPath(authored)
        layer.Save()
        root = sized_parent(case_root, 'accepted-scene', 160)
        store = immutable_scene_store.ImmutableSceneStore(root)
        accepted = store.capture(layer_path)

        def owned(snapshot):
            stage = Usd.Stage.Open(snapshot.path)
            value = stage.GetPrimAtPath('/Texture').GetAttribute('inputs:file').Get()
            resolved = str(Ar.GetResolver().Resolve(value.path))
            assert resolved and value.resolvedPath
            path = Path(resolved)
            assert not str(path).startswith('\\\\?\\') and len(str(path)) < 260
            return path

        accepted_asset = owned(accepted)
        assert accepted_asset.read_bytes() == payload
        accepted_pixels = decode(accepted_asset)
        assert accepted_pixels['size'] == [1, 1]
        extended(source).write_bytes(candidate_bytes)
        candidate = store.capture(layer_path)
        candidate_asset = owned(candidate)
        assert candidate_asset != accepted_asset
        assert candidate_asset.read_bytes() == candidate_bytes
        candidate_pixels = decode(candidate_asset)
        assert candidate_pixels != accepted_pixels
        store.retain([accepted.path])
        assert decode(accepted_asset) == accepted_pixels
        assert not candidate_asset.exists() and not Path(candidate.path).exists()
        store.retain([])
        assert store.cleanup_errors == [] and not tuple(root.iterdir())
        report['cases'].append({'name': name, 'source_path': str(source), 'source_length': source_length,
            'ordinary_source_error': ordinary_error, 'root_length': len(str(root)),
            'old_temporary_length': len(str(root / ('asset-' + '0' * 32) / basename)),
            'old_final_length': len(str(root / ('assets-' + '0' * 64) / basename)),
            'resolved_owned_asset': str(accepted_asset), 'owned_length': len(str(accepted_asset)),
            'accepted_decode': accepted_pixels, 'candidate_decode': candidate_pixels,
            'rejected_candidate_removed': True, 'predecessor_retained_exact': True, 'cleanup_empty': True})
    report['sdk_modules'] = sorted(name for name in sys.modules
        if name.split('.')[0] in {'ovrtx', 'ovphysx', 'ovusd', 'ovmaterial', 'ovwarp'})
    assert not report['sdk_modules']
    report['status'] = 'passed'
except BaseException:
    report['status'] = 'failed'
    report['traceback'] = traceback.format_exc()
    raise
finally:
    (out / 'result.json').write_text(json.dumps(report, indent=2))
