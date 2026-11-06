"""Frozen viewport baseline provenance and same-policy DLSS comparisons."""
import ast
import hashlib
import json
from pathlib import Path

ORIGINAL_FIXTURE = 'd05a414dcc19e315338b36dbfa19d402ef6827b64916fe8dd639fcb0d3bec14c'
ORIGINAL_FUNCTIONS = '8fb23ad1dd515722ac3486aa042c8292a11d6ff08a823a5eb45d328d57122ada'
ORIGINAL_MODULE = 'd7169ad3748588f6311306e32f721abe8a411b62736356e7a948463307c63db2'


def verify_fixture_compatibility(path, reference_hash):
    """Normalize only reviewed DLSS evidence/readiness additions, then hash the whole fixture."""
    path = Path(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual == reference_hash:
        return {'fixture_sha256': actual}
    assert reference_hash == ORIGINAL_FIXTURE
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    compare = next(node for node in functions if node.name == 'compare')
    capture = next(node for node in functions if node.name == 'capture')
    ready_gate = ast.parse('if not dlss_frame_ready(name,instance,result,frame,accepted_loads,since,report):return False').body[0]
    loop = next(node for node in capture.body if isinstance(node,ast.For))
    gates = [node for node in loop.body if ast.dump(node) == ast.dump(ready_gate)]
    assert len(gates) == 1
    loop.body.remove(gates[0])
    expected_call = ast.parse("compare_dlss_policy(name,report,pixels_by_name,dlss_reference,noise['LdrColor'])").body[0]
    restored = set()
    for node in ast.walk(compare):
        if not isinstance(node, ast.If):
            continue
        condition = ast.unparse(node.test)
        if condition not in ("name == 'dlss_default'", "name == 'dlss_performance'"):
            continue
        assert ast.dump(node.body[-1]) == ast.dump(expected_call)
        node.body.pop()
        if condition == "name == 'dlss_default'":
            node.body.extend(ast.parse("assert delta(name,'resize_reset')>noise['LdrColor'],'DLSS preset has no visible effect'").body)
        restored.add(condition)
    assert len(restored) == 2
    digest = hashlib.sha256(ast.dump(ast.Module(body=functions,type_ignores=[]),include_attributes=False).encode()).hexdigest()
    assert digest == ORIGINAL_FUNCTIONS, 'Unreviewed viewport action/capture/comparison change'

    def remove_exact(body, text):
        expected = ast.dump(ast.parse(text).body[0])
        matches = [node for node in body if ast.dump(node) == expected]
        assert len(matches) == 1, text
        body.remove(matches[0])

    remove_exact(tree.body, 'from viewport_output_baseline import (verify_fixture_compatibility, load_dlss_reference, compare_dlss_policy, dlss_frame_ready)')
    remove_exact(tree.body, "parser.add_argument('--dlss-reference',type=Path)")
    remove_exact(tree.body, 'dlss_reference=None')
    admission = next(node for node in tree.body if isinstance(node,ast.If) and ast.unparse(node.test)=='not args.calibrate_baseline')
    expected = ast.dump(ast.parse("report['baseline_fixture_compatibility']=verify_fixture_compatibility(__file__,reference['fixture_sha256'])").body[0])
    matches = [node for node in admission.body if ast.dump(node)==expected]
    assert len(matches)==1
    admission.body[admission.body.index(matches[0])] = ast.parse("assert reference['fixture_sha256']==report['fixture_sha256']").body[0]
    remove_exact(admission.body,"if args.dlss_reference is None:raise ValueError('Viewport qualification requires pinned same-policy DLSS baseline evidence')")
    remove_exact(admission.body,"dlss_reference=load_dlss_reference(args.dlss_reference,report['shared_scene_sha256'])")
    module_digest=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
    assert module_digest==ORIGINAL_MODULE, 'Unreviewed viewport scene setup, stage list or numerical limit change'
    return {'original_fixture_sha256': reference_hash, 'normalized_functions_sha256': digest,
            'normalized_module_sha256':module_digest,
            'scope': 'Only reviewed DLSS evidence/readiness additions normalized; all original scene setup, stages, numerical limits and other assertions unchanged.'}


def load_dlss_reference(reference_path, shared_scene_hash, *, manifest_path=None):
    """Reject stale pixels, settings, source, or SDK identity before rendering."""
    import numpy as np
    reference_path = Path(reference_path).resolve()
    repo = next(parent.parent for parent in reference_path.parents if parent.name == '_build')
    manifest_path = manifest_path or Path(__file__).with_name('fixtures') / 'render_output_dlss_baseline.json'
    manifest = json.loads(Path(manifest_path).read_text())
    assert manifest['shared_scene_sha256'] == shared_scene_hash
    assert reference_path == (repo / manifest['reference_path']).resolve()
    for relative, expected in manifest['files'].items():
        artifact = (repo / relative).resolve()
        artifact.relative_to(repo)
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected, relative
    reference = json.loads(reference_path.read_text())
    receipt = json.loads((repo / manifest['receipt_path']).read_text())
    assert reference['status'] == 'passed' and reference['qualification'] is False
    assert reference['fixture_sha256'] == ORIGINAL_FIXTURE
    assert reference['shared_scene_sha256'] == shared_scene_hash
    assert receipt['loaded_baseline_changed'] == [] and receipt['source_changed'] == []
    assert receipt['original_fixtures_unchanged'] and receipt['reference_unchanged']
    retained = Path(receipt['environment_overrides']['OV_OUTPUT_ADDON_ROOT']).resolve()
    retained.relative_to(repo)
    for relative, expected in receipt['loaded_baseline_before'].items():
        assert hashlib.sha256((retained / relative).read_bytes()).hexdigest() == expected, relative
    assert receipt['baseline_worker_sha256'] == manifest['baseline_worker_sha256']
    assert receipt['baseline_addon_pin'] == manifest['baseline_addon_pin']
    policies = {}
    for name, expected in manifest['policies'].items():
        case = next(case for case in reference['cases'] if case['name'] == name)
        row = case['views'][0]
        assert row['dlss_enabled'] is expected['enabled']
        assert row['size'] == [321,241] and row['samples'] == 34
        assert row['render_var'] == 'LdrColor' and row['product_id'] == expected['product_id']
        assert row['dlss_authored_opinion']['value'] == expected['authored_opinion']
        config = json.loads((repo / expected['config_path']).read_text())
        assert config.get('rtx',{}).get('post',{}).get('dlss',{}).get('execMode') == expected['config_value']
        pixels = np.load(repo / expected['raw_path'], allow_pickle=False)
        assert pixels.shape == (241,321,4) and pixels.dtype == np.float32
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row['raw_sha256']
        assert np.isfinite(pixels).all()
        policies[name] = {'pixels':pixels,'enabled':expected['enabled'],'product_id':expected['product_id']}
    return {'policies':policies,'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'manifest_sha256':hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()}


def dlss_frame_ready(name, instance, result, frame, loads, since, report):
    """A redrawn predecessor is not a completed policy transition."""
    if name not in {'dlss_default','dlss_performance'}:
        return True
    if not loads or loads[-1]['accepted_at'] <= since or frame is None:
        return False
    expected = None if name == 'dlss_default' else 'performance'
    settings = loads[-1]['settings']
    if settings.get('/rtx/post/dlss/execMode') != expected:
        return False
    previous_name = 'resize_reset' if name == 'dlss_default' else 'dlss_default'
    previous = next(case for case in report['cases'] if case['name'] == previous_name)['views'][0]
    runtime = result.native_timings
    if not runtime.get('native_stage_id') or runtime['native_stage_id'] == previous['runtime'].get('native_stage_id'):
        return False
    if runtime.get('scene_path') != instance._ovrtx_scene_composition.composed_scene_path:
        return False
    if frame.render_result is not result:
        return False
    from pxr import Usd
    stage = Usd.Stage.Open(instance._ovrtx_scene_composition.composed_scene_path)
    product = stage.GetPrimAtPath(instance._viewport_request.render_product_path)
    attribute = product.GetAttribute('omni:rtx:post:dlss:execMode')
    opinion = attribute.Get() if attribute and attribute.HasAuthoredValueOpinion() else None
    return opinion == expected


def compare_dlss_policy(name, report, pixels_by_name, reference, limit):
    import numpy as np
    expected = reference['policies'][name]
    row = report['cases'][-1]['views'][0]
    assert row['dlss_enabled'] is expected['enabled']
    assert row['product_id'] == expected['product_id'] and row['size'] == [321,241]
    actual = pixels_by_name[name][0]
    assert actual.shape == expected['pixels'].shape and np.isfinite(actual).all()
    error = float(np.abs(actual.astype(float)-expected['pixels'].astype(float)).mean())
    report.setdefault('dlss_baseline_comparisons',[]).append({'policy':name,'mae':error,'limit':limit,
        'reference_sha256':reference['reference_sha256'],'manifest_sha256':reference['manifest_sha256']})
    assert error <= limit, (name,error,limit)
