"""Candidate-only assertions must not weaken the retained native baseline gate."""
import ast
import hashlib
from pathlib import Path
import pytest

FIXTURE=Path(__file__).with_name('blender_final_output_pixels.py')

def verify(path):
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                  and node.name=='verify_frozen_baseline_capture')
    scope=dict(ast=ast,hashlib=hashlib,Path=Path,__file__=str(path),
        reference={'fixture_sha256':'80da8948e5abcd10a05c9729f6eb42c559268ba3d20bd425055114006a5c7b90'},
        report={'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    scope['verify_frozen_baseline_capture']()
    return scope['report']

def test_reviewed_animation_correction_preserves_still_baseline():
    assert verify(FIXTURE)['baseline_fixture_compatibility']['baseline_capture_ast_sha256']==\
        '86cd0dbaa91d11520edb2484eaeeedb06748c21e8fe0c30e8f25f6b7ebbb0e71'

@pytest.mark.parametrize('old,new',[
    ('max(.005,noise*3)','max(.1,noise*3)'),
    ('minimum_blue_pixels=50','minimum_blue_pixels=0'),
    ("assert result.render_var == 'HdrColor'","assert result.render_var == 'LdrColor'"),
])
def test_changed_baseline_behavior_is_refused(tmp_path,old,new):
    text=FIXTURE.read_text()
    assert old in text
    changed=tmp_path/'fixture.py';changed.write_text(text.replace(old,new))
    with pytest.raises(AssertionError):verify(changed)
