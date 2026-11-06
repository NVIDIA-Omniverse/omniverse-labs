"""Strict operator cancellation handling without executing Blender or native code."""
import ast
import hashlib
from pathlib import Path

import pytest

FIXTURE=Path(__file__).with_name('blender_final_output_pixels.py')


def functions():
    return {node.name:node for node in ast.parse(FIXTURE.read_text()).body if isinstance(node,ast.FunctionDef)}


def extracted(name,namespace=None):
    scope=dict(namespace or {})
    exec(compile(ast.Module(body=[functions()[name]],type_ignores=[]),str(FIXTURE),'exec'),scope)
    return scope[name]


@pytest.mark.parametrize('message',['Error: OVRTX render cancelled','Error: OVRTX render cancelled\n'])
def test_expected_error_preserved_without_fabricated_operator_result(message):
    calls=[]
    def render():
        calls.append(1)
        raise RuntimeError(message)
    result,error=extracted('invoke_expected_cancellation')(render)
    assert calls==[1] and result is None and error==message


@pytest.mark.parametrize('error',[RuntimeError('different render error'),ValueError('Error: OVRTX render cancelled'),RuntimeError('Error: OVRTX render cancelled extra')])
def test_unexpected_exception_propagates_unchanged(error):
    def render():raise error
    with pytest.raises(type(error)) as caught:extracted('invoke_expected_cancellation')(render)
    assert caught.value is error


def test_actual_operator_return_is_preserved():
    result={'FINISHED'}
    assert extracted('invoke_expected_cancellation')(lambda:result)==(result,None)


def test_baseline_capture_ast_remains_original():
    report={'fixture_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest()}
    extracted('verify_frozen_baseline_capture',dict(reference={'fixture_sha256':'80da8948e5abcd10a05c9729f6eb42c559268ba3d20bd425055114006a5c7b90'},report=report,ast=ast,Path=Path,__file__=str(FIXTURE),hashlib=hashlib))()
    assert report['baseline_fixture_compatibility']['baseline_capture_ast_sha256']=='86cd0dbaa91d11520edb2484eaeeedb06748c21e8fe0c30e8f25f6b7ebbb0e71'
