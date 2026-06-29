import importlib.util
import os
import unittest
from pathlib import Path

import nanousd  # noqa: F401 - registers the pxr compatibility shim

UPSTREAM = Path(__file__).parent / "openusd_compat" / "upstream"

COPIED_TESTS = [
    "pxr/usd/sdf/testenv/testSdfPath.py",
    "pxr/usd/sdf/testenv/testSdfListOp.py",
    "pxr/usd/sdf/testenv/testSdfAssetPath.py",
    "pxr/usd/usd/testenv/testUsdStage.py",
    "pxr/usd/usd/testenv/testUsdPrims.py",
    "pxr/usd/usd/testenv/testUsdTimeCode.py",
    "pxr/usd/usd/testenv/testUsdTimeSamples.py",
    "pxr/usd/usd/testenv/testUsdRelationships.py",
    "pxr/usd/usd/testenv/testUsdAppliedAPISchemas.py",
    "pxr/usd/usdGeom/testenv/testUsdGeomConsts.py",
    "pxr/usd/usdGeom/testenv/testUsdGeomXformable.py",
    "pxr/usd/usdGeom/testenv/testUsdGeomPurposeVisibility.py",
    "pxr/usd/usdShade/testenv/testUsdShadeMaterialBinding.py",
]


def _load_copied_module(relpath: str):
    path = UPSTREAM / relpath
    spec = importlib.util.spec_from_file_location("openusd_copied_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_copied_module(relpath: str):
    module = _load_copied_module(relpath)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    result = unittest.TestResult()
    suite.run(result)
    assert result.wasSuccessful(), result.errors + result.failures


def test_copied_openusd_tests_are_present_with_license_headers():
    for relpath in COPIED_TESTS:
        path = UPSTREAM / relpath
        assert path.exists(), relpath
        text = path.read_text(encoding="utf-8")
        assert "Copyright" in text
        assert "openusd.org/license" in text


def test_copied_openusd_timecode_module_passes():
    _run_copied_module("pxr/usd/usd/testenv/testUsdTimeCode.py")


def test_copied_openusd_asset_path_module_passes():
    _run_copied_module("pxr/usd/sdf/testenv/testSdfAssetPath.py")


def test_copied_openusd_listop_module_passes():
    _run_copied_module("pxr/usd/sdf/testenv/testSdfListOp.py")


def test_copied_openusd_path_module_passes():
    _run_copied_module("pxr/usd/sdf/testenv/testSdfPath.py")


def test_copied_openusd_stage_url_identifier_test_passes(tmp_path, monkeypatch):
    module = _load_copied_module("pxr/usd/usd/testenv/testUsdStage.py")
    case = module.TestUsdStage(methodName="test_URLEncodedIdentifiers")

    monkeypatch.chdir(tmp_path)
    case.test_URLEncodedIdentifiers()
    assert (tmp_path / "Libeccio%20LowFBX.usda").exists()
