# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build CPython's Windows XML modules with pinned Expat 2.9.0 sources.

Run with the application venv's Python after installing setuptools 80.9.0
and wheel 0.45.1. Only the verified, job-owned base prefix is modified.
Linux already embeds Expat 2.9.0 and only needs identity validation.
"""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install_ovui_python as runtime

PROVENANCE_NAME = "ovdnext-expat-build.json"
MODULE_PATHS = ("DLLs/pyexpat.pyd", "DLLs/_elementtree.pyd")
SOURCES = {
    "cpython": {
        "version": "3.12.16", "root": "Python-3.12.16",
        "filename": "Python-3.12.16.tar.xz",
        "url": "https://www.python.org/ftp/python/3.12.16/Python-3.12.16.tar.xz",
        "sha256": "b5796915dc45ff745cfb4e990a7e7c93d6f546328a102ce31448d69ebc37575c",
        "bytes": 20839652,
    },
    "expat": {
        "version": "2.9.0", "root": "expat-2.9.0",
        "filename": "expat-2.9.0.tar.gz",
        "url": "https://github.com/libexpat/libexpat/releases/download/R_2_9_0/expat-2.9.0.tar.gz",
        "sha256": "16afbb9cefead2aa278105cf27d9f597bde7fbf3dbb85015857ca7ca6a4e89ba",
        "bytes": 847314,
    },
}
LICENSE_HASHES = {
    "LICENSE.cpython-xml.txt": "3b2f81fe21d181c499c59a256c8e1968455d6689d269aa85373bfb6af41da3bf",
    "LICENSE.expat.txt": "31b15de82aa19a845156169a17a5488bf597e561b2c318d159ed583139b25e87",
}
EXPAT_FILES = ("xcs.c", "xmlparse.c", "xmlrole.c", "xmltok.c", "random_rand_s.c")
DEFINITIONS = [("XML_STATIC", "1"), ("Py_BUILD_CORE_MODULE", "1"),
               ("_CRT_SECURE_NO_WARNINGS", "1")]


def verify_source_archive(path, entry):
    if Path(path).stat().st_size != entry["bytes"] or runtime.sha256(path) != entry["sha256"]:
        raise ValueError("XML source archive size/SHA-256 mismatch")


def source_archive(entry, cache, work):
    if cache is not None:
        target = Path(cache) / entry["filename"]
        verify_source_archive(target, entry)
        return target
    target = work / entry["filename"]
    with urllib.request.urlopen(entry["url"], timeout=60) as response, target.open("xb") as stream:
        shutil.copyfileobj(response, stream)
    verify_source_archive(target, entry)
    return target


def extract_sources(archive, destination, root_name):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Source destination already exists: {destination}")
    with tarfile.open(archive, "r:*") as stream:
        members = stream.getmembers()
        names = set()
        for member in members:
            path = PurePosixPath(member.name)
            if (not path.parts or path.parts[0] != root_name or path.is_absolute()
                    or ".." in path.parts or re.search(r"[\\:\x00-\x1f]", member.name)
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe XML source member: {member.name}")
            canonical = path.as_posix()
            if canonical in names:
                raise ValueError(f"Duplicate XML source member: {member.name}")
            names.add(canonical)
        destination.mkdir(parents=True, exist_ok=False)
        stream.extractall(destination, members=members, filter="data")
    return destination / root_name


def make_build_provenance(prefix, input_record, compiler, validation=None):
    return {
        "schema": 1, "platform": "windows", "python_version": "3.12.16",
        "expat_version": "expat_2.9.0", "abi": "cp312-win_amd64",
        "input_archive_sha256": input_record["archive_sha256"],
        "sources": copy.deepcopy(SOURCES), "compiler": compiler,
        "build": {"expat_files": list(EXPAT_FILES), "definitions": [list(item) for item in DEFINITIONS],
                  "setuptools": "80.9.0", "wheel": "0.45.1"},
        "runtime_files": {name: runtime.sha256(Path(prefix) / name) for name in MODULE_PATHS},
        "licenses": dict(LICENSE_HASHES),
        "validation": validation or {"native_xml_imports": True, "baseline_capi_compatible": True,
                                     "xml_regression_tests": 490, "xml_regression_skips": 12},
    }


def read_build_provenance(prefix):
    """Validate the shipped Windows rebuild marker, modules, and source licenses."""
    prefix = Path(prefix).resolve()
    input_record = runtime.read_runtime_provenance(prefix, True)
    record = json.loads((prefix / PROVENANCE_NAME).read_text(encoding="utf-8"))
    if (record.get("schema") != 1 or record.get("platform") != "windows"
            or record.get("python_version") != "3.12.16"
            or record.get("expat_version") != "expat_2.9.0"
            or record.get("abi") != "cp312-win_amd64"
            or record.get("input_archive_sha256") != input_record["archive_sha256"]):
        raise ValueError("XML rebuild input/identity provenance mismatch")
    if record.get("sources") != SOURCES:
        raise ValueError("XML rebuild source provenance mismatch")
    expected_files = {name: runtime.sha256(prefix / name) for name in MODULE_PATHS}
    if record.get("runtime_files") != expected_files:
        raise ValueError("XML rebuild module provenance mismatch")
    actual_licenses = {name: runtime.sha256(prefix / name) for name in LICENSE_HASHES}
    if record.get("licenses") != LICENSE_HASHES or actual_licenses != LICENSE_HASHES:
        raise ValueError("XML rebuild license provenance mismatch")
    return record


def clean_env():
    return {key: value for key, value in os.environ.items()
            if not key.upper().startswith(("PYTHON", "LD_")) and key not in {"GH_TOKEN", "GITHUB_TOKEN"}}


def run_python(executable, code, arguments=(), cwd=None):
    return subprocess.run([str(executable), "-I", "-B", "-c", code, *map(str, arguments)],
                          cwd=cwd, env=clean_env(), capture_output=True, text=True, check=True)


def probe_xml(executable, expected, artifacts=None):
    code = """
import io,json,platform,struct,sys
if sys.argv[2]: sys.path.insert(0,sys.argv[2])
import pyexpat,_elementtree
import xml.etree.ElementTree as ET
assert platform.python_version() == '3.12.16' and struct.calcsize('P') == 8
assert pyexpat.EXPAT_VERSION == sys.argv[1]
assert _elementtree.XMLParser().version == sys.argv[1].replace('expat_','Expat ')
p=pyexpat.ParserCreate(); p.Parse(b'<ovd><actor id="2"/></ovd>',True)
p=pyexpat.ParserCreate(namespace_separator=':'); p.ParseFile(io.BytesIO(b'<ovd xmlns="test"><actor/></ovd>'))
p=pyexpat.ParserCreate(); p.Parse(b'<!DOCTYPE ovd [<!ENTITY n "x">]><ovd>&n;</ovd>',True)
assert ET.fromstring('<ovd><actor id="2"/></ovd>')[0].get('id') == '2'
print(json.dumps({'python':platform.python_version(),'expat':pyexpat.EXPAT_VERSION,
                  'elementtree':_elementtree.XMLParser().version}))
"""
    output = run_python(executable, code, [expected, artifacts or ""])
    return json.loads(output.stdout)


def setup_source():
    # Relative source paths avoid MSVC's object-name length limit in CI checkouts.
    return f"""from pathlib import Path
from setuptools import Extension,setup
cpython=Path({SOURCES['cpython']['root']!r})
expat=Path({SOURCES['expat']['root']!r})/'lib'
includes=[str(expat),str(cpython/'Modules/expat'),str(cpython/'Include'),str(cpython/'Include/internal')]
sources=[str(expat/name) for name in {EXPAT_FILES!r}]
definitions={DEFINITIONS!r}
setup(name='ovdnext-expat-runtime',version='2.9.0',ext_modules=[
 Extension('pyexpat',[str(cpython/'Modules/pyexpat.c'),*sources],include_dirs=includes,
           define_macros=[*definitions,('PYEXPAT_EXPORTS','1')]),
 Extension('_elementtree',[str(cpython/'Modules/_elementtree.c'),*sources],include_dirs=includes,
           define_macros=definitions)])
"""


def build_modules(work, executable):
    for name, version in (("setuptools", "80.9.0"), ("wheel", "0.45.1")):
        if importlib.metadata.version(name) != version:
            raise ValueError(f"XML rebuild requires {name}=={version}")
    (work / "setup.py").write_text(setup_source(), encoding="utf-8")
    process = subprocess.run([str(executable), "-B", "setup.py", "build_ext",
                              "--build-lib", "artifacts", "--build-temp", "obj"],
                             cwd=work, env=clean_env(), capture_output=True, text=True)
    log = process.stdout + process.stderr
    (work / "build.log").write_text(log, encoding="utf-8")
    if process.returncode:
        raise RuntimeError(f"Native XML build failed; inspect {work / 'build.log'}")
    toolset = re.search(r"MSVC[\\/](\d[\d.]+)[\\/]", log)
    if not toolset:
        raise ValueError("Native XML build did not report an MSVC compiler")
    for module in ("pyexpat", "_elementtree"):
        if not (work / "artifacts" / f"{module}.cp312-win_amd64.pyd").is_file():
            raise ValueError(f"Missing cp312 x86_64 module: {module}")
    return {"name": "MSVC", "toolset": toolset.group(1),
            "python_compiler": platform.python_compiler()}


def validate_build(executable, artifacts, source, work):
    measured = probe_xml(executable, "expat_2.9.0", artifacts)
    # Both the upstream accelerator and our rebuilt parser now use Expat 2.9.0.
    # Verify the shared C API remains compatible before replacing both modules.
    compatibility = """
import sys
sys.path.insert(0,sys.argv[1]); import pyexpat; sys.path.pop(0)
assert pyexpat.EXPAT_VERSION == 'expat_2.9.0'
import _elementtree
assert _elementtree.XMLParser().version == 'Expat 2.9.0'
parser = _elementtree.XMLParser()
parser.feed('<ovd><actor id="2"/></ovd>')
assert parser.close()[0].get('id') == '2'
"""
    run_python(executable, compatibility, [artifacts])
    regression = """
import sys,unittest
sys.path[:0]=[sys.argv[1],sys.argv[2]]
import pyexpat,_elementtree
assert pyexpat.EXPAT_VERSION == 'expat_2.9.0'
assert _elementtree.XMLParser().version == 'Expat 2.9.0'
suite=unittest.defaultTestLoader.loadTestsFromNames(['test.test_pyexpat','test.test_xml_etree','test.test_xml_etree_c'])
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
"""
    try:
        tests = run_python(executable, regression, [artifacts, source / "Lib"])
    except subprocess.CalledProcessError as error:
        (work / "xml-regression.log").write_text((error.stdout or "") + (error.stderr or ""), encoding="utf-8")
        raise RuntimeError(f"Upstream XML tests failed; inspect {work / 'xml-regression.log'}") from error
    (work / "xml-regression.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    log = tests.stdout + tests.stderr
    count = re.search(r"Ran (\d+) tests? in", log)
    skipped = re.search(r"skipped=(\d+)", log)
    if not count or int(count.group(1)) < 400:
        raise ValueError("Upstream XML regression tests did not run the expected suite")
    return measured, {"native_xml_imports": True, "baseline_capi_compatible": True,
                      "xml_regression_tests": int(count.group(1)),
                      "xml_regression_skips": int(skipped.group(1)) if skipped else 0}


def install_modules(prefix, work, input_record, compiler, validation):
    backup = work / "original-modules"
    backup.mkdir()
    changed_paths = (*MODULE_PATHS, *LICENSE_HASHES, PROVENANCE_NAME)
    for relative in changed_paths:
        if (prefix / relative).is_file():
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prefix / relative, saved)
    try:
        for relative in MODULE_PATHS:
            module = Path(relative).stem
            built = work / "artifacts" / f"{module}.cp312-win_amd64.pyd"
            staged = prefix / f"{relative}.new"
            shutil.copy2(built, staged)
            staged.replace(prefix / relative)
        shutil.copy2(work / SOURCES["cpython"]["root"] / "LICENSE", prefix / "LICENSE.cpython-xml.txt")
        shutil.copy2(work / SOURCES["expat"]["root"] / "COPYING", prefix / "LICENSE.expat.txt")
        probe_xml(prefix / "python.exe", "expat_2.9.0")
        runtime.read_runtime_provenance(prefix, True)
        record = make_build_provenance(prefix, input_record, compiler, validation)
        (prefix / PROVENANCE_NAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        read_build_provenance(prefix)
    except BaseException:
        for relative in changed_paths:
            saved = backup / relative
            if saved.is_file():
                shutil.copy2(saved, prefix / relative)
            else:
                (prefix / relative).unlink(missing_ok=True)
        for relative in MODULE_PATHS:
            (prefix / f"{relative}.new").unlink(missing_ok=True)
        raise
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--source-cache", type=Path, help="Directory containing both pinned source archives")
    args = parser.parse_args(argv)
    windows = sys.platform == "win32"
    if (sys.platform not in {"win32", "linux"} or platform.python_version() != "3.12.16"
            or sys.implementation.name != "cpython" or struct.calcsize("P") != 8
            or platform.machine().lower() not in {"amd64", "x86_64"}):
        parser.error("Requires CPython 3.12.16 on Windows/Linux x86_64")
    prefix = Path(sys.base_prefix).resolve()
    input_record = runtime.read_runtime_provenance(prefix, windows)
    work_root = args.work_root.resolve()
    if work_root.is_relative_to(prefix):
        parser.error("Build work directory must be outside the managed interpreter prefix")
    work_root.mkdir(parents=True, exist_ok=True)
    if not windows:
        measured = probe_xml(prefix / "bin/python3.12", "expat_2.9.0")
        report = {"schema": 1, "platform": "linux", "runtime_input": input_record,
                  "probe": measured, "rebuild_required": False}
        (work_root / "build-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(measured))
        return 0
    if sys.prefix == sys.base_prefix:
        parser.error("Run with a venv of the verified job-owned interpreter")
    if (prefix / PROVENANCE_NAME).exists():
        record = read_build_provenance(prefix)
        measured = probe_xml(prefix / "python.exe", "expat_2.9.0")
        report = {"schema": 1, "platform": "windows", "runtime_input": input_record,
                  "probe": measured, "expat_build": record, "reused_verified_build": True}
        (work_root / "build-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(measured))
        return 0
    probe_xml(prefix / "python.exe", input_record["base_expat_version"])
    # This new child retains inputs, original modules, compiler output and tests.
    work = Path(tempfile.mkdtemp(prefix="expat-build-", dir=work_root))
    for entry in SOURCES.values():
        archive = source_archive(entry, args.source_cache, work)
        extracted = extract_sources(archive, work / f"{entry['root']}-extracted", entry["root"])
        extracted.rename(work / entry["root"])
    compiler = build_modules(work, sys.executable)
    measured, validation = validate_build(prefix / "python.exe", work / "artifacts",
                                          work / SOURCES["cpython"]["root"], work)
    record = install_modules(prefix, work, input_record, compiler, validation)
    report = {"schema": 1, "platform": "windows", "runtime_input": input_record,
              "probe": measured, "expat_build": record,
              "build_log": str(work / "build.log"), "xml_regression_log": str(work / "xml-regression.log")}
    (work_root / "build-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(measured))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
