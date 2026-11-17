# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the real isolated Pillow capability probe on selected application bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pyexpat
import shutil
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "install_ovui_pillow.py"
FEATURES = {"zlib", "jpg", "jpg_2000", "libtiff", "freetype2", "webp", "littlecms2",
            "raqm", "libimagequant", "xcb", "avif"}


def helper():
    spec = importlib.util.spec_from_file_location("ovui_pillow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def app_tree(tmp_path, *, license_header=True, extra_dist=False, with_pil=True, corrupt_png=False):
    PIL = pytest.importorskip("PIL")
    app = tmp_path / "app"
    app.mkdir()
    source = Path(PIL.__file__).resolve().parent
    if with_pil:
        shutil.copytree(source, app / "PIL", ignore=shutil.ignore_patterns("__pycache__"))
        libs = source.parent / "pillow.libs"
        if libs.is_dir():
            shutil.copytree(libs, app / "pillow.libs")
        if corrupt_png:
            # Decode a real PNG first, then corrupt its returned pixel bytes.
            # The isolated child imports only this staged Image implementation.
            with (app / "PIL/Image.py").open("a", encoding="utf-8") as output:
                output.write('''
_probe_original_open = open
def open(*args, **kwargs):
    image = _probe_original_open(*args, **kwargs)
    if image.format == "PNG":
        pixels = bytearray(image.convert("RGBA").tobytes())
        pixels[0] ^= 1
        corrupt = frombytes("RGBA", image.size, bytes(pixels))
        corrupt.format = "PNG"
        image.close()
        return corrupt
    return image
''')
    provenance = {"fixture": "provenance"}
    license_bytes = b"zlib-ng fixture license\n"
    names = [f"pillow-{PIL.__version__}.dist-info"] + (["pillow-0.0.1.dist-info"] if extra_dist else [])
    for name in names:
        info = app / name
        (info / "licenses").mkdir(parents=True)
        header = "License-File: ZLIB-NG.txt\n" if license_header else ""
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: pillow\nVersion: {PIL.__version__}\n{header}", encoding="utf-8")
        (info / "licenses/ZLIB-NG.txt").write_bytes(license_bytes)
        (info / "ovdnext-build.json").write_text(json.dumps(provenance), encoding="utf-8")
    return app, provenance, license_bytes


def probe(app, *, code_prefix=""):
    return subprocess.run([sys.executable, "-I", "-B", "-c", code_prefix + helper().PILLOW_PROBE, str(app)],
                          text=True, capture_output=True, timeout=30)


def append_to_staged(path, text):
    with path.open("a", encoding="utf-8") as output:
        output.write(text)


def test_probe_reports_selected_tree_evidence_and_actual_pixels(tmp_path):
    app, provenance, license_bytes = app_tree(tmp_path)
    result = probe(app)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)["pillow"]
    assert evidence["build_provenance"] == provenance
    assert evidence["vendor_license_sha256"] == hashlib.sha256(license_bytes).hexdigest()
    assert evidence["vendor_license_declared"] is True
    assert all(evidence[name] is True for name in ("png_decode", "image_color", "raw_rgba"))
    assert set(evidence["features"]) == FEATURES


def test_probe_reports_interpreter_expat_before_selecting_application_tree(tmp_path):
    app, _, _ = app_tree(tmp_path)
    (app / "pyexpat.py").write_text('EXPAT_VERSION = "application impostor"\n', encoding="utf-8")
    result = probe(app)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["expat"] == pyexpat.EXPAT_VERSION


def test_probe_rejects_unusable_accelerated_xml_parser(tmp_path):
    app, _, _ = app_tree(tmp_path)
    result = probe(app, code_prefix='''
import _elementtree
def broken_parser(*args, **kwargs):
    raise RuntimeError("unusable XML extension")
_elementtree.XMLParser = broken_parser
''')
    assert result.returncode != 0 and "unusable XML extension" in result.stderr


def test_probe_reports_an_enabled_openjpeg_codec(tmp_path):
    app, _, _ = app_tree(tmp_path)
    # Exercise the real isolated probe with an enabled codec in its staged PIL.
    append_to_staged(app / "PIL/features.py", '''
_probe_original_check = check
def check(feature):
    return True if feature == "jpg_2000" else _probe_original_check(feature)
''')
    result = probe(app)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["pillow"]["features"]["jpg_2000"] is True


def test_probe_rejects_unknown_feature_names(tmp_path):
    app, _, _ = app_tree(tmp_path)
    # A renamed upstream codec must fail rather than appear disabled.
    append_to_staged(app / "PIL/features.py", '\ncodecs["jpeg2000"] = codecs.pop("jpg_2000")\n')
    result = probe(app)
    assert result.returncode != 0 and "Unknown Pillow feature names" in result.stderr


def test_probe_reports_undeclared_vendor_license(tmp_path):
    app, _, _ = app_tree(tmp_path, license_header=False)
    result = probe(app)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["pillow"]["vendor_license_declared"] is False


def test_probe_rejects_ambiguous_distribution(tmp_path):
    app, _, _ = app_tree(tmp_path, extra_dist=True)
    result = probe(app)
    assert result.returncode != 0 and "Exactly one Pillow distribution" in result.stderr


def test_probe_rejects_pillow_outside_selected_tree(tmp_path):
    app, _, _ = app_tree(tmp_path, with_pil=False)
    result = probe(app)
    assert result.returncode != 0 and "not imported from the selected application tree" in result.stderr


def test_probe_reports_corrupt_decoded_png_pixels(tmp_path):
    app, _, _ = app_tree(tmp_path, corrupt_png=True)
    result = probe(app)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)["pillow"]
    assert evidence["png_decode"] is False
    assert evidence["image_color"] is True and evidence["raw_rgba"] is True
