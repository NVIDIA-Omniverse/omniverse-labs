# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed-runtime smoke check: no source imports, window or network listener."""
from __future__ import annotations
import argparse
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile


def validate_notices(package):
    for name in ("OvdNext-LICENSE.txt", "OvdNext-NOTICE.txt",
                 "OVUI-LICENSE.txt", "OVUI-THIRD_PARTY_NOTICES.md", "Physics-LICENSE.md",
                 "PVD-source-notice.txt", "PxShared-source-notice.txt",
                 "pybind11-LICENSE.txt", "CHANGELOG.md", "NOTICES.txt"):
        assert (package / "notices" / name).stat().st_size > 100, name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--legacy", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    modules = {}
    for name in ("ovd_ui", "_pvd_native", "ovd_ui.application", "ovd_ui.renderer",
                 "omni.ui", "ovui_widgets.common.style", "ovui_data_adapters.services", "numpy", "moderngl", "PIL"):
        module = importlib.import_module(name)
        file = Path(module.__file__).resolve()
        if not file.is_relative_to(prefix):
            raise RuntimeError(f"Import escaped the clean venv: {name} -> {file}")
        modules[name] = str(file)
    import ovd_ui
    from _pvd_native import Session
    from ovd_ui.ingestion import load_capture
    from ovd_ui.palette import ColorPalette, DEFAULT_PALETTE_PATH
    from PIL import Image
    package = Path(ovd_ui.__file__).parent
    fonts = json.loads((package / "assets/fonts/manifest.json").read_text())["fonts"]
    for name, spec in fonts.items():
        assert hashlib.sha256((package / "assets/fonts" / name).read_bytes()).hexdigest() == spec["sha256"], name
    icons = sorted((package / "assets/icons").glob("*.png"))
    expected_icons = {"first", "previous", "reverse", "play", "pause", "next", "last",
                      "theme", "live", "live_paused", "dots", "close",
                      "visible", "hidden", "opaque", "transparent"}
    assert {path.stem for path in icons} == expected_icons
    assert {path.stem for path in (package / "assets/icons").glob("*.svg")} == expected_icons
    for path in icons:
        with Image.open(path) as icon:
            icon.verify()
    validate_notices(package)
    with tempfile.TemporaryDirectory(prefix="pvd-installed-probe-") as temp:
        os.environ["XDG_DATA_HOME"] = str(Path(temp) / "fresh-user-data")
        assert DEFAULT_PALETTE_PATH.is_relative_to(package)
        palette = ColorPalette()
        assert not palette.load_error and not palette.issues
        assert palette.path.read_bytes() == DEFAULT_PALETTE_PATH.read_bytes()
        palette.edit({("actors", "rigidStatic"): "red"})
        palette.reset_defaults()
        assert palette.document == json.loads(DEFAULT_PALETTE_PATH.read_bytes())
        session = Session(str(args.capture.resolve()), temp)
        scenes = session.scenes()
        assert scenes and session.info().ovd
        assert session.seek(scenes[0].id, min(1, scenes[0].frame_count - 1))
        geometry = session.geometry_snapshot()
        assert session.info().instance_count > 0 and len(geometry.object_ids) > 0
        native = {"instances": session.info().instance_count, "draw_objects": len(geometry.object_ids), "scenes": len(scenes)}
        archive_result = "not exercised: pass --legacy for a real PXD2 archive"
        if args.legacy:
            legacy = Session(str(args.legacy.resolve()), temp)
            assert not legacy.info().ovd and legacy.scenes()
            assert legacy.seek(legacy.scenes()[0].id, 1)
            native["legacy_instances"] = legacy.info().instance_count
            assert native["legacy_instances"] > 0
            archive = Path(temp) / "installed-probe.pvd"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
                output.write(args.legacy, "capture.pxd2")
            opened = load_capture(archive, temporary_directory=temp)
            assert not opened.session.info().ovd
            payload = opened.data_path
            assert payload.is_file() and payload != archive
            opened.close()
            assert not payload.exists()
            archive_result = "opened legacy PXD2 archive and released temporary payload"
    for excluded in ("pytest", "pybind11", "fastapi", "uvicorn", "requests"):
        try:
            metadata.distribution(excluded)
        except metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(f"Development dependency unexpectedly installed: {excluded}")
    result = {"result": "pass", "prefix": str(prefix), "version": metadata.version("pvd-ovui"),
              "isolated": bool(sys.flags.isolated), "modules": modules, "fonts": len(fonts),
              "icons": len(icons), "native": native, "archive": archive_result,
              "fresh_user_palette": "initialized and reset from installed canonical colors.json",
              "gui": "not started; requires separate desktop launch"}
    if not result["isolated"]:
        raise RuntimeError("Run this probe with Python -I")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
