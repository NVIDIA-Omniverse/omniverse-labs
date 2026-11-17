# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package a previously built native module without recompiling it in pip.

Portable runtime packages use ci/package_ovui_portable.py after preparing the
application venv with ci/install_ovui_pillow.py. This wheel hook is historical;
the retired ci/package_ovui_linux.py cannot create new wheelhouse packages.
The extension declaration deliberately produces a CPython/platform wheel, never
a pure-Python or manylinux wheel. Native ABI and driver requirements still apply.
"""
import os
from pathlib import Path
import shutil
import sys
import sysconfig

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class PrebuiltNative(build_ext):
    def build_extension(self, ext):
        if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("PVD runtime packaging currently requires Linux CPython 3.12")
        source = Path(os.environ.get("OVD_NATIVE_MODULE", ""))
        expected = "_pvd_native" + sysconfig.get_config_var("EXT_SUFFIX")
        if not source.is_file() or source.name != expected:
            raise RuntimeError(f"OVD_NATIVE_MODULE must name the built {expected}; use ci/package_ovui_portable.py for portable packaging")
        if not Path("ovd_ui/runtime-build.json").is_file():
            raise RuntimeError("This historical wheel hook requires staged runtime provenance/notices; use ci/package_ovui_portable.py for portable packaging")
        destination = Path(self.get_ext_fullpath(ext.name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


setup(ext_modules=[Extension("_pvd_native", sources=[])], cmdclass={"build_ext": PrebuiltNative})
