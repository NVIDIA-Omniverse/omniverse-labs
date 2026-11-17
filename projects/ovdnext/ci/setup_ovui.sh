#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
command -v uv >/dev/null || { echo 'Install uv, then run ci/setup_ovui.sh again.' >&2; exit 1; }
mkdir -p _ci/deps _ci/logs
physics_ref=a12f79ce585ffb79e5ffa4848cb45efc3de2f632
ovui_ref=a81db103a4d99ff2d75368a444e167889536ff42
if [[ ! -d _ci/deps/physics ]]; then
    ./ci/fetch_physics.sh --repository-url https://github.com/NVIDIA-Omniverse/physx-internal.git \
        --ref "$physics_ref" --destination "$root/_ci/deps/physics"
fi
[[ $(git -C _ci/deps/physics rev-parse HEAD) == "$physics_ref" ]] || { echo 'Physics checkout has an unexpected pin.' >&2; exit 1; }
if [[ ! -d _ci/deps/ovui ]]; then
    git init _ci/deps/ovui
    git -C _ci/deps/ovui remote add origin https://github.com/NVIDIA-Omniverse/ovui.git
    git -C _ci/deps/ovui fetch --depth 1 origin "$ovui_ref"
    git -C _ci/deps/ovui checkout --detach FETCH_HEAD
fi
[[ $(git -C _ci/deps/ovui rev-parse HEAD) == "$ovui_ref" ]] || { echo 'OVUI checkout has an unexpected pin.' >&2; exit 1; }
if [[ ! -x _ci/venv/bin/python ]]; then
    uv python install 3.12.16
    bootstrap=$(uv python find --managed-python 3.12.16)
    runtime_python=$("$bootstrap" -B ci/install_ovui_python.py --reuse-verified --work-root "${OVUI_PYTHON_INPUT_ROOT:-$root/_ci/python-inputs-fa506e08}")
    uv venv --python "$runtime_python" _ci/venv
fi
[[ $(_ci/venv/bin/python -c 'import platform; print(platform.python_version())') == 3.12.16 ]] || { echo 'CPython 3.12.16 is required; move _ci/venv aside and rerun ci/setup_ovui.sh.' >&2; exit 1; }
_ci/venv/bin/python -B -c 'from pathlib import Path; import sys; from ci.install_ovui_python import read_runtime_provenance, probe_runtime; read_runtime_provenance(Path(sys.base_prefix), False); probe_runtime(Path(sys.base_prefix), False)' || {
    echo '_ci/venv is not based on the verified CPython 3.12.16 input runtime; move _ci/venv aside and rerun ci/setup_ovui.sh.' >&2
    exit 1
}
uv pip install --python _ci/venv/bin/python -r python/requirements.txt \
    'setuptools==80.9.0' 'wheel==0.45.1' 'cmake==3.25.2' 'ninja==1.10.2.4' \
    ./_ci/deps/ovui/ovui-data-adapters/dist/common \
    ./_ci/deps/ovui/ovui-data-adapters/dist/services \
    ./_ci/deps/ovui/ovui-widgets/dist/common
PATH="$root/_ci/venv/bin:$PATH" _ci/venv/bin/python -B ci/install_ovui_pillow.py --work-root "$(mktemp -d "$root/_ci/pillow-build-XXXXXX")/build" --source-cache "$root/_ci/pillow-sources" --uv "$(command -v uv)"
cmake -S . -B _ci/build/ovui -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPVD3_BUILD_PYTHON=ON -DPVD3_BUILD_TESTS=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_Qt5=ON -DPVD3_REQUIRE_OVD_SUPPORT=ON \
    -DPVDRUNTIME_ROOT="$root/_ci/deps/physics/physx/pvdruntime" \
    -DPVDDOM_ROOT="$root/_ci/deps/physics/physx/pvddom" \
    -DPython_EXECUTABLE="$root/_ci/venv/bin/python" \
    -Dpybind11_DIR="$("$root/_ci/venv/bin/python" -m pybind11 --cmakedir)"
cmake --build _ci/build/ovui --target _pvd_native --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
PYTHONPATH="$root/python:$root/_ci/build/ovui/python" _ci/venv/bin/python -B -c 'import _pvd_native; print("Headless native import smoke passed")'
printf 'Full verification: use ci/build_ovui.py and ci/test_ovui.py. This setup runs an import smoke only.\n'
printf 'Run ./ci/run_ovui.sh test/data/test_scene.ovd from a desktop terminal.\n'
