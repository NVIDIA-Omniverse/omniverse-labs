# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private source-backed population for custom FMI schemas."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _isolated_usd_environment() -> dict[str, str]:
    """Return an environment without renderer-owned USD plugin paths."""
    env = os.environ.copy()
    env.pop("PXR_PLUGINPATH_NAME", None)
    for name in tuple(env):
        if name.startswith("OV_PXR_PLUGINPATH_"):
            env.pop(name)
    return env


def parse_source(source_asset: str) -> dict:
    """Parse FMI schema and initial values in an isolated USD process."""
    source = str(Path(source_asset).expanduser().resolve())
    script_dir = Path(__file__).resolve().parent
    parser = script_dir / "_parse_fmi_schema.py"
    usd_python = os.environ.get("USD_PYTHON", sys.executable)
    launcher = (
        "import runpy,sys; script=sys.argv.pop(1); "
        "runpy.run_path(script, run_name='__main__')"
    )
    result = subprocess.run(
        [usd_python, "-c", launcher, str(parser), source],
        capture_output=True,
        text=True,
        timeout=60,
        env=_isolated_usd_environment(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FMI schema parsing failed (exit {result.returncode}):\n{result.stderr}"
        )
    return json.loads(result.stdout)


def deserialise_instances(raw: dict, *, root_prim: str, enable_ssp: bool) -> dict:
    from ._parser import (
        FmuDirection,
        FmuParserConnection,
        FmuParserInstance,
        FmuParserMapping,
    )

    root = root_prim.rstrip("/") or "/"
    instances = {}
    for path, item in raw.items():
        if root != "/" and path != root and not path.startswith(root + "/"):
            continue
        is_ssp = bool(item.get("ssp"))
        if is_ssp and not enable_ssp:
            continue
        connections = []
        for connection in item["connections"]:
            mappings = [
                FmuParserMapping(
                    fmiAttributeName=mapping["fmiAttributeName"],
                    usdAttributeName=mapping["usdAttributeName"],
                    direction=FmuDirection(mapping["direction"]),
                    usdMapping=tuple(mapping["usdMapping"]),
                )
                for mapping in connection["mappings"]
            ]
            connections.append(
                FmuParserConnection(
                    enabled=connection["enabled"],
                    targets=connection["targets"],
                    mappings=mappings,
                )
            )
        parser_instance = FmuParserInstance(
            enabled=item["enabled"],
            fmu=item.get("ssp") or item.get("fmu"),
            path=item["path"],
            connections=connections,
        )
        parser_instance._is_ssp = is_ssp
        instances[path] = parser_instance
    return instances
