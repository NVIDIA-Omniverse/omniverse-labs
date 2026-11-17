# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable displayed build information from this imported package's resources.

The release version is canonical package source; packaging checks it against
CHANGELOG.md. Build metadata is recorded provenance, not a runtime binary audit.
No working-directory lookup, git command or invented compilation date is used.
"""
from dataclasses import dataclass
from importlib import resources
import json
import re

from . import __version__


PRODUCT_NAME = "PVD on OVUI"
MANUAL_STATUS = "The original PVD 3 User Manual.pdf is not included in this distribution."
_MAX_BUILD_BYTES = 65536
_REVISION_FIELDS = (("source_commit", "Source revision", 40),
                    ("physics_commit", "Physics revision", 40),
                    ("ovui_commit", "OVUI revision", 40),
                    ("native_sha256", "Native SHA-256", 64))


@dataclass(frozen=True)
class AboutInfo:
    rows: tuple[tuple[str, str], ...]
    provenance_status: str

    @property
    def copy_text(self):
        return "\n".join((PRODUCT_NAME, *(f"{label}: {value}" for label, value in self.rows)))


def parse_about_info(payload: bytes | None) -> AboutInfo:
    """Accept the complete required provenance subset, or display none of it."""
    version = (("Version", __version__),)
    if payload is None:
        return AboutInfo(version + (("Build information", "Unavailable"),),
                         "This copy does not include packaged build information.")
    try:
        if len(payload) > _MAX_BUILD_BYTES:
            raise ValueError("Oversized build metadata")
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("version") != __version__:
            raise ValueError("Build version mismatch")
        if type(value.get("source_dirty")) is not bool:
            raise ValueError("Missing source state")
        for key, _, length in _REVISION_FIELDS:
            if not isinstance(value.get(key), str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value[key]):
                raise ValueError("Invalid recorded identity")
        rows = [*version, ("Source revision", value["source_commit"]),
                ("Source state", "Modified" if value["source_dirty"] else "Clean")]
        rows.extend((label, value[key]) for key, label, _ in _REVISION_FIELDS[1:])
        return AboutInfo(tuple(rows), "Build information recorded by this package.")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return AboutInfo(version + (("Build information", "Unavailable"),),
                         "Packaged build information is invalid or does not match this version.")


def load_about_info() -> AboutInfo:
    try:
        resource = resources.files("ovd_ui").joinpath("runtime-build.json")
        with resource.open("rb") as stream:
            payload = stream.read(_MAX_BUILD_BYTES + 1)
    except (OSError, ValueError):
        payload = None
    return parse_about_info(payload)
