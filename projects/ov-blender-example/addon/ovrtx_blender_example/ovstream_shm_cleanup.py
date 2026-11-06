# SPDX-License-Identifier: Apache-2.0
"""Cleanup only a confirmed-dead owned worker's recorded OVStream resources.

This belongs to the process owner, never the frame reader. The caller supplies
its own epoch directory and the PID of its reaped subprocess. No global SHM
scan, prefix sweep, vendor import, or cleanup of a live process is permitted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import struct
import sys


def cleanup_registry(registry_path: str | Path, *, expected_pid: int,
                     expected_epoch: str | None = None,
                     shm_directory: str | Path = "/dev/shm",
                     control_directory: str | Path | None = None) -> list[str]:
    """Unlink exact registered resources after independently confirming death."""
    if expected_pid <= 0:
        raise ValueError("A positive owned worker PID is required")
    windows = sys.platform == "win32"
    if windows:
        from .win32_shm import process_exists
        if process_exists(expected_pid):
            raise RuntimeError("Refusing SHM cleanup while worker PID still exists")
    else:
        try:
            os.kill(expected_pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Refusing SHM cleanup while worker PID still exists")
    registry_path = Path(registry_path)
    try:
        before = registry_path.lstat()
        if stat.S_ISLNK(before.st_mode) or getattr(before, "st_file_attributes", 0) & 0x400:
            raise ValueError("Invalid owned stream registry: symlink/reparse point")
        fd = os.open(registry_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return []
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if (not stat.S_ISREG(info.st_mode) or (not windows and info.st_uid != os.getuid())
                or info.st_size > 65536 or (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino)):
            raise ValueError("Invalid owned stream registry")
        registry = json.load(handle)
    if registry.get("schema") != 1 or registry.get("pid") != expected_pid or not registry.get("epoch"):
        raise ValueError("Stream registry does not belong to the expected worker")
    if expected_epoch is not None and registry['epoch'] != expected_epoch:
        raise ValueError("Stream registry belongs to a stale worker session")
    names = registry.get("streams")
    if not isinstance(names, list) or any(not isinstance(n, str) or not re.fullmatch(r"blender-[0-9a-f]{32}", n) for n in names):
        raise ValueError("Invalid recorded SHM stream names")
    if windows:
        # Named mappings, events and pipe instances are kernel objects, not
        # files to unlink. Worker death closes its handles; readers must close
        # their own mappings. Never create/open objects solely to delete them.
        return []
    shm = Path(shm_directory)
    control = Path(control_directory or os.environ.get("TMPDIR") or "/tmp")
    candidates = []
    # Validate the complete set before deleting anything.
    for name in set(names):
        region = shm / ("ovstream-" + name)
        try:
            fd = os.open(region, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            pass  # Worker may have died during partial creation or cleanup.
        else:
            try:
                header = os.pread(fd, 60, 0)
                if len(header) != 60:
                    raise ValueError("Recorded SHM region has a truncated header")
                words = struct.unpack("<8I2Q3I", header)
                if words[:3] != (0x4D53564F, 2, 1) or words[-1] != expected_pid:
                    raise ValueError("Recorded SHM region belongs to another producer")
            finally:
                os.close(fd)
        for path in (region, shm / ("sem.ovstream-" + name + "-wake"), control / ("ovstream-" + name + ".sock")):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if info.st_uid != os.getuid() or stat.S_ISLNK(info.st_mode):
                raise ValueError("Recorded SHM resource has unexpected ownership or is a symlink")
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISSOCK(info.st_mode)):
                raise ValueError("Recorded SHM resource has unsupported type")
            candidates.append((path, info.st_dev, info.st_ino))
    removed = []
    for path, device, inode in candidates:
        try:
            info = path.lstat()
            if (info.st_dev, info.st_ino) != (device, inode):
                raise ValueError("Recorded SHM resource changed during cleanup")
            path.unlink()
            removed.append(str(path))
        except FileNotFoundError:
            pass
    return removed
