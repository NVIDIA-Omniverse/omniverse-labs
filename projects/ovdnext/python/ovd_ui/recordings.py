# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exact native recording files and bounded, safe library operations."""
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid

from .settings import application_data_directory


@dataclass(frozen=True)
class RecordingEntry:
    path: Path
    recorded_at: float
    size_bytes: int
    status: str = "Saved"


def absolute_path(path):
    """Normalize without following a final symlink, which needs explicit rejection."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def regular_file(path):
    try:
        return stat.S_ISREG(Path(path).lstat().st_mode)
    except FileNotFoundError:
        return False


class RecordingLibrary:
    def __init__(self, directory=None):
        self.directory = absolute_path(directory or application_data_directory() / "recordings")
        self._entries = {}
        self.load_error = ""
        try:
            self.reload()
        except OSError as error:
            # A broken optional disk library must not block RAM capture startup.
            self.load_error = str(error)

    @property
    def entries(self):
        return tuple(sorted(self._entries.values(), key=lambda entry: (entry.recorded_at, str(entry.path)),
                            reverse=True))

    def ensure_directory(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise ValueError("The recordings directory must be a real directory")
        return self.directory

    def contains(self, path):
        return absolute_path(path).parent == self.directory

    def candidate_path(self, recorded_at=None):
        stamp = datetime.fromtimestamp(recorded_at or time.time(), timezone.utc).strftime("%Y_%m_%d_%H_%M_%S_%f")
        return self.directory / f"ovdnext_{stamp}_{uuid.uuid4().hex}.ovd"

    def reload(self):
        previous = self._entries
        entries = {}
        if self.directory.exists() and not self.directory.is_symlink():
            for path in self.directory.iterdir():
                if path.suffix.lower() == ".ovd" and regular_file(path):
                    info = path.stat()
                    old = previous.get(path)
                    entries[path] = RecordingEntry(path, old.recorded_at if old else info.st_mtime,
                                                   info.st_size, old.status if old else "Saved")
        self._entries = entries
        return self.entries

    def upsert(self, path, status="Saved", recorded_at=None, source=None):
        path = absolute_path(path)
        if not self.contains(path) or path.suffix.lower() != ".ovd":
            raise ValueError("Recording must be a direct .ovd file in the library")
        size = Path(source or path).stat().st_size if regular_file(source or path) else 0
        old = self._entries.get(path)
        self._entries[path] = RecordingEntry(path, recorded_at or (old.recorded_at if old else time.time()), size, status)

    def forget(self, path):
        self._entries.pop(absolute_path(path), None)

    def delete(self, path, protected=()):
        path = absolute_path(path)
        protected = {absolute_path(item) for item in protected if item}
        same_file = any(path.exists() and item.exists() and os.path.samefile(path, item) for item in protected)
        if (path in protected or same_file or not self.contains(path) or path.suffix.lower() != ".ovd"
                or not regular_file(path)):
            raise ValueError("Only inactive, undisplayed library recordings can be deleted")
        self.ensure_directory()
        path.unlink()
        self.forget(path)


def validate_capture_destination(source, destination, *, overwrite=False):
    source = absolute_path(source) if source is not None else None
    destination = absolute_path(destination)
    if source is not None:
        if not regular_file(source):
            raise ValueError("Capture source is missing or is not a regular file")
        if source == destination or (destination.exists() and os.path.samefile(source, destination)):
            raise ValueError("Choose a destination different from the recording source")
    if destination.is_symlink() or (destination.exists() and not regular_file(destination)):
        raise ValueError("Capture destination must be a regular file")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Capture already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("Capture destination directory does not exist")
    return source, destination


def copy_capture(source, destination, *, overwrite=False):
    """Copy a stopped native stream without truncating source, links or existing files.

    An explicit overwrite publishes an independently completed temporary file with
    replace; the original destination survives a failed read/write. The default is
    no-replace and always safe to call without a confirmation dialog.
    """
    if source is None:
        raise ValueError("Capture source is missing or is not a regular file")
    source, destination = validate_capture_destination(source, destination, overwrite=overwrite)
    descriptor, temporary = tempfile.mkstemp(prefix=".ovd-save-", dir=destination.parent)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        with os.fdopen(os.open(source, flags), "rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            descriptor = -1
            if not stat.S_ISREG(os.fstat(input_stream.fileno()).st_mode):
                raise ValueError("Capture source is not a regular file")
            remaining = os.fstat(input_stream.fileno()).st_size
            while remaining:
                block = input_stream.read(min(remaining, 1024 * 1024))
                if not block:
                    raise OSError("Capture source changed during save")
                output_stream.write(block)
                remaining -= len(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination, follow_symlinks=False)
        return destination
    finally:
        if descriptor != -1:
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)
