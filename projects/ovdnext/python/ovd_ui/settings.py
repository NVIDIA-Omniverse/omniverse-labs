# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned frontend preferences; legacy Qt preferences remain independent."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile


def application_data_directory():
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return root / "NVIDIA/PVD-OVUI"


def parse_port(endpoint):
    """Accept a port or legacy host:port, while rejecting truncation/overflow."""
    text = str(endpoint).strip().rsplit(":", 1)[-1]
    if not text.isascii() or not text.isdecimal() or not 1 <= int(text) <= 65535:
        raise ValueError("Listen port must be between 1 and 65535")
    return int(text)


class Preferences:
    DEFAULTS = {"live_port": 5425, "keep_recordings": False, "recent_files": [],
                "theme": "dark", "open_directory": "", "save_directory": ""}

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else application_data_directory() / "preferences.json"
        self.values = deepcopy(self.DEFAULTS)
        self.load_error = ""
        self.load()

    def load(self):
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if document.get("version") != 1 or not isinstance(document.get("values"), dict):
                raise ValueError("Unsupported preferences file")
            values = document["values"]
            self.values.update(values)
            self.values["live_port"] = parse_port(values.get("live_port", 5425))
            if not isinstance(values.get("keep_recordings", False), bool):
                self.values["keep_recordings"] = False
            recents = values.get("recent_files", [])
            self.values["recent_files"] = list(dict.fromkeys(
                item for item in recents if isinstance(item, str)))[:8] if isinstance(recents, list) else []
            if self.values.get("theme") not in ("dark", "light"):
                self.values["theme"] = "dark"
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self.values = deepcopy(self.DEFAULTS)
            self.load_error = f"Could not load preferences: {error}"

    def get(self, key, default=None):
        return deepcopy(self.values.get(key, default))

    def set(self, key, value):
        self.set_many({key: value})

    def set_many(self, values):
        """Commit one validated batch; failed writes retain the previous state."""
        changes = deepcopy(dict(values))
        if not changes:
            return
        if "live_port" in changes:
            changes["live_port"] = parse_port(changes["live_port"])
        if "keep_recordings" in changes and not isinstance(changes["keep_recordings"], bool):
            raise ValueError("Keep recordings must be a boolean")
        # Write first; a failure must not make memory disagree with persisted UI state.
        updated = deepcopy(self.values)
        updated.update(changes)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".preferences-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump({"version": 1, "values": updated}, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.values = updated

    @property
    def recent_files(self):
        return tuple(self.values["recent_files"])

    def add_recent(self, path):
        path = str(Path(path).expanduser().resolve())
        self.set("recent_files", [path] + [item for item in self.recent_files if item != path][:7])

    def clear_recent(self):
        self.set("recent_files", [])
