# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original colors.json semantics with independent, atomically saved UI settings."""
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np
from PIL import ImageColor

from .settings import application_data_directory

ACTOR_KEYS = ("rigidStatic", "rigidDynamicActive", "rigidDynamicSleep", "kinematicActive",
              "kinematicSleep", "articulation", "cloth", "particle", "deformableActive",
              "deformableSleep", "fallback")
GIZMO_KEYS = ("contactPoint", "contactNormal", "jointParentFrame", "jointChildFrame",
              "boundingBox", "centerOfMass", "velocity", "axisX", "axisY", "axisZ", "selectionTint")
PALETTE_KEYS = {"actors": ACTOR_KEYS, "gizmos": GIZMO_KEYS}
UNUSED_ACTOR_KEYS = frozenset(("kinematicSleep", "particle", "deformableActive", "deformableSleep", "fallback"))
# PVDRenderer::loadOvdParticles supplies this directly; actors.particle is unused.
OVD_PARTICLE_BUFFER_COLOR = (.2, .6, 1., 1.)
MAGENTA = (1., 0., 1., 1.)
PROFILE_FUNCTION_DEFAULT = "#00FF00"  # Separate legacy QSettings key, not colors.json.
DEFAULT_PALETTE_PATH = Path(__file__).resolve().parent / "assets/colors.json"
# Must match the explicit native OverlayColorRole enum; never infer roles from RGB.
OVERLAY_ROLE_KEYS = {1: "boundingBox", 2: "centerOfMass", 3: "velocity", 4: "axisX",
                     5: "axisY", 6: "axisZ", 7: "jointParentFrame", 8: "jointChildFrame",
                     9: "contactPoint", 10: "contactNormal"}


def parse_color(value):
    """QColor's SVG names and RGB/ARGB hex forms, including 12/16-bit RGB.

    Verified against Qt 5.15 QColor; CSS functional colors and rebeccapurple are
    intentionally not accepted. Named colors ignore spaces/tabs as Qt does.
    """
    if not isinstance(value, str):
        raise ValueError("Color must be a string")
    if value.startswith("#"):
        text = value[1:]
        if len(text) not in (3, 6, 8, 9, 12) or not re.fullmatch(r"[0-9a-fA-F]+", text):
            raise ValueError("Use an RGB or ARGB hex color")
        alpha = 1.
        if len(text) == 8:
            alpha, text = int(text[:2], 16) / 255., text[2:]
        width = len(text) // 3
        components = [int(text[i:i + width], 16) for i in range(0, len(text), width)]
        if width == 3:
            components = [(value << 4) | (value >> 8) for value in components]
            maximum = 65535
        else:
            maximum = (16 ** width) - 1
        return (*[component / maximum for component in components], alpha)
    name = value.lower().replace(" ", "").replace("\t", "")
    if name == "transparent":
        return (0., 0., 0., 0.)
    if name == "rebeccapurple" or name not in ImageColor.colormap:
        raise ValueError("Unknown SVG color name")
    return (*[component / 255. for component in ImageColor.getrgb(name)], 1.)


def color_hex(rgba):
    components = [int(np.clip(float(value), 0, 1) * 255 + .5) for value in rgba]
    red, green, blue, alpha = components
    return (f"#{alpha:02X}" if alpha != 255 else "#") + f"{red:02X}{green:02X}{blue:02X}"


@dataclass(frozen=True)
class PaletteIssue:
    section: str
    key: str
    message: str


class ColorPalette:
    """Poll on the UI/render thread; inode-aware checks survive editor replace.

    An explicit path is read as-is, including missing or invalid files. With no
    path, initialize a separate user palette from the shipped JSON on first use.
    No legacy Qt settings or source palette file is ever written implicitly.
    """
    def __init__(self, path=None, *, defaults_path=DEFAULT_PALETTE_PATH, poll_interval=.25):
        self.path = Path(path) if path is not None else application_data_directory() / "colors.json"
        self.defaults_path = Path(defaults_path)
        self.poll_interval = float(poll_interval)
        self.revision = 0
        self.document = {}
        self.issues = ()
        self.load_error = ""
        self._colors = {}
        self._stamp = None
        self._payload = None
        self._next_poll = 0.
        if path is None and not self.path.exists() and self.defaults_path.is_file():
            self._write(self.defaults_path.read_bytes())
        self.reload()

    def _stat(self):
        try:
            stat = self.path.stat()
            return self._identity(stat)
        except OSError:
            return None

    @staticmethod
    def _identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def poll(self, *, force=False):
        now = time.monotonic()
        if not force and now < self._next_poll:
            return False
        self._next_poll = now + self.poll_interval
        if force or self._stat() != self._stamp:
            return self.reload()
        return False

    def reload(self):
        error = ""
        stamp = self._stat()
        try:
            with self.path.open("rb") as stream:
                payload = stream.read()
                # Retain the opened inode's identity. If an editor replaces the
                # path during this read, the following poll sees the new inode.
                stamp = self._identity(os.fstat(stream.fileno()))
            document = json.loads(payload)
            if not isinstance(document, dict):
                raise ValueError("Palette root must be a JSON object")
        except (OSError, ValueError, UnicodeError) as exception:
            payload = None
            document = {}
            error = str(exception)
        if self.revision and (payload, error, stamp) == (self._payload, self.load_error, self._stamp):
            return False
        self.document, self.load_error = document, error
        self._payload, self._stamp = payload, stamp
        colors, issues = {}, []
        for section, keys in PALETTE_KEYS.items():
            values = document.get(section)
            for key in keys:
                value = values.get(key) if isinstance(values, dict) else None
                try:
                    colors[section, key] = parse_color(value)
                except ValueError as exception:
                    colors[section, key] = MAGENTA
                    issues.append(PaletteIssue(section, key, "Missing color" if value is None else str(exception)))
        self._colors, self.issues = colors, tuple(issues)
        self.revision += 1
        return True

    def get(self, section, key):
        if (section, key) in self._colors:
            return self._colors[section, key]
        try:
            return parse_color(self.raw(section, key))
        except ValueError:
            return MAGENTA

    def actor(self, key):
        return self.get("actors", key)

    def gizmo(self, key):
        return self.get("gizmos", key)

    def raw(self, section, key):
        values = self.document.get(section)
        value = values.get(key) if isinstance(values, dict) else None
        return value if isinstance(value, str) else ""

    def _write(self, payload):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".colors-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def edit(self, changes):
        """Atomically merge edited keys into the latest document, retaining extras."""
        for (section, key), value in changes.items():
            if key not in PALETTE_KEYS.get(section, ()):
                raise ValueError(f"Unknown palette entry: {section}.{key}")
            parse_color(value)
        self.poll(force=True)
        updated = deepcopy(self.document)
        for (section, key), value in changes.items():
            if not isinstance(updated.get(section), dict):
                updated[section] = {}
            updated[section][key] = value
        self._write((json.dumps(updated, indent=4) + "\n").encode("utf-8"))
        self.reload()

    def default_changes(self):
        document = json.loads(self.defaults_path.read_bytes())
        return {(section, key): document[section][key] for section, keys in PALETTE_KEYS.items() for key in keys}

    def reset_defaults(self):
        self.edit(self.default_changes())

    def import_file(self, path):
        payload = Path(path).read_bytes()
        if not isinstance(json.loads(payload), dict):
            raise ValueError("Palette root must be a JSON object")
        self._write(payload)
        self.reload()

    def geometry_colors(self, snapshot, class_names):
        """Reference actor palette and explicit OVD particle-buffer base color."""
        count = len(snapshot.object_ids)
        result = np.tile(self.actor("articulation"), (count, 1)).astype("f4")
        classes = getattr(snapshot, "actor_classes", np.zeros(count, dtype="i4"))
        sleeping = np.asarray(getattr(snapshot, "sleeping", np.zeros(count)), dtype=bool)
        flags = np.asarray(getattr(snapshot, "actor_flags", np.zeros(count)))
        for class_id in np.unique(classes):
            name = class_names.get(int(class_id), "")
            rows = classes == class_id
            if name == "PxRigidStatic":
                result[rows] = self.actor("rigidStatic")
            elif name == "PxCloth":
                result[rows] = self.actor("cloth")
            elif name == "PxRigidDynamic":
                result[rows] = self.actor("rigidDynamicActive")
                result[rows & (flags != 0)] = self.actor("kinematicActive")
                # The source overwrites sleeping kinematics with this key too.
                result[rows & sleeping] = self.actor("rigidDynamicSleep")
            elif name == "PxPBDParticleSystem":
                # The native OVD path submits blue particle-buffer billboards
                # outside updateRender's ordinary actor palette dispatch. Require
                # its owned native row witnesses; other points keep their colors.
                fields = [getattr(snapshot, field, ()) for field in
                          ("kinds", "particle_radius_modes", "shape_ids", "geometry_ids")]
                if all(len(field) == count for field in fields):
                    kinds, radius_modes, shapes, geometries = map(np.asarray, fields)
                    buffers = (kinds == 7) & (radius_modes == 2) & (geometries == 0)
                    buffers &= (shapes > 0) & (shapes != snapshot.object_ids)
                    result[rows & buffers] = OVD_PARTICLE_BUFFER_COLOR
        result[:, 3] = 1.  # Actor alpha is controlled by visualization settings.
        return result

    def overlay_colors(self, snapshot):
        colors = snapshot.colors
        roles = getattr(snapshot, "color_roles", ())
        if len(roles) != len(snapshot.kinds) or not np.any(roles):
            return colors
        result = colors.copy()
        for role, key in OVERLAY_ROLE_KEYS.items():
            color = self.gizmo(key)
            if key == "centerOfMass":
                color = (*color[:3], 1.)  # Original COM rendering ignores QColor alpha.
            result[roles == role] = color
        return result
