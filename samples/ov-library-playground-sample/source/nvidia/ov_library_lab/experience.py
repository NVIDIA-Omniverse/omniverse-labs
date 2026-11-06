# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Customer-facing configuration; deliberately free of native SDK imports."""
from dataclasses import dataclass
import json
from pathlib import Path
from .campus_assets import CAMPUS_HIDDEN_ROOMS
from .extensions import value

APP_NAME = 'Omniverse Libraries Playground'
APP_SUBTITLE = 'Explore Omniverse libraries through interactive, real-world examples.'
VERSION = '0.14.0-preview'
PROFILES = ('public', 'internal') if value('ROOM_RESTRICTIONS') else ('public',)
ROOM_RESTRICTIONS = value('ROOM_RESTRICTIONS', {})
# Compatibility for internal contributor tools; public catalog is independently filtered.
INTERNAL_ROOMS = ROOM_RESTRICTIONS


@dataclass(frozen=True)
class Quality:
    width: int
    height: int
    bounces: int


QUALITIES = {
    'Performance': Quality(1024, 669, 2),
    'Balanced': Quality(1280, 836, 4),
    'Studio': Quality(1824, 1191, 6),
}


class ExperienceSettings:
    def __init__(self, root, profile='public', quality=None):
        if profile not in PROFILES:
            raise ValueError('Unknown release profile: '+profile)
        self.path = Path(root)/'preferences.json'
        self.profile = profile
        self.quality = 'Balanced'
        self.text_scale = 1.0
        self.mouse_sensitivity = 1.0
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict):
                raise ValueError('Preferences must be an object')
            if data.get('quality') in QUALITIES:
                self.quality = data['quality']
            self.text_scale = max(1., min(1.25, float(data.get('text_scale', 1.))))
            self.mouse_sensitivity = max(.3, min(1.5, float(data.get('mouse_sensitivity', 1.))))
        except (OSError, ValueError, TypeError):
            pass
        if quality in QUALITIES:
            self.quality = quality

    def allowed(self, key):
        from .content import ROOMS
        return key in ROOMS and key not in CAMPUS_HIDDEN_ROOMS and (self.profile == 'internal' or key not in ROOM_RESTRICTIONS)

    def render_spec(self, moving=False, now=None):
        # Resizing this render product rebuilds RTX resources (~1.7 s on the
        # tested laptop). Walking and stopping must never trigger a resize.
        # Ignore legacy "adaptive" preferences; preserve the chosen preset.
        quality = QUALITIES[self.quality]
        return (quality.width, quality.height), quality.bounces

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps({key: getattr(self, key) for key in
                        ('quality', 'text_scale', 'mouse_sensitivity')}, indent=2), encoding='utf-8')
        temp.replace(self.path)
