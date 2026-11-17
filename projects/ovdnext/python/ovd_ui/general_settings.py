# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One validated settings draft and one accepted preference transaction."""
from dataclasses import dataclass, field
import re

from .array_settings import ArrayDisplaySettings
from .formatting import FLOAT_PRECISION_MIN, FLOAT_PRECISION_MAX, ValueFormat
from .palette import PROFILE_FUNCTION_DEFAULT, parse_color
from .render_settings import RenderingSettings


def integer_text(text, label):
    """Reject fractional drafts instead of truncating them through an IntModel."""
    text = str(text).strip()
    if not re.fullmatch(r"\+?[0-9]+", text) or len(text.lstrip("+0")) > 10:
        raise ValueError(f"{label} must be a whole decimal number")
    return int(text.lstrip("+0") or "0")


@dataclass(frozen=True)
class GeneralSettings:
    rendering: RenderingSettings = field(default_factory=RenderingSettings)
    arrays: ArrayDisplaySettings = field(default_factory=ArrayDisplaySettings)
    value_format: ValueFormat = field(default_factory=ValueFormat)
    profile_function_color: str = PROFILE_FUNCTION_DEFAULT

    def __post_init__(self):
        if not isinstance(self.rendering, RenderingSettings):
            raise ValueError("Rendering settings must be validated")
        if not isinstance(self.arrays, ArrayDisplaySettings):
            raise ValueError("Array settings must be validated")
        if not isinstance(self.value_format, ValueFormat):
            raise ValueError("Property format must be validated")
        value = self.value_format
        if (isinstance(value.precision, bool) or not isinstance(value.precision, int)
                or not FLOAT_PRECISION_MIN <= value.precision <= FLOAT_PRECISION_MAX):
            raise ValueError("Float precision must be a whole number from 0 to 25")
        if value.float_format not in ("auto", "fixed", "scientific"):
            raise ValueError("Choose Auto, Fixed or Scientific float format")
        if not isinstance(value.show_hex, bool):
            raise ValueError("Hexadecimal display must be boolean")
        parse_color(self.profile_function_color)

    @classmethod
    def from_preferences(cls, preferences):
        """Read accepted values without writing or importing Qt's preferences."""
        try:
            rendering = RenderingSettings.from_preferences(preferences.get("rendering", {}))
        except (ValueError, TypeError):
            rendering = RenderingSettings()
        value_format = ValueFormat(preferences.get("float_precision", 6),
                                   preferences.get("float_format", "auto"),
                                   preferences.get("show_hex", False))
        try:
            cls(value_format=value_format)
        except ValueError:
            value_format = ValueFormat()
        color = preferences.get("ProfileFunctionColor", PROFILE_FUNCTION_DEFAULT)
        try:
            parse_color(color)
        except ValueError:
            color = PROFILE_FUNCTION_DEFAULT
        try:
            arrays = ArrayDisplaySettings.from_preferences(preferences)
        except ValueError:
            arrays = ArrayDisplaySettings()
        return cls(rendering, arrays, value_format, color)

    def to_preferences(self):
        return {"rendering": self.rendering.to_preferences(),
                "array_group_size": self.arrays.group_size,
                "max_array_to_load": self.arrays.max_array_to_load,
                "float_precision": self.value_format.precision,
                "float_format": self.value_format.float_format,
                "show_hex": self.value_format.show_hex,
                "ProfileFunctionColor": self.profile_function_color}

    def save(self, preferences):
        """All fields are already validated; publish only after this succeeds."""
        preferences.set_many(self.to_preferences())
