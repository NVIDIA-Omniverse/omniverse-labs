# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Array display policy; independent of native publication/read budgets."""
from dataclasses import dataclass
from enum import Enum

# A logical group may contain 9,999 elements. One visible property window must
# remain small regardless of that choice; native request limits are separate.
VISIBLE_ARRAY_WINDOW = 64
MAX_ARRAY_WINDOW_BYTES = 8 * 1024 * 1024


class _Missing(Enum):
    VALUE = 0  # Preferences.get deep-copies defaults; enum identity survives.


@dataclass(frozen=True)
class ArrayDisplaySettings:
    group_size: int = 10
    max_array_to_load: int = 1000

    def __post_init__(self):
        for key, low, high in (("group_size", 10, 9999), ("max_array_to_load", 0, 99999)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be an integer between {low} and {high}")

    @classmethod
    def from_preferences(cls, preferences):
        """Read canonical keys, or migrate a valid old window size in memory.

        Invalid canonical values raise, allowing the preferences owner to report
        or recover them. An invalid obsolete key is ignored. No key is written.
        """
        missing = _Missing.VALUE
        group = preferences.get("array_group_size", missing)
        if group is missing:
            old = preferences.get("array_window", 10)
            group = max(10, old) if type(old) is int and 1 <= old <= 1000 else 10
        return cls(group, preferences.get("max_array_to_load", 1000))

    def to_preferences(self):
        return {"array_group_size": self.group_size, "max_array_to_load": self.max_array_to_load}

    def allows_expansion(self, count):
        return 0 <= count <= self.max_array_to_load

    def range_span(self, count):
        """Inclusive-index groups have at most group_size logical children."""
        span = 1
        while (count + span - 1) // span > self.group_size:
            span *= self.group_size
        return span

    def window(self, count, position=0, *, element_bytes=1):
        """One bounded leaf window, clamped to its logical group and native total."""
        if count <= 0 or element_bytes > MAX_ARRAY_WINDOW_BYTES:
            return 0, 0
        position = max(0, min(count - 1, int(position)))
        first = position // self.group_size * self.group_size
        last = min(first + self.group_size, count)
        size = min(VISIBLE_ARRAY_WINDOW, last - first, MAX_ARRAY_WINDOW_BYTES // max(1, element_bytes))
        start = max(first, min(position, last - size))
        return start, min(size, count - start)
