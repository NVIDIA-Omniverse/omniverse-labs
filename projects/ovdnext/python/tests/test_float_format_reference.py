# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every supported precision is compared with the actual original C++ printer."""
import struct
import sys

import pytest

from ovd_ui.formatting import ValueFormat, format_value
from reference_data import load


@pytest.fixture(scope="module")
def reference():
    rows = load("float_format_windows" if sys.platform == "win32" else "float_format_linux")
    return {(r["kind"], r["mode"], r["precision"], r["raw_hex_le"]): r["expected"] for r in rows}


@pytest.mark.parametrize("mode", ("auto", "fixed", "scientific"))
@pytest.mark.parametrize("kind,type_name", (("f", "PvdF32"), ("d", "PxF64")))
def test_all_precisions_match_original_scalar_printer(reference, mode, kind, type_name):
    # Include precision-zero rounding, negative zero, both large-value threshold
    # directions, IEEE infinity/NaN and the original PhysX sentinel convention.
    numbers = (0., -0., 1., -1.25, 1.2345678901234567, .0000001,
        999999999999., 1e12, 1000000000001., -1e12, -1000000000001.,
        1e20, -1e20, float("inf"), -float("inf"), float("nan"))
    raw = [struct.pack("<" + kind, value) for value in numbers]
    raw += [bytes.fromhex(bits) for bits in (("ffff7f7f", "ffff7fff", "01000000", "0000c0ff")
        if kind == "f" else ("ffffffffffffef7f", "ffffffffffffefff", "0100000000000000", "000000000000f8ff"))]
    cases = [(precision, value) for precision in range(26) for value in raw]
    expected = [reference[kind, mode, precision, value.hex()] for precision, value in cases]
    assert len(expected) == len(cases)
    for (precision, value), text in zip(cases, expected):
        assert format_value(type_name, value, ValueFormat(precision, mode)) == text, (kind, mode, precision, value.hex(), text)


def test_compound_components_use_the_same_fractional_precision():
    raw = struct.pack("<3f", 1.25, -.5, 2.)
    assert format_value("PxVec3", raw, ValueFormat(0, "auto")) == "1, -0, 2"
    assert format_value("PxVec3", raw, ValueFormat(2, "auto")) == "1.25, -0.50, 2.00"
