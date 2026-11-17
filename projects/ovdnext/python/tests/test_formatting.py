# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
from ovd_ui.formatting import ValueFormat,element_size,format_value


def test_native_integer_precision_signedness_and_sample_width():
    assert format_value("PvdU64",struct.pack("<Q",2**64-1))=="18446744073709551615"
    assert format_value("PxI64",struct.pack("<q",-2**63))=="-9223372036854775808"
    assert format_value("PvdU32",b"\xff")=="255"
    assert format_value("PvdI16",b"\xff")=="-1"
    assert format_value("PvdI8",b"\xff",ValueFormat(show_hex=True))=="-1 (0xff)"


def test_boolean_recorded_width_and_scalar_arrays():
    assert format_value("PvdBool",b"\1\0\0\0")=="true"
    assert format_value("PvdBool",b"\0")=="false"
    assert format_value("PvdBool",b"\1\0",array=True)=="true, false"
    assert format_value("U32Array4",struct.pack("<4I",1,2,3,4))=="1, 2, 3, 4"
    assert element_size("U32Array4")==16


def test_float_format_and_bounded_uninterpreted_data():
    assert format_value("PxF64",struct.pack("<d",1.25),ValueFormat(3,"fixed"))=="1.250"
    assert format_value("PvdF32",struct.pack("<f",1.25),ValueFormat(2,"scientific"))=="1.25e+00"
    assert format_value("PxF32",b"\0").startswith("Incomplete recorded value")
    assert format_value("PvdU16",struct.pack("<20H",*range(20))).endswith("… [20 values]")
    assert format_value("Unknown",b"\x55"*100).endswith("… [100 bytes]")
    assert element_size("PxTransform")==28
