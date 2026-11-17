# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded text presentation for owned native property values."""
import math
import sys
import struct
from dataclasses import dataclass

_SCALARS={"PvdBool":"?","PvdI8":"b","PvdU8":"B","PvdI16":"h","PvdU16":"H",
          "PvdI32":"i","PvdU32":"I","PvdI64":"q","PvdU64":"Q","PvdF32":"f","PvdF64":"d"}
_SCALARS.update({"Px"+name[3:]:code for name,code in tuple(_SCALARS.items())})
_SCALARS["U32Array4"]="I"
_FLOAT_TYPES={"PxVec3","PxVec4","PxQuat","PxTransform","PxBounds3","PxMat3","PxMat33","PxMat44"}
_COMPONENTS={"PxVec3":3,"PxVec4":4,"PxQuat":4,"PxTransform":7,"PxBounds3":6,"PxMat3":9,"PxMat33":9,"PxMat44":16,"U32Array4":4}
FLOAT_PRECISION_MIN, FLOAT_PRECISION_MAX = 0, 25
_F32_MAX = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
_F64_MAX = float.fromhex("0x1.fffffffffffffp+1023")
_F32_AUTO_THRESHOLD = struct.unpack("<f", struct.pack("<f", 1e12))[0]


def _float_text(value, code, options):
    """Match PrettyPrinter's fractional precision and recorded PhysX sentinels."""
    precision=max(FLOAT_PRECISION_MIN,min(FLOAT_PRECISION_MAX,int(options.precision)))
    if code=="f" and abs(value)==_F32_MAX:
        return "-Infinity" if value<0 else "Infinity"
    if code=="d" and value==_F64_MAX:
        return "Infinity"
    if math.isnan(value):
        return ("-nan(ind)" if sys.platform == "win32" else "-nan") if math.copysign(1.,value)<0 else "nan"
    mode="e" if options.float_format=="scientific" else "f"
    if options.float_format=="auto":
        # The reference applies abs() only to F32, and compares with a float32
        # threshold. Keep that recorded-value convention for existing captures.
        if (abs(value)>_F32_AUTO_THRESHOLD if code=="f" else value>1e12):mode="e"
    return format(value,f".{precision}{mode}")


@dataclass(frozen=True)
class ValueFormat:
    precision:int=6
    float_format:str="auto"
    show_hex:bool=False


def element_size(type_name):
    code="f" if type_name in _FLOAT_TYPES else _SCALARS.get(type_name)
    if type_name in ("ObjectRef","StringHandle"):return 4
    return struct.calcsize(code)*_COMPONENTS.get(type_name,1) if code else 0


def format_value(type_name,raw,options=ValueFormat(),*,array=False,limit=16,total_bytes=None):
    """Format owned bytes; total_bytes preserves native size for raw previews."""
    if not raw:return "—"
    if type_name in ("PvdBool","PxBool") and not array:
        return "true" if int.from_bytes(raw,"little") else "false"
    code="f" if type_name in _FLOAT_TYPES else _SCALARS.get(type_name)
    if code:
        width=struct.calcsize(code)
        if len(raw)<width and code not in "fd":
            values=(int.from_bytes(raw,"little",signed=code in "bhiq"),)
            count=1
        elif len(raw)%width==0:
            count=len(raw)//width
            values=struct.unpack("<"+code*min(count,limit),raw[:width*limit])
        else:return "Incomplete recorded value: "+raw.hex(" ")
        def number(value):
            if code in "fd":return _float_text(value,code,options)
            if code=="?":return "true" if value else "false"
            text=str(value)
            return f"{text} (0x{value & ((1 << (width*8))-1):0{width*2}x})" if options.show_hex else text
        return ", ".join(number(value) for value in values)+(f" … [{count:,} values]" if count>limit else "")
    preview=raw[:32]
    if total_bytes is not None and total_bytes>len(raw):
        suffix=f" … [{len(preview):,} of {total_bytes:,} bytes]"
    else:
        suffix=f" … [{len(raw):,} bytes]" if len(raw)>32 else ""
    return preview.hex(" ")+suffix


def format_record_field(field,precision=6,limit=16):
    """Use native field types and bytes; never decode engine memory layouts here."""
    if field.type=="StringHandle":
        return field.string_value or "—"
    return format_value(field.type,field.data,ValueFormat(precision),array=field.array,limit=limit)
