# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only application protocol shared with the isolated SDK worker.

Unknown fields are forbidden except inside optional `extensions`. Required
capabilities cannot be hidden in extensions. See docs/frame-protocol.md.
"""
from collections.abc import Mapping
import re

BOOTSTRAP_VERSION = 2
CONTROL_VERSION = 1
TYPED_MAJOR = 1
TYPED_MINOR = 0
CARRIER_ABI = 2
MAX_PAYLOAD_BYTES = 256 * 1024 * 1024
MAX_CARRIER_BYTES = 64 * 1024 * 1024
FORMAT_BYTES = {"rgba8": 4, "rgba16f": 8, "rgba32f": 16, "r32f": 4, "r32u": 4}
CAPABILITIES = {"control": 1, "owned_frames": 1, "typed_frames": 1,
                "contiguous_chunks": 1, "rgba16f_bits": 1}


class ProtocolError(ValueError):
    """Application version/capability/descriptor failure, never SDK tuple failure."""


def fields(value, required, optional=()):
    if not isinstance(value, Mapping) or not set(required) <= value.keys():
        raise ProtocolError("FRAME_DESCRIPTOR: missing required fields")
    if set(value) - set(required) - set(optional) - {"extensions"}:
        raise ProtocolError("FRAME_DESCRIPTOR: unknown field; use optional extensions")
    if "extensions" in value and not isinstance(value["extensions"], dict):
        raise ProtocolError("FRAME_DESCRIPTOR: extensions must be an object")


def integer(value, name, low=1, high=(1 << 64) - 1):
    if type(value) is not int or not low <= value <= high:
        raise ProtocolError(f"FRAME_BOUNDS: {name} must be an integer in [{low},{high}]")
    return value


def offer():
    return dict(control=CONTROL_VERSION, typed_major=TYPED_MAJOR, typed_minor=TYPED_MINOR,
                carrier_abi=CARRIER_ABI, required=dict(CAPABILITIES), optional={})


def negotiate(peer):
    if not isinstance(peer, Mapping):
        raise ProtocolError("PROTOCOL_MISMATCH: application negotiation is required")
    fields(peer, ("control", "typed_major", "typed_minor", "carrier_abi", "required", "optional"))
    for key, expected in (("control", CONTROL_VERSION), ("typed_major", TYPED_MAJOR), ("carrier_abi", CARRIER_ABI)):
        if type(peer[key]) is not int or peer[key] != expected:
            raise ProtocolError(f"{'CARRIER_ABI_MISMATCH' if key == 'carrier_abi' else 'PROTOCOL_MISMATCH'}: unsupported {key} {peer[key]!r}, expected {expected}")
    integer(peer["typed_minor"], "typed minor", 0, 65535)
    required, optional = peer["required"], peer["optional"]
    if not isinstance(required, dict) or not isinstance(optional, dict):
        raise ProtocolError("CAPABILITY_MISMATCH: invalid capability objects")
    if set(required) & set(optional) or not CAPABILITIES.keys() <= required.keys():
        raise ProtocolError("CAPABILITY_MISMATCH: missing or ambiguous mandatory capability")
    for key, value in required.items():
        if type(value) is not int or CAPABILITIES.get(key) != value:
            raise ProtocolError(f"CAPABILITY_MISMATCH: unknown required capability {key}={value!r}")
    for key, value in optional.items():
        if not isinstance(key, str) or len(key) > 64 or type(value) is not int or not 1 <= value <= 65535:
            raise ProtocolError("CAPABILITY_MISMATCH: malformed optional capability")
    # Higher minor peers must emit the selected lower minor semantics. Optional
    # additions are ignorable; a new required meaning needs a new major/capability.
    return offer()


def payload_header(product_id, aov):
    return dict(version={"major": TYPED_MAJOR, "minor": TYPED_MINOR}, carrier_abi=CARRIER_ABI,
                required_capabilities=dict(CAPABILITIES), product_id=product_id, aov=aov)


def validate_layout(source):
    fields(source, ("schema", "encoding", "byte_order", "storage_width", "storage_height",
                    "byte_count", "planes", "version", "carrier_abi", "required_capabilities", "product_id", "aov"))
    if type(source["schema"]) is not int or source["schema"] != 1 or source["encoding"] != "raw_bytes_over_bgra8" or source["byte_order"] != "little":
        raise ProtocolError("PROTOCOL_MISMATCH: unsupported raw-frame encoding or schema")
    fields(source["version"], ("major", "minor"))
    version = source["version"]
    negotiate(dict(control=CONTROL_VERSION, typed_major=version["major"], typed_minor=version["minor"],
                   carrier_abi=source["carrier_abi"], required=source["required_capabilities"], optional={}))
    if version["minor"] != TYPED_MINOR:
        raise ProtocolError("PROTOCOL_MISMATCH: payload did not use negotiated minor version")
    product = source["product_id"]
    if (not isinstance(product, str) or len(product) > 1024 or not product.startswith('/')
            or not all(component.isidentifier() for component in product[1:].split('/'))):
        raise ProtocolError("FRAME_DESCRIPTOR: invalid absolute product identifier")
    if source["aov"] not in ("LdrColor", "HdrColor"):
        raise ProtocolError("FRAME_DESCRIPTOR: unsupported AOV")
    width = integer(source["storage_width"], "storage_width", high=8192)
    height = integer(source["storage_height"], "storage_height", high=8192)
    count = integer(source["byte_count"], "byte_count", high=MAX_PAYLOAD_BYTES)
    if not count <= width * height * 4 < count + width * 4:
        raise ProtocolError("FRAME_BOUNDS: raw storage must tightly cover the payload")
    records = source["planes"]
    if not isinstance(records, list) or not 1 <= len(records) <= 32:
        raise ProtocolError("FRAME_DESCRIPTOR: expected one to 32 planes")
    end, names = 0, set()
    for item in records:
        fields(item, ("name", "format", "width", "height", "offset_bytes", "row_bytes", "byte_count"))
        name, fmt = item["name"], item["format"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) or name in names:
            raise ProtocolError("FRAME_DESCRIPTOR: plane names must be unique identifiers")
        if not isinstance(fmt, str) or fmt not in FORMAT_BYTES:
            raise ProtocolError("FRAME_DESCRIPTOR: unsupported plane format")
        pw = integer(item["width"], "plane width", high=16384)
        ph = integer(item["height"], "plane height", high=16384)
        offset = integer(item["offset_bytes"], "plane offset", 0, MAX_PAYLOAD_BYTES)
        row = integer(item["row_bytes"], "plane stride", high=MAX_PAYLOAD_BYTES)
        size = integer(item["byte_count"], "plane size", high=MAX_PAYLOAD_BYTES)
        if offset != end or row != pw * FORMAT_BYTES[fmt] or size != row * ph or offset + size > count:
            raise ProtocolError("FRAME_BOUNDS: plane bounds, stride, or layout is inconsistent")
        end += size
        names.add(name)
    if end != count:
        raise ProtocolError("FRAME_BOUNDS: planes do not cover the payload")


def validate_held(frame):
    if not isinstance(frame.stream_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,63}", frame.stream_name):
        raise ProtocolError("FRAME_IDENTITY: invalid stream name")
    for name in ("epoch", "copy_token"):
        value = getattr(frame, name)
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise ProtocolError(f"FRAME_IDENTITY: invalid {name}")
    integer(frame.producer_pid, "producer PID", high=(1 << 32) - 1)
    integer(frame.sequence, "sequence")
    integer(frame.capture_timestamp_ns, "timestamp", 0)
    integer(frame.width, "carrier width", high=8192)
    integer(frame.height, "carrier height", high=8192)
    if frame.width * frame.height * 4 > MAX_CARRIER_BYTES:
        raise ProtocolError("FRAME_BOUNDS: carrier budget exceeds 64 MiB")


def validate_product(source, *, product_id, aov, width, height):
    """Bind the delivered MVP planes to the outstanding render request."""
    validate_layout(source)
    if source['product_id'] != product_id or source['aov'] != aov:
        raise ProtocolError('FRAME_IDENTITY: product/AOV differs from requested output')
    allowed = ({'linear_rgba16f': 'rgba16f', 'rgba8': 'rgba8', 'presentation_rgba32f': 'rgba32f'}
               if aov == 'HdrColor' else {'rgba8': 'rgba8'})
    required = 'linear_rgba16f' if aov == 'HdrColor' else 'rgba8'
    if required not in {plane['name'] for plane in source['planes']}:
        raise ProtocolError('FRAME_DESCRIPTOR: missing required color plane')
    for plane in source['planes']:
        if (allowed.get(plane['name']) != plane['format']
                or (plane['width'], plane['height']) != (width, height)):
            raise ProtocolError('FRAME_DESCRIPTOR: unsupported plane, format or requested dimensions')
