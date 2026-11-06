# SPDX-License-Identifier: Apache-2.0
"""Explicit application raw-byte framing over official OVStream BGRA8 SHM.

OVStream still publishes its documented uint8 BGRA transport. This application
encoding uses those byte positions as a lossless container, with a separate
typed descriptor. It does not claim native HDR pixel-format support in OVStream.
No numerical conversion, channel swizzle, or color transform is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Any

from . import frame_protocol as protocol
from .ovstream_shm import HeldFrame, OvstreamShmReader, ShmFrameError


ENCODING = "raw_bytes_over_bgra8"
FORMAT_BYTES = protocol.FORMAT_BYTES
MAX_PAYLOAD_BYTES = protocol.MAX_PAYLOAD_BYTES
MAX_CARRIER_BYTES = protocol.MAX_CARRIER_BYTES


@dataclass(frozen=True)
class RawPlane:
    name: str
    format: str
    width: int
    height: int
    offset_bytes: int
    row_bytes: int
    byte_count: int


@dataclass(frozen=True)
class RawFrameLayout:
    storage_width: int
    storage_height: int
    byte_count: int
    planes: tuple[RawPlane, ...]
    product_id: str
    aov: str

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any]) -> RawFrameLayout:
        try:
            protocol.validate_layout(source)
        except protocol.ProtocolError as exc:
            raise ShmFrameError(str(exc)) from exc
        planes = tuple(RawPlane(**{name: item[name] for name in RawPlane.__dataclass_fields__})
                       for item in source['planes'])
        return cls(source['storage_width'], source['storage_height'], source['byte_count'],
                   planes, source['product_id'], source['aov'])



@dataclass(frozen=True)
class RawPlanePayload:
    plane: RawPlane
    data: bytes | memoryview


@dataclass(frozen=True)
class CopiedRawFrame:
    layout: RawFrameLayout
    planes: tuple[RawPlanePayload, ...]
    epoch: str
    sequence: int

    def payload(self, name: str, *, pixel_format: str, width: int, height: int) -> bytes | memoryview:
        for item in self.planes:
            if item.plane.name == name:
                if (item.plane.format, item.plane.width, item.plane.height) != (pixel_format, width, height):
                    raise ShmFrameError(f"Raw plane {name} does not match requested format and dimensions")
                return item.data
        raise ShmFrameError(f"Missing raw plane {name}")


def copy_raw_frame(reader: OvstreamShmReader, held: HeldFrame, source: Mapping[str, Any]) -> CopiedRawFrame:
    try:
        protocol.validate_held(held)
    except protocol.ProtocolError as exc:
        raise ShmFrameError(str(exc)) from exc
    layout = RawFrameLayout.from_mapping(source)
    if (held.width, held.height) != (layout.storage_width, layout.storage_height):
        raise ShmFrameError("Raw descriptor and held transport dimensions disagree")
    copied = reader.copy_bytes(held)
    if len(copied.data) != layout.storage_width * layout.storage_height * 4:
        raise ShmFrameError("Raw transport returned an incomplete buffer")
    view = memoryview(copied.data).toreadonly()
    planes = tuple(RawPlanePayload(plane, view[plane.offset_bytes:plane.offset_bytes+plane.byte_count])
                   for plane in layout.planes)
    return CopiedRawFrame(layout, planes, copied.epoch, copied.sequence)


class RawFrameAssembler:
    """Bounded owned assembly of an explicitly identified multi-chunk frame.

    The session owner serializes all chunk requests, copies, and acknowledgments.
    This object never sends control messages and never releases a producer hold.
    """

    def __init__(self, held: HeldFrame, layout: Mapping[str, Any], chunk: Mapping[str, Any]):
        try:
            protocol.validate_held(held)
        except protocol.ProtocolError as exc:
            raise ShmFrameError(str(exc)) from exc
        self.layout = RawFrameLayout.from_mapping(layout)
        self._identity = (held.stream_name, held.epoch, held.producer_pid, held.width, held.height)
        self._capacity = held.width * held.height * 4
        if not (0 < held.width <= 8192 and 0 < held.height <= 8192) or self._capacity > MAX_CARRIER_BYTES:
            raise ShmFrameError("Raw chunk exceeds the 64 MiB carrier budget")
        self._count = (self.layout.byte_count + self._capacity - 1) // self._capacity
        if self._count > 65536:
            raise ShmFrameError("Raw chunk count exceeds 65536 segment budget")
        self._transfer = chunk.get("transfer_id") if isinstance(chunk, Mapping) else None
        if not isinstance(self._transfer, str) or not re.fullmatch(r"[0-9a-f]{32}", self._transfer):
            raise ShmFrameError("Raw transfer requires a unique identifier")
        self._next = 0
        self._last_sequence = 0
        self._tokens = set()
        self._timestamp = held.capture_timestamp_ns
        self._finished = False
        self._copy_failed = False
        self._validate_chunk(held, layout, chunk)
        self._buffer = bytearray(self.layout.byte_count)

    def _validate_chunk(self, held, layout, chunk):
        try:
            protocol.validate_held(held)
            protocol.fields(chunk, ("transfer_id", "index", "count", "offset_bytes", "byte_count", "total_bytes"))
        except protocol.ProtocolError as exc:
            raise ShmFrameError(str(exc)) from exc
        if held.copy_token in self._tokens:
            raise ShmFrameError("Duplicate or stale raw chunk copy token")
        if held.capture_timestamp_ns < self._timestamp:
            raise ShmFrameError("Raw chunk timestamp is stale")
        if not isinstance(chunk, Mapping) or chunk.get("transfer_id") != self._transfer:
            raise ShmFrameError("Raw chunk belongs to another transfer")
        if (held.stream_name, held.epoch, held.producer_pid, held.width, held.height) != self._identity:
            raise ShmFrameError("Raw chunk stream identity or dimensions changed")
        if RawFrameLayout.from_mapping(layout) != self.layout:
            raise ShmFrameError("Raw frame layout changed during transfer")
        offset = self._next * self._capacity
        expected = dict(index=self._next, count=self._count, offset_bytes=offset,
                        byte_count=min(self._capacity, self.layout.byte_count-offset),
                        total_bytes=self.layout.byte_count)
        for name, value in expected.items():
            if type(chunk.get(name)) is not int or chunk[name] != value:
                raise ShmFrameError(f"Raw chunk {name} is not the expected contiguous frame segment")

    def append(self, reader: OvstreamShmReader, held: HeldFrame,
               layout: Mapping[str, Any], chunk: Mapping[str, Any]) -> bool:
        """Copy one current chunk; return True when every byte has been received."""
        if self._copy_failed:
            raise ShmFrameError('Raw frame copy failed; discard the assembly')
        if self._finished or self._next >= self._count:
            raise ShmFrameError("Raw frame assembly is already complete")
        self._validate_chunk(held, layout, chunk)
        if held.sequence <= self._last_sequence:
            raise ShmFrameError("Raw chunk sequence is stale")
        start, count = chunk["offset_bytes"], chunk["byte_count"]
        try:
            direct_copy = getattr(reader, 'copy_bytes_into', None)
            if callable(direct_copy):
                direct_copy(held, self._buffer, destination_offset=start, count=count)
            else:
                copied = reader.copy_bytes(held)
                if len(copied.data) != self._capacity:
                    raise ShmFrameError("Raw chunk carrier is truncated")
                self._buffer[start:start+count] = memoryview(copied.data)[:count]
        except BaseException:
            self._copy_failed = True
            self._buffer = bytearray()
            raise
        self._tokens.add(held.copy_token)
        self._timestamp = held.capture_timestamp_ns
        self._next += 1
        self._last_sequence = held.sequence
        return self._next == self._count

    def finish(self) -> CopiedRawFrame:
        if self._copy_failed or self._finished or self._next != self._count:
            raise ShmFrameError("Raw frame is incomplete or was already consumed")
        view = memoryview(self._buffer).toreadonly()
        planes = tuple(RawPlanePayload(p, view[p.offset_bytes:p.offset_bytes+p.byte_count])
                       for p in self.layout.planes)
        del view
        # Plane views retain the owned assembly allocation, never the SDK ring.
        self._buffer = bytearray()
        self._finished = True
        return CopiedRawFrame(self.layout, planes, self._identity[1], self._last_sequence)
