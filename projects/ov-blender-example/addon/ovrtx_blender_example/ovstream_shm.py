# SPDX-License-Identifier: Apache-2.0
"""Owned frame copies from the public OVStream SHM v2 wire contract.

No NVIDIA library is loaded in this process. Linux uses mmap/Unix sockets;
Windows uses read-only existing Win32 mappings and a bounded named pipe. The producer MUST hold publication
until it receives the copy acknowledgment for ``HeldFrame.copy_token``. The
OVStream ring itself does not grant a read lease: sequence checks alone cannot
make arbitrary Python mmap reads portable atomic/acquire operations.

The session owner supplies a unique stream name and epoch after worker startup.
It receives HeldFrame over its ordered control connection after stream_video
returns, calls copy_frame, and only then acknowledges/relinquishes the hold.
Failure also requires an explicit cancel/release through that owner. This module
never resumes the producer, removes producer resources, or manages a session.

Wire reference: NVIDIA-Omniverse/ovstream, docs/ARCHITECTURE.md, SHM transport.
"""

from __future__ import annotations

from dataclasses import dataclass
import mmap
import math
import os
from pathlib import Path
import re
import socket
import struct
import sys


class ShmFrameError(RuntimeError):
    """Invalid, stale, unavailable, or unheld shared-memory frame."""


@dataclass(frozen=True)
class HeldFrame:
    stream_name: str
    epoch: str
    sequence: int
    producer_pid: int
    width: int
    height: int
    capture_timestamp_ns: int
    copy_token: str


@dataclass(frozen=True)
class CopiedFrame:
    """Top-down tightly packed RGBA bytes; existing result boundary flips rows."""

    width: int
    height: int
    rgba8: bytes
    epoch: str
    sequence: int
    capture_timestamp_ns: int
    source_pitch_bytes: int


@dataclass(frozen=True)
class CopiedBytes:
    """Owned slot bytes without pixel-channel interpretation or row padding."""

    width: int
    height: int
    data: bytes
    epoch: str
    sequence: int
    capture_timestamp_ns: int
    source_pitch_bytes: int


@dataclass(frozen=True)
class _Region:
    slots: int
    max_width: int
    max_height: int
    max_pitch: int
    stride: int
    offset: int
    sequence: int
    latest_slot: int
    alive: int
    pid: int


_REGION = struct.Struct("<8I2Q3I")
_SLOT = struct.Struct("<QQ4I")
_MAGIC = 0x4D53564F
_MAX_REGION_BYTES = 1024 * 1024 * 1024
_STREAM_NAME = re.compile(r"[A-Za-z0-9_-]{1,63}\Z")


class OvstreamShmReader:
    """Read-only mapping whose lifetime belongs to one worker/stream epoch."""

    def __init__(self, *, stream_name: str, epoch: str, producer_pid: int,
                 shm_directory: str | Path = "/dev/shm",
                 control_directory: str | Path | None = None,
                 timeout_seconds: float = 5.0) -> None:
        if not (sys.platform.startswith("linux") or sys.platform == "win32"):
            raise ShmFrameError("SHM reader supports Linux and Windows only")
        if not _STREAM_NAME.fullmatch(stream_name):
            raise ShmFrameError("Invalid SHM stream name")
        if not epoch or producer_pid <= 0 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ShmFrameError("Stream epoch, producer PID, and positive timeout are required")
        self.stream_name = stream_name
        self.epoch = epoch
        self.producer_pid = producer_pid
        self._last_sequence = 0
        self._mapping: mmap.mmap | None = None
        self._control: socket.socket | None = None
        endpoint = Path(control_directory or os.environ.get("TMPDIR") or "/tmp")
        endpoint = endpoint / f"ovstream-{stream_name}.sock"
        try:
            if sys.platform == "win32":
                from . import win32_shm
                control = win32_shm.ControlPipe(stream_name, timeout_seconds)
            else:
                control = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._control = control
            if sys.platform != "win32":
                control.settimeout(timeout_seconds)
                control.connect(str(endpoint))
            control.sendall(b"ATTACH protocolVersion=2\n")
            reply = bytearray()
            while not reply.endswith(b"\n"):
                char = control.recv(1)
                if not char or len(reply) >= 256:
                    raise ShmFrameError("Truncated or oversized SHM attach response")
                reply.extend(char)
            match = re.fullmatch(rb"ATTACH_OK protocolVersion=2 slotCount=([2-8])\r?\n", reply)
            if match is None:
                raise ShmFrameError("OVStream refused the required SHM v2 handshake")
            if sys.platform == "win32":
                self._mapping = win32_shm.ReadOnlyMapping(stream_name, max_bytes=_MAX_REGION_BYTES)
            else:
                path = Path(shm_directory) / f"ovstream-{stream_name}"
                fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                try:
                    size = os.fstat(fd).st_size
                    if not 128 <= size <= _MAX_REGION_BYTES:
                        raise ShmFrameError("SHM region size is outside supported bounds")
                    self._mapping = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
                finally:
                    os.close(fd)
            region = self._read_region()
            if region.slots != int(match.group(1)):
                raise ShmFrameError("Attach ring depth does not match mapped region")
        except BaseException:
            self.close()
            raise

    def _read_region(self) -> _Region:
        mapping = self._mapping
        if mapping is None:
            raise ShmFrameError("SHM reader is closed")
        values = _REGION.unpack(mapping[:_REGION.size])
        magic, version, pixel_format = values[:3]
        if (magic, version, pixel_format) != (_MAGIC, 2, 1):
            raise ShmFrameError("Unsupported SHM magic, protocol, or pixel format")
        region = _Region(*values[3:])
        if not 2 <= region.slots <= 8:
            raise ShmFrameError("Invalid SHM ring depth")
        if not (0 < region.max_width <= 32768 and 0 < region.max_height <= 32768):
            raise ShmFrameError("Invalid SHM dimension capacity")
        if region.max_pitch < region.max_width * 4:
            raise ShmFrameError("SHM pitch capacity is too small")
        if region.stride < 64 + region.max_pitch * region.max_height:
            raise ShmFrameError("SHM slot stride is too small")
        if region.offset < 128 or region.offset + region.slots * region.stride > len(mapping):
            raise ShmFrameError("SHM slots exceed mapped bounds")
        if region.latest_slot >= region.slots or region.alive != 1:
            raise ShmFrameError("SHM producer stopped or latest slot is invalid")
        if region.pid != self.producer_pid:
            raise ShmFrameError("SHM producer identity changed; reattach with a new epoch")
        if sys.platform == "win32":
            from .win32_shm import process_exists
            if not process_exists(region.pid):
                raise ShmFrameError("SHM producer no longer exists")
        else:
            try:
                os.kill(region.pid, 0)
            except ProcessLookupError as exc:
                raise ShmFrameError("SHM producer no longer exists") from exc
            except PermissionError:
                pass  # Existence is established even when signaling is forbidden.
        return region

    def copy_frame(self, frame: HeldFrame) -> CopiedFrame:
        """Copy an ordinary BGRA8 image and convert channel order to RGBA8."""
        raw = self.copy_bytes(frame)
        rgba = bytearray(raw.data)
        rgba[0::4] = raw.data[2::4]
        rgba[2::4] = raw.data[0::4]
        return CopiedFrame(raw.width, raw.height, bytes(rgba), raw.epoch, raw.sequence,
                           raw.capture_timestamp_ns, raw.source_pitch_bytes)

    def copy_bytes(self, frame: HeldFrame) -> CopiedBytes:
        """Copy one ACK-held frame. Caller acknowledges only after this returns.

        A failed call publishes nothing and does not advance the reader cursor.
        The caller must cancel/release the hold or abandon the worker connection.
        """
        return self._copy_bytes(frame)

    def copy_bytes_into(self, frame: HeldFrame, destination: bytearray, *,
                        destination_offset: int = 0, count: int | None = None) -> None:
        """Copy held packed bytes directly into caller-owned assembly storage.

        The destination may contain partial bytes on failure and must be
        discarded. Success alone advances the accepted sequence; ACK remains
        the serialized session owner's responsibility.
        """
        if type(destination) is not bytearray:
            raise ShmFrameError('SHM destination must be an owned bytearray')
        if type(destination_offset) is not int or destination_offset < 0:
            raise ShmFrameError('Invalid owned destination offset')
        if count is None:
            count = frame.width * frame.height * 4
        if type(count) is not int or count < 0 or destination_offset + count > len(destination):
            raise ShmFrameError('SHM copy exceeds owned destination bounds')
        self._copy_bytes(frame, destination, destination_offset, count)

    def _copy_bytes(self, frame, destination=None, destination_offset=0, count=None):
        if not frame.copy_token:
            raise ShmFrameError("Producer copy hold is required")
        if (frame.stream_name, frame.epoch, frame.producer_pid) != (
            self.stream_name, self.epoch, self.producer_pid
        ):
            raise ShmFrameError("Held frame belongs to another stream epoch")
        if frame.sequence <= self._last_sequence:
            raise ShmFrameError("Stale or duplicate SHM frame sequence")
        region = self._read_region()
        if region.sequence != frame.sequence:
            raise ShmFrameError("Published SHM sequence does not match held frame")
        mapping = self._mapping
        assert mapping is not None
        base = region.offset + region.latest_slot * region.stride
        sequence, timestamp, width, height, pitch, reserved = _SLOT.unpack(mapping[base:base + _SLOT.size])
        if (sequence, timestamp, width, height) != (
            frame.sequence, frame.capture_timestamp_ns, frame.width, frame.height
        ):
            raise ShmFrameError("SHM slot metadata does not match held frame")
        if not (0 < width <= region.max_width and 0 < height <= region.max_height):
            raise ShmFrameError("SHM frame dimensions exceed capacity")
        if not width * 4 <= pitch <= region.max_pitch or reserved:
            raise ShmFrameError("Invalid SHM frame pitch or reserved bits")
        pixels_start = base + 64
        if pixels_start + pitch * height > base + region.stride:
            raise ShmFrameError("SHM frame exceeds slot bounds")
        if destination is not None:
            if count > width * height * 4:
                raise ShmFrameError('SHM copy exceeds packed frame bounds')
            target = memoryview(destination)
            try:
                remaining, output = count, destination_offset
                for row in range(1 if pitch == width * 4 else height):
                    size = remaining if pitch == width * 4 else min(remaining, width * 4)
                    if not size:
                        break
                    source = pixels_start + row * pitch
                    if hasattr(mapping, 'copy_into'):
                        mapping.copy_into(destination, destination_offset=output,
                                          source_offset=source, count=size)
                    else:
                        # mmap exports only a scoped read-only view; no
                        # intermediate payload bytes and no ring alias escape.
                        source_view = memoryview(mapping)[source:source + size]
                        try:
                            target[output:output + size] = source_view
                        finally:
                            source_view.release()
                    remaining -= size
                    output += size
            finally:
                target.release()
        elif pitch == width * 4:
            # One owned copy when rows are tight; keep the held-slot checks.
            bgra = mapping[pixels_start:pixels_start + pitch * height]
        else:
            bgra = b"".join(mapping[pixels_start + row * pitch:pixels_start + row * pitch + width * 4]
                            for row in range(height))
        # This verifies the ACK contract defensively; it is not a read lease.
        if _SLOT.unpack(mapping[base:base + _SLOT.size]) != (sequence, timestamp, width, height, pitch, reserved):
            raise ShmFrameError("SHM producer modified a held frame during copy")
        after = self._read_region()
        if after != region:
            raise ShmFrameError("SHM producer advanced or stopped during held copy")
        result = (None if destination is not None else
                  CopiedBytes(width, height, bgra, frame.epoch, sequence, timestamp, pitch))
        self._last_sequence = sequence
        return result

    def close(self) -> None:
        try:
            if self._mapping is not None:
                self._mapping.close()
                self._mapping = None
        finally:
            if self._control is not None:
                try:
                    self._control.sendall(b"DETACH\n")
                except OSError:
                    pass
                finally:
                    self._control.close()
                    self._control = None

    def __enter__(self) -> OvstreamShmReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
