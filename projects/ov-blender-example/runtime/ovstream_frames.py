# SPDX-License-Identifier: Apache-2.0
"""External-process OVStream publisher; never import this module in Blender.

One successful publication grants an application-level copy hold. All callers
must release its token after Blender owns the pixels before publishing again.
The official ring's release_frame API is not a lease and is not used here.
"""
from __future__ import annotations

import os
import functools
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
import uuid
from dataclasses import dataclass

import ovstream
import warp as wp


import runpy
from functools import lru_cache


@lru_cache(maxsize=1)
def _frame_protocol():
    root = Path(__file__).resolve().parents[1]
    path = root / 'addon/ovrtx_blender_example/frame_protocol.py'
    if not path.is_file():
        path = root / 'ovrtx_blender_example/frame_protocol.py'
    return runpy.run_path(str(path))


@wp.kernel
def _copy_rgba_to_bgra(source: wp.array3d(dtype=wp.uint8), target: wp.array3d(dtype=wp.uint8)):
    row, column = wp.tid()
    target[row, column, 0] = source[row, column, 2]
    target[row, column, 1] = source[row, column, 1]
    target[row, column, 2] = source[row, column, 0]
    target[row, column, 3] = source[row, column, 3]


@wp.kernel
def _pack_raw_plane(source: wp.array2d(dtype=wp.uint8), target: wp.array(dtype=wp.uint8),
                    offset: int, row_bytes: int):
    row, byte = wp.tid()
    index = offset + row * row_bytes + byte
    target[index] = source[row, byte]


@wp.kernel
def _prepare_hdr_upload(source: wp.array3d(dtype=wp.float16), target: wp.array3d(dtype=wp.float32)):
    row, column = wp.tid()
    target[row, column, 0] = wp.float32(source[row, column, 0])
    target[row, column, 1] = wp.float32(source[row, column, 1])
    target[row, column, 2] = wp.float32(source[row, column, 2])
    target[row, column, 3] = 1.0


_LIBRARY_LOCK = threading.Lock()
_RING_SLOTS = 2  # Official documented minimum; application ACK prevents overwrite.
_RAW_CHUNK_BYTES = 4 * 1024 * 1024
_WINDOWS_RAW_CHUNK_BYTES = 64 * 1024 * 1024


def _hdr_upload_fits(width: int, height: int, raw_bytes: int) -> bool:
    # Carrier capacity limits a chunk, not the complete owned message. An
    # explicit staging request either supplies its plane or fails before any
    # allocation; it must not silently become the staging-off experiment.
    return raw_bytes + width * height * 16 <= 256 * 1024 * 1024


@functools.lru_cache(maxsize=1)
def _windows_shm():
    # Reuse our stdlib platform primitives without importing the Blender addon
    # package or loading a second vendor library in the external worker.
    from runtime_layout import addon_helper_path
    path = addon_helper_path(__file__, 'win32_shm.py')
    spec = importlib.util.spec_from_file_location('_ovstream_publisher_win32', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _RawTransfer:
    identity: str
    layout: dict
    payload: object
    storage_width: int
    storage_height: int
    capacity: int
    count: int
    stream: object = None
    index: int = 0
    failed: bool = False


def _posix_shm_api():
    import ctypes
    import ctypes.util
    # shm_open moved from librt into libc in newer glibc. Resolve symbols,
    # never guess a mount point or use a CPython-private descriptor field.
    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, 'shm_open') or not hasattr(library, 'shm_unlink'):
        name = ctypes.util.find_library('rt')
        if not name:
            raise RuntimeError('POSIX shared-memory API is unavailable')
        library = ctypes.CDLL(name, use_errno=True)
    create = library.shm_open
    create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_uint]
    create.restype = ctypes.c_int
    unlink = library.shm_unlink
    unlink.argtypes = [ctypes.c_char_p]
    unlink.restype = ctypes.c_int
    return create, unlink


def _posix_available_memory():
    import ctypes
    create, unlink = _posix_shm_api()
    name = ('/blender-ovstream-capacity-' + uuid.uuid4().hex).encode('ascii')
    fd = create(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), name.decode('ascii'))
    unlinked = False
    primary = None
    try:
        if unlink(name) != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), name.decode('ascii'))
        unlinked = True
        stats = os.fstatvfs(fd)
        if stats.f_bavail < 0 or stats.f_frsize <= 0:
            raise RuntimeError('Invalid POSIX shared-memory capacity')
        return stats.f_bavail * stats.f_frsize
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        # Retry only our own exclusively-created name after failed unlink.
        if not unlinked:
            try:
                if unlink(name) != 0:
                    code = ctypes.get_errno()
                    errors.append(OSError(code, os.strerror(code), name.decode('ascii')))
            except Exception as error:
                errors.append(error)
        try:
            os.close(fd)
        except OSError as error:
            errors.append(error)
        if errors:
            if primary is not None:
                for error in errors:
                    primary.add_note('Capacity probe cleanup failed: ' + repr(error))
            else:
                raise ExceptionGroup('Capacity probe cleanup failed', errors)


def _check_allocation(width: int, height: int, *, max_region_bytes: int,
                      min_free_bytes: int) -> int:
    if not (0 < width <= 8192 and 0 < height <= 8192):
        raise ValueError("SHM spike dimensions must be in [1,8192]")
    pitch = ((width * 4 + 63) // 64) * 64
    required = 128 + _RING_SLOTS * (64 + pitch * height)
    if required > max_region_bytes:
        raise ValueError(f"SHM ring needs {required} bytes, exceeds {max_region_bytes} byte budget")
    if sys.platform == 'win32':
        available = _windows_shm().available_memory()
        resource = 'Windows shared-memory capacity'
    else:
        available = _posix_available_memory()
        resource = 'POSIX shared-memory capacity'
    if required + min_free_bytes > available:
        raise RuntimeError(f"Insufficient {resource}: {required} bytes plus "
                           f"{min_free_bytes} reserve required; {available} available")
    return required


class ShmPublisher:
    """ACK-held images or typed raw payloads using official CUDA-to-host copy."""

    def __init__(self, *, device: str = "cuda:0", epoch: str | None = None,
                 max_region_bytes: int = 256 * 1024 * 1024,
                 min_free_bytes: int = 16 * 1024 * 1024,
                 registry_path: str | Path | None = None,
                 raw_chunk_bytes: int | None = None):
        if max_region_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("SHM allocation budget must be positive and reserve nonnegative")
        self.epoch = epoch or uuid.uuid4().hex
        self._max_region_bytes = max_region_bytes
        self._min_free_bytes = min_free_bytes
        if raw_chunk_bytes is None:
            raw_chunk_bytes = (_WINDOWS_RAW_CHUNK_BYTES if sys.platform == 'win32' else _RAW_CHUNK_BYTES)
        if type(raw_chunk_bytes) is not int or not 4096 <= raw_chunk_bytes <= _WINDOWS_RAW_CHUNK_BYTES:
            raise ValueError("Raw carrier budget must be between 4 KiB and 64 MiB")
        self._raw_chunk_bytes = raw_chunk_bytes
        self._registry_path = Path(registry_path) if registry_path is not None else None
        self._registry_streams = set()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("OVStream SHM publication requires a CUDA producer")
        self._lock = threading.Lock()
        self._server = None
        self._buffer = None
        self._event = None
        self._stream_name = ""
        self._dimensions = None
        self._publication_sequence = 0
        self._held_token = None
        self._raw_transfer = None
        self._raw_payload_cache = None
        self._closed = False
        with _LIBRARY_LOCK:
            ovstream.initialize()

    def _start(self, width: int, height: int) -> None:
        # Validate before retiring the previous stream or allocating CUDA/SHM.
        _check_allocation(width, height, max_region_bytes=self._max_region_bytes,
                          min_free_bytes=self._min_free_bytes)
        self._stop_stream()
        self._stream_name = "blender-" + uuid.uuid4().hex
        # Persist ownership before SDK allocation so SIGKILL between allocation
        # and first frame response does not leave untraceable resources.
        self._registry_streams.add(self._stream_name)
        self._write_registry()
        server = ovstream.Server(ovstream.ServerType.SHM)
        try:
            server.start(ovstream.ServerConfig(
                width=width, height=height, cuda_device=self.device.ordinal,
                cuda_context=int(self.device.context), shm_stream_name=self._stream_name,
                shm_slot_count=_RING_SLOTS,
            ))
            self._buffer = wp.empty((height, width, 4), dtype=wp.uint8, device=self.device)
            self._event = wp.Event(device=self.device)
        except BaseException:
            server.close()
            raise
        self._server = server
        self._dimensions = (width, height)
        self._publication_sequence = 0

    def publish_rgba_cuda(self, source, width: int, height: int) -> dict:
        """Return HeldFrame fields after completed SDK publication, with no CPU pixel readback."""
        with self._lock:
            if self._closed:
                raise RuntimeError("SHM publisher is closed")
            if self._held_token is not None:
                raise RuntimeError("Previous SHM frame is still held; copy acknowledgment required")
            if (width <= 0 or height <= 0 or source.shape != (height, width, 4)
                    or source.dtype != wp.uint8 or source.device != self.device):
                raise ValueError("Expected producer-device RGBA8 CUDA array with shape (height,width,4)")
            if self._dimensions != (width, height):
                self._start(width, height)
            stream = wp.get_stream(self.device)
            wp.launch(_copy_rgba_to_bgra, dim=(height, width), inputs=[source, self._buffer],
                      device=self.device, stream=stream)
            return self._publish_buffer(width, height, stream)

    def publish_raw_planes_cuda(self, planes, *, presentation_rgba32f: bool = False,
                                product_id: str = "/Render/Color", aov: str | None = None) -> dict:
        """Pack typed CUDA planes byte-for-byte into one explicitly described frame.

        Each plane is (name, format, Warp array). This is an application byte
        encoding, not native OVStream HDR support. Even NaN payload bits survive
        because the kernel only loads/stores uint8 byte views. Input row padding
        is omitted; every declared plane is tightly packed in the result.
        Optional float32 presentation is derived on CUDA into the owned packing
        allocation. The original half-float plane remains byte-for-byte exact.
        """
        formats = {"rgba8": (wp.uint8, 4, 1), "rgba16f": (wp.float16, 4, 2),
                   "rgba32f": (wp.float32, 4, 4), "r32f": (wp.float32, 1, 4),
                   "r32u": (wp.uint32, 1, 4)}
        import re
        if type(presentation_rgba32f) is not bool:
            raise ValueError("HDR presentation selection must be boolean")
        with self._lock:
            if self._closed:
                raise RuntimeError("SHM publisher is closed")
            if self._held_token is not None:
                raise RuntimeError("Previous SHM frame is still held; copy acknowledgment required")
            if not isinstance(planes, (list, tuple)) or not 1 <= len(planes) <= 32:
                raise ValueError("Raw publication needs one to 32 typed planes")
            records, views, names, total = [], [], set(), 0
            for name, pixel_format, source in planes:
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name) or name in names:
                    raise ValueError("Raw plane names must be unique identifiers")
                if pixel_format not in formats:
                    raise ValueError("Unsupported raw plane format")
                dtype, channels, item_bytes = formats[pixel_format]
                if source.device != self.device or source.dtype != dtype or source.ndim not in (2, 3):
                    raise ValueError("Raw plane dtype/device/rank disagrees with its descriptor")
                height, width = source.shape[:2]
                if not 0 < width <= 16384 or not 0 < height <= 16384:
                    raise ValueError("Raw plane dimensions are outside supported bounds")
                if ((source.ndim == 2 and channels != 1)
                        or (source.ndim == 3 and (source.shape[2] != channels or source.strides[2] != item_bytes))
                        or source.strides[1] != channels * item_bytes):
                    raise ValueError("Raw plane channels must be contiguous")
                row_bytes = width * channels * item_bytes
                if source.strides[0] < row_bytes:
                    raise ValueError("Raw plane row stride is too small or reversed")
                size = row_bytes * height
                records.append(dict(name=name, format=pixel_format, width=width, height=height,
                    offset_bytes=total, row_bytes=row_bytes, byte_count=size))
                views.append(wp.array(ptr=source.ptr, shape=(height, row_bytes),
                    strides=(source.strides[0], 1), dtype=wp.uint8, device=self.device, copy=False))
                total += size
                names.add(name)
            prepared_source = None
            if presentation_rgba32f:
                if len(planes) != 1 or tuple(planes[0][:2]) != ("linear_rgba16f", "rgba16f"):
                    raise ValueError("HDR presentation requires one exact half-float plane")
                prepared_source = planes[0][2]
                height, width = prepared_source.shape[:2]
                size = height * width * 16
                if not _hdr_upload_fits(width, height, total):
                    raise ValueError('Requested HDR staging exceeds the 256 MiB owned-payload budget')
                records.append(dict(name="presentation_rgba32f", format="rgba32f",
                    width=width, height=height, offset_bytes=total,
                    row_bytes=width * 16, byte_count=size))
                total += size
            if total > 256 * 1024 * 1024:
                raise ValueError("Raw frame exceeds the 256 MiB owned-payload budget")
            logical_width = max(min(max(record["width"] for record in records), 8192),
                                (total + 8192*4-1)//(8192*4))
            layout = dict(schema=1, encoding="raw_bytes_over_bgra8", byte_order="little",
                storage_width=logical_width, storage_height=(total + logical_width*4-1)//(logical_width*4),
                byte_count=total, planes=records)
            policy = _frame_protocol()
            layout.update(policy['payload_header'](product_id, aov or (
                "HdrColor" if any(r['format'] == 'rgba16f' for r in records) else "LdrColor")))
            policy['validate_layout'](layout)
            policy['validate_product'](layout, product_id=layout['product_id'], aov=layout['aov'],
                                       width=records[0]['width'], height=records[0]['height'])
            # A bounded official ring is independent of full image size. Pack
            # once on the GPU so borrowed renderer maps can close before chunks
            # cross the process boundary. No host-file fallback exists.
            # Windows mappings are pagefile-backed, unlike the small Linux
            # POSIX shared-memory budget. A wide, pitch-aligned carrier avoids repeated
            # GPU readbacks/control round trips and row-by-row host copies.
            carrier_budget = getattr(self, '_raw_chunk_bytes', _RAW_CHUNK_BYTES)
            width = (min(8192, ((logical_width + 15) // 16) * 16)
                     if carrier_budget > _RAW_CHUNK_BYTES else min(logical_width, 1024))
            height = max(1, min((total + width*4-1)//(width*4), carrier_budget//(width*4), 8192))
            capacity = width * height * 4
            if self._dimensions != (width, height):
                self._start(width, height)
            payload = self._take_raw_payload(total)
            stream = wp.get_stream(self.device)
            self._raw_transfer = _RawTransfer(uuid.uuid4().hex, layout, payload,
                width, height, capacity, (total + capacity - 1) // capacity, stream=stream)
            try:
                for record, view in zip(records, views):
                    wp.launch(_pack_raw_plane, dim=(record["height"], record["row_bytes"]),
                        inputs=[view, payload, record["offset_bytes"], record["row_bytes"]],
                        device=self.device, stream=stream)
                if prepared_source is not None:
                    prepared = records[-1]
                    target = wp.array(ptr=payload.ptr + prepared["offset_bytes"],
                        shape=(prepared["height"], prepared["width"], 4),
                        dtype=wp.float32, device=self.device, copy=False)
                    wp.launch(_prepare_hdr_upload, dim=(prepared["height"], prepared["width"]),
                        inputs=[prepared_source, target], device=self.device, stream=stream)
                return self._publish_raw_chunk(stream)
            except BaseException:
                # stream_video can fail after enqueueing a GPU read. Complete
                # that read before dropping the owned source allocation.
                wp.synchronize_stream(stream)
                self._raw_transfer = None
                raise

    def _take_raw_payload(self, total):
        # One bounded owned allocation, never borrowed renderer map storage.
        # The cache is populated only after final ACK and GPU completion.
        if self._held_token is not None or self._raw_transfer is not None:
            raise RuntimeError("Raw packing storage is still in use")
        cached = getattr(self, "_raw_payload_cache", None)
        self._raw_payload_cache = None
        if cached is not None and cached.size == total:
            return cached
        return wp.empty(total, dtype=wp.uint8, device=self.device)

    def _publish_raw_chunk(self, stream):
        transfer = self._raw_transfer
        if transfer is None:
            raise RuntimeError("No raw frame transfer is active")
        offset = transfer.index * transfer.capacity
        size = min(transfer.capacity, transfer.layout["byte_count"] - offset)
        if size < transfer.capacity:
            # Full chunks overwrite every carrier byte; only a partial final
            # chunk needs clearing to avoid exposing a prior frame's tail.
            self._buffer.zero_()
        target = wp.array(ptr=self._buffer.ptr, shape=(transfer.capacity,), dtype=wp.uint8,
                          device=self.device, copy=False)
        wp.copy(target, transfer.payload, src_offset=offset, count=size, stream=stream)
        held = self._publish_buffer(transfer.storage_width, transfer.storage_height, stream)
        chunk = dict(transfer_id=transfer.identity, index=transfer.index, count=transfer.count,
                     offset_bytes=offset, byte_count=size, total_bytes=transfer.layout["byte_count"])
        return dict(shm=held, raw_frame=transfer.layout, chunk=chunk)

    def next_raw_chunk(self, copy_token: str) -> dict:
        """Acknowledge the current chunk and publish the next without rendering."""
        with self._lock:
            if not copy_token or copy_token != self._held_token or self._raw_transfer is None:
                raise RuntimeError("Raw chunk acknowledgment does not match the active transfer")
            if self._raw_transfer.failed:
                raise RuntimeError("Failed raw chunk publication requires cancellation")
            if self._raw_transfer.index + 1 >= self._raw_transfer.count:
                raise RuntimeError("Last raw chunk requires final release, not another chunk")
            self._raw_transfer.index += 1
            try:
                return self._publish_raw_chunk(self._raw_transfer.stream)
            except BaseException:
                # The SDK may have overwritten a slot before reporting failure.
                # Never retry this token and silently skip an unpublished chunk.
                self._raw_transfer.failed = True
                raise

    def cancel_raw_transfer(self, copy_token: str) -> None:
        with self._lock:
            if not copy_token or copy_token != self._held_token:
                raise RuntimeError("Raw cancellation does not match the held frame")
            # A failed SDK publication may leave CUDA work pending. Do not
            # recycle its allocation through the successful-transfer cache.
            if self._raw_transfer is not None and getattr(self._raw_transfer, 'stream', None) is not None:
                # A copy can fail before recording the next event. Synchronize
                # the captured packing stream, not the prior chunk's event or
                # the calling thread's possibly different current stream.
                wp.synchronize_stream(self._raw_transfer.stream)
            self._raw_payload_cache = None
            self._raw_transfer = None
            self._held_token = None

    def _publish_buffer(self, width, height, stream):
        stream.record_event(self._event)
        timestamp = time.monotonic_ns()
        video_frame = ovstream.VideoFrame.from_cuda_array(
            self._buffer, start_timestamp_ns=timestamp, ended_timestamp_ns=timestamp,
            sync=ovstream.CudaSync(stream=stream.cuda_stream, wait_event=self._event.cuda_event),
        )
        try:
            self._server.stream_video(video_frame)
        except BaseException:
            # Failure can follow a native slot write. Retire this stream before
            # another transfer rather than guessing whether its sequence moved.
            self._dimensions = None
            raise
        # OVStream guarantees one sequence per successful stream_video, starting
        # at 1 after start. Only this serialized publisher writes the stream.
        # Blender still validates the actual ring/slot identity before copying.
        self._publication_sequence += 1
        token = uuid.uuid4().hex
        self._held_token = token
        return dict(stream_name=self._stream_name, epoch=self.epoch,
                    sequence=self._publication_sequence, producer_pid=os.getpid(),
                    width=width, height=height,
                    capture_timestamp_ns=timestamp, copy_token=token)

    def release(self, copy_token: str) -> None:
        """Allow next publication after the owner has copied or cancelled this frame."""
        with self._lock:
            if not copy_token or copy_token != self._held_token:
                raise RuntimeError("Copy acknowledgment does not match the held SHM frame")
            if self._raw_transfer is not None and (self._raw_transfer.failed
                    or self._raw_transfer.index + 1 != self._raw_transfer.count):
                raise RuntimeError("Incomplete raw frame requires explicit cancellation")
            if self._raw_transfer is not None and getattr(self, "_event", None) is not None:
                # ACK establishes consumer ownership; the event establishes
                # producer completion, separately. Never recycle on failure.
                wp.synchronize_event(self._event)
                self._raw_payload_cache = self._raw_transfer.payload
            self._raw_transfer = None
            self._held_token = None

    def _stop_stream(self) -> None:
        server, self._server = self._server, None
        try:
            if server is not None:
                try:
                    server.stop()
                finally:
                    server.close()
                self._registry_streams.discard(self._stream_name)
                self._write_registry()
        finally:
            self._buffer = self._event = None
            self._dimensions = None

    def diagnostics(self) -> dict:
        """Owned allocations/leases only; no driver-wide or borrowed-map claims."""
        with self._lock:
            transfer = self._raw_transfer
            cache = self._raw_payload_cache
            buffer = getattr(self, '_buffer', None)
            width, height = ((buffer.shape[1], buffer.shape[0]) if buffer is not None
                             else self._dimensions or (0, 0))
            pitch = ((width * 4 + 63) // 64) * 64
            return dict(closed=self._closed, held_leases=int(self._held_token is not None),
                active_transfers=int(transfer is not None),
                active_payload_bytes=0 if transfer is None else transfer.layout['byte_count'],
                cached_payload_bytes=0 if cache is None else cache.size,
                carrier_gpu_bytes=width * height * 4,
                shm_region_bytes=0 if self._server is None else 128 + _RING_SLOTS * (64 + pitch * height),
                pending_chunks=0 if transfer is None else transfer.count-transfer.index-1,
                registered_streams=len(self._registry_streams))

    def _write_registry(self) -> None:
        if self._registry_path is None:
            return
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._registry_path.with_name(self._registry_path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(dict(schema=1, pid=os.getpid(), epoch=self.epoch,
                                                streams=sorted(self._registry_streams))) + "\n")
            # Windows readers/AV may briefly hold the old file without delete
            # sharing. Keep the previous complete registry visible and retry
            # only those OS errors, for at most 310 ms. Permanent denial remains
            # a teardown error; it must never be reported as clean shutdown.
            delays = (.01, .02, .04, .08, .16)
            for attempt in range(len(delays) + 1):
                try:
                    os.replace(temporary, self._registry_path)
                    break
                except PermissionError as error:
                    if (sys.platform != 'win32' or getattr(error, 'winerror', None) not in (5, 32, 33)
                            or attempt == len(delays)):
                        raise
                    time.sleep(delays[attempt])
        finally:
            # Only our unique temporary file; never sweep another writer's files.
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                try:
                    transfer = self._raw_transfer
                    if transfer is not None and getattr(transfer, 'stream', None) is not None:
                        wp.synchronize_stream(transfer.stream)
                finally:
                    self._stop_stream()
            finally:
                self._held_token = None
                self._raw_transfer = None
                self._raw_payload_cache = None
                with _LIBRARY_LOCK:
                    ovstream.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
