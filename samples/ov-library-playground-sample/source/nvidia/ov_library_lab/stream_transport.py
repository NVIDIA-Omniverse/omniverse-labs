# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual ovstream SHM transport. No network listener or simulated stream."""
import ctypes
from pathlib import Path
import os
import time
from collections import deque

import numpy as np


class LocalStream:
    def __init__(self, width=640, height=360):
        import ovstream
        import ovstream._bindings as bindings
        self.sdk = ovstream
        self.width, self.height = width, height
        self.messages = deque()
        self.sent = self.received = 0
        self.last_sequence = 0
        self.last_received = self.transfer_ms = 0.
        self.server = self.client = None
        self.cuda = None
        self.pointer = ctypes.c_void_p()
        self.initialized = False
        try:
            folder = Path(bindings._find_library()).parent
            self.cuda = ctypes.CDLL(str(next(folder.glob('cudart64_*.dll'))))
            self.cuda.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self.cuda.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self.cuda.cudaFree.argtypes = [ctypes.c_void_p]
            if self.cuda.cudaMalloc(ctypes.byref(self.pointer), width*height*4):
                raise RuntimeError('CUDA frame allocation failed')
            ovstream.initialize(log_min_severity=ovstream.LogLevel.ERROR)
            self.initialized = True
            self.server = ovstream.Server(ovstream.ServerType.SHM)
            self.server.on_message = self.messages.append
            self.name = f'ov-library-lab-{os.getpid()}-{time.monotonic_ns()}'
            self.server.start(ovstream.ServerConfig(width=width, height=height, shm_stream_name=self.name))
            self.client = ovstream.Client(ovstream.ClientType.SHM, stream_name=self.name)
        except Exception:
            self.close()
            raise

    def send(self, rgba):
        image = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        bgra = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]])
        started = time.perf_counter()
        if self.cuda.cudaMemcpy(self.pointer, bgra.ctypes.data, bgra.nbytes, 1):
            raise RuntimeError('CUDA upload failed')
        self.server.stream_video(self.sdk.VideoFrame(buffer=self.pointer.value, width=self.width,
                                                     height=self.height, pitch_bytes=self.width*4))
        self.sent += 1
        frame = self.client.wait_frame(timeout_ms=0)
        if frame is None or frame.sequence <= self.last_sequence:
            return None
        self.last_sequence = frame.sequence
        self.received += 1
        self.last_received = time.monotonic()
        self.transfer_ms = (time.perf_counter()-started)*1000.
        return np.ascontiguousarray(frame.as_numpy()[:, :frame.width, [2, 1, 0, 3]])

    def command(self, text):
        self.client.send_message(text)

    def close(self):
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.server is not None:
            self.server.stop()
            self.server.close()
            self.server = None
        if self.pointer.value and self.cuda is not None:
            self.cuda.cudaFree(self.pointer)
            self.pointer = ctypes.c_void_p()
        if self.initialized:
            self.sdk.shutdown()
            self.initialized = False
