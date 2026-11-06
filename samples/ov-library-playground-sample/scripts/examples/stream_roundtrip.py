# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send a known RGBA frame and a return command over real ovstream shared memory.

Run with libraries-env/Scripts/python.exe. This local transport example reuses
the application's adapter; it does not start a network server or RTX renderer.
Read source/nvidia/ov_library_lab/stream_transport.py for the SDK calls.
"""
import hashlib
import time
import numpy as np
from _common import ROOT  # Configures the pinned native DLL search paths.
from nvidia.ov_library_lab.stream_transport import LocalStream


def main():
    width, height = 128, 72
    y, x = np.indices((height, width))
    frame = np.stack((x*2, y*3, (x+y)%256, np.full_like(x, 255)), axis=-1).astype(np.uint8)
    transport = LocalStream(width, height)
    try:
        deadline = time.monotonic()+5
        received = None
        # SHM may expose its initial empty frame before the first video upload
        # reaches the consumer. Wait for our payload, not just any frame.
        while time.monotonic()<deadline:
            received = transport.send(frame)
            if received is not None and np.array_equal(received, frame):
                break
            time.sleep(.01)
        assert received is not None, 'No frame arrived within five seconds'
        np.testing.assert_array_equal(received, frame)
        command = 'pause-demo'
        transport.command(command)
        deadline = time.monotonic()+5
        while command not in transport.messages and time.monotonic()<deadline:
            transport.send(frame)
            time.sleep(.01)
        assert command in transport.messages, 'The return command did not arrive'
        print('PASS: ovstream transferred identical RGBA pixels and returned:', command)
        print('Frame SHA-256:', hashlib.sha256(received.tobytes()).hexdigest())
    finally:
        transport.close()


if __name__ == '__main__':
    main()
