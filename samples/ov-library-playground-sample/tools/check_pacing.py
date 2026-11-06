# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for scoped Windows UI/asyncio clocks and shutdown."""
import asyncio
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source'))
from nvidia.ov_library_lab import pacing


class FramePacing(unittest.TestCase):
    def test_runner_uses_precise_deadlines_and_closes_owned_loop(self):
        loops = []
        async def probe():
            loop = asyncio.get_running_loop()
            loops.append(loop)
            self.assertIs(loop.time, time.perf_counter)
            self.assertEqual(loop._clock_resolution, time.get_clock_info('perf_counter').resolution)
            before = time.perf_counter()
            await asyncio.sleep(.015)
            self.assertGreaterEqual(time.perf_counter()-before, .014)
            return 42
        with patch.object(pacing, 'high_resolution_clock', return_value=True):
            self.assertEqual(pacing.run_event_loop(probe()), 42)
        self.assertTrue(loops[0].is_closed())

    def test_sdk_clock_is_local_and_restored_on_failure(self):
        sdk = SimpleNamespace(time=time)
        original = time.monotonic
        with patch.dict(sys.modules, {'omni.ui.standalone': sdk}), patch.object(pacing, 'high_resolution_clock', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'shutdown'):
                with pacing.precise_timers():
                    self.assertIs(sdk.time.monotonic, time.perf_counter)
                    self.assertIs(sdk.time.sleep, time.sleep)
                    self.assertIs(time.monotonic, original)
                    raise RuntimeError('shutdown')
            self.assertIs(sdk.time, time)

    def test_diagnostic_opt_out(self):
        with patch.dict(os.environ, {'OVLAB_LEGACY_CLOCK': '1'}):
            self.assertFalse(pacing.high_resolution_clock())

    def test_fine_clock_keeps_sdk_reference(self):
        sdk = SimpleNamespace(time=time)
        with patch.dict(sys.modules, {'omni.ui.standalone': sdk}), patch.object(pacing, 'high_resolution_clock', return_value=False):
            with pacing.precise_timers():
                self.assertIs(sdk.time, time)

    def test_runner_closes_loop_on_error(self):
        loops = []
        async def probe():
            loops.append(asyncio.get_running_loop())
            raise ValueError('expected')
        with self.assertRaisesRegex(ValueError, 'expected'):
            pacing.run_event_loop(probe())
        self.assertTrue(loops[0].is_closed())


if __name__ == '__main__':
    unittest.main()
