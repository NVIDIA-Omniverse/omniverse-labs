# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Tests for Python ovRTX error-handling snippets."""

from pathlib import Path

import pytest

MISSING_USD_PATH = str((Path(__file__).parent / "../data/this-file-does-not-exist.usda").resolve())


def test_sync_method_raises_runtime_error(renderer):
    """Synchronous Python APIs surface operation failures as RuntimeError."""
    # [snippet:doc-python-sync-runtime-error]
    with pytest.raises(RuntimeError, match="open_usd"):
        renderer.open_usd(MISSING_USD_PATH)
    # [/snippet:doc-python-sync-runtime-error]


def test_async_wait_raises_runtime_error(renderer):
    """Async Python APIs surface operation failures when wait() is called."""
    # [snippet:doc-python-async-operation-error]
    op = renderer.open_usd_async(MISSING_USD_PATH)
    with pytest.raises(RuntimeError, match="open_usd"):
        op.wait()
    # [/snippet:doc-python-async-operation-error]
