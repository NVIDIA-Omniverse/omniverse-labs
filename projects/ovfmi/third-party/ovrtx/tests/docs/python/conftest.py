# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Pytest configuration for ovrtx documentation tests."""

from pathlib import Path

import ovrtx
import pytest


@pytest.fixture(scope="session")
def output_dir():
    """Return the _output directory, creating it if needed."""
    d = Path(__file__).parent / "_output"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def renderer(output_dir):
    """Create a Renderer for an individual test."""
    config = ovrtx.RendererConfig(
        log_file_path=str(output_dir / "python-doc-tests-ovrtx.log"),
    )
    r = ovrtx.Renderer(config=config)
    yield r
    del r
