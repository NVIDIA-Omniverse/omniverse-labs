# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit host capability boundary; never suppress a product failure."""
import pytest


def symlink(link, target, *, target_is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        if getattr(error, 'winerror', None) == 1314:
            pytest.skip('This Windows test token lacks symbolic-link privilege (1314)')
        raise
