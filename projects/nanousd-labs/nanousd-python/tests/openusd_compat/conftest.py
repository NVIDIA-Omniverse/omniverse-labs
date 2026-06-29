# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def pytest_ignore_collect(collection_path, config):
    path = Path(str(collection_path))
    return "openusd_compat" in path.parts and "upstream" in path.parts
