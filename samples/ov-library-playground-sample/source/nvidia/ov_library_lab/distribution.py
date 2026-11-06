# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-identifying build metadata for local pilot feedback."""
import json
from pathlib import Path


def identity():
    path=Path(__file__).resolve().parents[3]/'config/distribution.json'
    if path.is_file():
        return json.loads(path.read_text(encoding='utf-8'))
    return {'audience':'development','build_id':'working-tree','automatic_uploads':False}
