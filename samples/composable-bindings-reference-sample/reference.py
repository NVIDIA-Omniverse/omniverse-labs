#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the default sample without installing packages (Python 3.9+, Linux/Windows)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
from composable_bindings.reference.cli import main
if __name__ == '__main__':
    main()
