# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local security defaults for the viewer's entry points."""
import sys


def configure_runtime_security():
    """Keep Windows MIME detection out of user-writable POSIX default paths."""
    if sys.platform != "win32":
        return
    import mimetypes

    # init([]) still prepends knownfiles on first initialization. Clear the
    # defaults and rebuild even if another import initialized the database.
    # Built-in mappings and the normal Windows registry lookup remain enabled.
    mimetypes.knownfiles = []
    mimetypes.init()
