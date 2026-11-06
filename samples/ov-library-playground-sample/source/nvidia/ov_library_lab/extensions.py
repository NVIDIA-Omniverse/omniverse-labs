# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional exhibit hooks; an absent extension adds no rooms or implementations."""
from importlib import import_module
from importlib.util import find_spec
import os

# Public distributions omit this module and all of its dependencies. Only the
# module's absence is optional: a broken installed extension must raise visibly.
_name = __package__ + '.internal_features'
# Match the public source edition when previewing a release in this private
# checkout. Packaging still omits the extension and all of its dependencies.
_release_preview = os.environ.get('OVLAB_RELEASE_PREVIEW') == '1'
_backend = import_module(_name) if not _release_preview and find_spec(_name) is not None else None


def value(name, default=None):
    return getattr(_backend, name, default) if _backend else default


def call(name, *args):
    callback = value(name)
    return callback(*args) if callback else None
