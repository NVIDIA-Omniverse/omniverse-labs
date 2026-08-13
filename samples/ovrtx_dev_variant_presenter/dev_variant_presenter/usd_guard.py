# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-wide lock serializing pxr USD *stage authoring*.

pxr's Sdf change manager / TfNotice dispatch is process-global and NOT thread-safe:
two threads composing or authoring stages at once (e.g. the background variant
classifier calling SetVariantSelection while the render thread runs build_composite +
open_usd) corrupt it and crash the process (observed: SetVariantSelection ->
_OpenChangeBlock -> Tf_PyInvokeImpl access violation).

Every code path that OPENS, composes, or AUTHORS a Usd stage / Sdf layer must hold this
lock for that work. Steady-state ovrtx `step()` and attribute writes do NOT (they don't
touch the pxr change manager — live camera writes run safely during classify). RLock so
a render-thread path that nests (e.g. _do_batch -> _reopen) doesn't self-deadlock.
"""
import threading

USD_LOCK = threading.RLock()
