# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire the RenderRuntime + event bridge + FastAPI app, and start the render thread."""
from __future__ import annotations

import queue

from dev_variant_presenter.api.routes import create_app
from dev_variant_presenter.config import Settings
from dev_variant_presenter.render.runtime import RenderRuntime


def build_app(settings: Settings | None = None):
    settings = settings or Settings()
    events: "queue.Queue" = queue.Queue()
    runtime = RenderRuntime(settings, on_event=events.put)
    app = create_app(runtime, settings, events)
    runtime.start()  # constructs the ovrtx.Renderer + ovstream server on the render thread
    from dev_variant_presenter import session
    session.start_checkpointer(app.state, runtime)   # crash recovery: see session.py
    return app
