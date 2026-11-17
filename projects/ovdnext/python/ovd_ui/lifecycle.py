# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Own the public standalone pump, application task and backend lifetime.

Both factories return coroutines; setup, close and shutdown are synchronous
main-thread operations. No task or GL resource outlives backend shutdown.
The public OVUI ``run_async`` pump deliberately performs no backend teardown.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any


RUNNER_ID = "owned-public-asyncio-v1"


def raise_errors(message: str, errors: list[BaseException]) -> None:
    """Keep original failures, including non-Exception cancellation/interrupts."""
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup(message, errors)


def _task(factory: Callable[[], Coroutine[Any, Any, Any]], name: str):
    coroutine = factory()
    try:
        return asyncio.create_task(coroutine, name=name)
    except BaseException:
        # A rejected task factory has not taken ownership of this coroutine.
        coroutine.close()
        raise


async def run_owned(application_factory, pump_factory, *, setup, close, shutdown):
    """Join both tasks, close resources, then shut down the backend.

Only cancellations requested here are normal. Every other task/setup/cleanup
failure is retrieved and propagated. Cleanup is shielded and joined even if
the owner receives another cancellation while shutting down.
"""
    application = pump = None
    errors = []
    try:
        setup()
        application = _task(application_factory, "pvd-application")
        pump = _task(pump_factory, "pvd-standalone-pump")
        await asyncio.wait((application, pump), return_when=asyncio.FIRST_COMPLETED)
    except BaseException as error:
        errors.append(error)

    owner = asyncio.current_task()
    cancellations_seen = owner.cancelling()
    # Stop native ticks first. No new cleanup task is needed: even a task
    # factory failure must still reach resource cleanup and backend shutdown.
    for task in (pump, application):
        if task is None:
            continue
        if not task.done() and task.cancelling():
            # asyncio.timeout briefly cancels its own task, then uncancels it
            # when translating CancelledError to TimeoutError. Let that ready
            # cancellation run before deciding the task is already stopping.
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError as error:
                if owner.cancelling() > cancellations_seen:
                    errors.append(error)
                    cancellations_seen = owner.cancelling()
        # An independently cancelling task retains its original failure and
        # must not receive a second cancellation inside its finally block.
        requested = not task.done() and task.cancelling() == 0
        if requested:
            task.cancel()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if owner.cancelling() > cancellations_seen:
                    errors.append(error)
                    cancellations_seen = owner.cancelling()
            except BaseException:
                # Retrieve the original task failure exactly once below.
                break
        try:
            task.result()
        except asyncio.CancelledError as error:
            if not requested:
                errors.append(error)
        except BaseException as error:
            errors.append(error)
    for name, callback in (("Application resource cleanup", close),
                           ("Standalone backend shutdown", shutdown)):
        try:
            callback()
        except BaseException as error:
            error.add_note(name)
            errors.append(error)
    raise_errors("Standalone lifecycle failed", errors)


def run_standalone(application_factory, pump_factory, *, setup, close, shutdown):
    """Blocking product entry point; embedded callers may await run_owned."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("run_standalone requires no running event loop; await run_owned instead")
    return asyncio.run(run_owned(application_factory, pump_factory,
                                 setup=setup, close=close, shutdown=shutdown))
