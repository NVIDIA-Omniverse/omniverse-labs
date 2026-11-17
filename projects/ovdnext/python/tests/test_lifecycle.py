# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task ordering and error ownership; no OVUI/backend/GPU is initialized."""
import asyncio
import inspect

import pytest

from ovd_ui.lifecycle import run_owned, run_standalone


def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


def callbacks(events, *, setup_error=None, close_error=None, shutdown_error=None):
    def action(name, error):
        def invoke():
            events.append(name)
            if error is not None:raise error
        return invoke
    return dict(setup=action("setup", setup_error), close=action("close", close_error),
                shutdown=action("shutdown", shutdown_error))


@pytest.mark.parametrize("first", ["application", "pump"])
def test_either_completion_joins_both_tasks_before_cleanup(first):
    events=[]
    async def scenario():
        ready={name:asyncio.Event() for name in ("application","pump")}
        async def work(name):
            try:
                events.append(name+" entered")
                ready[name].set()
                await ready["pump" if name=="application" else "application"].wait()
                if name!=first:await asyncio.Future()
            finally:
                # Async cleanup cannot be abandoned by simply closing a coro.
                await asyncio.sleep(0)
                events.append(name+" joined")
        await run_owned(lambda:work("application"),lambda:work("pump"),**callbacks(events))
        assert asyncio.all_tasks()=={asyncio.current_task()}
    asyncio.run(scenario())
    assert events[0]=="setup"
    assert events[-2:]==["close","shutdown"]
    assert set(events[-4:-2])=={"application joined","pump joined"}


@pytest.mark.parametrize("first", ["application", "pump"])
def test_task_failure_is_preserved_and_other_task_settles(first):
    failure=ValueError(first)
    events=[]
    async def bad():raise failure
    async def wait():
        try:await asyncio.Future()
        finally:events.append("other joined")
    with pytest.raises(ValueError) as caught:
        run_standalone(bad if first=="application" else wait,
                       bad if first=="pump" else wait,**callbacks(events))
    assert caught.value is failure
    assert events[-3:]==["other joined","close","shutdown"]


def test_simultaneous_task_and_cleanup_failures_are_all_retrieved():
    failures=[ValueError("application"),RuntimeError("pump"),OSError("close"),LookupError("shutdown")]
    events=[]
    async def fail(error):raise error
    with pytest.raises(ExceptionGroup) as caught:
        run_standalone(lambda:fail(failures[0]),lambda:fail(failures[1]),
            **callbacks(events,close_error=failures[2],shutdown_error=failures[3]))
    assert {id(error) for error in leaves(caught.value)}=={id(error) for error in failures}
    assert events==["setup","close","shutdown"]


def test_setup_failure_does_not_construct_coroutines_but_closes_resources():
    events=[]
    failure=RuntimeError("partial setup")
    def forbidden():pytest.fail("factory called after setup failure")
    with pytest.raises(RuntimeError) as caught:
        run_standalone(forbidden,forbidden,**callbacks(events,setup_error=failure))
    assert caught.value is failure
    assert events==["setup","close","shutdown"]


def test_application_factory_failure_still_cleans_up():
    events=[]
    def fail():raise ValueError("factory failed")
    with pytest.raises(ValueError,match="factory failed"):
        run_standalone(fail,fail,**callbacks(events))
    assert events==["setup","close","shutdown"]


@pytest.mark.parametrize("failed_creation", [1,2])
def test_task_creation_failure_closes_unowned_coroutine_and_never_needs_cleanup_task(monkeypatch,failed_creation):
    events=[]
    created=[]
    async def work():await asyncio.Future()
    def factory():
        coroutine=work();created.append(coroutine);return coroutine
    async def scenario():
        original=asyncio.create_task
        count=0
        def reject(coroutine,**kwargs):
            nonlocal count
            count+=1
            if count>=failed_creation:raise RuntimeError("task factory failed")
            return original(coroutine,**kwargs)
        monkeypatch.setattr(asyncio,"create_task",reject)
        with pytest.raises(RuntimeError,match="task factory failed"):
            await run_owned(factory,factory,**callbacks(events))
        assert asyncio.all_tasks()=={asyncio.current_task()}
    asyncio.run(scenario())
    assert events==["setup","close","shutdown"]
    assert all(inspect.getcoroutinestate(coro)==inspect.CORO_CLOSED for coro in created)


def test_external_cancellation_is_propagated_after_async_task_cleanup():
    events=[]
    async def scenario():
        entered=asyncio.Event()
        async def application():
            entered.set()
            try:await asyncio.Future()
            finally:events.append("application joined")
        async def pump():
            try:await asyncio.Future()
            finally:
                await asyncio.sleep(0)
                events.append("pump joined")
        owner=asyncio.create_task(run_owned(application,pump,**callbacks(events)))
        await entered.wait()
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):await owner
        assert asyncio.all_tasks()=={asyncio.current_task()}
    asyncio.run(scenario())
    assert events==["setup","pump joined","application joined","close","shutdown"]


def test_repeated_owner_cancel_does_not_cancel_or_abandon_pending_task_cleanup():
    events=[]
    async def scenario():
        entered=asyncio.Event();joining=asyncio.Event();release=asyncio.Event()
        async def application():
            entered.set()
            try:await asyncio.Future()
            finally:events.append("application joined")
        async def pump():
            try:await asyncio.Future()
            finally:
                joining.set()
                await release.wait()
                events.append("pump joined")
        owner=asyncio.create_task(run_owned(application,pump,**callbacks(events)))
        await entered.wait();owner.cancel();await joining.wait();owner.cancel()
        await asyncio.sleep(0)
        assert events==["setup"]
        release.set()
        with pytest.raises(BaseExceptionGroup) as caught:await owner
        assert len(leaves(caught.value))==2
        assert all(isinstance(error,asyncio.CancelledError) for error in leaves(caught.value))
        assert asyncio.all_tasks()=={asyncio.current_task()}
    asyncio.run(scenario())
    assert events[-4:]==["pump joined","application joined","close","shutdown"]


def test_unrequested_application_cancellation_is_not_reported_as_success():
    events=[]
    async def cancelled():raise asyncio.CancelledError("independent cancellation")
    async def pump():await asyncio.Future()
    with pytest.raises(asyncio.CancelledError):
        run_standalone(cancelled,pump,**callbacks(events))
    assert events[-2:]==["close","shutdown"]


@pytest.mark.parametrize("first", ["application", "pump"])
def test_already_cancelling_task_retains_original_cancel_and_finishes_finally(first):
    events=[]
    async def scenario():
        joining=asyncio.Event()
        async def application():
            asyncio.current_task().cancel("independent")
            try:await asyncio.sleep(0)
            finally:
                joining.set()
                for _ in range(5):await asyncio.sleep(0)
                events.append("application finally finished")
        async def pump():await joining.wait()
        with pytest.raises(asyncio.CancelledError,match="independent"):
            await run_owned(application if first=="application" else pump,
                            pump if first=="application" else application,**callbacks(events))
    asyncio.run(scenario())
    assert events==["setup","application finally finished","close","shutdown"]


@pytest.mark.parametrize("target", ["application", "pump"])
@pytest.mark.parametrize("cancel_owner", [False, True])
def test_internal_timeout_during_shutdown_does_not_suppress_owned_cancellation(target, cancel_owner):
    events=[]
    async def scenario():
        armed=asyncio.Event();release=asyncio.Event()
        async def finishes():
            await armed.wait()
        async def pacing():
            armed.set()
            # Complete the other task, then expire a real timeout in the same
            # loop turn that wakes run_owned. This reproduces frame pacing's
            # transient Task.cancelling() without replacing asyncio internals.
            await asyncio.sleep(0)
            try:
                async with asyncio.timeout(0):
                    await asyncio.Future()
            except TimeoutError:
                events.append("timeout handled")
                if cancel_owner:owner.cancel("during timeout settlement")
            try:
                await release.wait()
            finally:
                events.append("pacing joined")
        owner=asyncio.create_task(run_owned(
            pacing if target=="application" else finishes,
            finishes if target=="application" else pacing,**callbacks(events)))
        try:
            done,_=await asyncio.wait({owner},timeout=1)
        finally:
            # The regression must not itself hang when run against old code.
            release.set()
            results=await asyncio.gather(owner,return_exceptions=True)
        assert done, "Internal timeout caused the owner to skip its stop request"
        if cancel_owner:
            assert isinstance(results[0],asyncio.CancelledError)
        else:
            assert results==[None]
        assert asyncio.all_tasks()=={asyncio.current_task()}
    asyncio.run(scenario())
    assert events==["setup","timeout handled","pacing joined","close","shutdown"]


def test_nested_blocking_runner_rejects_before_constructing_coroutines():
    async def scenario():
        def forbidden():pytest.fail("constructed a coroutine")
        with pytest.raises(RuntimeError,match="await run_owned"):
            run_standalone(forbidden,forbidden,**callbacks([]))
    asyncio.run(scenario())
