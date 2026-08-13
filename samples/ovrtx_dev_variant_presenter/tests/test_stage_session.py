# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StageSession ordinal / write-floor publish order (fakes, no GPU)."""
from __future__ import annotations

import sys
import types

import pytest

import dev_variant_presenter.render.stage_session as ss


class FakeOp:
    def wait(self):
        return self


class FakeStage:
    def __init__(self, name):
        self.name = name
        self.floor = 0
        self.writes = []
        self.destroyed = False

    def advance_write_floor(self, ordinal, *a, **k):
        self.floor = ordinal
        return FakeOp()

    def write_attribute(self, query, attribute, ordinal, tensors, **k):
        self.writes.append((query, attribute, ordinal, tensors, k))
        return FakeOp()

    def destroy(self):
        self.destroyed = True

    def get_path_dictionary(self):
        return FakePathDict()

    def query_from_path_list(self, path_list):
        return ("query", path_list)


class FakePathDict:
    def __init__(self, stage=None):
        self.stage = stage

    def create_path_list_from_strings(self, paths):
        return ("paths", tuple(paths))


class FakePopulation:
    def __init__(self):
        self.opens = []

    class PopulationDomain:
        RENDERING = 1
        PHYSICS = 2

    def open_usd(self, stage, path, ordinal=1, time_code=float("nan"), domains=1):
        self.opens.append(("file", path, ordinal, domains))
        return None

    def open_usd_from_string(self, stage, usda, ordinal=1, time_code=float("nan"), domains=1):
        self.opens.append(("str", usda, ordinal, domains))
        return None

    def update_from_usd_time(self, stage, ordinal, time_code):
        self.opens.append(("time", time_code, ordinal))
        return None

    def apply_usd_changes(self, stage, ordinal=1):
        self.opens.append(("apply", ordinal))
        return None


class FakeRenderer:
    def __init__(self):
        self.attached = None
        self.detached = False

    def attach_ovstage(self, stage):
        self.attached = stage

    def detach_ovstage(self):
        self.detached = True


@pytest.fixture
def patched(monkeypatch):
    fake_pop = FakePopulation()
    fake_mod = types.ModuleType("ovstage")
    fake_mod.Stage = FakeStage
    fake_mod.PathDictionary = FakePathDict
    fake_mod.Scope = types.SimpleNamespace(ALL=0)
    fake_mod.PrimMode = types.SimpleNamespace(UPSERT=0, INSERT=1)
    fake_mod.population = fake_pop

    pop_mod = types.ModuleType("ovstage.population")
    pop_mod.open_usd = fake_pop.open_usd
    pop_mod.open_usd_from_string = fake_pop.open_usd_from_string
    pop_mod.update_from_usd_time = fake_pop.update_from_usd_time
    pop_mod.apply_usd_changes = fake_pop.apply_usd_changes
    pop_mod.PopulationDomain = FakePopulation.PopulationDomain
    fake_mod.population = pop_mod

    monkeypatch.setitem(sys.modules, "ovstage", fake_mod)
    monkeypatch.setitem(sys.modules, "ovstage.population", pop_mod)
    return fake_pop


def test_populate_advances_floor(patched):
    renderer = FakeRenderer()
    session = ss.StageSession("t")
    session.create_and_attach(renderer)
    assert renderer.attached is session.stage

    ord1 = session.populate_usd("/tmp/x.usda")
    assert ord1 == 1
    assert session.committed_ordinal == 1
    assert session.stage.floor == 1
    assert patched.opens[-1][0] == "file"
    assert patched.opens[-1][2] == 1


def test_write_then_advance(patched):
    renderer = FakeRenderer()
    session = ss.StageSession("t")
    session.create_and_attach(renderer)
    session.populate_usd("/tmp/x.usda")

    ord2 = session.write_attribute("q", "omni:xform", [1.0], is_array=False)
    assert ord2 == 2
    assert session.committed_ordinal == 2
    assert len(session.stage.writes) == 1
    assert session.stage.writes[0][2] == 2


def test_query_from_paths_uses_path_dictionary_wrapper(patched):
    renderer = FakeRenderer()
    session = ss.StageSession("t")
    session.create_and_attach(renderer)
    q = session.query_from_paths(["/World/Cam"])
    assert q == ("query", ("paths", ("/World/Cam",)))


def test_detach_and_destroy(patched):
    renderer = FakeRenderer()
    session = ss.StageSession("t")
    session.create_and_attach(renderer)
    stage = session.stage
    session.detach_and_destroy(renderer)
    assert renderer.detached is True
    assert stage.destroyed is True
    assert session.stage is None
    assert session.committed_ordinal == 0
