# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""App-owned ovstage lifecycle for Dev Variant Presenter.

Owns one `ovstage.Stage`, publishes mutations at monotonically increasing ordinals,
advances the write floor, and attaches/detaches the ovrtx renderer. Render-thread
only — never call from HTTP handlers.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def _wait(op: Any) -> None:
    """Wait on an ovstage Operation if one was returned; population helpers return None."""
    if op is None:
        return
    wait = getattr(op, "wait", None)
    if callable(wait):
        wait()


class StageSession:
    """Thin coordinator around ovstage population + data-plane + write floor."""

    def __init__(self, name: str = "dev_variant_presenter"):
        self._name = name
        self._stage: Any = None
        self._ordinal = 0          # last allocated ordinal (0 = none yet)
        self._committed = 0        # last ordinal published via advance_write_floor
        self._attached = False

    @property
    def stage(self) -> Any:
        return self._stage

    @property
    def committed_ordinal(self) -> int:
        return self._committed

    def create_and_attach(self, renderer) -> None:
        import ovstage
        if self._stage is not None:
            raise RuntimeError("StageSession already has a stage")
        self._stage = ovstage.Stage(self._name)
        renderer.attach_ovstage(self._stage)
        self._attached = True

    def next_ordinal(self) -> int:
        self._ordinal += 1
        return self._ordinal

    def populate_usd(self, path: str, *, domains=None) -> int:
        import ovstage
        from ovstage import population
        if self._stage is None:
            raise RuntimeError("StageSession has no stage — call create_and_attach first")
        ordinal = self.next_ordinal()
        if domains is None:
            domains = population.PopulationDomain.RENDERING
        population.open_usd(self._stage, path, ordinal=ordinal, domains=domains)
        self.advance(ordinal)
        return ordinal

    def populate_usda(self, usda: str, *, domains=None) -> int:
        from ovstage import population
        if self._stage is None:
            raise RuntimeError("StageSession has no stage — call create_and_attach first")
        ordinal = self.next_ordinal()
        if domains is None:
            domains = population.PopulationDomain.RENDERING
        population.open_usd_from_string(self._stage, usda, ordinal=ordinal, domains=domains)
        self.advance(ordinal)
        return ordinal

    def apply_usd_changes(self) -> int:
        from ovstage import population
        if self._stage is None:
            raise RuntimeError("StageSession has no stage")
        ordinal = self.next_ordinal()
        population.apply_usd_changes(self._stage, ordinal=ordinal)
        self.advance(ordinal)
        return ordinal

    def update_from_usd_time(self, time_code: float) -> int:
        from ovstage import population
        if self._stage is None:
            raise RuntimeError("StageSession has no stage")
        ordinal = self.next_ordinal()
        population.update_from_usd_time(self._stage, ordinal, float(time_code))
        self.advance(ordinal)
        return ordinal

    def write_attribute(
        self,
        query,
        attribute: str | int,
        tensors,
        *,
        is_array: bool,
        ordinal: Optional[int] = None,
        prim_mode=None,
        semantic: int = 0,
        index_map: Optional[Sequence[int]] = None,
        mask: Optional[Sequence[int]] = None,
        count: Optional[int] = None,
        cuda_event: Optional[int] = None,
        cuda_stream: Optional[int] = None,
        advance: bool = True,
    ) -> int:
        import ovstage
        if self._stage is None:
            raise RuntimeError("StageSession has no stage")
        if ordinal is None:
            ordinal = self.next_ordinal()
        kwargs = {
            "is_array": is_array,
            "semantic": semantic,
            "index_map": index_map,
            "mask": mask,
            "count": count,
            "cuda_event": cuda_event,
            "cuda_stream": cuda_stream,
        }
        if prim_mode is None:
            prim_mode = ovstage.PrimMode.UPSERT
        kwargs["prim_mode"] = prim_mode
        op = self._stage.write_attribute(query, attribute, ordinal, tensors, **kwargs)
        _wait(op)
        if advance:
            self.advance(ordinal)
        return ordinal

    def write_omni_xform(self, prim_path: str, matrix_4x4, *, advance: bool = True) -> int:
        """Write `omni:xform` as one 16-lane float64 matrix element (ovstage recipe)."""
        import numpy as np
        from ovstage import (
            AttributeSemantic, DLDataType, DLDataTypeCode, make_dltensor,
        )
        m = np.asarray(matrix_4x4, dtype=np.float64).reshape(16)
        tensor = make_dltensor(
            m,
            dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=64, lanes=16),
            shape=[1],
        )
        query = self.query_from_paths([prim_path])
        try:
            return self.write_attribute(
                query,
                "omni:xform",
                tensor,
                is_array=False,
                semantic=int(AttributeSemantic.MATRIX),
                advance=advance,
            )
        finally:
            self._release_query(query)

    def write_scalar_attrs(
        self,
        prim_path: str,
        attrs: dict[str, float],
        *,
        advance: bool = True,
    ) -> int:
        """Write one or more float scalar attributes on a prim at a single ordinal."""
        import numpy as np
        if not attrs:
            raise ValueError("attrs must be non-empty")
        query = self.query_from_paths([prim_path])
        try:
            ordinal = self.next_ordinal()
            for name, value in attrs.items():
                tensor = np.array([float(value)], dtype=np.float32)
                self.write_attribute(
                    query,
                    name,
                    tensor,
                    is_array=False,
                    ordinal=ordinal,
                    advance=False,
                )
            if advance:
                self.advance(ordinal)
            return ordinal
        finally:
            self._release_query(query)

    def _release_query(self, query) -> None:
        if self._stage is None or query is None:
            return
        try:
            op = self._stage.release_query(query)
            _wait(op)
        except Exception:  # noqa: BLE001
            pass

    def advance(self, ordinal: int) -> None:
        import ovstage
        if self._stage is None:
            raise RuntimeError("StageSession has no stage")
        op = self._stage.advance_write_floor(ordinal, ovstage.Scope.ALL)
        _wait(op)
        self._committed = ordinal
        if ordinal > self._ordinal:
            self._ordinal = ordinal

    def query_from_paths(self, paths: Sequence[str]):
        """Build an ovstage query from prim path strings (caller must release)."""
        import ovstage
        if self._stage is None:
            raise RuntimeError("StageSession has no stage")
        # get_path_dictionary() returns a raw C bundle; PathDictionary(stage) is the
        # Python wrapper that owns create_path_list_from_strings / intern_path.
        pd = ovstage.PathDictionary(self._stage)
        path_list = pd.create_path_list_from_strings(list(paths))
        return self._stage.query_from_path_list(path_list)

    def detach_and_destroy(self, renderer) -> None:
        if self._attached:
            try:
                renderer.detach_ovstage()
            except Exception:  # noqa: BLE001
                pass
            self._attached = False
        if self._stage is not None:
            try:
                self._stage.destroy()
            except Exception:  # noqa: BLE001
                pass
            self._stage = None
        self._ordinal = 0
        self._committed = 0
