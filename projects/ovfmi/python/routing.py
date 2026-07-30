# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""USD-space input and output routing for the public ovfmi data plane."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .types import AttributeWrite, MissingInputPolicy, ReadGroup


def _component(values: list, offset: int, count: int):
    if count == 1:
        return values[offset]
    return values[offset : offset + count] if count else values


def _as_rows(values) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim == 0:
        rows = rows.reshape(1, 1)
    elif rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    else:
        rows = rows.reshape(rows.shape[0], -1)
    return rows


def _ovstage_column(ovstage, values):
    data = np.ascontiguousarray(values)
    if data.dtype.kind not in "fiu":
        data = data.astype(np.float64)
    leading = data.shape[0]
    lanes = 1 if data.ndim == 1 else int(np.prod(data.shape[1:]))
    codes = {
        "f": ovstage.DLDataTypeCode.kDLFloat,
        "i": ovstage.DLDataTypeCode.kDLInt,
        "u": ovstage.DLDataTypeCode.kDLUInt,
    }
    return ovstage.make_dltensor(
        data,
        dtype=ovstage.DLDataType(codes[data.dtype.kind], data.dtype.itemsize * 8, lanes),
        shape=[leading],
        ndim=1,
    )


def _read_stage_attribute(stage, prim_paths, attribute_name, ordinal):
    import ovstage

    if not prim_paths:
        return {}
    dictionary = ovstage.PathDictionary(stage)
    path_list = dictionary.create_path_list_from_strings(prim_paths)
    query = stage.query_from_path_list(path_list)
    rows = {}
    try:
        attribute = dictionary.intern_token(attribute_name)
        with stage.read_attributes(
            query,
            [attribute],
            ovstage.OrdinalRange.latest(ordinal),
        ) as read:
            read.wait()
            for group in read.groups():
                try:
                    if group.is_delete or group.is_array or group.tensor_count != 1:
                        continue
                    values = _as_rows(group.array(0))
                    paths = dictionary.get_path_strings(group.prim_list)
                    for local_index in range(group.prim_count):
                        prim_index = group.prim_index(local_index)
                        data_index = group.data_row_index(local_index)
                        if prim_index < len(paths) and data_index < len(values):
                            rows[paths[prim_index]] = values[data_index].tolist()
                finally:
                    stage.release_group(group)
        return rows
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(path_list)
        dictionary.destroy()


def write_stage_group(stage, group: ReadGroup, ordinal: int) -> None:
    import ovstage

    if not group.prim_paths or not group.tensors:
        return
    attribute_name = group.attribute_name
    values = group.tensors[0]
    semantic = group.semantic
    # ovrtx consumes the canonical local-matrix column.  Preserve USD-space
    # xformOp identity in read(), but translate the convenience stage write.
    if attribute_name == "xformOp:translate":
        translations = _as_rows(values)
        matrices = np.zeros((len(translations), 4, 4), dtype=np.float64)
        matrices[:, 0, 0] = 1.0
        matrices[:, 1, 1] = 1.0
        matrices[:, 2, 2] = 1.0
        matrices[:, 3, 3] = 1.0
        matrices[:, 3, : min(3, translations.shape[1])] = translations[:, :3]
        attribute_name = "omni:xform"
        values = matrices
        semantic = int(ovstage.AttributeSemantic.MATRIX)
    dictionary = ovstage.PathDictionary(stage)
    path_list = dictionary.create_path_list_from_strings(group.prim_paths)
    query = stage.query_from_path_list(path_list)
    try:
        stage.write_attribute(
            query,
            attribute_name,
            ordinal,
            _ovstage_column(ovstage, values),
            is_array=group.is_array,
            semantic=semantic,
        ).wait()
    finally:
        stage.release_query(query).wait()
        dictionary.destroy_path_list(path_list)
        dictionary.destroy()


class InputRouter:
    def __init__(self, initial_values: dict, policy: MissingInputPolicy):
        self._values = {
            prim: {attribute: list(values) for attribute, values in attributes.items()}
            for prim, attributes in initial_values.items()
        }
        self._policy = policy
        self._maps: dict[str, dict[str, tuple[str, str, int, int]]] = {}

    def cache_connections(self, instance) -> None:
        from ._parser import FmuDirection

        mapped = {}
        for connection in instance.get_parser_instance().connections:
            if not connection.enabled:
                continue
            for target in connection.targets:
                for mapping in connection.mappings:
                    if mapping.direction == FmuDirection.INPUT:
                        offset, count = mapping.usdMapping
                        mapped[mapping.fmiAttributeName] = (
                            target,
                            mapping.usdAttributeName,
                            offset,
                            count,
                        )
        self._maps[instance.get_parser_instance().path] = mapped

    def empty_cache_for(self, instance) -> None:
        self._maps.pop(instance.get_parser_instance().path, None)

    def write_start_values(self, instance) -> None:
        values = {}
        for name, (prim, attribute, offset, count) in self._maps.get(
            instance.get_parser_instance().path, {}
        ).items():
            row = self._values.get(prim, {}).get(attribute)
            if row is None:
                if self._policy == MissingInputPolicy.ERROR:
                    raise ValueError(f"missing mapped FMI input {prim}.{attribute}")
                if self._policy == MissingInputPolicy.ZERO:
                    values[name] = 0.0 if count in (0, 1) else [0.0] * count
                continue
            values[name] = _component(row, offset, count)
        instance.set_start_values(values)

    def get_inputs_for(self, instance) -> dict:
        inputs = {}
        for name, (prim, attribute, offset, count) in self._maps.get(
            instance.get_parser_instance().path, {}
        ).items():
            row = self._values.get(prim, {}).get(attribute)
            if row is not None:
                inputs[name] = _component(row, offset, count)
        return inputs

    def update_value(self, prim: str, attribute: str, values) -> None:
        self._values.setdefault(prim, {})[attribute] = [
            item.item() if hasattr(item, "item") else item
            for item in np.asarray(values).reshape(-1)
        ]

    def write(self, writes: list[AttributeWrite]) -> None:
        for write in writes:
            if write.is_array:
                raise NotImplementedError("ragged array FMI inputs are not implemented")
            rows = _as_rows(write.values)
            if len(rows) != len(write.prim_paths):
                raise ValueError(
                    f"{write.attribute_name}: expected {len(write.prim_paths)} rows, "
                    f"received {len(rows)}"
                )
            for index, prim in enumerate(write.prim_paths):
                self.update_value(prim, write.attribute_name, rows[index])

    def update_from_stage(self, stage, from_ordinal: int, to_ordinal: int) -> None:
        if from_ordinal > to_ordinal:
            raise ValueError("from_ordinal must be <= to_ordinal")
        by_attribute: OrderedDict[str, list[str]] = OrderedDict()
        for mappings in self._maps.values():
            for prim, attribute, _offset, _count in mappings.values():
                paths = by_attribute.setdefault(attribute, [])
                if prim not in paths:
                    paths.append(prim)
        for attribute, paths in by_attribute.items():
            for prim, values in _read_stage_attribute(
                stage, paths, attribute, to_ordinal
            ).items():
                self.update_value(prim, attribute, values)

    def mapped_prim_paths(self, attribute_name: str) -> set[str]:
        return {
            prim
            for mappings in self._maps.values()
            for prim, attribute, _offset, _count in mappings.values()
            if attribute == attribute_name
        }


class OutputRouter:
    def __init__(self, initial_values: dict, input_router: InputRouter, strict: bool):
        self._baseline = {
            prim: {attribute: list(values) for attribute, values in attributes.items()}
            for prim, attributes in initial_values.items()
        }
        self._input_router = input_router
        self._strict = strict
        self._maps: dict[str, list] = {}
        self._latest: OrderedDict[tuple[str, str], list] = OrderedDict()
        self._owners: dict[tuple[str, str, int], str] = {}

    def cache_connections(self, instance) -> None:
        from ._parser import FmuDirection

        mappings = []
        for connection in instance.get_parser_instance().connections:
            if not connection.enabled:
                continue
            for mapping in connection.mappings:
                if mapping.direction != FmuDirection.OUTPUT:
                    continue
                offset, count = mapping.usdMapping
                targets = []
                for target in connection.targets:
                    components = range(offset, offset + max(count, 1))
                    for component in components:
                        key = (target, mapping.usdAttributeName, component)
                        previous = self._owners.get(key)
                        if self._strict and previous is not None and previous != mapping.fmiAttributeName:
                            raise ValueError(
                                "overlapping output mapping for "
                                f"{target}.{mapping.usdAttributeName}[{component}]"
                            )
                        self._owners[key] = mapping.fmiAttributeName
                    targets.append((target, mapping.usdAttributeName, offset, count))
                mappings.append((mapping.fmiAttributeName, targets))
        self._maps[instance.get_parser_instance().path] = mappings

    def empty_cache_for(self, instance) -> None:
        self._maps.pop(instance.get_parser_instance().path, None)

    def get_outputs_for(self, instance) -> list[str]:
        return [name for name, _targets in self._maps.get(instance.get_parser_instance().path, [])]

    def write_outputs(self, instance, _outputs, result) -> None:
        names = getattr(result.dtype, "names", ()) or ()
        for name, targets in self._maps.get(instance.get_parser_instance().path, []):
            if name not in names:
                continue
            produced = np.asarray(result[name]).reshape(-1).tolist()
            for prim, attribute, offset, count in targets:
                key = (prim, attribute)
                if count == 0:
                    row = list(produced)
                else:
                    width = max(offset + count, len(self._baseline.get(prim, {}).get(attribute, [])))
                    row = list(
                        self._latest.get(
                            key,
                            self._baseline.get(prim, {}).get(attribute, [0.0] * width),
                        )
                    )
                    if len(row) < width:
                        row.extend([0.0] * (width - len(row)))
                    payload = produced if len(produced) > 1 else produced * count
                    row[offset : offset + count] = payload[:count]
                self._latest[key] = row
                self._input_router.update_value(prim, attribute, row)

    def snapshot(self, prim_paths=None, attribute_names=None) -> list[ReadGroup]:
        path_filter = set(prim_paths) if prim_paths is not None else None
        attribute_filter = set(attribute_names) if attribute_names is not None else None
        grouped: OrderedDict[str, list[tuple[str, list]]] = OrderedDict()
        for (prim, attribute), values in self._latest.items():
            if path_filter is not None and prim not in path_filter:
                continue
            if attribute_filter is not None and attribute not in attribute_filter:
                continue
            grouped.setdefault(attribute, []).append((prim, list(values)))

        groups = []
        for attribute, rows in grouped.items():
            widths = {len(values) for _prim, values in rows}
            if len(widths) != 1:
                tensors = tuple(np.asarray(values) for _prim, values in rows)
                is_array = True
            else:
                dtype = np.float64 if attribute == "omni:xform" else np.float32
                tensors = (np.asarray([values for _prim, values in rows], dtype=dtype),)
                is_array = False
            semantic = 0
            if attribute == "omni:xform":
                try:
                    import ovstage

                    semantic = int(ovstage.AttributeSemantic.MATRIX)
                except ImportError:
                    pass
            groups.append(
                ReadGroup(
                    prim_paths=tuple(prim for prim, _values in rows),
                    attribute_name=attribute,
                    tensors=tensors,
                    is_array=is_array,
                    semantic=semantic,
                )
            )
        return groups
