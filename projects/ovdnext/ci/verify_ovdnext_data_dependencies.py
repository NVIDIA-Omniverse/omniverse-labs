#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the standalone OvdNext Data probe dependency boundary."""

import argparse
from pathlib import Path
import re


class DependencyError(Exception):
    pass


ALLOWED_TARGETS = {
    "ovdnext_data_probe",
    "OvdDataOmniPvd",
    "OvdData",
    "PVDRuntime",
    "Ws2_32",
    "dl",
    "rt",
    "pthread",
    "Threads::Threads",
}
REQUIRED_TARGETS = {"OvdData", "OvdDataOmniPvd", "PVDRuntime"}
FORBIDDEN_EXACT = {"PVDCore", "PVDNext"}
FORBIDDEN_FRAGMENTS = ("qt", "python", "ovui", "ovrtx", "pvddom", "opengl", "renderer")


def _unescape_label(value):
    return value.replace(r'\"', '"').replace(r"\\", "\\")


def _parse_graph(path):
    try:
        lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise DependencyError("unable to read dependency graph: %s" % error)
    nodes = {}
    edges = {}
    node_pattern = re.compile(r'^\s*"([^"]+)"\s*\[\s*label\s*=\s*"((?:\\.|[^"])*)"')
    edge_pattern = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"')
    for line in lines:
        node = node_pattern.search(line)
        if node:
            nodes[node.group(1)] = _unescape_label(node.group(2))
        edge = edge_pattern.search(line)
        if edge:
            edges.setdefault(edge.group(1), []).append(edge.group(2))
    if not nodes:
        raise DependencyError("dependency graph contains no quoted target labels")
    return nodes, edges


def _forbidden_fragment(value):
    lowered = value.lower()
    for fragment in FORBIDDEN_FRAGMENTS:
        if fragment in lowered:
            return fragment
    return None


def verify_dependency_boundary(graph_path, root_target, imports_path):
    nodes, edges = _parse_graph(graph_path)
    root_nodes = [node for node, label in nodes.items() if label == root_target]
    if len(root_nodes) != 1:
        raise DependencyError("dependency graph must contain exactly one root target: %s" % root_target)
    root_node = root_nodes[0]
    closure = set()
    paths = {root_node: [root_target]}
    pending = [root_node]
    while pending:
        node = pending.pop()
        if node in closure:
            continue
        closure.add(node)
        for dependency in edges.get(node, []):
            if dependency not in nodes:
                raise DependencyError("dependency graph edge references an unknown node: %s" % dependency)
            if dependency not in paths:
                paths[dependency] = paths[node] + [nodes[dependency]]
            pending.append(dependency)

    labels = {nodes[node] for node in closure}
    for node in closure:
        label = nodes[node]
        path = " -> ".join(paths[node])
        if label in FORBIDDEN_EXACT:
            raise DependencyError("forbidden dependency path: %s" % path)
        fragment = _forbidden_fragment(label)
        if fragment:
            raise DependencyError("forbidden dependency path: %s" % path)
        if label not in ALLOWED_TARGETS:
            raise DependencyError("unknown dependency path: %s" % path)
    missing = sorted(REQUIRED_TARGETS - labels)
    if missing:
        raise DependencyError("dependency graph is missing required target: %s" % missing[0])

    try:
        imports = Path(imports_path).read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise DependencyError("unable to read binary imports: %s" % error)
    for imported in imports:
        fragment = _forbidden_fragment(imported)
        if fragment:
            raise DependencyError("forbidden binary import: %s" % imported.strip())


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--root-target", required=True)
    parser.add_argument("--imports", required=True)
    return parser


def main(arguments=None):
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        verify_dependency_boundary(args.graph, args.root_target, args.imports)
    except DependencyError as error:
        parser.exit(1, "ERROR: %s\n" % error)


if __name__ == "__main__":
    main()
