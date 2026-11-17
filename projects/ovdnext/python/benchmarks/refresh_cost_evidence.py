# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in benchmark observations; never change production code or GC controls."""
import ast
import gc
import hashlib
import threading
from time import perf_counter


BEGIN = "_pvd_benchmark_graph_begin"
END = "_pvd_benchmark_graph_end"


def instrument_graph(original):
    """Insert two reversible calls, preserving every original source byte."""
    if BEGIN.encode() in original or END.encode() in original:
        raise ValueError("Graph observer names already occur in Application source")
    newline = b"\r\n" if b"\r\n" in original else b"\n"
    anchors = [b"            children=dict(native_children)",
               b"            query_ids=[record.query_id for record in self.adapter.query_records()]"]
    anchors = [a + newline for a in anchors]
    if any(original.count(anchor) != 1 for anchor in anchors):
        raise ValueError("Graph preparation source anchors are absent or duplicated")
    start, stop = [original.index(anchor) for anchor in anchors]
    if stop <= start:
        raise ValueError("Graph preparation anchors are out of order")
    tree = ast.parse(original)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Application")
    refresh = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_refresh")
    first_line, last_line = original[:start].count(b"\n") + 1, original[:stop].count(b"\n") + 1
    if not refresh.lineno < first_line < last_line <= refresh.end_lineno:
        raise ValueError("Graph anchors are outside Application._refresh")
    statements = [node for node in ast.walk(refresh) if isinstance(node, ast.stmt)
                  and first_line <= node.lineno < last_line and node.col_offset == 12]
    statements.sort(key=lambda node: node.lineno)
    if [type(node) for node in statements] != [ast.Assign, ast.Assign, ast.If]:
        raise ValueError("Graph preparation statement boundary changed; audit it first")
    if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await))
           for statement in statements for node in ast.walk(statement)):
        raise ValueError("Graph preparation can leave its observation span")
    result, edits = original, []
    for offset, name in ((stop, END), (start, BEGIN)):
        addition = b"            " + name.encode() + b"()" + newline
        result = result[:offset] + addition + result[offset:]
        edits.append(dict(offset=offset, addition_hex=addition.hex()))
    restored = result
    for edit in reversed(edits):
        offset, addition = edit["offset"], bytes.fromhex(edit["addition_hex"])
        if restored[offset:offset + len(addition)] != addition:
            raise AssertionError("Graph observer insertion cannot be reversed")
        restored = restored[:offset] + restored[offset + len(addition):]
    if restored != original:
        raise AssertionError("Graph observer changed original source bytes")

    class RemoveObserver(ast.NodeTransformer):
        count = 0
        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id in (BEGIN, END) and not call.args and not call.keywords):
                self.count += 1
                return None
            return self.generic_visit(node)
    remover = RemoveObserver()
    stripped = remover.visit(ast.parse(result))
    if remover.count != 2 or ast.dump(tree) != ast.dump(stripped):
        raise AssertionError("Graph observer changed the original Application AST")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    return result, dict(original_sha256=sha(original), generated_sha256=sha(result),
        original_span_sha256=sha(original[start:stop]), first_line=first_line,
        stop_line_exclusive=last_line, insertions=edits, byte_reversal_verified=True,
        ast_equivalence_verified=True, scope="Children/owners shallow copies and native geometry membership supplementation; excludes cached getter, query IDs and InteractionState validation")


def _overlap(intervals, start, stop):
    clipped = sorted((max(a, start), min(b, stop)) for a, b in intervals if b > start and a < stop)
    total, end = 0., start
    for a, b in clipped:
        total += max(0., b - max(a, end))
        end = max(end, b)
    return total * 1000


def gc_control_evidence(before):
    """Read-only metadata for both observer modes; never install a callback."""
    before = dict(enabled=before["enabled"], thresholds=list(before["thresholds"]), debug=before["debug"])
    after = GcObservation.controls()
    return dict(before=before, after=after, unchanged=before == after,
                scope="Outside-clock observations immediately before and after the application event loop; no collection or control mutation")


class GcObservation:
    def __init__(self, *, max_events=200000, clock=perf_counter):
        if type(max_events) is not int or max_events < 2:
            raise ValueError("GC observation needs a positive bounded event capacity")
        self.max_events, self.clock = max_events, clock
        self.origin = clock()
        self.events = []
        self.dropped_events = 0
        self.protocol_errors = 0
        self.running = False
        self.callback = self._event
        self.before = self.controls()
        self.after = None

    @staticmethod
    def controls():
        return dict(enabled=gc.isenabled(), thresholds=list(gc.get_threshold()), debug=gc.get_debug())

    def start(self):
        if self.running:
            raise RuntimeError("GC observer already installed")
        self.before = self.controls()
        gc.callbacks.append(self.callback)
        self.running = True

    def stop(self):
        if self.running:
            index = next((i for i, callback in enumerate(gc.callbacks) if callback is self.callback), None)
            if index is None:
                self.protocol_errors += 1
            else:
                del gc.callbacks[index]
            self.running = False
        self.after = self.controls()

    def _event(self, phase, info):
        # No exceptions may escape a CPython GC callback. Once full, keep an
        # explicit dropped count rather than allocating an unbounded side log.
        if len(self.events) >= self.max_events:
            self.dropped_events += 1
            return
        if phase not in ("start", "stop") or info.get("generation") not in (0, 1, 2):
            self.protocol_errors += 1
            return
        self.events.append((self.clock(), phase, info["generation"], threading.get_ident(),
                            info.get("collected", 0), info.get("uncollectable", 0)))

    def window(self, start, stop, phases=()):
        """Wall-clock intersections are correlations, not GC CPU attribution."""
        active, intervals, collections = {}, [], []
        errors = self.protocol_errors
        for at, phase, generation, thread, collected, uncollectable in tuple(self.events):
            key = (thread, generation)
            if phase == "start":
                if key in active:
                    errors += 1
                active[key] = at
            elif key not in active:
                errors += 1
            else:
                began = active.pop(key)
                if at < began:
                    errors += 1
                elif at > start and began < stop:
                    intervals.append((began, at))
                    collections.append(dict(start_ms=(began - self.origin) * 1000,
                        stop_ms=(at - self.origin) * 1000, generation=generation, thread_id=thread,
                        collected=collected, uncollectable=uncollectable,
                        overlap_ms=_overlap([(began, at)], start, stop)))
        unfinished = [dict(start_ms=(at - self.origin) * 1000, generation=key[1], thread_id=key[0])
                      for key, at in active.items() if at <= stop]
        names = sorted({name for name, _, _ in phases})
        phase_overlap = {}
        for name in names:
            parts = [(max(a, c, start), min(b, d, stop)) for a, b in intervals
                     for candidate, c, d in phases if candidate == name and min(b, d, stop) > max(a, c, start)]
            phase_overlap[name] = _overlap(parts, start, stop)
        return dict(complete=not (self.dropped_events or errors or unfinished or self.controls() != self.before),
            dropped_events=self.dropped_events, protocol_errors=errors,
            sample_start_ms=(start - self.origin) * 1000, sample_stop_ms=(stop - self.origin) * 1000,
            overlap_ms=_overlap(intervals, start, stop), overlap_by_phase_ms=phase_overlap,
            collections=collections, unfinished_collections=unfinished)

    def summary(self):
        after = self.after if self.after is not None else self.controls()
        trace = self.window(self.origin, self.clock())
        return dict(enabled=True, complete=trace["complete"] and after == self.before,
            controls_before=self.before, controls_after=after, max_events=self.max_events,
            recorded_events=len(self.events), dropped_events=self.dropped_events,
            protocol_errors=trace["protocol_errors"], unfinished_collections=trace["unfinished_collections"],
            normal_gc_unchanged=after == self.before,
            clock="time.perf_counter monotonic seconds; serialized offsets in milliseconds",
            events=[dict(at_ms=(at - self.origin) * 1000, phase=phase, generation=generation,
                         thread_id=thread, collected=collected, uncollectable=uncollectable)
                    for at, phase, generation, thread, collected, uncollectable in self.events])
