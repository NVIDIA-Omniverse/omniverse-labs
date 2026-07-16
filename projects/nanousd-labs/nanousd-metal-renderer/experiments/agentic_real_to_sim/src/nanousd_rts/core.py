"""Typed scene contract and immutable workspace storage."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
NODE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
ROLES = {"background", "static", "movable", "articulated"}
JOINT_KINDS = {"prismatic", "revolute"}
SELECTION_MODES = {"stable-reference", "bounds"}
AXES = {"X": 0, "Y": 1, "Z": 2}


class RealToSimError(RuntimeError):
    """Expected, user-actionable failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_digest(value: Any) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value).encode()).hexdigest()}"


def _vec(value: Iterable[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise RealToSimError(f"{name} must contain {length} finite numbers")
    return result


@dataclass(frozen=True, slots=True)
class Bounds:
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _vec(self.minimum, 3, "bounds.minimum")
        maximum = _vec(self.maximum, 3, "bounds.maximum")
        if any(lo >= hi for lo, hi in zip(minimum, maximum, strict=True)):
            raise RealToSimError("every bounds minimum must be smaller than its maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Bounds":
        return cls(tuple(value["min"]), tuple(value["max"]))

    @classmethod
    def from_center_size(
        cls,
        center: Iterable[float],
        size: Iterable[float],
    ) -> "Bounds":
        center_array = np.asarray(_vec(center, 3, "center"), dtype=np.float64)
        size_array = np.asarray(_vec(size, 3, "size"), dtype=np.float64)
        if np.any(size_array <= 0):
            raise RealToSimError("collider size values must be positive")
        half = size_array * 0.5
        return cls(tuple(center_array - half), tuple(center_array + half))

    @property
    def center(self) -> tuple[float, float, float]:
        return tuple((np.asarray(self.minimum) + np.asarray(self.maximum)) * 0.5)

    @property
    def size(self) -> tuple[float, float, float]:
        return tuple(np.asarray(self.maximum) - np.asarray(self.minimum))

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(np.asarray(self.size)))

    def translated(self, delta: Iterable[float]) -> "Bounds":
        offset = np.asarray(_vec(delta, 3, "translation"), dtype=np.float64)
        return Bounds(tuple(np.asarray(self.minimum) + offset), tuple(np.asarray(self.maximum) + offset))

    def overlap(self, other: "Bounds") -> tuple[float, float, float]:
        lower = np.maximum(self.minimum, other.minimum)
        upper = np.minimum(self.maximum, other.maximum)
        return tuple(np.maximum(upper - lower, 0.0))

    def overlap_volume(self, other: "Bounds") -> float:
        return float(np.prod(self.overlap(other)))

    def horizontal_overlap_fraction(self, other: "Bounds", up_axis: str) -> float:
        axes = [index for index in range(3) if index != AXES[up_axis]]
        overlap = np.asarray(self.overlap(other))[axes]
        base = np.minimum(np.asarray(self.size)[axes], np.asarray(other.size)[axes])
        if np.any(base <= 0):
            return 0.0
        return float(np.prod(overlap / base))

    def to_json(self) -> dict[str, list[float]]:
        return {"min": list(self.minimum), "max": list(self.maximum)}


@dataclass(frozen=True, slots=True)
class Collider:
    kind: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    provenance: str = "agent-authored"
    confidence: float = 0.5
    collision_mode: str = "solid"

    def __post_init__(self) -> None:
        if self.kind != "box":
            raise RealToSimError("the local oracle currently supports explicit box colliders only")
        center = _vec(self.center, 3, "collider.center")
        size = _vec(self.size, 3, "collider.size")
        if any(value <= 0 for value in size):
            raise RealToSimError("collider size values must be positive")
        rotation = np.asarray(_vec(self.rotation_wxyz, 4, "collider.rotation_wxyz"))
        norm = float(np.linalg.norm(rotation))
        if norm <= 1e-12:
            raise RealToSimError("collider quaternion must be nonzero")
        rotation /= norm
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise RealToSimError("collider confidence must be in [0, 1]")
        if self.collision_mode not in {"solid", "support", "shell"}:
            raise RealToSimError("collision_mode must be solid, support, or shell")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "rotation_wxyz", tuple(float(item) for item in rotation))

    @property
    def bounds(self) -> Bounds:
        # Verification uses the conservative world AABB. USDA preserves rotation.
        w, x, y, z = self.rotation_wxyz
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        half = np.abs(rotation) @ (np.asarray(self.size) * 0.5)
        center = np.asarray(self.center)
        return Bounds(tuple(center - half), tuple(center + half))

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Collider":
        return cls(
            kind=value["kind"],
            center=tuple(value["center"]),
            size=tuple(value["size"]),
            rotation_wxyz=tuple(value.get("rotation_wxyz", (1, 0, 0, 0))),
            provenance=value.get("provenance", "agent-authored"),
            confidence=float(value.get("confidence", 0.5)),
            collision_mode=value.get("collision_mode", "solid"),
        )


@dataclass(frozen=True, slots=True)
class Joint:
    kind: str
    parent: str
    axis: str
    axis_sign: int
    origin: tuple[float, float, float]
    lower: float
    upper: float
    confidence: float
    provenance: str
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in JOINT_KINDS:
            raise RealToSimError(f"joint kind must be one of {sorted(JOINT_KINDS)}")
        if self.axis not in AXES:
            raise RealToSimError("joint axis must be X, Y, or Z")
        if self.axis_sign not in {-1, 1}:
            raise RealToSimError("joint axis_sign must be -1 or 1")
        if not NODE_ID.fullmatch(self.parent):
            raise RealToSimError("joint parent is not a valid node id")
        origin = _vec(self.origin, 3, "joint.origin")
        if not math.isfinite(self.lower) or not math.isfinite(self.upper) or self.lower >= self.upper:
            raise RealToSimError("joint lower must be finite and smaller than upper")
        if not 0.0 <= self.confidence <= 1.0:
            raise RealToSimError("joint confidence must be in [0, 1]")
        object.__setattr__(self, "origin", origin)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Joint":
        return cls(
            kind=value["kind"],
            parent=value["parent"],
            axis=value["axis"],
            axis_sign=int(value.get("axis_sign", 1)),
            origin=tuple(value["origin"]),
            lower=float(value["lower"]),
            upper=float(value["upper"]),
            confidence=float(value["confidence"]),
            provenance=value["provenance"],
            diagnostics=tuple(value.get("diagnostics", ())),
        )


@dataclass(frozen=True, slots=True)
class SceneNode:
    node_id: str
    label: str
    role: str
    visual_bounds: Bounds
    selection_file: str
    selected_gaussians: int
    collider: Collider | None = None
    support_parent: str | None = None
    joint: Joint | None = None
    tags: tuple[str, ...] = ()
    selection_mode: str = "stable-reference"
    selection_bounds: Bounds | None = None

    def __post_init__(self) -> None:
        if not NODE_ID.fullmatch(self.node_id):
            raise RealToSimError(f"invalid node id: {self.node_id}")
        if self.role not in ROLES:
            raise RealToSimError(f"node role must be one of {sorted(ROLES)}")
        if self.selected_gaussians < 0:
            raise RealToSimError("selected_gaussians cannot be negative")
        if self.support_parent is not None and not NODE_ID.fullmatch(self.support_parent):
            raise RealToSimError("support parent is not a valid node id")
        if self.selection_mode not in SELECTION_MODES:
            raise RealToSimError(
                f"selection_mode must be one of {sorted(SELECTION_MODES)}"
            )
        if self.selection_mode == "bounds" and self.selection_bounds is None:
            raise RealToSimError("bounds selection mode requires selection_bounds")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "SceneNode":
        return cls(
            node_id=value["id"],
            label=value.get("label", value["id"]),
            role=value["role"],
            visual_bounds=Bounds.from_json(value["visual_bounds"]),
            selection_file=value["selection_file"],
            selected_gaussians=int(value["selected_gaussians"]),
            collider=Collider.from_json(value["collider"]) if value.get("collider") else None,
            support_parent=value.get("support_parent"),
            joint=Joint.from_json(value["joint"]) if value.get("joint") else None,
            tags=tuple(value.get("tags", ())),
            selection_mode=value.get("selection_mode", "stable-reference"),
            selection_bounds=(
                Bounds.from_json(value["selection_bounds"])
                if value.get("selection_bounds")
                else None
            ),
        )

    def to_json(self) -> dict[str, Any]:
        value = {
            "id": self.node_id,
            "label": self.label,
            "role": self.role,
            "visual_bounds": self.visual_bounds.to_json(),
            "selection_file": self.selection_file,
            "selected_gaussians": self.selected_gaussians,
            "support_parent": self.support_parent,
            "tags": list(self.tags),
            "selection_mode": self.selection_mode,
        }
        if self.selection_bounds:
            value["selection_bounds"] = self.selection_bounds.to_json()
        if self.collider:
            value["collider"] = asdict(self.collider)
        if self.joint:
            value["joint"] = asdict(self.joint)
        return value


@dataclass(slots=True)
class Workspace:
    root: Path
    state: dict[str, Any] = field(repr=False)

    @classmethod
    def open(cls, root: Path) -> "Workspace":
        root = Path(root).resolve()
        state_path = root / "scene.json"
        if not state_path.is_file():
            raise RealToSimError(f"workspace scene is missing: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("schema_version") != SCHEMA_VERSION:
            raise RealToSimError("unsupported real-to-sim workspace schema")
        workspace = cls(root, state)
        workspace.verify_source()
        return workspace

    @classmethod
    def create(
        cls,
        root: Path,
        source_ply: Path,
        source_report: dict[str, Any],
        *,
        source_provenance: dict[str, Any],
        up_axis: str = "Y",
        meters_per_unit: float = 1.0,
        replace: bool = False,
    ) -> "Workspace":
        root = Path(root).resolve()
        if root.exists() and replace:
            shutil.rmtree(root)
        if root.exists() and any(root.iterdir()) and not replace:
            raise RealToSimError(f"workspace is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        for name in ("source", "selections", "evidence", "exports", "trace"):
            (root / name).mkdir(exist_ok=True)
        canonical_source = root / "source" / "source.ply"
        shutil.copy2(Path(source_ply).resolve(), canonical_source)
        source_digest = sha256_file(canonical_source)
        state = {
            "schema_version": SCHEMA_VERSION,
            "created_unix": time.time(),
            "up_axis": up_axis,
            "meters_per_unit": float(meters_per_unit),
            "source": {
                "path": "source/source.ply",
                "sha256": source_digest,
                "bytes": canonical_source.stat().st_size,
                "report": source_report,
                "provenance": source_provenance,
            },
            "nodes": [],
            "support_edges": [],
            "completion_candidates": [],
            "scene_revision": 0,
        }
        workspace = cls(root, state)
        workspace.save()
        workspace.trace("ingest", {"source": source_provenance}, {"source_sha256": source_digest})
        return workspace

    @property
    def source_path(self) -> Path:
        return self.root / self.state["source"]["path"]

    @property
    def up_axis(self) -> str:
        return self.state["up_axis"]

    @property
    def nodes(self) -> list[SceneNode]:
        return [SceneNode.from_json(item) for item in self.state.get("nodes", [])]

    def node(self, node_id: str) -> SceneNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise RealToSimError(f"unknown scene node: {node_id}")

    def put_node(self, node: SceneNode) -> None:
        values = [item for item in self.state.get("nodes", []) if item["id"] != node.node_id]
        values.append(node.to_json())
        self.state["nodes"] = sorted(values, key=lambda item: item["id"])
        self.state["scene_revision"] = int(self.state.get("scene_revision", 0)) + 1
        self._rebuild_support_edges()
        self.save()

    def remove_node(self, node_id: str) -> None:
        before = len(self.state.get("nodes", []))
        self.state["nodes"] = [item for item in self.state.get("nodes", []) if item["id"] != node_id]
        if len(self.state["nodes"]) == before:
            raise RealToSimError(f"unknown scene node: {node_id}")
        self.state["scene_revision"] = int(self.state.get("scene_revision", 0)) + 1
        self._rebuild_support_edges()
        self.save()

    @property
    def completions(self) -> list[dict[str, Any]]:
        return list(self.state.get("completion_candidates", []))

    def put_completion(self, record: dict[str, Any]) -> None:
        completion_id = record.get("id")
        if not isinstance(completion_id, str) or not NODE_ID.fullmatch(completion_id):
            raise RealToSimError(f"invalid completion candidate id: {completion_id}")
        values = [
            item
            for item in self.state.get("completion_candidates", [])
            if item.get("id") != completion_id
        ]
        values.append(record)
        self.state["completion_candidates"] = sorted(values, key=lambda item: item["id"])
        self.state["scene_revision"] = int(self.state.get("scene_revision", 0)) + 1
        self.save()

    def remove_completions_for_nodes(self, node_ids: Iterable[str]) -> int:
        selected = set(node_ids)
        before = len(self.state.get("completion_candidates", []))
        self.state["completion_candidates"] = [
            item
            for item in self.state.get("completion_candidates", [])
            if item.get("node") not in selected
        ]
        removed = before - len(self.state["completion_candidates"])
        if removed:
            self.state["scene_revision"] = int(self.state.get("scene_revision", 0)) + 1
            self.save()
        return removed

    def _rebuild_support_edges(self) -> None:
        self.state["support_edges"] = sorted(
            [
                {"parent": node.support_parent, "child": node.node_id}
                for node in self.nodes
                if node.support_parent
            ],
            key=lambda item: (item["parent"], item["child"]),
        )

    def selection_path(self, node_id: str) -> Path:
        return self.root / "selections" / f"{node_id}.npy"

    def save_selection(self, node_id: str, source_indices: np.ndarray) -> str:
        indices = np.unique(np.asarray(source_indices, dtype=np.uint32))
        relative = Path("selections") / f"{node_id}.npy"
        np.save(self.root / relative, indices, allow_pickle=False)
        return relative.as_posix()

    def load_selection(self, node: SceneNode | str) -> np.ndarray:
        record = self.node(node) if isinstance(node, str) else node
        path = self.root / record.selection_file
        if not path.is_file():
            raise RealToSimError(f"node selection is missing: {path}")
        return np.load(path, allow_pickle=False)

    def verify_source(self) -> None:
        path = self.source_path
        if not path.is_file():
            raise RealToSimError(f"immutable source PLY is missing: {path}")
        expected = self.state["source"]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RealToSimError(f"immutable source checksum changed: expected {expected}, got {actual}")

    def save(self) -> None:
        self.state["logical_digest"] = content_digest(
            {key: value for key, value in self.state.items() if key not in {"logical_digest", "created_unix"}}
        )
        (self.root / "scene.json").write_text(
            json.dumps(self.state, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def trace(self, tool: str, inputs: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        trace_path = self.root / "trace" / "operations.jsonl"
        sequence = 0
        if trace_path.exists():
            with trace_path.open("r", encoding="utf-8") as stream:
                sequence = sum(1 for _ in stream)
        record = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "tool": tool,
            "inputs": inputs,
            "outputs": outputs,
            "scene_revision": self.state.get("scene_revision", 0),
            "scene_digest": self.state.get("logical_digest"),
            "unix_time": time.time(),
        }
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(canonical_json(record) + "\n")
        return record
