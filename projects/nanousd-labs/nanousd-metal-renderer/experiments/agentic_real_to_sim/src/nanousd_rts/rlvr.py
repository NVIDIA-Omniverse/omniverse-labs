"""Single-use multi-turn RLVR episode adapter for Tinker-style rollouts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .core import RealToSimError, Workspace
from .sim import verify
from .workflow import TOOL_CATALOG, execute_action


ARTIFACT_KINDS = {"preview", "usda", "usda_render", "voxel"}


@dataclass(frozen=True)
class EpisodeRequirements:
    """Hidden evaluator requirements for one scene-building task."""

    required_nodes: tuple[str, ...] = ()
    required_interactive_nodes: tuple[str, ...] = ()
    required_mesh_pbr_nodes: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ("usda", "usda_render", "preview")
    minimum_interactive_nodes: int = 1
    max_turns: int = 12

    def __post_init__(self) -> None:
        unknown = set(self.required_artifacts) - ARTIFACT_KINDS
        if unknown:
            raise RealToSimError(f"unknown RLVR artifact requirements: {sorted(unknown)}")
        if self.minimum_interactive_nodes < 0:
            raise RealToSimError("minimum_interactive_nodes cannot be negative")
        if self.max_turns <= 0:
            raise RealToSimError("max_turns must be positive")

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "EpisodeRequirements":
        return cls(
            required_nodes=tuple(value.get("required_nodes", ())),
            required_interactive_nodes=tuple(value.get("required_interactive_nodes", ())),
            required_mesh_pbr_nodes=tuple(value.get("required_mesh_pbr_nodes", ())),
            required_artifacts=tuple(
                value.get("required_artifacts", ("usda", "usda_render", "preview"))
            ),
            minimum_interactive_nodes=int(value.get("minimum_interactive_nodes", 1)),
            max_turns=int(value.get("max_turns", 12)),
        )


@dataclass(frozen=True)
class RewardSnapshot:
    """Trainer-only reward state; do not place hidden requirements in the prompt."""

    turn: int
    dense_score: float
    terminal_reward: float
    terminal_passed: bool
    components: dict[str, float]
    gates: dict[str, bool]
    failed_gates: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["failed_gates"] = list(self.failed_gates)
        return value


def _read_passed(path: Path, *keys: str) -> bool:
    if not path.is_file():
        return False
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        for key in keys:
            value = value[key]
        return bool(value)
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _artifact_gates(workspace: Workspace) -> dict[str, bool]:
    voxel_reports = list(
        (workspace.root / "exports" / "voxel").glob("*/collision-report.json")
    )
    return {
        "preview": (workspace.root / "preview" / "index.html").is_file(),
        "usda": _read_passed(
            workspace.root / "exports" / "scene.manifest.json",
            "validation",
            "passed",
        ),
        "usda_render": _read_passed(
            workspace.root / "evidence" / "usda" / "render.json",
            "passed",
        ),
        "voxel": bool(voxel_reports)
        and all(_read_passed(path, "passed") for path in voxel_reports),
    }


def _failed_verification(error: Exception) -> dict[str, Any]:
    gates = {
        "immutable_source": False,
        "scene_has_nodes": False,
        "support_graph_acyclic": False,
        "all_node_contracts": False,
        "all_joint_sweeps": False,
        "completion_assets_valid": False,
        "accepted_completions_linked": False,
        "visual_segmentation_review": False,
    }
    return {
        "passed": False,
        "gates": gates,
        "nodes": [],
        "joint_sweeps": [],
        "continuous_scores": {"mean_visual_collider_coverage": 0.0},
        "errors": [f"{type(error).__name__}: {error}"],
    }


class RealToSimEpisode:
    """One isolated, stateful scene-building episode.

    The adapter deliberately mirrors the cookbook's single-use environment
    lifecycle. A caller can expose ``TOOL_CATALOG`` as structured model tools,
    call :meth:`step` sequentially, attach ``dense_score`` to each turn for
    Kevin-style future credit, and use ``terminal_reward`` for fail-closed
    submission scoring.
    """

    def __init__(
        self,
        workspace: Workspace,
        requirements: EpisodeRequirements | None = None,
    ):
        self.workspace = workspace
        self.requirements = requirements or EpisodeRequirements()
        self.turn = 0
        self.done = False

    def _verification(self) -> dict[str, Any]:
        try:
            return verify(self.workspace, run_sweeps=True)
        except Exception as exc:
            return _failed_verification(exc)

    def reward_snapshot(self, *, final: bool = False) -> RewardSnapshot:
        report = self._verification()
        verification_gates = {
            str(name): bool(passed)
            for name, passed in report.get("gates", {}).items()
        }
        nodes = {node.node_id: node for node in self.workspace.nodes}
        sweep_pass = {
            item["node"]: bool(item["passed"])
            for item in report.get("joint_sweeps", [])
        }
        interactive = {
            node_id
            for node_id, node in nodes.items()
            if node.role in {"movable", "articulated"}
            and node.joint is not None
            and sweep_pass.get(node_id, False)
        }
        mesh_pbr_nodes = {
            item["node"]
            for item in self.workspace.completions
            if item.get("status") == "accepted"
            and item.get("representation", {}).get("type")
            == "mesh-bound-gaussian-pbr"
        }
        required_nodes_ok = all(
            node_id in nodes for node_id in self.requirements.required_nodes
        )
        required_interactive_ok = all(
            node_id in interactive
            for node_id in self.requirements.required_interactive_nodes
        )
        required_mesh_pbr_ok = all(
            node_id in mesh_pbr_nodes
            for node_id in self.requirements.required_mesh_pbr_nodes
        )
        minimum_interactivity_ok = (
            len(interactive) >= self.requirements.minimum_interactive_nodes
        )
        artifact_state = _artifact_gates(self.workspace)
        required_artifacts_ok = all(
            artifact_state[kind] for kind in self.requirements.required_artifacts
        )

        if self.requirements.required_nodes:
            semantic = sum(
                node_id in nodes for node_id in self.requirements.required_nodes
            ) / len(self.requirements.required_nodes)
        else:
            semantic = min(1.0, len(nodes) / 3.0)
        if self.requirements.required_interactive_nodes:
            interactivity = sum(
                node_id in interactive
                for node_id in self.requirements.required_interactive_nodes
            ) / len(self.requirements.required_interactive_nodes)
        else:
            denominator = max(1, self.requirements.minimum_interactive_nodes)
            interactivity = min(1.0, len(interactive) / denominator)
        if self.requirements.required_artifacts:
            artifacts = sum(
                artifact_state[kind]
                for kind in self.requirements.required_artifacts
            ) / len(self.requirements.required_artifacts)
        else:
            artifacts = 1.0
        if self.requirements.required_mesh_pbr_nodes:
            mesh_pbr_fidelity = sum(
                node_id in mesh_pbr_nodes
                for node_id in self.requirements.required_mesh_pbr_nodes
            ) / len(self.requirements.required_mesh_pbr_nodes)
        else:
            mesh_pbr_fidelity = 1.0
        gate_values = list(verification_gates.values())
        local_verification = (
            sum(gate_values) / len(gate_values) if gate_values else 0.0
        )
        fidelity = float(
            report.get("continuous_scores", {}).get(
                "mean_visual_collider_coverage",
                0.0,
            )
        )
        efficiency = max(
            0.0,
            1.0 - self.turn / self.requirements.max_turns,
        )
        components = {
            "source_provenance": float(
                verification_gates.get("immutable_source", False)
            ),
            "semantic_completeness": float(semantic),
            "interactivity": float(interactivity),
            "local_verification": float(local_verification),
            "artifact_completeness": float(artifacts),
            "visual_collider_fidelity": float(fidelity),
            "mesh_pbr_fidelity": float(mesh_pbr_fidelity),
            "tool_efficiency": float(efficiency),
        }
        raw_dense_score = (
            components["source_provenance"] * 0.13
            + components["semantic_completeness"] * 0.13
            + components["interactivity"] * 0.18
            + components["local_verification"] * 0.18
            + components["artifact_completeness"] * 0.13
            + components["visual_collider_fidelity"] * 0.08
            + components["mesh_pbr_fidelity"] * 0.12
            + components["tool_efficiency"] * 0.05
        )
        invariant_safety = all(
            verification_gates.get(name, False)
            for name in (
                "immutable_source",
                "support_graph_acyclic",
                "all_node_contracts",
                "completion_assets_valid",
                "accepted_completions_linked",
            )
        )
        dense_score = raw_dense_score if invariant_safety else 0.0
        gates = {
            **verification_gates,
            "invariant_safety": invariant_safety,
            "required_nodes": required_nodes_ok,
            "required_interactive_nodes": required_interactive_ok,
            "required_mesh_pbr_nodes": required_mesh_pbr_ok,
            "minimum_interactivity": minimum_interactivity_ok,
            "required_artifacts": required_artifacts_ok,
        }
        terminal_passed = invariant_safety and bool(report.get("passed")) and all(
            (
                required_nodes_ok,
                required_interactive_ok,
                required_mesh_pbr_ok,
                minimum_interactivity_ok,
                required_artifacts_ok,
            )
        )
        failed = tuple(sorted(name for name, passed in gates.items() if not passed))
        return RewardSnapshot(
            turn=self.turn,
            dense_score=float(dense_score),
            terminal_reward=float(dense_score if final and terminal_passed else 0.0),
            terminal_passed=terminal_passed,
            components=components,
            gates=gates,
            failed_gates=failed,
        )

    def public_observation(
        self,
        snapshot: RewardSnapshot | None = None,
    ) -> dict[str, Any]:
        snapshot = snapshot or self.reward_snapshot(final=False)
        artifacts = _artifact_gates(self.workspace)
        completion_types = {
            item["node"]: item.get("representation", {}).get("type")
            for item in self.workspace.completions
            if item.get("status") == "accepted"
        }
        return {
            "schema_version": 1,
            "turn": self.turn,
            "turns_remaining": max(
                0,
                self.requirements.max_turns - self.turn,
            ),
            "scene_digest": self.workspace.state["logical_digest"],
            "nodes": [
                {
                    "id": node.node_id,
                    "role": node.role,
                    "selected_gaussians": node.selected_gaussians,
                    "support_parent": node.support_parent,
                    "joint": node.joint.kind if node.joint else None,
                    "completion_representation": completion_types.get(node.node_id),
                }
                for node in self.workspace.nodes
            ],
            "failed_public_gates": list(snapshot.failed_gates),
            "artifacts": artifacts,
            "available_tools": dict(TOOL_CATALOG),
        }

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.done:
            raise RealToSimError("RLVR episode is already complete")
        self.turn += 1
        try:
            result = execute_action(self.workspace, action)
        except Exception as exc:
            self.done = True
            failure = {
                "schema_version": 1,
                "turn": self.turn,
                "status": "failed",
                "tool": action.get("tool"),
                "error": f"{type(exc).__name__}: {exc}",
                "trainer_reward": {
                    "turn": self.turn,
                    "dense_score": 0.0,
                    "terminal_reward": 0.0,
                    "terminal_passed": False,
                    "components": {},
                    "gates": {"tool_action_valid": False},
                    "failed_gates": ["tool_action_valid"],
                },
            }
            self.workspace.trace("rlvr-step", action, failure)
            return failure

        requested_submission = action.get("tool") == "verify"
        self.done = requested_submission or self.turn >= self.requirements.max_turns
        reward = self.reward_snapshot(final=self.done)
        response = {
            "schema_version": 1,
            "turn": self.turn,
            "status": "ok",
            "tool": action.get("tool"),
            "result": result,
            "done": self.done,
            "observation": self.public_observation(reward),
            "trainer_reward": reward.to_json(),
        }
        self.workspace.trace(
            "rlvr-step",
            action,
            {
                "done": self.done,
                "dense_score": reward.dense_score,
                "terminal_reward": reward.terminal_reward,
                "terminal_passed": reward.terminal_passed,
            },
        )
        return response
