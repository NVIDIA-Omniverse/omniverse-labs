from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nanousd_rts.collision import (
    SPLAT_TRANSFORM_GLB_TO_NANOUSD,
    _registration_for_transform,
)
from nanousd_rts.completion import (
    _union_bounds,
    accept_completion,
    completion_report,
    propose_hidden_interiors,
)
from nanousd_rts.core import Bounds, RealToSimError, Workspace
from nanousd_rts.gaussian import (
    GaussianScene,
    ingest,
    load_gaussians,
    make_drawer_fixture,
    write_mesh_bound_gaussians,
    write_surface_patch_gaussians,
)
from nanousd_rts.segmentation import PLANAR_REFINER, refine_planar_selection
from nanousd_rts.segmentation_review import (
    accept_segmentation_review,
    create_segmentation_review_plan,
    segmentation_review_status,
)
from nanousd_rts.mesh_completion import (
    EXTERNAL_MATERIAL_PROVIDER,
    LOCAL_MATERIAL_PROVIDER,
    fit_mesh_pbr_completion,
)
from nanousd_rts.preview import write_preview
from nanousd_rts.rlvr import EpisodeRequirements, RealToSimEpisode
from nanousd_rts.sim import add_node, fit_joint, set_support, sweep_joint, verify
from nanousd_rts.usda import compile_usda
from nanousd_rts.visual_articulation import _nearest_reference_labels
from nanousd_rts.visual_completion import author_visual_completion
from nanousd_rts.workflow import run_plan


class RealToSimCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="nanousd-rts-test-")
        self.root = Path(self.temporary.name)
        source = self.root / "fixture.ply"
        make_drawer_fixture(source)
        self.workspace = ingest(source, self.root / "workspace")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_ply_decode_is_finite_and_bounded(self) -> None:
        scene = load_gaussians(self.workspace.source_path)
        self.assertEqual(scene.count, 3750)
        self.assertEqual(scene.sh_degree, 0)
        self.assertTrue(scene.report()["finite"])
        self.assertGreater(scene.bounds.diagonal, 1.0)

    def test_drawer_joint_sweep_compile_and_verify(self) -> None:
        add_node(
            self.workspace,
            node_id="floor",
            label="floor",
            role="static",
            source_indices=np.arange(0, 900, dtype=np.uint32),
            collision_mode="support",
        )
        add_node(
            self.workspace,
            node_id="cabinet",
            label="cabinet",
            role="static",
            source_indices=np.arange(900, 2100, dtype=np.uint32),
            collision_mode="shell",
        )
        add_node(
            self.workspace,
            node_id="drawer",
            label="drawer",
            role="movable",
            source_indices=np.arange(2100, 2750, dtype=np.uint32),
            tags=("drawer",),
        )
        set_support(self.workspace, child_id="cabinet", parent_id="floor")
        set_support(self.workspace, child_id="drawer", parent_id="cabinet")
        joint = fit_joint(self.workspace, node_id="drawer")
        self.assertEqual(joint.kind, "prismatic")
        self.assertEqual(joint.axis, "Z")
        self.assertEqual(joint.axis_sign, -1)
        self.assertTrue(sweep_joint(self.workspace, node_id="drawer")["passed"])
        candidates = propose_hidden_interiors(self.workspace, node_id="drawer")
        accepted = accept_completion(self.workspace, completion_id=candidates[0]["id"])
        self.assertEqual(accepted["status"], "accepted")
        self.assertFalse(accepted["provenance"]["measured"])
        manifest = compile_usda(self.workspace)
        self.assertTrue(manifest["validation"]["passed"])
        text = Path(manifest["usda"]).read_text()
        self.assertIn('def PhysicsPrismaticJoint "drawer_joint"', text)
        self.assertIn("nanousdRts:gaussianSource", text)
        self.assertIn("nanousdRts:hiddenCompletion", text)
        portable = compile_usda(
            self.workspace,
            output=self.root / "portable-export" / "scene.usda",
        )
        portable_text = Path(portable["usda"]).read_text()
        self.assertIn("@../workspace/source/source.ply@", portable_text)
        self.assertIn("@../workspace/selections/drawer.npy@", portable_text)
        report = verify(self.workspace)
        self.assertTrue(report["passed"])
        preview = write_preview(self.workspace)
        self.assertTrue(preview.is_file())
        preview_text = preview.read_text()
        self.assertIn("drawer", preview_text)
        self.assertIn('id="canvasFrame"', preview_text)
        self.assertIn("observe(canvasFrame)", preview_text)
        self.assertNotIn("observe(canvas);", preview_text)
        episode = RealToSimEpisode(
            self.workspace,
            EpisodeRequirements(
                required_nodes=("floor", "cabinet", "drawer"),
                required_interactive_nodes=("drawer",),
                required_artifacts=("usda", "preview"),
            ),
        )
        reward = episode.reward_snapshot(final=True)
        self.assertTrue(reward.terminal_passed)
        self.assertGreater(reward.terminal_reward, 0.8)

    def test_plan_fails_closed_on_unknown_tool(self) -> None:
        plan = self.root / "bad-plan.json"
        plan.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "actions": [{"tool": "shell", "command": "echo not-allowed"}],
                }
            )
        )
        with self.assertRaises(RealToSimError):
            run_plan(self.workspace, plan)
        result = json.loads((self.workspace.root / "trace" / "plan-result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failed_action"], 0)

    def test_rlvr_episode_fails_closed_on_unknown_tool(self) -> None:
        episode = RealToSimEpisode(
            self.workspace,
            EpisodeRequirements(required_artifacts=(), minimum_interactive_nodes=0),
        )
        result = episode.step({"tool": "shell", "command": "echo not-allowed"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["trainer_reward"]["terminal_reward"], 0.0)
        self.assertTrue(episode.done)

    def test_bounds_reject_degenerate_box(self) -> None:
        with self.assertRaises(RealToSimError):
            Bounds((0, 0, 0), (0, 1, 1))

    def test_hidden_completion_union_cannot_shrink_measured_collider(self) -> None:
        measured = Bounds((-2.0, -1.0, -0.5), (2.0, 1.0, 0.5))
        generated = Bounds((-0.25, -0.8, -0.4), (0.25, 0.8, 0.4))
        combined = _union_bounds(measured, generated)
        self.assertEqual(combined, measured)

    def test_collision_registration_restores_asymmetric_ply_coordinates(self) -> None:
        source = Bounds((-7.0, -3.0, 2.0), (1.0, 5.0, 11.0))
        derived = Bounds((-1.0, -5.0, 2.0), (7.0, 3.0, 11.0))
        registration = _registration_for_transform(
            source,
            derived,
            SPLAT_TRANSFORM_GLB_TO_NANOUSD,
        )
        self.assertAlmostEqual(registration["normalized_residual"], 0.0)
        self.assertEqual(
            registration["matrix"],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
        )

    def test_nearest_reference_labels_transfer_stable_object_selection(self) -> None:
        references = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.asarray([True, False, True])
        queries = np.asarray(
            [
                [0.1, 0.0, 0.0],
                [0.9, 0.0, 0.0],
                [9.5, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            _nearest_reference_labels(queries, references, labels, batch_size=2),
            np.asarray([True, False, True]),
        )

    def test_planar_refinement_keeps_positive_and_negative_references(self) -> None:
        rng = np.random.default_rng(7)
        front_yz = rng.uniform(-0.8, 0.8, size=(180, 2))
        front = np.column_stack(
            (rng.normal(0.10, 0.0015, size=180), front_yz)
        )
        carcass_yz = rng.uniform(-0.8, 0.8, size=(90, 2))
        carcass = np.column_stack(
            (rng.normal(-0.08, 0.004, size=90), carcass_yz)
        )
        floaters_yz = rng.uniform(-0.8, 0.8, size=(12, 2))
        floaters = np.column_stack(
            (rng.normal(0.22, 0.002, size=12), floaters_yz)
        )
        positions = np.asarray(
            np.concatenate((front, carcass, floaters), axis=0),
            dtype=np.float32,
        )
        count = len(positions)
        scene = GaussianScene(
            source_path=self.root / "planar-fixture.ply",
            source_sha256="sha256:fixture",
            positions=positions,
            scales=np.tile(np.asarray([[0.002, 0.018, 0.021]], dtype=np.float32), (count, 1)),
            orientations=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (count, 1)),
            opacities=np.ones(count, dtype=np.float32),
            sh_coefficients=np.zeros((count, 3), dtype=np.float32),
            sh_degree=0,
        )
        refined = refine_planar_selection(
            scene,
            Bounds((-0.2, -1.0, -1.0), (0.25, 1.0, 1.0)),
            front_axis="X",
            outward_sign=1,
            kind="cabinet-door",
        )
        self.assertEqual(refined.diagnostics["refiner"], PLANAR_REFINER)
        self.assertGreater(refined.diagnostics["positive_references"], 150)
        self.assertGreater(refined.diagnostics["negative_references"], 80)
        self.assertAlmostEqual(refined.diagnostics["front_plane"], 0.10, delta=0.015)
        self.assertGreater(refined.diagnostics["front_alignment_median"], 0.95)

    def test_segmentation_review_is_pose_complete_and_digest_bound(self) -> None:
        add_node(
            self.workspace,
            node_id="review_parent",
            label="review parent",
            role="static",
            source_indices=np.arange(900, 2100, dtype=np.uint32),
            collision_mode="shell",
        )
        add_node(
            self.workspace,
            node_id="review_door",
            label="review door",
            role="movable",
            source_indices=np.arange(2100, 2750, dtype=np.uint32),
            tags=("visual-refined", PLANAR_REFINER, "cabinet-door"),
        )
        set_support(self.workspace, child_id="review_door", parent_id="review_parent")
        fit_joint(
            self.workspace,
            node_id="review_door",
            parent_id="review_parent",
            kind="revolute",
            axis="Y",
            axis_sign=1,
            origin=(0.0, 0.0, 0.0),
            lower=0.0,
            upper=90.0,
        )
        plan = create_segmentation_review_plan(self.workspace)
        self.assertEqual([part["id"] for part in plan["parts"]], ["review_door"])
        gradient = np.tile(
            np.linspace(25, 225, 800, dtype=np.uint8),
            (500, 1),
        )
        for index, pose in enumerate(("closed", "half", "open")):
            rgb = np.repeat(gradient[:, :, None], 3, axis=2)
            rgb[120:360, 80 + index * 120 : 200 + index * 120, :] = (35, 120, 210)
            path = (
                self.workspace.root
                / "evidence"
                / "segmentation"
                / "review"
                / "review_door"
                / f"{pose}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            from PIL import Image

            Image.fromarray(rgb).save(path)
        accepted = accept_segmentation_review(
            self.workspace,
            reviewer="codex-browser-test",
            note="Door remains coherent across all three articulated poses.",
        )
        self.assertTrue(accepted["passed"])
        open_path = (
            self.workspace.root
            / "evidence"
            / "segmentation"
            / "review"
            / "review_door"
            / "open.png"
        )
        with open_path.open("ab") as stream:
            stream.write(b"changed")
        stale = segmentation_review_status(self.workspace)
        self.assertFalse(stale["passed"])
        self.assertEqual(stale["stale_or_invalid_nodes"], ["review_door"])

    def test_source_is_immutable(self) -> None:
        with self.workspace.source_path.open("ab") as stream:
            stream.write(b"x")
        with self.assertRaises(RealToSimError):
            Workspace.open(self.workspace.root)
        episode = RealToSimEpisode(
            self.workspace,
            EpisodeRequirements(required_artifacts=(), minimum_interactive_nodes=0),
        )
        reward = episode.reward_snapshot(final=True)
        self.assertEqual(reward.dense_score, 0.0)
        self.assertEqual(reward.terminal_reward, 0.0)

    def test_surface_patch_writer_preserves_requested_count(self) -> None:
        output = self.root / "generated-surfaces.ply"
        bounds = Bounds((0.0, 0.0, 0.0), (1.0, 0.8, 0.6))
        write_surface_patch_gaussians(
            output,
            [
                {
                    "bounds": bounds,
                    "axis": "X",
                    "side": -1,
                    "color": (0.8, 0.7, 0.6),
                },
                {
                    "bounds": bounds,
                    "axis": "Z",
                    "side": 1,
                    "color": (0.2, 0.3, 0.4),
                },
            ],
            count=240,
            seed=12,
        )
        scene = load_gaussians(output)
        self.assertEqual(scene.count, 240)
        self.assertTrue(scene.report()["finite"])

    def test_mesh_bound_writer_retains_face_barycentric_and_uv_associations(self) -> None:
        output = self.root / "mesh-bound.ply"
        sidecar = self.root / "mesh-bindings.npz"
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.2],
                [1.0, 1.0, 0.4],
                [0.0, 1.0, 0.2],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        uvs = np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ],
            dtype=np.float64,
        )
        texture = np.zeros((32, 32, 3), dtype=np.uint8)
        texture[..., 0] = np.arange(32, dtype=np.uint8)[None, :] * 8
        report = write_mesh_bound_gaussians(
            output,
            vertices,
            faces,
            face_colors=np.asarray([[0.5, 0.4, 0.3], [0.3, 0.4, 0.5]]),
            count=120,
            association_path=sidecar,
            face_uvs=uvs,
            base_color_texture=texture,
            face_groups=np.asarray([4, 9], dtype=np.uint32),
            seed=3,
        )
        self.assertEqual(report["gaussian_count"], 120)
        scene = load_gaussians(output)
        self.assertEqual(scene.count, 120)
        self.assertTrue(scene.report()["finite"])
        associations = np.load(sidecar, allow_pickle=False)
        self.assertEqual(associations["face_indices"].shape, (120,))
        self.assertEqual(associations["barycentric"].shape, (120, 3))
        np.testing.assert_allclose(
            associations["barycentric"].sum(axis=1),
            np.ones(120),
            atol=1e-6,
        )
        np.testing.assert_array_equal(associations["face_groups"], np.asarray([4, 9]))
        self.assertTrue(np.isfinite(associations["uv"]).all())
        quaternion = scene.orientations[0]
        w, x, y, z = quaternion
        renderer_rows = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        first_face = associations["face_indices"][0]
        np.testing.assert_allclose(
            renderer_rows,
            associations["face_frames"][first_face].T,
            atol=1e-6,
        )

    def test_amodal_drawer_completion_separates_static_and_moving_assets(self) -> None:
        add_node(
            self.workspace,
            node_id="cabinet",
            label="cabinet",
            role="static",
            source_indices=np.arange(900, 2100, dtype=np.uint32),
            collision_mode="shell",
        )
        add_node(
            self.workspace,
            node_id="drawer",
            label="drawer",
            role="movable",
            source_indices=np.arange(2100, 2750, dtype=np.uint32),
            tags=("drawer",),
        )
        set_support(self.workspace, child_id="drawer", parent_id="cabinet")
        fit_joint(self.workspace, node_id="drawer")
        completion = author_visual_completion(
            self.workspace,
            node_id="drawer",
            kind="drawer",
            front_axis="Z",
            outward_sign=-1,
            depth=0.45,
            up_sign=1,
            shelf_count=0,
            static_gaussians=400,
            moving_gaussians=400,
            background_occlusion_bounds=Bounds(
                (-0.45, 0.15, -0.6),
                (0.45, 0.85, -0.35),
            ),
        )
        self.assertEqual(completion["status"], "accepted")
        self.assertFalse(completion["provenance"]["measured"])
        self.assertEqual(
            {item["attachment"] for item in completion["assets"]},
            {"world", "joint"},
        )
        report = completion_report(self.workspace)
        self.assertTrue(report["all_assets_valid"])
        self.assertEqual(
            self.workspace.node("drawer").collider.provenance,
            f"accepted-completion:{completion['id']}",
        )
        self.assertEqual(
            completion["visual_profile"]["background_occlusion_bounds"],
            {
                "min": [-0.45, 0.15, -0.6],
                "max": [0.45, 0.85, -0.35],
            },
        )
        upgraded = fit_mesh_pbr_completion(
            self.workspace,
            node_id="drawer",
            material_provider=LOCAL_MATERIAL_PROVIDER,
            texture_size=128,
        )
        self.assertEqual(
            upgraded["representation"]["type"],
            "mesh-bound-gaussian-pbr",
        )
        self.assertTrue(upgraded["representation"]["mesh_face_association"])
        self.assertEqual(len(upgraded["mesh_assets"]), 2)
        for asset in upgraded["mesh_assets"]:
            self.assertTrue((self.workspace.root / asset["mesh"]["path"]).is_file())
            self.assertTrue(
                (self.workspace.root / asset["associations"]["path"]).is_file()
            )
            self.assertEqual(set(asset["pbr_maps"]), {
                "baseColor.png",
                "roughness.png",
                "metallic.png",
                "normal.png",
                "ao.png",
            })
        self.assertTrue(completion_report(self.workspace)["all_assets_valid"])
        roughness = (
            self.workspace.root
            / upgraded["mesh_assets"][0]["pbr_maps"]["roughness.png"]["path"]
        )
        roughness_bytes = roughness.read_bytes()
        roughness.unlink()
        self.assertFalse(completion_report(self.workspace)["all_assets_valid"])
        roughness.write_bytes(roughness_bytes)
        self.assertTrue(completion_report(self.workspace)["all_assets_valid"])
        mesh_usda = compile_usda(
            self.workspace,
            output=self.root / "mesh-pbr-export" / "scene.usda",
        )
        mesh_usda_text = Path(mesh_usda["usda"]).read_text()
        self.assertIn("nanousdRts:staticCavityMesh", mesh_usda_text)
        self.assertIn("nanousdRts:movingInteriorBaseColor", mesh_usda_text)
        self.assertIn("nanousdRts:movingInteriorAssociations", mesh_usda_text)
        self.assertEqual(
            mesh_usda["mesh_pbr_completions"],
            [completion["id"]],
        )
        external_bundle = (
            self.workspace.root
            / "generated"
            / "mesh-pbr-completions"
            / "drawer"
        )
        external = fit_mesh_pbr_completion(
            self.workspace,
            node_id="drawer",
            material_provider=EXTERNAL_MATERIAL_PROVIDER,
            external_material_bundle=external_bundle,
            texture_size=128,
        )
        self.assertEqual(
            external["representation"]["material_provider"],
            EXTERNAL_MATERIAL_PROVIDER,
        )
        self.assertTrue(completion_report(self.workspace)["all_assets_valid"])
        fidelity_episode = RealToSimEpisode(
            self.workspace,
            EpisodeRequirements(
                required_nodes=("drawer",),
                required_interactive_nodes=("drawer",),
                required_mesh_pbr_nodes=("drawer",),
                required_artifacts=(),
            ),
        )
        fidelity_reward = fidelity_episode.reward_snapshot(final=True)
        self.assertTrue(fidelity_reward.gates["required_mesh_pbr_nodes"])
        self.assertEqual(fidelity_reward.components["mesh_pbr_fidelity"], 1.0)
        self.assertTrue(fidelity_reward.terminal_passed)


if __name__ == "__main__":
    unittest.main()
