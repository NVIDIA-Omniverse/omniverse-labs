#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative inventory for the repository's semantic validation suites."""

from __future__ import annotations

from pathlib import Path


SUITES = (
    "unit",
    "golden-small",
    "golden-large",
    "ov-integration",
    "blender-integration",
    "performance-small",
    "performance-large",
)

# These are selected, not marked, so product tests remain ignorant of orchestration.
BLENDER_TESTS = (
    "tests/test_color_management_panel.py::test_panel_draws_view_settings_controls_headless",
    "tests/test_dlss_toggle.py::test_dlss_property_registers_defaults_true_and_persists",
    "tests/test_interactive_edit_bridge_persistence.py::test_live_edit_bridge_survives_file_load",
    "tests/test_ldr_color_gating.py::test_panel_gating_each_resolved_mode_headless",
    "tests/test_rtpt_render_panel.py::test_panel_draws_quality_controls_headless",
    "tests/test_rtpt_scene_properties.py::test_properties_register_default_and_persist_through_blend",
    "tests/test_scene_generation_contract.py::test_real_blender_scene_generation_contract",
    "tests/test_stock_panel_compat.py::test_stock_panel_compat_register_unregister_headless",
    "tests/test_stock_panel_poll.py",
    "tests/test_user_messages.py::test_report_operator_and_pump_execute_headless",
)

OV_INTEGRATION_TESTS = (
    "tests/test_physics_viewport_sync.py",
    "tests/test_ovphysx_runtime_client.py",
    "tests/test_ovrtx_runtime_client.py",
    "tests/test_ovrtx_session.py",
    "tests/test_direct_runtime_session.py",
    "tests/test_shared_stage_composition.py",
)

GOLDENS = {
    "golden-small": (
        ("demo_stair_drop_1280x720", "demo_stair_drop_1280x720"),
        ("hero_cube", "hero_cube"),
    ),
    "golden-large": (
        ("perf_junk_shop_1280x720", "perf_junk_shop_1280x720"),
        ("perf_blender_classroom_1280x720", "perf_blender_classroom_1280x720"),
    ),
}

INTEGRATION_PROBES = ()  # Installed native qualification uses the root paired-package verifier.

PERFORMANCE = {
    "performance-small": {
        "existing-light-edit-responsiveness": (
            "run_blender_light_edit_responsiveness.py",
            None,
            None,
        ),
    },
    "performance-large": {
        "blender-navigation-ldr": (
            "run_blender_navigation.py",
            "ldr_rgba8_display_passthrough",
            "report_navigation.py",
        ),
        "blender-navigation-hdr": (
            "run_blender_navigation.py",
            "scene_linear_hdr",
            "report_navigation.py",
        ),
    },
}

# Standalone diagnostics deliberately excluded from product validation.
EXCLUDED_PROBES = {
    "run_ovrtx_live_transform_probe.py": "historical SIL native comparison; disconnected from supported OVStream qualification",
    "run_physics_stock_workflows.py": "separately invoked required Windows stock Blender physics authoring/Action/restore gate; no vendor SDK imports, requires --repo and --output",
    "run_physics_workflows.py": "separately invoked required Windows production physics workflow gate: fixed-step bake, presentation, kinematic sampling, offline Actions and viewport rewind; requires --repo and --output",
    "run_physics_continuity_recovery.py": "separately invoked required Windows renderer-recovery and physics-continuity native gate: admitted scene-owned physics identity, timestamp and body-state preservation; requires --repo and --output",
    "run_bounded_recovery.py": "separately invoked required Windows Blender recovery gate: owned producer faults, real deadlines, raw viewport pixels, cancellation and asynchronous lifecycle cleanup; requires --repo and --output",
    "run_scene_transaction_probe.py": "separately invoked required Windows scene transaction gate: locked native readback, immutable source and asset rollback, pixel equivalence and failed-rollback quarantine; requires --output and --usd-site-packages",
    "run_public_sdk_first_frame_smoke.py": "separately invoked Windows public SDK admission gate: authenticated worker, owned first-frame copy and ACK; requires --python and --output",
    "run_owned_frame_failure_smoke.py": "separately invoked Windows public SDK frame-failure gate: real renderer and owned SHM cleanup; requires --python and --output",
    "run_ovstream_shm_smoke.py": "opt-in public SDK GPU transport diagnostic",
    "run_ovstream_raw_frame_smoke.py": "opt-in public SDK bit-exact raw HDR SHM diagnostic",
    "run_public_sdk_recovery_smoke.py": "opt-in public SDK renderer crash/replay diagnostic",
    "run_public_sdk_async_smoke.py": "opt-in public SDK async render and cancellation diagnostic",
    "run_public_sdk_semantic_smoke.py": "opt-in public SDK lossless semantic SHM and label-map diagnostic",
    "run_ovstream_receiver_recovery_smoke.py": "opt-in public SDK receiver reconnect/backpressure diagnostic",
    "run_ovstage_scene_updates_smoke.py": "opt-in public SDK same-stage source replacement diagnostic",
    "run_physics_viewport_sync_probe.py": "driver used by the native physics viewport pytest test",
    "run_blender_renderer_switch.py": "standalone installed-package acceptance; requires the normal enabled Blender profile",
    "run_scene_generation_contract.py": "driver used by the Blender pytest test",
    "run_shared_stage_composition_probe.py": "historical SIL composition diagnostic; retired orchestration",
    "run_blender_orthographic_view_parity_probe.py": "manual viewport parity diagnostic",
    "run_ovphysx_drop_probe.py": "historical SIL physics diagnostic; retired orchestration",
    "run_ovrtx_color_presentation_probe.py": "historical SIL rendering diagnostic; retired orchestration",
    "run_ovrtx_light_value_probe.py": "historical SIL rendering diagnostic; retired orchestration",
    "run_ovrtx_material_value_probe.py": "historical SIL rendering diagnostic; retired orchestration",
    "run_ovrtx_operator_seam_probe.py": "historical SIL GUI diagnostic; retired orchestration",
    "run_ovrtx_orthographic_camera_probe.py": "historical SIL rendering diagnostic; retired orchestration",
    "run_ovrtx_primvars_st_probe.py": "historical SIL rendering diagnostic; retired orchestration",
    "run_ovrtx_world_dome_probe.py": "historical SIL rendering diagnostic; retired orchestration",
}

EXCLUDED_TESTS = {
    "tests/test_generated_presentation_defs.py::test_generated_presentation_composes_a_defined_traversable_product": "requires an OpenUSD Python environment",
    "tests/test_simready_physics_conversion.py::test_convert_scene_unibody_authors_usd_physics_surface": "requires an OpenUSD Python environment",
    "tests/test_session_lifecycle.py::test_pid_running_never_signals_on_windows": "Windows-only platform pin",
    "tests/test_worker_plugins_search_path.py::test_apply_worker_runtime_environment_prepends_plugins_to_path": "Windows-only platform pin",
}


def unit_tests(root: Path) -> tuple[str, ...]:
    integration = set(OV_INTEGRATION_TESTS)
    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted((root / "tests").glob("test_*.py"))
        if path.relative_to(root).as_posix() not in integration
    )
