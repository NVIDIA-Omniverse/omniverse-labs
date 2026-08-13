# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from dev_variant_presenter.batch.engine import (
    DEFAULT_EXPLOSION_THRESHOLD, ExplosionError, count_permutations,
    estimate_seconds, expand_labeled, expand_matrix, guard_count,
)
from dev_variant_presenter.batch.jobs import BatchJob, MatrixMode
from dev_variant_presenter.models import QualitySpec, VariantChoice, VariantSetInfo

SETS = (
    VariantSetInfo("Carpaint", "/W/Looks", ("Noir", "Sakura", "Green"), "Noir"),
    VariantSetInfo("Doors", "/W/Body", ("Open", "Closed"), "Closed"),
    VariantSetInfo("Wheels", "/W/Wheels", ("A", "B", "C", "D"), "A"),
)
BASE = (
    VariantChoice("/W/Looks", "Carpaint", "Noir"),
    VariantChoice("/W/Body", "Doors", "Closed"),
    VariantChoice("/W/Wheels", "Wheels", "A"),
)


def _job(mode, included, curated=()):
    return BatchJob(mode=mode, base_selection=BASE, included=included,
                    cameras=["/Cam"], quality=QualitySpec(),
                    frame_mode="single", out_dir="C:/out", curated=curated)


# --- full cartesian ---
def test_full_cartesian_count_is_product():
    job = _job(MatrixMode.FULL_CARTESIAN, {"Carpaint": ("Noir", "Sakura"), "Wheels": ("A", "B", "C")})
    assert count_permutations(job, SETS) == 6  # 2 * 3


def test_full_cartesian_expands_to_pinned_plus_swept():
    job = _job(MatrixMode.FULL_CARTESIAN, {"Carpaint": ("Noir", "Sakura")})
    perms = expand_matrix(job, BASE, SETS)
    assert len(perms) == 2
    for sel in perms:
        by_set = {c.set_name: c.variant for c in sel}
        assert by_set["Doors"] == "Closed"
        assert by_set["Wheels"] == "A"
        assert set(c.set_name for c in sel) == {"Carpaint", "Doors", "Wheels"}
    assert {dict((c.set_name, c.variant) for c in s)["Carpaint"] for s in perms} == {"Noir", "Sakura"}


# --- one at a time ---
def test_one_at_a_time_count_is_sum_not_product():
    job = _job(MatrixMode.ONE_AT_A_TIME, {"Carpaint": ("Noir", "Sakura"), "Wheels": ("A", "B", "C")})
    assert count_permutations(job, SETS) == 5  # 2 + 3, not 6


def test_one_at_a_time_varies_one_set_others_pinned():
    job = _job(MatrixMode.ONE_AT_A_TIME, {"Carpaint": ("Noir", "Sakura"), "Wheels": ("A", "B")})
    perms = expand_matrix(job, BASE, SETS)
    assert len(perms) == 4  # 2 + 2
    for sel in perms:
        by_set = {c.set_name: c.variant for c in sel}
        assert by_set["Doors"] == "Closed"  # untouched set always pinned
    base_vals = {"Carpaint": "Noir", "Wheels": "A"}
    for sel in perms:
        by_set = {c.set_name: c.variant for c in sel}
        diffs = [k for k in ("Carpaint", "Wheels") if by_set[k] != base_vals[k]]
        assert len(diffs) <= 1  # at most one swept set differs from base


# --- curated ---
def test_curated_count_and_expansion():
    c1 = (VariantChoice("/W/Looks", "Carpaint", "Noir"), VariantChoice("/W/Body", "Doors", "Open"))
    c2 = (VariantChoice("/W/Looks", "Carpaint", "Sakura"), VariantChoice("/W/Body", "Doors", "Closed"))
    job = _job(MatrixMode.CURATED, {}, curated=(c1, c2))
    assert count_permutations(job, SETS) == 2
    assert expand_matrix(job, BASE, SETS) == [c1, c2]


# --- explosion guard ---
def test_default_threshold_is_500():
    assert DEFAULT_EXPLOSION_THRESHOLD == 500


def test_guard_passes_under_threshold():
    guard_count(499)  # no raise


def test_guard_raises_over_threshold():
    with pytest.raises(ExplosionError):
        guard_count(501)


def test_guard_confirm_bypasses():
    guard_count(10_000, confirm=True)  # no raise


def test_guard_custom_threshold():
    with pytest.raises(ExplosionError):
        guard_count(11, threshold=10)


# --- estimate ---
def test_estimate_seconds_scales_with_count_and_frames():
    job_single = _job(MatrixMode.ONE_AT_A_TIME, {"Carpaint": ("Noir", "Sakura")})  # 2 perms
    single = estimate_seconds(job_single, SETS, per_frame_s=4.0)
    assert single == pytest.approx(2 * 1 * 1 * 4.0)


def test_labels_one_at_a_time_name_the_swept_variant():
    # one-at-a-time names by the swept set, even when the value equals base (Noir)
    job = _job(MatrixMode.ONE_AT_A_TIME, {"Carpaint": ("Noir", "Sakura")})
    labels = sorted(label for _, label in expand_labeled(job, BASE, SETS))
    assert labels == ["Carpaint-Noir", "Carpaint-Sakura"]


def test_labels_cartesian_name_all_included_sets():
    job = _job(MatrixMode.FULL_CARTESIAN, {"Carpaint": ("Noir", "Sakura"), "Wheels": ("A", "B")})
    labels = {label for _, label in expand_labeled(job, BASE, SETS)}
    assert labels == {"Carpaint-Noir_Wheels-A", "Carpaint-Noir_Wheels-B",
                      "Carpaint-Sakura_Wheels-A", "Carpaint-Sakura_Wheels-B"}


def test_estimate_animation_range_multiplies_frames():
    job = _job(MatrixMode.ONE_AT_A_TIME, {"Carpaint": ("Noir", "Sakura")})
    object.__setattr__(job, "frame_mode", "animation_range")
    est = estimate_seconds(job, SETS, per_frame_s=2.0, frame_count=10)
    assert est == pytest.approx(2 * 10 * 1 * 2.0)


# --- animation batch auto-encodes the sequence mp4 (fps = stage fps / step) ---
def test_animation_batch_auto_encodes_mp4_at_stage_fps_over_step(tmp_path, monkeypatch):
    from pathlib import Path

    from PIL import Image

    from dev_variant_presenter.batch import engine
    from dev_variant_presenter.models import StageInfo
    from dev_variant_presenter.post import processing

    monkeypatch.setattr(engine, "_converge", lambda *a, **k: None)

    def fake_save(renderer, rp, path, delta_time):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 18), (180, 40, 40)).save(p)
    monkeypatch.setattr(engine, "_save_frame", fake_save)

    class StubBackend:
        """Matches engine.RenderBackend — no renderer.open_usd / update_from_usd_time."""
        def open_composite(self, path): pass
        def set_time(self, time_code): pass
        def reset(self): pass
        def step_product(self, rp, dt): return None   # _converge/_save_frame are stubbed too

    class StubComposer:
        @staticmethod
        def build_composite(user_usd, sel, **kw):
            Path(kw["out_path"]).write_text("#usda 1.0")

    job = BatchJob(mode=MatrixMode.ONE_AT_A_TIME, base_selection=BASE,
                   included={"Carpaint": ("Noir",)}, cameras=["/Cam"],
                   quality=QualitySpec(), frame_mode="animation_range",
                   out_dir=str(tmp_path), frame_start=1, frame_end=4, frame_step=2)
    info = StageInfo(str(tmp_path / "x.usd"), "World", "Y", 1, 4, 60.0, SETS, ())
    engine.run_batch(job, BASE, info, backend=StubBackend(),
                     render_product_path="/Render/RP", emit=lambda e: None,
                     is_cancelled=lambda: False, composer=StubComposer,
                     user_usd="x.usd")

    seq = tmp_path / "Carpaint-Noir"
    assert len(list(seq.glob("*.png"))) == 2          # frames 1 and 3
    mp4 = tmp_path / "Carpaint-Noir.mp4"
    assert mp4.is_file() and mp4.stat().st_size > 0   # assembled at render time
    assert processing.video_fps(str(mp4)) == 30       # 60 fps stage / step 2
