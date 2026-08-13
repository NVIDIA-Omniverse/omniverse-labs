# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from PIL import Image

from dev_variant_presenter.post import processing as post
from dev_variant_presenter.post.processing import (
    convert_all_to_videos, frames_to_video, list_results,
    overlay_all, overlay_text_on_image, overlay_text_on_permutation_folder,
)


def _png(path: Path, size=(2, 2), color=(10, 20, 30)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


# ---- overlay ----
def test_overlay_text_on_image_changes_pixels_and_file_still_opens(tmp_path):
    p = _png(tmp_path / "frame.png", size=(64, 32))
    before = list(Image.open(p).getdata())
    overlay_text_on_image(str(p), "Carpaint: Noir", font_size=12)
    img = Image.open(p); img.load()
    after = list(img.getdata())
    assert img.size == (64, 32)
    assert before != after


def test_overlay_text_on_image_tiny_2x2_does_not_raise(tmp_path):
    p = _png(tmp_path / "tiny.png")
    overlay_text_on_image(str(p), "X")
    assert Image.open(p).size == (2, 2)


def test_overlay_folder_name_parses_to_label(tmp_path):
    perm = tmp_path / "Carpaint-Noir"; perm.mkdir()
    for i in range(3):
        _png(perm / f"{i:04d}.png", size=(48, 24))
    assert overlay_text_on_permutation_folder(str(perm)) == "Carpaint: Noir"


def test_overlay_folder_multiset_label(tmp_path):
    perm = tmp_path / "Carpaint-Noir_Wheel_Colors-Gold"; perm.mkdir()
    _png(perm / "0000.png", size=(48, 24))
    assert overlay_text_on_permutation_folder(str(perm)) == "Carpaint: Noir | Wheel_Colors: Gold"


def test_overlay_all_processes_only_perm_folders(tmp_path):
    good = tmp_path / "Carpaint-Noir"; good.mkdir(); _png(good / "0000.png", size=(48, 24))
    scratch = tmp_path / "_mixer"; scratch.mkdir(); _png(scratch / "0000.png", size=(48, 24))
    empty = tmp_path / "Doors-Open"; empty.mkdir()
    assert overlay_all(str(tmp_path)) == 1


def test_overlay_all_is_non_destructive(tmp_path):
    # originals stay pristine; labeled copies land under _labeled/ mirroring the layout
    folder = tmp_path / "Carpaint-Noir"; folder.mkdir()
    frame = _png(folder / "0000.png", size=(48, 24))
    still = _png(tmp_path / "Carpaint-Sakura.png", size=(48, 24))
    before_frame = list(Image.open(frame).getdata())
    before_still = list(Image.open(still).getdata())
    assert overlay_all(str(tmp_path)) == 2          # folder + still
    assert list(Image.open(frame).getdata()) == before_frame      # original untouched
    assert list(Image.open(still).getdata()) == before_still      # original untouched
    labeled_frame = tmp_path / "_labeled" / "Carpaint-Noir" / "0000.png"
    labeled_still = tmp_path / "_labeled" / "Carpaint-Sakura.png"
    assert labeled_frame.is_file() and labeled_still.is_file()
    assert list(Image.open(labeled_still).getdata()) != before_still   # label burned into the COPY


# ---- video ----
def test_frames_to_video_writes_nonempty_mp4(tmp_path):
    src = tmp_path / "Carpaint-Noir"; src.mkdir()
    for i in range(3):
        _png(src / f"{i:04d}.png", size=(48, 24), color=(i * 40, 0, 0))
    out = tmp_path / "Carpaint-Noir.mp4"
    assert frames_to_video(str(src), str(out), fps=24) is True
    assert out.exists() and out.stat().st_size > 0


def test_frames_to_video_empty_dir_returns_false(tmp_path):
    src = tmp_path / "empty"; src.mkdir()
    assert frames_to_video(str(src), str(tmp_path / "x.mp4")) is False


def test_convert_all_to_videos_counts_and_writes(tmp_path):
    perm = tmp_path / "Carpaint-Noir"; perm.mkdir()
    for i in range(3):
        _png(perm / f"{i:04d}.png", size=(48, 24))
    scratch = tmp_path / "_mixer"; scratch.mkdir(); _png(scratch / "0000.png", size=(48, 24))
    assert convert_all_to_videos(str(tmp_path), fps=24) == 1
    assert (tmp_path / "Carpaint-Noir.mp4").exists()


# ---- compress (no-ffmpeg path) ----
def test_compress_video_no_ffmpeg_returns_none(tmp_path, monkeypatch):
    src = tmp_path / "clip.mp4"; src.write_bytes(b"\x00\x00")
    monkeypatch.setattr(post.shutil, "which", lambda _name: None)
    assert post.compress_video(str(src)) is None


# ---- results discovery ----
def test_list_results_reports_frames_video_and_label(tmp_path):
    perm = tmp_path / "Carpaint-Noir"; perm.mkdir()
    for i in range(2):
        _png(perm / f"{i:04d}.png", size=(32, 16))
    frames_to_video(str(perm), str(tmp_path / "Carpaint-Noir.mp4"))
    _png(tmp_path / "Doors-Open.png", size=(32, 16))
    results = {r["name"]: r for r in list_results(str(tmp_path))}
    assert results["Carpaint-Noir"]["label"] == "Carpaint: Noir"
    assert results["Carpaint-Noir"]["frame_count"] == 2
    assert results["Carpaint-Noir"]["video"].endswith("Carpaint-Noir.mp4")
    assert results["Doors-Open"]["label"] == "Doors: Open"
    assert results["Doors-Open"]["frame_count"] == 1


def test_list_results_skips_nested_results_dir(tmp_path):
    """A subfolder of NAMED stills is a nested results dir (e.g. a cartesian sub-batch),
    not one permutation's animation — it must NOT collapse into a single tile (the bug
    that leaked a 'folder' into the cutsheet). Only numbered frame sequences are perms."""
    nested = tmp_path / "cartesian"; nested.mkdir()
    _png(nested / "Carpaint-Noir_Light-Red.png", size=(32, 16))
    _png(nested / "Carpaint-Noir_Light-Blue.png", size=(32, 16))
    anim = tmp_path / "Carpaint-Noir"; anim.mkdir()
    for i in range(2):
        _png(anim / f"{i:04d}.png", size=(32, 16))
    _png(tmp_path / "Doors-Open.png", size=(32, 16))
    names = {r["name"] for r in list_results(str(tmp_path))}
    assert "cartesian" not in names              # nested results dir excluded
    assert names == {"Carpaint-Noir", "Doors-Open"}


def test_cut_sheet_excludes_nested_results_dir(tmp_path):
    from dev_variant_presenter.post.processing import make_cut_sheet
    from PIL import Image
    nested = tmp_path / "cartesian"; nested.mkdir()
    _png(nested / "Carpaint-Noir_Light-Red.png", size=(40, 24))
    _png(tmp_path / "Doors-Open.png", size=(40, 24))
    _png(tmp_path / "Doors-Closed.png", size=(40, 24))
    out = make_cut_sheet(str(tmp_path), columns=1)
    assert out is not None
    _, h = Image.open(out).size
    # 1 column: height == one row per tile. Two named stills -> 2 rows; if 'cartesian'
    # leaked as a 3rd tile the sheet would be a full row taller.
    cell_h = 24 + 40                             # ch + label_h (defaults)
    assert h < 3 * cell_h                        # 2 rows fit; a 3rd would exceed


# --- fps probe + overlay re-encodes the sibling mp4 ---
def test_video_fps_probe_round_trips(tmp_path):
    src = tmp_path / "seq"; src.mkdir()
    for i in range(3):
        _png(src / f"{i:04d}.png", size=(48, 24))
    out = tmp_path / "seq.mp4"
    assert frames_to_video(str(src), str(out), fps=30)
    assert post.video_fps(str(out)) == 30


def test_overlay_all_writes_labeled_video_preserving_fps(tmp_path):
    perm = tmp_path / "Carpaint-Noir"; perm.mkdir()
    for i in range(3):
        _png(perm / f"{i:04d}.png", size=(64, 32))
    mp4 = tmp_path / "Carpaint-Noir.mp4"
    assert frames_to_video(str(perm), str(mp4), fps=12)
    before = mp4.read_bytes()
    assert overlay_all(str(tmp_path)) == 1
    assert mp4.read_bytes() == before                 # original video untouched
    labeled = tmp_path / "_labeled" / "Carpaint-Noir.mp4"
    assert labeled.is_file()                          # labeled re-encode lives in _labeled/
    assert post.video_fps(str(labeled)) == 12         # original playback rate preserved


def test_cut_sheet_composes_labeled_grid_without_touching_sources(tmp_path):
    from dev_variant_presenter.post.processing import make_cut_sheet
    stills = [_png(tmp_path / f"Carpaint-{v}.png", size=(64, 32), color=(i * 60, 90, 40))
              for i, v in enumerate(["Noir", "Sakura", "Green"])]
    anim = tmp_path / "Doors-Open"; anim.mkdir(); _png(anim / "0000.png", size=(64, 32))
    before = [list(Image.open(p).getdata()) for p in stills]
    out = make_cut_sheet(str(tmp_path))
    sheet = Image.open(out); sheet.load()
    # 4 cells -> 2x2 grid: cells 64x(32+40 label) + 12px pads
    assert sheet.size == (2 * 64 + 3 * 12, 2 * 72 + 3 * 12)
    assert [list(Image.open(p).getdata()) for p in stills] == before   # sources untouched
    out2 = make_cut_sheet(str(tmp_path))               # rerun: excludes itself, same layout
    assert Image.open(out2).size == sheet.size


def test_cut_sheet_empty_dir_returns_none(tmp_path):
    from dev_variant_presenter.post.processing import make_cut_sheet
    assert make_cut_sheet(str(tmp_path)) is None
