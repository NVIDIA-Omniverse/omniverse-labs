# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-processing for rendered variant-permutation frames.

Ported from the old post_processing.py / post_rendering_utils.py. Pure PIL / OpenCV /
ffmpeg-binary — no ovrtx, no GPU; runs in the FastAPI thread-pool executor. Mixer/slideshow
are NOT here — they are expressed as Timeline presets instead. `frames_to_video` stays
public because the timeline renderer reuses it.

Naming contract (matches batch.jobs.permutation_name): a permutation dir/still is
`{set}-{variant}` tokens joined by `_` (e.g. `Carpaint-Noir_Wheel_Colors-Gold`). The label
parser splits on `_`, then each token once on `-`, rendering `"{set}: {variant}"` joined by
`" | "`. Folders starting with `_`, or with no `*.png`, are skipped.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont

NVIDIA_GREEN = (118, 185, 0)


def _label_from_perm_name(name: str) -> str:
    """Inverse of permutation_name. Set names can contain '_' (e.g. Wheel_Colors), so we
    can't naively split on '_': accumulate '_'-segments into a token until one contains '-'
    (the set-variant boundary), then split that token once on '-'. Correct when variants
    have no '_' (holds for the ConceptCar; single-word variants)."""
    parts, buf = [], []
    for seg in name.split("_"):
        buf.append(seg)
        if "-" in seg:
            set_name, variant = "_".join(buf).split("-", 1)
            parts.append(f"{set_name}: {variant}")
            buf = []
    return " | ".join(parts) if parts else name


# First one that loads wins: Windows, then the two fonts a Linux box almost always has
# (DejaVu on Debian/Ubuntu, Liberation on RHEL), then macOS. Bare names resolve through the
# platform font search; the absolute paths cover installs that search doesn't see.
_FONT_CANDIDATES = (
    "arial.ttf", "Arial.ttf",
    "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
)


def _font(size: int):
    """A font at the REQUESTED size on any platform; never raises. The last resort is
    Pillow's built-in font sized via load_default(size) — plain load_default() returns a
    ~11px bitmap whatever you ask for, which makes labels unreadable at 720p+ and throws
    off the text-extent math the overlay/cut-sheet layouts do."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size)   # Pillow >= 10.1 (pinned 12.3.0)
    except Exception:
        return ImageFont.load_default()


def overlay_text_on_image(path: str, text: str, font_size: int = 25, dst: str | None = None) -> None:
    """Burn `text` onto the image bottom-left, NVIDIA-green on a dark plate. Writes to `dst`
    when given (non-destructive copy), else in place."""
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin = 10
    x = margin
    y = max(0, img.height - text_h - margin * 2)
    draw.rectangle([x - 4, y - 4, x + text_w + 8, y + text_h + 8], fill=(0, 0, 0))
    draw.text((x, y), text, fill=NVIDIA_GREEN, font=font)
    out = Path(dst) if dst else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))


def overlay_text_on_permutation_folder(perm_dir: str, dst_dir: str | None = None) -> str:
    """Overlay the folder's parsed label onto every PNG. Labeled frames land in `dst_dir`
    when given (non-destructive), else in place. Returns the label."""
    perm = Path(perm_dir)
    label = _label_from_perm_name(perm.name)
    if perm.is_dir():
        for png in natsorted(perm.glob("*.png")):
            dst = str(Path(dst_dir) / png.name) if dst_dir else None
            overlay_text_on_image(str(png), label, dst=dst)
    return label


def overlay_all(out_dir: str) -> int:
    """NON-DESTRUCTIVE label burn-in: originals stay pristine; labeled copies land under
    `out_dir/_labeled/` mirroring the layout — frame folders, top-level stills, and (for
    folders with a sibling {name}.mp4) a re-encoded labeled video at the original's fps.
    Returns the number of permutations labeled."""
    root = Path(out_dir)
    dst_root = root / "_labeled"
    count = 0
    for d in natsorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and any(d.glob("*.png")):
            labeled_dir = dst_root / d.name
            overlay_text_on_permutation_folder(str(d), dst_dir=str(labeled_dir))
            sibling = root / f"{d.name}.mp4"
            if sibling.is_file():
                frames_to_video(str(labeled_dir), str(dst_root / f"{d.name}.mp4"),
                                fps=video_fps(str(sibling)))
            count += 1
    for f in natsorted(root.glob("*.png")):   # single stills live at the top level
        if not f.name.startswith("_") and f.stem != "cutsheet":
            overlay_text_on_image(str(f), _label_from_perm_name(f.stem),
                                  dst=str(dst_root / f.name))
            count += 1
    return count


def make_cut_sheet(out_dir: str, *, columns: int | None = None, label_h: int = 40,
                   pad: int = 12, font_size: int = 24) -> str | None:
    """Compose every permutation's image (single still, or an animation's first frame) into
    ONE labeled contact-sheet grid — `out_dir/cutsheet.png`. Sources are untouched (labels
    are drawn on the sheet, not the renders). Returns the path, or None with no images."""
    import math
    perms = [p for p in list_results(out_dir)
             if p["first_frame"] and p["name"] != "cutsheet"]
    if not perms:
        return None
    imgs = [Image.open(p["first_frame"]).convert("RGB") for p in perms]
    cw = max(i.width for i in imgs)
    ch = max(i.height for i in imgs)
    n = len(imgs)
    cols = columns or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cell_h = ch + label_h
    W = cols * cw + pad * (cols + 1)
    H = rows * cell_h + pad * (rows + 1)
    sheet = Image.new("RGB", (W, H), (13, 17, 23))
    draw = ImageDraw.Draw(sheet)
    font = _font(font_size)
    for i, (img, p) in enumerate(zip(imgs, perms)):
        r, c = divmod(i, cols)
        x = pad + c * (cw + pad)
        y = pad + r * (cell_h + pad)
        sheet.paste(img, (x + (cw - img.width) // 2, y + (ch - img.height) // 2))
        label = p["label"]
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + max(0, (cw - tw) // 2), y + ch + (label_h - (bbox[3] - bbox[1])) // 2),
                  label, fill=NVIDIA_GREEN, font=font)
    out = Path(out_dir) / "cutsheet.png"
    sheet.save(str(out))
    return str(out)


def video_fps(video_path: str, default: int = 24) -> int:
    """Frame rate of an existing video, probed via ffmpeg's stream banner ('... 30 fps, ...').
    Falls back to `default` if ffmpeg is missing or the rate can't be parsed."""
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        return default
    try:
        # `ffmpeg -i <file>` with no output exits non-zero but prints stream info on stderr
        r = subprocess.run([ffmpeg, "-i", video_path], capture_output=True, text=True)
        m = re.search(r"(\d+(?:\.\d+)?)\s*fps", r.stderr)
        return max(1, round(float(m.group(1)))) if m else default
    except (OSError, ValueError):
        return default


def frames_to_video(image_dir: str, output_path: str, fps: int = 24) -> bool:
    """Encode natsorted PNGs into an H.264 MP4 — browsers only decode H.264/VP9/AV1, so
    OpenCV's mp4v (MPEG-4 Part 2) plays in desktop players but NOT in Chrome's <video>.
    Uses the static ffmpeg bundled with imageio-ffmpeg; falls back to cv2/mp4v if absent.
    Public — the timeline renderer reuses it. Returns False on no frames / no encoder."""
    frames = natsorted(Path(image_dir).glob("*.png"))
    if not frames:
        return False
    try:
        return _frames_to_h264(frames, output_path, fps)
    except ImportError:
        return _frames_to_mp4v(frames, output_path, fps)


def _frames_to_h264(frames: list, output_path: str, fps: int) -> bool:
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    first = np.asarray(Image.open(frames[0]).convert("RGB"))
    h, w = first.shape[:2]
    # yuv420p is the universally-decodable pixel format but needs even dims;
    # macro_block_size=2 pads at most 1px instead of ffmpeg's default 16px snap.
    # quality=None disables imageio's own -crf so ours wins: CRF 17 is visually
    # lossless for rendered content (the default ~23-25 visibly degrades vs the PNGs)
    writer = imageio_ffmpeg.write_frames(
        output_path, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
        macro_block_size=2, quality=None,
        output_params=["-crf", "17", "-preset", "slow", "-movflags", "+faststart"])
    writer.send(None)   # prime the generator
    try:
        for f in frames:
            img = np.asarray(Image.open(f).convert("RGB"))
            if img.shape[:2] != (h, w):   # tolerate stray mixed-size frames
                img = np.asarray(Image.fromarray(img).resize((w, h)))
            writer.send(np.ascontiguousarray(img))
    finally:
        writer.close()
    return True


def _frames_to_mp4v(frames: list, output_path: str, fps: int) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    first = cv2.imread(str(frames[0]))
    if first is None:
        return False
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        for f in frames:
            img = cv2.imread(str(f))
            if img is not None:
                writer.write(img)
    finally:
        writer.release()
    return True


def convert_all_to_videos(out_dir: str, fps: int = 24) -> int:
    """Encode every permutation folder under out_dir to a sibling {name}.mp4. Returns count."""
    root = Path(out_dir)
    count = 0
    for d in natsorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("_") and any(d.glob("*.png")):
            if frames_to_video(str(d), str(root / f"{d.name}.mp4"), fps):
                count += 1
    return count


def _ffmpeg_exe() -> str | None:
    """ffmpeg from PATH, else the static binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return None


def compress_video(video_path: str, *, target_bitrate: str = "2M",
                   suffix: str = "_cmp", two_pass: bool = True) -> str | None:
    """Two-pass libx264 compress via the ffmpeg binary (PATH or bundled). Returns the output
    path, or None if ffmpeg is absent or the encode fails (graceful — never raises)."""
    ffmpeg = _ffmpeg_exe()
    if ffmpeg is None:
        return None
    src = Path(video_path)
    if not src.is_file():
        return None
    out = src.with_name(f"{src.stem}{suffix}.mp4")
    null_sink = "NUL" if os.name == "nt" else "/dev/null"
    passlog = str(src.with_suffix("")) + suffix
    base = [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264", "-b:v", target_bitrate]
    try:
        if two_pass:
            subprocess.run(base + ["-pass", "1", "-passlogfile", passlog, "-an", "-f", "mp4", null_sink],
                           check=True, capture_output=True)
            subprocess.run(base + ["-pass", "2", "-passlogfile", passlog, "-c:a", "aac", str(out)],
                           check=True, capture_output=True)
        else:
            subprocess.run(base + ["-c:a", "aac", str(out)], check=True, capture_output=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    finally:
        for log in src.parent.glob(f"{Path(passlog).name}*.log*"):
            try:
                log.unlink()
            except OSError:
                pass
    return str(out) if out.is_file() else None


def list_results(out_dir: str) -> list[dict]:
    """Enumerate permutation results: folder-of-frames and/or single still, paired with a
    sibling {name}.mp4. Each: {name,label,frames,frame_count,first_frame,video}."""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    results: dict[str, dict] = {}

    def _entry(name: str) -> dict:
        return results.setdefault(name, {
            "name": name, "label": _label_from_perm_name(name),
            "frames": [], "frame_count": 0, "first_frame": None, "video": None})

    for d in natsorted(root.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            frames = [str(p) for p in natsorted(d.glob("*.png"))]
            # only a NUMBERED frame sequence (0000.png, 0001.png, ...) is one permutation's
            # animation. A folder of NAMED stills is a nested results dir (e.g. a cartesian
            # sub-batch) — collapsing it to a single first-frame tile leaked a bogus "folder"
            # entry into the cutsheet + Results list. Point those tools at the subfolder itself.
            if frames and all(Path(f).stem.isdigit() for f in frames):
                e = _entry(d.name)
                e["frames"] = frames
                e["frame_count"] = len(frames)
                e["first_frame"] = frames[0]

    for f in natsorted(root.iterdir()):
        if not f.is_file() or f.name.startswith("_"):
            continue
        if f.suffix.lower() == ".png":
            e = _entry(f.stem)
            if not e["frames"]:
                e["frames"] = [str(f)]
                e["frame_count"] = 1
                e["first_frame"] = str(f)
        elif f.suffix.lower() == ".mp4":
            _entry(f.stem)["video"] = str(f)

    return [results[k] for k in natsorted(results)]
