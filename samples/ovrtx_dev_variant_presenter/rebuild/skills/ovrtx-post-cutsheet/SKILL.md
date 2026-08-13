---
name: ovrtx-post-cutsheet
description: >
  Post-process rendered variant permutations: browse a results folder of stills/videos,
  burn variant-label plates onto renders (originals untouched), assemble frame sequences into
  MP4, and compose a labeled contact sheet ("cut sheet"). Pure PIL/OpenCV/ffmpeg, off the
  render thread. Use when adding a Results / labeling / contact-sheet feature to a variant
  presenter.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, ovrtx, post-processing, labels, contact-sheet, video]
  domain: ai-ml
  languages: [python]
---

# Results + post-processing

No ovrtx; pure PIL / OpenCV / (optional) ffmpeg. Run in a thread-pool executor so it never
blocks the control plane or the render thread.

## Browse — `list_results(dir)`
Enumerate the rendered stills (`{label}.png` and `{label}/####.png` sequences) and videos
(`{label}.mp4`) in a folder for the Results UI. Serve files via `GET /api/frame?path=` /
`GET /api/video?path=` (validate the suffix + that the file exists).

## Overlay labels — `overlay_all(dir) -> count`
Burn a label plate (e.g. NVIDIA-green `"{set}: {variant}"`) onto each render and write the
labeled copy into a `_labeled/` subfolder. **Originals are never modified** (the Results UI
keeps showing the clean renders; labels are a derived artifact). Handle BOTH layouts: a
folder of frames AND **top-level single stills** `{label}.png` — a folder-only implementation
silently overlays nothing for the common single-still / Cartesian batch.

**Label parsing** (recover `{set}: {variant}` from the filename): set names contain `_` (e.g.
`Wheel_Colors`), so do NOT split on `_`. Accumulate `_`-segments until a `-` boundary between
`{set}-{variant}` pairs (the inverse of the batch naming convention). Cache-bust the served
URL (`&t=`) so an overlaid frame reloads in the browser.

## Video — `frames_to_video(frame_dir, out_path, fps)`
Assemble a numbered frame sequence into an MP4: **H.264 (`libx264`) + `pix_fmt yuv420p` via
imageio-ffmpeg — NEVER OpenCV `cv2.VideoWriter` with `mp4v`.** MPEG-4 Part 2 writes a
valid-looking file that `/api/video` serves fine and Chrome's `<video>` silently cannot decode →
the Results player shows NOTHING. yuv420p needs even dimensions, so pad
minimally (`macro_block_size=2`) rather than letting ffmpeg snap to 16. `mp4v` survives ONLY as an
explicit last-resort fallback for when no ffmpeg binary is available at all (imageio-ffmpeg not
importable). This is PUBLIC and reused by the batch animation_range path AND the timeline render —
a codec regression here breaks both. `convert_all_to_videos(dir, fps)` walks a results folder and
encodes each sequence. `compress_video` (ffmpeg 2-pass) is optional — return gracefully (no-op) if
ffmpeg is absent rather than failing.

## Cut sheet — `make_cut_sheet(dir) -> path`
Compose all variant stills in a folder into one labeled contact-sheet image (a grid with each
cell labeled by its `{set}: {variant}`). One image to compare the whole sweep at a glance.

## API
`POST /api/post/overlay {out_dir} -> {count}` · `POST /api/post/cutsheet {out_dir} -> {path}`
· `POST /api/post/video {out_dir,fps} -> {count}` · `POST /api/post/compress {video_path} ->
{path}` — all `run_in_executor`. The Results tab drives these and previews via the frame/video
routes.
