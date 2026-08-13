// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Timeline core — client-side mirror of dev_variant_presenter/sequence/timeline.py.
// Pure logic so scrubbing computes state_at instantly in the browser; toDict() produces the
// exact JSON shape /api/timeline/render + Timeline.from_dict consume. Node-testable (UMD).
(function (root, factory) {
  const api = factory();
  root.TL = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const DEFAULT_CLIP_S = 2.0;
  const EPS = 1e-9;

  // clip covering t, else the previous clip (a gap holds), else null — mirrors the
  // server's camera_clip_at so scrub and render agree on clip-relative time
  function clipAt(track, t) {
    let held = null;
    const clips = [...(track.clips || [])].sort((a, b) => a.start_s - b.start_s);
    for (const c of clips) {
      if (c.start_s <= t + EPS) {
        held = c;
        if (t < c.start_s + c.duration_s - EPS) return c;
      } else break;
    }
    return held;
  }
  function valueAt(track, t) {
    const c = clipAt(track, t);
    return c ? c.value : null;
  }

  // base: [{prim_path,set_name,variant}] -> {selection, camera, cameraClipStart}
  function stateAt(tl, t, base) {
    const trackValues = {};
    let camera = null, cameraClipStart = 0;
    for (const track of tl.tracks || []) {
      if (track.kind === "camera") {
        let c = clipAt(track, t);
        if (!c && track.clips && track.clips.length) {
          // lead-in: the FIRST camera clip governs before it starts (mirrors the server)
          c = [...track.clips].sort((a, b) => a.start_s - b.start_s)[0];
        }
        if (c) { camera = c.value; cameraClipStart = c.start_s; }
        continue;
      }
      const v = valueAt(track, t);
      if (v != null && track.set_name != null) trackValues[track.set_name] = v;
    }
    const selection = base.map((b) => ({
      prim_path: b.prim_path, set_name: b.set_name,
      variant: track_value(trackValues, b.set_name, b.variant),
    }));
    return { selection, camera, cameraClipStart };
  }
  function track_value(tv, set_name, fallback) {
    return Object.prototype.hasOwnProperty.call(tv, set_name) ? tv[set_name] : fallback;
  }

  function frameCount(tl) { return Math.round(tl.duration_s * tl.fps); }

  // true if `tl` is already initialized for a stage with these variant sets (same set names),
  // so a SAME-stage panel rebuild (e.g. authoring/updating the turntable camera re-runs
  // buildPanels) can preserve existing clips instead of wiping them. False for an
  // empty/uninitialized tl or a genuinely different stage.
  function tracksMatchStage(tl, variantSets) {
    const have = (variantSets || []).map((s) => s.set_name);
    const cur = ((tl && tl.tracks) || []).filter((t) => t.kind === "variant_set").map((t) => t.set_name);
    return cur.length > 0 && cur.length === have.length && have.every((s) => cur.includes(s));
  }

  // advance the playhead by dt seconds; clamp/stop at the end, or wrap when looping.
  // pure: drives the transport rAF loop. duration<=0 (empty timeline) -> parked + stop.
  function nextPlayheadTime(t0, dt, duration, loop) {
    if (!(duration > 0)) return { t: 0, stop: true };
    const t = t0 + dt;
    if (t < duration) return { t, stop: false };
    if (loop) return { t: t % duration, stop: false };
    return { t: duration, stop: true };
  }

  // snap t to the 1/fps frame grid, step one frame in `dir`, clamp to [0, duration]
  function frameStep(t, fps, dir, duration) {
    const f = Math.round(t * fps) + dir;
    return Math.max(0, Math.min(f / fps, duration));
  }

  // Append-at-playhead placement: drop a clip of `value` at `start` for `dur` seconds,
  // OVERWRITING whatever it covers. The new clip always lands exactly at `start` (the
  // playhead). Overlapped clips are trimmed (head/tail survives), split (a clip straddling
  // the insert becomes two of the same value), or dropped (fully covered). Returns a new,
  // start-sorted clips array; the input is not mutated. Unlike clampClip (which resolves
  // overlaps for manual drags by shoving the clip to a free slot), this never moves the new
  // clip off the playhead.
  function placeClipOverwrite(clips, value, start, dur) {
    const end = start + dur;
    const out = [];
    for (const c of clips || []) {
      const cEnd = c.start_s + c.duration_s;
      if (cEnd <= start + EPS || c.start_s >= end - EPS) { out.push(c); continue; }  // no overlap
      if (c.start_s < start - EPS) out.push({ ...c, duration_s: start - c.start_s }); // head survives
      if (cEnd > end + EPS) out.push({ ...c, start_s: end, duration_s: cEnd - end });  // tail survives
      // fully-covered middle is dropped
    }
    out.push({ value, start_s: start, duration_s: dur });
    out.sort((a, b) => a.start_s - b.start_s);
    return out;
  }

  function fillTrackAllVariants(setInfo, clip_s = DEFAULT_CLIP_S) {
    return {
      kind: "variant_set", set_name: setInfo.set_name, prim_path: setInfo.prim_path,
      clips: setInfo.variants.map((v, i) => ({ value: v, start_s: i * clip_s, duration_s: clip_s })),
    };
  }

  function makeSlideshow(sets, clip_s = DEFAULT_CLIP_S) {
    // KEEP one track per set; lay every variant out sequentially across the whole timeline
    // (one change at a time), each clip on its OWN set's track — a staircase: all of set A's
    // variants, then all of set B's, etc. (loadPreset prepends the camera track.)
    let t = 0;
    const tracks = [];
    for (const s of sets) {
      const clips = [];
      for (const v of s.variants) { clips.push({ value: v, start_s: t, duration_s: clip_s }); t += clip_s; }
      tracks.push({ kind: "variant_set", set_name: s.set_name, prim_path: s.prim_path, clips });
    }
    return { duration_s: t, fps: 30.0, tracks };
  }

  function makeMixer(sets, clip_s = DEFAULT_CLIP_S) {
    const steps = sets.reduce((m, s) => Math.max(m, s.variants.length), 0);
    const tracks = sets.map((s) => {
      const raw = [];
      for (let i = 0; i < steps; i++) raw.push({ value: s.variants[Math.min(i, s.variants.length - 1)], start_s: i * clip_s, duration_s: clip_s });
      const merged = [];
      for (const c of raw) {
        const last = merged[merged.length - 1];
        if (last && last.value === c.value && Math.abs(last.start_s + last.duration_s - c.start_s) < EPS) last.duration_s += c.duration_s;
        else merged.push({ ...c });
      }
      return { kind: "variant_set", set_name: s.set_name, prim_path: s.prim_path, clips: merged };
    });
    return { duration_s: steps * clip_s, fps: 30.0, tracks };
  }

  return { DEFAULT_CLIP_S, valueAt, clipAt, stateAt, frameCount, fillTrackAllVariants, makeSlideshow, makeMixer, nextPlayheadTime, frameStep, tracksMatchStage, placeClipOverwrite };
});
