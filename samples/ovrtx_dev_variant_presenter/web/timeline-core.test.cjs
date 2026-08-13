// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Node test: the JS timeline core must match dev_variant_presenter/sequence/timeline.py.
// Run: node web/timeline-core.test.cjs
const assert = require("assert");
const TL = require("./timeline-core.js");

const base = [
  { prim_path: "/World/Looks", set_name: "Carpaint", variant: "Noir" },
  { prim_path: "/World/Looks", set_name: "Wheel_Colors", variant: "Chrome" },
  { prim_path: "/World/Doors", set_name: "Doors", variant: "Closed" },
];
const v = (sel, name) => (sel.find((c) => c.set_name === name) || {}).variant;
const T = (kind, set_name, prim_path, clips) => ({ kind, set_name, prim_path, clips });
const C = (value, start_s, duration_s) => ({ value, start_s, duration_s });

// clip under playhead
let tl = { duration_s: 8, fps: 30, tracks: [T("variant_set", "Carpaint", "/World/Looks", [C("Sakura", 0, 4), C("Sky", 4, 4)])] };
assert.equal(v(TL.stateAt(tl, 1, base).selection, "Carpaint"), "Sakura");
assert.equal(v(TL.stateAt(tl, 5, base).selection, "Carpaint"), "Sky");

// gap holds previous
tl = { duration_s: 10, fps: 30, tracks: [T("variant_set", "Carpaint", "/World/Looks", [C("Sakura", 0, 3), C("Sky", 6, 3)])] };
assert.equal(v(TL.stateAt(tl, 4.5, base).selection, "Carpaint"), "Sakura");

// before first clip -> base
tl = { duration_s: 10, fps: 30, tracks: [T("variant_set", "Carpaint", "/World/Looks", [C("Sakura", 2, 3)])] };
assert.equal(v(TL.stateAt(tl, 0.5, base).selection, "Carpaint"), "Noir");

// untracked set pinned to base
tl = { duration_s: 8, fps: 30, tracks: [T("variant_set", "Carpaint", "/World/Looks", [C("Sakura", 0, 8)])] };
let s = TL.stateAt(tl, 1, base).selection;
assert.equal(v(s, "Carpaint"), "Sakura");
assert.equal(v(s, "Wheel_Colors"), "Chrome");
assert.equal(v(s, "Doors"), "Closed");
assert.equal(s.length, 3);

// camera track
tl = { duration_s: 10, fps: 30, tracks: [T("camera", null, null, [C("/Cam/Main", 0, 5), C("/Cam/Close", 5, 5)])] };
assert.equal(TL.stateAt(tl, 1, base).camera, "/Cam/Main");
assert.equal(TL.stateAt(tl, 6, base).camera, "/Cam/Close");
assert.equal(TL.stateAt({ duration_s: 5, fps: 30, tracks: [] }, 1, base).camera, null);

// frameCount + presets
assert.equal(TL.frameCount({ duration_s: 2, fps: 30 }), 60);
const carpaint = { set_name: "Carpaint", prim_path: "/World/Looks", variants: ["Noir", "Sakura", "Sky"] };
const doors = { set_name: "Doors", prim_path: "/World/Doors", variants: ["Closed", "Open"] };
const ft = TL.fillTrackAllVariants(carpaint, 2.0);
assert.deepEqual(ft.clips.map((c) => c.value), ["Noir", "Sakura", "Sky"]);
assert.deepEqual(ft.clips.map((c) => c.start_s), [0, 2, 4]);
// slideshow: KEEP one track per set; lay every variant out sequentially across the whole
// timeline (one change at a time), each clip on its OWN set's track (a staircase).
const ss = TL.makeSlideshow([carpaint, doors], 1.5);
assert.equal(ss.tracks.length, 2);                                        // tracks kept, not collapsed
assert.deepEqual(ss.tracks.map((t) => t.set_name), ["Carpaint", "Doors"]);
assert.deepEqual(ss.tracks[0].clips.map((c) => c.value), ["Noir", "Sakura", "Sky"]);
assert.deepEqual(ss.tracks[0].clips.map((c) => c.start_s), [0, 1.5, 3.0]);
assert.deepEqual(ss.tracks[1].clips.map((c) => c.value), ["Closed", "Open"]);
assert.deepEqual(ss.tracks[1].clips.map((c) => c.start_s), [4.5, 6.0]);   // after Carpaint's three
assert.ok(Math.abs(ss.duration_s - 7.5) < 1e-9);                          // 5 variants * 1.5
const mx = TL.makeMixer([carpaint, doors], 2.0);
assert.equal(mx.tracks.length, 2);
assert.ok(Math.abs(mx.duration_s - 6.0) < 1e-9);

// nextPlayheadTime: advance, stop-at-end, loop-wrap, empty
assert.deepEqual(TL.nextPlayheadTime(0, 1, 10, false), { t: 1, stop: false });
assert.deepEqual(TL.nextPlayheadTime(2, 3, 10, false), { t: 5, stop: false });
assert.deepEqual(TL.nextPlayheadTime(8, 5, 10, false), { t: 10, stop: true });   // clamp + stop
assert.deepEqual(TL.nextPlayheadTime(8, 4, 10, true), { t: 2, stop: false });    // wrap (12 % 10)
assert.deepEqual(TL.nextPlayheadTime(0, 5, 0, false), { t: 0, stop: true });     // empty timeline

// frameStep: grid snap, direction, clamp both ends (fps 30 -> 1/30 s grid)
assert.ok(Math.abs(TL.frameStep(1.00, 30, 1, 10) - (31 / 30)) < 1e-9);   // 1.0s -> frame 30 -> 31
assert.ok(Math.abs(TL.frameStep(1.00, 30, -1, 10) - (29 / 30)) < 1e-9);  // -> frame 29
assert.ok(Math.abs(TL.frameStep(1.01, 30, 1, 10) - (31 / 30)) < 1e-9);   // off-grid 1.01s -> nearest frame 30 -> +1 -> 31
assert.equal(TL.frameStep(0, 30, -1, 10), 0);                            // clamp low
assert.equal(TL.frameStep(10, 30, 1, 10), 10);                           // clamp high

// tracksMatchStage: preserve the timeline only on a SAME-stage panel rebuild
const sets2 = [carpaint, doors];
assert.equal(TL.tracksMatchStage({ tracks: [] }, sets2), false);   // uninitialized -> reset
assert.equal(TL.tracksMatchStage({ tracks: [
  T("camera", null, null, []),
  T("variant_set", "Carpaint", "/p", []),
  T("variant_set", "Doors", "/p", []),
] }, sets2), true);                                                 // same sets -> preserve clips
assert.equal(TL.tracksMatchStage({ tracks: [T("variant_set", "Foo", "/p", [])] }, sets2), false); // other stage -> reset

// placeClipOverwrite: append-at-playhead drops the clip at `start`, OVERWRITING (trim/split/
// drop) whatever it covers. The new clip ALWAYS lands exactly at `start` and the result never
// overlaps. (Regression: the old addClip ran clampClip and shoved the clip off the playhead
// whenever it landed on an existing clip.)
const noOverlap = (cs) => {
  const o = [...cs].sort((a, b) => a.start_s - b.start_s);
  for (let i = 1; i < o.length; i++)
    assert.ok(o[i].start_s + 1e-9 >= o[i - 1].start_s + o[i - 1].duration_s, "clips overlap: " + JSON.stringify(o));
};
const placedAt = (cs, start) => cs.find((c) => c.value === "NEW" && Math.abs(c.start_s - start) < 1e-9);
const shape = (cs) => [...cs].sort((a, b) => a.start_s - b.start_s).map((c) => [c.value, c.start_s, c.duration_s]);

let pc = TL.placeClipOverwrite([], "NEW", 5, 2);                                  // empty track
assert.ok(placedAt(pc, 5)); assert.equal(pc.length, 1);

pc = TL.placeClipOverwrite([C("A", 0, 2), C("B", 2, 2)], "NEW", 6, 2);           // past the end -> gap
assert.ok(placedAt(pc, 6)); noOverlap(pc);

pc = TL.placeClipOverwrite([C("A", 0, 2), C("B", 6, 2)], "NEW", 4, 2);           // into a gap
assert.ok(placedAt(pc, 4)); noOverlap(pc);

pc = TL.placeClipOverwrite([C("A", 0, 2), C("B", 2, 2)], "NEW", 3, 2);           // inside last clip (looked dead)
assert.ok(placedAt(pc, 3)); noOverlap(pc);
assert.deepEqual(shape(pc), [["A", 0, 2], ["B", 2, 1], ["NEW", 3, 2]]);          // B trimmed back to the playhead

pc = TL.placeClipOverwrite([C("A", 0, 10)], "NEW", 5, 2);                        // mid a long clip -> split
assert.ok(placedAt(pc, 5)); noOverlap(pc);
assert.deepEqual(shape(pc), [["A", 0, 5], ["NEW", 5, 2], ["A", 7, 3]]);

pc = TL.placeClipOverwrite([C("A", 2, 2)], "NEW", 1, 4);                         // fully covers a clip -> drop it
assert.ok(placedAt(pc, 1)); assert.equal(pc.length, 1);

pc = TL.placeClipOverwrite([C("A", 0, 4)], "NEW", 2, 2);                         // exactly on a clip edge
assert.ok(placedAt(pc, 2)); noOverlap(pc);
assert.deepEqual(shape(pc), [["A", 0, 2], ["NEW", 2, 2]]);

console.log("timeline-core.test.cjs: ALL PASS");
