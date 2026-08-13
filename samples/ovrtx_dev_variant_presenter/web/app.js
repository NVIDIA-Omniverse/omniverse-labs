// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Dev Variant Presenter — control UI over the FastAPI plane + ovstream WebRTC video.
// IIFE: NVIDIA's streaming lib (fetched at setup, not shipped here — see
// THIRD-PARTY-NOTICES.md; an ESM bundle loaded as a classic script) leaks top-level
// minified globals like `el` into global scope; wrapping keeps our declarations local
// and avoids redeclaration SyntaxErrors that abort the whole file.
(() => {
const { AppStreamer, StreamType } = window.OVWebRTC;

// Windows localhost->::1 trap: our FastAPI WS server and the ovstream signaling server
// bind IPv4 only, but Chrome resolves `localhost` to IPv6 ::1 first for WebSockets.
// Normalize to the IPv4 loopback so either URL you paste (localhost OR 127.0.0.1) works.
const toIPv4 = (s) => s.replace(/^localhost/i, "127.0.0.1").replace(/^\[?::1\]?/, "127.0.0.1");

// The signaling port may be auto-shifted off 49100 if it was busy; learn the real one.
let signalPort = 49100;
async function loadConfig() {
  try {
    const c = await (await fetch("/api/config")).json();
    if (c && c.signal_port) signalPort = c.signal_port;
  } catch (e) { console.warn("config fetch failed; using default signaling port", e); }
}

const el = (id) => document.getElementById(id);
const statusEl = el("status");
const overlay = el("overlay");

let stage = null;
let selection = {};      // key `${prim_path}@@${set_name}` -> {prim_path,set_name,variant}
let quality = { mode: "RealTimePathTracing", samples_per_pixel: 64, max_bounces: 4, resolution: [1280, 720] };
let connected = false;
let streamLive = false;
let connectWatch = null;    // warns if the single-client video slot is held by another tab
let variantSwatches = {};   // set_name -> { variant: "#rrggbb" }, filled by the `classified` event

function setStatus(cls, text) { statusEl.className = "status " + cls; statusEl.textContent = text; }
function setOverlay(text) { overlay.style.display = text ? "block" : "none"; overlay.textContent = text || ""; }

// ---- control API ----
async function api(path, body) {
  const r = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${path} -> ${r.status} ${await r.text()}`);
  return r.json();
}

async function openStage(seed, extra) {
  const usd_path = el("usd-path").value.trim();
  if (!usd_path) return;
  // same-stage Open = attach, not reopen: a full /api/open recomposes + re-warms the
  // renderer (~30s of black viewport) for zero change. The SERVER must confirm — the
  // client's own state is stale after a server restart, and trusting it left recovery
  // attaching to a stageless server forever. Explicit state (a project restore's
  // seed/extra) still goes through the real open.
  if (!extra && !(Array.isArray(seed) && seed.length) && stage &&
      (usd_path === stage.source_url || usd_path === stage.usd_path)) {
    try {
      const st = await (await fetch("/api/stage")).json();
      if (st.open && st.ready && st.usd_path === stage.usd_path) {
        serverStageReady = true;
        if (!connected) connectStream();
        else setOverlay("");
        return;
      }
    } catch (e) { /* server unreachable — fall through to the real open */ }
  }
  setStatus("warming", "opening");
  setOverlay("Opening stage…");
  try {
    const sel = Array.isArray(seed) ? seed : null;
    const body = { usd_path };
    if (sel && sel.length) body.selections = sel;
    if (extra && extra.camera) body.camera_path = extra.camera;       // project restore rides in
    if (extra && extra.looks) body.looks = extra.looks;               // with the open -> the first
    if (extra && extra.xforms) body.xforms = extra.xforms;            // frame IS the saved state
    stage = await api("/api/open", body);
    // a mirrored https:// stage keeps its URL as the user-facing identity (the local
    // junction path is plumbing); local stages reflect whatever the server resolved
    const shown = stage.source_url || stage.usd_path;
    if (shown && shown !== usd_path) el("usd-path").value = shown;
    try { localStorage.setItem("dvp_last_stage", shown || usd_path); } catch (e) { /* ignore */ }
  } catch (e) {
    setStatus("error", "error"); setOverlay("Open failed: " + e.message); return;
  }
  try {
    // a bare reopen (crash recovery, manual re-open) may have had its session restored
    // server-side - reflect the restored selection + camera instead of stage defaults
    const seeded = Array.isArray(seed) && seed.length;
    buildPanels(seeded ? seed : (stage.selection && stage.selection.length ? stage.selection : seed));
    if (stage.camera && !(extra && extra.camera)) el("camera-select").value = stage.camera;
    ttRestoreFromStage(stage.turntable);   // authored rig -> pivot gizmo + tools come back
  } catch (e) {
    setStatus("error", "error");
    setOverlay("UI build error: " + (e && e.message ? e.message : e));
    console.error("openStage build error:", e);
    return;
  }
  if (!connected) connectStream();
  else setOverlay("");
}

function buildPanels(seed) {
  try { ttStopPreview(); ttHideGizmo(); } catch (e) { /* nothing to tear down */ }
  updateStageInfo();
  buildCameras();
  buildVariantCards(seed);
  buildGridPanel();
  buildTimelinePanel();
  // a stage can open while the Timeline tab is ALREADY active (Open Project lives there) —
  // sync the strip, which is otherwise only toggled by tab clicks (hidden until tab-bounce)
  el("timeline-strip").classList.toggle("show", timelineActive());
}

// A reloaded tab should ATTACH to the running session — ask the server what's open and
// rebuild the UI around it (field, panels, selection, camera) without re-opening anything.
// On same-stage recovery the panels (and the working timeline!) are left untouched.
async function attachIfOpen() {
  try {
    const st = await (await fetch("/api/stage")).json();
    if (!st.open) return false;
    if (!stage || stage.usd_path !== st.usd_path) {
      el("usd-path").value = st.source_url || st.usd_path;
      stage = st.info;
      stage.source_url = st.source_url || "";   // same shape as the /api/open response
      stage.camera = st.camera || stage.camera;  // so buildGridPanel can default-tick the active camera
      buildPanels(st.selection || []);
      if (st.camera) el("camera-select").value = st.camera;
      ttRestoreFromStage(st.turntable);   // authored rig -> pivot gizmo + tools come back
    }
    const res = st.display && st.display.resolution;
    if (res) {   // reattach: match the live stream size + ALWAYS re-sync the batch output size
      // grid-w/grid-h must sync UNCONDITIONALLY here. On a reload the browser can restore the
      // disp-res <select> to the right value while resetting these number inputs to their
      // 1280x720 HTML default; gating the sync on a disp-res MISMATCH leaves the grid fields
      // stale, and the next batch then silently renders at 1280x720 instead of the aspect the
      // UI is showing.
      if (el("disp-res").value !== res.join("x")) el("disp-res").value = res.join("x");
      el("grid-w").value = res[0]; el("grid-h").value = res[1];
    }
    serverStageReady = !!st.ready;
    setStatus("warming", "attaching");
    connectStream();
    return true;
  } catch (e) { console.warn("attach failed", e); return false; }
}

// Self-healing stream: when WebRTC drops (server restart, watchdog relaunch, expired
// library retries), keep re-attaching until it's back — never strand at "Reconnecting…".
let recoveryTimer = null;
function scheduleStreamRecovery(why) {
  if (recoveryTimer) return;                  // single-flight
  serverStageReady = false;   // stream lost = server state unknown until /api/stage answers
  setStatus("warming", "reconnecting");
  setOverlay(`Stream lost (${why}) — reconnecting…`);
  // tear down NOW and reconnect only after a real backoff: the stream server is
  // single-client, and a fresh session arriving mid-teardown of the old one gets
  // accepted at signaling but never gets media (the LIVE-but-black half-open)
  try { AppStreamer.terminate(); } catch (e) { /* ignore */ }
  connected = false; streamLive = false;
  recoveryTimer = setTimeout(async () => {
    recoveryTimer = null;
    if (connected || connecting) return;    // another path (e.g. resolution resize) beat us to it
    if (await attachIfOpen()) return;         // re-synced + connecting
    try {
      const st = await (await fetch("/api/stage")).json();
      if (!st.open) {
        const last = localStorage.getItem("dvp_last_stage");
        if (last) {                           // server restarted: reopen the last stage hands-off
          el("usd-path").value = last;
          setOverlay("Server restarted — reopening the last stage…");
          openStage();
          return;
        }
        setStatus("", "no stage"); setOverlay("Server restarted — open a stage or a project."); return;
      }
    } catch (e) { /* server still down — keep trying */ }
    scheduleStreamRecovery("retrying");
  }, 9000);
}

// metersPerUnit -> a human unit name (USD stores the scale, not the name)
function unitLabel(mpu) {
  if (!mpu) return "";
  const known = [[1, "meters"], [0.01, "centimeters"], [0.001, "millimeters"],
                 [0.0254, "inches"], [0.3048, "feet"]];
  for (const [k, name] of known) if (Math.abs(mpu - k) < 1e-9) return name;
  return mpu + " m/unit";
}
// read-only stage timebase + units, so the authored 60fps / Y-up / cm is never a mystery
function updateStageInfo() {
  const box = el("stage-info");
  if (!box) return;
  if (!stage || !stage.fps) { box.textContent = ""; return; }
  const fps = stage.fps;
  const s = Math.round(stage.start_time), e = Math.round(stage.end_time);
  const nframes = Math.max(1, e - s + 1);
  const parts = [`${(+fps).toFixed(fps % 1 ? 2 : 0)} fps`];
  if (nframes > 1) parts.push(`${nframes} frames (${s}–${e})`);
  if (stage.up_axis) parts.push(`${stage.up_axis}-up`);
  const u = unitLabel(stage.meters_per_unit);
  if (u) parts.push(u);
  box.textContent = "Stage: " + parts.join("  ·  ");
}

function buildCameras() {
  const sel = el("camera-select");
  sel.innerHTML = "";
  (stage.cameras || []).forEach((c) => {
    const o = document.createElement("option");
    o.value = c.path; o.textContent = `${c.name}  (${c.path})`;
    o.dataset.animated = c.animated ? "1" : "";
    sel.appendChild(o);
  });
  refreshCameraBadges();
  ttSyncCreateLabel();
}

// once the rig exists, the same button UPDATES it — and for the (animated) turntable
// camera this IS the save-framing action: Save framing only applies to stage cameras
const TT_CAM = "/TurntableRig/Turntable";
function ttSyncCreateLabel() {
  const b = el("tt-add");
  if (stage && stage.cameras && stage.cameras.some((c) => c.path === TT_CAM)) {
    b.innerHTML = "&#8635; Update camera from this view";
    b.setAttribute("data-help", "Re-authors the turntable camera from YOUR CURRENT VIEW " +
      "(exact pose, pan offsets included) and the current fields. This is how you save a " +
      "new framing for the turntable camera — Save framing above only applies to " +
      "regular stage cameras.");
  } else {
    b.innerHTML = "&#127909; Create camera from this view";
    b.setAttribute("data-help", "Creates the turntable camera using YOUR CURRENT VIEW as " +
      "the first frame — the exact pose, including any pan offset you composed.");
  }
}

// state-at-a-glance in the camera list: ↻ animated · ◆ saved framing · ✱ optics overrides
async function refreshCameraBadges() {
  let lookCams = [], framedCams = [];
  try {
    const st = await (await fetch("/api/stage")).json();
    lookCams = st.look_cams || [];
    framedCams = st.framed_cams || [];
  } catch (e) { /* badge-less is fine */ }
  document.querySelectorAll("#camera-select option").forEach((o) => {
    if (!o.value) return;
    const c = (stage.cameras || []).find((x) => x.path === o.value);
    if (!c) return;
    let badges = "";
    if (o.dataset.animated) badges += " \u21BB";                 // animated
    if (framedCams.includes(o.value)) badges += " \u25C6";       // saved framing
    if (lookCams.includes(o.value)) badges += " \u2731";         // optics overrides
    o.textContent = `${c.name}  (${c.path})${badges}`;
  });
}

function buildVariantCards(seed) {
  selection = {};
  variantSwatches = {};   // new stage — swatches arrive later via the `classified` event
  const seedMap = {};
  if (Array.isArray(seed)) seed.forEach((c) => { seedMap[`${c.prim_path}@@${c.set_name}`] = c.variant; });
  for (const vs of stage.variant_sets) {
    const key = `${vs.prim_path}@@${vs.set_name}`;
    selection[key] = { prim_path: vs.prim_path, set_name: vs.set_name, variant: seedMap[key] || vs.current };
  }
  el("vset-count").textContent = stage.variant_sets.length;
  const root = el("variant-cards");
  root.innerHTML = "";
  stage.variant_sets.forEach((vs) => {
    const key = `${vs.prim_path}@@${vs.set_name}`;
    const card = document.createElement("div"); card.className = "vcard";
    const name = document.createElement("div"); name.className = "name"; name.textContent = vs.set_name;
    const prim = document.createElement("div"); prim.className = "prim"; prim.textContent = vs.prim_path;
    const chips = document.createElement("div"); chips.className = "chips";
    vs.variants.forEach((v) => {
      const chip = document.createElement("span");
      chip.className = "chip" + (v === selection[key].variant ? " on" : "");
      chip.dataset.sname = vs.set_name; chip.dataset.variant = v;
      const dot = document.createElement("span"); dot.className = "swatch";  // hidden until colored
      const label = document.createElement("span"); label.textContent = v;
      chip.append(dot, label);
      chip.onclick = () => chooseVariant(key, v, chips);
      chips.appendChild(chip);
    });
    card.append(name, prim, chips);
    root.appendChild(card);
  });
}

// Color the chip dots once the classifier reports per-variant swatches (both panes).
function applySwatches() {
  document.querySelectorAll(".chip").forEach((chip) => {
    const c = variantSwatches[chip.dataset.sname] && variantSwatches[chip.dataset.sname][chip.dataset.variant];
    const dot = chip.querySelector(".swatch");
    if (c && dot) { dot.style.background = c; dot.style.display = "inline-block"; }
  });
}

function chooseVariant(key, variant, chipsEl) {
  selection[key].variant = variant;
  markDirty();   // base look belongs to the project
  [...chipsEl.children].forEach((c) => c.classList.toggle("on", c.dataset.variant === variant));
  api("/api/variant", { selections: Object.values(selection) }).catch((e) => console.error(e));
}

el("open-btn").onclick = () => openStage();   // don't pass the click event as `seed`
el("camera-select").onchange = (e) => {
  if (e.target.value) { markDirty(); api("/api/camera/snap", { camera_path: e.target.value }).catch(console.error); }
};
document.querySelectorAll(".mode-btn").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".mode-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    quality.mode = b.dataset.mode;
    api("/api/render-mode", { quality }).catch(console.error);
  };
});

// ---- Display settings: resolution/aspect + camera exposure / FOV / DOF ----
async function postDisplay(body) {
  markDirty();   // any display/optics change belongs to the project
  try {
    await fetch("/api/display", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (e) { console.error("display", e); }
}
el("disp-iso").oninput = () => { el("disp-iso-v").textContent = el("disp-iso").value; };
el("disp-iso").onchange = () => postDisplay({ iso: +el("disp-iso").value });
el("disp-fl").oninput = () => { el("disp-fl-v").textContent = el("disp-fl").value; };
el("disp-fl").onchange = () => postDisplay({ focal_length: +el("disp-fl").value });
el("disp-fs").oninput = () => { const v = +el("disp-fs").value; el("disp-fs-v").textContent = v > 0 ? v.toFixed(1) : "off"; };
el("disp-fs").onchange = () => postDisplay({ f_stop: +el("disp-fs").value });   // 0 => DOF off
el("disp-fd").onchange = () => { const val = el("disp-fd").value.trim(); if (val) postDisplay({ focus_distance: +val }); };
el("disp-save-framing").onclick = async () => {
  markDirty();   // a committed framing belongs to the project
  try { await api("/api/camera/save-framing", {}); } catch (e) { console.error(e); }
};
el("disp-reset").onclick = () => {
  el("disp-iso").value = 100; el("disp-iso-v").textContent = "100";
  el("disp-fl").value = 50; el("disp-fl-v").textContent = "50";
  el("disp-fs").value = 0; el("disp-fs-v").textContent = "off";
  el("disp-fd").value = "";
  // explicit nulls clear the active camera's optics override -> back to the authored, sharp camera
  postDisplay({ exposure: null, iso: null, focal_length: null, f_stop: null, focus_distance: null });
  // ...and drop its framing override -> snap back to the authored transform
  const cam = el("camera-select").value;
  if (cam) api("/api/camera/snap", { camera_path: cam, reset: true }).catch(console.error);
};
function scheduleStreamReconnect() {
  // the server rebuilds the streamer at the new size — drop our side and re-handshake
  setTimeout(() => {
    try { AppStreamer.terminate(); } catch (e) { /* ignore */ }
    connected = false; streamLive = false; setOverlay("Resizing stream…"); connectStream();
  }, 1800);
}
el("disp-res").onchange = () => {
  const [w, h] = el("disp-res").value.split("x").map(Number);
  el("grid-w").value = w; el("grid-h").value = h;   // pre-fill batch output size (editable)
  postDisplay({ resolution: [w, h] });
  scheduleStreamReconnect();
};
// --- eye-dropper: click a point on the model to set focus distance ---
// worldPositionM is dead in this ovrtx build, so the server resolves the picked prim and
// measures its bbox; here we just arm an overlay, send normalized click coords, and show the result.
let pickArmed = false;
let pickMode = "focus";   // "focus" (eye-dropper) | "pivot" (turntable)
function setPickArmed(on, mode) {
  pickArmed = on;
  if (mode) pickMode = mode;
  el("pick-overlay").classList.toggle("show", on);
  el("disp-pick").classList.toggle("armed", on && pickMode === "focus");
  el("tt-pick").classList.toggle("armed", on && pickMode === "pivot");
  if (on) setOverlay(pickMode === "pivot"
    ? "Click the asset to place the turntable pivot  ·  Esc to cancel"
    : "Click a point on the model to focus there  ·  Esc to cancel");
  else if ((overlay.textContent || "").startsWith("Click")) setOverlay("");
}
el("disp-pick").onclick = () => setPickArmed(!(pickArmed && pickMode === "focus"), "focus");
el("tt-pick").onclick = () => setPickArmed(!(pickArmed && pickMode === "pivot"), "pivot");
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && pickArmed) setPickArmed(false); });
el("pick-overlay").onclick = (ev) => {
  const r = el("remote-video").getBoundingClientRect();
  const nx = (ev.clientX - r.left) / r.width, ny = (ev.clientY - r.top) / r.height;
  const mode = pickMode;
  setPickArmed(false);
  if (!(nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1)) { setOverlay(""); return; }   // clicked the letterbox
  if (mode === "pivot") {
    setOverlay("Placing pivot…");
    api("/api/pick-point", { nx, ny }).then((j) => {
      if (j && j.ok) { setOverlay(""); ttSetPivot(j.point, j.size); }
      else { setOverlay((j && j.reason) || "Couldn't place the pivot"); setTimeout(() => setOverlay(""), 2500); }
    }).catch((e) => { console.error("pick-point", e); setOverlay(""); });
    return;
  }
  setOverlay("Focusing…");
  fetch("/api/pick-focus", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ nx, ny }) })
    .then((res) => res.json())
    .then((j) => {
      if (j && j.ok && j.distance != null) { el("disp-fd").value = Math.round(j.distance); setOverlay(""); }
      else { const msg = (j && j.reason) || "Couldn't focus there"; setOverlay(msg); setTimeout(() => { if (overlay.textContent === msg) setOverlay(""); }, 2500); }
    })
    .catch((e) => { console.error("pick-focus", e); setOverlay(""); });
};

function setHint(html) { const h = el("hint"); h.innerHTML = html; h.classList.add("show"); }
function clearHint() { el("hint").classList.remove("show"); }

// ---- turntable: pivot gizmo (screen-space projection of 3D axes = X-ray by construction:
// an overlay can never be occluded by scene geometry) ----
let ttPivot = null;
let gizmoTimer = null;
async function ttSetPivot(point, size) {
  ttPivot = point.slice();
  const ext = Math.max(size[0], size[1], size[2], 1);
  el("tt-step").value = Math.max(1, Math.round(ext / 40));
  if (!el("tt-radius").value) el("tt-radius").value = Math.round(ext * 1.6);
  el("tt-tools").style.display = "block";
  ttShowPivot();
  if (!gizmoTimer) gizmoTimer = setInterval(updateGizmo, 300);   // tracks live camera moves
  updateGizmo();
  // drop the viewer INTO the future turntable camera's frame 0 (suggested orbit around the
  // pivot, keeping the current approach direction) — Add is WYSIWYG from this moment on
  try {
    const j = await api("/api/camera/look-at", {
      target: ttPivot, radius: +el("tt-radius").value || Math.round(ext * 1.6),
      height: +el("tt-height").value || 0 });
    if (j && j.start_deg != null) ttStartDeg = j.start_deg;
  } catch (e) { /* ignore */ }
  setHint("<b>You are looking through the turntable camera's first frame.</b><br>" +
    "Drag the gizmo to move the pivot &middot; orbit/zoom until the view looks right &middot; " +
    "then <b>&#127909; Create camera from this view</b>.");
}
function ttShowPivot() {
  el("tt-pivot").textContent = ttPivot ? ttPivot.map((v) => v.toFixed(1)).join(", ") : "no pivot";
}
function ttHideGizmo() {
  clearHint();
  ttPivot = null;
  clearInterval(gizmoTimer); gizmoTimer = null;
  el("gizmo").style.display = "none";
  el("gizmo").innerHTML = "";   // drop the stale axes so a late/blocked redraw can't flash the old pivot
  el("tt-tools").style.display = "none";
  ttShowPivot();
}
// frames is in STAGE frames (the rig's timebase) — show the resulting seconds so nobody
// multiplies by the TIMELINE fps by mistake (240 @ 60fps stage = 4s/rev, at ANY output fps)
function ttFramesSeconds() {
  const f = +el("tt-frames").value || 0;
  const fps = (stage && stage.fps) || 0;
  // spell out the whole relationship so the stage fps (not the timeline fps) is the
  // obvious divisor: "240 frames ÷ 60 fps stage = 4.0 s / revolution"
  el("tt-frames-s").textContent = (f > 0 && fps > 0)
    ? `${f} frames ÷ ${(+fps).toFixed(fps % 1 ? 2 : 0)} fps stage = ${(f / fps).toFixed(1)} s / revolution`
    : "";
}
el("tt-frames").addEventListener("input", ttFramesSeconds);

function ttKeepGizmo(pv) {   // re-arm the pivot UI after buildPanels (which hides it):
  ttPivot = pv;              // Create promises "press again any time to update" — the
  el("tt-tools").style.display = "block";   // pivot must survive the camera's creation
  ttFramesSeconds();
  ttShowPivot();
  if (!gizmoTimer) gizmoTimer = setInterval(updateGizmo, 300);
  updateGizmo();
}

// the pivot lives IN the authored rig — after a reload or project load, restore the
// pivot gizmo + tools (and the authored frames count) instead of demanding a re-pick
function ttRestoreFromStage(info) {
  if (!info || !info.pivot || ttPivot) return;
  if (info.frames) el("tt-frames").value = info.frames;
  if (info.start_deg != null) ttStartDeg = info.start_deg;
  ttKeepGizmo(info.pivot.slice());
}
function configureActive() {
  return document.querySelector('.tab[data-pane="configure"]').classList.contains("active");
}
async function updateGizmo() {
  // serverStageReady gates the timer across server restarts: a relaunched server has no
  // stage yet, and hammering it with project/probe calls is pointless (it gets 400s).
  // The gizmo is a Configure-tab authoring tool — on other tabs it must hide (it floats
  // ABOVE the Results player, photobombing rendered videos) and stop its API chatter.
  if (!configureActive()) { el("gizmo").style.display = "none"; return; }
  if (!ttPivot || !stage || !serverStageReady) return;
  const L = (+el("tt-step").value || 5) * 4;
  const [x, y, z] = ttPivot;
  const pts = [[x, y, z], [x - L, y, z], [x + L, y, z], [x, y - L, z], [x, y + L, z], [x, y, z - L], [x, y, z + L]];
  try {
    const sc = (await api("/api/project", { points: pts })).screen;
    // the pivot may have been cleared (opening a new stage runs ttHideGizmo) or the tab
    // switched WHILE this projection was in flight — don't redraw a dead gizmo onto the
    // fresh stage (drawGizmo force-shows the overlay, so a late redraw strands it on screen)
    if (!ttPivot || !configureActive()) return;
    drawGizmo(sc);
    probeOcclusion(sc[0]);
  } catch (e) { /* ignore */ }
}
// shade the gizmo "submerged" when the pivot sits behind/inside geometry — the X-ray
// overlay otherwise hides exactly that information
let occBusy = false;
async function probeOcclusion(center) {
  if (occBusy || !ttPivot || !center || !center[2]) return;
  occBusy = true;
  try {
    const j = await api("/api/probe-occlusion", { point: ttPivot, nx: center[0], ny: center[1] });
    el("gizmo").classList.toggle("occluded", !!(j && j.ok && j.occluded));
  } catch (e) { /* ignore */ }
  finally { occBusy = false; }
}
let gizmoPx = null;   // last drawn pixel geometry: {c:[x,y], ends:[[x,y]x6 in +X,-X,+Y,-Y,+Z,-Z... ]}
function drawGizmo(sc) {
  const g = el("gizmo");
  if (!ttPivot) { g.style.display = "none"; return; }   // no pivot -> never show the overlay (covers late async redraws)
  const r = el("remote-video").getBoundingClientRect();
  const host = g.getBoundingClientRect();
  const X = (p) => (r.left - host.left) + p[0] * r.width;
  const Y = (p) => (r.top - host.top) + p[1] * r.height;
  const px = sc.map((p) => [X(p), Y(p), p[2]]);
  gizmoPx = px;
  const seg = (a, b, color, ax) =>
    `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#000" stroke-width="4" stroke-opacity="0.55"/>` +
    `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="${color}" stroke-width="2"/>` +
    `<line class="handle" data-ax="${ax}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}" stroke="#fff" stroke-opacity="0" stroke-width="14"/>`;
  let svg = "";
  if (px[1][2] || px[2][2]) svg += seg(px[1], px[2], "#ff5252", 0);
  if (px[3][2] || px[4][2]) svg += seg(px[3], px[4], "#76b900", 1);
  if (px[5][2] || px[6][2]) svg += seg(px[5], px[6], "#4f8cff", 2);
  if (px[0][2]) svg += `<circle cx="${px[0][0]}" cy="${px[0][1]}" r="5" fill="#fff" stroke="#000" stroke-width="2"/>` +
    `<circle class="handle dot" data-ax="screen" cx="${px[0][0]}" cy="${px[0][1]}" r="12" fill="#fff" fill-opacity="0"/>`;
  g.innerHTML = svg;
  g.style.display = "block";
}

// ---- direct manipulation: drag an axis handle to slide the pivot along that axis;
// drag the center dot to move it in the camera plane. World delta = mouse delta
// projected onto the axis's SCREEN direction, scaled by world-units-per-pixel taken
// from the live projection — no extra server round-trips during the drag.
let gizmoDrag = null;
el("gizmo").addEventListener("pointerdown", async (ev) => {
  const h = ev.target.closest(".handle");
  if (!h || !ttPivot || !gizmoPx) return;
  ev.preventDefault(); ev.stopPropagation();
  const L = (+el("tt-step").value || 5) * 4;
  const c = gizmoPx[0];
  const mkAxis = (endIdx, world) => {
    const e = gizmoPx[endIdx];
    const d = [e[0] - c[0], e[1] - c[1]];
    const len = Math.hypot(d[0], d[1]);
    if (len < 8) return null;                       // axis end-on to the camera: degenerate
    return { dir: [d[0] / len, d[1] / len], wpp: L / len, world };
  };
  let axes = [];
  if (h.dataset.ax === "screen") {
    try {
      const pose = await (await fetch("/api/camera-pose")).json();
      const pts = [ttPivot,
        [ttPivot[0] + pose.right[0] * L, ttPivot[1] + pose.right[1] * L, ttPivot[2] + pose.right[2] * L],
        [ttPivot[0] + pose.up[0] * L, ttPivot[1] + pose.up[1] * L, ttPivot[2] + pose.up[2] * L]];
      const sc = (await api("/api/project", { points: pts })).screen;
      const r = el("remote-video").getBoundingClientRect();
      const host = el("gizmo").getBoundingClientRect();
      const P = (p) => [(r.left - host.left) + p[0] * r.width, (r.top - host.top) + p[1] * r.height];
      const cc = P(sc[0]), rr = P(sc[1]), uu = P(sc[2]);
      const mk = (e, world) => {
        const d = [e[0] - cc[0], e[1] - cc[1]];
        const len = Math.hypot(d[0], d[1]);
        return len < 8 ? null : { dir: [d[0] / len, d[1] / len], wpp: L / len, world };
      };
      axes = [mk(rr, pose.right), mk(uu, pose.up)].filter(Boolean);
    } catch (e) { return; }
  } else {
    const ax = +h.dataset.ax;
    const world = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][ax];
    const a = mkAxis([2, 4, 6][ax], world);
    if (!a) return;
    axes = [a];
  }
  gizmoDrag = { axes, start: [ev.clientX, ev.clientY], pivot0: ttPivot.slice() };
  el("gizmo").classList.add("dragging");
  el("gizmo").setPointerCapture(ev.pointerId);
});
el("gizmo").addEventListener("pointermove", (ev) => {
  if (!gizmoDrag) return;
  const dx = ev.clientX - gizmoDrag.start[0], dy = ev.clientY - gizmoDrag.start[1];
  const p = gizmoDrag.pivot0.slice();
  for (const a of gizmoDrag.axes) {
    const along = dx * a.dir[0] + dy * a.dir[1];    // mouse travel along the axis on screen
    const t = along * a.wpp;                        // -> world units
    p[0] += a.world[0] * t; p[1] += a.world[1] * t; p[2] += a.world[2] * t;
  }
  ttPivot = p;
  ttShowPivot();
  if (!updateGizmo._busy) {                         // ~live re-projection, single-flight
    updateGizmo._busy = true;
    updateGizmo().finally(() => { updateGizmo._busy = false; });
  }
});
const endGizmoDrag = () => {
  if (!gizmoDrag) return;
  gizmoDrag = null;
  el("gizmo").classList.remove("dragging");
  updateGizmo();
};
el("gizmo").addEventListener("pointerup", endGizmoDrag);
el("gizmo").addEventListener("pointercancel", endGizmoDrag);
document.querySelectorAll(".tt-nudge button[data-ax]").forEach((b) => {
  b.onclick = () => {
    if (!ttPivot) return;
    ttPivot[+b.dataset.ax] += (+b.dataset.d) * (+el("tt-step").value || 1);
    ttShowPivot();
    updateGizmo();
  };
});
let ttStartDeg = 0;   // spin start azimuth (frame 0 = the view at Create time)
// derive the orbit (radius / height / start angle) from the CURRENT live view
let ttCamWorld = null;   // full 16-float pose captured at create time (exact framing)
async function ttOrbitFromView() {
  const pose = await (await fetch("/api/camera-pose")).json();
  ttCamWorld = pose.m || null;
  const off = [pose.eye[0] - ttPivot[0], pose.eye[1] - ttPivot[1], pose.eye[2] - ttPivot[2]];
  const zUp = (stage && stage.up_axis === "Z");
  const radius = zUp ? Math.hypot(off[0], off[1]) : Math.hypot(off[0], off[2]);
  const height = zUp ? off[2] : off[1];
  ttStartDeg = (zUp ? Math.atan2(off[1], off[0]) : Math.atan2(off[0], off[2])) * 180 / Math.PI;
  el("tt-radius").value = Math.round(radius);
  el("tt-height").value = Math.round(height);
  return { radius, height };
}
let ttPlaying = false;
function ttStopPreview() {
  if (!ttPlaying) return;
  ttPlaying = false;
  el("tt-preview").innerHTML = "&#9654; Preview spin";
  fetch("/api/playback", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ playing: false }) }).catch(() => {});
}
el("tt-preview").onclick = async () => {
  if (ttPlaying) { ttStopPreview(); return; }
  // WYSIWYG: the spin previews the camera AS IT WILL RENDER — re-author it from the
  // current view + fields first, so reframing (pan offsets included) and a changed
  // frames count take effect immediately instead of silently previewing stale state
  if (ttPivot) {
    el("tt-preview").disabled = true;
    const ok = await ttCreate(true);
    el("tt-preview").disabled = false;
    if (!ok) return;
  }
  ttPlaying = true;
  el("tt-preview").innerHTML = "&#9632; Stop preview";
  api("/api/playback", { playing: true }).catch(() => {});
};
el("tt-remove").onclick = async () => {
  el("tt-msg").textContent = "removing turntable camera…";
  try {
    const resp = await api("/api/turntable/remove", {});
    stage = resp;
    buildPanels(Object.values(selection));
    if (resp.camera) el("camera-select").value = resp.camera;
    markDirty();
    ttHideGizmo();
    el("tt-msg").textContent = "turntable camera removed";
  } catch (e) { el("tt-msg").textContent = "remove failed: " + e.message; }
};
async function ttCreate(quiet) {
  // author (or re-author) the turntable camera from the CURRENT view + fields.
  // Returns true on success. quiet = no hint churn (preview auto-updates silently).
  if (!ttPivot) { el("tt-msg").textContent = "pick a pivot first"; return false; }
  try { await ttOrbitFromView(); }                 // the view you SEE becomes frame 0
  catch (e) { el("tt-msg").textContent = "couldn't read the camera pose"; return false; }
  const body = { pivot: ttPivot, radius: +el("tt-radius").value || 100,
                 height: +el("tt-height").value || 0, frames: +el("tt-frames").value || 120,
                 start_deg: ttStartDeg, focal_length: +el("disp-fl").value || 35,
                 camera_world: ttCamWorld };       // EXACT pose: pan offsets preserved
  const existed = stage && stage.cameras && stage.cameras.some((c) => c.path === TT_CAM);
  el("tt-msg").textContent = quiet ? "updating turntable camera…"
    : (existed ? "updating camera from this view…" : "creating camera from this view…");
  const pv = ttPivot.slice();   // buildPanels hides the pivot UI — keep it: Create is repeatable
  try {
    const resp = await api("/api/turntable", body);
    stage = resp;
    buildPanels(Object.values(selection));
    if (resp.camera) el("camera-select").value = resp.camera;
    markDirty();
    ttKeepGizmo(pv);
    if (!quiet) {
      setHint("<b>Turntable camera added.</b> Preview spin plays it exactly as it will render — " +
        "raise <b>frames</b> for a slower revolution. Nudge the pivot or reframe, then " +
        "<b>Create</b> (or just Preview) to update. Render via <b>Grid + Animation range</b>.");
      setTimeout(clearHint, 12000);
      el("tt-msg").textContent = (existed ? "turntable camera updated" :
        "turntable camera added") + " — use Grid + Animation range to render the revolution";
    } else {
      el("tt-msg").textContent = "";
    }
    return true;
  } catch (e) { el("tt-msg").textContent = "failed: " + e.message; return false; }
}
el("tt-add").onclick = () => ttCreate(false);

function gatherDisplay() {
  const [w, h] = el("disp-res").value.split("x").map(Number);
  const fd = el("disp-fd").value.trim();
  return { resolution: [w, h], iso: +el("disp-iso").value, focal_length: +el("disp-fl").value,
           f_stop: +el("disp-fs").value, focus_distance: fd ? +fd : null };
}
function applyDisplay(d) {   // restore display controls + push to the renderer (project open)
  if (!d) return;
  // a REAL resolution change rebuilds the server-side streamer (drops WebRTC) — schedule the
  // same reconnect the manual dropdown does, or a project open strands at "Reconnecting…".
  // (same-size posts are a server-side no-op and need nothing.)
  const resChanged = d.resolution && el("disp-res").value !== d.resolution.join("x");
  if (d.resolution) el("disp-res").value = d.resolution.join("x");
  if (d.iso != null) { el("disp-iso").value = d.iso; el("disp-iso-v").textContent = d.iso; }
  if (d.focal_length != null) { el("disp-fl").value = d.focal_length; el("disp-fl-v").textContent = d.focal_length; }
  if (d.f_stop != null) { el("disp-fs").value = d.f_stop; el("disp-fs-v").textContent = (+d.f_stop) > 0 ? (+d.f_stop).toFixed(1) : "off"; }
  if (d.focus_distance != null) el("disp-fd").value = d.focus_distance;
  postDisplay(d);
  if (resChanged) scheduleStreamReconnect();
}
// Repopulate the Display sliders from a camera's look (absent fields -> defaults). NO posting —
// the server emits camera_params whenever the active camera (or its look) changes, so switching
// cameras shows that camera's optics.
function setDisplayControls(look) {
  look = look || {};
  const iso = look.iso != null ? look.iso : 100;
  el("disp-iso").value = iso; el("disp-iso-v").textContent = iso;
  const fl = look.focal_length != null ? look.focal_length : 50;
  el("disp-fl").value = fl; el("disp-fl-v").textContent = fl;
  const fs = look.f_stop != null ? look.f_stop : 0;
  el("disp-fs").value = fs; el("disp-fs-v").textContent = (+fs) > 0 ? (+fs).toFixed(1) : "off";
  el("disp-fd").value = look.focus_distance != null ? Math.round(look.focus_distance) : "";
}

// ---- events WS (runtime -> UI) ----
function connectEvents() {
  const ws = new WebSocket(`ws://${toIPv4(location.host)}/events`);
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === "warmup") { setStatus("warming", "warming up"); if (!streamLive) setOverlay("Warming up — compiling shaders…"); }
    else if (e.type === "ready") {
      // first frame of a newly opened stage — if our stream is already attached, the open
      // is DONE (an open with the stream connected otherwise leaves "opening" stuck forever)
      serverStageReady = true;
      stopWarming();
      if (streamLive) { setStatus("live", "live"); setOverlay(""); }
    }
    else if (e.type === "stage_open") { serverStageReady = false; if (streamLive) showWarming(); }
    else if (e.type === "resolution") {   // streamer rebuilt: frameless until reopen
      serverStageReady = false;
      // the server is the source of truth for stream size (a restored session may differ
      // from the UI) - sync the select + batch defaults so renders inherit the real size.
      // grid-w/grid-h sync every time (not only on a disp-res mismatch) — see attachIfOpen.
      if (e.width) {
        if (el("disp-res").value !== `${e.width}x${e.height}`) el("disp-res").value = `${e.width}x${e.height}`;
        el("grid-w").value = e.width; el("grid-h").value = e.height;
      }
    }
    else if (e.type === "classified") { variantSwatches = e.swatches || {}; applySwatches(); }
    else if (e.type === "batch_progress") { onBatchProgress(e); }
    else if (e.type === "batch_done") { onBatchDone(e); }
    else if (e.type === "timeline_progress") { onTimelineProgress(e); }
    else if (e.type === "timeline_done") { onTimelineDone(e); }
    else if (e.type === "mirror_progress") { setOverlay(`Downloading stage… ${e.downloaded} files  ·  ${e.file || ""}`); }
    else if (e.type === "framing_saved") { setOverlay("Framing saved"); setTimeout(() => { if (overlay.textContent === "Framing saved") setOverlay(""); }, 1200); refreshCameraBadges(); }
    else if (e.type === "framing_skipped") { setOverlay(e.message); setTimeout(() => { if (overlay.textContent === e.message) setOverlay(""); }, 3500); }
    else if (e.type === "camera_params") { setDisplayControls(e.params); refreshCameraBadges(); }   // switched camera / look changed
    else if (e.type === "focus_picked") { if (e.distance != null) { el("disp-fd").value = Math.round(e.distance); markDirty(); } }
    else if (e.type === "error") { setStatus("error", "error"); console.error("runtime:", e.message); }
  };
  ws.onclose = () => setTimeout(connectEvents, 1500);
}

// ---- ovstream WebRTC video ----
let connecting = false;   // single-flight: recovery + resolution-reconnect must not race
async function connectStream() {
  if (connecting) return;
  connecting = true;
  setStatus("warming", "connecting");
  setOverlay("Connecting to live viewport…");
  // evict any half-open ghost before EVERY handshake: the server is single-client, and
  // whenever we are connecting our previous session is dead-or-dying by definition —
  // page reload, resolution rebuild, recovery, all of them can leave the slot wedged
  // (signaling lands, media never flows -> black viewport). Costs ~1.5s per connect.
  try {
    // the route replies only AFTER the rebuild executed on the render thread — no
    // fixed sleep to lose against a slow warmup drain
    await fetch("/api/stream/restart", { method: "POST" });
    await new Promise((r) => setTimeout(r, 250));   // let the fresh server start listening
  } catch (e) { /* server briefly down — the normal connect path + recovery handle it */ }
  await loadConfig();  // ensure we use the actual (possibly auto-shifted) signaling port
  clearTimeout(connectWatch);
  connectWatch = setTimeout(() => {
    if (connected) return;
    // never strand in a half-open connect: the library can hang in signaling/ICE forever
    // (no success, no error) — tear it down and let the recovery loop keep trying
    connecting = false;
    try { AppStreamer.terminate(); } catch (e) { /* ignore */ }
    scheduleStreamRecovery("connect timed out — if another tab holds the stream, close it");
  }, 15000);
  try {
    await AppStreamer.connect({
      streamSource: StreamType.DIRECT,
      logLevel: 2,
      streamConfig: {
        videoElementId: "remote-video",
        server: toIPv4(location.hostname) || "127.0.0.1",
        signalingPort: signalPort,
        fps: 60,
        maxReconnects: 5,
        onStart: (msg) => {
          if (msg.action !== "start") return;
          if (msg.status === "success") {
            clearTimeout(connectWatch);
            connecting = false;
            connected = true; streamLive = true;
            lastVideoFrame = performance.now();   // frame-flow watchdog baseline
            if (serverStageReady) { setStatus("live", "live"); setOverlay(""); }
            else { setStatus("warming", "warming up"); showWarming(); }   // black-but-working
            // keep MUTED + play(): Chrome blocks autoplay of unmuted media (no user gesture),
            // which left the <video> paused on frame 0 = black viewport with only the gizmo
            // overlay showing. The stream is video-only, so muted has no downside.
            const v = el("remote-video"); if (v) { v.muted = true; v.play().catch(() => {}); v.focus(); }
          } else if (msg.status === "error") {
            connecting = false;
            scheduleStreamRecovery(msg.info || "stream error");
          }
        },
        onStop: () => {
          connecting = false; connected = false; streamLive = false;
          scheduleStreamRecovery("stream stopped");
        },
        onUpdate: () => {},
      },
    });
  } catch (e) {
    connecting = false;
    scheduleStreamRecovery(e.message || "connect failed");
  }
}

// ====================== Grid / batch permutation rendering + Results ======================
let gridMode = "one_at_a_time";       // matrix mode
let gridQmode = "RealTimePathTracing"; // batch render mode
let included = {};                    // set_name -> Set(variant) cherry-pick
let batchRunning = false;
let lastPerms = 0;                    // permutation count (what the 500 guard checks)
let results = [];                     // [{name,label,frames,frame_count,first_frame}]

// ---- tabs ----
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    const pane = t.dataset.pane;
    document.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.id === `pane-${pane}`));
    renderResultsMedia();   // show the still/video overlay only on the Results tab
    if (pane !== "configure") el("gizmo").style.display = "none";   // instant, timer re-shows on return
    const onTimeline = pane === "timeline" && !!stage;
    el("timeline-strip").classList.toggle("show", onTimeline);
    if (onTimeline) { tlBaseSel = tlBaseFromLive(); renderTimeline(); }   // snapshot live look as scrub base
    if (pane === "configure" && stage && tlTouchedLive) {
      // timeline scrubbing drove the viewport (variants + camera cuts) — coming back to
      // Configure, snap the live view back to what THIS tab says is selected.
      tlTouchedLive = false; tlLastCamera = null;
      api("/api/variant", { selections: Object.values(selection) }).catch(() => {});
      const cam = el("camera-select").value;
      if (cam) api("/api/camera/snap", { camera_path: cam }).catch(() => {});
    }
  };
});
document.querySelectorAll(".grid-mode-btn").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".grid-mode-btn").forEach((x) => x.classList.toggle("active", x === b));
  gridMode = b.dataset.mode; updateGridCount();
});
document.querySelectorAll(".grid-q-btn").forEach((b) => b.onclick = () => {
  document.querySelectorAll(".grid-q-btn").forEach((x) => x.classList.toggle("active", x === b));
  gridQmode = b.dataset.mode;
});

// ---- grid panel build (mirrors stage; cherry-pick include + per-variant) ----
function buildGridPanel() {
  included = {};
  const cams = el("grid-cameras"); cams.innerHTML = "";
  // Default-tick the camera being previewed in Configure (WYSIWYG), falling back to the first
  // authored one; still fully editable below. Read it from stage.camera (the server's active
  // camera), not from the <select>: buildCameras() has just reset that to its first option and
  // it is not restored to the real selection until after buildPanels runs.
  const selCam = (stage && stage.camera) || el("camera-select").value || "";
  const camList = stage.cameras || [];
  const selMatches = selCam && camList.some((c) => c.path === selCam);
  camList.forEach((c, i) => {
    const lab = document.createElement("label"); lab.className = "check";
    const cb = document.createElement("input"); cb.type = "checkbox"; cb.value = c.path;
    cb.checked = selMatches ? (c.path === selCam) : (i === 0);
    cb.onchange = updateGridCount;
    lab.append(cb, document.createTextNode(" " + c.name));
    cams.appendChild(lab);
  });
  const root = el("grid-sets"); root.innerHTML = "";
  stage.variant_sets.forEach((vs) => {
    const card = document.createElement("div"); card.className = "vcard";
    const head = document.createElement("label"); head.className = "check setinc";
    const inc = document.createElement("input"); inc.type = "checkbox";
    const nm = document.createElement("span"); nm.className = "name"; nm.textContent = " " + vs.set_name;
    head.append(inc, nm);
    const chips = document.createElement("div"); chips.className = "chips";
    vs.variants.forEach((vname) => {
      const chip = document.createElement("span"); chip.className = "chip";
      chip.dataset.sname = vs.set_name; chip.dataset.variant = vname;
      const dot = document.createElement("span"); dot.className = "swatch";
      const lb = document.createElement("span"); lb.textContent = vname;
      chip.append(dot, lb);
      chip.onclick = () => {
        const set = included[vs.set_name];
        if (!set) return;                       // ignore until the set is included
        if (set.has(vname)) set.delete(vname); else set.add(vname);
        chip.classList.toggle("on", set.has(vname));
        updateGridCount();
      };
      chips.appendChild(chip);
    });
    inc.onchange = () => {
      if (inc.checked) included[vs.set_name] = new Set(vs.variants);  // default: sweep all
      else delete included[vs.set_name];
      card.classList.toggle("included", inc.checked);
      [...chips.children].forEach((c) => c.classList.toggle("on", inc.checked));
      updateGridCount();
    };
    card.append(head, chips);
    root.appendChild(card);
  });
  el("grid-fstart").value = Math.round(stage.start_time || 0);   // default to the stage span
  el("grid-fend").value = Math.round(stage.end_time || 0);
  applySwatches();
  updateGridCount();
}

function selectedCameras() {
  return [...document.querySelectorAll("#grid-cameras input:checked")].map((c) => c.value);
}
function includedMap() {
  return Object.fromEntries(
    Object.entries(included).map(([k, s]) => [k, [...s]]).filter(([, a]) => a.length));
}
function fmtTime(s) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}
function updateGridCount() {
  const counts = Object.values(includedMap()).map((a) => a.length);
  let perms = 0;
  if (counts.length) perms = gridMode === "full_cartesian"
    ? counts.reduce((a, b) => a * b, 1) : counts.reduce((a, b) => a + b, 0);
  lastPerms = perms;
  const cams = Math.max(1, selectedCameras().length);
  let frames = 1;
  if (el("grid-anim").checked) {
    const fs = parseInt(el("grid-fstart").value) || 0;
    const fe = parseInt(el("grid-fend").value) || 0;
    const st = Math.max(1, parseInt(el("grid-fstep").value) || 1);
    frames = Math.max(1, Math.floor((fe - fs) / st) + 1);
  }
  const total = perms * cams * frames;   // actual render count (frames/cams included)
  el("grid-count").textContent = total;
  el("grid-estimate").textContent = total
    ? `${perms} perm${perms > 1 ? "s" : ""} → ${total} render${total > 1 ? "s" : ""} • ~${fmtTime(total * 3)} (rough)`
      + (perms > 500 ? " • exceeds 500-perm guard" : "")
    : "include one or more sets to sweep";
  el("grid-render").disabled = !total || batchRunning;
}

// ---- run / cancel ----
async function runBatch(confirm = false) {
  const out_dir = el("grid-out").value.trim();
  const cameras = selectedCameras();
  if (!out_dir) { el("grid-status").textContent = "set an output folder"; return; }
  if (!cameras.length) { el("grid-status").textContent = "select at least one camera"; return; }
  if (!confirm && lastPerms > 500) {   // confirm up front; avoids a scary 409 round-trip
    if (!window.confirm(`${lastPerms} permutations exceeds the 500-permutation guard. Render anyway?`)) {
      el("grid-status").textContent = "cancelled (over guard)"; return;
    }
    confirm = true;
  }
  // belt-and-suspenders: if a W/H field is ever blank, fall back to the SHOWN display
  // resolution, never a hard-coded literal that silently changes the aspect.
  const [dispW, dispH] = el("disp-res").value.split("x").map(Number);
  const job = {
    mode: gridMode, base_selection: [], included: includedMap(), cameras,
    quality: {
      mode: gridQmode, samples_per_pixel: parseInt(el("grid-spp").value) || 64, max_bounces: 4,
      resolution: [parseInt(el("grid-w").value) || dispW || 1280, parseInt(el("grid-h").value) || dispH || 720],
    },
    frame_mode: el("grid-anim").checked ? "animation_range" : "single",
    frame_start: parseInt(el("grid-fstart").value),
    frame_end: parseInt(el("grid-fend").value),
    frame_step: Math.max(1, parseInt(el("grid-fstep").value) || 1),
    out_dir, confirm,
  };
  let r;
  try {
    r = await fetch("/api/batch", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ job }),
    });
  } catch (e) { el("grid-status").textContent = "batch request failed: " + e.message; return; }
  if (r.status === 409) {
    if (window.confirm(`This batch exceeds the 500-permutation guard. Render anyway?`)) return runBatch(true);
    el("grid-status").textContent = "cancelled (over guard)"; return;
  }
  if (!r.ok) { el("grid-status").textContent = "batch error: " + (await r.text()); return; }
  const { count } = await r.json();
  batchRunning = true;
  el("grid-render").disabled = true; el("grid-cancel").disabled = false;
  el("grid-bar").style.width = "0%";
  el("grid-status").textContent = `rendering ${count} permutation${count > 1 ? "s" : ""}…`;
  el("results-dir").value = out_dir;   // pre-fill the Results tab
  setStatus("warming", "batch");
}
async function cancelBatch() {
  try { await fetch("/api/batch/cancel", { method: "POST" }); } catch (e) { /* ignore */ }
  el("grid-status").textContent = "cancelling after the current permutation…";
}
function onBatchProgress(e) {
  const pct = e.total ? Math.round((e.done / e.total) * 100) : 0;
  el("grid-bar").style.width = pct + "%";
  el("grid-status").textContent = `${e.done}/${e.total} — ${e.name} (${e.phase})`;
}
function onBatchDone(e) {
  batchRunning = false;
  el("grid-cancel").disabled = true;
  el("grid-bar").style.width = "100%";
  el("grid-status").textContent = e.cancelled
    ? `cancelled — ${e.done}/${e.total} rendered → ${e.out_dir}`
    : `done — ${e.done} rendered → ${e.out_dir}`;
  updateGridCount();
  if (el("results-dir").value.trim() === e.out_dir) refreshResults();
}
el("grid-render").onclick = () => runBatch(false);
el("grid-cancel").onclick = cancelBatch;
["grid-w", "grid-h", "grid-fstart", "grid-fend", "grid-fstep"].forEach((id) => {
  el(id).oninput = updateGridCount; el(id).onchange = updateGridCount;
});
el("grid-anim").onchange = () => {
  el("grid-frame-range").style.display = el("grid-anim").checked ? "flex" : "none";
  updateGridCount();
};

// ---- results browser ----
let curResult = -1;
let mediaBust = 0;   // bump on refresh so overlaid/encoded files reload past the browser cache

function resultsActive() {
  return document.querySelector('.tab[data-pane="results"]').classList.contains("active");
}
function frameUrl(path) { return `/api/frame?path=${encodeURIComponent(path)}&t=${mediaBust}`; }
function videoUrl(path) { return `/api/video?path=${encodeURIComponent(path)}&t=${mediaBust}`; }

async function refreshResults() {
  const dir = el("results-dir").value.trim();
  if (!dir) return;
  let j;
  try { j = await (await fetch(`/api/results?dir=${encodeURIComponent(dir)}`)).json(); }
  catch (e) { el("results-empty").textContent = "results fetch failed: " + e.message; return; }
  mediaBust++;
  results = j.permutations || [];
  const sel = el("results-select"); sel.innerHTML = "";
  results.forEach((p, i) => {
    const o = document.createElement("option");
    const tag = p.video ? " ▸" : (p.frame_count > 1 ? `  (${p.frame_count}f)` : "");
    o.value = i; o.textContent = p.label + tag;
    sel.appendChild(o);
  });
  el("results-empty").style.display = results.length ? "none" : "block";
  if (!results.length) el("results-empty").textContent = "nothing rendered in that folder yet.";
  // grey out Overlay when the folder has nothing it applies to
  // (a timeline render is a single mp4 — no frame folders / stills to label)
  const hasImages = results.some((p) => p.frame_count > 0);
  el("post-overlay").disabled = !hasImages;
  el("post-cutsheet").disabled = !hasImages;
  if (results.length) showResult(Math.min(curResult < 0 ? 0 : curResult, results.length - 1));
  else { curResult = -1; renderResultsMedia(); }
}
function showResult(i) {
  curResult = i;
  const p = results[i]; if (!p) return;
  el("results-select").value = i;
  const hasVideo = !!p.video;
  const multi = p.frame_count > 1 && !hasVideo;
  el("results-frame-controls").style.display = multi ? "flex" : "none";
  if (hasVideo) {
    el("results-video").src = videoUrl(p.video);
  } else {
    if (multi) {
      const sl = el("results-slider"); sl.max = p.frame_count - 1; sl.value = 0;
      el("results-frame-label").textContent = `1 / ${p.frame_count}`;
    }
    el("results-img").src = frameUrl(p.frames[0] || p.first_frame);
  }
  renderResultsMedia();
}
function showFrame(fi) {
  const p = results[curResult]; if (!p) return;
  el("results-img").src = frameUrl(p.frames[fi] || p.first_frame);
  renderResultsMedia();
}
// Show the still OR the video, only on the Results tab; never both, never over other tabs.
function renderResultsMedia() {
  const p = results[curResult];
  const active = resultsActive() && !!p;
  const hasVideo = active && !!p.video;
  el("results-video").classList.toggle("show", hasVideo);
  el("results-img").classList.toggle("show", active && !hasVideo);
  if (!hasVideo) { try { el("results-video").pause(); } catch (e) { /* ignore */ } }
}
async function browseFolder(inputId, after) {
  if (browseFolder._busy) return;   // one native folder dialog at a time (single-flight)
  browseFolder._busy = true;
  try {
    const j = await (await fetch("/api/browse-folder", { method: "POST" })).json();
    if (j.path) { el(inputId).value = j.path; if (after) after(); }
  } catch (e) { console.warn("browse failed", e); }
  finally { browseFolder._busy = false; }
}
el("results-browse").onclick = () => browseFolder("results-dir", refreshResults);
el("grid-browse").onclick = () => browseFolder("grid-out");
el("tl-browse").onclick = () => browseFolder("tl-out");
el("results-refresh").onclick = refreshResults;
el("results-select").onchange = (e) => showResult(parseInt(e.target.value) || 0);
el("results-slider").oninput = (e) => {
  const fi = parseInt(e.target.value) || 0;
  el("results-frame-label").textContent = `${fi + 1} / ${results[curResult].frame_count}`;
  showFrame(fi);
};

// ---- post-processing ----
async function postOp(path, body, label) {
  const btns = ["post-overlay", "post-cutsheet"].map(el);
  btns.forEach((b) => (b.disabled = true));
  el("post-log").textContent = label + "…";
  try {
    return await (await fetch(path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    })).json();
  } catch (e) { el("post-log").textContent = label + " failed: " + e.message; return null; }
  finally { btns.forEach((b) => (b.disabled = false)); }
}
el("post-overlay").onclick = async () => {
  const dir = el("results-dir").value.trim();
  if (!dir) { el("post-log").textContent = "set the results folder first"; return; }
  const j = await postOp("/api/post/overlay", { out_dir: dir }, "overlaying labels");
  if (j) { el("post-log").textContent = `labeled ${j.count} render(s) — copies in the _labeled folder`; refreshResults(); }
};
el("post-cutsheet").onclick = async () => {
  const dir = el("results-dir").value.trim();
  if (!dir) { el("post-log").textContent = "set the results folder first"; return; }
  const j = await postOp("/api/post/cutsheet", { out_dir: dir }, "composing cut sheet");
  if (j && j.path) {
    el("post-log").textContent = "cut sheet ready";
    await refreshResults();
    const i = results.findIndex((p) => p.name === "cutsheet");
    if (i >= 0) showResult(i);                       // straight into the viewport
  } else if (j) { el("post-log").textContent = "no rendered images in that folder yet"; }
};
// (no → MP4 button / fps field: animation batches assemble each sequence's mp4 at render
// time at the KNOWN rate — stage fps / frame step — and Overlay re-encodes a labeled
// folder's sibling mp4 automatically. /api/post/video remains for API use.)
// (no Compress button: every video we emit is already libx264/crf-23 — a fixed-bitrate
// re-encode only loses quality. compress_video + /api/post/compress remain for API use.)

// ====================== Timeline: multi-track non-linear editor ======================
const TL = window.TL;
const PPS = 70;        // pixels per second
let LABEL_W = 230;     // sticky track-label gutter width (drag-resizable)
let tl = { duration_s: 0, fps: 30, tracks: [] };
let tlBaseSel = [];    // stable base snapshot (untracked sets) for scrub — NOT the mutating live
let tlClipS = 2.0;
let tlLastCamera = null;
let tlTouchedLive = false;   // a scrub/edit pushed timeline state to the viewport
let tlPlayT = 0;       // current playhead time (for "append at playhead")
let appendAtPlayhead = false;
let scrubTimer = null;
let tlAnim = null;        // requestAnimationFrame handle while playing (null = paused)
let tlPlayStartWall = 0;  // performance.now() at play start (wall-clock origin)
let tlPlayStartT = 0;     // playhead time at play start
let tlLoop = false;       // loop toggle
let tlLastPost = 0;       // last applyStateAt wall-clock (playback throttle)
let tlRendering = false;  // an MP4 render is in progress -> transport disabled
let drag = null;
let selClip = null;    // { track, clip } object refs of the selected clip (drag highlight)

function tlBaseFromLive() {
  return Object.values(selection).map((c) => ({ prim_path: c.prim_path, set_name: c.set_name, variant: c.variant }));
}
function tlSetInfo(name) { return (stage.variant_sets || []).find((v) => v.set_name === name); }
function tlVariantsFor(track) {
  if (track.kind === "camera") return (stage.cameras || []).map((c) => c.path);
  const si = tlSetInfo(track.set_name); return si ? si.variants : [];
}
function tlRecalcDuration() {
  let end = 0;
  for (const t of tl.tracks) for (const c of t.clips) end = Math.max(end, c.start_s + c.duration_s);
  tl.duration_s = end;
}
function tlContentWidth() { return LABEL_W + Math.max(tl.duration_s, 4) * PPS + 80; }

function buildTimelinePanel() {
  // pre-populate a lane per variant set + a camera track. PRESERVE an existing timeline across
  // a SAME-stage rebuild (authoring/updating the turntable camera re-runs buildPanels) — only
  // re-initialize when the stage's variant sets actually change, so tuning the camera in
  // Configure never wipes your clips (and Play keeps working on the kept timeline).
  if (!TL.tracksMatchStage(tl, stage.variant_sets)) {
    tl = { duration_s: 0, fps: parseInt(el("tl-fps").value) || 30, tracks: [
      { kind: "camera", set_name: null, prim_path: null, clips: [] },
      ...(stage.variant_sets || []).map((vs) => (
        { kind: "variant_set", set_name: vs.set_name, prim_path: vs.prim_path, clips: [] })),
    ] };
    tlBaseSel = tlBaseFromLive();
    tlLastCamera = null;
    selClip = null;
  }
  renderTimeline();
  loadViews();
  loadProjects();
}

function renderTimeline() {
  tl.fps = parseInt(el("tl-fps").value) || 30;
  tlClipS = parseFloat(el("tl-clip").value) || 2.0;
  tlRecalcDuration();
  const W = tlContentWidth();
  const ruler = el("tl-ruler"); ruler.innerHTML = ""; ruler.style.width = W + "px";
  for (let s = 0; s <= Math.ceil(Math.max(tl.duration_s, 4)); s++) {
    const tick = document.createElement("div"); tick.className = "tl-tick";
    tick.style.left = (LABEL_W + s * PPS) + "px"; tick.textContent = s + "s";
    ruler.appendChild(tick);
  }
  const root = el("tl-tracks"); root.innerHTML = "";
  let visible = 0;
  tl.tracks.forEach((track, ti) => {
    if (track.hidden) return;
    visible++;
    const row = document.createElement("div"); row.className = "tl-track"; row.style.width = W + "px";
    const label = document.createElement("div"); label.className = "tl-track-label";
    const hide = document.createElement("button"); hide.className = "tl-hide"; hide.textContent = "−";
    hide.title = "hide this track"; hide.onclick = () => { track.hidden = true; markDirty(); renderTimeline(); };
    const name = document.createElement("span"); name.className = "name";
    name.textContent = track.kind === "camera" ? "Camera" : track.set_name;
    const vsel = document.createElement("select");
    tlVariantsFor(track).forEach((val) => {
      const o = document.createElement("option"); o.value = val;
      o.textContent = track.kind === "camera" ? val.split("/").pop() : val; vsel.appendChild(o);
    });
    const add = document.createElement("button"); add.className = "tl-add-clip"; add.textContent = "Append";
    add.title = "append this variant as a clip"; add.onclick = () => addClip(ti, vsel.value);
    label.append(hide, name, vsel, add);
    row.appendChild(label);
    track.clips.forEach((clip, ci) => row.appendChild(makeClipEl(ti, ci, clip, track)));
    root.appendChild(row);
  });
  // "show hidden…" picker
  const show = el("tl-show-track");
  show.innerHTML = '<option value="">show hidden…</option>';
  tl.tracks.forEach((track, ti) => {
    if (!track.hidden) return;
    const o = document.createElement("option"); o.value = ti;
    o.textContent = track.kind === "camera" ? "Camera" : track.set_name; show.appendChild(o);
  });
  el("tl-dur").textContent = tl.duration_s.toFixed(1) + "s";
  el("tl-del-clip").disabled = !selClip;   // delete button live-tracks the selection
  syncRuler();
  // any edit (move/resize/reorder/retarget/add) refreshes the viewport at the playhead
  if (stage && timelineActive() && tlBaseSel.length) postStateAt(tlPlayT);
}

function makeClipEl(ti, ci, clip, track) {
  const div = document.createElement("div");
  div.className = "tl-clip" + (track.kind === "camera" ? " cam" : "") + (selClip && selClip.clip === clip ? " sel" : "");
  div.style.left = (LABEL_W + clip.start_s * PPS) + "px";
  div.style.width = Math.max(clip.duration_s * PPS, 30) + "px";   // keep room for the ▾
  if (track.kind !== "camera") {
    const sw = variantSwatches[track.set_name] && variantSwatches[track.set_name][clip.value];
    if (sw) { const d = document.createElement("span"); d.className = "sw"; d.style.background = sw; div.appendChild(d); }
  }
  const lbl = document.createElement("span"); lbl.className = "clab";
  lbl.textContent = track.kind === "camera" ? clip.value.split("/").pop() : clip.value;
  div.appendChild(lbl);
  // per-clip variant dropdown shown as a small ▾ icon (no clipped label); the native popup
  // list (readable) opens on click. Transparent select overlays the icon.
  const cvWrap = document.createElement("span"); cvWrap.className = "clip-var-wrap"; cvWrap.title = "change variant";
  const caret = document.createElement("span"); caret.className = "caret"; caret.textContent = "▾";
  const cv = document.createElement("select"); cv.className = "clip-var";
  tlVariantsFor(track).forEach((val) => {
    const o = document.createElement("option"); o.value = val;
    o.textContent = track.kind === "camera" ? val.split("/").pop() : val;
    if (val === clip.value) o.selected = true; cv.appendChild(o);
  });
  cv.onpointerdown = (e) => e.stopPropagation();   // don't start a drag from the dropdown
  cv.onchange = (e) => { clip.value = e.target.value; markDirty(); renderTimeline(); };
  cvWrap.append(caret, cv);
  div.appendChild(cvWrap);
  const rs = document.createElement("div"); rs.className = "rs"; div.appendChild(rs);
  // (no dblclick-to-delete here — see deleteSelectedClip for why)
  div.onpointerdown = (e) => startClipDrag(e, ti, ci, e.target === rs ? "resize" : "move");
  return div;
}

function addClip(ti, value) {
  if (!value) return;
  const track = tl.tracks[ti]; const clips = track.clips;
  let dur = tlClipS;
  if (track.kind === "camera") {
    const info = (stage.cameras || []).find((c) => c.path === value);
    if (info && info.animated && stage.fps) {
      // an animated camera's natural clip length is one full lap as authored
      // (frames / stage fps) — e.g. a 240-frame turntable at 60fps drops in as 4s
      dur = Math.max(0.1, snap((stage.end_time - stage.start_time + 1) / stage.fps));
    }
  }
  if (appendAtPlayhead) {
    // drop the clip exactly at the playhead, overwriting (trim/split/drop) whatever it covers.
    // NOT clampClip — that's the drag/resize overlap resolver and would shove the clip off the
    // playhead onto the nearest free slot whenever the playhead sits on an existing clip.
    track.clips = TL.placeClipOverwrite(clips, value, snap(tlPlayT), dur);
  } else {
    // stack: butt up against the last clip (never overlaps, so no clamp needed)
    const start = clips.length ? clips[clips.length - 1].start_s + clips[clips.length - 1].duration_s : 0;
    clips.push({ value, start_s: start, duration_s: dur });
  }
  markDirty();
  renderTimeline();
}

function snap(s) { return Math.max(0, Math.round(s)); }   // clips snap to whole seconds
function clampClip(track, ci) {
  const clip = track.clips[ci];
  const others = track.clips.filter((_, i) => i !== ci);
  const prev = others.filter((o) => o.start_s <= clip.start_s).sort((a, b) => a.start_s - b.start_s).pop();
  const next = others.filter((o) => o.start_s > clip.start_s).sort((a, b) => a.start_s - b.start_s)[0];
  const minStart = prev ? prev.start_s + prev.duration_s : 0;
  if (clip.start_s < minStart) clip.start_s = minStart;
  const maxEnd = next ? next.start_s : Infinity;
  if (clip.start_s + clip.duration_s > maxEnd) {
    if (drag && drag.mode === "resize") clip.duration_s = Math.max(0.25, maxEnd - clip.start_s);
    else clip.start_s = Math.max(minStart, maxEnd - clip.duration_s);
  }
}
function startClipDrag(e, ti, ci, mode) {
  e.preventDefault();
  const track = tl.tracks[ti]; const clip = track.clips[ci];
  selClip = { track, clip };   // click selects (green outline); then may drag
  drag = { ti, ci, mode, x0: e.clientX, start0: clip.start_s, dur0: clip.duration_s };
  renderTimeline();
  window.addEventListener("pointermove", onClipDrag);
  window.addEventListener("pointerup", endClipDrag, { once: true });
}
function onClipDrag(e) {
  if (!drag) return;
  markDirty();   // a real move/resize (selection clicks alone never get here)
  const dt = (e.clientX - drag.x0) / PPS;
  const track = tl.tracks[drag.ti]; const clip = track.clips[drag.ci];
  if (drag.mode === "move") clip.start_s = Math.max(0, snap(drag.start0 + dt));
  else clip.duration_s = Math.max(0.25, snap(drag.dur0 + dt));
  clampClip(track, drag.ci);
  renderTimeline();
}
function endClipDrag() { drag = null; window.removeEventListener("pointermove", onClipDrag); }

// delete the selected clip (click a clip to select it -> green outline). The on-clip double-click
// can't fire reliably because the pointerdown re-renders the clip element out from under the second
// click, so the toolbar button + Delete key (both keyed on selClip) are the dependable path.
function deleteSelectedClip() {
  if (!selClip) return;
  markDirty();
  const ci = selClip.track.clips.indexOf(selClip.clip);
  if (ci >= 0) selClip.track.clips.splice(ci, 1);
  selClip = null;
  renderTimeline();
}
el("tl-del-clip").onclick = deleteSelectedClip;
document.addEventListener("keydown", (e) => {
  if (e.key !== "Delete" && e.key !== "Backspace") return;
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;   // don't hijack typing
  if (selClip) { e.preventDefault(); deleteSelectedClip(); }
});
// transport keyboard shortcuts — Timeline tab only, never while typing in a field
document.addEventListener("keydown", (e) => {
  if (!timelineActive()) return;
  const tag = (e.target && e.target.tagName) || "";
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key === " ") { e.preventDefault(); tlTogglePlay(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); tlStep(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); tlStep(1); }
  else if (e.key === "Home") { e.preventDefault(); tlSeek(0); }
  else if (e.key === "End") { e.preventDefault(); tlSeek(tl.duration_s); }
});

// ---- scrub (ruler) -> live viewport via /api/variant + /api/camera/snap (no new endpoint) ----
function timelineActive() {
  return document.querySelector('.tab[data-pane="timeline"]').classList.contains("active");
}
function setPlayhead(t) {
  tlPlayT = t;
  positionPlayhead();
  el("tl-playtime").textContent = t.toFixed(2) + "s";
}
function positionPlayhead() {
  const sl = el("tl-scroll").scrollLeft;
  const x = LABEL_W + tlPlayT * PPS - sl;
  const ph = el("tl-playhead");
  if (x < LABEL_W - 0.5) ph.style.display = "none";
  else { ph.style.display = "block"; ph.style.left = x + "px"; }
}
function syncRuler() {   // keep the fixed ruler + playhead aligned with horizontal track scroll
  el("tl-ruler").style.transform = `translateX(${-el("tl-scroll").scrollLeft}px)`;
  positionPlayhead();
}
// apply the composed state at time t to the live viewport IMMEDIATELY (no debounce).
// used directly by the transport (throttled) and frame-step/seek; postStateAt wraps it
// in the scrub debounce.
function applyStateAt(t) {
  const st = TL.stateAt(tl, t, tlBaseSel);
  tlTouchedLive = true;   // viewport now shows timeline state, not the Configure tab's
  api("/api/variant", { selections: st.selection }).catch(() => {});
  if (st.camera) {
    const info = (stage.cameras || []).find((c) => c.path === st.camera);
    if (info && info.animated) {
      // animated clip (turntable rig OR authored camera move): re-pose the rig at the
      // clip-relative stage time — the camera moves under the playhead
      tlLastCamera = st.camera;
      api("/api/camera/snap", { camera_path: st.camera,
                                at_s: Math.max(0, t - (st.cameraClipStart || 0)) }).catch(() => {});
    } else if (st.camera !== tlLastCamera) {
      tlLastCamera = st.camera;
      api("/api/camera/snap", { camera_path: st.camera }).catch(() => {});
    }
  }
}
// post the composed state at time t to the live viewport (debounced) — used by scrub AND edits
function postStateAt(t) {
  clearTimeout(scrubTimer);
  scrubTimer = setTimeout(() => applyStateAt(t), 90);
}
function scrubTo(e) {
  const rect = el("tl-rulerbar").getBoundingClientRect();
  const sl = el("tl-scroll").scrollLeft;
  let t = (e.clientX - rect.left - LABEL_W + sl) / PPS;
  t = Math.max(0, Math.min(t, tl.duration_s));
  setPlayhead(t);
  postStateAt(t);
}
el("tl-rulerbar").onpointerdown = (e) => {   // scrubber lives in the fixed header — always reachable
  if (e.target.closest("#tl-transport")) return;   // clicks on transport buttons don't scrub
  e.preventDefault();                              // stop the drag from selecting page text
  document.body.classList.add("scrubbing");
  scrubTo(e);
  window.addEventListener("pointermove", scrubTo);
  window.addEventListener("pointerup", () => {
    window.removeEventListener("pointermove", scrubTo);
    document.body.classList.remove("scrubbing");
  }, { once: true });
};
el("tl-scroll").addEventListener("scroll", syncRuler);

// ---- transport: drive the playhead through the scrub path (camera + variant swaps) ----
function tlSetTransportEnabled(on) {
  ["tl-to-start", "tl-step-back", "tl-play", "tl-step-fwd", "tl-to-end", "tl-loop"]
    .forEach((id) => { el(id).disabled = !on; });
}
function tlFollow(t) {   // keep the playhead in view while playing
  const sc = el("tl-scroll");
  const screenX = LABEL_W + t * PPS - sc.scrollLeft;   // mirror positionPlayhead's math
  const margin = 40;
  if (screenX > sc.clientWidth - margin) sc.scrollLeft += screenX - (sc.clientWidth - margin);
  else if (screenX < LABEL_W + margin) sc.scrollLeft = Math.max(0, sc.scrollLeft - ((LABEL_W + margin) - screenX));
  // setting scrollLeft fires the scroll listener -> syncRuler -> positionPlayhead realigns
}
function tlFrame() {
  if (!timelineActive()) { tlPause(); return; }   // left the tab -> stop
  const now = performance.now();
  const dt = (now - tlPlayStartWall) / 1000;
  const r = TL.nextPlayheadTime(tlPlayStartT, dt, tl.duration_s, tlLoop);
  setPlayhead(r.t);
  tlFollow(r.t);
  if (now - tlLastPost >= 90) { tlLastPost = now; applyStateAt(r.t); }   // throttle, not debounce
  if (r.stop) { tlPause(); applyStateAt(r.t); return; }                  // park + final post
  tlAnim = requestAnimationFrame(tlFrame);
}
function tlPlay() {
  if (tlAnim || tlRendering || !tl.duration_s) return;
  clearTimeout(scrubTimer);   // drop any pending debounced scrub post
  // a transport play and the turntable Preview spin must not both drive the camera
  fetch("/api/playback", { method: "POST", headers: { "Content-Type": "application/json" },
                           body: JSON.stringify({ playing: false }) }).catch(() => {});
  if (tlPlayT >= tl.duration_s - 1e-6) setPlayhead(0);   // restart if parked at the end
  tlPlayStartWall = performance.now();
  tlPlayStartT = tlPlayT;
  tlLastPost = 0;             // post on the first frame
  el("tl-play").innerHTML = "&#9208;";   // pause glyph
  tlAnim = requestAnimationFrame(tlFrame);
}
function tlPause() {
  if (tlAnim) cancelAnimationFrame(tlAnim);
  tlAnim = null;
  el("tl-play").innerHTML = "&#9654;";   // play glyph
}
function tlTogglePlay() { tlAnim ? tlPause() : tlPlay(); }
function tlSeek(t) {
  tlPause();
  t = Math.max(0, Math.min(t, tl.duration_s));
  setPlayhead(t); tlFollow(t); applyStateAt(t);
}
function tlStep(dir) { tlSeek(TL.frameStep(tlPlayT, tl.fps || 30, dir, tl.duration_s)); }
function tlSetLoop(on) { tlLoop = on; el("tl-loop").classList.toggle("active", on); }
el("tl-to-start").onclick = () => tlSeek(0);
el("tl-step-back").onclick = () => tlStep(-1);
el("tl-play").onclick = tlTogglePlay;
el("tl-step-fwd").onclick = () => tlStep(1);
el("tl-to-end").onclick = () => tlSeek(tl.duration_s);
el("tl-loop").onclick = () => tlSetLoop(!tlLoop);

// ---- presets + add-track ----
function loadPreset(preset) {
  tl = preset;
  markDirty();
  if (!tl.tracks.some((t) => t.kind === "camera")) {
    tl.tracks.unshift({ kind: "camera", set_name: null, prim_path: null, clips: [] });
  }
  tl.fps = parseInt(el("tl-fps").value) || 30;
  selClip = null; renderTimeline();
}
function hasClips() { return tl.tracks.some((t) => t.clips.length); }
function confirmReplace() { return !hasClips() || window.confirm("This replaces the current track view. Continue?"); }
el("tl-slideshow").onclick = () => { if (confirmReplace()) loadPreset(TL.makeSlideshow(stage.variant_sets, tlClipS)); };
el("tl-mixer").onclick = () => { if (confirmReplace()) loadPreset(TL.makeMixer(stage.variant_sets, tlClipS)); };
el("tl-clear").onclick = () => {   // keep the tracks, empty the clips
  if (!confirmReplace()) return;
  tl.tracks.forEach((t) => { t.clips = []; });
  markDirty();
  selClip = null; renderTimeline();
};

// ---- track views: a view belongs to the OPEN PROJECT (its clips reference this stage's
// variant sets + cameras and lean on the project's per-camera overrides — so views can't
// be global). No project open => no named views; the workspace timeline still saves with
// the project. ----
// show the "Saved track views" section only when a project is open (views live in the
// project); otherwise hide it and show a hint pointing at Project — the dependency made
// visible, instead of presenting a dead, empty section.
function syncProjectGate() {
  const open = !!curProject;
  el("tlv-block").style.display = open ? "" : "none";
  el("proj-hint").style.display = open ? "none" : "";
  el("tlv-project-name").textContent = open ? ` — ${curProject}` : "";
}
async function loadViews() {
  syncProjectGate();
  const sel = el("tlv-list"); sel.innerHTML = "";
  if (!curProject) return;   // nothing to list until a project is open
  try {
    const j = await (await fetch(`/api/timelines?project=${encodeURIComponent(curProject)}`)).json();
    (j.views || []).forEach((v) => { const o = document.createElement("option"); o.value = v.name; o.textContent = v.name; sel.appendChild(o); });
  } catch (e) { /* ignore */ }
}
el("tlv-save").onclick = async () => {
  if (!curProject) { el("tlv-msg").textContent = "open or save a project first — views are saved in the project"; return; }
  const name = el("tlv-name").value.trim();
  if (!name) { el("tlv-msg").textContent = "enter a name"; return; }
  try {
    const res = await fetch("/api/timelines/save", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, timeline: tl, project: curProject }) });
    if (!res.ok) { el("tlv-msg").textContent = "save failed: " + (await res.json()).detail; return; }
    el("tlv-msg").textContent = `saved "${name}" in project "${curProject}"`; el("tlv-name").value = ""; loadViews();
  } catch (e) { el("tlv-msg").textContent = "save failed: " + e.message; }
};
el("tlv-load").onclick = async () => {
  const name = el("tlv-list").value;
  if (!name || !curProject || !confirmReplace()) return;
  try {
    const r = await (await fetch(`/api/timelines/load?name=${encodeURIComponent(name)}&project=${encodeURIComponent(curProject)}`)).json();
    if (r && r.timeline) {
      tl = r.timeline; tl.fps = tl.fps || 30; selClip = null; markDirty(); renderTimeline();
      el("tlv-msg").textContent = `loaded "${name}"`;
    }
  } catch (e) { el("tlv-msg").textContent = "load failed: " + e.message; }
};
el("tlv-del").onclick = async () => {
  const name = el("tlv-list").value;
  if (!name || !curProject || !window.confirm(`Delete view "${name}" from project "${curProject}"?`)) return;
  try {
    await fetch("/api/timelines/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, project: curProject }) });
    el("tlv-msg").textContent = `deleted "${name}"`; loadViews();
  } catch (e) { /* ignore */ }
};

// ---- projects (USD + base look + display + all saved views) ----
let curProject = null;   // the open project; an empty-name Save updates it (Save vs Save-As)
let workspaceDirty = false;   // workspace drifted since the last project save/open
function markDirty() {
  workspaceDirty = true;
  el("proj-save").classList.add("dirty");
}
function clearDirty() {
  workspaceDirty = false;
  el("proj-save").classList.remove("dirty");
}
function syncProjPlaceholder() {
  el("proj-name").placeholder = curProject ? `re-save "${curProject}"` : "project name";
}
async function loadProjects() {
  try {
    const j = await (await fetch("/api/projects")).json();
    const sel = el("proj-list"); sel.innerHTML = "";
    (j.projects || []).forEach((p) => { const o = document.createElement("option"); o.value = p.name; o.textContent = p.name; sel.appendChild(o); });
  } catch (e) { /* ignore */ }
}
el("proj-save").onclick = async () => {
  // typed name = save as that project; empty = update the open one (workspace drifts the moment
  // you keep dialing looks / reframing / editing clips — this is its Ctrl+S)
  const name = el("proj-name").value.trim() || curProject;
  if (!name) { el("proj-msg").textContent = "enter a project name"; return; }
  if (!stage) { el("proj-msg").textContent = "open a stage first"; return; }
  try {
    await fetch("/api/projects/save", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, base_selection: Object.values(selection), display: gatherDisplay(),
                             timeline: tl, camera: el("camera-select").value }) });
    el("proj-msg").textContent = `saved project "${name}"`; el("proj-name").value = "";
    curProject = name; syncProjPlaceholder(); clearDirty(); loadProjects();
    loadViews();   // a freshly-saved project enables its (initially empty) view library
  } catch (e) { el("proj-msg").textContent = "save failed: " + e.message; }
};
el("proj-open").onclick = async () => {
  const name = el("proj-list").value;
  if (!name) return;
  if (stage && !window.confirm("Opening a project reloads the stage and replaces the current look, display, and views. Continue?")) return;
  try {
    const rec = await (await fetch(`/api/projects/load?name=${encodeURIComponent(name)}`)).json();
    if (rec.usd_path) el("usd-path").value = rec.usd_path;
    const looks = rec.looks || {}, xforms = rec.xforms || {};
    const newFormat = Object.keys(looks).length || Object.keys(xforms).length || rec.camera;
    // per-camera state + selected camera ride in WITH the open — ONE reopen, the first frame
    // is already the saved look/framing (no original-view-then-settle phase)
    await openStage(rec.base_selection || [], { camera: rec.camera, looks, xforms });
    if (rec.camera) el("camera-select").value = rec.camera;   // UI sync; server opened on it
    if (newFormat) {
      // optics arrived with the open; only a genuine stream-size change needs an extra step
      const res = rec.display && rec.display.resolution;
      if (res && el("disp-res").value !== res.join("x")) {
        el("disp-res").value = res.join("x");
        postDisplay({ resolution: res });
        scheduleStreamReconnect();
      }
    } else {
      applyDisplay(rec.display);                  // legacy project: single global display
    }
    if (rec.timeline) {                           // workspace: the editor strip's working timeline
      tl = rec.timeline; tl.fps = tl.fps || 30; selClip = null; tlLastCamera = null;
      renderTimeline();
    }
    curProject = name;                            // set BEFORE loadViews — views are project-scoped
    loadViews();                                  // the project's own track views
    syncProjPlaceholder(); clearDirty();          // freshly opened == saved state
    el("proj-msg").textContent = `opened "${name}"`;
  } catch (e) { el("proj-msg").textContent = "open failed: " + e.message; }
};
el("proj-del").onclick = async () => {
  const name = el("proj-list").value;
  if (!name || !window.confirm(`Delete project "${name}"?`)) return;
  try {
    await fetch("/api/projects/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }) });
    if (name === curProject) { curProject = null; syncProjPlaceholder(); }
    el("proj-msg").textContent = `deleted "${name}"`; loadProjects();
  } catch (e) { /* ignore */ }
};
syncProjPlaceholder();
el("tl-clip").oninput = renderTimeline;
el("tl-fps").oninput = renderTimeline;

// append-mode toggle
function setAppendMode(playhead) {
  appendAtPlayhead = playhead;
  el("tl-mode-stack").classList.toggle("active", !playhead);
  el("tl-mode-playhead").classList.toggle("active", playhead);
}
el("tl-mode-stack").onclick = () => setAppendMode(false);
el("tl-mode-playhead").onclick = () => setAppendMode(true);

// show a hidden track again
el("tl-show-track").onchange = (e) => {
  const ti = parseInt(e.target.value);
  if (!Number.isNaN(ti) && tl.tracks[ti]) { tl.tracks[ti].hidden = false; markDirty(); renderTimeline(); }
};

// resize the Tracks column (label gutter): updates LABEL_W (layout) + --label-w (CSS widths)
function setLabelW(px) {
  LABEL_W = Math.max(150, Math.min(px, 420));
  document.documentElement.style.setProperty("--label-w", LABEL_W + "px");
  renderTimeline();
}
el("tl-gutter-resize").onpointerdown = (e) => {
  e.preventDefault();
  const x0 = e.clientX, w0 = LABEL_W;
  const mv = (ev) => setLabelW(w0 + (ev.clientX - x0));
  window.addEventListener("pointermove", mv);
  window.addEventListener("pointerup", () => window.removeEventListener("pointermove", mv), { once: true });
};
// resize the right panel by dragging its left edge
el("panel-resize").onpointerdown = (e) => {
  e.preventDefault();
  const panel = document.querySelector(".panel");
  const x0 = e.clientX, w0 = panel.getBoundingClientRect().width;
  const mv = (ev) => { panel.style.width = Math.max(260, Math.min(w0 + (x0 - ev.clientX), window.innerWidth * 0.6)) + "px"; };
  window.addEventListener("pointermove", mv);
  window.addEventListener("pointerup", () => window.removeEventListener("pointermove", mv), { once: true });
};

// resize the timeline strip by dragging its top edge
el("tl-resize").onpointerdown = (e) => {
  e.preventDefault();
  const strip = el("timeline-strip");
  const startY = e.clientY, startH = strip.getBoundingClientRect().height;
  const onMove = (ev) => {
    const h = Math.max(140, Math.min(startH + (startY - ev.clientY), window.innerHeight * 0.75));
    strip.style.height = h + "px";
  };
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", () => window.removeEventListener("pointermove", onMove), { once: true });
};

// ---- render to MP4 ----
async function tlRender() {
  const out = el("tl-out").value.trim();
  if (!out) { el("tl-status").textContent = "set an output folder"; return; }
  if (!tl.duration_s) { el("tl-status").textContent = "add some clips first"; return; }
  // what-you-see-is-what-renders: the MP4 inherits the viewport's aspect/resolution
  const [rw, rh] = el("disp-res").value.split("x").map(Number);
  const quality = { mode: "RealTimePathTracing", samples_per_pixel: 48, max_bounces: 4, resolution: [rw, rh] };
  let r;
  try {
    r = await fetch("/api/timeline/render", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timeline: tl, quality, out_dir: out }),
    });
  } catch (e) { el("tl-status").textContent = "render request failed: " + e.message; return; }
  if (r.status === 422) { el("tl-status").textContent = "invalid timeline: " + (await r.text()); return; }
  if (!r.ok) { el("tl-status").textContent = "error: " + (await r.text()); return; }
  const { frames } = await r.json();
  tlPause(); tlRendering = true; tlSetTransportEnabled(false);   // transport must not fight the render's USD_LOCK
  el("tl-render").disabled = true; el("tl-cancel").disabled = false;
  el("tl-bar").style.width = "0%"; el("tl-status").textContent = `rendering ${frames} frames…`;
  el("results-dir").value = out;
}
function onTimelineProgress(e) {
  const pct = e.total ? Math.round((e.frame / e.total) * 100) : 0;
  el("tl-bar").style.width = pct + "%";
  el("tl-status").textContent = `${e.frame}/${e.total} frames`;
}
async function onTimelineDone(e) {
  tlRendering = false; tlSetTransportEnabled(true);
  el("tl-render").disabled = false; el("tl-cancel").disabled = true;
  el("tl-bar").style.width = "100%";
  el("tl-status").textContent = e.cancelled ? `cancelled — ${e.frames} frames → ${e.video}` : `done → ${e.video}`;
  if (!e.video) return;
  // surface the result: derive the folder from the event (robust to page reloads wiping
  // results-dir), select the new mp4, and bring it into the viewport via the Results tab.
  el("results-dir").value = e.video.replace(/[\\/][^\\/]*$/, "");
  await refreshResults();
  const i = results.findIndex((p) => p.video === e.video);
  if (i >= 0) showResult(i);
  if (!e.cancelled) document.querySelector('.tab[data-pane="results"]').click();
}
el("tl-render").onclick = tlRender;
el("tl-cancel").onclick = () => { fetch("/api/timeline/cancel", { method: "POST" }).catch(() => {}); el("tl-status").textContent = "cancelling…"; };

const v = el("remote-video");
// Frame-flow watchdog: a HALF-OPEN WebRTC session (signaling succeeded -> we show LIVE, but
// the media transport never established server-side) emits no onStop and no error — the only
// honest signal is that video frames stop arriving. LIVE + dry pipe for 10s => reconnect.
let lastVideoFrame = 0;
let serverStageReady = false;   // server produced a frame of the open stage (ready event /api/stage)
let dryPipeCycles = 0;          // consecutive no-frame recoveries -> escalate to a streamer rebuild
let restartRequested = false;   // at most ONE server-side rebuild per dry streak (no restart storms)
if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
  const tick = () => { lastVideoFrame = performance.now(); dryPipeCycles = 0; restartRequested = false; v.requestVideoFrameCallback(tick); };
  v.requestVideoFrameCallback(tick);
  setInterval(() => {
    // only meaningful when the server HAS frames to send — during a long first open there
    // are legitimately none, and recovering in a loop just spams "stream lost"
    if (!streamLive || !serverStageReady) return;
    if (performance.now() - lastVideoFrame > 10000) {
      connected = false; streamLive = false;
      dryPipeCycles++;
      if (dryPipeCycles >= 2 && !restartRequested) {
        // repeated half-open sessions: the SERVER side is wedged — rebuild its streamer
        // ONCE per streak (frames arriving resets the latch)
        restartRequested = true;
        fetch("/api/stream/restart", { method: "POST" }).catch(() => {});
      } else if (dryPipeCycles >= 4) {
        // the rebuild didn't clear it — only a process restart ever has; the watchdog
        // relaunches the server and recovery auto-reopens the last stage
        dryPipeCycles = 0;
        setOverlay("Stream wedged — restarting the render server…");
        fetch("/api/restart", { method: "POST" }).catch(() => {});
      }
      scheduleStreamRecovery("no video frames");
    }
  }, 5000);
}
v.addEventListener("mouseenter", () => { if (connected) v.focus(); });
// navigation is FREE — framing only persists via the explicit Save framing button
window.addEventListener("beforeunload", () => { if (connected) AppStreamer.terminate(); });

// ---- info popovers: one fixed tooltip for every [data-help] element. JS-positioned
// (viewport-clamped, flips upward near the bottom) because a CSS ::after child clips
// against the sidebar's overflow for low blocks like Turntable.
const helpPop = document.createElement("div");
helpPop.id = "help-pop";
document.body.appendChild(helpPop);
let helpTarget = null;
document.addEventListener("mouseover", (ev) => {
  const t = ev.target && ev.target.closest ? ev.target.closest("[data-help]") : null;
  if (t === helpTarget) return;
  helpTarget = t;
  if (!t) { helpPop.style.display = "none"; return; }
  helpPop.textContent = t.getAttribute("data-help");
  helpPop.style.display = "block";
  const r = t.getBoundingClientRect();
  const x = Math.max(8, Math.min(r.left, window.innerWidth - helpPop.offsetWidth - 8));
  let y = r.bottom + 7;
  if (y + helpPop.offsetHeight > window.innerHeight - 8) y = r.top - helpPop.offsetHeight - 7;
  helpPop.style.left = x + "px"; helpPop.style.top = Math.max(8, y) + "px";
});
document.documentElement.addEventListener("mouseleave", () => {
  helpPop.style.display = "none"; helpTarget = null;
});

// ---- warming ticker: a connected stream with no stage frame yet is WORKING, not dead.
// Show what's happening and for how long, so a composing reopen never reads as a failure.
let warmTicker = null, warmT0 = 0;
function showWarming() {
  if (warmTicker) return;
  warmT0 = performance.now();
  const tick = () => {
    if (serverStageReady || !connected) { stopWarming(); return; }
    const sec = Math.round((performance.now() - warmT0) / 1000);
    setOverlay(`Stage warming up — composing + rendering the first frame (${sec}s, ~30s is normal)`);
    if (sec > 0 && sec % 5 === 0) {
      // self-heal: the ready signal is a one-shot WS event — a tab that missed it
      // (reconnect gap, second tab) would tick forever. Ask the server directly.
      fetch("/api/stage").then((r) => r.json()).then((st) => {
        if (st.ready) { serverStageReady = true; stopWarming(); setStatus("live", "live"); }
      }).catch(() => {});
    }
  };
  tick();
  warmTicker = setInterval(tick, 1000);
}
function stopWarming() {
  clearInterval(warmTicker); warmTicker = null;
  if ((overlay.textContent || "").startsWith("Stage warming")) setOverlay("");
}

// ---- collapsible Configure & Grid blocks: collapse the upper settings (Camera / Render mode /
// Display / Turntable in Configure; Matrix mode / Quality & output / Cameras in Grid) to a
// header-only row so the .grow list below (Variant sets / Include sets) gets the freed space.
// The .grow block is the thing we're making room for, so it stays open. Collapsed state
// persists per block in localStorage, keyed by the header's title text (keys are unique across
// both panes, e.g. Configure "Camera" vs Grid "Cameras").
(function setupCollapsibleBlocks() {
  document.querySelectorAll("#pane-configure .block:not(.grow), #pane-grid .block:not(.grow)").forEach((block) => {
    const h2 = block.querySelector("h2");
    if (!h2) return;
    const title = ((h2.childNodes[0] && h2.childNodes[0].textContent) || "").trim();
    if (!title) return;                       // need a stable persistence key
    const key = "dvp_collapse_" + title.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    block.classList.add("collapsible");
    const caret = document.createElement("span");
    caret.className = "block-caret";
    caret.textContent = "▾";             // ▾ (rotates to ▸ via CSS when collapsed)
    h2.appendChild(caret);
    let collapsed = false;
    try { collapsed = localStorage.getItem(key) === "1"; } catch (e) { /* ignore */ }
    block.classList.toggle("collapsed", collapsed);
    h2.addEventListener("click", () => {
      collapsed = !block.classList.contains("collapsed");
      block.classList.toggle("collapsed", collapsed);
      try { localStorage.setItem(key, collapsed ? "1" : "0"); } catch (e) { /* ignore */ }
    });
  });
})();

connectEvents();
loadProjects();   // populate the project list so you can open one before any stage
syncProjectGate();   // track-views section stays hidden until a project is open
attachIfOpen().then((attached) => { if (!attached) setOverlay("Open a stage to begin"); });
})();
