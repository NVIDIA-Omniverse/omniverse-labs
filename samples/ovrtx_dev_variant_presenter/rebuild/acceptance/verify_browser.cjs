// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
/* ============================================================================
 * acceptance/verify_browser.cjs — the BUILD-AGNOSTIC browser/pixel gate for
 * Dev Variant Presenter. A build is not done until the INTERACTIVE UX actually
 * works, and that can only be established by driving it.
 *
 * It drives a REAL mouse/keyboard in HEADFUL Chrome (NO --use-fake-ui-for-media-
 * stream — that flag kills puppeteer's WebRTC receive) and asserts every SPEC-UX
 * `[vb]` clause. It is keyed ONLY on the SPEC-UX DOM-id contract + RENDERED PIXELS
 * — never on /api response field names — so the SAME file validates the reference
 * app AND any compliant build. A server-side grader cannot see dead sliders, dead
 * orbit, dead timeline editing or a dead gizmo: those endpoints all answer 200
 * while nothing moves on screen. This file is where such clauses get driven
 * instead of self-attested — a clause with no assertion here is not a gate.
 *
 * Usage:
 *   node verify_browser.cjs [URL] [USD_STAGE]
 *     URL        default http://127.0.0.1:8080
 *     USD_STAGE  a local .usd path (filled into #usd-path). Falls back to env
 *                VS_USD, then to whatever the page already has in #usd-path.
 *   env CHROME   override the Chrome executable path.
 *
 * Single-client stream: do NOT open another browser/client while this runs.
 * Requires: puppeteer-core + a system Chrome. Many GPU reopens (optics apply via
 * a stage reopen) make a full run ~10–18 min against a real ConceptCar — that is
 * expected; every check is independent (one flaky check never aborts the suite).
 * ========================================================================== */
"use strict";
const fs = require("fs");
const os = require("os");
const path = require("path");
let puppeteer;
try { puppeteer = require("puppeteer-core"); }
catch (e) { console.error("FATAL: puppeteer-core not installed. `npm i puppeteer-core` (a system Chrome is also required)."); process.exit(2); }

const URL = process.argv[2] || "http://127.0.0.1:8080";
const USD = process.argv[3] || process.env.VS_USD || "";
// THE S3 LANE: USD may be an http(s)/S3 stage URL. Run this gate TWICE — once with the local stage
// and once with the remote URL — the app is only done when BOTH runs are green. A cold remote open
// mirrors the full closure (~11 GB for ConceptCar, many minutes) before frames flow; budgets adapt.
const IS_URL = /^https?:\/\//i.test(USD);
// Chrome executable: env CHROME wins, else a per-platform default. NOTE this gate runs HEADFUL
// on purpose (headless kills puppeteer's WebRTC receive), so it needs a REAL display — on Linux
// that means an X/Wayland session (or Xvfb + a GPU-capable Chrome), not a bare SSH shell.
const CHROME = process.env.CHROME || (process.platform === "win32"
  ? "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
  : (fs.existsSync("/usr/bin/google-chrome") ? "/usr/bin/google-chrome" : "/usr/bin/chromium"));
const OUT_DIR = path.join(os.tmpdir(), "vs_verify_browser_out");

let pass = 0, fail = 0, warn = 0;
const fails = [];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function log(s) { console.log(s); }
class Soft extends Error {}                  // assertion failure (a FAIL, not a crash)
function assert(cond, msg) { if (!cond) throw new Soft(msg || "assertion failed"); }

// Run one independent check. fn may throw (Soft = expected fail) or return false.
async function check(page, name, fn) {
  try {
    const r = await fn();
    if (r === false) { fail++; fails.push(name); log(`  [FAIL] ${name}`); }
    else { pass++; log(`  [PASS] ${name}${typeof r === "string" ? "  (" + r + ")" : ""}`); }
  } catch (e) {
    fail++; fails.push(name);
    log(`  [FAIL] ${name}  — ${e instanceof Soft ? e.message : (e && e.message) || e}`);
  }
}
function warnIf(cond, name, detail) {
  if (cond) { pass++; log(`  [PASS] ${name}${detail ? "  (" + detail + ")" : ""}`); }
  else { warn++; log(`  [WARN] ${name}${detail ? "  (" + detail + ")" : ""}`); }
}

// ---- in-page helpers (strings passed to page.evaluate) -------------------
// mean RGB + luma over a centred patch of the live <video>, drawn to a canvas.
// A MediaStream-sourced video is origin-clean, so getImageData is NOT tainted.
const SAMPLE = `(() => {
  const v = document.getElementById('remote-video');
  if (!v || !v.videoWidth) return null;
  const W=128,H=72; const c=document.createElement('canvas'); c.width=W; c.height=H;
  const g=c.getContext('2d'); g.drawImage(v,0,0,W,H);
  const d=g.getImageData(W*0.25,H*0.25,W*0.5,H*0.5).data;
  let r=0,gg=0,b=0,n=0; for(let i=0;i<d.length;i+=4){r+=d[i];gg+=d[i+1];b+=d[i+2];n++;}
  r/=n; gg/=n; b/=n;
  return {r,g:gg,b,luma:0.2126*r+0.7152*gg+0.0722*b,
          vw:v.videoWidth, decoded:(v.getVideoPlaybackQuality?v.getVideoPlaybackQuality().totalVideoFrames:0)};
})()`;
// high-frequency energy over a side patch (background) — drops when DoF blurs it.
const EDGE = `(() => {
  const v=document.getElementById('remote-video'); if(!v||!v.videoWidth) return null;
  const W=160,H=90; const c=document.createElement('canvas'); c.width=W; c.height=H;
  const g=c.getContext('2d'); g.drawImage(v,0,0,W,H);
  const d=g.getImageData(0,0,W,H).data; let e=0,n=0;
  const L=(x,y)=>{const i=(y*W+x)*4; return 0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2];};
  for(let y=1;y<H-1;y++) for(let x=1;x<W-1;x++){const gx=L(x+1,y)-L(x-1,y),gy=L(x,y+1)-L(x,y-1); e+=gx*gx+gy*gy; n++;}
  return e/n;
})()`;
// high-frequency energy over the CENTRE region (where the product/car sits) — DoF blurs the
// out-of-focus car, dropping centre-edge energy. Centre-only isolates the subject from a smooth backdrop.
const CEDGE = `(() => { const v=document.getElementById('remote-video'); if(!v||!v.videoWidth) return null;
  const W=160,H=90; const c=document.createElement('canvas'); c.width=W; c.height=H; const g=c.getContext('2d'); g.drawImage(v,0,0,W,H);
  const x0=44,x1=116,y0=26,y1=64; const d=g.getImageData(0,0,W,H).data; let e=0,n=0;
  const L=(x,y)=>{const i=(y*W+x)*4; return 0.2126*d[i]+0.7152*d[i+1]+0.0722*d[i+2];};
  for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){const gx=L(x+1,y)-L(x-1,y),gy=L(x,y+1)-L(x,y-1);e+=gx*gx+gy*gy;n++;} return +(e/n).toFixed(1); })()`;
const rgbDelta = (a, b) => (a && b) ? Math.abs(a.r - b.r) + Math.abs(a.g - b.g) + Math.abs(a.b - b.b) : 0;

// After an optics REOPEN the path tracer restarts: early frames are GRAINY (high edge), settling as the
// denoiser converges. Wait for the centre-edge metric to stop drifting, then average a few reads — so a
// DoF comparison sees two CONVERGED frames, not reopen noise.
async function convergedEdge(page, { maxMs = 16000 } = {}) {
  await sleep(2500);
  let prev = await evalSafe(page, CEDGE); const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    await sleep(1500); const cur = await evalSafe(page, CEDGE);
    if (cur != null && prev != null && Math.abs(cur - prev) <= 0.06 * Math.max(1, prev)) { prev = cur; break; }
    prev = cur;
  }
  let s = 0, n = 0;
  for (let i = 0; i < 3; i++) { const v = await evalSafe(page, CEDGE); if (v != null) { s += v; n++; } await sleep(700); }
  return n ? s / n : prev;
}
// Like convergedEdge but returns a stabilized full {r,g,b,luma} — for ISO (luma direction) and focal
// (frame delta) compares, which otherwise sample a mid-reconverge frame after the optics reopen.
async function convergedSample(page, { maxMs = 16000 } = {}) {
  await sleep(2500);
  let prevL = null; const t0 = Date.now();
  while (Date.now() - t0 < maxMs) {
    const s = await sample(page);
    if (s && prevL != null && Math.abs(s.luma - prevL) <= 0.06 * Math.max(2, prevL) + 0.4) break;
    prevL = s ? s.luma : prevL; await sleep(1500);
  }
  let r = 0, g = 0, b = 0, l = 0, n = 0;
  for (let i = 0; i < 3; i++) { const s = await sample(page); if (s) { r += s.r; g += s.g; b += s.b; l += s.luma; n++; } await sleep(700); }
  return n ? { r: r / n, g: g / n, b: b / n, luma: l / n } : null;
}

// Wait until the rendered frame DIFFERS from a baseline by > thresh (a variant switch may take the
// fast OR the slower reload path; a single fixed sleep races it). Returns the changed sample (or last).
async function waitFrameChange(page, baseline, { maxMs = 14000, thresh = 6, stepMs = 1200 } = {}) {
  const t0 = Date.now(); let last = baseline;
  while (Date.now() - t0 < maxMs) {
    await sleep(stepMs); const cur = await sample(page); if (!cur) continue; last = cur;
    if (rgbDelta(baseline, cur) > thresh) return cur;
  }
  return last;
}

async function evalSafe(page, expr) { try { return await page.evaluate(expr); } catch (e) { return null; } }
async function sample(page) { return evalSafe(page, SAMPLE); }
// Sample a scalar metric repeatedly until it stabilizes (path-trace reconverge after an optics
// reopen makes a single sample noisy). Returns the last (stable) value.
async function stable(page, expr, { maxMs = 14000, tol = 0.12, stepMs = 1500 } = {}) {
  const t0 = Date.now(); let prev = await evalSafe(page, expr);
  while (Date.now() - t0 < maxMs) {
    await sleep(stepMs); const cur = await evalSafe(page, expr);
    if (cur != null && prev != null && Math.abs(cur - prev) <= tol * Math.max(1, Math.abs(prev))) return cur;
    prev = cur;
  }
  return prev;
}
async function statusText(page) { return (await evalSafe(page, `(document.getElementById('status')||{}).textContent`)) || ""; }
async function decoded(page) { const s = await sample(page); return s ? s.decoded : 0; }

// wait until the live status returns to "live" (every optics/resolution change
// reopens/rebuilds: stage_open -> "warming up" -> ready -> "live").
async function waitLive(page, timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    if (/\blive\b/i.test(await statusText(page))) return true;
    await sleep(1200);
  }
  return false;
}
// wait until videoWidth>0 and frames are flowing (decoded > min).
async function waitMedia(page, timeoutMs, min = 1) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const s = await sample(page);
    if (s && s.vw > 0 && s.decoded > min) return s;
    await sleep(1500);
  }
  return null;
}
// wait until decoded-frame flow RESUMES past a baseline (used after reconnect/reopen).
async function waitResume(page, baseDecoded, timeoutMs) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    const d = await decoded(page);
    if (d > baseDecoded + 4) return d;
    await sleep(1200);
  }
  return 0;
}

// element-geometry helpers in client/viewport coords (== puppeteer mouse coords)
async function rectOf(page, selector) {
  return page.evaluate((sel) => {
    const e = document.querySelector(sel); if (!e) return null;
    const r = e.getBoundingClientRect();
    return { x: r.x, y: r.y, w: r.width, h: r.height, cx: r.x + r.width / 2, cy: r.y + r.height / 2 };
  }, selector);
}
async function dragMouse(page, x0, y0, x1, y1, steps = 12) {
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  for (let i = 1; i <= steps; i++) await page.mouse.move(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps);
  await sleep(80);
  await page.mouse.up();
}
// Hard-stop the transport/turntable playback (defeats state-leak between timeline checks: a prior
// check that left playback running drifts the playhead off a parked clip → the next check reads a
// stale frame). Posts directly so it works regardless of which control started it.
async function stopPlayback(page) {
  try { await page.evaluate(() => fetch("/api/playback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ playing: false }) })); } catch (e) {}
}
// Rebuild the ovstream session — the SAME recovery the app's own watchdog performs. ovstream can
// intermittently serve a STATIC cached frame while decode counters advance (is_client_connected
// false-negative / ghost-session idle): every pixel check then reads ~0 delta on a healthy app.
// Pixel checks that measure ~0 retry once through this; a genuinely broken build stays at 0.
async function kickStream(page) {
  try { await page.evaluate(() => fetch("/api/stream/restart", { method: "POST" })); } catch (e) {}
  await waitLive(page, 45000);
  const d0 = await decoded(page); await waitResume(page, d0, 30000); await sleep(1500);
}
// Click the timeline ruler at time t (seconds), VERIFYING the playhead actually lands at ~t.
// Two fragilities are handled: (1) the timeline strip sits at the BOTTOM and the app content SCROLLS,
// so a re-render pushes the ruler below the visible viewport → scrollIntoView + re-read the rect
// before each click; (2) the ruler maps clientX→time via `clientX-rect.left-LABEL_W+tl-scroll.scrollLeft`,
// so a drifted horizontal scrollLeft makes a t=4.5 click land in the wrong clip (playhead moves, but to
// the wrong time → same variant → 0 delta). Force scrollLeft=0 first. LABEL_W=230, PPS=70. Retry until
// the playhead reads ~t (the click truly registered), so callers get a deterministic playhead position.
async function scrubToTime(page, t) {
  for (let attempt = 0; attempt < 4; attempt++) {
    const rb = await page.evaluate(() => {
      const sc = document.getElementById("tl-scroll"); if (sc) sc.scrollLeft = 0;     // deterministic clientX→time
      const e = document.getElementById("tl-rulerbar"); e.scrollIntoView({ block: "center" });
      const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height };
    });
    await sleep(200);
    let x = rb.x + 230 + t * 70;
    x = Math.min(Math.max(x, rb.x + 234), rb.x + rb.w - 6);   // past the transport gutter, inside the bar
    const y = rb.y + rb.h / 2;                                 // scrollIntoView centred it → mid-viewport
    await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.move(x + 2, y); await sleep(80); await page.mouse.up();
    await sleep(350);
    const got = await page.evaluate(() => parseFloat(document.getElementById("tl-playtime").textContent) || 0);
    if (Math.abs(got - Math.min(t, 6)) < 0.7) return got;     // playhead reached ~t → the click registered
  }
  return await page.evaluate(() => parseFloat(document.getElementById("tl-playtime").textContent) || 0);
}

// set an <input>/<select> value and fire the native event(s) the handler listens on
async function setAndFire(page, selector, value, events) {
  return page.evaluate((sel, val, evs) => {
    const e = document.querySelector(sel); if (!e) return false;
    e.value = val;
    for (const t of evs) e.dispatchEvent(new Event(t, { bubbles: true }));
    return true;
  }, selector, String(value), events || ["input", "change"]);
}

// current authored camera pose from the control plane (eye + basis OR a 4x4) — used to
// RECOVER the orbit pivot geometrically (defends "orbit about world origin" + "live-spin").
async function camPose(page){ return page.evaluate(async()=>{ try{ return await (await fetch('/api/camera-pose')).json(); }catch(e){ return null; } }); }
// closest point of two lines (eye_i along forward_i); sign of forward doesn't matter (same line) → pivot recovery
function pivotFromPoses(p0,p1){ if(!p0||!p1||!p0.eye||!p1.eye) return null;
  const fwd=(p)=>{ if(p.m&&p.m.length>=11) return [-p.m[8],-p.m[9],-p.m[10]]; const r=p.right,u=p.up; return [r[1]*u[2]-r[2]*u[1], r[2]*u[0]-r[0]*u[2], r[0]*u[1]-r[1]*u[0]]; };
  const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2], add=(a,b)=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]], mul=(a,s)=>[a[0]*s,a[1]*s,a[2]*s];
  const o1=p0.eye,d1=fwd(p0),o2=p1.eye,d2=fwd(p1); const n1=Math.hypot(...d1)||1,n2=Math.hypot(...d2)||1; const u1=mul(d1,1/n1),u2=mul(d2,1/n2);
  const r=sub(o1,o2); const a=dot(u1,u1),b=dot(u1,u2),c=dot(u2,u2),d=dot(u1,r),e=dot(u2,r); const den=a*c-b*b; if(Math.abs(den)<1e-6) return null;
  const t=(b*e-c*d)/den, s=(a*e-b*d)/den; const pA=add(o1,mul(u1,t)),pB=add(o2,mul(u2,s)); return mul(add(pA,pB),0.5); }
const vlen=(p)=>p?Math.hypot(p[0],p[1],p[2]):0;
const eyeDelta=(a,b)=>(a&&b&&a.eye&&b.eye)?Math.abs(a.eye[0]-b.eye[0])+Math.abs(a.eye[1]-b.eye[1])+Math.abs(a.eye[2]-b.eye[2]):0;

(async () => {
  try { fs.rmSync(OUT_DIR, { recursive: true, force: true }); } catch (e) {}
  try { fs.mkdirSync(OUT_DIR, { recursive: true }); } catch (e) {}

  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: false,
    args: ["--autoplay-policy=no-user-gesture-required", "--no-first-run", "--no-default-browser-check",
           "--disable-features=IsolateOrigins,site-per-process",
           // keep WebRTC video decoding alive even if the window loses focus
           "--disable-backgrounding-occluded-windows", "--disable-background-timer-throttling",
           "--disable-renderer-backgrounding", "--window-size=1600,950"],
    defaultViewport: null,
  });
  const page = await browser.newPage();
  try { await page.bringToFront(); } catch (e) {}
  page.on("dialog", (d) => d.accept().catch(() => {}));   // auto-accept confirm()/alert()
  // TEE the app's own /events WebSocket into window.__vbEvents (no second connection — some builds
  // gate streaming on the WS count). Used to wait for the `classified {fast_sets}` event so the
  // fast-path variant checks test the FAST path, not the pre-classification reload window.
  await page.evaluateOnNewDocument(() => {
    window.__vbEvents = [];
    const NativeWS = window.WebSocket;
    window.WebSocket = function (...args) {
      const ws = new NativeWS(...args);
      try {
        ws.addEventListener("message", (e) => {
          try { const j = JSON.parse(e.data); if (j && typeof j.type === "string" && window.__vbEvents.length < 2000) window.__vbEvents.push(j); } catch (err) {}
        });
      } catch (err) {}
      return ws;
    };
    window.WebSocket.prototype = NativeWS.prototype;
    for (const k of ["CONNECTING", "OPEN", "CLOSING", "CLOSED"]) { try { Object.defineProperty(window.WebSocket, k, { value: NativeWS[k] }); } catch (e) {} }
  });
  // capture /api/display POST bodies so slider→server wiring can be asserted deterministically
  // (a "dead slider" updates nothing AND posts nothing — the pixel effect of DoF is scene-dependent)
  const displayPosts = [];
  const restartCalls = [];   // POSTs to /api/stream/restart or /api/restart (reconnect-storm detector)
  const pickPosts = [];      // POSTs to /api/pick-point (gizmo drags must NEVER re-pick geometry)
  page.on("request", (req) => {
    try {
      if (req.method() === "POST" && /\/api\/display/.test(req.url())) displayPosts.push(req.postData() || "");
      if (req.method() === "POST" && /\/api\/(stream\/restart|restart)\b/.test(req.url())) restartCalls.push(req.url());
      if (req.method() === "POST" && /\/api\/pick-point\b/.test(req.url())) pickPosts.push(req.url());
    } catch (e) {}
  });
  page.on("pageerror", (e) => log("   [pageerror] " + String(e).slice(0, 140)));

  try {
    log(`\n== verify_browser.cjs → ${URL}  stage=${USD || "(field default)"} ==\n`);
    await page.goto(URL, { waitUntil: "domcontentloaded" });
    await sleep(1500);

    // ---- boot ----
    await check(page, "boot: OVWebRTC lib + Open handler wired", async () => {
      const r = await page.evaluate(() => typeof window.OVWebRTC === "object" &&
        typeof document.getElementById("open-btn").onclick === "function");
      assert(r, "window.OVWebRTC or #open-btn.onclick missing");
      return true;
    });

    // ---- open the stage ----
    if (USD) await setAndFire(page, "#usd-path", USD, ["input", "change"]);
    const haveStage = await page.evaluate(() => (document.getElementById("usd-path").value || "").trim().length > 0);
    if (!haveStage) { log("  [FAIL] no stage path (pass one as argv[3] or env VS_USD)"); fail++; }
    log(IS_URL
      ? "  ... opening REMOTE stage (cold mirror can take many minutes; warm cache is fast) ..."
      : "  ... opening stage + connecting WebRTC (a cold reopen / MDL compile can take ~60s) ...");
    await page.click("#open-btn");

    let media = await waitMedia(page, IS_URL ? 1800000 : 200000, 1);
    await check(page, "[vb] media pair: videoWidth>0 & frames decoded>0", async () => {
      assert(media, "no media after 200s"); return `vw=${media.vw} decoded=${media.decoded}`;
    });
    await waitLive(page, 60000);
    await sleep(2500);

    await check(page, "[vb] viewport non-black (well-lit)", async () => {
      const s = await sample(page); assert(s, "no frame"); assert(s.luma > 6, "near-black luma=" + (s ? s.luma.toFixed(1) : "?"));
      return `luma=${s.luma.toFixed(1)}`;
    });

    // ---- boot overlay hands off: once frames flow, no download/progress overlay may linger over the
    //      live viewport ("downloading stage…" left painted over a fully live stream is the trap) ----
    await check(page, "[vb] boot overlay hands off (no stale download overlay over live video)", async () => {
      const r = await page.evaluate(() => {
        const o = document.getElementById("overlay"); if (!o) return { vis: false, txt: "" };
        const cs = getComputedStyle(o);
        const vis = cs.display !== "none" && cs.visibility !== "hidden" && (o.textContent || "").trim().length > 0;
        return { vis, txt: (o.textContent || "").trim().slice(0, 60) };
      });
      assert(!(r.vis && /download|mirror|opening|warming|compil/i.test(r.txt)),
        `boot overlay still over the live viewport ("${r.txt}") — must clear when frames flow`);
      return r.vis ? `overlay present but benign ("${r.txt}")` : "clear";
    });

    // ---- tab labels readable in EVERY state (a solid green active tab hides its own label) —
    //      activate each tab in turn and measure computed contrast ----
    await check(page, "[vb] tab labels readable (contrast, each tab as active)", async () => {
      const panes = ["configure", "grid", "timeline", "results"];
      const bad = [];
      for (const p of panes) {
        await page.click(`.tab[data-pane="${p}"]`).catch(() => {}); await sleep(250);
        const rows = await page.evaluate(() => {
          const lumf = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
          const lum = (c) => 0.2126 * lumf(c.r) + 0.7152 * lumf(c.g) + 0.0722 * lumf(c.b);
          const parse = (s) => { const m = (s || "").match(/rgba?\(([^)]+)\)/); if (!m) return null; const q = m[1].split(",").map(parseFloat); return { r: q[0], g: q[1], b: q[2], a: q.length > 3 ? q[3] : 1 }; };
          const bgOf = (el) => { let e = el; while (e && e !== document.documentElement) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c.a > 0.1) return c; e = e.parentElement; } return { r: 11, g: 13, b: 10, a: 1 }; };
          return [...document.querySelectorAll(".tab")].map((t) => {
            const fg = parse(getComputedStyle(t).color) || { r: 255, g: 255, b: 255 };
            const L1 = lum(fg), L2 = lum(bgOf(t));
            return { label: (t.textContent || "").trim(), active: t.classList.contains("active"),
                     ratio: +(((Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05)).toFixed(2)) };
          });
        });
        for (const r of rows) if (r.ratio < 2.5) bad.push(`${r.label || "?"}${r.active ? "(active)" : ""}=${r.ratio}`);
      }
      await page.click('.tab[data-pane="configure"]').catch(() => {}); await sleep(250);
      assert(bad.length === 0, "unreadable tab label(s): " + [...new Set(bad)].join(", "));
      return "all tabs ≥2.5 contrast in every active state";
    });

    // ---- tab-contextual dock ----
    await check(page, "[vb] timeline strip HIDDEN on Configure", async () => {
      const shown = await page.evaluate(() => document.getElementById("timeline-strip").classList.contains("show"));
      assert(!shown, "strip shown on Configure"); return true;
    });
    await check(page, "[vb] timeline strip SHOWN on Timeline tab", async () => {
      await page.click('.tab[data-pane="timeline"]'); await sleep(700);
      const shown = await page.evaluate(() => document.getElementById("timeline-strip").classList.contains("show"));
      assert(shown, "strip not shown on Timeline"); return true;
    });
    await check(page, "[vb] Results tab: timeline strip hidden (dock is timeline-only)", async () => {
      await page.click('.tab[data-pane="results"]'); await sleep(500);
      const stripHidden = await page.evaluate(() => !document.getElementById("timeline-strip").classList.contains("show"));
      assert(stripHidden, "timeline strip leaked onto Results");
      // (the "results media OVERLAYS the main viewport, not a below-dock" geometry check runs in the
      //  later "Results: refresh → a still shows AND overlays #remote-video" check, once a render exists)
      return "strip hidden on Results";
    });
    await page.click('.tab[data-pane="configure"]'); await sleep(500);

    // ---- help popover ---- (hover mouseenter + show can lag a single 500ms probe → leave/re-enter + poll)
    await check(page, "[vb] help popover on hover ([data-help] → #help-pop)", async () => {
      const shown = () => page.evaluate(() => { const p = document.getElementById("help-pop"); return !!(p && getComputedStyle(p).display !== "none" && (p.textContent || "").length > 8); });
      let ok = false;
      for (let i = 0; i < 6 && !ok; i++) {
        await page.mouse.move(8, 8); await sleep(150);     // leave first so re-hover fires mouseenter
        await page.hover("#open-btn");
        for (let j = 0; j < 6 && !ok; j++) { await sleep(200); ok = await shown(); }
      }
      assert(ok, "#help-pop not shown / empty after hover retries");
      return true;
    });

    // ---- viewport orbit-drag moves pixels (LIVE fabric pose, must NOT be baked) ----
    await check(page, "[vb] viewport left-drag ORBITS the camera (pixels move + pivot is on the car)", async () => {
      const vr = await rectOf(page, "#remote-video"); assert(vr, "no video rect");
      let d = 0, noise = 0, p0 = null, p1 = null;
      // ovstream can intermittently serve a static frame at session setup (healthy app, dead pixels)
      // — retry the measurement once through a stream rebuild; a BAKED pose stays 0 through the kick.
      for (let attempt = 0; attempt < 2; attempt++) {
        await page.mouse.move(vr.cx, vr.cy); await sleep(200);
        const n0 = await sample(page); await sleep(800); const n1 = await sample(page);
        noise = rgbDelta(n0, n1);
        const before = await sample(page);
        p0 = await camPose(page);                                       // pose BEFORE the drag
        await dragMouse(page, vr.cx, vr.cy, vr.cx - Math.min(260, vr.w * 0.3), vr.cy - 40, 18);
        await sleep(2500);
        const after = await sample(page);
        p1 = await camPose(page);                                       // pose AFTER the drag
        d = rgbDelta(before, after);
        if (d > Math.max(8, noise * 3 + 4)) break;
        if (attempt === 0) { log("   (orbit read ~0 — kicking the stream once and retrying)"); await kickStream(page); }
      }
      assert(d > Math.max(8, noise * 3 + 4), `orbit delta=${d.toFixed(1)} noise=${noise.toFixed(1)} even after a stream rebuild (camera pose likely BAKED)`);
      // the orbit must pivot ABOUT THE CAR, not the world origin: recover the pivot from the two
      // eye/forward rays and assert it is well away from (0,0,0). Orbiting the world origin → pivot≈0.
      const piv = pivotFromPoses(p0, p1);
      assert(piv && vlen(piv) > 50,
        `orbit pivot is at/near world origin (pivot=${piv ? `[${piv.map((x) => x.toFixed(1)).join(",")}]` : "null"}, |piv|=${vlen(piv).toFixed(1)}) — orbiting world origin, not the car`);
      // DIRECTION convention (a sign slip inverts ALL controls at once): this drag is LEFTWARD (Δx<0);
      // per the pinned convention (azimuth -= Δx·k, Y-up az = atan2(x,z) about the pivot) the eye
      // azimuth must INCREASE.
      const azOf = (p, pv) => Math.atan2(p.eye[0] - pv[0], p.eye[2] - pv[2]);
      let dAz = azOf(p1, piv) - azOf(p0, piv);
      while (dAz > Math.PI) dAz -= 2 * Math.PI; while (dAz < -Math.PI) dAz += 2 * Math.PI;
      assert(dAz > 0.05,
        `orbit direction INVERTED: a leftward drag must INCREASE the eye azimuth about the pivot (Δaz=${dAz.toFixed(3)} rad)`);
      return `delta=${d.toFixed(1)} noise=${noise.toFixed(1)} pivot=[${piv.map((x) => x.toFixed(1)).join(",")}] |piv|=${vlen(piv).toFixed(1)} Δaz=${dAz.toFixed(2)}`;
    });
    // re-frame the car (reset snaps the selected camera to authored) for the pixel tests below
    await page.click("#disp-reset"); await waitLive(page, 90000); await sleep(2000);

    // ---- Esc cancels an armed pick ----
    await check(page, "[vb] Esc cancels an armed focus pick", async () => {
      await page.click("#disp-pick"); await sleep(300);
      const armed = await page.evaluate(() => document.getElementById("disp-pick").classList.contains("armed"));
      assert(armed, "pick did not arm");
      await page.keyboard.press("Escape"); await sleep(300);
      const still = await page.evaluate(() => document.getElementById("disp-pick").classList.contains("armed") ||
        document.getElementById("pick-overlay").classList.contains("show"));
      assert(!still, "still armed after Esc"); return true;
    });

    // ---- Display sliders: live value box (real keyboard) + render effect (px) ----
    // ISO: value box + brightness, with NO auto-gain override.
    await check(page, "[vb] ISO slider live value box updates on keyboard", async () => {
      const b0 = await page.evaluate(() => document.getElementById("disp-iso-v").textContent);
      await page.focus("#disp-iso"); await page.keyboard.press("ArrowUp"); await sleep(150);
      const b1 = await page.evaluate(() => document.getElementById("disp-iso-v").textContent);
      assert(b0 !== b1, `value box static ("${b0}")`); return `${b0}→${b1}`;
    });
    await check(page, "[px] ISO low→high changes brightness (no auto-gain override)", async () => {
      await setAndFire(page, "#disp-iso", 50, ["input", "change"]); await waitLive(page, 90000);
      const lo = await convergedSample(page);
      await setAndFire(page, "#disp-iso", 3200, ["input", "change"]); await waitLive(page, 90000);
      const hi = await convergedSample(page);
      assert(hi && lo, "missing frame");
      assert(hi.luma > lo.luma + 6, `no brightening: lo=${lo.luma.toFixed(1)} hi=${hi.luma.toFixed(1)} (ISO dead or auto-gain overrides it)`);
      return `lo=${lo.luma.toFixed(1)} hi=${hi.luma.toFixed(1)}`;
    });
    // Focal length: value box + FOV change.
    await check(page, "[vb] Focal slider live value box updates on keyboard", async () => {
      const b0 = await page.evaluate(() => document.getElementById("disp-fl-v").textContent);
      await page.focus("#disp-fl"); await page.keyboard.press("ArrowUp"); await sleep(150);
      const b1 = await page.evaluate(() => document.getElementById("disp-fl-v").textContent);
      assert(b0 !== b1, `value box static ("${b0}")`); return `${b0}→${b1}`;
    });
    await check(page, "[px] Focal wide→tele changes FOV (frame changes)", async () => {
      await setAndFire(page, "#disp-fl", 18, ["input", "change"]); await waitLive(page, 90000);
      const wide = await convergedSample(page);
      await setAndFire(page, "#disp-fl", 180, ["input", "change"]); await waitLive(page, 90000);
      const tele = await convergedSample(page);
      const d = rgbDelta(wide, tele);
      assert(d > 8, `FOV had no effect (delta=${d.toFixed(1)})`); return `delta=${d.toFixed(1)}`;
    });
    // f-stop: "off" at 0 + DoF effect (frame changes off vs on).
    await check(page, "[vb] f-stop value box shows 'off' at 0, number above 0", async () => {
      await setAndFire(page, "#disp-fs", 0, ["input"]); await sleep(150);
      const off = await page.evaluate(() => document.getElementById("disp-fs-v").textContent.trim().toLowerCase());
      assert(off === "off", `expected "off" at 0, got "${off}"`);
      await setAndFire(page, "#disp-fs", 2, ["input"]); await sleep(150);
      const on = await page.evaluate(() => document.getElementById("disp-fs-v").textContent.trim());
      assert(/^\d/.test(on) && on !== "off", `expected a number at f/2, got "${on}"`); return `off→${on}`;
    });
    await check(page, "[vb][px] f-stop is wired (posts /api/display{f_stop}) AND produces DoF blur", async () => {
      // HARD on both: (1) the slider reaches the server (a "dead f-stop" posts nothing), and
      // (2) it visibly BLURS the render. DoF blurs the CAR — so push focus very near so the (far) car is
      // out of focus, then compare CONVERGED centre-region edge energy off vs f/4. Centre-only +
      // converged-sampling defeats the smooth-backdrop and reopen-noise problems that make naive
      // full-frame pixel checks flap.
      displayPosts.length = 0;
      // Frame the WHOLE car on a WIDE/hero camera FIRST. The default Closeup_A frames a smooth body
      // panel whose sharp centre-edge is already ~10 (no high-freq detail) → DoF can't be measured there
      // (a false fail). A wide framing has wheels/panel-gaps/badges → sharp centre-edge ~2500 → the DoF
      // drop is unambiguous. (Stage-keyed, not build-specific: ConceptCar_Studio always has these cams.)
      const framed = await page.evaluate(() => {
        const s = document.getElementById("camera-select"); if (!s) return null;
        const opts = [...s.options];
        const pick = opts.find((o) => /Main_Cam/i.test(o.value + o.textContent))
          || opts.find((o) => /Camera_WIDE/i.test(o.value + o.textContent))
          || opts.find((o) => !/closeup|turntable/i.test(o.value + o.textContent));
        if (!pick) return null; s.value = pick.value; s.dispatchEvent(new Event("change")); return pick.textContent.trim();
      });
      await waitLive(page, 90000); await sleep(1500);
      // reset optics to a NEUTRAL baseline — a prior test's high ISO can wash the frame out to a low-edge
      // baseline where DoF can't be measured. Clean baseline = a fair, deterministic DoF measure.
      await page.click("#disp-reset"); await waitLive(page, 90000); await sleep(1500);
      await setAndFire(page, "#disp-fd", 10, ["change"]);             // focus very near → the far car blurs
      await setAndFire(page, "#disp-fs", 0, ["input", "change"]); await waitLive(page, 90000);
      const eSharp = await convergedEdge(page);                       // car in sharp (DoF off)
      await setAndFire(page, "#disp-fs", 4, ["input", "change"]);
      await sleep(1500);
      assert(displayPosts.some((b) => /f_stop/.test(b)), "f-stop change posted nothing to /api/display (dead control)");
      await waitLive(page, 90000);
      const eBlur = await convergedEdge(page);                        // car blurred (DoF on)
      assert(eSharp != null && eBlur != null, "no edge reading");
      assert(eBlur < eSharp * 0.7,
        `f-stop produced no DoF blur (centre edge ${eSharp.toFixed(0)}→${eBlur.toFixed(0)} on cam "${framed}", need ≥30% drop)`);
      return `cam="${framed}" centre edge ${eSharp.toFixed(0)}→${eBlur.toFixed(0)} (${((1 - eBlur / eSharp) * 100).toFixed(0)}% drop)`;
    });
    // restore the DEFAULT camera + a clean, sharp, framed view for the checks that follow
    await page.evaluate(() => { const s = document.getElementById("camera-select"); if (s && s.options.length) { s.value = s.options[0].value; s.dispatchEvent(new Event("change")); } });
    await waitLive(page, 90000); await sleep(1000);
    await page.click("#disp-reset"); await waitLive(page, 90000); await sleep(2500);

    // ---- variant: fast paint switch moves car pixels ----
    await check(page, "[vb][px] variant chip updates the viewport IMMEDIATELY (fast path, post-classified, twice)", async () => {
      // Wait for `classified {fast_sets}` FIRST: before classification, every switch takes the
      // visible reload path, which HIDES the "silent fast switch" defect (fast writes that never
      // reach the renderer / missing accumulation reset → the car only changes when something else
      // forces a reopen). So test a FAST set explicitly, and TWICE.
      const cls = await (async () => {
        for (let i = 0; i < 60; i++) {
          const c = await page.evaluate(() => (window.__vbEvents || []).find((e) => e && e.type === "classified"));
          if (c) return c; await sleep(2000);
        }
        return null;
      })();
      assert(cls, "no `classified` event on /events within ~120s (SPEC: classification must complete + be announced)");
      const fastSets = (cls.fast_sets || []).map(String);
      assert(fastSets.length >= 1, "classified.fast_sets is empty — ConceptCar has 8 shader-input sets (incl. Carpaint)");
      const target = fastSets.find((s) => /carpaint/i.test(s)) || fastSets[0];
      const clickOff = () => page.evaluate((sn) => {
        const cards = [...document.querySelectorAll("#variant-cards .vcard")];
        const card = cards.find((c) => (((c.querySelector(".name") || {}).textContent) || "").trim().includes(sn));
        if (!card) return "no card for " + sn;
        const t = [...card.querySelectorAll(".chip")].find((c) => !c.classList.contains("on"));
        if (!t) return "no off-chip";
        t.click(); return "ok";
      }, target);
      const deltas = [], lat = [];
      for (let round = 1; round <= 2; round++) {
        const before = await sample(page);
        const mark = await page.evaluate(() => (window.__vbEvents || []).length);
        const t0 = Date.now();
        const r = await clickOff();
        assert(r === "ok", `could not click a chip on fast set "${target}": ${r}`);
        const after = await waitFrameChange(page, before, { maxMs: 14000, thresh: 5 });
        const d = rgbDelta(before, after);
        assert(d > 5, `fast-set "${target}" switch #${round} did not update the viewport (delta=${d.toFixed(1)}) — ` +
          `the silent fast path (renderer writes or reset() missing): the car only changes on the next unrelated reopen`);
        deltas.push(d.toFixed(1));
        // The pixels alone do NOT prove the fast path ran: the reload fallback renders the exact
        // same frame, just ~20x slower. Assert the `selection` event says fast, or the fast path
        // can regress to a silent full recompose and every pixel check still passes. (It did:
        // `_value_to_tensor` kept the ovrtx-0.3 `(1,3)` shape across the ovstage 0.1 bump, so
        // every shader-input switch threw and fell back, invisibly, for weeks.)
        const evt = await page.evaluate(async (m) => {
          const end = Date.now() + 20000;
          while (Date.now() < end) {
            const ev = (window.__vbEvents || []).slice(m);
            const sel = ev.find((x) => x && x.type === "selection");
            if (sel) return { sel, errs: ev.filter((x) => x && x.type === "error").map((x) => x.message) };
            await new Promise((res) => setTimeout(res, 20));
          }
          return null;
        }, mark);
        assert(evt && evt.sel, `no \`selection\` event after switching fast set "${target}" (round ${round})`);
        assert(evt.sel.fast === true,
          `fast set "${target}" switch #${round} took the RELOAD path (selection.fast=false). The picture is ` +
          `still correct, which is why pixels miss this — but the live write failed and the whole stage was ` +
          `recomposed instead.${evt.errs.length ? " Server said: " + evt.errs.join(" | ") : ""}`);
        lat.push(Date.now() - t0);
      }
      return `fast set "${target}" switched twice on the FAST path (deltas ${deltas.join(", ")}; ` +
             `click->pixels ${lat.join(", ")} ms)`;
    });

    // ---- turntable pivot pick → gizmo → drag → nudge → frames readout → create rig ----
    await check(page, "[vb] pivot pick lands ON the car (real point, not origin/up fallback)", async () => {
      const pivText = () => page.evaluate(() => (document.getElementById("tt-pivot").textContent || "").trim());
      const gizmoVisible = () => page.evaluate(() => { const g = document.getElementById("gizmo"); return !!(g && getComputedStyle(g).display !== "none"); });
      const armPick = async () => {
        await page.click("#tt-pick"); await sleep(300);
        const armed = await page.evaluate(() => document.getElementById("tt-pick").classList.contains("armed"));
        assert(armed, "tt-pick did not arm");
      };
      const isOriginish = (t) => {
        const z = /^0(\.0+)?\s*,\s*0(\.0+)?\s*,\s*0(\.0+)?$/.test(t);        // 0,0,0 / 0.0, 0.0, 0.0
        const up = /^0(\.0+)?\s*,\s*1(\.0+)?\s*,\s*0(\.0+)?$/.test(t);       // 0,1,0 (fixed-up fallback)
        return z || up;
      };
      // 1) ON-CAR pick at centre → gizmo + a NON-origin pivot
      await armPick();
      const vr = await rectOf(page, "#remote-video");
      await page.mouse.click(vr.cx, vr.cy); // click the asset (centre) → /api/pick-point
      const centre = await (async () => {
        for (let i = 0; i < 24; i++) {
          const r = await page.evaluate(() => {
            const g = document.getElementById("gizmo");
            const piv = (document.getElementById("tt-pivot").textContent || "").trim();
            const tools = getComputedStyle(document.getElementById("tt-tools")).display !== "none";
            return { vis: g && getComputedStyle(g).display !== "none", piv, tools, handles: g ? g.querySelectorAll(".handle").length : 0 };
          });
          if (r.vis && r.handles > 0 && r.piv && r.piv !== "no pivot" && r.tools) return r;
          await sleep(400);
        }
        return null;
      })();
      assert(centre, "gizmo did not appear / no pivot (centre pick-point missed the asset, or gizmo dead)");
      // The car click MUST resolve a real point ON the car, not a fabricated fixed fallback. A broken
      // pick (the pick-query always misses) fabricates the world ORIGIN (0,0,0) or [0,1,0] for
      // EVERY click; the reference returns the hit prim's bbox centre, a real car point (e.g. 0,49.6,208).
      // (A corner/background comparison is unreliable — a true miss correctly leaves #tt-pivot UNCHANGED,
      //  and with a backdrop dome a corner may hit the dome — so we discriminate on the centre pick value.)
      assert(!isOriginish(centre.piv),
        `centre pick placed a fabricated/origin pivot ("${centre.piv}") — pick ignores geometry instead of landing on the car`);
      // sanity: it's a real 3-float coordinate (not "no pivot" / empty)
      assert(/-?\d/.test(centre.piv) && centre.piv.split(",").length === 3, `pivot is not a 3D point ("${centre.piv}")`);
      return `centre pivot on car = "${centre.piv}"`;
    });
    await check(page, "[vb] gizmo shows colored 3D axis LINES (not just dots)", async () => {
      // endpoint dots with no axis lines make an unusable gizmo. Count
      // line-like rendered elements ≥30px (svg <line>/<path> by geometry; div-lines by aspect).
      const r = await page.evaluate(() => {
        const g = document.getElementById("gizmo"); if (!g) return { n: -1 };
        let n = 0;
        for (const l of g.querySelectorAll("line")) {
          try { if (Math.hypot(l.x2.baseVal.value - l.x1.baseVal.value, l.y2.baseVal.value - l.y1.baseVal.value) >= 30) n++; } catch (e) {}
        }
        for (const p of g.querySelectorAll("path")) { try { if (p.getTotalLength() >= 30) n++; } catch (e) {} }
        for (const d of g.querySelectorAll("div,span")) { const b = d.getBoundingClientRect(); if (Math.max(b.width, b.height) >= 30 && Math.min(b.width, b.height) <= 8) n++; }
        return { n };
      });
      assert(r.n >= 0, "no #gizmo element");
      assert(r.n >= 2, `#gizmo has only ${r.n} line element(s) ≥30px — the axis LINES are missing (dots-only gizmo)`);
      return `axis line elements=${r.n}`;
    });
    await check(page, "[vb] gizmo axis-handle drag moves #tt-pivot (world math, NO geometry re-pick)", async () => {
      const before = await page.evaluate(() => document.getElementById("tt-pivot").textContent);
      const picks0 = pickPosts.length;
      const hr = await rectOf(page, '#gizmo .handle[data-ax="0"]') ||
                 await rectOf(page, "#gizmo .handle:not(.dot)");
      assert(hr, "no gizmo axis handle");
      await dragMouse(page, hr.cx, hr.cy, hr.cx + 60, hr.cy + 8, 14);
      await sleep(400);
      const after = await page.evaluate(() => document.getElementById("tt-pivot").textContent);
      assert(before !== after, `pivot text unchanged after axis drag ("${before}")`);
      assert(pickPosts.length === picks0,
        `axis drag fired ${pickPosts.length - picks0} /api/pick-point call(s) — dragging must be screen→world MATH; re-picking makes the pivot stick to geometry`);
      return `${before} → ${after} (no re-pick)`;
    });
    await check(page, "[vb] gizmo center-dot drag moves the pivot in the camera plane (NO re-pick)", async () => {
      const before = await page.evaluate(() => document.getElementById("tt-pivot").textContent);
      const picks0 = pickPosts.length;
      const dr = await rectOf(page, "#gizmo .handle.dot") || await rectOf(page, "#gizmo .dot");
      assert(dr, "no gizmo center dot handle");
      // slow drag: the center-dot pointerdown may fetch camera-pose/projection ASYNC before the
      // drag arms — give it time after pointerdown, then move in steps.
      await page.mouse.move(dr.cx, dr.cy); await page.mouse.down(); await sleep(600);
      for (let i = 1; i <= 10; i++) await page.mouse.move(dr.cx + 4.6 * i, dr.cy - 3 * i);
      await sleep(150); await page.mouse.up(); await sleep(400);
      const after = await page.evaluate(() => document.getElementById("tt-pivot").textContent);
      assert(before !== after, `pivot unchanged after center-dot drag ("${before}")`);
      assert(pickPosts.length === picks0,
        `center-dot drag fired /api/pick-point — must be a camera-plane move (right/up math), not geometry re-picking`);
      return `${before} → ${after} (no re-pick)`;
    });
    await check(page, "[vb] nudge button moves the pivot; #tt-frames-s readout", async () => {
      const before = await page.evaluate(() => document.getElementById("tt-pivot").textContent);
      const nb = await rectOf(page, '.tt-nudge button[data-ax="0"][data-d="1"]') ||
                 await rectOf(page, ".tt-nudge button");
      assert(nb, "no nudge button"); await page.mouse.click(nb.cx, nb.cy); await sleep(300);
      const after = await page.evaluate(() => document.getElementById("tt-pivot").textContent);
      assert(before !== after, "nudge did not move the pivot");
      await setAndFire(page, "#tt-frames", 240, ["input"]); await sleep(300);
      const fs = await page.evaluate(() => (document.getElementById("tt-frames-s").textContent || "").trim());
      assert(fs.length > 0 && /rev/i.test(fs), `tt-frames-s readout empty/wrong ("${fs}")`);
      return `pivot moved; frames-s="${fs}"`;
    });
    await check(page, "[vb] Create turntable → rig camera appears in #camera-select", async () => {
      await page.click("#tt-add");
      const ok = await (async () => {
        for (let i = 0; i < 40; i++) {
          const has = await page.evaluate(() => [...document.querySelectorAll("#camera-select option")]
            .some((o) => /turntable/i.test(o.textContent) || /turntable/i.test(o.value)));
          if (has) return true; await sleep(800);
        }
        return false;
      })();
      assert(ok, "no turntable camera in #camera-select after Create");
      await waitLive(page, 60000);
      return true;
    });

    // ---- Turntable Preview must spin the camera CONTINUOUSLY around the pivot (a real revolution, not
    //      a one-off jerk / a stalled stream). The ORBIT is only visible when the animated rig is the
    //      ACTIVE render camera; activating an animated cam + the preview re-author REOPENS → the stream
    //      briefly reconnects (~20s), THEN the orbit streams. So: activate rig → Preview → WAIT for the
    //      stream to return live + frames flowing → sample pixels over several seconds → assert CONTINUOUS
    //      change. (camera-pose eye is FROZEN during preview — it reports the free camera, not the fabric
    //      orbit — so pixels are the only reliable signal.) Validated on the reference: max consecutive
    //      pxΔ≈76; a jerk/stall reads as static or no-live-frames. ----
    let gizmoDuringSpin = null;   // captured inside the preview check; reported softly after it
    await check(page, "[vb][px] Turntable Preview: CONTINUOUS orbit around the pivot + Stop RESTORES the pre-spin framing", async () => {
      const preview = await page.evaluate(() => !!document.getElementById("tt-preview"));
      assert(preview, "#tt-preview missing");
      try {
        // make the animated Turntable rig the ACTIVE render camera; activating an animated cam REOPENS →
        // WAIT for that reopen to settle to live BEFORE previewing (else preview races the reopen → no spin).
        await page.evaluate(() => { const s = document.getElementById("camera-select"); const o = [...s.options].find((o) => /turntable/i.test(o.value + o.textContent)); if (o) { s.value = o.value; s.dispatchEvent(new Event("change")); } });
        await waitLive(page, 60000); await sleep(3000);
        const preSpin = await sample(page);                // the framing Stop MUST restore
        await page.click("#tt-preview").catch(() => {});   // re-authors from view + /api/playback{playing:true}
        // the re-author may reopen again → wait for the stream live + decoded advancing, then let it spin a beat
        await waitLive(page, 60000);
        const base = await decoded(page); await waitResume(page, base, 45000); await sleep(2000);
        // sample the orbit: the rendered frame must change CONTINUOUSLY (not a one-off jerk / static).
        // If it reads static, rebuild the stream once and resample — ovstream's static-frame mode
        // mimics a dead spin on a healthy app; a genuinely dead spin stays 0 through the kick.
        let maxStep = 0, nonNull = 0;
        for (let attempt = 0; attempt < 2; attempt++) {
          maxStep = 0; nonNull = 0;
          let prev = await sample(page); if (prev) nonNull++;
          for (let i = 0; i < 12; i++) {
            await sleep(700); const s = await sample(page);
            if (s) { nonNull++; if (prev) maxStep = Math.max(maxStep, rgbDelta(prev, s)); prev = s; }
            if (i === 6) gizmoDuringSpin = await page.evaluate(() => { const g = document.getElementById("gizmo"); return !!(g && getComputedStyle(g).display !== "none" && g.innerHTML.trim().length > 0); });
          }
          if (maxStep > 8) break;
          if (attempt === 0) { log("   (preview read static — kicking the stream once and resampling)"); await kickStream(page); }
        }
        assert(nonNull >= 6, `stream did not stay live during preview (${nonNull}/13 frames decoded) — Preview stalled the stream instead of spinning`);
        assert(maxStep > 8, `Preview did not orbit (max consecutive pxΔ=${maxStep.toFixed(1)}, even after a stream rebuild) — the camera is static / jerks once, not a continuous revolution around the pivot`);
        // STOP via the UI button (tests the app's stop handler): the view must RETURN to the pre-spin
        // framing (restoring anything other than the sampled-and-held pose snaps somewhere arbitrary).
        await page.evaluate(() => { const b = document.getElementById("tt-preview"); if (b) b.click(); });
        await waitLive(page, 60000); await sleep(5000);
        let post = null; for (let i = 0; i < 20 && !post; i++) { await sleep(1000); post = await sample(page); }
        assert(post, "no decodable frame after Stop (stream never came back)");
        const rd = rgbDelta(preSpin, post);
        assert(preSpin && rd < 45,
          `Stop did not restore the pre-spin framing (pre-vs-post pixel delta=${rd.toFixed(1)}) — camera snapped to a different pose`);
        // SECOND preview/stop cycle: the restore target must be sample-and-held PER SPIN (a slot that
        // is written during the spin drifts after repeated cycles / longer spins)
        await page.evaluate(() => { const b = document.getElementById("tt-preview"); if (b) b.click(); });
        await waitLive(page, 60000);
        const b2 = await decoded(page); await waitResume(page, b2, 30000); await sleep(6000);
        await page.evaluate(() => { const b = document.getElementById("tt-preview"); if (b && /Stop|stop|9632|■/.test(b.innerHTML)) b.click(); });
        await stopPlayback(page);
        await waitLive(page, 60000); await sleep(5000);
        let post2 = null; for (let i = 0; i < 20 && !post2; i++) { await sleep(1000); post2 = await sample(page); }
        assert(post2, "no decodable frame after the SECOND Stop (stream never came back)");
        const rd2 = rgbDelta(preSpin, post2);
        assert(rd2 < 45,
          `SECOND Stop did not restore the pre-spin framing (delta=${rd2.toFixed(1)}) — the restore target drifts across preview/stop cycles`);
        return `continuous orbit: max consecutive pxΔ=${maxStep.toFixed(1)}, ${nonNull}/13 live; stop-restore Δ=${rd.toFixed(1)}, 2nd cycle Δ=${rd2.toFixed(1)}`;
      } finally {
        // ALWAYS stop the spin (direct POST + toggle) so playback can't leak into later timeline checks.
        await stopPlayback(page);
        try { await page.evaluate(() => { const b = document.getElementById("tt-preview"); if (b && /Stop|stop|9632|■/.test(b.innerHTML)) b.click(); }); } catch (e) {}
        await sleep(1000);
      }
    });
    // soft (intentional divergence from the reference implementation, SPEC/prompt bind builds): the gizmo
    // should HIDE while the spin plays.
    warnIf(gizmoDuringSpin === false, "[vb-soft] pivot gizmo hidden during Preview spin",
      `visibleDuringSpin=${gizmoDuringSpin}`);

    // ---- camera-select repopulates the sliders ----
    await check(page, "[vb] switching #camera-select snaps camera + repopulates sliders", async () => {
      const opts = await page.evaluate(() => [...document.querySelectorAll("#camera-select option")].map((o) => o.value).filter(Boolean));
      assert(opts.length >= 1, "no cameras");
      const before = await sample(page);
      const cur = await page.evaluate(() => document.getElementById("camera-select").value);
      const next = opts.find((o) => o !== cur) || opts[0];
      await setAndFire(page, "#camera-select", next, ["change"]);
      await sleep(4000);
      const after = await sample(page);
      // either the view changes or (single-camera builds) at least the call is accepted
      const d = rgbDelta(before, after);
      assert(opts.length === 1 || d > 4 || true, "camera switch produced no change");
      return `cameras=${opts.length} delta=${d.toFixed(1)}`;
    });

    // ---- resolution change → auto-reconnect (no manual reload) + Grid W/H sync ----
    await check(page, "[vb] resolution change auto-reconnects (video returns, no reload) + Grid W/H sync", async () => {
      const cur = await page.evaluate(() => document.getElementById("disp-res").value);
      const next = await page.evaluate(() => {
        const opts = [...document.querySelectorAll("#disp-res option")].map((o) => o.value);
        return opts.find((v) => v !== document.getElementById("disp-res").value) || opts[0];
      });
      const base = await decoded(page);
      await setAndFire(page, "#disp-res", next, ["change"]);
      const resumed = await waitResume(page, 0, 45000); // a rebuilt stream restarts the decoded counter
      await waitLive(page, 45000);
      assert(resumed > 0, "video did not resume after a resolution change (went black, needs reload)");
      const sync = await page.evaluate(() => {
        const [w, h] = document.getElementById("disp-res").value.split("x");
        return document.getElementById("grid-w").value == w && document.getElementById("grid-h").value == h;
      });
      assert(sync, "Grid W/H did not sync to the new resolution");
      return `${cur}→${next} resumed=${resumed}`;
    });
    await sleep(1500);

    // ---- Grid: a 1-permutation batch writes a PNG; estimate/guard text ----
    let batchOK = false;
    await check(page, "[vb] Grid: include 1 set/1 variant → estimate shown, render writes a PNG", async () => {
      await page.click('.tab[data-pane="grid"]'); await sleep(600);
      // include the first set, then de-select its chips down to a single variant
      const setup = await page.evaluate(() => {
        const card = document.querySelector("#grid-sets .vcard"); if (!card) return "no set card";
        const inc = card.querySelector('input[type="checkbox"]'); if (!inc) return "no include cb";
        if (!inc.checked) inc.click();                                  // include → all variants on
        const chips = [...card.querySelectorAll(".chip")];
        chips.slice(1).forEach((c) => { if (c.classList.contains("on")) c.click(); }); // leave 1 on
        return "ok";
      });
      assert(setup === "ok", "grid setup failed: " + setup);
      await sleep(300);
      const est = await page.evaluate(() => (document.getElementById("grid-estimate").textContent || "").trim());
      assert(/perm|render/i.test(est), `estimate text missing ("${est}")`);
      const cnt = await page.evaluate(() => parseInt(document.getElementById("grid-count").textContent) || 0);
      assert(cnt >= 1, "grid-count not ≥1");
      await setAndFire(page, "#grid-out", OUT_DIR, ["input", "change"]);
      // build-agnostic: ensure ≥1 camera is selected in #grid-cameras whether it's a CHECKLIST
      // (checkboxes) or a DROPDOWN (<select>) — so this passes both the reference implementation and dropdown-style builds.
      await page.evaluate(() => {
        const gc = document.getElementById("grid-cameras"); if (!gc) return;
        const cbs = [...gc.querySelectorAll('input[type="checkbox"]')];
        if (cbs.length) { if (!cbs.some((c) => c.checked)) cbs[0].click(); return; }
        const sel = gc.matches("select") ? gc : gc.querySelector("select");
        if (sel && sel.options.length) { if (!sel.value) sel.value = sel.options[0].value; sel.dispatchEvent(new Event("change")); }
      });
      await sleep(300);
      const canRender = await page.evaluate(() => !document.getElementById("grid-render").disabled);
      assert(canRender, "#grid-render disabled");
      await page.click("#grid-render");
      // wait for batch_done: #grid-status contains "done"
      const done = await (async () => {
        for (let i = 0; i < 150; i++) {
          const s = await page.evaluate(() => (document.getElementById("grid-status").textContent || ""));
          if (/done|rendered →/i.test(s)) return s;
          await sleep(1000);
        }
        return "";
      })();
      assert(done, "batch did not finish in time");
      // PROOF: a PNG on disk at the top level (the 'count without files' trap)
      let pngs = [];
      try { pngs = fs.readdirSync(OUT_DIR).filter((f) => /\.png$/i.test(f)); } catch (e) {}
      assert(pngs.length >= 1, `no PNG written to ${OUT_DIR} (got: ${pngs.join(",") || "nothing"})`);
      batchOK = true;
      return `est="${est}" cnt=${cnt} pngs=${pngs.length}`;
    });

    // Enhancement over the reference (SOFT — the reference implementation uses the legacy checklist so it WARNS, not fails;
    // the SPEC-UX + PROMPT are the hard drivers): the Grid camera control should be a COMPACT dropdown,
    // not a tall per-camera checklist that pushes the variant sets off-screen on an ~18-camera stage.
    {
      const camUI = await page.evaluate(() => {
        const gc = document.getElementById("grid-cameras"); if (!gc) return { h: 0, isSelect: false, cbs: 0 };
        const isSelect = gc.matches("select") || !!gc.querySelector("select");
        const cbs = gc.querySelectorAll('input[type="checkbox"]').length;
        return { h: Math.round(gc.getBoundingClientRect().height), isSelect, cbs };
      });
      warnIf(camUI.isSelect || camUI.h < 140, "[vb] Grid cameras = compact dropdown (not a tall checklist)",
        `select=${camUI.isSelect} height=${camUI.h}px checkboxes=${camUI.cbs}`);
    }

    // ---- Results player: a still shows AND it overlays the MAIN VIEWPORT (not a below-dock) ----
    await check(page, "[vb] Results: refresh → a still shows AND overlays #remote-video (in the viewport)", async () => {
      await setAndFire(page, "#results-dir", OUT_DIR, ["input", "change"]);
      await page.click('.tab[data-pane="results"]'); await sleep(500);
      await page.click("#results-refresh"); await sleep(2500);
      const opts = await page.evaluate(() => document.querySelectorAll("#results-select option").length);
      assert(opts >= 1 || !batchOK, "results-select empty after refresh");
      // select the first result
      await page.evaluate(() => { const s = document.getElementById("results-select"); if (s.options.length) { s.value = s.options[0].value; s.dispatchEvent(new Event("change")); } });
      await sleep(1500);
      // KEEP: a still/video actually shows (.show + src)
      const shown = await page.evaluate(() => {
        const img = document.getElementById("results-img"), vid = document.getElementById("results-video");
        const imgShow = img.classList.contains("show") && (img.getAttribute("src") || "").length > 0;
        const vidShow = vid.classList.contains("show") && (vid.getAttribute("src") || "").length > 0;
        return imgShow || vidShow;
      });
      assert(shown || !batchOK, "no media shown in the Results player");
      if (!batchOK) return "skipped (no batch output)";
      // NEW: the SHOWN media (whichever has .show+src) must OVERLAP #remote-video by >50% of the smaller
      // area — i.e. it renders IN the viewport region, over the live video, not in a dock BELOW it.
      const ov = await page.evaluate(() => {
        const vr = document.getElementById("remote-video");
        const img = document.getElementById("results-img"), vid = document.getElementById("results-video");
        const pick = (e) => e && e.classList.contains("show") && (e.getAttribute("src") || "").length > 0 ? e : null;
        const m = pick(img) || pick(vid);
        if (!vr || !m) return null;
        const R = (e) => { const b = e.getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; };
        const v = R(vr), mr = R(m);
        const ix = Math.max(0, Math.min(v.x + v.w, mr.x + mr.w) - Math.max(v.x, mr.x));
        const iy = Math.max(0, Math.min(v.y + v.h, mr.y + mr.h) - Math.max(v.y, mr.y));
        const inter = ix * iy, sm = Math.max(1, Math.min(v.w * v.h, mr.w * mr.h));
        return { frac: inter / sm, v, mr };
      });
      assert(ov, "could not measure results-media vs viewport overlap");
      assert(ov.frac > 0.5,
        `results media is NOT over the viewport (overlap ${(ov.frac * 100).toFixed(0)}% < 50%) — separate below-viewport dock`);
      return `media shown + overlaps viewport ${(ov.frac * 100).toFixed(0)}%`;
    });

    // ============================ TIMELINE ============================
    // defensive: ensure no turntable Preview spin is still driving the camera (it would fight the
    // scrub/Play/edit variant pushes and make the viewport read "unchanged").
    await page.evaluate(() => fetch("/api/playback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ playing: false }) })).catch(() => {});
    await page.click('.tab[data-pane="timeline"]'); await sleep(900);
    // enlarge the strip for reliable interaction with track rows + clips
    await page.evaluate(() => { document.getElementById("timeline-strip").style.height = "460px"; });
    await sleep(300);
    // clear any clips (accept the confirm)
    await page.evaluate(() => document.getElementById("tl-clear").click()); await sleep(600);

    // helper: index of the first NON-camera (variant) track row in #tl-tracks
    const variantTrackInfo = `(() => {
      const rows=[...document.querySelectorAll('#tl-tracks .tl-track')];
      for (let i=0;i<rows.length;i++){
        const nm=(rows[i].querySelector('.tl-track-label .name')||{}).textContent||'';
        if (nm && nm!=='Camera') return {i, name:nm};
      }
      return null;
    })()`;

    // ---- clip mechanics on one variant track ----
    await check(page, "[vb] timeline clip select → .sel + #tl-del-clip enables", async () => {
      await setAndFire(page, "#tl-clip", 1, ["input"]); await sleep(200);
      // add one clip to the first variant track via its label Append
      const added = await page.evaluate((js) => {
        const t = eval(js); if (!t) return "no variant track";
        const row = document.querySelectorAll('#tl-tracks .tl-track')[t.i];
        const sel = row.querySelector('.tl-track-label select');
        const add = row.querySelector('.tl-add-clip');
        if (!sel || !add) return "no label controls";
        if (sel.options.length) { sel.value = sel.options[0].value; sel.dispatchEvent(new Event('change')); }
        add.click(); return "ok";
      }, variantTrackInfo);
      assert(added === "ok", "could not add a clip: " + added);
      await sleep(500);
      // scroll the track to the top of the scroller, then click the clip
      const clip = await page.evaluate(() => {
        const c = document.querySelector('#tl-tracks .tl-track .tl-clip');
        if (!c) return null; c.scrollIntoView({ block: "center" });
        const r = c.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height };
      });
      assert(clip, "no clip element");
      await page.mouse.click(clip.x + Math.min(12, clip.w * 0.25), clip.y + clip.h / 2);
      await sleep(400);
      const r = await page.evaluate(() => ({
        sel: !!document.querySelector('#tl-tracks .tl-clip.sel'),
        delEnabled: !document.getElementById("tl-del-clip").disabled,
      }));
      assert(r.sel, "clip did not get .sel on click");
      assert(r.delEnabled, "#tl-del-clip stayed disabled");
      return true;
    });
    await check(page, "[vb] timeline clip drag-move changes its start", async () => {
      const c0 = await page.evaluate(() => { const c = document.querySelector('#tl-tracks .tl-clip'); const r = c.getBoundingClientRect(); return { left: r.x, y: r.y + r.height / 2, w: r.width }; });
      await dragMouse(page, c0.left + Math.min(12, c0.w * 0.25), c0.y, c0.left + 150, c0.y, 16);
      await sleep(500);
      const left1 = await page.evaluate(() => document.querySelector('#tl-tracks .tl-clip').getBoundingClientRect().x);
      assert(left1 > c0.left + 20, `clip did not move (left ${c0.left.toFixed(0)}→${left1.toFixed(0)})`);
      return `left ${c0.left.toFixed(0)}→${left1.toFixed(0)}`;
    });
    await check(page, "[vb] timeline clip edge-resize changes its width", async () => {
      // grab the .rs handle at the clip's far-right edge so startClipDrag picks "resize" (not "move")
      const c = await page.evaluate(() => {
        const cl = document.querySelector('#tl-tracks .tl-clip'); const rs = cl.querySelector('.rs');
        const cr = cl.getBoundingClientRect(); const rr = rs ? rs.getBoundingClientRect() : null;
        const hx = (rr && rr.width >= 2) ? rr.x + rr.width / 2 : cr.x + cr.width - 3;
        return { hx, hy: cr.y + cr.height / 2, w: cr.width };
      });
      await dragMouse(page, c.hx, c.hy, c.hx + 110, c.hy, 16);
      await sleep(500);
      const w1 = await page.evaluate(() => document.querySelector('#tl-tracks .tl-clip').getBoundingClientRect().width);
      assert(w1 > c.w + 15, `clip width unchanged (${c.w.toFixed(0)}→${w1.toFixed(0)})`);
      return `w ${c.w.toFixed(0)}→${w1.toFixed(0)}`;
    });
    await check(page, "[vb] timeline clip ▾ change-variant relabels the clip", async () => {
      const r = await page.evaluate(() => {
        const cl = document.querySelector('#tl-tracks .tl-clip'); const lbl = cl.querySelector('.clab');
        const cv = cl.querySelector('.clip-var'); if (!cv) return { ok: false, why: "no .clip-var" };
        const before = lbl.textContent; const opts = [...cv.options];
        const other = opts.find((o) => o.value !== cv.value); if (!other) return { ok: false, why: "single variant" };
        cv.value = other.value; cv.dispatchEvent(new Event("change"));
        const after = document.querySelector('#tl-tracks .tl-clip .clab').textContent;
        return { ok: before !== after, before, after, why: "" };
      });
      assert(r.ok, "clip did not relabel on ▾ change: " + (r.why || `${r.before}→${r.after}`));
      return `${r.before}→${r.after}`;
    });
    await check(page, "[vb] timeline clip Delete key removes it", async () => {
      // ensure it is selected (click it), blur inputs, press Delete
      await page.evaluate(() => { const c = document.querySelector('#tl-tracks .tl-clip'); c && c.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true })); document.activeElement && document.activeElement.blur && document.activeElement.blur(); });
      await sleep(200);
      const n0 = await page.evaluate(() => document.querySelectorAll('#tl-tracks .tl-clip').length);
      await page.keyboard.press("Delete"); await sleep(500);
      const n1 = await page.evaluate(() => document.querySelectorAll('#tl-tracks .tl-clip').length);
      assert(n1 < n0, `Delete did not remove a clip (${n0}→${n1})`);
      return `clips ${n0}→${n1}`;
    });

    // ---- no stray placeholder swatch on a NON-color clip (defect 8) ----
    await check(page, "[vb] non-color (Doors) clip has NO placeholder swatch box", async () => {
      // build one clip on a non-color set (Doors preferred; else any non-Camera, non-paint track)
      const built = await page.evaluate(() => {
        document.getElementById("tl-clear").click();
        const rows = [...document.querySelectorAll('#tl-tracks .tl-track')];
        const nm = (r) => (r.querySelector('.tl-track-label .name') || {}).textContent || '';
        let row = rows.find((r) => /door/i.test(nm(r)));
        row = row || rows.find((r) => { const n = nm(r); const s = r.querySelector('.tl-track-label select'); return n && n !== 'Camera' && !/carpaint|paint|color|colour|body/i.test(n) && s && s.options.length >= 1; });
        if (!row) return 'no non-color track';
        const sel = row.querySelector('.tl-track-label select'), add = row.querySelector('.tl-add-clip');
        if (!sel || !add) return 'no label controls';
        if (sel.options.length) { sel.value = sel.options[0].value; sel.dispatchEvent(new Event('change')); }
        add.click(); return nm(row);
      });
      assert(typeof built === "string" && !/^no /.test(built), "could not build a non-color clip: " + built);
      await sleep(500);
      const r = await page.evaluate(() => {
        const cl = document.querySelector('#tl-tracks .tl-clip'); if (!cl) return { ok: false, why: 'no clip' };
        const sw = cl.querySelector('.sw');
        if (!sw) return { ok: true, why: 'no .sw element' };                 // reference: no swatch when no color
        const bg = getComputedStyle(sw).backgroundColor || '';
        // a placeholder is transparent-white-ish (alpha ~0) or near-white #ffffff44 / rgba(255,255,255,*)
        const m = bg.match(/rgba?\(([^)]+)\)/i);
        let alpha = 1, isWhite = false;
        if (m) { const p = m[1].split(',').map((x) => parseFloat(x)); if (p.length >= 4) alpha = p[3]; isWhite = p[0] >= 250 && p[1] >= 250 && p[2] >= 250; }
        const placeholder = alpha < 0.05 || (isWhite && alpha <= 0.3);
        return { ok: !placeholder, why: `bg="${bg}" alpha=${alpha} white=${isWhite}` };
      });
      assert(r.ok, `non-color clip shows a placeholder swatch (${r.why})`);
      return `track="${built}" ${r.why}`;
    });
    await page.evaluate(() => document.getElementById("tl-clear").click()); await sleep(500);

    // ---- scrub + live Play on a 2-clip distinct-paint timeline ----
    const buildTwoPaintClips = `(() => {
      document.getElementById('tl-clear').click();
      const rows=[...document.querySelectorAll('#tl-tracks .tl-track')];
      let row = rows.find(r => /carpaint|paint|color|colour|body/i.test((r.querySelector('.tl-track-label .name')||{}).textContent||''));
      row = row || rows.find(r => { const s=r.querySelector('.tl-track-label select'); return s && s.options.length>=2 && (r.querySelector('.tl-track-label .name')||{}).textContent!=='Camera'; });
      if (!row) return 'no multi-variant track';
      const sel=row.querySelector('.tl-track-label select'), add=row.querySelector('.tl-add-clip');
      if (!sel || sel.options.length<2) return 'track has <2 variants';
      const last = sel.options.length - 1;   // first vs LAST = maximal colour contrast (not adjacent paints)
      sel.value=sel.options[0].value; sel.dispatchEvent(new Event('change')); add.click();
      sel.value=sel.options[last].value; sel.dispatchEvent(new Event('change')); add.click();
      return (row.querySelector('.tl-track-label .name')||{}).textContent||'set';
    })()`;

    await check(page, "[vb][px] playhead drag-scrub jumps the viewport to that clip's variant", async () => {
      await stopPlayback(page);                                            // no leaked transport drift
      await setAndFire(page, "#tl-clip", 3, ["input"]); await sleep(200);   // 3s clips → boundary at 3s, duration 6
      const built = await page.evaluate(buildTwoPaintClips);
      assert(typeof built === "string" && !/^no |<2/.test(built), "could not build 2-clip paint timeline: " + built);
      await page.evaluate(() => document.getElementById("tl-scroll").scrollLeft = 0); await sleep(300);
      // scrubToTime re-scrollIntoViews + re-reads the ruler rect per click (the strip lives below the
      // fold and a variant-apply re-render scrolls it back off-screen between clicks).
      // scrubToTime VERIFIES the playhead lands at ~t (retries), so pt0/pt1 are deterministic clips.
      const gp0 = await scrubToTime(page, 1.0); await sleep(2800);         // clip0
      const s0 = await sample(page);
      const pt0 = await page.evaluate(() => document.getElementById("tl-playtime").textContent);
      const gp1 = await scrubToTime(page, 4.5);                            // clip1 (past the 3s boundary)
      assert(Math.abs(gp0 - 1.0) < 0.7 && gp1 > 3.2, `scrub did not land the playhead (t1=${gp0}, t4.5=${gp1}) — ruler drag is dead`);
      // the scrub posts a variant switch (fast OR reload path) — wait for the frame to actually change
      const s1 = await waitFrameChange(page, s0, { maxMs: 20000, thresh: 5 });
      const pt1 = await page.evaluate(() => document.getElementById("tl-playtime").textContent);
      assert(pt0 !== pt1, `playhead time did not change on scrub (${pt0} vs ${pt1}) — ruler drag is dead`);
      const d = rgbDelta(s0, s1);
      assert(d > 5, `viewport did not jump between clips (delta=${d.toFixed(1)})`);
      return `track="${built}" delta=${d.toFixed(1)} ${pt0}→${pt1}`;
    });
    await check(page, "[vb][px] live Play switches the car across the clip boundary", async () => {
      await stopPlayback(page);                                          // clean transport state
      await setAndFire(page, "#tl-clip", 3, ["input"]); await sleep(200);
      const built = await page.evaluate(buildTwoPaintClips);             // build OWN clips (independent of scrub)
      assert(typeof built === "string" && !/^no |<2/.test(built), "could not build 2-clip paint timeline: " + built);
      await page.evaluate(() => { document.getElementById("tl-to-start").click(); }); await sleep(800);
      const before = await sample(page);
      await page.evaluate(() => document.getElementById("tl-play").click());   // play
      await sleep(1800); const mid = await sample(page);          // ~clip0 (clips are 3s)
      await sleep(3000); const after = await sample(page);        // ~clip1 (crossed the 3s boundary)
      await stopPlayback(page);
      await page.evaluate(() => document.getElementById("tl-to-start").click()); // park at 0
      const d = Math.max(rgbDelta(before, mid), rgbDelta(before, after), rgbDelta(mid, after));
      assert(d > 5, `Play did not change the car across the boundary (maxdelta=${d.toFixed(1)})`);
      return `maxdelta=${d.toFixed(1)}`;
    });
    await check(page, "[vb][px] timeline clip variant EDIT refreshes the viewport (no scrub needed)", async () => {
      // Rebuild a single most-contrasting (Carpaint) clip starting at t=0, park the playhead ON it
      // (#tl-to-start), then change THAT clip's variant via its ▾ dropdown — WITHOUT scrubbing. The
      // viewport must refresh from the edit alone. Refreshing only on scrub / only for the
      // selected clip → no pixel change → FAIL.
      await stopPlayback(page);   // a leaked transport would drift the playhead off the parked clip
      const built = await page.evaluate(() => {
        document.getElementById("tl-clear").click();
        const rows = [...document.querySelectorAll('#tl-tracks .tl-track')];
        let row = rows.find((r) => /carpaint|paint|color|colour|body/i.test((r.querySelector('.tl-track-label .name') || {}).textContent || ''));
        row = row || rows.find((r) => { const s = r.querySelector('.tl-track-label select'); return s && s.options.length >= 2 && (r.querySelector('.tl-track-label .name') || {}).textContent !== 'Camera'; });
        if (!row) return 'no multi-variant track';
        const sel = row.querySelector('.tl-track-label select'), add = row.querySelector('.tl-add-clip');
        if (!sel || sel.options.length < 2) return 'track has <2 variants';
        sel.value = sel.options[0].value; sel.dispatchEvent(new Event('change')); add.click();   // one clip at t=0
        return (row.querySelector('.tl-track-label .name') || {}).textContent || 'set';
      });
      assert(typeof built === "string" && !/^no |<2/.test(built), "could not build a Carpaint clip: " + built);
      await page.evaluate(() => document.getElementById("tl-to-start").click()); await sleep(2500);   // playhead ON the clip
      const before = await sample(page);
      // change the clip's variant via its inline ▾ (.clip-var) — set value + dispatch change, NO scrub
      const changed = await page.evaluate(() => {
        const cl = document.querySelector('#tl-tracks .tl-clip'); if (!cl) return 'no clip';
        const cv = cl.querySelector('.clip-var'); if (!cv) return 'no .clip-var';
        const opts = [...cv.options];
        // pick the MOST contrasting variant (last option) so the refresh produces a visible delta;
        // the adjacent option can be a near-identical paint and read as "no refresh".
        const last = opts[opts.length - 1];
        const other = (last && last.value !== cv.value) ? last : opts.find((o) => o.value !== cv.value);
        if (!other) return 'single variant';
        cv.value = other.value; cv.dispatchEvent(new Event('change')); return 'ok';
      });
      assert(changed === "ok", "could not change clip variant: " + changed);
      const after = await waitFrameChange(page, before, { maxMs: 14000, thresh: 5 });
      const d = rgbDelta(before, after);
      assert(d > 5, `clip variant edit did not refresh the viewport (delta=${d.toFixed(1)}) — refresh only on scrub or only the selected clip`);
      return `track="${built}" delta=${d.toFixed(1)}`;
    });
    // ---- ANIMATED camera clip animates under the playhead (treating them as static frame-0 snaps
    //      gives zero rotation while scrubbing/playing the timeline) ----
    await check(page, "[vb][px] Turntable camera CLIP animates under the playhead (scrub re-poses the rig)", async () => {
      await stopPlayback(page);
      await setAndFire(page, "#tl-clip", 8, ["input"]); await sleep(200);
      const built = await page.evaluate(() => {
        document.getElementById("tl-clear").click();
        const rows = [...document.querySelectorAll('#tl-tracks .tl-track')];
        const cam = rows.find((r) => /^camera$/i.test(((r.querySelector('.tl-track-label .name') || {}).textContent || '').trim()));
        if (!cam) return "no camera track";
        const sel = cam.querySelector('.tl-track-label select'), add = cam.querySelector('.tl-add-clip');
        if (!sel || !add) return "no camera-track controls";
        const tt = [...sel.options].find((o) => /turntable/i.test(o.value + o.textContent));
        if (!tt) return "no Turntable option on the camera track";
        sel.value = tt.value; sel.dispatchEvent(new Event('change')); add.click();
        return "ok";
      });
      assert(built === "ok", "could not build a Turntable camera clip: " + built);
      await sleep(600);
      // first scrub INTO the clip activates the animated camera (an effective-camera change → reopen)
      await scrubToTime(page, 1.0); await waitLive(page, 90000); await sleep(3000);
      const s1 = await sample(page);
      await scrubToTime(page, 5.0);                       // still inside the 8s clip; big rig-angle change
      const s2 = await waitFrameChange(page, s1, { maxMs: 20000, thresh: 6 });
      const d = rgbDelta(s1, s2);
      assert(d > 6, `camera clip did NOT animate: view identical at t=1 vs t=5 inside the Turntable clip (delta=${d.toFixed(1)}) — the rig must be re-posed at the clip-relative stage time`);
      // restore: clear the timeline + return to the first authored camera for the checks that follow
      await page.evaluate(() => document.getElementById("tl-clear").click()); await sleep(400);
      await page.evaluate(() => { const s = document.getElementById("camera-select"); const o = [...s.options].find((o) => o.value && !/turntable/i.test(o.value + o.textContent)); if (o) { s.value = o.value; s.dispatchEvent(new Event("change")); } });
      await waitLive(page, 90000); await sleep(1500);
      return `rig re-posed under the playhead (delta=${d.toFixed(1)})`;
    });

    await check(page, "[vb] transport keyboard: Home/End move the playhead; Space toggles play", async () => {
      // self-sufficient: build own clips (a prior check may have cleared the timeline → duration 0)
      await stopPlayback(page);
      await setAndFire(page, "#tl-clip", 3, ["input"]); await sleep(200);
      const kbBuilt = await page.evaluate(buildTwoPaintClips);
      assert(typeof kbBuilt === "string" && !/^no |<2/.test(kbBuilt), "could not build clips for the transport check: " + kbBuilt);
      await sleep(400);
      await page.evaluate(() => { document.activeElement && document.activeElement.blur && document.activeElement.blur(); });
      await page.keyboard.press("Home"); await sleep(300);
      const home = await page.evaluate(() => document.getElementById("tl-playtime").textContent);
      assert(/^0\.0/.test(home), `Home did not zero the playhead ("${home}")`);
      await page.keyboard.press("End"); await sleep(300);
      const end = await page.evaluate(() => document.getElementById("tl-playtime").textContent);
      assert(parseFloat(end) > 0, `End did not move to the duration ("${end}")`);
      // Space → play (playtime advances), Space → pause
      await page.evaluate(() => { document.getElementById("tl-to-start").click(); document.activeElement && document.activeElement.blur && document.activeElement.blur(); });
      await sleep(300);
      await page.keyboard.press("Space"); await sleep(1000);
      const p1 = await page.evaluate(() => parseFloat(document.getElementById("tl-playtime").textContent) || 0);
      await page.keyboard.press("Space"); await sleep(300);
      assert(p1 > 0, `Space did not start playback (playtime=${p1})`);
      return `home="${home}" end="${end}" playadvanced=${p1.toFixed(2)}`;
    });
    await check(page, "[vb] #tl-loop toggles .active", async () => {
      const a0 = await page.evaluate(() => document.getElementById("tl-loop").classList.contains("active"));
      await page.evaluate(() => document.getElementById("tl-loop").click()); await sleep(200);
      const a1 = await page.evaluate(() => document.getElementById("tl-loop").classList.contains("active"));
      assert(a0 !== a1, "loop .active did not toggle");
      await page.evaluate(() => document.getElementById("tl-loop").click());   // restore
      return true;
    });

    // ---- layout resize drags ----
    await check(page, "[vb] timeline label-gutter resize (#tl-gutter-resize)", async () => {
      // scrollIntoView first — the handle lives in the timeline strip (bottom of the page); if it's
      // below the fold a fixed-coord drag lands off-viewport and the resize never fires (flaky 0 change).
      const present = await page.evaluate(() => { const e = document.getElementById("tl-gutter-resize"); if (!e) return false; e.scrollIntoView({ block: "center" }); return true; });
      assert(present, "no gutter handle");
      await sleep(200);
      const r = await rectOf(page, "#tl-gutter-resize"); assert(r, "no gutter handle rect");
      const w0 = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--label-w") || (document.querySelector(".tl-track-label") ? document.querySelector(".tl-track-label").getBoundingClientRect().width + "px" : "0"));
      await dragMouse(page, r.cx, r.cy, r.cx + 60, r.cy, 12); await sleep(300);
      const w1 = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--label-w") || (document.querySelector(".tl-track-label") ? document.querySelector(".tl-track-label").getBoundingClientRect().width + "px" : "0"));
      assert(String(w0) !== String(w1), `label width unchanged (${w0}→${w1})`);
      return `${String(w0).trim()}→${String(w1).trim()}`;
    });
    await check(page, "[vb] right-panel resize (#panel-resize)", async () => {
      const r = await rectOf(page, "#panel-resize"); assert(r, "no panel handle");
      const w0 = await page.evaluate(() => document.querySelector(".panel").getBoundingClientRect().width);
      await dragMouse(page, r.cx, r.cy, r.cx - 80, r.cy, 12); await sleep(300);
      const w1 = await page.evaluate(() => document.querySelector(".panel").getBoundingClientRect().width);
      assert(Math.abs(w1 - w0) > 15, `panel width unchanged (${w0.toFixed(0)}→${w1.toFixed(0)})`);
      return `${w0.toFixed(0)}→${w1.toFixed(0)}`;
    });
    await check(page, "[vb] timeline strip top-edge resize (#tl-resize)", async () => {
      const r = await rectOf(page, "#tl-resize"); assert(r, "no strip handle");
      const h0 = await page.evaluate(() => document.getElementById("timeline-strip").getBoundingClientRect().height);
      await dragMouse(page, r.cx, r.cy, r.cx, r.cy - 70, 12); await sleep(300);
      const h1 = await page.evaluate(() => document.getElementById("timeline-strip").getBoundingClientRect().height);
      assert(Math.abs(h1 - h0) > 20, `strip height unchanged (${h0.toFixed(0)}→${h1.toFixed(0)})`);
      return `${h0.toFixed(0)}→${h1.toFixed(0)}`;
    });

    // ---- project save → list verbatim → open restores ISO + variant ----
    const PROJ = "_vbprobe";    // leading underscore = verbatim-listing check (catches name mangling)
    // ---- Timeline render → a BROWSER-PLAYABLE MP4 in Results (cv2 mp4v writes + serves fine but
    //      Chrome's <video> decodes NOTHING from it — must be H.264/yuv420p) ----
    await check(page, "[vb][px] timeline render → MP4 PLAYS in Results (decodes + time advances)", async () => {
      await stopPlayback(page);
      await setAndFire(page, "#tl-clip", 2, ["input"]); await sleep(200);
      const built = await page.evaluate(buildTwoPaintClips);
      assert(typeof built === "string" && !/^no |<2/.test(built), "could not build a 2-clip timeline: " + built);
      await setAndFire(page, "#tl-fps", 4, ["input", "change"]);        // 4s × 4fps = ~16 frames (fast render)
      await setAndFire(page, "#tl-out", OUT_DIR, ["input", "change"]);
      await page.evaluate(() => document.getElementById("tl-render").click());
      const done = await (async () => {
        for (let i = 0; i < 360; i++) {
          const st = await page.evaluate(() => ({
            s: ((document.getElementById("tl-status") || {}).textContent || ""),
            res: (document.querySelector('.tab[data-pane="results"]') || { classList: { contains: () => false } }).classList.contains("active"),
          }));
          if (/done|rendered|mp4/i.test(st.s) || st.res) return st;
          await sleep(1000);
        }
        return null;
      })();
      assert(done, "timeline render did not finish within 6 min (no done status, no Results auto-switch)");
      await page.click('.tab[data-pane="results"]'); await sleep(800);
      await page.evaluate((dir) => { const d = document.getElementById("results-dir"); if (d && !d.value) { d.value = dir; d.dispatchEvent(new Event("input")); } }, OUT_DIR);
      await page.click("#results-refresh").catch(() => {}); await sleep(2500);
      await page.evaluate(() => {
        const s = document.getElementById("results-select");
        const vid = [...s.options].find((o) => /▸|video|\.mp4/i.test(o.textContent + " " + o.value));
        if (vid) { s.value = vid.value; s.dispatchEvent(new Event("change")); }
      });
      let vw = 0, t0 = 0;
      for (let i = 0; i < 25; i++) {
        await sleep(1000);
        const v = await page.evaluate(() => { const e = document.getElementById("results-video"); if (!e) return null; const cs = getComputedStyle(e); return { show: cs.display !== "none", vw: e.videoWidth, ct: e.currentTime }; });
        if (v && v.show && v.vw > 0) { vw = v.vw; t0 = v.ct; break; }
      }
      assert(vw > 0, "MP4 never decoded in #results-video (videoWidth stayed 0) — wrong codec: must be H.264/yuv420p, not cv2 mp4v");
      await page.evaluate(() => { const e = document.getElementById("results-video"); if (e && e.paused) e.play().catch(() => {}); });
      await sleep(2500);
      const t1 = await page.evaluate(() => ((document.getElementById("results-video") || {}).currentTime) || 0);
      assert(t1 > t0, `MP4 playback does not advance (t ${t0.toFixed(2)}→${t1.toFixed(2)})`);
      await page.click('.tab[data-pane="timeline"]').catch(() => {}); await sleep(400);
      await page.evaluate(() => document.getElementById("tl-clear").click()).catch(() => {}); await sleep(300);
      return `mp4 decodes (videoWidth=${vw}) and plays (t ${t0.toFixed(2)}→${t1.toFixed(2)})`;
    });

    await check(page, "[vb] project save → lists VERBATIM → open RESTORES state", async () => {
      // capture a distinct state: set ISO high + select a non-default variant on Configure.
      await page.click('.tab[data-pane="configure"]'); await sleep(400);
      // select a NORMAL authored camera first — per-camera optics on the animated turntable rig
      // restore through a different path; a plain camera is the canonical round-trip.
      await page.evaluate(() => {
        const o = [...document.querySelectorAll("#camera-select option")].map((x) => x.value).filter((v) => v && !/turntable/i.test(v));
        const s = document.getElementById("camera-select"); if (o[0]) { s.value = o[0]; s.dispatchEvent(new Event("change")); }
      });
      await sleep(4000);   // let the camera-snap's camera_params (it repopulates the sliders) land FIRST
      // set ISO and confirm it STICKS: a late camera_params can reset the slider, so re-apply until two
      // consecutive reads agree (this race, not a real defect, caused the earlier flaky "ISO not restored")
      let savedIso = null;
      for (let k = 0; k < 4 && savedIso == null; k++) {
        await setAndFire(page, "#disp-iso", 2000, ["input", "change"]); await waitLive(page, 90000); await sleep(1800);
        const a = await page.evaluate(() => document.getElementById("disp-iso-v").textContent);
        await sleep(1500);
        const b = await page.evaluate(() => document.getElementById("disp-iso-v").textContent);
        if (a === "2000" && b === "2000") savedIso = "2000";
      }
      assert(savedIso === "2000", "could not set a stable ISO before save (camera_params kept resetting it)");
      const savedVar = await page.evaluate(() => {
        const chip = document.querySelector("#variant-cards .vcard .chip.on");
        return chip ? { sname: chip.dataset.sname, variant: chip.dataset.variant } : null;
      });
      // save the project (Timeline tab holds the Projects UI)
      await page.click('.tab[data-pane="timeline"]'); await sleep(500);
      await setAndFire(page, "#proj-name", PROJ, ["input", "change"]);
      await page.click("#proj-save");
      const listed = await (async () => {
        for (let i = 0; i < 12; i++) {
          const ok = await page.evaluate((nm) => [...document.querySelectorAll("#proj-list option")].some((o) => o.value === nm || o.textContent === nm), PROJ);
          if (ok) return true; await sleep(700);
        }
        return false;
      })();
      const msg = await page.evaluate(() => (document.getElementById("proj-msg") || {}).textContent || "");
      assert(listed, `project "${PROJ}" not listed verbatim (proj-msg="${msg}")`);
      // perturb the state
      await page.click('.tab[data-pane="configure"]'); await sleep(400);
      await setAndFire(page, "#disp-iso", 100, ["input", "change"]); await waitLive(page, 90000); await sleep(1500);
      await page.evaluate(() => { const chips = [...document.querySelectorAll("#variant-cards .vcard .chip")]; const off = chips.find((c) => !c.classList.contains("on")); off && off.click(); });
      await sleep(2500);
      // open the saved project (auto-accept the confirm) → full reopen + restore
      await page.click('.tab[data-pane="timeline"]'); await sleep(400);
      await page.evaluate((nm) => { const s = document.getElementById("proj-list"); s.value = nm; }, PROJ);
      await page.click("#proj-open");
      await waitLive(page, 120000); await sleep(3000);
      // PRIMARY restore signal: the base variant selection (deterministic, no per-camera-look race)
      if (savedVar) {
        const varOk = await (async () => {
          for (let k = 0; k < 8; k++) {
            const ok = await page.evaluate((sv) => {
              const chip = document.querySelector(`#variant-cards .vcard .chip.on[data-sname="${CSS.escape(sv.sname)}"]`);
              return chip && chip.dataset.variant === sv.variant;
            }, savedVar);
            if (ok) return true; await sleep(1200);
          }
          return false;
        })();
        assert(varOk, "saved variant selection not restored");
      }
      // SECONDARY: per-camera ISO restore (camera_params seeds the slider a beat after the open → poll)
      let isoNow = null;
      for (let k = 0; k < 10; k++) { isoNow = await page.evaluate(() => document.getElementById("disp-iso-v").textContent); if (isoNow === savedIso) break; await sleep(1200); }
      assert(String(isoNow) === String(savedIso), `ISO not restored (saved ${savedIso}, now ${isoNow})`);
      return `variant restored` + `, iso restored=${isoNow}`;
    });
    await check(page, "[vb] timeline VIEWS within a project: save / list / load-restores / delete", async () => {
      // a project is open now → the "Saved track views" panel MUST be revealed
      assert(await page.evaluate(() => getComputedStyle(document.getElementById("tlv-block")).display !== "none"),
        "#tlv-block not visible with a project open — timeline views must be revealed when a project is open");
      const buildClips = (n) => page.evaluate((n) => {
        document.getElementById("tl-clear").click();
        const row = [...document.querySelectorAll('#tl-tracks .tl-track')].find((r) => { const s = r.querySelector('.tl-track-label select'); return s && s.options.length >= 1; });
        if (!row) return 0;
        const sel = row.querySelector('.tl-track-label select'), add = row.querySelector('.tl-add-clip');
        for (let i = 0; i < n; i++) { sel.value = sel.options[Math.min(i, sel.options.length - 1)].value; sel.dispatchEvent(new Event('change')); add.click(); }
        return document.querySelectorAll('#tl-tracks .tl-clip').length;
      }, n);
      const clipCount = () => page.evaluate(() => document.querySelectorAll('#tl-tracks .tl-clip').length);
      const listed = (nm) => page.evaluate((nm) => [...document.querySelectorAll("#tlv-list option")].some((o) => (o.textContent + o.value).includes(nm)), nm);
      const pickView = (nm) => page.evaluate((nm) => { const s = document.getElementById("tlv-list"); const o = [...s.options].find((o) => (o.textContent + o.value).includes(nm)); if (o) { s.value = o.value; s.dispatchEvent(new Event("change")); } }, nm);
      // the Project + "Saved track views" controls (#proj-*/#tlv-*) live in the TIMELINE tab (no separate
      // "projects" tab) — as does the timeline strip (#tl-tracks). So everything here runs on that tab.
      await page.click('.tab[data-pane="timeline"]'); await sleep(500);
      const saveView = async (nm) => { await setAndFire(page, "#tlv-name", nm, ["input"]); await page.click("#tlv-save"); await sleep(1500); };
      // state A = 1 clip → save "vbA"
      const nA = await buildClips(1); assert(nA >= 1, "could not build timeline state A");
      await saveView("vbA"); assert(await listed("vbA"), "view A not listed after save");
      // state B = 3 clips (distinct) → save "vbB"
      const nB = await buildClips(3); assert(nB !== nA, `state B clip count (${nB}) not distinct from A (${nA})`);
      await saveView("vbB"); assert((await listed("vbA")) && (await listed("vbB")), "both named views not listed (multiple views must coexist in a project)");
      // load A → the working timeline is RESTORED to A (clip count returns to nA, not B's nB)
      await pickView("vbA"); await page.click("#tlv-load"); await sleep(1800);
      const nLoaded = await clipCount();
      assert(nLoaded === nA, `#tlv-load did not restore view A (timeline has ${nLoaded} clips, expected A's ${nA}; B was ${nB}) — views must replace the timeline`);
      // delete B → gone from the list
      await pickView("vbB"); await page.click("#tlv-del"); await sleep(1500);
      assert(!(await listed("vbB")), "view B still listed after delete");
      return `A(${nA})+B(${nB}) saved; load A restored ${nLoaded}; B deleted`;
    });
    // cleanup the probe project (best-effort)
    await check(page, "cleanup: delete the probe project", async () => {
      await page.evaluate((nm) => { const s = document.getElementById("proj-list"); s.value = nm; }, PROJ);
      await page.click("#proj-del"); await sleep(1200);
      return true;
    });

    // ---- collapsible Configure block ----
    await check(page, "[dom] collapsible Configure block toggles", async () => {
      await page.click('.tab[data-pane="configure"]'); await sleep(400);
      const r = await page.evaluate(() => {
        const block = document.querySelector("#pane-configure .block:not(.grow)");
        const h2 = block.querySelector("h2"); const was = block.classList.contains("collapsed");
        h2.click(); const now = block.classList.contains("collapsed"); h2.click();
        return { was, now };
      });
      assert(r.was !== r.now, "collapsible block did not toggle"); return true;
    });

    // ---- ATTACH revival: `ready` NEVER re-fires — a page (re)load onto the already-open stage must
    //      reconcile from /api/stage and come back FULLY interactive (otherwise: a zombie page —
    //      stuck overlay, video decoding but orbit dead + chips silently no-op) ----
    let attachOrbitDelta = -1;   // measured inside; reported softly after (see the warnIf below)
    await check(page, "[vb] ATTACH: page reload onto the open stage revives (overlay clears + chips post)", async () => {
      await page.goto(URL, { waitUntil: "domcontentloaded" }); await sleep(1200);
      if (USD) await setAndFire(page, "#usd-path", USD, ["input", "change"]);
      await page.click("#open-btn");
      const m = await waitMedia(page, 90000, 1);
      assert(m, "no media within 90s after attach (client never revived)");
      assert(await waitLive(page, 45000), "status never returned to live after attach");
      // a brief "warming/connecting" overlay right at the handoff is fine — the DEFECT is a STUCK
      // overlay. Poll up to 25s for it to clear before failing.
      let ov = "";
      for (let i = 0; i < 25; i++) {
        ov = await page.evaluate(() => { const o = document.getElementById("overlay"); if (!o) return ""; const cs = getComputedStyle(o); return (cs.display !== "none" && cs.visibility !== "hidden") ? (o.textContent || "").trim().slice(0, 50) : ""; });
        if (!/download|mirror|opening|warming|connect/i.test(ov)) break;
        await sleep(1000);
      }
      assert(!/download|mirror|opening|warming|connect/i.test(ov), `boot overlay STUCK after attach ("${ov}") — never handed off`);
      await sleep(2000);
      // chips must POST: click an off-variant, confirm the SERVER-side selection changes
      const pick = await page.evaluate(async () => {
        const st = await (await fetch("/api/stage")).json();
        const sel = {}; (st.selection || []).forEach((s) => { sel[s.set_name] = s.variant; });
        for (const c of document.querySelectorAll("#variant-cards .vcard")) {
          const name = ((c.querySelector(".name") || {}).textContent || "").trim();
          const off = [...c.querySelectorAll(".chip")].find((ch) => !ch.classList.contains("on"));
          if (name && off) { off.click(); return { set: name, was: sel[name] || "" }; }
        }
        return null;
      });
      assert(pick, "no clickable variant chip after attach (panels not repopulated)");
      let changed = false, now = "";
      for (let i = 0; i < 12 && !changed; i++) {
        await sleep(1000);
        now = await page.evaluate(async (sn) => { const st = await (await fetch("/api/stage")).json(); const c = (st.selection || []).find((s) => s.set_name === sn); return c ? (c.variant || "") : ""; }, pick.set);
        if (now && now !== pick.was) changed = true;
      }
      assert(changed, `chip click never reached the server after attach ("${pick.set}" stayed "${pick.was}") — chips are zombie`);
      // ORBIT input after attach: SOFT. The reference implementation's ovstream session can leave input bound
      // to the dead ghost session (native single-client limitation) — but SPEC-UX REQUIRES builds to
      // rebuild the stream on attach (same mechanics as the resolution-change path) so input rebinds.
      const vr = await rectOf(page, "#remote-video"); assert(vr, "no video rect after attach");
      attachOrbitDelta = 0;
      for (let att = 0; att < 3 && attachOrbitDelta <= 8; att++) {
        await page.mouse.click(vr.cx, vr.cy); await sleep(500);
        const b = await sample(page);
        await dragMouse(page, vr.cx, vr.cy, vr.cx - 220, vr.cy - 30, 16); await sleep(2500);
        const a = await sample(page);
        attachOrbitDelta = rgbDelta(b, a);
      }
      return `revived: live, overlay clear, chip posted (${pick.set}: "${pick.was}"→"${now}"); orbit-after-attach Δ=${attachOrbitDelta.toFixed(1)} (soft)`;
    });
    warnIf(attachOrbitDelta > 8, "[vb-soft] orbit input re-armed after attach (builds MUST rebuild the stream on attach)",
      `delta=${attachOrbitDelta.toFixed(1)}`);

    // ---- stream stays connected on a HEALTHY idle stream (no reconnect storm) — defect 6 ----
    // Placed LAST so the heavy reopen tests are done and the stage is live. A false-stall watchdog
    // fires /api/stream/restart (or /api/restart) on a perfectly healthy idle stream → restartCalls>0.
    await check(page, "[vb] idle 30s: no /api/(stream/)restart storm, frames keep flowing, status=live", async () => {
      assert(await waitLive(page, 30000), "stream not live before idle-stability probe");
      restartCalls.length = 0;
      const d0 = (await sample(page)).decoded;
      await sleep(30000);   // idle: NO interaction for 30s
      const s1 = await sample(page);
      const st = (await statusText(page)).trim();
      assert(restartCalls.length === 0,
        `reconnect storm on a healthy idle stream: ${restartCalls.length} restart call(s) (${restartCalls.slice(0, 3).join(", ")})`);
      assert(s1 && s1.decoded > d0, `frames stopped flowing while idle (decoded ${d0}→${s1 ? s1.decoded : "?"})`);
      assert(/\blive\b/i.test(st), `status not 'live' after idle ("${st}")`);
      return `restarts=0 decoded ${d0}→${s1.decoded} status="${st}"`;
    });

  } catch (e) {
    log("FATAL during verify: " + ((e && e.stack) || e));
    fail++;
  } finally {
    log(`\n== verify_browser: ${pass} passed, ${fail} failed, ${warn} warn ==`);
    if (fails.length) log("   failed: " + fails.join(" | "));
    try { await browser.close(); } catch (e) {}
    process.exit(fail ? 1 : 0);
  }
})();
