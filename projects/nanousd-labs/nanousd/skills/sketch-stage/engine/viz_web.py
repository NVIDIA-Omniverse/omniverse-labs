# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2D top-down + 3D live placement viz.

Tiny stdlib HTTP server. Endpoints:
  /          — index.html with split panes (2D top-down + 3D Three.js) + JS
  /events    — Server-Sent Events stream tailing events.jsonl

Open http://localhost:<port>/ in a browser. Both panes redraw as each
event arrives. The 3D pane fetches /template from the session HTTP
(default http://<host>:8766/template) for shell + zone bounds.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# OS-temp-dir-based default so the path resolves on Linux / macOS / Windows.
_DEFAULT_FOLLOW_ROOT = str(Path(tempfile.gettempdir()) / "sketch_live")


INDEX_HTML = r"""
<!doctype html>
<html><head><meta charset='utf-8'><title>sketch placement live (2D + 3D)</title>
<style>
html, body { margin:0; padding:0; height:100%; font-family:monospace; background:#0b1220; color:#e5e7eb; overflow:hidden; }
#legend-bar { display:flex; align-items:center; gap:8px; padding:4px 8px; background:#111827; font-size:11px; }
#legend-bar button { background:#1f2937; color:#e5e7eb; border:1px solid #374151; padding:2px 8px; cursor:pointer; font-family:monospace; font-size:11px; border-radius:3px; }
#legend-bar button:hover { background:#374151; }
#legend-count { color:#9ca3af; }
#legend { display:none; flex-wrap:wrap; gap:6px; padding:5px 8px; background:#111827; max-height:25vh; overflow-y:auto; border-top:1px solid #1f2937; }
#legend.open { display:flex; }
.lg { display:flex; align-items:center; gap:3px; font-size:11px; white-space:nowrap; }
.sw { display:inline-block; width:10px; height:10px; flex-shrink:0; }
#status { padding:5px 8px; background:#111827; font-size:11px; }
#wrap { display:flex; height: calc(100vh - 56px); }
.pane { position:relative; flex:1; border-right:1px solid #1f2937; }
.pane:last-child { border-right:none; }
.pane h2 { position:absolute; top:5px; left:8px; margin:0; font-size:12px;
           background:rgba(17,24,39,0.7); color:#cbd5e1; padding:3px 7px; z-index:10; pointer-events:none; }
canvas { display:block; width:100%; height:100%; cursor:grab; }
canvas:active { cursor:grabbing; }
</style>
<script src="/static/three.min.js"></script>
<script src="/static/OrbitControls.js"></script>
</head>
<body>
<div id='legend-bar'>
  <button id='legend-toggle' onclick="document.getElementById('legend').classList.toggle('open'); this.textContent=document.getElementById('legend').classList.contains('open')?'Hide legend':'Show legend';">Show legend</button>
  <span id='legend-count'>0 archetypes</span>
</div>
<div id='legend'></div>
<div id='wrap'>
  <div class='pane'><h2>top-down 2D</h2><canvas id='c2d'></canvas></div>
  <div class='pane'><h2>3D (Three.js, drag=rotate, wheel=zoom)</h2><canvas id='c3d'></canvas></div>
</div>
<div id='status'>connecting…</div>
<script>
// Structural archetypes get fixed colors; everything else gets a deterministic
// hue from the archetype name so the legend stays consistent within a session.
const COLOR_FIXED = { "<rootcell>":"#10b981", container:"#8b5cf6" };
const COLOR = {};
const _legendSeen = new Set();
function _archeColor(arche){
  if (COLOR_FIXED[arche]) return COLOR_FIXED[arche];
  if (COLOR[arche]) return COLOR[arche];
  let h = 0; for (let i=0;i<arche.length;i++) h = (h*131 + arche.charCodeAt(i)) >>> 0;
  const hue = h % 360, sat = 55 + (h>>>8) % 30, lt = 45 + (h>>>16) % 15;
  const c = `hsl(${hue},${sat}%,${lt}%)`;
  COLOR[arche] = c;
  return c;
}
const lg = document.getElementById('legend');
const lgCount = document.getElementById('legend-count');
function _legendAdd(arche){
  if (_legendSeen.has(arche)) return;
  _legendSeen.add(arche);
  const sw = document.createElement('span'); sw.className = 'sw';
  sw.style.background = _archeColor(arche);
  const e = document.createElement('span'); e.className = 'lg';
  e.appendChild(sw); e.appendChild(document.createTextNode(arche));
  lg.appendChild(e);
  if (lgCount) lgCount.textContent = `${_legendSeen.size} archetypes`;
}
for (const k of Object.keys(COLOR_FIXED)) _legendAdd(k);

// ============================================================================
//  shared state
// ============================================================================
const items = new Map();       // id -> {arche, x, y, w, d, h, yaw}
const recent = [];             // {until, x, y, w, d}
const counts = {placed:0, rejected:0, removed:0, events:0};
let template = null;
let SESSION_PORT = __SESSION_PORT__;

async function fetchTemplate(){
  try {
    const r = await fetch(`http://${location.hostname}:${SESSION_PORT}/template`);
    if (!r.ok) return null;
    return await r.json();
  } catch(_) { return null; }
}
async function fetchStageGraph(){
  try {
    const r = await fetch(`http://${location.hostname}:${SESSION_PORT}/tool/query_stage_graph`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}',
    });
    if (!r.ok) return [];
    const data = await r.json();
    return Array.isArray(data) ? data : [];
  } catch(_) { return []; }
}

// ============================================================================
//  2D top-down
// ============================================================================
const c2d = document.getElementById('c2d'); const ctx = c2d.getContext('2d');
function fit2D(){ c2d.width = c2d.parentElement.clientWidth; c2d.height = c2d.parentElement.clientHeight; }
fit2D(); window.addEventListener('resize', () => { fit2D(); fit3D(); render2D(); render3D(); });
let view2 = {scale: 8, tx: 100, ty: 100, fitDone: false};
function w2c(x, y){ return [view2.tx + x*view2.scale, c2d.height - (view2.ty + y*view2.scale)]; }
function fitAll2(){
  if (items.size < 8) return;
  let xs = [], ys = [];
  for (const b of items.values()){ xs.push(b.x - b.w/2, b.x + b.w/2); ys.push(b.y - b.d/2, b.y + b.d/2); }
  let x0=Math.min(...xs), x1=Math.max(...xs), y0=Math.min(...ys), y1=Math.max(...ys);
  const margin=30;
  view2.scale = Math.min((c2d.width-2*margin)/(x1-x0+0.5),(c2d.height-2*margin)/(y1-y0+0.5))*0.95;
  view2.tx = c2d.width/2 - ((x0+x1)/2)*view2.scale;
  view2.ty = c2d.height/2 - ((y0+y1)/2)*view2.scale;
  view2.fitDone = true;
}
function render2D(){
  ctx.fillStyle='#0b1220'; ctx.fillRect(0,0,c2d.width,c2d.height);
  if (template){
    for (const sh of template.shells || []){
      const ox = sh.originWorldM ? sh.originWorldM[0] : 0;
      const oy = sh.originWorldM ? sh.originWorldM[1] : 0;
      const w = sh.boundsM.widthM, d = sh.boundsM.depthM;
      const [a,b1] = w2c(ox, oy); const [c,e] = w2c(ox+w, oy+d);
      const xx = Math.min(a,c), yy = Math.min(b1,e); const ww = Math.abs(c-a), hh = Math.abs(e-b1);
      ctx.fillStyle = 'rgba(255,255,255,0.04)'; ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 2;
      ctx.fillRect(xx,yy,ww,hh); ctx.strokeRect(xx,yy,ww,hh);
    }
    for (const z of template.zones || []){
      const ox = z.originWorldM ? z.originWorldM[0] : 0; const oy = z.originWorldM ? z.originWorldM[1] : 0;
      const w = z.boundsM.widthM, d = z.boundsM.depthM;
      const [a,b1] = w2c(ox, oy); const [c,e] = w2c(ox+w, oy+d);
      const xx = Math.min(a,c), yy = Math.min(b1,e); const ww = Math.abs(c-a), hh = Math.abs(e-b1);
      ctx.strokeStyle = 'rgba(56,189,248,0.55)'; ctx.setLineDash([8,5]); ctx.lineWidth = 1.5;
      ctx.strokeRect(xx,yy,ww,hh); ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(56,189,248,0.85)'; ctx.font = '12px monospace';
      ctx.fillText(z.id, xx+5, yy+15);
    }
  }
  // Draw rootcells first (as outlined frames), then archetype placements on top.
  const order = [...items.values()].sort((a,b) => (a.arche === '<rootcell>' ? -1 : 1) - (b.arche === '<rootcell>' ? -1 : 1));
  for (const b of order){
    const [a,b1] = w2c(b.x - b.w/2, b.y - b.d/2); const [cc,d2] = w2c(b.x + b.w/2, b.y + b.d/2);
    const baseColor = _archeColor(b.arche);
    _legendAdd(b.arche);
    const xx = Math.min(a,cc), yy = Math.min(b1,d2); const ww = Math.abs(cc-a), hh = Math.abs(d2-b1);
    // 2D wireframe rendering for container-style archetypes so contents
    // stay visible. The set is the structural rootcell + any archetype
    // tagged isContainer=true by the pack (see scaffolds below).
    if (CONTAINER_ARCHETYPES.has(b.arche)){
      ctx.strokeStyle = baseColor;
      ctx.lineWidth = b.arche === '<rootcell>' ? 2 : 1;
      if (b.arche === '<rootcell>') ctx.setLineDash([6, 4]);
      ctx.strokeRect(xx, yy, ww, hh);
      ctx.setLineDash([]);
      if (ww > 38 && hh > 14){
        ctx.fillStyle = baseColor; ctx.font = '10px monospace';
        ctx.fillText(b.arche, xx + 3, yy + 11);
      }
      continue;
    }
    ctx.fillStyle = b.anchor ? baseColor + 'AA' : baseColor;
    ctx.strokeStyle = '#0b1220'; ctx.lineWidth = 0.5;
    ctx.fillRect(xx,yy,ww,hh); ctx.strokeRect(xx,yy,ww,hh);
    if (ww > 38 && hh > 14){
      ctx.fillStyle = '#0b1220'; ctx.font = '10px monospace';
      ctx.fillText(b.arche, xx + 3, yy + 11);
    }
  }
  const now = performance.now()/1000;
  for (const r of recent){
    if (r.until < now) continue;
    const [a,b1] = w2c(r.x - r.w/2, r.y - r.d/2); const [cc,d2] = w2c(r.x + r.w/2, r.y + r.d/2);
    ctx.fillStyle = 'rgba(220,38,38,0.55)'; ctx.strokeStyle = '#fee2e2'; ctx.lineWidth = 1;
    const xx = Math.min(a,cc), yy = Math.min(b1,d2); const ww = Math.abs(cc-a), hh = Math.abs(d2-b1);
    ctx.fillRect(xx,yy,ww,hh); ctx.strokeRect(xx,yy,ww,hh);
  }
}

// 2D zoom + pan
let dragging = false, pStart = null;
c2d.addEventListener('wheel', (e) => {
  e.preventDefault();
  const f = e.deltaY < 0 ? 1.18 : 1/1.18;
  const r = c2d.getBoundingClientRect();
  const cx = e.clientX - r.left, cy = e.clientY - r.top;
  const wx = (cx - view2.tx)/view2.scale; const wy = ((c2d.height - cy) - view2.ty)/view2.scale;
  view2.scale *= f; view2.tx = cx - wx*view2.scale; view2.ty = (c2d.height - cy) - wy*view2.scale;
}, {passive:false});
c2d.addEventListener('mousedown', (e) => { dragging = true; pStart = {x: e.clientX, y: e.clientY, tx: view2.tx, ty: view2.ty}; });
c2d.addEventListener('mousemove', (e) => { if(!dragging) return;
  view2.tx = pStart.tx + (e.clientX - pStart.x); view2.ty = pStart.ty - (e.clientY - pStart.y);
});
c2d.addEventListener('mouseup', () => { dragging = false; });

// ============================================================================
//  3D Three.js
// ============================================================================
const c3d = document.getElementById('c3d');
function fit3D(){ c3d.width = c3d.parentElement.clientWidth; c3d.height = c3d.parentElement.clientHeight; }
fit3D();
const HAS_THREE = typeof THREE !== 'undefined'
  && typeof THREE.Scene === 'function'
  && typeof THREE.WebGLRenderer === 'function'
  && typeof THREE.OrbitControls === 'function';
let scene3 = null, camera3 = null, renderer3 = null, controls = null, template3 = null;
const placement3 = new Map();   // id -> Mesh/LineSegments when 3D is available

function draw3DUnavailable(){
  const g = c3d.getContext('2d');
  g.fillStyle = '#0b1220'; g.fillRect(0, 0, c3d.width, c3d.height);
  g.fillStyle = '#cbd5e1'; g.font = '14px monospace';
  g.fillText('3D unavailable: missing Three.js static assets.', 18, 36);
  g.fillStyle = '#94a3b8'; g.font = '12px monospace';
  g.fillText('2D live view still updates from placement events.', 18, 58);
}

if (HAS_THREE){
  scene3 = new THREE.Scene(); scene3.background = new THREE.Color('#0b1220');
  camera3 = new THREE.PerspectiveCamera(50, c3d.width / c3d.height, 0.1, 5000);
  camera3.position.set(80, 80, 60); camera3.up.set(0, 0, 1); camera3.lookAt(35, 50, 0);
  renderer3 = new THREE.WebGLRenderer({canvas: c3d, antialias: true});
  renderer3.setSize(c3d.width, c3d.height, false);
  controls = new THREE.OrbitControls(camera3, renderer3.domElement);
  controls.target.set(35, 50, 1); controls.update();
  window.addEventListener('resize', () => {
    camera3.aspect = c3d.width / c3d.height; camera3.updateProjectionMatrix();
    renderer3.setSize(c3d.width, c3d.height, false);
  });
  // Lambert needs lights; one ambient + one directional is plenty.
  const ambient = new THREE.AmbientLight(0xffffff, 0.7); scene3.add(ambient);
  const dir = new THREE.DirectionalLight(0xffffff, 0.6); dir.position.set(60, 60, 100); scene3.add(dir);
  // Z-up grid (lying on XY plane)
  const grid = new THREE.GridHelper(200, 40, 0x334155, 0x1f2937);
  grid.rotation.x = Math.PI/2; scene3.add(grid);
  // reference axes
  const axes = new THREE.AxesHelper(5); scene3.add(axes);
  template3 = new THREE.Group(); scene3.add(template3);
} else {
  window.addEventListener('resize', draw3DUnavailable);
  draw3DUnavailable();
}

function rebuildTemplate3(){
  if (!HAS_THREE || !template3) return;
  while (template3.children.length) template3.remove(template3.children[0]);
  if (!template) return;
  for (const sh of template.shells || []){
    const ox = sh.originWorldM ? sh.originWorldM[0] : 0;
    const oy = sh.originWorldM ? sh.originWorldM[1] : 0;
    const w = sh.boundsM.widthM, d = sh.boundsM.depthM, h = sh.boundsM.heightM || 6;
    // floor + 4 thin walls + no roof
    const wallMat = new THREE.MeshStandardMaterial({color: 0x60a5fa, transparent:true, opacity:0.18});
    const floorMat = new THREE.MeshStandardMaterial({color: 0x1e293b});
    const floor = new THREE.Mesh(new THREE.BoxGeometry(w, d, 0.05), floorMat);
    floor.position.set(ox + w/2, oy + d/2, -0.025); template3.add(floor);
    const ws = 0.08;
    const sWall = (sx, sy, sz, px, py, pz) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), wallMat);
      m.position.set(px, py, pz); template3.add(m);
    };
    sWall(w, ws, h, ox + w/2, oy + ws/2, h/2);              // south
    sWall(w, ws, h, ox + w/2, oy + d - ws/2, h/2);          // north
    sWall(ws, d, h, ox + ws/2, oy + d/2, h/2);              // west
    sWall(ws, d, h, ox + w - ws/2, oy + d/2, h/2);          // east
    // wall outlines
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(w, d, h)),
      new THREE.LineBasicMaterial({color: 0x60a5fa, transparent:true, opacity:0.6}));
    edges.position.set(ox + w/2, oy + d/2, h/2);
    template3.add(edges);
  }
  for (const z of template.zones || []){
    const ox = z.originWorldM ? z.originWorldM[0] : 0;
    const oy = z.originWorldM ? z.originWorldM[1] : 0;
    const w = z.boundsM.widthM, d = z.boundsM.depthM;
    const geom = new THREE.PlaneGeometry(w, d);
    const mat = new THREE.MeshStandardMaterial({color: 0x38bdf8, transparent:true, opacity:0.07, side: THREE.DoubleSide});
    const plane = new THREE.Mesh(geom, mat);
    plane.position.set(ox + w/2, oy + d/2, 0.01);
    template3.add(plane);
    const eg = new THREE.LineSegments(
      new THREE.EdgesGeometry(geom),
      new THREE.LineDashedMaterial({color: 0x38bdf8, dashSize: 1, gapSize: 0.6}));
    eg.position.set(ox + w/2, oy + d/2, 0.02);
    eg.computeLineDistances();
    template3.add(eg);
  }
  // re-aim camera at the template centroid
  const xs=[], ys=[];
  for (const z of template.zones || []){ const ox=z.originWorldM[0], oy=z.originWorldM[1]; xs.push(ox, ox+z.boundsM.widthM); ys.push(oy, oy+z.boundsM.depthM); }
  for (const sh of template.shells || []){ const ox=sh.originWorldM[0], oy=sh.originWorldM[1]; xs.push(ox, ox+sh.boundsM.widthM); ys.push(oy, oy+sh.boundsM.depthM); }
  if (xs.length){
    const cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2;
    controls.target.set(cx, cy, 1);
    const sz = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
    camera3.position.set(cx + sz*0.6, cy - sz*0.6, sz*0.5);
    controls.update();
  }
}

// Cache one geometry+material per (archetype, isAnchor) combo so we re-use
// across hundreds of placements (no per-item GPU allocation).
const _matCache = new Map();
function _getMat(arche, isAnchor){
  const key = arche + (isAnchor ? '_a' : '_n');
  let m = _matCache.get(key);
  if (!m){
    const c = new THREE.Color(_archeColor(arche));
    _legendAdd(arche);
    m = new THREE.MeshLambertMaterial({
      color: c,
      transparent: isAnchor,
      opacity: isAnchor ? 0.5 : 1.0,
    });
    _matCache.set(key, m);
  }
  return m;
}
// Archetypes that act as containers / large structures — render as wireframes
// so the smaller placements inside or beside them remain visible. The set is
// the structural rootcell plus any archetype the pack tagged isContainer=true
// (the server sends a `containerArchetypes` payload at session open).
const CONTAINER_ARCHETYPES = new Set(['<rootcell>']);
function add3D(id, arche, x, y, w, d, h, yaw, isAnchor){
  if (!HAS_THREE || !scene3) return;
  if (CONTAINER_ARCHETYPES.has(arche)){
    const colorHex = _archeColor(arche);
    _legendAdd(arche);
    const geom = new THREE.BoxGeometry(w, d, h);
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geom),
      new THREE.LineBasicMaterial({color: new THREE.Color(colorHex)}));
    edges.position.set(x, y, h / 2);
    edges.rotation.z = yaw * Math.PI / 180;
    scene3.add(edges);
    placement3.set(id, edges);
    return;
  }
  // Standard placements: shared per-archetype material, single Box mesh.
  const geom = new THREE.BoxGeometry(w, d, h);
  const mesh = new THREE.Mesh(geom, _getMat(arche, isAnchor));
  mesh.position.set(x, y, h / 2);
  mesh.rotation.z = yaw * Math.PI / 180;
  scene3.add(mesh);
  placement3.set(id, mesh);
}
function remove3D(id){
  if (!HAS_THREE || !scene3) return;
  const m = placement3.get(id); if(!m) return;
  scene3.remove(m);
  if (m.geometry) m.geometry.dispose();
  // material is shared in _matCache; do not dispose
  placement3.delete(id);
}
function flash3D(x, y, w, d){
  if (!HAS_THREE || !scene3) return;
  const mat = new THREE.MeshStandardMaterial({color: 0xdc2626, transparent:true, opacity:0.55});
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, d, 0.05), mat);
  m.position.set(x, y, 0.05); scene3.add(m);
  setTimeout(() => { scene3.remove(m); m.geometry.dispose(); m.material.dispose(); }, 1500);
}

function animate(){
  requestAnimationFrame(animate);
  if (!HAS_THREE) return;
  controls.update(); renderer3.render(scene3, camera3);
}
animate();

// ============================================================================
//  event tail + dispatch
// ============================================================================
function clearAll(){
  for (const m of placement3.values()){
    scene3.remove(m); if(m.geometry) m.geometry.dispose();
  }
  placement3.clear();
  items.clear();
  recent.length = 0;
  counts.placed = 0; counts.rejected = 0; counts.removed = 0; counts.events = 0;
  view2.fitDone = false;
}
function fit3DCamera(){
  if (!HAS_THREE) return;
  if (items.size === 0) return;
  let xs = [], ys = [], zs = [];
  for (const b of items.values()){
    xs.push(b.x - b.w/2, b.x + b.w/2);
    ys.push(b.y - b.d/2, b.y + b.d/2);
    zs.push(0, b.h);
  }
  const cx = (Math.min(...xs)+Math.max(...xs))/2;
  const cy = (Math.min(...ys)+Math.max(...ys))/2;
  const sz = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys));
  controls.target.set(cx, cy, 1);
  camera3.position.set(cx + sz*0.7, cy - sz*0.7, sz*0.6);
  controls.update();
}
function applyEv(m){
  counts.events++;
  if (m.type === 'session_open'){
    clearAll();
    if (m.sessionPort && m.sessionPort !== SESSION_PORT){
      SESSION_PORT = m.sessionPort;
    }
    // Re-seed the viz from the new session's stage graph regardless of
    // whether the port changed. A resumed session loads anchor placements
    // silently into the index (no per-placement 'placed' events are
    // emitted at load time), so the SSE stream alone would never show
    // them — they only exist via the HTTP query.
    (async () => {
      template = await fetchTemplate();
      rebuildTemplate3();
      const seed = await fetchStageGraph();
      for (const p of seed){
        const [w,d,h] = p.slotM; const [x,y] = p.posM;
        items.set(p.id, {arche: p.archetype, x, y, w, d, h, yaw: p.yawDeg || 0, anchor: true});
        add3D(p.id, p.archetype, x, y, w, d, h, p.yawDeg || 0, true);
      }
      view2.fitDone = false;
      if (items.size) fitAll2();
    })();
    return;
  }
  if (m.type === 'placed'){
    const [w,d,h] = m.slotM; const [x,y] = m.posM;
    items.set(m.id, {arche: m.archetype, x, y, w, d, h, yaw: m.yawDeg || 0});
    add3D(m.id, m.archetype, x, y, w, d, h, m.yawDeg || 0);
    counts.placed++;
    if (!view2.fitDone && items.size >= 8) fitAll2();
  } else if (m.type === 'rejected'){
    const sl = m.slotM || [0.5, 0.5, 0.5]; const [x,y] = m.posM;
    recent.push({until: performance.now()/1000 + 1.4, x, y, w: sl[0], d: sl[1]});
    flash3D(x, y, sl[0], sl[1]);
    counts.rejected++;
  } else if (m.type === 'removed'){
    items.delete(m.id);
    remove3D(m.id);
    counts.removed++;
  }
}

// Buffer SSE events and drain in batches so 200+ rapid arrivals don't
// trigger 200+ scene mutations between frames (visible stutter).
const _evQueue = [];
const es = new EventSource('/events');
es.onmessage = (ev) => {
  let m; try { m = JSON.parse(ev.data); } catch (_) { return; }
  _evQueue.push(m);
};
es.onerror = () => { document.getElementById('status').textContent = 'event stream disconnected'; };
function drainEvents(){
  // process up to N per frame to keep UI responsive
  const N = 80;
  for (let i = 0; i < N && _evQueue.length; i++){
    applyEv(_evQueue.shift());
  }
  if (_evQueue.length > 0){
    document.getElementById('status').textContent =
      `processing... ${_evQueue.length} events queued (placed:${counts.placed})`;
  }
  requestAnimationFrame(drainEvents);
}
requestAnimationFrame(drainEvents);

setInterval(() => {
  while (recent.length && recent[0].until < performance.now()/1000) recent.shift();
  render2D();
  document.getElementById('status').textContent =
    `events:${counts.events}  placed:${counts.placed}  rejected:${counts.rejected}  removed:${counts.removed}  index:${items.size}  | left=2D, right=3D | 2D: scroll/drag, 3D: drag rotate, wheel zoom`;
}, 60);

// load template + initial stage graph on startup
async function bootstrap(){
  template = await fetchTemplate();
  rebuildTemplate3();
  // seed anchor placements (drawn as muted-color placeholders before any events arrive)
  const seed = await fetchStageGraph();
  for (const p of seed){
    const [w,d,h] = p.slotM; const [x,y] = p.posM;
    items.set(p.id, {arche: p.archetype, x, y, w, d, h, yaw: p.yawDeg || 0, anchor: true});
    add3D(p.id, p.archetype, x, y, w, d, h, p.yawDeg || 0, true);
  }
  if (items.size && !view2.fitDone) fitAll2();
}
bootstrap();
</script></body></html>
"""


class _Handler(BaseHTTPRequestHandler):
    events_path: Path = Path()  # set by main()
    session_port: int = 8766
    # Follow-latest mode: when set, _stream_events periodically re-scans
    # follow_dir for the newest events.jsonl and switches to it. Used to
    # auto-track concurrent or successive sessions without restarting
    # viz_web manually.
    follow_dir: Path | None = None
    follow_interval_s: float = 1.0

    def log_message(self, *args, **kwargs):
        pass  # silence access log

    @classmethod
    def _scan_latest(cls) -> tuple[Path | None, int | None]:
        """Find the most recently-modified events.jsonl under follow_dir.
        Returns (path, session_port_or_None). The session port is
        inferred from the parent directory name when that name parses
        as a valid integer (the per-port live-events convention).
        """
        if cls.follow_dir is None or not cls.follow_dir.exists():
            return None, None
        best: tuple[float, Path] | None = None
        for p in cls.follow_dir.rglob("events.jsonl"):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if best is None or mt > best[0]:
                best = (mt, p)
        if best is None:
            return None, None
        path = best[1]
        port: int | None = None
        try:
            port = int(path.parent.name)
        except ValueError:
            port = None
        return path, port

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = INDEX_HTML.replace("__SESSION_PORT__", str(self.session_port)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/static/"):
            name = self.path[len("/static/"):]
            # disallow path traversal
            if "/" in name or ".." in name:
                self.send_response(403); self.end_headers(); return
            static_dir = Path(__file__).resolve().parent / "static"
            file_path = static_dir / name
            if not file_path.exists():
                self.send_response(404); self.end_headers(); return
            ct = "application/javascript" if name.endswith(".js") else "application/octet-stream"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self._cors()
            self.end_headers()
            self._stream_events()
            return
        self.send_response(404)
        self.end_headers()

    def _stream_events(self):
        offset = 0
        last_follow_check = 0.0
        current_path = type(self).events_path  # read once from class state
        try:
            while True:
                # Follow-latest: periodically check whether a newer
                # events.jsonl appeared under follow_dir. If so, switch
                # to it and emit synthetic session_open so the viz
                # clears state and rebinds session_port + template.
                if type(self).follow_dir is not None and (
                        time.time() - last_follow_check > type(self).follow_interval_s):
                    last_follow_check = time.time()
                    latest, port = type(self)._scan_latest()
                    if latest is not None and latest != current_path:
                        current_path = latest
                        offset = 0
                        type(self).events_path = latest
                        if port is not None:
                            type(self).session_port = port
                        self.wfile.write(
                            b"data: {\"type\": \"session_open\", "
                            b"\"reason\": \"viz_followed_to_newer_session\", "
                            b"\"path\": " + json.dumps(str(latest)).encode("utf-8") + b", "
                            b"\"sessionPort\": " + str(port or 0).encode("utf-8") + b"}\n\n")
                        self.wfile.flush()

                if current_path.exists():
                    try:
                        # Detect file truncation/rotation: if file is now
                        # smaller than our offset, the run was reset — emit a
                        # synthetic session_open so the viz clears state.
                        size = current_path.stat().st_size
                        if size < offset:
                            offset = 0
                            self.wfile.write(b"data: {\"type\": \"session_open\", \"reason\": \"file_truncated\"}\n\n")
                            self.wfile.flush()
                        with current_path.open("rb") as f:
                            f.seek(offset)
                            chunk = f.read()
                            offset = f.tell()
                    except FileNotFoundError:
                        chunk = b""
                    if chunk:
                        for line in chunk.decode("utf-8", errors="replace").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            self.wfile.write(b"data: " + line.encode("utf-8") + b"\n\n")
                        self.wfile.flush()
                time.sleep(0.12)
        except (BrokenPipeError, ConnectionResetError):
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events", nargs="?", default=None,
                    help="path to events.jsonl (will be tailed). Omit when "
                         "--follow-latest is set.")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--session-port", type=int, default=8766)
    ap.add_argument("--follow-latest", nargs="?", const=_DEFAULT_FOLLOW_ROOT,
                    default=None, metavar="DIR",
                    help="Auto-follow the most-recently-modified events.jsonl "
                         "anywhere under DIR (default $TMPDIR/sketch_live). When "
                         "a new session writes a newer events.jsonl, viz "
                         "switches to it and emits a synthetic session_open. "
                         "Session port auto-binds to <DIR>/<port>/events.jsonl "
                         "when the parent directory name parses as an int. "
                         "Use this instead of the positional `events` arg "
                         "for run.sh / multi-session workflows.")
    args = ap.parse_args()
    if args.follow_latest:
        _Handler.follow_dir = Path(args.follow_latest)
        _Handler.follow_dir.mkdir(parents=True, exist_ok=True)
        latest, port = _Handler._scan_latest()
        if latest is not None:
            _Handler.events_path = latest
            if port is not None:
                _Handler.session_port = port
        else:
            _Handler.events_path = _Handler.follow_dir / "events.jsonl"
            _Handler.session_port = args.session_port
        print(f"viz running at http://localhost:{args.port}/  "
              f"(follow-latest: {args.follow_latest}; "
              f"initial: {_Handler.events_path}; "
              f"session-port: {_Handler.session_port})", flush=True)
    else:
        if args.events is None:
            ap.error("either positional `events` or --follow-latest is required")
        _Handler.events_path = Path(args.events)
        _Handler.session_port = args.session_port
        print(f"viz running at http://localhost:{args.port}/  "
              f"(events: {args.events})", flush=True)
    # Bind IPv6 dual-stack so browsers that resolve "localhost" to ::1 don't
    # stall on a refused IPv6 connect (Chrome/Firefox try ::1 first and may
    # hang while waiting to fall back to 127.0.0.1).
    class _DualStackServer(ThreadingHTTPServer):
        address_family = socket.AF_INET6
        def server_bind(self):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except (AttributeError, OSError):
                pass
            super().server_bind()
    try:
        httpd = _DualStackServer(("::", args.port), _Handler)
    except OSError:
        # IPv6 unavailable — fall back to IPv4 only.
        httpd = ThreadingHTTPServer(("0.0.0.0", args.port), _Handler)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
