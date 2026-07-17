"""Write a dependency-free interactive scene/joint inspection page."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from .core import Workspace
from .sim import sweep_joint


def _data_uri(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _latest_render(workspace: Workspace) -> Path | None:
    root = workspace.root / "evidence" / "render"
    candidates = sorted(root.glob("*/rgb.png"), key=lambda item: item.stat().st_mtime)
    return candidates[-1] if candidates else None


def _verification(workspace: Workspace) -> dict[str, Any] | None:
    path = workspace.root / "evidence" / "verification" / "report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if report.get("scene_digest") != workspace.state["logical_digest"]:
        return None
    return report


def write_preview(workspace: Workspace, *, output: Path | None = None) -> Path:
    sweeps = {}
    for node in workspace.nodes:
        if node.joint is not None:
            path = workspace.root / "evidence" / "sweeps" / node.node_id / "sweep.json"
            report = json.loads(path.read_text()) if path.is_file() else sweep_joint(workspace, node_id=node.node_id)
            sweeps[node.node_id] = report
    payload = {
        "scene": workspace.state,
        "nodes": [node.to_json() for node in workspace.nodes],
        "sweeps": sweeps,
        "verification": _verification(workspace),
        "gaussianRender": _data_uri(_latest_render(workspace)),
    }
    data = json.dumps(payload, sort_keys=True).replace("</", "<\\/")
    output = Path(output).resolve() if output else workspace.root / "preview" / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NanoUSD Agentic Real-to-Sim</title>
<style>
:root {{
  color-scheme: dark;
  --bg:#080b10; --panel:#101722; --line:#25344a; --text:#e7eef9;
  --muted:#91a1b7; --orange:#ff7b32; --blue:#3097ff; --green:#44d17a;
}}
* {{ box-sizing:border-box; }}
html,body {{ width:100%; height:100%; }}
body {{ margin:0; display:grid; grid-template-rows:auto minmax(0,1fr); overflow:hidden; font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--bg); color:var(--text); }}
header {{ padding:18px 22px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; }}
h1 {{ margin:0; font:600 18px/1.2 system-ui,sans-serif; letter-spacing:.01em; }}
.sub {{ color:var(--muted); margin-top:4px; }}
.badge {{ align-self:center; padding:5px 9px; border:1px solid var(--line); border-radius:999px; }}
main {{ min-width:0; min-height:0; display:grid; grid-template-columns:minmax(0,1fr) minmax(300px,360px); }}
.stage {{ min-width:0; min-height:0; padding:18px; display:grid; grid-template-rows:minmax(0,1fr) auto; gap:14px; }}
.canvas-frame {{ min-width:0; min-height:0; overflow:hidden; background:radial-gradient(circle at 50% 45%,#172235,#090d14 70%); border:1px solid var(--line); border-radius:10px; }}
canvas {{ display:block; width:100%; height:100%; }}
.controls {{ display:grid; grid-template-columns:160px 1fr auto; gap:12px; align-items:center; }}
select,input,button {{ accent-color:var(--orange); background:#101722; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; }}
button.active {{ border-color:var(--orange); color:#fff; }}
aside {{ min-width:0; min-height:0; border-left:1px solid var(--line); padding:18px; overflow:auto; background:var(--panel); }}
section {{ margin-bottom:22px; }}
h2 {{ font:600 13px/1.2 system-ui,sans-serif; text-transform:uppercase; letter-spacing:.09em; color:var(--muted); margin:0 0 10px; }}
.row {{ display:flex; justify-content:space-between; gap:10px; border-bottom:1px solid #1b2637; padding:7px 0; }}
.tree {{ padding-left:12px; border-left:2px solid #2b3b52; margin:5px 0; }}
.ok {{ color:var(--green); }} .bad {{ color:#ff5b62; }} .muted {{ color:var(--muted); }}
.photo {{ width:100%; border-radius:8px; border:1px solid var(--line); margin-top:8px; }}
.legend span {{ display:inline-flex; align-items:center; margin:0 12px 6px 0; }}
.dot {{ width:10px; height:10px; border-radius:2px; margin-right:6px; }}
pre {{ white-space:pre-wrap; word-break:break-word; font-size:11px; color:var(--muted); }}
@media(max-width:900px) {{
  html,body {{ height:auto; min-height:100%; }}
  body {{ display:block; overflow:auto; }}
  main {{ display:block; }}
  .stage {{ height:clamp(520px,calc(100vh - 72px),760px); }}
  aside {{ overflow:visible; border-left:0; border-top:1px solid var(--line); }}
}}
</style>
</head>
<body>
<header>
  <div><h1>NanoUSD Agentic Real-to-Sim</h1><div class="sub">Gaussian visual truth + registered physical scene truth</div></div>
  <div class="badge" id="status">local deterministic oracle</div>
</header>
<main>
  <div class="stage">
    <div class="canvas-frame" id="canvasFrame"><canvas id="scene"></canvas></div>
    <div>
      <div class="controls">
        <select id="joint"></select>
        <input id="state" type="range" min="0" max="1000" value="0">
        <output id="value">closed</output>
      </div>
      <div style="margin-top:10px">
        <button data-view="front" class="active">front</button>
        <button data-view="top">top</button>
        <button data-view="side">side</button>
      </div>
    </div>
  </div>
  <aside>
    <section><h2>Hard gates</h2><div id="gates"></div></section>
    <section><h2>Scene graph</h2><div id="tree"></div></section>
    <section><h2>Hidden completion</h2><div id="completions"></div></section>
    <section><h2>Legend</h2><div class="legend">
      <span><i class="dot" style="background:#596579"></i>static</span>
      <span><i class="dot" style="background:#3097ff"></i>movable</span>
      <span><i class="dot" style="background:#ff7b32"></i>articulated</span>
    </div></section>
    <section id="photoSection"><h2>Gaussian evidence</h2><img id="photo" class="photo"></section>
    <section><h2>Scope</h2><div class="muted">Interactive conservative collider and joint sweep preview. It is a local development oracle, not a claim of PhysX contact fidelity.</div></section>
    <section><h2>Selected joint</h2><pre id="jointInfo"></pre></section>
  </aside>
</main>
<script>
const data = {data};
const canvas = document.querySelector("#scene");
const canvasFrame = document.querySelector("#canvasFrame");
const ctx = canvas.getContext("2d");
const jointSelect = document.querySelector("#joint");
const slider = document.querySelector("#state");
const valueOut = document.querySelector("#value");
const colors = {{background:"#303846", static:"#596579", movable:"#3097ff", articulated:"#ff7b32"}};
let view = "front";
let resizeRequest = 0;

function gateRows() {{
  const gates = data.verification?.gates || {{}};
  const root = document.querySelector("#gates");
  if (!Object.keys(gates).length) {{ root.innerHTML='<div class="muted">Run verify to populate gates.</div>'; return; }}
  root.innerHTML = Object.entries(gates).map(([name,ok]) =>
    `<div class="row"><span>${{name}}</span><strong class="${{ok?'ok':'bad'}}">${{ok?'PASS':'FAIL'}}</strong></div>`
  ).join("");
  document.querySelector("#status").textContent = data.verification.passed ? "PASS · fail-closed gates" : "FAIL · not promotable";
  document.querySelector("#status").className = "badge " + (data.verification.passed ? "ok" : "bad");
}}

function treeRows() {{
  const byParent = {{}};
  for (const n of data.nodes) (byParent[n.support_parent || "ROOT"] ||= []).push(n);
  function draw(parent) {{
    return (byParent[parent] || []).map(n =>
      `<div class="tree"><div><strong>${{n.id}}</strong> <span class="muted">${{n.role}}</span></div>` +
      `<div class="muted">${{n.selected_gaussians.toLocaleString()}} Gaussians${{n.joint ? ' · '+n.joint.kind : ''}}</div>` +
      draw(n.id) + `</div>`
    ).join("");
  }}
  document.querySelector("#tree").innerHTML = draw("ROOT") || '<div class="muted">No nodes.</div>';
}}

function setupJoints() {{
  const ids = Object.keys(data.sweeps);
  jointSelect.innerHTML = ids.length
    ? ids.map(id => `<option value="${{id}}">${{id}}</option>`).join("")
    : '<option value="">no articulated nodes</option>';
  slider.disabled = !ids.length;
  updateJointInfo();
}}
function completionRows() {{
  const items=data.scene.completion_candidates || [];
  document.querySelector("#completions").innerHTML = items.length ? items.map(item =>
    `<div class="row"><span>${{item.id}}<br><small class="muted">${{item.generated_gaussians}} generated · not measured</small></span>`+
    `<strong class="${{item.status==='accepted'?'ok':'muted'}}">${{item.status}}</strong></div>`
  ).join("") : '<div class="muted">No generated candidates.</div>';
}}

function currentBounds(node) {{
  const sweep = data.sweeps[node.id];
  if (!sweep || jointSelect.value !== node.id) return node.collider
    ? boundsFromCenterSize(node.collider.center,node.collider.size)
    : node.visual_bounds;
  const t = Number(slider.value)/1000;
  const index = Math.round(t*(sweep.samples.length-1));
  return sweep.samples[index].bounds;
}}
function boundsFromCenterSize(c,s) {{
  return {{min:c.map((v,i)=>v-s[i]/2), max:c.map((v,i)=>v+s[i]/2)}};
}}
function axesForView() {{
  const up = {{X:0,Y:1,Z:2}}[data.scene.up_axis];
  const h = [0,1,2].filter(i=>i!==up);
  if (view==="top") return [h[0],h[1]];
  if (view==="side") return [h[1],up];
  return [h[0],up];
}}
function resize() {{
  if(resizeRequest) cancelAnimationFrame(resizeRequest);
  resizeRequest=requestAnimationFrame(()=>{{
    resizeRequest=0;
    const r=canvasFrame.getBoundingClientRect(), d=Math.min(window.devicePixelRatio||1,2);
    const width=Math.max(1,Math.floor(r.width*d)), height=Math.max(1,Math.floor(r.height*d));
    if(canvas.width!==width || canvas.height!==height){{canvas.width=width;canvas.height=height;}}
    ctx.setTransform(d,0,0,d,0,0); draw();
  }});
}}
function draw() {{
  const w=canvas.clientWidth,h=canvas.clientHeight; ctx.clearRect(0,0,w,h);
  const [ax,ay]=axesForView();
  const boxes=data.nodes.filter(n=>n.collider).map(n=>({{node:n,b:currentBounds(n)}}));
  if (!boxes.length) return;
  const min=[Math.min(...boxes.map(x=>x.b.min[ax])),Math.min(...boxes.map(x=>x.b.min[ay]))];
  const max=[Math.max(...boxes.map(x=>x.b.max[ax])),Math.max(...boxes.map(x=>x.b.max[ay]))];
  const span=[Math.max(max[0]-min[0],1e-6),Math.max(max[1]-min[1],1e-6)];
  const scale=Math.min((w-80)/span[0],(h-80)/span[1]);
  const map=(x,y)=>[40+(x-min[0])*scale,h-40-(y-min[1])*scale];
  ctx.lineWidth=1;
  ctx.strokeStyle="#26364d";
  for(let i=0;i<=10;i++){{const x=40+(w-80)*i/10;ctx.beginPath();ctx.moveTo(x,20);ctx.lineTo(x,h-20);ctx.stroke();}}
  for(let i=0;i<=10;i++){{const y=40+(h-80)*i/10;ctx.beginPath();ctx.moveTo(20,y);ctx.lineTo(w-20,y);ctx.stroke();}}
  for(const {{node,b}} of boxes){{
    const p0=map(b.min[ax],b.min[ay]),p1=map(b.max[ax],b.max[ay]);
    const x=p0[0],y=p1[1],rw=p1[0]-p0[0],rh=p0[1]-p1[1];
    ctx.fillStyle=colors[node.role]+"88";ctx.strokeStyle=colors[node.role];ctx.lineWidth=node.id===jointSelect.value?3:1.5;
    ctx.fillRect(x,y,rw,rh);ctx.strokeRect(x,y,rw,rh);
    ctx.fillStyle="#f2f6ff";ctx.font="12px ui-monospace";ctx.fillText(node.id,x+5,y+15);
  }}
  const edgeByChild=Object.fromEntries(data.scene.support_edges.map(e=>[e.child,e.parent]));
  for(const child of boxes){{
    const parent=boxes.find(x=>x.node.id===edgeByChild[child.node.id]);if(!parent)continue;
    const cc=[(child.b.min[ax]+child.b.max[ax])/2,(child.b.min[ay]+child.b.max[ay])/2];
    const pc=[(parent.b.min[ax]+parent.b.max[ax])/2,(parent.b.min[ay]+parent.b.max[ay])/2];
    const a=map(...cc),b=map(...pc);ctx.strokeStyle="#9fb2ca99";ctx.lineWidth=1;ctx.setLineDash([5,5]);
    ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.setLineDash([]);
  }}
}}
function updateJointInfo() {{
  const id=jointSelect.value, sweep=data.sweeps[id];
  if(!sweep){{document.querySelector("#jointInfo").textContent="No joint selected.";valueOut.textContent="closed";draw();return;}}
  const t=Number(slider.value)/1000, index=Math.round(t*(sweep.samples.length-1)), sample=sweep.samples[index];
  valueOut.textContent=`${{sample.value.toFixed(3)}} ${{sweep.joint.kind==="revolute"?"deg":"m"}}`;
  document.querySelector("#jointInfo").textContent=JSON.stringify({{
    kind:sweep.joint.kind, axis:`${{sweep.joint.axis_sign<0?'-':''}}${{sweep.joint.axis}}`,
    origin:sweep.joint.origin, limits:[sweep.joint.lower,sweep.joint.upper],
    fit_confidence:sweep.joint.confidence, sweep_gates:sweep.gates,
    collisions:sample.forbidden_collisions
  }},null,2); draw();
}}
document.querySelectorAll("button[data-view]").forEach(b=>b.addEventListener("click",()=>{{
  document.querySelectorAll("button[data-view]").forEach(x=>x.classList.remove("active"));b.classList.add("active");view=b.dataset.view;draw();
}}));
slider.addEventListener("input",updateJointInfo);jointSelect.addEventListener("change",()=>{{slider.value=0;updateJointInfo();}});
if(data.gaussianRender){{document.querySelector("#photo").src=data.gaussianRender;}}else{{document.querySelector("#photoSection").style.display="none";}}
gateRows();treeRows();completionRows();setupJoints();
new ResizeObserver(resize).observe(canvasFrame);
window.addEventListener("resize",resize);
resize();
</script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
    workspace.trace(
        "preview",
        {"output": str(output)},
        {"html": str(output), "joint_count": len(sweeps), "embedded_gaussian_render": payload["gaussianRender"] is not None},
    )
    return output
