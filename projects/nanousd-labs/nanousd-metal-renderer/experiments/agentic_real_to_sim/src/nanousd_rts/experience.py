"""High-fidelity streamed Gaussian viewer plus articulation inspection UI."""

from __future__ import annotations

import html
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .core import RealToSimError, Workspace
from .preview import write_preview
from .sim import sweep_joint
from .visual_articulation import materialize_articulated_sog
from .visual_completion import materialize_visual_completions


SUPERSPLAT_VIEWER_PACKAGE = "@playcanvas/supersplat-viewer@1.27.1"
SUPERSPLAT_NANOUSD_PATCH = 2


def _visual_source(workspace: Workspace) -> tuple[Path, str, dict[str, Any]]:
    provenance = workspace.state["source"].get("provenance", {})
    original = Path(provenance.get("original_path", workspace.source_path)).expanduser().resolve()
    if original.is_dir() and (original / "lod-meta.json").is_file():
        metadata = json.loads((original / "lod-meta.json").read_text(encoding="utf-8"))
        return original, "lod-meta.json", {
            "kind": "playcanvas-sog-lod",
            "gaussian_count": int(metadata.get("count", 0)),
            "lod_counts": [int(value) for value in metadata.get("counts", [])],
            "lod_levels": int(metadata.get("lodLevels", 0)),
            "generator": metadata.get("asset", {}).get("generator"),
        }
    if original.is_file() and original.suffix.lower() == ".ply":
        return original, "content.ply", {
            "kind": "standard-3dgs-ply",
            "gaussian_count": int(workspace.state["source"]["report"]["particle_count"]),
            "lod_counts": [],
            "lod_levels": 1,
            "generator": None,
        }
    raise RealToSimError(
        "high-fidelity preview requires the original SOG/LOD directory or PLY source; "
        f"recorded source is unavailable: {original}"
    )


def _replace_link(link: Path, target: Path) -> None:
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    link.symlink_to(target, target_is_directory=target.is_dir())


def _materialize_supersplat_viewer(destination: Path) -> dict[str, Any]:
    marker = destination / "viewer-package.json"
    required = (destination / "index.html", destination / "index.css", destination / "index.js")
    if marker.is_file() and all(path.is_file() for path in required):
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if (
            metadata.get("package") == SUPERSPLAT_VIEWER_PACKAGE
            and metadata.get("nanousd_patch") == SUPERSPLAT_NANOUSD_PATCH
        ):
            return metadata

    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nanousd-supersplat-viewer-") as temporary:
        completed = subprocess.run(
            [
                "npm",
                "pack",
                SUPERSPLAT_VIEWER_PACKAGE,
                "--silent",
                "--pack-destination",
                temporary,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RealToSimError(
                "failed to materialize the pinned SuperSplat viewer package:\n"
                + (completed.stderr or completed.stdout or "npm pack failed")
            )
        archives = sorted(Path(temporary).glob("*.tgz"))
        if len(archives) != 1:
            raise RealToSimError("npm pack did not produce exactly one viewer archive")
        with tarfile.open(archives[0], mode="r:gz") as archive:
            for source_name, output_name in (
                ("package/public/index.html", "index.html"),
                ("package/public/index.css", "index.css"),
                ("package/public/index.js", "index.js"),
                ("package/LICENSE", "LICENSE.supersplat-viewer"),
            ):
                member = archive.getmember(source_name)
                stream = archive.extractfile(member)
                if stream is None:
                    raise RealToSimError(f"viewer package is missing {source_name}")
                payload = stream.read()
                if output_name == "index.html":
                    text = payload.decode("utf-8")
                    import_needle = "import { main } from './index.js';"
                    import_replacement = "import { main, Asset, Entity } from './index.js';"
                    if import_needle not in text:
                        raise RealToSimError("viewer module import contract changed")
                    text = text.replace(import_needle, import_replacement, 1)
                    needle = "const viewer = await main(canvas, settingsJson, config);"
                    replacement = (
                        "const viewer = await main(canvas, settingsJson, config);\n"
                        "                window.nanousdViewer = viewer;\n"
                        "                window.nanousdPlayCanvas = { Asset, Entity };\n"
                        "                window.dispatchEvent(new CustomEvent("
                        "'nanousd-viewer-ready', { detail: viewer }));"
                    )
                    if needle not in text:
                        raise RealToSimError("viewer bootstrap contract changed; expected main() call not found")
                    text = text.replace(needle, replacement, 1)
                    text = text.replace("<title>SuperSplat Viewer</title>", "<title>NanoUSD Visual Truth</title>", 1)
                    payload = text.encode("utf-8")
                elif output_name == "index.js":
                    text = payload.decode("utf-8")
                    export_needle = "export { main };"
                    if export_needle not in text:
                        raise RealToSimError("viewer module export contract changed")
                    payload = text.replace(
                        export_needle,
                        "export { main, Asset, Entity };",
                        1,
                    ).encode("utf-8")
                (destination / output_name).write_bytes(payload)

    metadata = {
        "package": SUPERSPLAT_VIEWER_PACKAGE,
        "license": "MIT",
        "source": "npm pack",
        "nanousd_patch": SUPERSPLAT_NANOUSD_PATCH,
    }
    marker.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return metadata


def _settings() -> dict[str, Any]:
    return {
        "version": 2,
        "tonemapping": "none",
        "highPrecisionRendering": True,
        "background": {"color": [0.004, 0.006, 0.01]},
        "postEffectSettings": {
            "sharpness": {"enabled": True, "amount": 0.2},
            "bloom": {"enabled": False, "intensity": 1, "blurLevel": 2},
            "grading": {
                "enabled": False,
                "brightness": 0,
                "contrast": 1,
                "saturation": 1,
                "tint": [1, 1, 1],
            },
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5},
        },
        "animTracks": [],
        "cameras": [
            {
                "initial": {
                    "position": [2, 1, -2],
                    "target": [0, 0, 0],
                    "fov": 60,
                }
            }
        ],
        "annotations": [],
        "startMode": "default",
    }


def _experience_payload(workspace: Workspace, visual: dict[str, Any], budget: float) -> dict[str, Any]:
    sweeps: dict[str, Any] = {}
    for node in workspace.nodes:
        if node.joint is None:
            continue
        sweeps[node.node_id] = sweep_joint(workspace, node_id=node.node_id)
    verification_path = workspace.root / "evidence" / "verification" / "report.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file()
        else None
    )
    return {
        "scene": {
            "up_axis": workspace.up_axis,
            "source_sha256": workspace.state["source"]["sha256"],
            "support_edges": workspace.state["support_edges"],
            "completion_candidates": workspace.state.get("completion_candidates", []),
        },
        "nodes": [node.to_json() for node in workspace.nodes],
        "sweeps": sweeps,
        "verification": verification,
        "visual": {**visual, "viewer_budget_millions": budget},
    }


EXPERIENCE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>NanoUSD Home Scan Real-to-Sim</title>
<style>
:root { color-scheme:dark; --bg:#080b10; --panel:#101722; --line:#29384e; --text:#edf3fc; --muted:#94a4ba; --orange:#ff7b32; --green:#48d37d; --red:#ff626a; }
* { box-sizing:border-box; }
html,body { width:100%; height:100%; margin:0; overflow:hidden; background:var(--bg); color:var(--text); font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; }
body { display:grid; grid-template-rows:auto minmax(0,1fr); }
header { min-width:0; display:flex; justify-content:space-between; gap:16px; align-items:center; padding:12px 16px; border-bottom:1px solid var(--line); background:#0b1018; }
h1 { margin:0; font:600 17px/1.2 system-ui,sans-serif; }
.sub { color:var(--muted); margin-top:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.badge { flex:none; padding:5px 9px; border:1px solid var(--line); border-radius:999px; }
.ok { color:var(--green); } .bad { color:var(--red); } .muted { color:var(--muted); }
main { min-width:0; min-height:0; display:grid; grid-template-columns:minmax(0,1fr) 350px; }
.visual { min-width:0; min-height:0; position:relative; background:#020305; }
iframe { display:block; width:100%; height:100%; border:0; }
.protocol-warning { display:none; position:absolute; inset:24px; z-index:4; place-items:center; text-align:center; padding:24px; background:#101722ee; border:1px solid var(--orange); border-radius:12px; }
.protocol-warning code { display:block; margin-top:12px; color:#fff; user-select:all; }
aside { min-width:0; min-height:0; overflow:auto; padding:16px; border-left:1px solid var(--line); background:var(--panel); }
section { margin-bottom:20px; }
h2 { margin:0 0 9px; color:var(--muted); font:600 12px/1.2 system-ui,sans-serif; text-transform:uppercase; letter-spacing:.09em; }
.row { display:flex; justify-content:space-between; gap:10px; padding:7px 0; border-bottom:1px solid #1c2838; }
select,input,button { width:100%; accent-color:var(--orange); color:var(--text); background:#0c121c; border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; }
button { cursor:pointer; }
button:hover:not(:disabled) { border-color:var(--orange); }
button:disabled,input:disabled { cursor:not-allowed; opacity:.45; }
.controls { display:grid; grid-template-columns:1fr auto; gap:9px; align-items:center; }
.controls select,.controls input { grid-column:1/-1; }
.joint-buttons { grid-column:1/-1; display:grid; grid-template-columns:1fr 1fr; gap:8px; }
output { color:var(--orange); }
.proxy-frame { height:230px; overflow:hidden; border:1px solid var(--line); border-radius:8px; background:radial-gradient(circle at 50% 45%,#172235,#090d14 72%); }
canvas { display:block; width:100%; height:100%; }
a { color:#76b9ff; }
.note { padding:10px; border:1px solid var(--line); border-radius:7px; color:var(--muted); background:#0b111a; }
@media(max-width:900px) {
  html,body { height:auto; min-height:100%; overflow:auto; }
  body { display:block; }
  main { display:block; }
  .visual { height:70vh; min-height:480px; }
  aside { border-left:0; border-top:1px solid var(--line); overflow:visible; }
}
</style>
</head>
<body>
<header>
  <div><h1>Home Scan · articulated real-to-sim</h1><div class="sub" id="qualityLine"></div></div>
  <div class="badge" id="status">verification pending</div>
</header>
<main>
  <div class="visual">
    <iframe id="viewer" title="High-fidelity streamed Gaussian scene" src="__VIEWER_URL__" allow="fullscreen; xr-spatial-tracking"></iframe>
    <div class="protocol-warning" id="protocolWarning">
      <div>
        <strong>The streamed SOG viewer requires local HTTP.</strong>
        <code>nanousd-rts serve-preview __WORKSPACE__ --open</code>
      </div>
    </div>
  </div>
  <aside>
    <section>
      <h2>Articulation oracle</h2>
      <div class="controls">
        <select id="joint"></select>
        <input id="state" type="range" min="0" max="1000" value="0">
        <div class="joint-buttons"><button id="closeJoint" type="button">Close</button><button id="openJoint" type="button">Open</button></div>
        <span class="muted">joint state</span><output id="value">closed</output>
      </div>
      <div class="note" id="visualState" style="margin-top:10px">Loading measured fronts and generated interiors…</div>
      <div class="proxy-frame" id="proxyFrame"><canvas id="proxy"></canvas></div>
    </section>
    <section><h2>Hard gates</h2><div id="gates"></div></section>
    <section><h2>Scene graph</h2><div id="tree"></div></section>
    <section><h2>Hidden geometry</h2><div id="completions"></div></section>
    <section>
      <h2>Representation contract</h2>
      <div class="note">The streamed background is the original measured SOG with articulated opacity regions losslessly masked. Door and drawer fronts remain measured LOD0 splat assets. Static cavities and moving backsides, liners, bins, racks, and drawer boxes are separate generated assets and are never relabeled as scanned evidence.</div>
      <div style="margin-top:9px"><a href="./physics.html" target="_blank">Open full physics inspector</a></div>
    </section>
  </aside>
</main>
<script>
const data=__PAYLOAD__;
const colors={background:"#303846",static:"#596579",movable:"#3097ff",articulated:"#ff7b32"};
const jointSelect=document.querySelector("#joint");
const slider=document.querySelector("#state");
const valueOut=document.querySelector("#value");
const closeButton=document.querySelector("#closeJoint");
const openButton=document.querySelector("#openJoint");
const visualState=document.querySelector("#visualState");
const viewerFrame=document.querySelector("#viewer");
const canvas=document.querySelector("#proxy");
const proxyFrame=document.querySelector("#proxyFrame");
const ctx=canvas.getContext("2d");
let resizeRequest=0;
let visualReady=false;
const visualJoints={};
const jointStates=Object.fromEntries(Object.keys(data.sweeps).map(id=>[id,0]));

const fmtCount=n=>new Intl.NumberFormat().format(n||0);
document.querySelector("#qualityLine").textContent=
  `${fmtCount(data.visual.gaussian_count)} source Gaussians · ${data.visual.lod_levels} LODs · ${data.visual.viewer_budget_millions}M live budget · ${data.visual.articulation?.objects?.length||0} measured fronts · ${data.visual.generated?.assets?.length||0} generated interior parts`;
if(location.protocol==="file:"){
  document.querySelector("#protocolWarning").style.display="grid";
  document.querySelector("#viewer").style.visibility="hidden";
}

function gates(){
  const values=data.verification?.gates||{};
  document.querySelector("#gates").innerHTML=Object.keys(values).length
    ? Object.entries(values).map(([name,ok])=>`<div class="row"><span>${name}</span><strong class="${ok?'ok':'bad'}">${ok?'PASS':'FAIL'}</strong></div>`).join("")
    : '<div class="muted">Run verify to populate gates.</div>';
  const status=document.querySelector("#status");
  if(data.verification){
    status.textContent=data.verification.passed?"PASS · fail-closed gates":"FAIL · not promotable";
    status.className="badge "+(data.verification.passed?"ok":"bad");
  }
}
function tree(){
  const byParent={};
  for(const n of data.nodes)(byParent[n.support_parent||"ROOT"]||=[]).push(n);
  const draw=parent=>(byParent[parent]||[]).map(n=>
    `<div style="padding:6px 0 6px 11px;border-left:2px solid #2b3b52"><strong>${n.id}</strong> <span class="muted">${n.role}</span>`+
    `<div class="muted">${fmtCount(n.selected_gaussians)} measured${n.joint?' · '+n.joint.kind:''}</div>${draw(n.id)}</div>`
  ).join("");
  document.querySelector("#tree").innerHTML=draw("ROOT");
}
function completions(){
  const items=data.scene.completion_candidates||[];
  document.querySelector("#completions").innerHTML=items.length?items.map(item=>
    `<div class="row"><span>${item.id}<br><small class="muted">${fmtCount(item.generated_gaussians)} generated · non-measured</small></span>`+
    `<strong class="${item.status==='accepted'?'ok':'muted'}">${item.status}</strong></div>`
  ).join(""):'<div class="muted">No generated candidates.</div>';
}
function boundsFromCenterSize(c,s){return {min:c.map((v,i)=>v-s[i]/2),max:c.map((v,i)=>v+s[i]/2)};}
function currentBounds(node){
  const sweep=data.sweeps[node.id];
  if(!sweep)return node.collider?boundsFromCenterSize(node.collider.center,node.collider.size):node.visual_bounds;
  const i=Math.round((jointStates[node.id]||0)*(sweep.samples.length-1));
  return sweep.samples[i].bounds;
}
function resize(){
  if(resizeRequest)cancelAnimationFrame(resizeRequest);
  resizeRequest=requestAnimationFrame(()=>{
    resizeRequest=0;
    const r=proxyFrame.getBoundingClientRect(),d=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.max(1,Math.floor(r.width*d));canvas.height=Math.max(1,Math.floor(r.height*d));
    ctx.setTransform(d,0,0,d,0,0);draw();
  });
}
function draw(){
  const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);
  const up={X:0,Y:1,Z:2}[data.scene.up_axis],horizontal=[0,1,2].filter(i=>i!==up),ax=horizontal[0],ay=up;
  const boxes=data.nodes.filter(n=>n.collider).map(node=>({node,b:currentBounds(node)}));
  if(!boxes.length)return;
  const min=[Math.min(...boxes.map(x=>x.b.min[ax])),Math.min(...boxes.map(x=>x.b.min[ay]))];
  const max=[Math.max(...boxes.map(x=>x.b.max[ax])),Math.max(...boxes.map(x=>x.b.max[ay]))];
  const span=[Math.max(max[0]-min[0],1e-6),Math.max(max[1]-min[1],1e-6)];
  const scale=Math.min((w-34)/span[0],(h-34)/span[1]);
  const map=(x,y)=>[17+(x-min[0])*scale,h-17-(y-min[1])*scale];
  for(const {node,b} of boxes){
    const p0=map(b.min[ax],b.min[ay]),p1=map(b.max[ax],b.max[ay]),x=p0[0],y=p1[1],rw=p1[0]-p0[0],rh=p0[1]-p1[1];
    ctx.fillStyle=colors[node.role]+"80";ctx.strokeStyle=colors[node.role];ctx.lineWidth=node.id===jointSelect.value?3:1;
    ctx.fillRect(x,y,rw,rh);ctx.strokeRect(x,y,rw,rh);
    if(node.id===jointSelect.value){ctx.fillStyle="#fff";ctx.font="11px ui-monospace";ctx.fillText(node.id,x+4,y+13);}
  }
}
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const enginePoint=point=>[-point[0],-point[1],point[2]];
function setVisualStatus(message,kind="muted"){
  visualState.textContent=message;
  visualState.className=`note ${kind}`;
}
async function waitForViewer(){
  for(let attempt=0;attempt<300;attempt++){
    const frameWindow=viewerFrame.contentWindow;
    const viewer=frameWindow?.nanousdViewer;
    const classes=frameWindow?.nanousdPlayCanvas;
    if(viewer?.global?.app&&classes?.Asset&&classes?.Entity)return {viewer,classes};
    await delay(100);
  }
  throw new Error("Timed out waiting for the streamed Gaussian viewer");
}
function loadSplatEntity(app,classes,item,{parent,enginePivot=null,name}){
  const {Asset,Entity}=classes;
  const url=new URL(item.url,viewerFrame.contentWindow.location.href).toString();
  return new Promise((resolve,reject)=>{
    const asset=new Asset(`nanousd:${item.id}`,"gsplat",{url,filename:"meta.json"});
    asset.once("load",()=>{
      const entity=new Entity(name);
      if(enginePivot)entity.setLocalPosition(-enginePivot[0],-enginePivot[1],-enginePivot[2]);
      entity.setLocalEulerAngles(0,0,180);
      entity.addComponent("gsplat",{unified:true,asset});
      parent.addChild(entity);
      resolve(entity);
    });
    asset.once("error",error=>reject(new Error(`${item.id}: ${error}`)));
    app.assets.add(asset);
    app.assets.load(asset);
  });
}
async function loadVisualObject(app,classes,item){
  const {Entity}=classes;
  const pivot=new Entity(`joint:${item.id}`);
  const enginePivot=enginePoint(item.joint.origin);
  pivot.setLocalPosition(...enginePivot);
  app.root.addChild(pivot);
  const entity=await loadSplatEntity(app,classes,item,{
    parent:pivot,
    enginePivot,
    name:`measured-splats:${item.id}`
  });
  return {app,pivot,entity,enginePivot,joint:item.joint,gaussianCount:item.gaussian_count};
}
function applyVisualJoint(id){
  const visual=visualJoints[id];
  if(!visual)return;
  const fraction=jointStates[id]||0;
  const joint=visual.joint;
  const value=joint.lower+(joint.upper-joint.lower)*fraction;
  const axisFactor=joint.axis==="Z"?1:-1;
  visual.pivot.setLocalPosition(...visual.enginePivot);
  visual.pivot.setLocalEulerAngles(0,0,0);
  if(joint.kind==="revolute"){
    const angle=value*joint.axis_sign*axisFactor;
    const euler={X:[angle,0,0],Y:[0,angle,0],Z:[0,0,angle]}[joint.axis];
    visual.pivot.setLocalEulerAngles(...euler);
  }else{
    const plyDelta={X:[value*joint.axis_sign,0,0],Y:[0,value*joint.axis_sign,0],Z:[0,0,value*joint.axis_sign]}[joint.axis];
    const delta=enginePoint(plyDelta);
    visual.pivot.setLocalPosition(
      visual.enginePivot[0]+delta[0],
      visual.enginePivot[1]+delta[1],
      visual.enginePivot[2]+delta[2]
    );
  }
  visual.app.renderNextFrame=true;
}
async function setupVisualArticulations(){
  const assets=data.visual.articulation?.objects||[];
  const generated=data.visual.generated||{static_assets:[],moving_assets:[],generated_gaussians:0};
  if(!assets.length){
    setVisualStatus("No movable Gaussian assets were generated for this source.","bad");
    return;
  }
  try{
    const {viewer,classes}=await waitForViewer();
    const loaded=await Promise.all(assets.map(item=>loadVisualObject(viewer.global.app,classes,item)));
    for(let index=0;index<assets.length;index++)visualJoints[assets[index].id]=loaded[index];
    await Promise.all((generated.static_assets||[]).map(item=>
      loadSplatEntity(viewer.global.app,classes,item,{
        parent:viewer.global.app.root,
        name:`generated-static:${item.id}`
      })
    ));
    await Promise.all((generated.moving_assets||[]).map(item=>{
      const visual=visualJoints[item.node];
      if(!visual)throw new Error(`Generated moving asset has no measured joint: ${item.node}`);
      return loadSplatEntity(viewer.global.app,classes,item,{
        parent:visual.pivot,
        enginePivot:visual.enginePivot,
        name:`generated-moving:${item.id}`
      });
    }));
    visualReady=true;
    slider.disabled=false;closeButton.disabled=false;openButton.disabled=false;
    for(const id of Object.keys(visualJoints))applyVisualJoint(id);
    setVisualStatus(
      `${assets.length} measured fronts · ${fmtCount(assets.reduce((sum,item)=>sum+item.gaussian_count,0))} measured LOD0 Gaussians · ${fmtCount(generated.generated_gaussians||0)} generated interior Gaussians`,
      "ok"
    );
  }catch(error){
    console.error(error);
    setVisualStatus(`Visual articulation failed: ${error.message}`,"bad");
  }
}
function setupJoints(){
  const ids=Object.keys(data.sweeps);
  jointSelect.innerHTML=ids.map(id=>`<option value="${id}">${id}</option>`).join("");
  slider.disabled=true;closeButton.disabled=true;openButton.disabled=true;updateJoint();
}
function updateJoint(){
  const sweep=data.sweeps[jointSelect.value];
  if(!sweep){valueOut.textContent="none";draw();return;}
  const fraction=Number(slider.value)/1000;
  jointStates[jointSelect.value]=fraction;
  const i=Math.round(fraction*(sweep.samples.length-1)),sample=sweep.samples[i];
  valueOut.textContent=`${sample.value.toFixed(3)} ${sweep.joint.kind==="revolute"?"deg":"m"}`;
  if(visualReady)applyVisualJoint(jointSelect.value);
  draw();
}
jointSelect.addEventListener("change",()=>{slider.value=Math.round((jointStates[jointSelect.value]||0)*1000);updateJoint();});
slider.addEventListener("input",updateJoint);
closeButton.addEventListener("click",()=>{slider.value=0;updateJoint();});
openButton.addEventListener("click",()=>{slider.value=1000;updateJoint();});
new ResizeObserver(resize).observe(proxyFrame);
window.addEventListener("resize",resize);
gates();tree();completions();setupJoints();resize();setupVisualArticulations();
</script>
</body>
</html>
"""


def write_experience(
    workspace: Workspace,
    *,
    output: Path | None = None,
    budget: float = 16.0,
) -> dict[str, Any]:
    if not budget > 0:
        raise RealToSimError("viewer budget must be positive")
    source, content_name, visual = _visual_source(workspace)
    output = Path(output).resolve() if output else None
    preview_root = output.parent if output else workspace.root / "preview"
    visual_root = preview_root / "visual"
    viewer_package = _materialize_supersplat_viewer(visual_root)
    if source.is_dir():
        _replace_link(visual_root / "content", source)
        background_occlusions = [
            {
                "id": item["id"],
                "bounds": item["visual_profile"]["background_occlusion_bounds"],
            }
            for item in workspace.completions
            if item.get("status") == "accepted"
            and isinstance(item.get("visual_profile"), dict)
            and item["visual_profile"].get("background_occlusion_bounds")
        ]
        articulation = materialize_articulated_sog(
            source,
            visual_root / "articulated",
            workspace.nodes,
            working_ply=workspace.source_path,
            selection_root=workspace.root,
            background_occlusions=background_occlusions,
        )
        content_url = articulation.get("background_url") or f"./content/{content_name}"
    else:
        _replace_link(visual_root / content_name, source)
        content_url = f"./{content_name}"
        articulation = {
            "schema_version": 1,
            "background_url": None,
            "objects": [],
            "masked_gaussians_by_lod": {},
            "background_occluded_gaussians_by_lod": {},
        }
    generated = materialize_visual_completions(
        workspace,
        visual_root / "generated",
    )
    (visual_root / "settings.json").write_text(
        json.dumps(_settings(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    physics = write_preview(workspace, output=preview_root / "physics.html")
    query: dict[str, Any] = {
        "content": content_url,
        "budget": f"{budget:g}",
        "hpr": "1",
    }
    voxel = workspace.root / "exports" / "voxel" / "scene" / "scene.voxel.json"
    if voxel.is_file():
        query["collision"] = Path(os.path.relpath(voxel, visual_root)).as_posix()
    viewer_url = f"./visual/index.html?{urlencode(query)}"
    visual.update(
        {
            "content": str(source),
            "viewer_package": viewer_package["package"],
            "collision": str(voxel) if voxel.is_file() else None,
            "articulation": articulation,
            "generated": generated,
        }
    )
    payload = json.dumps(
        _experience_payload(workspace, visual, budget),
        sort_keys=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    output = output or preview_root / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    document = (
        EXPERIENCE_HTML.replace("__VIEWER_URL__", html.escape(viewer_url, quote=True))
        .replace("__WORKSPACE__", html.escape(str(workspace.root), quote=False))
        .replace("__PAYLOAD__", payload)
    )
    output.write_text(document, encoding="utf-8")
    workspace.trace(
        "write-experience-preview",
        {"budget_millions": budget},
        {
            "html": str(output),
            "physics_html": str(physics),
            "visual_source": str(source),
            "viewer_package": viewer_package["package"],
        },
    )
    return {
        "html": str(output),
        "physics_html": str(physics),
        "visual_source": str(source),
        "viewer_package": viewer_package["package"],
        "budget_millions": budget,
        "requires_http": True,
    }


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        print(f"[nanousd-rts] {self.address_string()} {format % args}", flush=True)


def serve_preview(
    workspace: Workspace,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    budget: float = 16.0,
    open_browser: bool = False,
) -> dict[str, Any]:
    report = write_experience(workspace, budget=budget)
    handler = partial(_QuietHandler, directory=str(workspace.root))
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        raise RealToSimError(f"cannot serve preview on {host}:{port}: {exc}") from exc
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}/preview/index.html"
    print(
        json.dumps(
            {
                "status": "serving",
                "url": url,
                "workspace": str(workspace.root),
                "viewer_package": report["viewer_package"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if open_browser:
        subprocess.Popen(["open", url])
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {**report, "url": url, "status": "stopped"}
