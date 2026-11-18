// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";
(() => {
  const data = window.OVD_GUIDE_FIXTURE;
  const byId = id => document.getElementById(id);
  const all = selector => [...document.querySelectorAll(selector)];
  const fmt = value => value === null ? "Unavailable" : value.toFixed(3);
  let frame = 1, metric = "position", timer = null;
  const metrics = {position: {label: "Position · Y", get: f => f.position?.[1] ?? null}, velocity: {label: "Velocity · Y", get: f => f.velocity?.[1] ?? null}, mass: {label: "Mass", get: f => f.mass}};
  function pressed(selector, key, selected) { all(selector).forEach(button => button.setAttribute("aria-pressed", String(button.dataset[key] === selected))); }
  function svgElement(name, attributes) { const e = document.createElementNS("http://www.w3.org/2000/svg", name); for (const [key,value] of Object.entries(attributes)) e.setAttribute(key,String(value)); return e; }
  function render() {
    const sample = data.frames[frame], selected = metrics[metric];
    byId("frame").value = String(frame);
    byId("frame-label").textContent = `0${frame} / 03`;
    byId("inspect-frame").textContent = String(frame);
    byId("metric-title").textContent = selected.label;
    byId("plot-title").textContent = `Recorded ${selected.label.toLowerCase()} across four frames; selected frame ${frame}`;
    const values = data.frames.map(selected.get), available = values.filter(v => v !== null);
    let lo = Math.min(...available), hi = Math.max(...available);
    const padding = Math.max((hi-lo)*0.16, Math.abs(hi)*0.06, 0.1); lo -= padding; hi += padding;
    const px = i => 55+i*180, py = value => 205-(value-lo)/(hi-lo)*160;
    byId("axis-top").textContent = hi.toFixed(1);
    byId("axis-bottom").textContent = lo.toFixed(1);
    let path = "", connected = false;
    values.forEach((value,index) => { if(value === null) { connected=false; return; } path += `${connected?"L":"M"}${px(index)},${py(value)} `; connected=true; });
    byId("trace").setAttribute("d",path);
    byId("plot-points").replaceChildren();
    values.forEach((value,index) => {
      const x=px(index);
      if(value === null) { byId("plot-points").append(svgElement("path",{d:`M${x-5} 220L${x+5} 230M${x-5} 230L${x+5} 220`,class:"missing-point"})); }
      else byId("plot-points").append(svgElement("circle",{cx:x,cy:py(value),r:index===frame?9:6,class:`data-point ${index===frame?"selected":""}`}));
    });
    byId("cursor").setAttribute("x1",px(frame)); byId("cursor").setAttribute("x2",px(frame));
    byId("alive").textContent = sample.alive ? "Alive" : "Unavailable";
    byId("alive").classList.toggle("unavailable",!sample.alive);
    byId("object-name").textContent = sample.alive ? sample.name : "Actor 8";
    for(const [id,value] of [["inspect-position",sample.position?.[1]??null],["inspect-velocity",sample.velocity?.[1]??null],["inspect-mass",sample.mass]]) {
      byId(id).textContent=fmt(value); byId(id).classList.toggle("missing",value===null);
    }
    byId("inspect-shapes").textContent=sample.shapes === null ? "Unavailable" : JSON.stringify(sample.shapes);
    byId("inspect-force").textContent=sample.alive ? "No sample" : "Unavailable";
    byId("inspect-force").classList.add("missing");
    const explanations = ["Actor 8 is not available at this frame. There is no value to plot. This is not a position or velocity of zero.","The actor is alive. Shape [7] is a typed reference; force has no scalar sample at this frame.","Velocity Y is a recorded zero; mass is still 2. Force remains absent. Compare their availability, not just the numbers.","Actor 8 is no longer available here. Keep its earlier values attached to their own frames."];
    byId("frame-explanation").textContent=explanations[frame];
  }
  function stop() { if(timer!==null) clearInterval(timer); timer=null; byId("play").textContent="▶"; byId("play").setAttribute("aria-pressed","false"); byId("play").setAttribute("aria-label","Play recorded frames"); }
  byId("frame").addEventListener("input",event => {stop();frame=Number(event.target.value);render();});
  all("[data-metric]").forEach(button=>button.addEventListener("click",()=>{metric=button.dataset.metric;pressed("[data-metric]","metric",metric);render();}));
  byId("play").addEventListener("click",()=>{if(timer!==null){stop();return;}byId("play").textContent="Ⅱ";byId("play").setAttribute("aria-pressed","true");byId("play").setAttribute("aria-label","Pause recorded frames");timer=setInterval(()=>{frame=(frame+1)%data.frames.length;render();},1400);});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});

  function block(tag,text,cls){const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(cls)node.className=cls;return node;}
  const availability = {
    zero:["0.000","present","Zero is a value.","In the fixture, actor 8 has a recorded velocity Y of zero at frame 2. Keep the zero and its context; do not turn it into a missing-data marker.","Compare another recorded frame or component."],
    absent:["—","absent_at_frame","No scalar sample here.","The actor is alive in frames 1 and 2, but force has no scalar sample at those frames. That does not establish that force was zero, or that the producer records it on every path.","Check another frame and the producer's recording coverage."],
    projection:["?","not_projected","This view did not include it.","Measured in multi_geom.ovd, scene 3, frame 2: body #11 mass returns not_projected without an inspector request, then present with value 10 when inspect_ids=[11]. The capture is unchanged.","Request a new coherent view with the object or required byte window."]
  };
  Object.values(availability).forEach(s=>{const card=block("div",undefined,"availability-card"),value=block("div",undefined,"value-display"),body=block("div");value.append(block("span",s[0]),block("code",s[1]));body.append(block("h3",s[2]),block("p",s[3]),block("p","Next: "+s[4],"next-action"));card.append(value,body);byId("availability-list").append(card);});
  const commands={windows:`$bundle = 'C:/work/OvdNext-windows-x86_64'\n$capture = 'C:/captures/example.ovd'\n& "$bundle/python/python.exe" -I -B ./inspect_ovd.py $bundle $capture\nif ($LASTEXITCODE -ne 0) { throw 'OVD inspection failed' }`,linux:`bundle='/work/OvdNext-linux-x86_64'\ncapture='/captures/example.ovd'\nenv -u PYTHONHOME -u PYTHONPATH -u LD_PRELOAD \\\n  LD_LIBRARY_PATH="$bundle/lib:$bundle/python/lib" \\\n  "$bundle/python/bin/python3.12" -I -B ./inspect_ovd.py "$bundle" "$capture"`};
  for(const platform of ["windows","linux"])byId("command-code-"+platform).textContent=commands[platform];
  async function copy(sourceId,statusId) {
    const source=byId(sourceId), status=byId(statusId);
    try { if(!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable"); await navigator.clipboard.writeText(source.textContent); status.textContent="Copied."; }
    catch {const range=document.createRange();range.selectNodeContents(source);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);status.textContent="Text selected. Copy with Ctrl+C (or Command+C).";}
  }
  for(const platform of ["windows","linux"])byId("copy-command-"+platform).addEventListener("click",()=>copy("command-code-"+platform,"copy-command-status-"+platform));
  byId("copy-prompt").addEventListener("click",()=>copy("agent-prompt","copy-prompt-status"));
  render();
})();
