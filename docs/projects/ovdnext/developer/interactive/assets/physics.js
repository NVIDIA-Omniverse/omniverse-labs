// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";
(() => {
  const captures=window.OVD_PHYSICS_DATA.captures;
  const cfg={
    rigid:{capture:0,object:11,up:1,task:"Need the mass and collision margin of this body? Select #11 (mass 10), then follow shapes → #10 (contactOffset ≈ 0.02). Compare globalPose, linearVelocity, mass and massSpaceInertiaTensor as you scrub. The arrow shows velocity direction; its magnitude stays in the inspector.",label:"Velocity"},
    chain:{capture:0,object:39,up:1,task:"Why does the chain translate at this joint? Select joint #39: its Y axis is limited to −0.1…0.1. Follow parentLink → joint → childLink and compare jointPosition with the limits. Select #35 or #43 to inspect the revolute joints.",label:"Joint axes"},
    drives:{capture:1,object:15,up:1,task:"Why does only twist move? Inspect spherical joint #15. Twist is free; the other axes are locked. The recorded driveTarget is 0.22, driveStiffness is 1 and driveDamping is 2. Stored limits do not constrain a free axis.",label:"Joint axes"},
    contacts:{capture:2,object:14,up:2,task:"Which surfaces contact the box? In OVD Viewer, select or focus the matching Pre-sim frame to see the Contacts overlay. This lab inspects the raw report: body #14 contacts two static meshes. Pick one of the 12 contact points, follow its actor pair, and compare normal impulse and separation. Scrub back to frame 1 to see empty contact arrays.",label:"Contact normals"}
  };
  function initializeLab(root,lab) {
  const $=id=>root.querySelector(`[data-field="${id}"]`), $$=s=>[...root.querySelectorAll(s)];
  let sample=1,object=cfg[lab].object,joint=object,contact=0,camera="iso",timer=null;
  const cap=()=>captures[cfg[lab].capture], frame=()=>cap().frames[sample];
  const fmt=v=>typeof v==="number" ? (Object.is(v,-0)?"0":String(Number(v.toPrecision(7)))) : String(v);
  function valueText(v){if(v===null||v===undefined)return "Undecoded";if(Array.isArray(v))return `[${v.slice(0,12).map(x=>typeof x==="object"?JSON.stringify(x):fmt(x)).join(", ")}${v.length>12?", …":""}]`;if(typeof v==="object")return Object.entries(v).map(([k,x])=>`${k}: ${valueText(x)}`).join("\n");return fmt(v);}
  const info=id=>cap().instances[id];
  function properties(id){return Object.entries(frame().objects[id]||{}).map(([pid,vi])=>({id:pid,...cap().schema[pid],...cap().values[vi]}));}
  const prop=(id,name)=>properties(id).find(p=>p.name===name);
  const val=(id,name)=>{const p=prop(id,name);return p?.availability==="present"?p.v:null;};
  const refs=(id,name)=>prop(id,name)?.refs||[];
  function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
  function svg(tag,attrs={}){const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;}
  function press(sel,key,current){$$(sel).forEach(b=>b.setAttribute("aria-pressed",String(b.dataset[key]===current)));}
  function preserveSelectionFocus(){
    const active=document.activeElement,area=active?.closest?.('[data-field="object-list"], [data-field="object-summary"], [data-field="attribute-list"], [data-field="scene-svg"], [data-field="joint-chain"], [data-field="contact-rows"]');
    const key=active?.hasAttribute?.('data-contact')?'contact':active?.hasAttribute?.('data-object')?'object':null;
    if(!area||!key||!root.contains(active))return ()=>{};
    const areaId=area.dataset.field,id=active.dataset[key];
    return ()=>{
      const replacement=$(areaId).querySelector(`[data-${key}="${id}"]${areaId==='scene-svg'?'[tabindex]':''}`);
      if(replacement&&!replacement.disabled)replacement.focus({preventScroll:true});
      else {$("object-title").tabIndex=-1;$("object-title").focus({preventScroll:true});}
    };
  }
  function stop(){clearInterval(timer);timer=null;$("physics-play").textContent="▶";$("physics-play").setAttribute("aria-pressed","false");$("physics-play").setAttribute("aria-label","Play sampled frames");}
  function chooseObject(id){object=Number(id);if(info(object)?.class==="PxArticulationJointReducedCoordinate")joint=object;$("attribute-filter").value="";render();}
  function refButton(id,label){const b=element("button",label||`${info(id)?.class||"Object"} #${id}`);b.type="button";b.dataset.object=id;b.disabled=!frame().objects[id];b.addEventListener("click",()=>chooseObject(id));return b;}
  function selectable(){return Object.keys(frame().objects).map(Number).filter(id=>{const c=info(id).class;return lab==="chain"||lab==="drives" ? c==="PxArticulationLink"||c==="PxArticulationJointReducedCoordinate" : c==="PxRigidDynamic"||c==="PxRigidStatic";});}
  function geometry(){return frame().geometry.filter(g=>lab==="chain"||lab==="drives" ? info(g.object)?.class==="PxArticulationLink" : lab==="rigid" ? !info(g.object)?.class.startsWith("PxArticulation") : true);}
  function contacts(){
    if(lab!=="contacts")return [];
    const scene=cap().scene,counts=val(scene,"pairsContactCounts")||[],points=val(scene,"pairsContactPoints")||[],normals=val(scene,"pairsContactNormals")||[],seps=val(scene,"pairsContactSeparations")||[],imp=val(scene,"pairsContactImpulses")||[],actors=prop(scene,"pairsActors")?.orderedRefs||[],shapes=prop(scene,"pairsContactShapes")?.orderedRefs||[];
    let offset=0;const result=[];
    counts.forEach((count,pair)=>{for(let n=0;n<count;n++,offset++)result.push({id:offset,pair,actors:actors.slice(pair*2,pair*2+2),shapes:shapes.slice(offset*2,offset*2+2),point:points.slice(offset*3,offset*3+3),normal:normals.slice(offset*3,offset*3+3),separation:seps[offset],impulse:imp[offset]});});return result;
  }
  function renderInspector(){
    const classname=info(object)?.class||"Unavailable object";
    $("object-title").textContent=`${classname} #${object}`;$("object-frame").textContent=`FRAME ${frame().index}`;
    const summary=$("object-summary");summary.replaceChildren();
    const names=classname.includes("Joint")?["type","parentLink","childLink"]:classname==="PxShape"?["geom","contactOffset","restOffset"]:classname==="PxScene"?["pairCount","gravity"]:["globalPose","mass","linearVelocity","shapes"];
    names.forEach(name=>{const p=prop(object,name);if(!p)return;const d=element("div");d.append(element("b",name));if(p.type==="ObjectRef")d.append(...(p.refs||[]).map(id=>refButton(id,`#${id}`)));else d.append(element("span",p.availability==="present"?valueText(p.v):p.availability));summary.append(d);});
    if(!summary.children.length)summary.append(element("div",`Scene ${cap().scene} · ${frame().alive.includes(object)?"alive":"unavailable"} at recorded frame ${frame().index}`));
    renderAttributes();
  }
  function renderAttributes(){
    const filter=$("attribute-filter").value.trim().toLowerCase(),all=properties(object),list=all.filter(p=>`${p.name} ${p.native_type}`.toLowerCase().includes(filter));
    $("attribute-count").textContent=`${list.length} / ${all.length} attributes · exact decoded values · raw previews up to 64 bytes`;
    const target=$("attribute-list");target.replaceChildren();
    for(const p of list){const row=element("div",undefined,"attribute");row.dataset.property=p.name;row.append(element("div",p.name,"prop-name"),element("div",`${p.native_type||p.type}${p.array?"[]":""} · ${p.size} bytes`,"prop-type"));
      const value=element("div",undefined,"prop-value");
      if(p.availability!=="present"){value.textContent=p.availability;value.classList.add("unavailable");}
      else if(p.type==="ObjectRef"){
        // Ordered producer-handle arrays have a different representation from membership references.
        const r=p.orderedRefs||p.refs||[];
        if(!r.length)value.textContent="[]";
        else r.forEach(id=>value.append(id===null?element("span","Unresolved "):refButton(id,`#${id}`)));
      }else {value.textContent=p.v===null?"Undecoded":typeof p.v==="object"?JSON.stringify(p.v):String(p.v);if(p.names&&typeof p.v==="number"){const exact=p.names.find(([n,v])=>v===p.v);if(exact)value.append(element("span",` · ${exact[0]}`));}}
      row.append(value);const more=element("div",undefined,"raw-details");
      const raw=[`Availability: ${p.availability}`,`Hex: ${p.hex||"(no bytes)"}`];
      if(p.handles?.length)raw.push(`Producer handles: ${p.handles.join(", ")}`);
      if(p.previewTruncated)raw.push(`Raw preview: first 64 of ${p.size} bytes`);
      if(p.valueTruncated)raw.push("Decoded value truncated at 64 KiB");
      more.append(element("span","Raw evidence","raw-label"),element("pre",raw.join("\n")));row.append(more);target.append(row);
    }
    if(!list.length)target.append(element("p","No matching attributes.","sample-note"));
  }
  function renderJoints(){
    if(!$("joint-panel"))return;
    const type=prop(joint,"type"),name=type?.names?.find(([n,v])=>v===type.v)?.[0]||String(type?.v);$("joint-type").textContent=`Joint #${joint} · ${name}`;
    const chain=$("joint-chain");chain.replaceChildren();refs(joint,"parentLink").forEach(id=>chain.append(refButton(id,`parentLink #${id}`)));chain.append(element("span",`→ joint #${joint} →`));refs(joint,"childLink").forEach(id=>chain.append(refButton(id,`childLink #${id}`)));
    const rows=$("joint-rows");rows.replaceChildren();const axes=["TWIST · X rotation","SWING1 · Y rotation","SWING2 · Z rotation","X translation","Y translation","Z translation"],motion=val(joint,"motion")||[];
    const at=(name,i)=>{const a=val(joint,name);return Array.isArray(a)&&i<a.length?fmt(a[i]):"No sample";};
    axes.forEach((axis,i)=>{const tr=element("tr");if(motion[i]===1||motion[i]===2)tr.className="active-row";[axis,["LOCKED","LIMITED","FREE"][motion[i]]??"No sample",at("jointPosition",i),`${at("limitLow",i)} / ${at("limitHigh",i)}`,at("driveTarget",i),at("driveStiffness",i),at("driveDamping",i)].forEach(t=>tr.append(element("td",t)));rows.append(tr);});
  }
  function renderContacts(){
    if(!$("contact-panel"))return;const restoreFocus=preserveSelectionFocus(),points=contacts();if(contact>=points.length)contact=0;
    const rows=$("contact-rows");rows.replaceChildren();
    points.forEach(p=>{const tr=element("tr");if(p.id===contact)tr.className="selected-contact";const td=element("td"),b=element("button",`#${p.id}`);b.type="button";b.dataset.contact=p.id;b.setAttribute("aria-pressed",String(p.id===contact));b.addEventListener("click",()=>{contact=p.id;renderContacts();draw();});td.append(b);tr.append(td);[p.actors.join(" ↔ "),valueText(p.point),valueText(p.normal),fmt(p.separation),fmt(p.impulse)].forEach(v=>tr.append(element("td",v)));rows.append(tr);});
    const selected=points[contact];$("contact-detail").textContent=selected?`Point #${selected.id} · actors ${selected.actors.join(" ↔ ")} · shapes ${selected.shapes.join(" ↔ ")} · separation ${fmt(selected.separation)} · normal impulse ${fmt(selected.impulse)}`:`No contacts in the arrays at recorded frame ${frame().index}. This is an empty sample, not twelve zero impulses.`;
    restoreFocus();
  }
  // All geometry uses the native snapshot's transform and dimensions. Meshes are captured triangles;
  // spheres/capsules are a tessellated display of their recorded analytic geometry.
  const add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,s)=>a.map(x=>x*s),length=a=>Math.hypot(...a);
  function transform(m,v){return [0,1,2].map(i=>m[i][0]*v[0]+m[i][1]*v[1]+m[i][2]*v[2]+m[i][3]);}
  function triangles(g){
    const d=g.dimensions;
    if(g.kind===5){const m=cap().meshes[g.mesh];if(!m)return [];const out=[];for(let i=0;i<m.indices.length;i+=3)out.push(m.indices.slice(i,i+3).map(j=>m.vertices[j]));return out;}
    if(g.kind===2){const v=Array.from({length:8},(_,i)=>d.map((x,j)=>((i>>j)&1)?x:-x));return [[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]].map(f=>f.map(i=>v[i]));}
    if(g.kind===1||g.kind===3){const sides=16,vertices=[],rings=[];
      if(g.kind===3){for(let i=0;i<=5;i++)rings.push([i/10*Math.PI,d[1]]);for(let i=5;i<=10;i++)rings.push([i/10*Math.PI,-d[1]]);}
      else for(let i=0;i<=10;i++)rings.push([i/10*Math.PI,0]);
      for(const [theta,offset] of rings){const r=d[0]*Math.sin(theta),x=d[0]*Math.cos(theta)+offset;vertices.push(Array.from({length:sides},(_,j)=>[x,r*Math.cos(j/sides*2*Math.PI),r*Math.sin(j/sides*2*Math.PI)]));}
      const out=[];for(let i=0;i<vertices.length-1;i++)for(let j=0;j<sides;j++)out.push([vertices[i][j],vertices[i][(j+1)%sides],vertices[i+1][(j+1)%sides],vertices[i+1][j]]);return out;}
    return [];
  }
  function draw(){
    const restoreFocus=preserveSelectionFocus(),root=$("scene-svg"),geoms=geometry();root.replaceChildren();const defs=svg("defs"),marker=svg("marker",{id:`${lab}-physics-arrow`,markerWidth:6,markerHeight:6,refX:5,refY:3,orient:"auto",markerUnits:"strokeWidth"});marker.append(svg("path",{d:"M0,0 L6,3 L0,6 Z",fill:"context-stroke"}));defs.append(marker);root.append(defs);
    const relevant=geoms.filter(g=>g.kind!==4),selectedGeoms=relevant.filter(g=>g.object===object||g.shape===object||g.geometry===object||refs(object,"childLink").includes(g.object)||refs(object,"parentLink").includes(g.object));
    const fitted=$("focus-selection").checked&&selectedGeoms.length?selectedGeoms:relevant,bounds=fitted.flatMap(g=>g.bounds),low=[0,1,2].map(i=>Math.min(...bounds.map(p=>p[i]))),high=[0,1,2].map(i=>Math.max(...bounds.map(p=>p[i]))),center=mul(add(low,high),.5),span=Math.max(...sub(high,low),.01);
    const up=cfg[lab].up,yaw=camera==="iso"?.6:0,pitch=camera==="iso"?.38:camera==="top"?Math.PI/2:0;
    function orient(v){const d=sub(v,center),a=up===2?[d[0],d[2],-d[1]]:d;const x=a[0]*Math.cos(yaw)+a[2]*Math.sin(yaw),z=-a[0]*Math.sin(yaw)+a[2]*Math.cos(yaw);return [x,a[1]*Math.cos(pitch)-z*Math.sin(pitch),a[1]*Math.sin(pitch)+z*Math.cos(pitch)];}
    const corner=[];for(let i=0;i<8;i++)corner.push(orient(low.map((x,j)=>((i>>j)&1)?high[j]:x)));const sx=Math.max(...corner.map(p=>p[0]))-Math.min(...corner.map(p=>p[0])),sy=Math.max(...corner.map(p=>p[1]))-Math.min(...corner.map(p=>p[1])),scale=Math.min(610/Math.max(sx,.01),325/Math.max(sy,.01));
    const project=v=>{const p=orient(v);return [360+p[0]*scale,218-p[1]*scale,p[2]];};
    function line(a,b,color,width=1,dashed=false,arrow=false){const p=project(a),q=project(b),e=svg("line",{x1:p[0],y1:p[1],x2:q[0],y2:q[1],stroke:color,"stroke-width":width});if(dashed)e.setAttribute("stroke-dasharray","4 3");if(arrow)e.setAttribute("marker-end",`url(#${lab}-physics-arrow)`);root.append(e);return e;}
    const floor=low.slice();floor[up]-=span*.06;const horizontal=[0,1,2].filter(i=>i!==up),gridSpan=span*.65;
    for(let n=-5;n<=5;n++){for(const ax of horizontal){const other=horizontal.find(i=>i!==ax),a=center.slice(),b=center.slice();a[up]=b[up]=floor[up];a[ax]=b[ax]=center[ax]+n*gridSpan/5;a[other]-=gridSpan;b[other]+=gridSpan;line(a,b,"#dce6df");}}
    const faces=[],colors=["#90bdcf","#b7b9df","#d5c28c","#8fc8aa","#a9c3db"];
    const related=info(object)?.class.includes("Joint")?[...refs(object,"parentLink"),...refs(object,"childLink")]:[];
    geoms.forEach((g,gi)=>{const selected=g.object===object||g.shape===object||g.geometry===object,linked=related.includes(g.object);for(const face of triangles(g)){const points=face.map(v=>project(transform(g.matrix,v)));faces.push({g,selected,points,depth:points.reduce((s,p)=>s+p[2],0)/points.length,color:selected?"#82cba0":linked?"#b8d9c3":colors[gi%colors.length]});}});
    faces.sort((a,b)=>a.depth-b.depth);
    faces.forEach(f=>{const p=svg("polygon",{points:f.points.map(v=>v.slice(0,2).join(",")).join(" "),fill:$("show-wire").checked?"none":f.color,"fill-opacity":lab==="contacts"?.52:.78,stroke:f.selected?"#007d37":"#58736c","stroke-width":f.selected?1.3:.65,"data-object":f.g.object});if(f.selected)p.classList.add("selected-geometry");p.addEventListener("click",()=>chooseObject(f.g.object));root.append(p);});
    // Labels provide keyboard-accessible picking without tabbing through every polygon.
    const seen=new Set();geoms.forEach(g=>{if(seen.has(g.object)||g.kind===4)return;seen.add(g.object);const c=project(mul(add(g.bounds[0],g.bounds[1]),.5)),button=svg("g",{role:"button",tabindex:0,"aria-label":`Inspect ${info(g.object)?.class} ${g.object}`,"data-object":g.object});button.append(svg("rect",{x:c[0]-16,y:c[1]-12,width:32,height:20,rx:4,fill:"#ffffff","fill-opacity":.88,stroke:g.object===object?"#007f35":"#a9bdb1"}));const text=svg("text",{x:c[0],y:c[1]+2,"text-anchor":"middle"});text.textContent=`#${g.object}`;button.append(text);button.addEventListener("click",()=>chooseObject(g.object));button.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();chooseObject(g.object);}});root.append(button);});
    if($("show-vectors").checked){
      if(lab==="rigid") {const pose=val(object,"globalPose"),v=val(object,"linearVelocity");if(pose?.p&&Array.isArray(v)&&length(v)>0)line(pose.p,add(pose.p,mul(v,span*.2/length(v))),"#ba5826",3,false,true);}
      if(lab==="chain"||lab==="drives") {for(const j of frame().jointFrames){if(info(object)?.class.includes("Joint")&&j.object!==object)continue;const colors=["#ce5959","#448d59","#497db7"],axis=j.rgba.slice(0,3).indexOf(Math.max(...j.rgba.slice(0,3))),direction=sub(j.points[1],j.points[0]);if(length(direction)>0)line(j.points[0],add(j.points[0],mul(direction,span*(j.color===7?.14:.1)/length(direction))),colors[axis],2,j.color===7,true);}}
    }
    if(lab==="contacts")for(const p of contacts()){const xy=project(p.point),selected=p.id===contact;if($("show-vectors").checked)line(p.point,add(p.point,mul(p.normal,.12)),selected?"#a84a09":"#b67b42",selected?2.5:1.4,false,true);const dot=svg("circle",{cx:xy[0],cy:xy[1],r:selected?7:4,fill:selected?"#c56214":"#f4c188",stroke:selected?"#7e3c0a":"#995b27","stroke-width":1.3,"data-contact":p.id,role:"button",tabindex:0,"aria-label":`Inspect contact ${p.id}`});const choose=()=>{contact=p.id;renderContacts();draw();};dot.addEventListener("click",choose);dot.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();choose();}});root.append(dot);}
    const caption=svg("text",{x:16,y:22});caption.textContent=`${relevant.length} recorded shapes · ${up===2?"Z":"Y"} up`;root.append(caption);const foot=svg("text",{x:16,y:411});foot.textContent="Grid is a visual reference · click an object or its ID";root.append(foot);
    $("scene-legend").textContent=lab==="rigid"?"Orange: selected velocity direction (normalized glyph)":lab==="contacts"?"Orange: contact normal · glyph length 0.12 scene units":"Axes: X red · Y green · Z blue · parent dashed / child solid · normalized glyphs";
    restoreFocus();
  }
  function render(){
    const restoreFocus=preserveSelectionFocus(),c=cap(),f=frame();$("physics-task").textContent=cfg[lab].task;$("physics-capture").textContent=`${c.id==="contacts"?"contact fixture":c.file} · scene ${c.scene}`;$("physics-frame").max=c.frames.length-1;$("physics-frame").value=sample;$("physics-frame").setAttribute("aria-valuetext",`Recorded frame ${f.index}, sample ${sample+1} of ${c.frames.length}`);$("physics-frame-label").textContent=`Frame ${f.index} / ${c.frameCount-1}`;$("vector-label").textContent=cfg[lab].label;
    const list=$("object-list");list.replaceChildren();selectable().forEach(id=>{const cls=info(id).class.replace("PxArticulationJointReducedCoordinate","Joint").replace("PxArticulationLink","Link").replace("PxRigidDynamic","Dynamic").replace("PxRigidStatic","Static"),b=refButton(id,`${cls} #${id}`);b.setAttribute("aria-pressed",String(id===object));list.append(b);});
    $("physics-provenance").textContent=`${c.file} · SHA-256 ${c.sha256} · ${c.frames.length} sampled frames from ${c.frameCount} · runtime ${window.OVD_PHYSICS_DATA.release}`;
    renderInspector();renderJoints();renderContacts();draw();
    restoreFocus();
  }
  $$('[data-camera]').forEach(b=>b.addEventListener('click',()=>{camera=b.dataset.camera;press('[data-camera]',"camera",camera);draw();}));
  $("attribute-filter").addEventListener("input",renderAttributes);$("physics-frame").addEventListener("input",()=>{stop();sample=Number($("physics-frame").value);render();});
  $("physics-play").addEventListener("click",()=>{if(timer){stop();return;}$("physics-play").textContent="Ⅱ";$("physics-play").setAttribute("aria-pressed","true");$("physics-play").setAttribute("aria-label","Pause sampled frames");timer=setInterval(()=>{sample=(sample+1)%cap().frames.length;render();},450);});
  if(lab==="rigid"){
    for(const [button,id] of [["example-mass",11],["example-contact-offset",10]]){
      document.getElementById(button).addEventListener("click",()=>{
        stop();sample=cap().frames.findIndex(f=>f.index===2);chooseObject(id);
        $("object-title").tabIndex=-1;$("object-title").focus({preventScroll:true});
        $("object-title").scrollIntoView({block:"center"});
      });
    }
  }
  ["show-vectors","show-wire","focus-selection"].forEach(id=>$(id).addEventListener("change",draw));$("inspect-scene")?.addEventListener("click",()=>{chooseObject(cap().scene);$("attribute-filter").value="pairs";renderAttributes();root.querySelector(".raw-attributes").open=true;$("attribute-filter").focus({preventScroll:true});$("attribute-filter").scrollIntoView({block:"nearest"});});
  document.addEventListener("visibilitychange",()=>{if(document.hidden)stop();});
  $("focus-selection").checked=lab==="contacts";
  render();
  }
  document.querySelectorAll('[data-physics-lab]').forEach(root=>initializeLab(root,root.dataset.physicsLab));
})();
