// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
"use strict";
(() => {
  const data = window.OVD_WORKFLOW_DATA;
  const $ = id => document.getElementById(id);
  function row(name, value) {
    const group = document.createElement("div");
    const term = document.createElement("dt"), result = document.createElement("dd");
    term.textContent = name; result.textContent = value;
    group.append(term, result);
    return group;
  }
  function renderTcp() {
    const state = data.tcp.states[Number($("tcp-stage").value)];
    $("tcp-stage-label").textContent = state.label;
    $("tcp-stage").setAttribute("aria-valuetext", state.label);
    $("tcp-box").style.visibility = state.boxPresent ? "visible" : "hidden";
    $("tcp-ground").style.visibility = state.groundPresent ? "visible" : "hidden";
    $("tcp-scene-note").textContent = state.boxPresent ? "Last recorded values in this segment" : "Catalog only: no recorded geometry or motion";
    $("tcp-scene-title").textContent = `${state.label}: box ${state.boxPresent ? "recorded" : "not recorded"}, ground ${state.groundPresent ? "recorded" : "not recorded"}`;
    $("tcp-readout").replaceChildren(
      row("Catalog", data.tcp.catalog.map(r => r.name).join(" · ")),
      row("Request / capture epoch", state.requestId === null ? "No selection request" : `${state.requestId} / ${state.captureEpoch}`),
      row("Acknowledgement", state.ackStatus),
      row("Box position Y · last write", state.lastPose === null ? "No motion payload" : String(state.lastPose[5])),
      row("Box linearVelocity · last write", state.lastVelocity === null ? "No motion payload" : JSON.stringify(state.lastVelocity)),
      row("Box mass", state.mass === null ? "No recorded write in this segment" : String(state.mass[0])),
      row("Ground", state.groundPresent ? "PxRigidStatic recorded" : "Not recorded in this segment")
    );
  }
  function renderContactOffset() {
    const gap = Number($("offset-gap").value);
    const shapes = data.contactOffsets.shapes;
    const contactDistance = shapes.reduce((sum, shape) => sum + shape.contactOffset, 0);
    const restDistance = shapes.reduce((sum, shape) => sum + shape.restOffset, 0);
    const inside = gap < contactDistance;
    const label = gap.toFixed(3);
    const bottom = 230 - gap * 1600;
    $("offset-gap-value").textContent = label;
    $("offset-gap").setAttribute("aria-valuetext", `${label} scene units`);
    $("offset-box").setAttribute("transform", `translate(0 ${bottom - 60})`);
    $("offset-gap-line").setAttribute("d", `M344 ${bottom} H356 M350 ${bottom} V230 M344 230 H356`);
    $("offset-gap-label").setAttribute("y", String((bottom + 230) / 2 + 5));
    $("offset-gap-label").textContent = label;
    $("offset-diagram").dataset.contact = String(inside);
    const state = gap <= restDistance ? "At the target rest distance: surfaces touching." : inside ? "Within contact distance: contacts begin before the surfaces touch." : "Outside contact distance: no contacts from this margin yet.";
    $("offset-state").textContent = state;
    $("offset-diagram-title").textContent = `Illustrated gap ${label}. ${state}`;
  }
  $("tcp-stage").addEventListener("input", renderTcp);
  $("offset-gap").addEventListener("input", renderContactOffset);
  renderTcp(); renderContactOffset();
})();
