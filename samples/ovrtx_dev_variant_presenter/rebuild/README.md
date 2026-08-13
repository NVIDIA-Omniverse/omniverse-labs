# Dev Variant Presenter — releasable skills + prompt + acceptance gate

This bundle lets an agent **single-shot build** Dev Variant Presenter (a browser/WebRTC,
RTX-streamed USD variant-permutation presenter) on top of NVIDIA's `omniverse-realtime-viewer`
seed skill — at **production cadence**, not just a happy path.

## Contents
- `PROMPT.md` — the build prompt (hand to a capable long-runway coding agent). It leads with the
  requirement ledger — 18 subtle behaviours stated as the user experiences them when they are
  wrong — then the executable done-gate contract.
- `acceptance/` — the executable definition of DONE:
  - `grade_http.py` — the control-plane grader (endpoints, architecture gates, file outputs,
    frontend-asset-404 guard). Done = every area `full` except `area2`/`area3` (inherently HTTP-only,
    allowed `partial`), and zero `fail`.
  - `verify_browser.cjs` — a headful-Chrome verifier that drives a REAL mouse/keyboard through
    every user-facing flow (live WebRTC pixels, orbit/pan/dolly, variant chips, optics sliders,
    pick gizmo, timeline editing + transport, grid batch, MP4 playback, projects, attach/reload).
    Run it in **two lanes**: the local stage AND the remote S3 stage URL (the remote lane catches
    mirror/classifier/attach defects invisible locally). Done = `0 failed` in both.
  - `remote_mirror_probe.py` — standalone full-closure mirror probe for remote stages.
- `skills/` — 10 skills (the domain layer the seed skill lacks):
  - `omniverse-dev-variant-presenter` — orchestrator: architecture, the binding contracts
    (`references/SPEC-FUNCTIONAL.md` + `SPEC-UX.md`), the stability checklist (the hard-won
    "black viewport / stream lost / crash / DOA-frontend" fixes), a production-parity checklist,
    and a seed recap.
  - `omniverse-dev-variant-presenter-ui` — visual design + layout + the static-serving contract.
  - `usd-variant-scan-classify`, `usd-variant-live-switching` — variant scan +
    fast-path-vs-reload classification + live switching.
  - `ovrtx-grid-batch-render`, `ovrtx-timeline-nle`, `ovrtx-turntable-camera` — batch stills +
    explosion guard, multi-track timeline → H.264 MP4, the interactive orbit/turntable rig +
    pick gizmo.
  - `ovrtx-projects-session`, `usd-remote-stage-mirror`, `ovrtx-post-cutsheet` — projects +
    crash recovery, remote `http(s)` stage mirroring (full closure incl. MDL-internal deps),
    results labeling + contact sheet.

## Which skill for which task
The skills are also usable one at a time, outside a full build — hand a single `SKILL.md` to an agent
along with the task ("implement a timeline view on ovrtx") and it carries the recipes and traps for
that area on its own.

| If the task is… | Read this skill |
| --- | --- |
| A multi-track timeline / NLE strip: clips, tracks, scrub, transport, MP4 render | `ovrtx-timeline-nle` |
| An orbit/turntable camera rig, pivot picking, the axis gizmo, spin preview | `ovrtx-turntable-camera` |
| Batch stills over a permutation grid, the explosion guard, progress + cancel | `ovrtx-grid-batch-render` |
| Finding a stage's variant sets and deciding fast-path vs reload per set | `usd-variant-scan-classify` |
| Switching a variant LIVE without a reopen (fabric writes on shader inputs) | `usd-variant-live-switching` |
| Opening an `http(s)`/S3 stage by mirroring its full dependency closure locally | `usd-remote-stage-mirror` |
| Saving/restoring projects, named track views, crash-recovery checkpointing | `ovrtx-projects-session` |
| Browsing render results, labelling stills, building a contact sheet | `ovrtx-post-cutsheet` |
| Frontend layout, styling, the tab-contextual dock, the static-serving contract | `omniverse-dev-variant-presenter-ui` |
| Overall architecture, the render-thread ownership rules, the binding specs | `omniverse-dev-variant-presenter` (start here) |

**Debugging ovrtx / ovstream itself?** Go straight to
`skills/omniverse-dev-variant-presenter/references/stability-checklist.md`. It is the
symptom → cause → fix → diagnostic reference for the traps that cost the most time in this stack:
black viewport, "stream lost" / reconnect storms, native crashes on teardown, DOA frontends, a stage
that won't recompose, the idle-GPU throttle, `ovstream.initialize()` ordering, and more. It is worth
reading before you write any ovrtx code, and worth re-reading the moment something goes silently
wrong. The sibling `references/SPEC-FUNCTIONAL.md` + `SPEC-UX.md` are the binding server and browser
contracts; `references/seed-recap.md` recaps the upstream seed skill.

## Prerequisites
Windows 11 + an NVIDIA RTX GPU; `uv`; Python `>=3.11,<3.12` (ovrtx ships no cp312+ distribution);
PyPI access for `ovrtx>=0.4,<0.5` + `ovstage>=0.1,<0.2` + `ovstream>=0.4,<0.5` — all three resolve
from PyPI, so no extra index needs configuring, but installing also needs network access to
**`pypi.nvidia.com`**, because the `ovrtx` PyPI package is only a `wheel-stub` sdist whose build
backend downloads the real platform wheel from NVIDIA's index (`ovstage` / `ovstream` publish real
wheels to PyPI directly); Node.js + headful Chrome for the browser verifier. The seed
skill `omniverse-realtime-viewer` (https://github.com/NVIDIA/skills — which in turn references the
OVRTX/ovstage/ovstream skills/SDK docs). The app owns an `ovstage.Stage` and attaches the `ovrtx`
renderer to it (`Renderer.attach_ovstage`) — `ovstage` is a required dependency, not optional.

## How to use
1. Give the builder agent `PROMPT.md` verbatim, with `skills/` and `acceptance/` reachable at the
   paths the prompt names. The prompt is self-contained: spec first, skills second, gate always.
2. **Budget for a large one-time download.** No stage ships in this bundle. Before either gate can
   run, the builder must start its own server and `Open` the public ConceptCar URL once, letting the
   app mirror the stage locally — **~11 GB, many minutes**. That mirror is the local-stage argument
   both gates take; later opens hit the cache and are fast. (The build environment itself is another
   several GB on top.)
3. The builder iterates against the gate itself; when it reports done, **re-run both gates
   independently on a fresh server** — never accept a build's self-report.
4. Done = `grade_http` reports zero `fail` and no `partial` outside `area2`/`area3`, **and**
   `verify_browser` reports `0 failed` in BOTH lanes (local + S3).

## Provenance
These materials were refined empirically: the app was repeatedly rebuilt from nothing but this
prompt, these skills, and the upstream seed skill — never the reference source — and each result was
compared against a known-good reference implementation. Whatever the rebuild got wrong was folded
back into the spec, a skill recipe, a gate check, or the requirement ledger, and the cycle repeated.
Both gates are keyed on the HTTP contract, the DOM-id contract, and rendered pixels rather than on
any internal detail, so they validate any compliant build rather than one codebase.
