---
name: omniverse-dev-variant-presenter-ui
description: >
  The visual design + layout for the Dev Variant Presenter frontend — the GitHub-dark + NVIDIA-green
  theme, the header / viewport+timeline / right-side tabbed control panel layout, and the component
  styles (pill chips with swatches, segmented buttons, blocks, the NLE timeline strip). Use when
  building or styling the Variant Presenter browser UI — also known as the "variant studio" UI in
  the walkthrough video and earlier releases — so it looks and feels like the product, not
  just functions like it.
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [ui, css, layout, frontend, dev-variant-presenter]
  domain: ai-ml
  languages: [html, css, javascript]
---

# Dev Variant Presenter — visual design + layout

Build the look, not just the behavior. Vanilla HTML/CSS/JS (no framework). This is the target
design; match it closely (colors exact, layout + components as specified).

## ⚠ SERVING CONTRACT — get this right or the entire UI is silently DEAD (do this FIRST)
The frontend is one IIFE. If any `<script>` 404s, NOTHING runs — Open does nothing, no chips, no
stream — and it looks like "the stream won't connect," not a 404. **The base path the HTML references
MUST match where the server actually serves the files.** This is the single most common way the whole
app ships broken (it happened: `<script src="./app.js">` resolves to `/app.js`, but the assets were
mounted at `/web` → every script 404'd → 100% dead UI while every API test passed).

Pick ONE consistent scheme:
- **Root scheme (simplest):** mount `app.mount("/", StaticFiles(directory=WEB, html=True))` and reference
  scripts at root: `<script src="/app.js">`. Register ALL `/api/*` and `/events` routes BEFORE the
  mount so they take precedence; the `/` mount then serves `index.html` and every sibling asset.
- **Prefixed scheme:** mount at `/web` and reference every asset under it: `<script src="/web/app.js">`,
  `<link href="/web/styles.css">`, the fetched WebRTC lib at `/web/<lib>.js`, etc.
Never mix (root-relative `src` + a prefixed mount). After wiring, **`curl` every `<script src>`/`<link href>`
and confirm 200 + a JS/CSS content-type** — see stability-checklist items 11 & 12. Serve `index.html`
with `Cache-Control: no-cache, no-store, must-revalidate`. That alone is the cache story — do NOT add
`?v=N` query strings to the asset tags; they are redundant with it and advertise a revision count.

## Theme tokens (use these EXACT values — define as CSS variables)

```css
:root {
  --bg:#0d1117;      /* app background (near-black, GitHub-dark) */
  --panel:#161b22;   /* header, side panel, timeline strip */
  --line:#30363d;    /* all borders / dividers */
  --txt:#e6edf3;     /* primary text */
  --muted:#8b949e;   /* labels, secondary text, h2 headings */
  --green:#76b900;   /* NVIDIA green — THE accent: active/selected/live, sliders, progress */
  --accent:#1f6feb;  /* blue — timeline variant clips */
}
* { box-sizing:border-box; }
body { margin:0; font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:var(--bg);
       color:var(--txt); height:100vh; display:flex; flex-direction:column; }
```

- **The single accent is NVIDIA green `#76b900`.** Every active/selected/on state (tab, segmented
  button, chip, mode button, timeline transport) is green background with near-black text
  `#04130a`. Hover borders go green (`border-color:var(--green)`).
- Buttons: `background:#21262d; border:1px solid var(--line); border-radius:6px; padding:7px 12px;
  font-weight:600`. Inputs/selects: `background:#0d1117; border:1px solid var(--line);
  border-radius:6px`.
- Status badge (top-right pill, uppercase): default muted; `.live` = green bg / `#04130a`;
  `.warming` = `#fb8500`; `.error` = `#cf222e`.

## Layout (three regions)

```
┌ header ────────────────────────────────────────────────────────────────┐
│ h1 "Dev Variant Presenter" (green) │ [stage path input flex:1] [Open] │ STATUS pill │
├ main (flex row) ─────────────────────────────────────┬─ panel-resize ─┬ aside.panel (330px) ┐
│ .stage-col (flex:1, column)                          │ (6px col-resize)│ .tabs: Configure    │
│   .viewport (flex:1, black; <video> + results-img/video overlay) │       │   Grid Timeline     │
│   below-viewport DOCK (340px): TIMELINE strip ONLY    │                 │   Results           │
│     (Results media overlays the .viewport, not here)  │                 │ .pane (active)      │
│                                                       │                 │ .pane (active)      │
└───────────────────────────────────────────────────────────────────────┴─────────────────────┘
```

- `header`: flex row, `padding:10px 14px`, `background:var(--panel)`, bottom border. `h1`
  `font-size:16px; color:var(--green)`. The stage `<input>` is `flex:1`. Status pill has
  `margin-left:auto`.
- `main`: `flex:1; display:flex`. `.stage-col { flex:1; display:flex; flex-direction:column }`.
  A draggable `.panel-resize` (6px, `cursor:col-resize`, hover green) splits it from the panel.
- `.viewport`: `position:relative; flex:1; background:#000`, centers `#remote-video`
  (`max-width/height:100%`). Overlays live here: an `#overlay` message, a crosshair pick layer,
  an SVG gizmo, a hint chip. **The turntable pivot gizmo is a plain SVG/DOM overlay drawn from
  `/api/project` screen coordinates — do NOT adopt `ovui` (or any native/in-renderer widget kit) for
  it.** The gizmo is client-side UI over the ovstream video, not scene geometry; ovui widgets are for
  an in-viewport native overlay pipeline this app doesn't use. Keep `#gizmo` as SVG.
- `aside.panel`: `width:330px; border-left:1px solid var(--line); background:var(--panel);
  display:flex; flex-direction:column; overflow:hidden`.

## The right panel — 4 tabs

`.tabs` is a flex row of 4 buttons; the active tab is **green TEXT with a green
`border-bottom:2px` on the dark bar — NOT a solid green filled block**. Filling the active
tab with `#76b900` while leaving the label green/white-on-green makes the tab name UNREADABLE.
Whatever the styling, every tab label (active + inactive) must keep text/background contrast ≥ 3:1
(aim 4.5:1) — this is gate-checked on computed styles. Each `.pane` is `display:none` unless `.active`. Inside a pane, content is
grouped into `.block`s (`padding:12px 14px; border-bottom:1px solid var(--line)`) with an `h2`
(`font-size:12px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted)`). A
`.block.grow { flex:1; overflow-y:auto }` holds the scrolling list at the bottom of a pane.

1. **Configure** (live look): blocks — **Camera** (a `<select>` of authored cameras + Save
   framing / Reset camera) · **Render mode** (a `.seg` of Real-Time / Path Tracing) · **Display**
   (read-only stage info + an aspect/resolution `<select>` + range sliders for ISO / focal length
   / f-stop + a focus-distance number with a **Pick** button) · **Turntable** (Pick pivot / Remove
   + a nudge gizmo + frames + Create / Preview spin) · **Variant sets** (`.block.grow`: one
   `.vcard` per set, each a row of `.chip`s).
2. **Grid**: **Matrix mode** (`.seg` One-at-a-time / Full Cartesian) · **Quality & output**
   (`.seg` Real-Time / Path Tracing + W/H/SPP + an Animation-range checkbox + output folder) ·
   **Cameras** — a **COMPACT dropdown** (`#grid-cameras`; a multi-select dropdown or a dropdown button
   opening a checkbox popover, still selecting one OR several cameras) — **NOT a tall per-camera
   checklist**: an ~18-camera stage (ConceptCar) must NOT push **Include sets** off-screen ·
   **Include sets** (`.block.grow`, ticked sets get a green
   border) · an estimate line + Render / Cancel + a progress bar.
3. **Timeline**: **Project** (name + Save/Open/Delete) · **Saved track views** · explanatory help
   text pointing at the strip below the viewport.
4. **Results**: the SIDE PANEL holds only the controls + the list — **Results folder** (+ Refresh) ·
   **Post-process** (`.seg` Overlay labels / Cut sheet) · **Renders** (a `<select>` list of
   stills/sequences/videos; the list is good). **The chosen media takes over the MAIN VIEWPORT PANEL** —
   `#results-img`/`#results-video` are children of the `.viewport` section, overlaying `#remote-video`,
   shown (`.show`) only on the Results tab; selecting a render replaces the live viewport with the
   still/MP4 IN PLACE. It is **NOT** a separate player in a below-viewport dock (there is no
   `#results-dock`). The multi-frame `#results-slider`+`#results-frame-label` control it from the side
   panel. (Earlier guidance wrongly said "below-viewport dock" — align to the reference: results media =
   in the viewport panel, where `#remote-video` lives.)

## Components

- **Pill chip** (`.chip`): `display:inline-flex; align-items:center; font-size:11px; padding:3px
  9px; border-radius:12px; border:1px solid var(--line); background:#161b22; color:var(--muted)`.
  `.on` = green bg / `#04130a` / bold. A `.swatch` dot (11×11, `border-radius:3px`) shows the
  variant's representative color. **Wire these dots up** — the `usd-variant-scan-classify`
  classifier emits per-variant swatch hex (`swatches: {set:{variant:"#rrggbb"}}` on the
  `classified` event); set each chip's `.swatch` background from it (hide the dot only when no
  color was detected). Chips without swatch dots are an incomplete look.
- **Segmented control** (`.seg`): a flex row of equal buttons (`flex:1`); the active one is green.
- **Range sliders**: `accent-color:var(--green)`. **Progress bar**: a `.bar` filled green.
- **Section info popovers**: a small circled "i" (`.info`) and `data-help` attributes that show a
  fixed popover bordered green — optional but on-brand.

## The below-viewport dock is TAB-CONTEXTUAL — and it is the TIMELINE's, only
The strip area under the viewport (`#timeline-strip`) is NOT a permanent fixture:
- **Timeline tab active →** the NLE timeline strip shows there.
- **Configure / Grid / Results →** the dock is hidden; the viewport uses full height.
- **Results media does NOT go in this dock.** The selected still/MP4 OVERLAYS the MAIN VIEWPORT
  (`#results-img`/`#results-video` inside `.viewport`, over `#remote-video`, `.show` on the Results tab) —
  it replaces the live viewport in place. There is no separate `#results-dock` below the viewport.
Drive it off the active tab: `#timeline-strip` is shown only while the Timeline tab is `.active` (e.g.
toggle a class on switchTab), hidden on every other tab even with a stage open. (Leaving the
timeline strip visible on ALL tabs whenever a stage is open is the bug to avoid.)

## Timeline strip (the NLE — shown ONLY on the Timeline tab)

`#timeline-strip { height:340px; border-top:1px solid var(--line); background:var(--panel) }`,
shown **only when the Timeline tab is active** (hidden on Configure/Grid/Results even with a stage open),
top edge drag-resizable. Three parts:
- a **toolbar** (`.tl-toolbar`, wraps): playhead time (green, tabular-nums), length, append-mode
  toggle (Stack / At playhead), presets (Slideshow / Mixer / Clear), new-clip duration + fps,
  and a right-aligned render group (output folder + **Render to MP4** in green + a progress bar).
- a **ruler** row with a transport gutter on the left (to-start / step-back / play / step-fwd /
  to-end / loop, ~22px square buttons) and tick marks.
- **tracks**: one row per variant set + a camera track; each row has a sticky left label
  (`var(--label-w,230px)`) and absolutely-positioned **clips**. Variant clips are blue
  (`var(--accent)`), camera clips purple (`#8957e5`), the selected clip has a green outline. A
  green 2px **playhead** line overlays the tracks. Each clip carries a `▾` to change its variant (and a
  tiny COLORED swatch ONLY when that variant has a detected swatch color — never a placeholder white box);
  drag to move, drag the right edge to resize. Any edit re-applies the state at the playhead immediately.

## Layout: the viewport shrinks for the dock ONLY when the dock is showing (flex sizing — common bug)

When the dock IS showing (the Timeline strip, on the Timeline tab), the viewport must SHRINK to make room
at any window height and any stream resolution — otherwise a 1080p stream makes the viewport tall and
pushes the 340px strip below the fold. When the dock is hidden (Configure/Grid/Results), the viewport uses
full height. Results media overlays the viewport (absolute, inside `.viewport`) — it does NOT take dock
space. Required flex rules:
```css
.stage-col { flex:1; display:flex; flex-direction:column; min-height:0; overflow:hidden; }  /* min-height:0 + overflow:hidden are both essential */
.viewport  { flex:1; min-height:0; overflow:hidden; position:relative; }   /* shrinks; never min-content */
#remote-video { max-width:100%; max-height:100%; }                         /* letterboxed, never forces height */
#results-img, #results-video { position:absolute; inset:0; }               /* overlay the viewport on Results */
#timeline-strip { flex:none; height:340px; }                               /* the dock = TIMELINE only */
```
`min-height:0` alone is not enough: a flex child's intrinsic content size (a raw `<video>`/`<img>` at its
native 1920×1080) can still overflow and push `#tl-resize`'s drag handle (and the whole strip) off-screen
or force page scroll — `overflow:hidden` on BOTH `.stage-col` and `.viewport` is what actually clips the
oversized intrinsic size so the flexbox shrink takes effect. Verify at a normal window height (~800–900px):
on the Timeline tab the strip (and its `#tl-resize` top-edge handle) is fully visible and draggable without
scrolling, even with a 1920×1080 stream; on the Results tab the video + its scrubber are; on Configure/Grid
the viewport is full-height with no strip. (Make the dock height user-resizable via `#tl-resize`, but it
must never be clipped or push the page into scroll.)

## Aspect / resolution must offer real ASPECT changes (not just 16:9 sizes)

The Display "Aspect / resolution" `<select>` must include genuinely different **aspect ratios**,
matching production — not three 16:9 resolutions. Provide at least:
`1280×720` (16:9) · `1920×1080` (16:9) · `1080×1080` (1:1) · `1080×1920` (9:16) ·
`1440×1080` (4:3) · `1920×804` (2.39:1). Changing it rebuilds the ovstream `Server` at the new
size (brief reconnect via the stream-reconnect path) AND updates the camera's
`verticalAperture = horizontalAperture * h/w` so the framing isn't stretched; the Grid tab
inherits the current size as its render default. Verify switching to 1:1 / 9:16 actually changes
the streamed frame's shape, not just its pixel count.

## Pick flows must be wired end-to-end (backend working ≠ feature working)

- **Focus picker:** arm the Pick button → show the crosshair overlay over the video → on click,
  read normalized coords from the `<video>` rect → `POST /api/pick-focus {nx,ny}` → **apply the
  returned `focus_distance`** to the focus field AND the live camera (this last step is the one
  that gets dropped — the server returns a distance but the UI never uses it, so "focus picker
  does nothing"). Disarm + remove the crosshair after.
- **Turntable pivot:** same arm→click pattern but `POST /api/pick-point {nx,ny}` → drop the pivot
  gizmo at the returned `world` point. (pick-focus returns a distance, not a point — using it for
  the pivot is wrong.)

## Verify the look
Open the app and confirm: green `#76b900` accent on the active tab / selected chip / LIVE badge;
the dark `#0d1117` background; the header with the title + stage field + Open + status pill; the
330px right panel with the four named tabs; variant chips with swatch dots. Switch to the **Timeline**
tab → the NLE strip appears below the viewport (green playhead, blue/purple clips); switch to
**Configure/Grid** → the strip is GONE and the viewport is full height; switch to **Results**, pick an
MP4 → it plays IN THE VIEWPORT AREA, overlaying the stream, with its scrubber/transport below the
video (not in the side panel). It should be recognizably the same product, not a generic form over the stream.
