---
name: usd-variant-live-switching
description: >
  Switch USD variants live in an ovrtx viewport WITHOUT reloading the stage when possible —
  replaying shader-input attribute writes for the "fast path", and falling back to a
  heartbeated composite reopen for transform/visibility/binding/structural variants. Use
  when wiring variant chips to a live RTX stream (the configurator experience).
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, variants, ovrtx, live]
  domain: ai-ml
  languages: [python]
---

# Live variant switching

Depends on `usd-variant-scan-classify` for the per-set classification + the union-absolute
write table. Runs on the **render thread only** (the sole ovrtx owner).

## Decide: fast path vs reload

**Extract the variant from each selection ROBUSTLY** — selections arrive as JSON dicts (`{prim_path, set_name, variant}`), so use an isinstance-aware extractor (`s["variant"] if isinstance(s, dict) else getattr(s, "variant", None)`), NOT `getattr(s, "variant") or s["variant"]` (that `getattr` raises `AttributeError` on a dict before the `or` can fall through). And **skip any selection whose variant is empty/None** rather than authoring it — an empty-variant entry otherwise crashes the apply. (Both bite on the very first `/api/variant` or a project-restore with a partial selection.)

Given the new selection, diff it against the live selection to find the CHANGED sets. Then:

- **Fast path** — taken only if EVERY changed set is classified `shader_input`. Replay the
  recorded absolute writes for each changed set's target variant through the app-owned
  `ovstage.Stage` (via a `StageSession` coordinator), NEVER the deprecated
  `Renderer.write_attribute`:
  ```python
  ordinal = session.next_ordinal()
  queries = []
  try:
      for prim_path, writes in by_prim.items():           # one query per distinct prim
          query = session.query_from_paths([prim_path])   # ovstage.PathDictionary(stage) under the hood
          queries.append(query)
          for attr, value in writes:
              session.write_attribute(query, attr, _tensor(value),
                                      is_array=False, ordinal=ordinal, advance=False)
  finally:
      for q in queries:
          session._release_query(q)
  session.advance(ordinal)     # one shared ordinal + one advance_write_floor for the whole switch
  renderer.reset()              # re-converge PT/RT2 accumulation after the writes
  ```
  No `populate_usd`, no lock. A `color3f` writes fine with the DEFAULT semantic — there is no
  special color Semantic and you do not need one. Sub-second; the stream never stalls.

  **Tensor shape for shader inputs:** a VECTOR input is ONE MULTI-LANE element, and a plain numpy
  dtype cannot express that — you must declare the layout. `color3f`/`float3` → `lanes=3`,
  `float2`/`Vec2f` → `lanes=2`, `float4` → `lanes=4`, always `shape=[1]`, because
  `is_array=False` means you are writing one element. A scalar `float` is the one case a bare
  numpy array covers: `(1,)`.
  ```python
  def _value_to_tensor(v):
      if isinstance(v, bool):         return None          # unsupported
      if isinstance(v, (int, float)): return np.array([float(v)], dtype=np.float32)
      try:    comps = [float(x) for x in v]
      except (TypeError, ValueError): return None
      if not comps: return None
      from ovstage import DLDataType, DLDataTypeCode, make_dltensor
      return make_dltensor(np.array(comps, dtype=np.float32),
                           dtype=DLDataType(code=DLDataTypeCode.kDLFloat, bits=32,
                                            lanes=len(comps)),
                           shape=[1])
  ```
  **‼ Passing a bare `(1,3)` array is the ovrtx-0.3-era form and it FAILS on ovstage 0.1.** It is
  read as `lanes=1, bytesPerRow=12` and rejected against the existing 3-lane Fabric column:
  `write_attribute: existing attribute 'inputs:diffuse_reflection_color' has a different type`.
  The reload fallback then renders the CORRECT frame, so nothing looks broken — while every
  shader-input switch silently costs a full recompose instead of a live write. Measured on the
  ConceptCar with a live client: **fast path 68-76 ms, reload path ~1450 ms**, and a reload also
  delays the NEXT command by up to ~3 s while its first heavy frame lands.

  `omni:xform` uses the same mechanism (16 lanes, `shape=[1]`, plus `semantic=MATRIX`) — see the
  turntable skill. The only differences are the lane count and the semantic, not the technique.

  **‼ `write_attribute` returns an async `Operation` — you MUST wait on it.** If you don't, the
  write can be a **silent no-op**: nothing errors, nothing changes on screen, and the numpy
  tensor's keepalive can die before the write lands. That reproduces the "silent fast switch"
  regression below via a root cause you will not find by staring at the selection logic. Wait
  inside your `StageSession.write_attribute` wrapper so no caller can forget.

- **Reload path** — if any changed set is transform/visibility/binding/structural. Rebuild
  the composite with the **full current selection** applied and populate it via
  `ovstage.population.open_usd` (the `StageSession.populate_usd` helper), under
  `USD_LOCK` and **heartbeated** (see below). ~1–2 s.

Wrap the fast-path writes in try/except → on any failure, fall back to reload. Switches
that arrive before classification finishes always reload, then flip to fast once classified.

## ‼ The fast path must be IMMEDIATELY VISIBLE — the "silent fast switch" regression
The failure mode: clicking Carpaint changes NOTHING on screen until the user forces an unrelated
reopen (switching a Backdrop, previewing the turntable) — then the paint "catches up." The same thing
shows up in the timeline: geometry/environment (reload) clips update the viewport, material (fast)
clips don't. Root causes to rule out, in order:
1. The writes edit **pxr/USD state only** and never reach the renderer — the live ovrtx stage
   doesn't watch your composite; you MUST write through `ovstage.Stage.write_attribute(...)` (via a
   `StageSession`/`ovstage.PathDictionary(stage)` query) the shader inputs — never the deprecated
   `Renderer.write_attribute`, and never `stage.get_path_dictionary()`'s raw bundle for the query.
2. The writes land but **`renderer.reset()` is missing** — accumulation keeps converging on the
   OLD look; the change surfaces only at the next reopen/reset.
3. The self-test only exercised the **pre-classified window** (everything reloads → visible), so
   the broken post-`classified` fast path was never seen. Test a fast set TWICE, after the
   `classified` event, and require a pixel change within ~2 s each time.
The timeline is the same endpoint (`/api/variant`) — if chips are broken this way, scrub/play/edit
material updates are broken identically.

## ‼ The classifier must be TIME-BOUND — it silently hung forever on an 11 GB mirrored stage
Classification that takes ~12 s on a local stage can grind indefinitely on a mirrored S3 stage
(huge layer stack, junction paths, hundreds of MDLs). A build's background classifier hung with no
event, no error, no log line — so the fast path never engaged and every switch stayed a slow reload
that users read as "variants don't work." Rules: classify per-set with a time budget (a set that
blows the budget is classified reload-only and skipped); ALWAYS emit `classified` within the
deadline (~120 s), even with reduced/empty `fast_sets`; emit an `error` event on trouble; log
progress so a hang is diagnosable. Verify classification completes on the MIRRORED stage, not just
the local one.

## A RELOAD MUST REPRODUCE THE FULL CURRENT LOOK (the #1 correctness trap)

`ovstage.population.open_usd` **discards all fabric attribute writes** — so every prior fast-path
change (e.g. a paint color the user picked) is LOST on a reopen unless you restore it. A reload
that only rebuilds the composite + populates it will silently REVERT earlier fast switches:
the user picks Carpaint=Blanco (fast, looks right), then switches Doors (reload) and the
paint snaps back to default. This passed a shallow grader and is exactly the "setting a
variant doesn't work" bug. Two things are BOTH required:

1. **The composite must carry the full selection AND it must actually compose.** Authoring a
   `variantSelection` in the composite *root* layer over a prim defined in a sublayer does
   NOT always win/compose as expected. VERIFY it: after building, open the composite in pxr
   and assert each prim's `GetVariantSets().GetVariantSet(set).GetVariantSelection()` equals
   what you intended. If selections aren't composing, author them in a session sublayer that
   is stronger, or author the variant on the prim spec directly — don't trust a silent
   `SetVariantSelection` in a try/except.
2. **Re-apply ALL fast-path shader overrides AFTER the reopen (`populate_usd`).** For every
   shader-input (fast) set in the current selection, replay its absolute writes via
   `ovstage.Stage.write_attribute` once the reopen completes (and `renderer.reset()`), so a
   paint/leather/trim color the user chose survives the reopen. Don't rely on the variant
   selection alone if (1) can't be guaranteed.

**Preserve the live camera across a reload.** Build the composite with the live viewer pose
and write that pose at warmup START (before the first surfaced frame) — otherwise the
authored camera flashes into view for a few frames and "snaps back", which reads as a bug.

## ENVIRONMENT / lighting / Backdrops variants MUST visibly re-render (a hard case)

A "Backdrops" (environment) set swaps dome lights / sky HDRs and is reload-class. Switching it
MUST change the rendered sky + lighting — verify the **whole-frame** pixels change, not the
selection. Two traps make it silently no-op:
1. **The selection may not compose on a deep/nested prim.** A Backdrops set often lives on a deep
   prim (e.g. `/World/Backdrops/Lighting/Environment/Lighting_SETS`), possibly behind a
   reference/payload. Car-paint selections can compose while this one silently doesn't. After
   building the composite, OPEN it in pxr and assert
   `GetVariantSet("Backdrops").GetVariantSelection()` equals the target on the actual authoring
   prim; if it didn't take, author it where it composes (the prim that owns the variantSet), not a
   parent `over`.
2. **The renderer can cache the old environment/dome across a reopen.** If the selection composes
   but the sky doesn't change, force a clean re-open of the new composite (and `reset()` +
   warm-up) so ovrtx re-evaluates the dome lights — don't reuse cached lighting.
Also ensure the dome HDR texture actually resolves (prefer the already-local copy; see
`usd-remote-stage-mirror`). **Verify:** switch Backdrops → Pacific_Highway / Bay_Bridge and
confirm the whole-frame mean color shifts (sky/lighting changes), not delta≈0.

## VERIFY end-to-end (this is what catches the revert + must be in the test suite)
- Switch a FAST set (paint) to a visibly distinct color → the car changes in < ~2 s, no reopen.
- Then switch a RELOAD set (Doors / Wheel_Turns) → **the paint color from the previous step
  MUST persist** AND the reload variant must visibly change. Sample the car-region pixels
  before/after and assert the fast change survived.
- The live camera pose must not jump during the reload.
Do this against the real stage (ConceptCar), reading rendered pixels — a `{ "ok": true }`
HTTP response is NOT proof a variant applied.

## Heartbeat the reload (do not stall the stream)

A reopen runs synchronously on the render thread and sends no frames meanwhile; a heavy
reload (e.g. a lighting/Backdrops set) plus a lock wait easily exceeds ovstream's ~7 s
liveness → the client drops. Run the blocking reopen while a short-lived daemon thread
re-streams the last good frame:

```python
def with_usd_heartbeat(self, fn):
    stop = threading.Event()
    def pump():
        while not stop.is_set():
            try: self.streamer.submit_last()   # re-stream the cached BGRA buffer
            except Exception: pass
            stop.wait(2.0)
    t = threading.Thread(target=pump, daemon=True); t.start()
    try:
        return fn()                 # build_composite + populate_usd (ovstage.population.open_usd), under USD_LOCK
    finally:
        stop.set(); t.join()        # join BEFORE the render loop resumes stream_video
```

`submit_last()` just re-streams the cached buffer, so it is safe to call from the pump
thread while the render thread is blocked inside `populate_usd`. Join the pump before the loop
resumes so `stream_video` is never called from two threads at once.

## Wiring (render thread)

- On open: reset the action table; spawn the async classifier; default the live selection
  to the stage's current selections.
- On a `SetSelection` command: diff vs live; fast path if all changed sets are
  shader-input-classified, else reload-fallback. Coalesce a burst of switches in the
  command drain (apply the net selection once).
- Keep the live selection mirrored so the next diff is correct and so a project save / reopen
  can restore it.

## Verify
A fast switch shifts pixels toward the target color in < ~2 s with no `populate_usd` in the
logs; a reload switch changes the stage and the stream survives (heartbeat frames during the
reopen). Confirm with a GPU spike: render before/after and check the mean color moved.
