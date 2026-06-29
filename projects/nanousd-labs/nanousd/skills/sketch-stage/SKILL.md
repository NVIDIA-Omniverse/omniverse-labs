---
name: sketch-stage
description: Use to generate or densify an OpenUSD stage from sketch-stage inputs (bounds / scale / densification mode / asset pack / intent), with the LLM driving placement one object at a time through collision-gated spatial-query tools while the user watches a live 2D+3D browser viz. Trigger phrases include "generate a USD stage", "densify a scene", "fill an area with assets from <pack>", "scene generation with LLM", "watch placement live", "incremental placement", "absorb a USD as a sketch template".
---

# Sketch Incremental Placement (LLM-driven scene generation)

Sketch-stage generates or densifies OpenUSD scenes from bounds / scale / pack / intent inputs. The **engine is the LLM**: instead of computing all positions up front from one fixed ruleset, the LLM queries the stage and places objects through a tool surface, with a collision gate that rejects overlap and a live 2D+3D viz the user watches.

Its USD reading and writing currently go through **OpenUSD (pxr / `usd-core`)** — it doesn't use `nanousd`. Its **output** — large prim-count stages — is used as **test/benchmark input for `nanousd`**.

## Inputs

| Field | Default | Notes |
|---|---|---|
| `bounds` | — | **Empty-canvas mode.** `WxD` or `WxDxH` in meters (e.g. `10000x10000`). Synthesizes an anchor with no placements — just a footprint shell of the requested size. The LLM places everything guided by `intent` + the pack's archetype semantics. Requires `--pack`. This is the recommended starting point for any themed generation. |
| `scale` | `medium` | `small` (+30), `medium` (+120), `large` (+600), `huge` (+2400) — count of NEW placements the LLM aims for on top of the anchor |
| `intent` | — | Free-form NL goal: e.g. `"<your theme description in your own words>"` — drives placement decisions |
| `out` | `~/.cache/sketch-stage/demo` | Output dir; `events.jsonl`, snapshots, and realized USD live here. Pass `--out` to override. |
| `mode` | `auto` | Densification mode after all template placements are bound: `auto` / `fill` / `tile` / `passthrough`. See "Densification modes" below. |
| `absorb-source` | — | **Optional.** Local path or `http(s)://` URL to a USD. The skill runs `absorb_pack.py` on it first and uses the result as the anchor. **The source must be SimReady-tagged** — i.e. its ref/payload-bearing prims must carry one of: `customData["simReady"]`, `UsdSemantics` labels, or SimReady `Kind` markers (`component` / `assembly` / `group` / `subcomponent`). The absorber refuses any source where < 50% of ref/payload prims carry such metadata; without SimReady info the LLM has no reliable way to bind absorbed placements to a target pack downstream. The absorbed sketch captures **layout + placement semantics only** — archetype labels come from the source itself (asset filename stem, falling back to prim name), slot dimensions from the prim's measured world bbox, `sourcePath` preserved for hierarchy analysis. Structural prims (footprint > 30 m², typically walls/floors/ceilings/facade panels) are filtered out and used only to size the shell. **No original asset URL is carried forward** — binding to actual pack assets is a separate LLM-driven step. |
| `absorb-pack` | — | Optional asset pack tag recorded in the absorbed sketch's metadata so realize knows which pack to consult for binding. Does NOT influence what the absorber captures. |
| `resume` | — | **Optional.** Path to a prior run's `_snapshot.sketch.json` to pick up where the previous session left off. See "Resume from snapshot". |

### Densification modes

After every template placement is bound, the LLM has three ways to grow the scene up to `addTarget` / target prim count:

- `fill` — fill the empty space inside the existing template footprint with collision-checked, semantics-aware free placements. Footprint stays the same; density increases. Use when the user wants the same scene "fuller".
- `tile` — replicate the entire template (anchor placements + bound assets) on a non-overlapping grid. Footprint grows; layout repeats. Use for prim-count-target benchmarks where you want N copies of the same room.
- `passthrough` — bypass per-placement realize entirely; tile the pack's `rootStage` USD as a single unit. Use only when the pack declares `rootStage` in its `pack.json`.
- `auto` (default) — pick `fill` for `small`/`medium` scales; pick `tile` (or `passthrough` if the pack supports it) for `large`/`huge`. Prefer `fill` when the user's intent is qualitative ("denser", "fuller"), `tile` when quantitative ("hit 100k prims").

Trigger phrases for `fill`: *"fill the empty space"*, *"densify without changing the footprint"*, *"more objects in the same room"*, *"don't tile, just fill"*.

Kit is not part of generation. Use Kit only after realization to open, inspect, collect, or validate the generated USD.

---

## Three-tier workflow: LLM authors STRATEGY, scripts EXECUTE, LLM oversees

This is the most important discipline in the skill.

Pure LLM-controlled placement does not scale: 10,000 placements × ~200 tokens × $0.005/k ≈ $10 per scene and ~1.4 hours of inference time. Pure deterministic placement doesn't capture intent: a grid walker can't know that *"archetype A in aisles, not next to archetype B"* means anything until the rule is encoded in the pack's semantics.

The boundary that works is a three-tier hybrid:

| Tier | Who | What | Frequency |
|---|---|---|---|
| **1. Strategy** | LLM | Parse user prompt → decompose canvas into regions → per-region archetype mix → write `preferredNear`/`avoidNear`/`placementBias` into `pack.semantics` → pick the right script for each pass | once per task |
| **2. Execution** | Scripts | Per-cell coord math, weighted random archetype choice, semantic gate (`avoidNear` reject, `preferredNear` cluster), `placementBias` cell weighting, sibling collision, on-surface localXY sampling | thousands of times per task |
| **3. Oversight** | LLM | Look at realized output (or `query_stage_graph` summary), decide if another pass / different region / specific one-off placement is needed | once after each pass |

What the LLM authors during a task:
- **Plan JSON** — `random_fill --plan '{...}'`, `densify_zones --zones '{...}'`. Strategic input.
- **Pack-semantic edits** — small `pack.json` patches (preferredNear, avoidNear, placementBias) using `Edit` on the existing file. Once, not per placement.
- **Inline `place_many` JSON via curl** — for anomalies that don't fit a tool (one-off placement at a specific spot, manual correction). Use when fewer than ~30 placements need bespoke logic.

What the LLM **must not** do:
- ❌ Write Python files to `/tmp/<run>/something.py`. If you find yourself authoring a script to compute placements, stop. Chain existing scripts or POST inline `place_many` for the bespoke part.
- ❌ Reinvent grid math — `densify_zones.py` walks deterministic row-major grids over any region.
- ❌ Reinvent weighted scatter — `random_fill.py` shuffles cells, picks archetype by weight, applies the semantic gate.
- ❌ Reinvent on-surface placement — `surface_fill.py` iterates the runtime surface registry and posts `place_on` per slot.

### Pick the right script per pass

| Pattern you need | Script | Determinism |
|---|---|---|
| Regular grids (rack rows, parking spaces, server pods) | `densify_zones.py` | fully deterministic — row-major, no RNG |
| Irregular scatter, density-fill, "fill the empty space" | `random_fill.py --semantic-gate` | reproducible given `--seed` |
| On-surface stacking (boxes on shelves, cups on counter, monitors on desks) | `surface_fill.py` | reproducible given `--seed` |
| One-off placements, specific landmarks at known coordinates | inline `place_many` JSON via curl | n/a |
| Reusable pattern that doesn't fit any of the above AND likely to recur | NEW script committed to `skills/sketch-stage/scripts/` — never `/tmp` | up to author |

When in doubt: chain existing scripts. If a chain of 4-5 CLI invocations + one inline curl does the job, that's the right shape. **Compose from primitives** — the skill ships no domain-specific recipes; the workflow is the recipe.

**`session_http` API schema** lives in `references/session_api.md`. Read it BEFORE writing any inline Python/`curl` against the live session — the recurring failure mode is calling `.get(...)` on an endpoint that returns a list (≈ half of them do) or `len(...)` on a count that's an integer.

---

## Per-run flow

### Step 0 (one-time): provision the venv

The skill installs `usd-core`, `omniverse-asset-validator`, `mcp`, and `numpy` into a per-skill venv at `~/.cache/sketch-stage/venv`. `scripts/run.sh` calls `scripts/setup_venv.sh` automatically on every invocation; the setup script is idempotent.

```bash
bash $SKILL_DIR/scripts/setup_venv.sh   # explicit invocation if you want to pre-warm it
```

Override the location by exporting `SKETCH_INCREMENTAL_VENV` before the call.

### Step 1: launch session and viz

```bash
OUT="$PWD/sketch_runs/my_run"
bash $SKILL_DIR/scripts/run.sh \
    --bounds 200x150 --pack /path/to/asset_pack \
    --scale medium --intent "<your theme>" --out "$OUT"
```

The script prints a JSON config with: chosen anchor + pack, target additional placement count, session HTTP URL (`http://127.0.0.1:<session_port>/`), live viz URL (`http://localhost:<viz_port>/`), and the available tool names. Open the viz URL in a browser; the page renders **2D top-down on the left** and **3D Three.js on the right**, both with the template shell + zones as a backdrop.

**Multi-tenant ports.** Two concurrent `run.sh` invocations get isolated servers automatically — the script tries the defaults (`session=8766`, `viz=8765`) first and bumps to the next free port (up to +32) if either is taken. The chosen ports come back in the JSON config's `session.url` / `viz.url` fields; subsequent tool calls must use those URLs, not the hard-coded `8766`. Pass `--session-port N` / `--viz-port N` to demand a specific port (will fail loudly if taken). The legacy "kill anything on this port before binding" behaviour is opt-in via `--force-kill` — by default, run.sh never disturbs another user's session.

Per-session live-event logs land at `$TMPDIR/sketch_live/<session-port>/events.jsonl` so concurrent viz processes don't clobber each other. Override with the `SKETCH_LIVE_EVENTS` env var.

**The LLM MUST tell the user the viz URL immediately after launching the session, in plain text on its own line, like this:**

> Live viz: http://localhost:8765/   (open this in a browser to watch placements arrive)

Repeat this line whenever a new session starts.

### Step 2: orient

Always do these calls first:

- `GET /status` — anchor's placement count + footprint
- `POST /tool/template_view` — **site shells + zones with their world origins and bounds**. Required reading before placing in empty-canvas mode; tells you the valid XY region in which placements must lie.
- `POST /tool/query_zones` — zones to target (merges spatial zones + LLM-labelled regions from `templateZones[]` if `extract_regions.py` has run)
- `POST /tool/query_pack_archetypes` — what archetypes the pack offers; each entry includes a `semantics` block
- `POST /tool/query_archetype_semantics` (optional, no body) — full semantics map for every archetype
- `POST /tool/query_surfaces` — surfaces available for `place_on` (counters, table tops, rack tops, shelves). Filter by `regionId`, `minFreeAreaM2`, or `label`.

**Coordinate convention.** Every shell / zone defines its valid XY region as `[originWorldM[0], originWorldM[0] + boundsM.widthM] × [originWorldM[1], originWorldM[1] + boundsM.depthM]`. Empty-canvas mode ships with a single site shell at `originWorldM=[0,0,0]` extending to `(widthM, depthM)` — so valid coords are `[0, W]` × `[0, D]`, **NOT** `[-W/2, +W/2]`. The collision gate rejects placements whose `posM[0:2]` is outside every declared shell and zone with `reason="out_of_bounds"`; the cheap fix is to read `template_view` first.

**Read every archetype's `semantics.typicalContext` and `semantics.avoidNear` before you commit any placements.** Ignoring them produces nonsense layouts (objects placed where their `avoidNear` neighbors already sit, perimeter objects placed mid-room, etc.).

### Step 3: plan an archetype-mix distribution (intent expansion)

Translate the user's `intent` and `addTarget` into per-archetype goals. For larger scenes or short-but-evocative intent ("dense data center", "automotive parts depot"), borrow MesaTask's "object list completion" pattern (arXiv 2509.22281 §4.2): instead of asking the user to enumerate every archetype, expand the theme into the FULL set of co-occurring objects a realistic scene would contain — data center → server racks AND CDUs AND CRAHs AND PDUs AND fire extinguishers AND walkways — then weight per-region according to how a real scene of that theme looks.

**Division of labor** — scripts own everything deterministic so the LLM only spends tokens on theme reasoning:

| Step | Who | Why |
|---|---|---|
| Inventory pack archetypes, filter invalid bboxes, compute per-region attempt budgets | `pack_summary.py` | Deterministic. On a large pack (200+ archetypes), having the LLM read raw `pack.json` costs hundreds of lines of context for what's effectively a `jq` query. |
| Define regions, pick archetypes for the theme, map archetypes → regions, set per-region weights, write per-region notes, flag archetypes that *should* be in the pack but aren't | LLM | Non-deterministic. Requires theme priors no script can reproduce. Missing-archetype flagging is the most knowledge-intensive — it reasons about what's outside the pack. |
| Cross-check the LLM's plan against the pack (hallucinated archetypes, sentinel bboxes, omitted archetypes, malformed weights) | `validate_archetype_plan.py` | Deterministic. The LLM can't reliably re-check its own JSON without burning the same token cost as inventory. |
| Slice per-region into `random_fill.py`'s `--plan` form | `jq` one-liner | Trivial transform. |

**Workflow:**

1. Summarize the pack and compute the attempt budget:

   ```bash
   "$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
       "$SKILL_DIR/scripts/pack_summary.py" \
       --pack /path/to/asset_pack \
       --regions '{"<region_a>": {"weight": 0.80}, "<region_b>": {"weight": 0.15}, "<region_c>": {"weight": 0.05}}' \
       --target-placements 10000
   ```

   Output (compact, one record per line):
   ```
   3 archetype(s), 8 valid asset(s), upAxis=Z, mpu=1.0
     <arch_a>    : 3 assets · 1.3×1.2×3.2 m – 1.3×1.2×3.2 m
     <arch_b>    : 3 assets · 1.6×3.7×3.4 m – 0.92×2.5×2.1 m
     <arch_c>    : 2 assets · 1.6×0.66×2.4 m – 1.4×0.6×2.3 m

   per-region attempt budget (target=10000, ×1.3 cushion):
     <region_a> (weight 0.8): 10400 attempts
     <region_b> (weight 0.15): 1950 attempts
     <region_c> (weight 0.05): 650 attempts
   ```

2. LLM writes the plan to `$OUT/expanded_plan.json`. Schema (matches `random_fill.py`'s expected `--plan` after slicing each region):

   ```json
   {
     "intent": "<one-sentence theme description>",
     "targetPlacements": 10000,
     "perRegion": {
       "<region_a>": {
         "attempts": 10400,
         "archetypes": {"<arch_a>": 12, "<arch_b>": 3, "<arch_c>": 1},
         "note": "<region_a> dominates; <arch_b> interspersed every few rows"
       },
       "<region_b>": {
         "attempts": 1950,
         "archetypes": {"<arch_b>": 5, "<arch_c>": 2, "<arch_a>": 1},
         "note": "perimeter biased toward <arch_b>"
       }
     }
   }
   ```

   Use the per-region `attempts` values from `pack_summary.py` directly (cushion already applied). `note` is free-form LLM commentary.

   Include an `_llmNote` field listing archetypes that *would* belong in the theme but are absent from the supplied pack. This is the irreducible LLM step — neither `pack_summary.py` nor `validate_archetype_plan.py` can produce it, since both see only what's *in* the pack.

3. Validate the plan against the pack:

   ```bash
   "$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
       "$SKILL_DIR/scripts/validate_archetype_plan.py" \
       --pack /path/to/asset_pack \
       --plan $OUT/expanded_plan.json
   ```

   Exit codes: 0 ok, 1 hard errors (hallucinated archetypes, sentinel bboxes, bad weights — stop), 2 warnings only (omissions — caller decides).

4. Slice each region into `random_fill.py`'s `--plan` form and execute (see Step 4b).

### Step 4: place

There are two modes, and choosing between them is the most common boundary error:

| Use case | Mode | Section |
|---|---|---|
| >50 placements of the same archetype with shared constraints | **bulk-fill scripts** (skip per-placement `query_nearby`) | Step 4b |
| <30 bespoke placements (one-offs, specific landmarks, corrections) | **per-placement** with `place` + `query_nearby` first | Step 4a |
| On-surface placements (boxes on shelves, props on tables) | **surface_fill** (a bulk variant of `place_on`) | Step 4b |

**The semantic gate (avoidNear/preferredNear/placementBias) runs inside both bulk scripts and the place tool** — it's a property of the engine, not of the workflow. Per-placement `query_nearby` is only needed when the LLM is making contextual decisions (e.g. "what's near here that I should react to?"), NOT to enforce semantic constraints.

#### Step 4a: per-placement (small batches, bespoke logic)

For each placement target:

1. **Query nearby context FIRST.** `POST /tool/query_nearby {posM, radiusM, limit}`. Each returned placement includes a `semantics` block. Read neighbors' semantics, not just their archetype names.

2. **Apply semantic rules** before committing:
   - The current archetype's `preferredNear` should overlap with at least one neighbor's archetype, OR the spot should be near an open zone where that archetype typically lives.
   - The current archetype's `avoidNear` should NOT include any close neighbor's archetype within ~2 m. If it does, pick a different point.
   - General patterns: archetype whose `placementBias = "edge"` → near footprint walls/columns, never mid-zone. Archetype whose `placementBias = "center"` and whose `avoidNear` includes its perimeter counterparts → centre of the room.

3. **Then** `POST /tool/place` with `{archetype, posM, slotM, yawDeg, id}`. The collision gate (3D OBB-SAT) rejects overlaps — structural safety net; semantics are the reasonableness check on top.

4. **On rejection**, pick a different point AND re-check semantics for the new context (don't blindly retry far away from a thoughtful spot).

5. Stop retrying after a few attempts per goal; report unfilled goals back honestly.

For asset-specific picks (e.g. "fill rack slots with the 3m rack model, not the large variant"), the LLM should pick assets explicitly:

1. `POST /tool/query_pack_assets` — returns all candidates per archetype with absolute + relative paths.
2. For each placement, decide: which specific `assetPath` to use, and whether to `scaleM` it to fit a non-matching slot.
3. Pass to `place`: `{archetype, posM, slotM, yawDeg, id, assetPath, scaleM}`.

#### Step 4b: bulk fill (scripts)

Three bulk-placement scripts in the same family, picked by placement surface:

**`densify_zones.py` — regular grids on the floor.** Reads zones from the live session, infers each archetype's slot dims from existing placements (median `slotM`), walks a deterministic row-major grid over each region with cell pitch = `max(slotW, slotD)`, and POSTs the batch. Use when the user wants rows/grids/lattices.

```bash
"$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
    "$SKILL_DIR/scripts/densify_zones.py" \
    --session http://127.0.0.1:8766 \
    --zones '{"<region>":{"<archetype>":<count>, ...}, ...}'
```

Slot lookup falls back to `--slot <arch>=w,d,h` when an archetype has no existing placements. **No zones in the template** (empty-canvas mode, custom anchor without zone prims): declare ad-hoc rectangles with `--region NAME=ox,oy,w,d` (repeatable). Names in `--zones` resolve to a `--region` first, then to a session zone.

Patterns:

```bash
# Pack zones tighter — existing zones, slot dims inferred from placements
densify_zones.py --session http://127.0.0.1:8766 \
    --zones '{"<zone_a>":{"<arch_x>":<count>,"<arch_y>":<count>}, \
              "<zone_b>":{"<arch_x>":<count>,"<arch_y>":<count>}}'

# Empty-canvas — quadrant A for arch_a, quadrant B for arch_b, central spine of arch_c
densify_zones.py --session http://127.0.0.1:8766 \
    --region region_a=0,0,1000,1000 \
    --region region_b=1000,1000,1000,1000 \
    --region spine=0,950,2000,100 \
    --zones '{"region_a":{"<arch_a>":600}, \
              "region_b":{"<arch_b>":1200}, \
              "spine":{"<arch_c>":150}}'

# Mount <arch_w> along the south wall of region <r> — thin strip, slot specified
densify_zones.py --session http://127.0.0.1:8766 \
    --slot <arch_w>=0.2,0.2,0.6 \
    --region south_wall=0,0,144,1.5 \
    --zones '{"south_wall":{"<arch_w>":28}}'
```

Same session state + same args → identical output (no RNG).

**`random_fill.py` — irregular scatter, density-fill.** Takes one or more regions (declared inline with `--region`, picked up from `query_zones`, or auto-derived from the shell with `--auto-shell`), a per-region archetype weight mix, and an attempt budget; shuffles a coarse cell grid (seeded RNG) and POSTs a single `place_many` per region.

The **semantic gate** (default ON when `--pack` is given) consults `pack.semantics` per candidate cell:
- `avoidNear`: rejects the cell if any neighbor within `--neighbor-radius-m` (default 2 m) matches the candidate's avoidNear list. **Hard reject.**
- `preferredNear`: **soft** — only enforced once at least one neighbor exists. Lets the first placement bootstrap; subsequent placements have to cluster with at least one preferredNear archetype.
- `placementBias`: `"edge"` / `"center"` / `"corner"` / `null`. Modulates per-cell archetype-choice weights by the cell's edge-distance score. Cap is `0.05` so a strongly-biased archetype can still appear in a mismatched cell at low rate.

Pass `--pack <path>` to enable the gate. Pass `--no-semantic-gate` to revert to pure weighted random. The per-region report grows `semSkipAvoid` and `semSkipPreferredMiss` counts so you can see the gate's filter rate.

```bash
"$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
    "$SKILL_DIR/scripts/random_fill.py" \
    --session http://127.0.0.1:$SESSION_PORT \
    --pack /path/to/pack \
    --auto-shell \
    --plan '{"shell": {"attempts": 8000, "archetypes": {"<arch_a>": 5, "<arch_b>": 2, "<arch_c>": 1, "<arch_d>": 1}}}' \
    --seed 42
```

**Region-selection strategies for `random_fill` (config inside the plan):**

- `shell` (default, **required-at-least-once**) — sweep every cell across the whole site shell, including the strips between declared zones. Without it, inter-zone gaps stay empty and the realized stage looks like isolated lattices floating in a void. For a cell at `(x, y)`: if the cell falls inside a zone, use that zone's `allowedArchetypes`; if it falls in an inter-zone gap, use a generic mix of pack archetypes reasonable for an open floor.
- `zone` — `random.choice` one zone from `query_zones()`, densify only inside it. Use only when the user names the zone: *"fill the dock zone tighter."*
- `area` — pick a random sub-rectangle inside the footprint. Use when the user names a region by location: *"add more activity in the NW corner."*
- `mixed` — run several passes: 1× `shell` first for global coverage, then 1–2× `zone` passes biased toward zones still below target density, then optionally 1× `area` for visual variety. **The first pass must be `shell`.**

Whichever strategy you pick, **shuffle the resulting cell list** before greedy-placing — never iterate in index order.

**`surface_fill.py` — on-surface placements (boxes on shelves, cups on counters).** Walks the runtime surface registry, picks an archetype per slot, posts `place_on`. Requires the owning archetype's `semantics.surfaces` to be declared (see "Enrich pack with semantics") plus an explicit slot for the thing being placed.

```bash
"$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
    "$SKILL_DIR/scripts/surface_fill.py" \
    --session http://127.0.0.1:$SESSION_PORT \
    --surface-label shelf \
    --archetypes '{"Boxes": 6, "Containers": 2}' \
    --slot 'Boxes=0.5,0.5,0.4' \
    --slot 'Containers=0.6,0.6,0.7' \
    --per-surface-max 5 \
    --layout grid \
    --seed 42
```

`--layout grid` (default) lays a deterministic row-major lattice per surface sized to the slot. `--layout random` is the legacy mode — useful for organic scatter but saturates at ~1–2 placements per surface because subsequent random samples keep colliding with the first one.

`surface_fill` automatically passes the surface's `topWorldZ` through to `place_on` so multi-shelf owners (e.g. a rack with surfaces at z=1, 2, 3) get each shelf addressed individually.

`--slot ARCH=w,d,h` is mandatory for every archetype — the script has to know each on-surface placement's size to compute per-surface capacity AND to validate the placement fits within the surface's footprint.

Common patterns (trigger phrases → invocation):

| User phrase | Invocation |
|---|---|
| *"items on the shelves"* | `--surface-label shelf --archetypes '{"<arch>": 1}' --per-surface-max 5` |
| *"props on the table"* | `--surface-label top --archetypes '{"<arch_a>": 3, "<arch_b>": 2}' --layout random` |
| *"only fill the storage zone, not the office"* | add `--region storage_aisles` |

**Floor-first, surfaces-second.** Run floor placements before on-surface ones — anchors and their surfaces become available in the registry as they're inserted, so `surface_fill` can decorate freshly placed anchors. (The runtime surface registry updates incrementally on every `place` / `place_many`.)

### Step 5: realize and report

```
POST /tool/realize  {}
```

Returns `{ok, rootUsd, manifest, snapshot, summary, templateBreakdown}`. `manifest` is the **path** to `manifest.json` on disk (a string), NOT the parsed dict — load it with `json.load(open(d["manifest"]))` if you need its inner fields.

**Use `scripts/report.py` to summarize the run** instead of writing inline-python inspection blocks per check:

```bash
python3 $SKILL_DIR/scripts/report.py <out_dir>
```

It prints paths, placement counts, composition strategy, per-archetype real-asset breakdown, unfilled warnings, and live-session status — same fields as the table below:

| Field | Source | Meaning |
|---|---|---|
| `anchorPlacementCount` | `templateBreakdown.anchorPlacementCount` | placements the absorbed template gave you (the "1 tile" baseline) |
| `addedPlacementCount` | `templateBreakdown.addedPlacementCount` | placements you added on top via tools |
| `perTilePlacementCount` | same as anchor count when tiling | placements per tile copy |
| `tilesActuallyPlaced` | `addedPlacementCount / perTilePlacementCount + 1` | total tile copies materialized |
| `totalPlacementCount` | `templateBreakdown.totalPlacementCount` | session sum |
| `byArchetype` filled / picks | `summary.byArchetype` | per-archetype counts |
| `composedPrimCount_usdcore` | `manifest["composedPrimCount_usdcore"]` | usd-core composed prim count (undercount when references don't load locally) |
| `unfilled[]` | `summary.unfilled` | placements that didn't fit the pack and weren't synthesized |
| `rootUsd` | `rootUsd` | path to root.usd to open in Kit |
| `compositionStrategy` | `manifest["compositionStrategy"]` | composition arc type, hierarchy, instancing |
| `layers` | `manifest["layers"]` | root layer, bulk content layer, wrapper layers |

If you tiled to hit a target prim count, **also report**: tile count attempted, per-tile placements, placed/rejected per tile, total composed prim count vs target.

**Diagnosing "things look overlapped" after realize.** Run `python3 $SKILL_DIR/scripts/check_overlaps.py [--session URL] [--pack /path]` against the live session. It audits three failure modes the user can't tell apart by eye: (1) actual OBB-overlap (gate bug or gate bypass), (2) rendered-asset bbox bigger than its declared slot, (3) pivot drift between asset origin and bbox-center-floor. Prints the first 20 offending pairs + per-archetype slot-vs-asset-bbox table.

### Step 6: iterate (optional)

The user can prompt for more changes — *"add more `<arch>` along the central aisle"*, *"remove all the `<arch>`s"*, *"replace the south-row items with the larger variant"* — and the LLM repeats Steps 2–5.

### Step 7: teardown (REQUIRED at end of task)

After the user is done — and especially after `realize` if no follow-up is expected — **the LLM MUST shut the session server down**. Leaving the server running ties up its port forever and (over a long Claude Code session) leaks file descriptors.

```bash
# Path A (HTTP, what /sketch-stage uses internally):
curl -s -X POST http://127.0.0.1:$SESSION_PORT/tool/shutdown -d '{}'

# Also kill the viz process (its PID came back from run.sh's launch JSON
# under viz.pid):
kill $VIZ_PID 2>/dev/null || true
```

```
# Path B (native MCP):
mcp__sketch-placement__shutdown()
```

The `shutdown` tool exits the server in a background thread so the response is delivered first. Multiple calls are idempotent. If you forget, the next `run.sh` invocation will auto-bump to the next free port, so the cost is "an orphan process and a leaked port" rather than a hard failure.

---

## Special modes

### Empty-canvas / theme-driven generation

When there's no anchor template to start from (the recommended starting point for any non-absorbed scene), the skill synthesizes a shell-only anchor of the requested size via `--bounds`:

```bash
OUT="$PWD/sketch_runs/theme_run"
bash $SKILL_DIR/scripts/run.sh \
    --bounds 10000x10000 \
    --pack <asset-pack-root> \
    --intent "<your theme description — describe regions / sub-zones / archetype mix in plain language>" \
    --mode fill \
    --out "$OUT"
```

The LLM is then responsible for the entire layout. The workflow:

0. **Fetch the canvas bounds first.** Call `POST /tool/template_view` and read the single shell entry: `originXY = shell.originWorldM[0:2]` and `extentXY = (originXY[0] + shell.boundsM.widthM, originXY[1] + shell.boundsM.depthM)`. **All subsequent placements must have `posM[0:2]` inside `[originXY, extentXY]`.** Empty-canvas mode always uses `originWorldM = [0, 0, 0]` — the shell extends from `(0, 0)` to `(W, D)`, NOT centered at origin. The place-gate hard-rejects out-of-bounds placements with `reason="out_of_bounds"`.

1. `query_zones()` will return empty. The LLM partitions the canvas into theme-zones from the `intent` using `(originXY, extentXY)`. It can author logical zones inline by remembering footprint sub-rectangles in working memory — it does NOT need to author USD zone prims; the partitioning is conceptual, used to drive per-zone archetype mixes.

2. `query_pack_archetypes()` + `query_pack_assets()` + `query_archetype_semantics()` for the pack's archetypes + semantics.

3. Translate the intent into a per-zone archetype distribution (each zone gets its own archetype mix and weights). Use holistic, semantics-aware reasoning — see "Anchor-binding strategy" below for the cross-pack mapping pattern.

4. For each zone:
   - Pick anchor lines / aisles / clusters that match the theme.
   - Generate placement coordinates per the archetype's `preferredNear` and `avoidNear`. **Constrain every coord to `[originXY[i] + margin, extentXY[i] − margin]`** (margin ≥ slot/2 keeps bboxes off the wall).
   - `place_many` with explicit `assetPath` + `scaleM` + `yawDeg`. Collision gate prevents overlaps.

5. Realize. If the canvas still feels sparse vs the user's `intent`, run `mode=fill` (Step 4b) to densify.

**Don't try to literally fill 10km × 10km uniformly** — at 1 placement per 4m² that's 25M placements, far past authoring capacity. Aim for visual density: clusters of placements in zones, big empty corridors between, sparse scatter elsewhere.

For very large canvases (≥1km), tile your own theme blocks: build one 200×200m block of a coherent layout, then `place_many` translated copies across the canvas with deliberate spacing.

#### Asset pick must match canvas scale + theme

Real asset libraries hold parts at **true real-world size**, ranging from 5 cm (a knife) to 5 m (a counter run). For empty-canvas work, "fits the role semantically" is necessary but not sufficient — the asset must also be **visually appropriate at the target canvas scale**:

1. After `query_pack_archetypes` + `query_pack_assets`, inspect each candidate's `size`. Reject candidates whose native size is too small to be visible at the canvas's typical camera distance, *unless* you're going to scale them up to display dimensions.
2. For canvases ≥ 100 m, prefer assemblies (whole engines, full frames) over tiny sub-components (injectors, bolts). The user's "automotive parts depot" doesn't mean "scatter 5 cm fuel injectors across 1 km" — it means "showroom of recognizable parts at depot-display scale".
3. When multiple assets in the pack work for a role, **prefer the larger one** for big canvases and the smaller one for indoor/room-scale anchors.
4. Decide visibility scale per-zone: a "showroom display" at depot scale wants a slot of ~5–10 m on a side; a "shelf prop" wants 0.3–0.5 m. Match `slotM` to the intended display size, not the asset's raw size.
5. Bind with `bind_archetype(..., fitMode="fit-to-slot")` so the engine derives `scaleM = slot / asset_world_bbox` per placement. This decouples your asset choice from the placement's visible size.

**Canvas-to-display ratio rule.** A placement is visible from Kit's default camera framing when its largest dimension is roughly 1–3% of the canvas's largest dimension. Below ~0.5% it's too small to read; above ~10% it dominates and the layout collapses to a few items.

| Canvas | Comfortable display size | Example |
|---|---|---|
| 100 m | 1–3 m | indoor room — typical assets at native size |
| 1 km | 10–30 m | small depot — assemblies on display platforms |
| 2 km | 20–40 m | medium depot — same parts with bigger slots |
| 10 km | 100–300 m | rare; needs scaled-up clusters or proxy structures |

If the user picks a 10 km canvas with intent at "depot" granularity, push back and suggest 1–2 km — or the visible result is a few specks scattered across an otherwise empty plane. The empty-canvas mode is most useful at 200m–2km.

### Resume from snapshot

Every `realize()` writes `_snapshot.sketch.json` next to the realized `root.usd`. The snapshot contains the **full session state** at realize time — anchor placements, every added placement, every bound `assetPath` + `scaleM` + `yawDeg`, the zones, and the `compositionStrategy`.

To pick up where a prior session left off:

```bash
bash $SKILL_DIR/scripts/run.sh \
    --resume /path/to/prior_run/_snapshot.sketch.json \
    --scale small --intent "<incremental change>"
```

The new session loads the snapshot, the LLM can call `query_template_placements`/`query_stage_graph` to see what's already there, then add more placements via `place`/`place_many`. The `--pack` from the snapshot is reused unless overridden.

### Anchor-binding strategy: bind real assets vs tile-to-target

After the template anchor is loaded, the LLM must fill all template placements before adding any free placements. There are two binding strategies:

**A. Pack has real assets (the normal case).** Asset choice and scale are reasoned LLM decisions, not arithmetic.

**Cross-pack holistic mapping** is REQUIRED for any absorbed anchor when the source's archetype names don't match the pack's archetype names — which is the **default case** with the vocabulary-free absorber, since absorbed archetypes come from the source asset filenames (`Kitchen_Cabinet`, `Living_Couch`, ...) and pack archetypes come from the pack's own categories (`Cabinets`, `Counters`, ...).

The LLM is responsible for this binding. If you `realize()` an absorbed sketch against a pack without binding first, every placement comes back **unfilled** (no candidate matches the source archetype name) and the realized USD is empty / synthesized cubes. Always run the binding pass first.

**Binding pass:**

1. Pull **all** anchor archetypes — call `query_template_placements()` (no arg) and group by `archetype` field. Each unique source archetype needs one binding decision.
2. Pull **all** pack archetypes + their assets — `query_pack_archetypes()` + `query_pack_assets()` (no arg). Each entry includes `size` (bbox) and `semantics`.
3. For each source archetype, pick the **best-fitting pack asset** considering:
   - **Slot dimensions** — the source archetype's typical slot (visible via `query_template_placements`) versus each pack asset's `size`. Pick the asset whose native bbox is within roughly 3× of the slot on at least one axis. If nothing is close, pick the closest and let `fitMode="fit-to-slot"` scale it.
   - **Semantic role** — the source archetype name and `query_archetype_semantics` (description, typicalContext, preferredNear, avoidNear) versus the pack archetype's semantics (description, affordances, anchors). A "Living_Couch" maps best to a pack archetype with `affordances=["seating","rest"]`; a "Kitchen_Counter" to one with `surfaces` declared. When no semantic match exists, fall back to size + visual character.
   - **Holism** — reason about the whole mapping table together, not per-archetype-in-isolation. If a single pack archetype (e.g. `Equipment`) is the only sensible match for several distinct source archetypes (Refrigerator, Oven, Washer), you may want to bind them to different *assets* within `Equipment` rather than all to the same one. `query_pack_assets("Equipment")` returns every candidate so you can pick visually distinct ones for distinct roles.
4. For each source archetype, call `bind_archetype(archetype="<source_archetype>", assetPath="<chosen pack asset path>", fitMode="fit-to-slot")`. With `fit-to-slot`, the engine measures the asset's world-bbox at bind time and computes per-placement `scaleM` so the rendered geometry matches the source's slot exactly — no manual scale math needed regardless of source/pack mpu/upAxis differences.
5. Verify with `query_template_placements(unfilledOnly=true)` — should return empty. If anything's still unfilled, that archetype's binding failed (e.g. asset path was wrong); the response from `bind_archetype` lists which placements got updated and which didn't.
6. `realize()` — every placement now binds to a real mesh.

The pack's `archetypeAliases` (in `pack.json`) is a *fallback* convenience, not a substitute — the LLM's holistic mapping overrides it whenever the LLM has a better semantic match.

**Honest reporting when no good match exists.** If the source has archetypes the pack genuinely doesn't cover (e.g. apartment USD with `Bedding22` but the pack is industrial-only with no soft-furniture analog), the LLM should pick the least-bad fit AND flag it back to the user — *"I bound Bedding archetypes to Plywood_Crate as the closest visual match; consider running against a residential pack for better fidelity."* Don't silently leave placements unfilled; the user wants the layout populated, even if imperfectly.

**Up-axis correctness for `scaleM`.** If the pack has `upAxis: Y`, the engine rotates each asset +90° about X at realize time so its Y becomes world Z. When computing `scaleM = slot / asset_bbox`, compare slot dimensions to the asset's **world bbox after rotation**, not its raw bbox:

```
asset bbox (asset frame, Y-up): (asset_x, asset_y, asset_z)
asset bbox (world after rotation): (asset_x, asset_z, asset_y)   ← Y and Z swap

scaleM[0] = slot_widthM   / asset_x
scaleM[1] = slot_depthM   / asset_z      # not asset_y
scaleM[2] = slot_heightM  / asset_y      # not asset_z
```

The engine then permutes `scaleM` back to asset-local axes before applying the scale op, so `scaleM` is always *world-axis* factors from the caller's perspective.

Per-archetype binding flow:

```
1. query_template_placements()                   # all anchor positions
2. group placements by archetype                 # one decision per archetype
3. for each archetype encountered:
     a. query_archetype_semantics(archetype)
        → read description, typicalContext, preferredNear, avoidNear
     b. query_pack_assets(archetype)
        → candidates (path, relPath)
     c. PICK A SPECIFIC assetPath:
        - prefer the candidate whose bbox best matches the slotM
        - prefer candidates whose name + semantics matches the slot's role
     d. DECIDE scaleM (or leave None for no scaling):
        - if slot W/D/H matches asset bbox closely → no scale
        - if slot is bigger → scale up to fill, OR keep asset and accept gap
        - if slot is smaller → scale down to fit (requires explicit consent
          to override the no-distortion policy)
        - account for upAxis as above
        - size sanity: if asset_bbox is 5–10× smaller than slot, double-check
          intent. Prefer a candidate within ~3× of the slot in at least one
          axis, or use bind_archetype(..., fitMode="fit-to-slot") and accept
          the intentional scale-up.
     e. for every placement of this archetype:
          update_placement(id, assetPath=..., scaleM=...?)
4. realize → check composedPrimCount vs target
5. IF below target: tile copies (Step 4 mode=tile) — apply the SAME per-
   archetype asset choice from step 3 to tile copies via place_many.
6. realize again
```

Each archetype-level decision covers tens or hundreds of placements; this is genuine LLM reasoning, not loops over coords.

**B. Tile-to-target benchmark.** When you just want N copies of the
current sketch to hit a target composed prim count — typically against a
synth-only pack (pack.json with empty `archetypes` → every placement
becomes a `Cube + Xform` at realize time, 2 prims/placement) for benchmark
comparisons. No semantic reasoning; pure grid math:

```bash
"$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
    "$SKILL_DIR/scripts/tile_template.py" \
    --session http://127.0.0.1:$SESSION_PORT \
    --target-prims 100000 \
    [--prims-per-placement 2]    # 2 = synth Cube+Xform; override for
                                 # real-asset packs with deeper hierarchy
    [--gap 2.0]                  # meters between tile copies
```

The script reads the current anchor from `query_stage_graph`, computes
`tiles_needed = ceil(target_prims / prims_per_placement / placements_per_tile)`,
lays a roughly-square grid (`ceil(sqrt(tiles))` cols × `ceil(tiles/cols)` rows),
and emits a single `place_many` of all translated copies. Same anchor +
same args → identical output (no RNG). Caller realizes separately.

---

## One-time pack preparation

These steps prepare a pack for use with the skill. Each pack only needs them once (or whenever its archetype set changes). They produce a `pack.json` that the per-run flow above consumes.

### Generate `pack.json`

The session, MCP tools, and realize step all expect `pack.json` at the pack root, declaring `upAxis`, `metersPerUnit`, `archetypes`, and (optionally) `archetypeAliases` / `rootStage`.

`scripts/gen_pack_json.py` walks an asset directory and writes `pack.json` next to it. Independent of `--scale`, `--out`, or any per-run setting.

```bash
"$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" \
    "$SKILL_DIR/scripts/gen_pack_json.py" /path/to/assets \
    [--scenario tag] [--name custom_name] [--force] \
    [--flat] [--subpacks dir1 [dir2 ...]] [--subpack-min-assets N]
```

What it does:
- Walks the tree for `*.usd` (skipping `payloads/`, `looks/`, `materials/` and obvious stub names).
- Opens each candidate stage and reads its **own** `UsdGeom.GetStageUpAxis` + `GetStageMetersPerUnit`. The realizer trusts asset metadata over pack-level hints.
- Measures bbox via `UsdGeom.BBoxCache`, converts to meters, groups by parent-folder name.
- Writes `archetypes: { <category>: [{ path, size: { widthM, depthM, heightM } }] }`. Per-axis upAxis/mpu vote counts land in `_scanStats`.

**Nested theme-packs (auto-detected).** When the root contains two or more immediate child directories that each hold ≥ `--subpack-min-assets` (default 5) entry-point USDs, the script emits a *manifest* pack.json at the root listing `subpacks: [{name, theme, path, archetypeCount, assetCount, …}]`, and writes a flat pack.json **inside each sub-pack** with its own `theme` field. `--flat` forces single-pack output. `--subpacks DIR [DIR …]` overrides auto-detection.

**Downstream scripts auto-flatten manifests.** `pack_summary.py` and `validate_archetype_plan.py` accept a manifest root directly; they load each sub-pack's pack.json transparently and present archetypes namespaced as `<theme>.<archetype>`. Pass `--subpack <theme>` (repeatable) to restrict to chosen themes.

**Compact discovery (token-thrifty pack inspection).** Never grep `pack.json` directly; use these `pack_summary.py` modes:

```bash
# What themes does this pack support?
pack_summary.py --pack ROOT --themes
#   -> <theme_a>     3 archetype(s)  3 asset(s)  path=<theme_a>
#      <theme_b>     3 archetype(s)  3 asset(s)  path=<theme_b>

# What archetypes are in this theme (or this whole pack)?
pack_summary.py --pack ROOT --subpack <theme_a> --list-archetypes
#   -> <theme_a>.<archetype_x>   1 asset(s)
#      <theme_a>.<archetype_y>   1 asset(s)
#      <theme_a>.<archetype_z>   1 asset(s)

# What asset USDs back this one archetype?
pack_summary.py --pack ROOT --list-assets <theme_a>.<archetype_x>
#   -> <theme_a>/<archetype_x>/<asset>.usd
```

`run.sh` will also call this generator automatically when its `--pack` argument points at a directory without `pack.json` (or when `--auto-gen-pack` is passed). That convenience is for the full launch flow; it does NOT mean `--scale` factors into pack generation.

Do NOT inline-generate `pack.json` from the agent prompt; use the script so the logic stays in one place.

### Enrich `pack.json` with per-archetype semantics

`pack.json` from the previous step has structural data only (paths + bboxes). For richer scene generation — semantic region detection in absorbed templates, on-surface placement (cup-on-table), and the placement-side semantic gate (Step 4b) — each archetype needs a `semantics` block. The schema:

```json
"<archetype>": [{
  "path": "...", "size": {...},
  "semantics": {
    "anchors": "rack_row" | null,                   // vocabulary token; seeds regions in extract_regions
    "affordances": ["compute", "store_servers"],     // vocabulary tokens; reveal what this archetype DOES
    "surfaces": [                                    // top-faces other placements can rest on (place_on)
      {"label": "rack_top", "localTopZ": 2.45, "footprintM": [1.6, 0.7]}
    ],
    "preferredNear": ["server_rack", "cdu"],         // FREE-FORM archetype names (not vocab); SOFT clustering hint
    "avoidNear":    ["<archetype_x>", "<archetype_y>"],  // FREE-FORM archetype names; HARD reject if a neighbor matches
    "placementBias": "edge" | "center" | "corner" | null,  // random_fill cell-selection bias
    "_harvestSource": "Kind"                         // diagnostic — which USD source produced the harvest
  }
}]
```

The first three fields (`anchors` / `affordances` / `surfaces`) drive **region detection and on-surface placement**. The last three (`preferredNear` / `avoidNear` / `placementBias`) drive the **placement-side gate** in `random_fill`.

**Enrichment flow:**

1. **Harvest** deterministic semantics from each asset:
   ```bash
   "$VENV/bin/python" "$SKILL_DIR/scripts/harvest_pack_semantics.py" --pack /path/to/pack
   ```
   Scans every asset's `UsdSemantics` + `Kind` + `customData["simReady"]`, writes a `semantics: {anchors, surfaces, affordances, _harvestSource}` block into each archetype entry, and adds `_pendingSemanticGapFill: [{archetype, missing, harvestedHints}]` listing what still needs LLM input. SimReady-tagged packs come out mostly populated; generic packs come out mostly empty.

2. **Fill** the gaps with LLM judgment: read each sub-pack's `pack.json`, look at `_pendingSemanticGapFill`, and edit the `semantics` block of each archetype using values from `references/pack_semantics_vocabulary.json`. After reviewing an archetype, remove it from `_pendingSemanticGapFill` to acknowledge — including when you decide a field stays `null`.

3. **(Optional, strongly recommended for surface-heavy packs) Measure actual shelf/top positions** from the asset USDs:
   ```bash
   "$VENV/bin/python" "$SKILL_DIR/scripts/measure_surfaces.py" \
       --pack /path/to/pack --archetype Racks --label shelf --write
   ```
   The LLM gap-fill writes `surfaces.localTopZ` as analytical estimates (h/3, 2h/3, h). For monolithic-mesh assets (where the whole asset is one mesh prim with internal shelves authored as horizontal faces), the analytical values are off by enough to make on-surface props visibly float or clip at realize. `measure_surfaces.py` opens each asset's USD and finds horizontal mesh faces via three fallbacks: prim-name match → geometric flat-mesh prims → mesh-face vertex analysis. Use it whenever a pack has surface-bearing archetypes. For packs that ship with Windows-style backslash payload references, also create per-asset-dir symlinks first (`ln -s payloads/base.usda 'payloads\base.usda'`).

4. **Validate**:
   ```bash
   "$VENV/bin/python" "$SKILL_DIR/scripts/validate_pack_semantics.py" --pack /path/to/pack
   ```
   Catches off-vocabulary tokens, malformed surface entries, and remaining gaps. Exit 0 = clean, 1 = errors, 2 = warnings only.

Vocabulary lives in `references/pack_semantics_vocabulary.json` — extend by adding tokens; the validator picks up new tokens on next run.

### Enrich an absorbed sketch with semantic regions

When an absorbed template has no `templateZones[]` (or only coarse spatial zones), `extract_regions.py` runs a four-pass region pipeline:

```bash
"$VENV/bin/python" "$SKILL_DIR/scripts/extract_regions.py" \
    --sketch /path/to/absorbed.sketch.json \
    --pack   /path/to/pack \
    [--write]    # default dry-run; --write persists templateZones[]
```

Passes:
1. **Anchor** — each placement whose archetype declares `semantics.anchors=<region_type>` seeds a region around itself; nearby placements get absorbed.
2. **Cluster** — remaining placements get 1m-grid clustered, dominant archetype names the cluster.
3. **Gap** — large negative-space rectangles inside the shell become "circulation" regions.
4. **LLM region-label** (manual workflow) — after `--write`, the LLM reads each region in the sketch's `templateZones[]` and edits it in place to add `name`, `purpose`, `allowedArchetypes`, `note` using the same vocabulary file.

Then run `validate_absorbed_regions.py --sketch <path>` to confirm regions are well-formed and (optionally) labelled. The session's `query_zones` merges these labelled regions with spatial zones at runtime, and `query_surfaces` / `place_on` use `allowedArchetypes` to reject category mismatches.

---

## Tools (over HTTP and MCP)

| Tool | Purpose |
|---|---|
| `open_session(anchor?, pack?, out?, force?)` | (Re)point the server's active session at a different anchor/pack/out without restart. With an empty payload AND a session already open, the call is a **no-op** (preserves placements) — pass `force=true` to recreate. |
| `reload_pack_semantics()` | Re-read `pack.json` and rebuild the runtime surface registry over current placements WITHOUT dropping the spatial index. Use after editing `pack.json` mid-session. |
| `close_session()` | Drop the active session; subsequent tool calls error until next `open_session`. Does NOT exit the server. |
| `shutdown(graceSeconds?)` | Exit the server process. Call at the end of a task. See Step 7. |
| `query_stage_graph(archetype?)` | List every current placement |
| `query_nearby(posM, radiusM, archetype?, limit?)` | Placements within radius, sorted nearest-first |
| `query_collision(posM, slotM, yawDeg?)` | 3D OBB-SAT collision check (Z interval + yaw-aware XY SAT) |
| `query_zones()` | Available zones with bounds + world origin (merges spatial zones + LLM-authored zones) |
| `query_source_hierarchy(prefix?, maxDepth?)` | Tree implied by every placement's `sourcePath`. Per node: `children`, `placementIds`, `archetypeCounts`, `bboxM`, `depth`. LLM uses this to identify zone candidates from the absorbed source's structure. |
| `create_zone(id, boundsM?, originWorldM?, purpose?, allowedArchetypes?, attachFromSourcePathPrefix?)` | Author a zone. With `attachFromSourcePathPrefix`, every placement whose `sourcePath` starts with that prefix gets `parentZoneId` set to `id`. `boundsM`/`originWorldM` default to the bbox of matched placements. |
| `query_pack_archetypes()` | Asset pack's archetype list |
| `query_surfaces(regionId?, minFreeAreaM2?, label?)` | Surfaces available for `place_on` |
| `query_archetype_semantics(archetype?)` | Per-archetype description / typicalContext / preferredNear / avoidNear |
| `query_pack_assets(archetype?)` | Specific USD candidate paths per archetype |
| `query_template_placements(archetype?, filledOnly?, unfilledOnly?)` | Template-defined positions to bind assets to |
| `place(archetype, posM, slotM, yawDeg?, id?, assetPath?, scaleM?, parentZoneId?)` | Place an asset on the floor; rejected on collision. `slotM` is local-pre-yaw. |
| `place_on(parentPlacementId, surfaceLabel, localXY, yawDeg, archetype, archetypeSizeM, regionId?, surfaceIndex?, topWorldZ?)` | Parent–child placement onto an existing placement's declared surface. Pass `surfaceIndex` or `topWorldZ` when the parent has multiple surfaces with the same label. |
| `place_many(placements, onCollision?)` | Batch place |
| `update_placement(id, assetPath?, scaleM?, yawDeg?)` | Modify an existing placement's binding |
| `bind_archetype(archetype, assetPath, scaleM?, fitMode?)` | Bind every placement of an archetype to one specific asset |
| `remove(id)` | Remove a placement |
| `list_recent_events(n?)` | Tail of session events |
| `session_status()` | Counts + footprint summary |
| `template_view()` | Site shells + zones with world origins/bounds (read BEFORE placing in empty-canvas mode) |
| `realize()` | Snapshot + author root.usd |
| `realize_passthrough(targetComposedPrimCount?, absorbedTemplate?)` | Tile the pack's rootStage USD to hit a prim count target |

## Conventions and known limitations

- **Coordinate system**: Z-up, meters. `posM = [x, y, z]` is the slot's center-floor in the parent zone's frame.
- **`yawDeg`**: rotation about Z, degrees.
- **`slotM` is in the placement's LOCAL (pre-yaw) frame**, NOT world axes. When `yawDeg != 0`, the OBB is rotated about Z at collision time. Worked example: a placement with native bbox `3.49 × 1.2 × 3.0` that should run "long-edge along Y" needs `yawDeg=90` AND `slotM=[3.7, 1.3, 3.1]` (NOT `[1.3, 3.7, 3.1]`). After yaw=90 the long local-X axis rotates onto world-Y; world coverage becomes `1.3 (X) × 3.7 (Y)`. Getting this backwards silently inflates the OBB by the ratio of long/short edge.
- **Slot vs asset**: slot is the *reservation*; the matcher picks a candidate whose bbox fits `slot × marginFactor` (default 1.05). With pivot correction, the picked asset's bbox-center-floor lands at `posM`.
- **Anchor-positions convention**: absorbed placements use the source's **asset-origin** position, not bbox-center-floor. There's a small drift between collision check and rendered position. To be addressed in a future absorb pass.
- **Collision gate is 3D OBB-SAT**: Z interval test + yaw-aware 2D SAT on XY. Stacked-on-Z placements work as long as `posM[2]` puts the placement's Z extent fully above the surface beneath it. The placement still has to *exist over a surface* — the engine doesn't conjure floor under elevated placements.
- **Multi-surface owners**: an owner can declare several surfaces with the same label. `query_surfaces` returns one entry per surface with a unique `surfaceId` and `labelIndex`. When calling `place_on`, pass `surfaceIndex` (0-based) or `topWorldZ` to pick a specific one — without a disambiguator the call is rejected with `stage="ambiguous_surface"`. `surface_fill.py` forwards `topWorldZ` automatically.
- **`open_session({})` with no payload is a no-op when a session is already open** (preserves placements). Use `reload_pack_semantics()` to re-read pack.json without dropping state; pass `force=true` to recreate the session in-place.
- **`remove(id)` evicts the placement AND its slot on any parent surface**. Removing a child placed via `place_on` therefore frees the spot for future `place_on` calls. Removing a parent drops every surface it owned along with their occupants.
- **Two-pack scenarios**: the skill doesn't yet support pulling archetypes from multiple packs in one session; future extension.

## Reference

- MCP wiring: `references/mcp_setup.md`
- Tool reference: `references/tools.md`
- Session HTTP API: `references/session_api.md`
- Pack-semantics vocabulary: `references/pack_semantics_vocabulary.json`
- Underlying engine: `$SKILL_DIR/engine/{spatial_index,session,session_http,mcp_server,viz_web}.py`
