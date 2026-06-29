# session_http API reference

Authoritative schema for every `POST /tool/<name>` endpoint plus the two
HTTP-only `GET` routes. **Read this before writing inline Python or curl
against the session — the most common scripting failure is calling
`.get(...)` on an endpoint that returns a list instead of a dict, or
`len(...)` on a count that comes back as an int.**

The base URL is `http://127.0.0.1:$SESSION_PORT/` where `$SESSION_PORT`
is the chosen port from `run.sh`'s launch JSON (default 8766, auto-bumped
if taken). Every `POST` body is `application/json`. Every response is
JSON.

Endpoints below are grouped by return shape so the "is this a list or a
dict?" question is one column-scan away.

---

## Returns a **dict** (object)

### `GET /status`
**Body:** none.
**Returns:** `{placementCount, anchorPlacementCount, addedPlacementCount, zoneCount, archetypes: [<name>, ...], footprintM: {minX, maxX, minY, maxY} | null, outDir}`.

### `GET /template`
**Body:** none. Convenience alias for `template_view`.

### `GET /tools`
**Body:** none.
**Returns:** **list** of tool names (string).

### `POST /tool/open_session`
**Body:** `{anchor?: str|null, pack?: str|null, out?: str|null, force?: bool}` — all optional; empty body with an already-open session is a no-op, returns `{ok, note, ...}`.
**Returns:** `{ok, anchor, pack, out, placementCount, zoneCount, archetypeCount, surfaceCount}`.

### `POST /tool/close_session`
**Body:** `{}`.
**Returns:** `{ok}`.

### `POST /tool/shutdown`
**Body:** `{graceSeconds?: float}` (default 0.2).
**Returns:** `{ok, pid, graceSeconds, note}`. Server then exits ~`graceSeconds` later via SIGTERM.

### `POST /tool/reload_pack_semantics`
**Body:** `{}`.
**Returns:** `{ok, placementCount, surfaceCount}` or `{ok: false, error}`. Rebuilds the runtime surface registry from the current placements + the latest `pack.json` on disk — use after editing `pack.json` mid-session.

### `POST /tool/template_view`
**Body:** `{}`.
**Returns:** `{shells: [{ownerId, originWorldM: [x,y,z], boundsM: {widthM, depthM, heightM}}], zones: [...], footprintM: {minX, maxX, minY, maxY} | null}`.

### `POST /tool/query_collision`
**Body:** `{posM: [x, y, z], slotM: [w, d, h], yawDeg?: float}`.
**Returns:** `{ok: bool, colliders: [{id, archetype, overlapM2}]}` — `ok=true` means no overlap.

### `POST /tool/query_archetype_semantics`
**Body:** `{archetype?: str}` — omit to get the whole map.
**Returns:** `{<archetype>: {description, preferredNear: [...], avoidNear: [...], anchors?, surfaces?, affordances?, placementBias?, ...}}`. Always a dict, keyed by archetype name. Single-archetype queries still return `{<name>: {...}}`.

### `POST /tool/place`
**Body:** `{archetype, posM: [x,y,z], slotM: [w,d,h], yawDeg?: float, id?: str, onCollision?: "reject"|"skip"|"force", assetPath?: str, scaleM?: [sx, sy, sz], parentZoneId?: str}`.
**Returns on success:** `{ok: true, id}`.
**Returns on rejection:** `{ok: false, rejected: {reason: "out_of_bounds"|"collision"|"region_filter", ...detail}}`.

### `POST /tool/place_many`
**Body:** `{placements: [{archetype, posM, slotM, yawDeg?, id?, parentZoneId?, assetPath?, scaleM?}, ...], onCollision?: "reject"|"skip"|"force"}`.
**Returns:** `{placed: int, rejected: int, total: int, results: [<per-placement-result>]}`. **`placed` and `rejected` are integers — do not call `len()` on them.** Use `results` (list) to iterate per-item outcomes.

### `POST /tool/place_on`
**Body:** `{parentPlacementId, surfaceLabel, localXY: [x, y], yawDeg?: float, archetype, archetypeSizeM: [w, d, h], regionId?: str, placementId?: str, surfaceIndex?: int, topWorldZ?: float}`.
**Returns on success:** `{ok: true, placement: {id, archetype, parentPlacement, parentSurface, parentSurfaceIndex, parentSurfaceId, localXY, yawDeg, posM, slotM}}`.
**Returns on rejection:** `{ok: false, rejected: {reason: "lookup"|"region_filter"|"edge_check"|"sibling_collision"|"ambiguous_surface", stage, ...}}`.

### `POST /tool/update_placement`
**Body:** `{id, assetPath?, scaleM?: [sx,sy,sz], yawDeg?: float}`.
**Returns:** `{ok, id, applied: {<field>: <value>}}`.

### `POST /tool/bind_archetype`
**Body:** `{archetype, assetPath, scaleM?, fitMode?: "explicit"|"fit-to-slot"}`.
**Returns:** `{ok, archetype, assetPath, fitMode, assetBboxWorldM: [w,d,h]|null, placementsUpdated: int, surfaceRebind: {entryFound, refreshed, surfacesAdded, entryPath} | null}`.

### `POST /tool/remove`
**Body:** `{id}`.
**Returns:** `{ok: bool}`.

### `POST /tool/close`
**Body:** `{}`.
**Returns:** `{ok, finalPlacementCount}`.

### `POST /tool/realize`
**Body:** `{onMiss?: "skip"|"synth"|"abort", assetPackOverride?: str, compositionStrategy?: dict}`.
**Returns:** `{ok, rootUsd, snapshot, manifest, summary: {byArchetype, unfilled}, templateBreakdown: {anchorPlacementCount, addedPlacementCount, totalPlacementCount}}`. **`manifest` is a path string**, not the parsed manifest; `json.load(open(d["manifest"]))` to read fields like `composedPrimCount_usdcore`, `compositionStrategy`, `layers`.

### `POST /tool/realize_passthrough`
**Body:** `{targetComposedPrimCount?: int, absorbedTemplate?: str}`.
**Returns:** same shape as `/tool/realize`.

---

## Returns a **list** (array) — `.get(...)` on these is the recurring bug

### `POST /tool/query_stage_graph`
**Body:** `{archetype?: str}` — omit for all placements.
**Returns:** **list** of `{id, archetype, posM: [x,y,z], slotM: {widthM, depthM, heightM}, yawDeg, assetPath?, scaleM?, parentZoneId?}`.

```bash
# CORRECT
placements = _http(BASE + "/tool/query_stage_graph", {})
for p in placements:
    print(p["id"], p["archetype"])

# WRONG — AttributeError: 'list' object has no attribute 'get'
placements = _http(BASE + "/tool/query_stage_graph", {})
for pid, p in placements.items():
    ...
```

### `POST /tool/query_nearby`
**Body:** `{posM: [x,y,z], radiusM: float, archetype?: str, limit?: int}`.
**Returns:** **list** of `{id, archetype, posM, slotM, yawDeg, distance}` sorted nearest-first.

### `POST /tool/query_zones`
**Body:** `{}`.
**Returns:** **list** of `{id, boundsM: {widthM, depthM, heightM}, originWorldM: [x,y,z], type?, purpose?, allowedArchetypes?, observedArchetypes?, name?, footprintM?}`. Merges spatial zones with templateZones from `extract_regions`.

### `POST /tool/query_pack_archetypes`
**Body:** `{}`.
**Returns:** **list** of `{archetype, candidateCount, samplePath, semantics: {...}}`. The `semantics` block now comes from `pack.json` (post-Stage-1 cleanup); archetypes with no `pack.semantics` block return the "no semantics declared" placeholder.

### `POST /tool/query_pack_assets`
**Body:** `{archetype?: str}` — omit for all.
**Returns:** **dict** `{<archetype>: [{path, relPath, sizeM?: {widthM, depthM, heightM}}, ...]}`. *Exception to the rule:* this one IS a dict keyed by archetype because the LLM commonly wants `assets["rack"]`.

### `POST /tool/query_template_placements`
**Body:** `{archetype?, filledOnly?: bool, unfilledOnly?: bool}`.
**Returns:** **list** of `{id, archetype, posM, slotM, yawDeg, isFilled, assetPath?, scaleM?, parentZoneId?}`. Anchor-template placements only (not free additions).

### `POST /tool/query_surfaces`
**Body:** `{regionId?: str, minFreeAreaM2?: float, label?: str}`.
**Returns:** **list** of `{owner, ownerArchetype, label, labelIndex, surfaceId, centerWorldXY: [x,y], topWorldZ, footprintM: [w,d], freeAreaM2, ownerYawDeg}`. Filter empties when nothing matches.

### `POST /tool/list_recent_events`
**Body:** `{n?: int}` (default 20).
**Returns:** **list** of event records (timestamped JSON lines from `events.jsonl`).

---

## Common LLM gotchas

| What I see | Fix |
|---|---|
| `'list' object has no attribute 'get'` | The endpoint returns a list. Iterate it, don't `.get` it. Cross-check this doc's section header. |
| `'int' object is not subscriptable` / `has no len()` | `placed`/`rejected`/`total` in `place_many`'s response are integers. Use `results` (list) for per-item iteration. |
| `KeyError: 'archetype'` from `query_pack_assets[k]` | `query_pack_assets` returns a dict keyed by archetype name; you probably want `query_pack_archetypes` (list of archetype summaries) or `query_pack_assets(archetype=X)`. |
| `400 missing field: 'X'` from `urlopen` | The endpoint requires field X. Required fields are bracketed in this doc as `body["X"]` in the corresponding `engine/session_http.py` TOOLS line; defaults are listed via `.get`. |
| `500 ValueError: …` from `urlopen` | The Session method raised. Check `<out_dir>/_session.log` (if launched via `run.sh`) or the server process's stderr. |
| First call after launch errors `no sketch session is open` | The HTTP server is running but no anchor/pack is bound yet. Either `open_session` first or include `--anchor` / `--pack` on `session_http.py`'s launch args. |

## When to look at the response shape

Every `_http` helper in `scripts/random_fill.py`, `scripts/surface_fill.py`, and `scripts/densify_zones.py` already calls `json.loads(response.read())` and returns the parsed payload. So `response` is **whatever this doc says the endpoint returns** — no `.json()` call, no `.read()` call needed.

If you're writing an inline heredoc against `urllib.request.urlopen` directly, the same applies — parse with `json.load(urlopen(req))` and treat the result per the doc.
