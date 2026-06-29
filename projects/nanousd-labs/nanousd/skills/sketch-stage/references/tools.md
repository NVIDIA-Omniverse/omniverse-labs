# Tool reference

All tools are available over both **MCP** (`mcp__sketch-placement__*` once the server is registered in `~/.claude/settings.json`) and **HTTP** (`POST http://127.0.0.1:8766/tool/<name>` with JSON body when running locally without MCP).

## query_stage_graph

```json
{ "archetype": "<archetype>" }   // optional filter
```

→ array of `{ id, archetype, posM:[x,y,z], slotM:[w,d,h], yawDeg, parentZoneId? }`.

## query_nearby

```json
{ "posM": [10, 5, 0], "radiusM": 4.0, "archetype": "<archetype>", "limit": 20 }
```

→ same shape as `query_stage_graph` plus `distanceM`. Sorted nearest-first.

## query_collision

```json
{ "posM": [10, 5, 0], "slotM": [1.4, 1.2, 1.2] }
```

→ `{ ok: bool, colliders: [{ id, archetype, overlapM2 }] }`. Use **before** `place` to avoid wasted rejections.

## query_zones

→ array of `{ id, boundsM:{widthM,depthM,heightM}, originWorldM:[x,y,z], type, purpose?, allowedArchetypes? }`.

## query_pack_archetypes

→ array of `{ archetype, candidateCount, samplePath }`. The set of archetype names you can `place` with.

## place

```json
{
  "archetype": "<archetype>",
  "posM": [10.0, 5.0, 0.0],
  "slotM": [4.4, 2.3, 3.3],
  "yawDeg": 0.0,
  "id": "<archetype>_001",  // optional; auto-generated if omitted
  "parentZoneId": "<zone_id>"
}
```

→ on success: `{ ok: true, id }`. On collision: `{ ok: false, rejected: { reason: "collision", colliders } }`.

The slot's center-floor at `posM` is the anchor point; the realizer applies a corrective offset so the picked asset's bbox center-floor lands on this point regardless of the asset's authored origin.

When editing a DBT-imported sketch, always set `parentZoneId` to a zone returned by `query_zones()` so snapshots and realized USD keep the semantic zone grouping.

## remove

```json
{ "id": "<placement_id>" }
```

→ `{ ok: bool }`.

## list_recent_events

```json
{ "n": 20 }
```

→ array of recent events (placed/rejected/removed/session_open/session_close).

## session_status

→ `{ placementCount, zoneCount, archetypes, footprintM, outDir }`.

## Choosing slot dimensions

The "slot" is the placement's allotted volume; the realizer fits the picked asset into it (subject to `fitMode`). Pick the slot to match what the placement should *look like* at the canvas's scale, not the asset's raw bbox:

- Read each archetype's `size` from `pack.json` (or query via `query_pack_archetypes`) as a starting point.
- Tighten to the **small variant** of each archetype's range so back-to-back placements don't overlap when the realizer picks the larger variant.
- For empty-canvas / theme-driven work, override per-zone with the visibility-scale rule in SKILL.md ("Canvas-to-display ratio rule"): the largest slot dimension should be roughly 1–3% of the canvas's largest dimension.
- Bind with `bind_archetype(..., fitMode="fit-to-slot")` so the realizer scales the asset into the slot you chose, decoupling asset bbox from display size.
