#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# validate_geo_cache.sh — correctness gate for the NUSD_GEO_CACHE geometry cache.
#
# Proves the cache round-trip is bit-exact and the cache actually engages:
#   A  cold  (NUSD_GEO_CACHE=1) — must MISS and write the cache
#   B  warm  (NUSD_GEO_CACHE=1) — must HIT and skip the USD parse
#   C  off   (env unset)        — geo_cache must be fully inert
#   D  stale (after `touch`)    — must MISS again (invalidation)
# A and B renders must be byte-identical (B reconstructs A's cached scene); a
# scene's USD parse can be nondeterministic, so run C is the inertness check,
# not a byte comparison. Any failure exits non-zero.
#
# The autonomous meshlet-cache loop runs this before every commit — a non-zero
# exit means DO NOT COMMIT. See docs/planning/MESHLET_GEOMETRY_CACHE_PLAN.md.
#
# Usage: ./scripts/validate_geo_cache.sh [scene.usd]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="$(cd "$REPO/.." && pwd)"

# Default scene is the full DSX assembly (~128 K meshes) — the real workload.
# Override with $1 for a different scene.
SCENE="${1:-$HOME/dsx-assets/dsx_dataset_2.1/DSX_BP_/DSX_BP/Data_Center/Assembly_HAC_GPU_BLDG.usd}"
BIN="$REPO/build/test_headless_render"
OUT="$(mktemp -d)"
export DISPLAY="${DISPLAY:-:1}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

fail() { echo "VALIDATE FAIL: $*" >&2; rm -rf "$OUT"; exit 1; }
md5()  { md5sum "$1" | awk '{print $1}'; }

[[ -x "$BIN" ]]    || fail "test_headless_render not built — run ./build.sh first"
[[ -f "$SCENE" ]]  || fail "scene not found: $SCENE"

CACHE="$SCENE.nzgeo"
rm -f "$CACHE"

# A — cold: cache miss, writes the cache.
NUSD_GEO_CACHE=1 NUSD_LOAD_TIMING=1 "$BIN" "$SCENE" "$OUT/a.ppm" \
    > "$OUT/a.log" 2>&1 || fail "run A (cold) render failed — see $OUT/a.log"
grep -q '\[geo_cache\] MISS — wrote cache' "$OUT/a.log" \
    || fail "run A did not write the cache"
[[ -f "$CACHE" ]] || fail "cache file $CACHE was not created"
grep -qE '\[geo_cache\] meshlets built: [1-9]' "$OUT/a.log" \
    || fail "run A built 0 meshlets — meshlet preprocessing skipped"

# B — warm: cache hit, no USD parse.
NUSD_GEO_CACHE=1 NUSD_LOAD_TIMING=1 "$BIN" "$SCENE" "$OUT/b.ppm" \
    > "$OUT/b.log" 2>&1 || fail "run B (warm) render failed — see $OUT/b.log"
grep -q '\[geo_cache\] HIT' "$OUT/b.log" \
    || fail "run B did not hit the cache"
grep -q 'scene_timing.*nanousd_open' "$OUT/b.log" \
    && fail "run B still parsed USD — cache load path not taken"
grep -qE 'HIT:.* [1-9][0-9]* meshlets,' "$OUT/b.log" \
    || fail "run B loaded 0 meshlets — meshlet section not round-tripped"

# C — feature off: geo_cache must be fully inert.
"$BIN" "$SCENE" "$OUT/c.ppm" > "$OUT/c.log" 2>&1 \
    || fail "run C (feature off) render failed — see $OUT/c.log"
grep -q '\[geo_cache\]' "$OUT/c.log" \
    && fail "run C touched geo_cache with NUSD_GEO_CACHE unset"

# Byte-exact gate: the warm cache load must reproduce the cold parse render.
# B reconstructs A's exact cached Scene, so this is deterministic; an
# independent re-parse (run C) is not — see the header note.
MA=$(md5 "$OUT/a.ppm"); MB=$(md5 "$OUT/b.ppm")
[[ "$MA" = "$MB" ]] || fail "cache render != parse render ($MB != $MA)"

# D — stale invalidation: touching the source forces a fresh miss.
touch "$SCENE"
NUSD_GEO_CACHE=1 "$BIN" "$SCENE" "$OUT/d.ppm" \
    > "$OUT/d.log" 2>&1 || fail "run D (stale) render failed — see $OUT/d.log"
grep -q '\[geo_cache\] MISS' "$OUT/d.log" \
    || fail "stale cache not invalidated after touch"

# E — cook tool: geo_cook pre-warms a usable cache with no GPU/display.
rm -f "$CACHE"
"$REPO/build/geo_cook" "$SCENE" > "$OUT/e.log" 2>&1 \
    || fail "geo_cook failed — see $OUT/e.log"
grep -q 'geo_cook: OK' "$OUT/e.log" || fail "geo_cook did not report OK"
[[ -f "$CACHE" ]] || fail "geo_cook did not write a cache"

HIT_MS=$(grep -o 'HIT:.*ms' "$OUT/b.log" | grep -o '[0-9.]* ms' | tail -1)
echo "VALIDATE OK: round-trip bit-exact (md5 $MA); cache HIT confirmed (${HIT_MS:-n/a}); invalidation works"
rm -rf "$OUT"
exit 0
