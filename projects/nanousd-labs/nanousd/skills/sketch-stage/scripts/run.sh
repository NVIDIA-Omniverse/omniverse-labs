#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Set up an LLM-driven sketch-placement run.
# Starts a session HTTP + 2D/3D viz over the supplied pack/anchor/bounds,
# prints the configuration + scale targets the LLM should use.
#
# Usage:
#   run.sh --bounds WxD --pack /path/to/asset_pack --intent "<theme>" --out /path
#   run.sh --absorb-source /path/or/url.usd --pack /path/to/asset_pack --out /path
#   run.sh --resume /path/to/_snapshot.sketch.json --intent "..." --out /path
#
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SCALE=medium
MODE=auto
OUT="${HOME}/.cache/sketch-stage/demo"
INTENT=""
ABSORB_SOURCE=""
ABSORB_PACK=""
PACK_OVERRIDE=""
AUTO_GEN_PACK=0
RESUME_SNAPSHOT=""
BOUNDS=""    # e.g. "10000x10000" or "10000x10000x100" → empty canvas mode
SESSION_PORT=8766           # auto-bumped if taken (unless --session-port given explicitly)
VIZ_PORT=8765               # auto-bumped if taken (unless --viz-port given explicitly)
SESSION_PORT_EXPLICIT=0     # user passed --session-port?
VIZ_PORT_EXPLICIT=0         # user passed --viz-port?
KILL_EXISTING=0             # default: never kill another user's server. opt-in via --force-kill.

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scale)    SCALE="$2";    shift 2 ;;
    --mode)     MODE="$2";     shift 2 ;;
    --out)      OUT="$2";      shift 2 ;;
    --intent)   INTENT="$2";   shift 2 ;;
    --pack)            PACK_OVERRIDE="$2"; shift 2 ;;
    --auto-gen-pack)   AUTO_GEN_PACK=1;    shift ;;
    --resume)          RESUME_SNAPSHOT="$2"; shift 2 ;;
    --bounds)          BOUNDS="$2";       shift 2 ;;
    --absorb-source) ABSORB_SOURCE="$2"; shift 2 ;;
    --absorb-pack)   ABSORB_PACK="$2";   shift 2 ;;
    --session-port) SESSION_PORT="$2"; SESSION_PORT_EXPLICIT=1; shift 2 ;;
    --viz-port)     VIZ_PORT="$2";     VIZ_PORT_EXPLICIT=1;     shift 2 ;;
    --force-kill) KILL_EXISTING=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  auto|fill|tile|passthrough) ;;
  *) echo "unsupported mode: $MODE  (auto|fill|tile|passthrough)" >&2; exit 2 ;;
esac

# Optional absorb step: if --absorb-source given, run absorb_pack.py first
# and use the resulting sketch as the anchor template. The absorber is
# vocabulary-free — archetype labels come from the source stage itself
# (referenced asset filenames). The optional --absorb-pack tag is recorded
# in the sketch metadata so realize knows which pack to consult later.
if [[ -n "$ABSORB_SOURCE" ]]; then
  ABSORBED_OUT="${OUT}/absorbed.sketch.json"
  mkdir -p "$OUT"
  echo "[run.sh] absorbing $ABSORB_SOURCE → $ABSORBED_OUT" >&2
  EXTRA_ARGS=""
  [[ -n "$ABSORB_PACK" ]] && EXTRA_ARGS="$EXTRA_ARGS --target-pack $ABSORB_PACK"
  ABSORB_VENV="$(bash "$SKILL_DIR/scripts/setup_venv.sh")"
  "$ABSORB_VENV/bin/python" \
    $SKILL_DIR/engine/absorb_pack.py \
    --source "$ABSORB_SOURCE" --out "$ABSORBED_OUT" \
    $EXTRA_ARGS >&2
  ANCHOR_OVERRIDE="$ABSORBED_OUT"
else
  ANCHOR_OVERRIDE=""
fi

# The skill no longer ships any built-in scenario anchors. The caller
# supplies one of:
#   --absorb-source <url-or-path>   (anchor derived from a source USD)
#   --bounds WxD[xH]                (empty canvas of the given footprint)
#   --resume <_snapshot.sketch.json> (resume from a prior realize)
# At least one of these must be present, else there's nothing to populate.
ANCHOR=""
PACK=""
[[ -n "$ANCHOR_OVERRIDE" ]] && ANCHOR="$ANCHOR_OVERRIDE"
[[ -n "$ABSORB_PACK" ]] && PACK="$ABSORB_PACK"
[[ -n "$PACK_OVERRIDE" ]] && PACK="$PACK_OVERRIDE"

if [[ -z "$PACK" ]] && [[ -z "$ABSORB_PACK" ]] && [[ -z "$BOUNDS" ]] && [[ -z "$RESUME_SNAPSHOT" ]] && [[ -z "$ABSORB_SOURCE" ]]; then
  echo "[run.sh] nothing to populate. Pass one of:" >&2
  echo "         --absorb-source <usd>  (anchor from a source scene)" >&2
  echo "         --bounds WxD --pack <pack-root>  (empty-canvas mode)" >&2
  echo "         --resume <_snapshot.sketch.json>" >&2
  exit 2
fi
if [[ -z "$PACK" ]] && [[ -z "$RESUME_SNAPSHOT" ]]; then
  echo "[run.sh] no asset pack supplied. Pass --pack /path/to/asset_pack" >&2
  echo "         (or set SKETCH_STAGE_DEFAULT_PACK=/path in your environment)." >&2
  PACK="${SKETCH_STAGE_DEFAULT_PACK:-}"
  if [[ -z "$PACK" ]]; then exit 2; fi
fi

# --bounds WxD or WxDxH: empty-canvas mode. Synthesize an anchor with no
# placements, just a site shell of the requested size. The LLM then drives
# all population from `--intent` + the pack's archetype semantics.
if [[ -n "$BOUNDS" ]]; then
  W=$(echo "$BOUNDS" | cut -dx -f1)
  D=$(echo "$BOUNDS" | cut -dx -f2)
  H=$(echo "$BOUNDS" | cut -dx -f3)
  if [[ -z "$H" ]]; then
    # Default wall height scales with footprint: ~8% of the longer side,
    # clamped to [3 m, 16 m]. Avoids the old "always 100 m" default which
    # produced absurd 100 m walls for a small footprint.
    H=$(awk -v w="$W" -v d="$D" 'BEGIN {
      m = (w > d ? w : d);
      h = m * 0.08;
      if (h < 3) h = 3;
      if (h > 16) h = 16;
      printf "%.1f", h;
    }')
  fi
  if [[ -z "$W" ]] || [[ -z "$D" ]]; then
    echo "[run.sh] --bounds: expected WxD or WxDxH (in meters), got $BOUNDS" >&2
    exit 2
  fi
  if [[ -z "$PACK" ]]; then
    echo "[run.sh] --bounds: --pack is required for empty-canvas mode" >&2
    exit 2
  fi
  mkdir -p "$OUT"
  ANCHOR="$OUT/empty.sketch.json"
  cat > "$ANCHOR" <<JSON
{
  "schemaVersion": 1,
  "seed": 42,
  "assetPack": "$PACK",
  "tree": {
    "type": "site",
    "shell": {"boundsM": {"widthM": $W, "depthM": $D, "heightM": $H}},
    "children": []
  }
}
JSON
  echo "[run.sh] empty-canvas mode: ${W}x${D}x${H}m anchor at $ANCHOR  pack=$PACK" >&2
fi
# --resume: pick up where a previous run left off. Every realize() writes a
# _snapshot.sketch.json next to root.usd; pass that path here and the new
# session loads its full state (anchor + added placements + bound assets +
# scaleM + yaw + zones) as the starting point. Per-run --pack still overrides.
# Must run BEFORE the no-anchor check below so the snapshot can serve as the
# anchor (an explicit --absorb-source or --bounds still wins via the override).
if [[ -n "$RESUME_SNAPSHOT" ]]; then
  if [[ ! -f "$RESUME_SNAPSHOT" ]]; then
    echo "[run.sh] --resume: snapshot not found: $RESUME_SNAPSHOT" >&2
    exit 2
  fi
  [[ -z "$ANCHOR" ]] && ANCHOR="$RESUME_SNAPSHOT"
  if [[ -z "$PACK_OVERRIDE" ]]; then
    SNAP_PACK=$("$($SKILL_DIR/scripts/setup_venv.sh)/bin/python" -c "import json,sys; print(json.load(open('$RESUME_SNAPSHOT')).get('assetPack',''))" 2>/dev/null)
    [[ -n "$SNAP_PACK" ]] && PACK="$SNAP_PACK"
  fi
  echo "[run.sh] resuming from $RESUME_SNAPSHOT  pack=$PACK" >&2
fi
if [[ -z "$ANCHOR" ]]; then
  echo "[run.sh] no anchor: pass --absorb-source, --resume, or --bounds (with --pack)" >&2
  exit 2
fi

# Auto-generate pack.json if missing (or always when --auto-gen-pack is set
# and the pack is a raw asset directory). Trusted source for upAxis/mpu is
# the asset stages themselves, not any pack-level guess.
if [[ ! -f "$PACK/pack.json" ]] || [[ "$AUTO_GEN_PACK" = "1" ]]; then
  echo "[run.sh] no pack.json at $PACK (or --auto-gen-pack); generating..." >&2
  GEN_VENV="$(bash "$SKILL_DIR/scripts/setup_venv.sh")"
  GEN_FLAGS=""
  [[ "$AUTO_GEN_PACK" = "1" ]] && GEN_FLAGS="--force"
  "$GEN_VENV/bin/python" "$SKILL_DIR/scripts/gen_pack_json.py" "$PACK" \
      $GEN_FLAGS >&2 || {
    echo "[run.sh] gen_pack_json.py failed for $PACK" >&2
    exit 3
  }
fi

# scale → target ADDITIONAL placement count (LLM decides how to distribute)
case "$SCALE" in
  small)  ADD_TARGET=30;   ;;
  medium) ADD_TARGET=120;  ;;
  large)  ADD_TARGET=600;  ;;
  huge)   ADD_TARGET=2400; ;;
  *) echo "unsupported scale: $SCALE  (small|medium|large|huge)" >&2; exit 2 ;;
esac

mkdir -p "$OUT"

# Port resolution: when the user didn't pass --session-port / --viz-port
# explicitly, auto-bump to the next free port starting at the default. This
# lets two concurrent run.sh invocations get isolated servers instead of
# colliding on 8766/8765.
_find_free_port() {
  # Find a port that's neither currently listening nor in the caller-
  # supplied exclude list. The exclude list avoids the
  # session-port-vs-viz-port race where session_http hasn't bound yet
  # when viz_web picks its port, and both end up with the same number.
  local start_port="$1" max_port="$2" exclude="${3:-}"
  local p
  for p in $(seq "$start_port" "$max_port"); do
    case " $exclude " in *" $p "*) continue ;; esac
    if ! ss -tnlH "( sport = :$p )" 2>/dev/null | grep -q "LISTEN"; then
      echo "$p"
      return 0
    fi
  done
  echo "$start_port"   # fall through to default; the bind itself will fail loudly
  return 1
}

if [[ "$KILL_EXISTING" = "1" ]]; then
  ps aux | grep -E "session_http.py --port $SESSION_PORT" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null || true
  ps aux | grep -E "viz_web.py .* --port $VIZ_PORT"      | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null || true
  sleep 0.5
fi

if [[ "$SESSION_PORT_EXPLICIT" = "0" ]]; then
  NEW_SESSION_PORT="$(_find_free_port "$SESSION_PORT" $((SESSION_PORT + 32)))"
  if [[ "$NEW_SESSION_PORT" != "$SESSION_PORT" ]]; then
    echo "  (session port $SESSION_PORT taken; using $NEW_SESSION_PORT)" >&2
    SESSION_PORT="$NEW_SESSION_PORT"
  fi
fi
if [[ "$VIZ_PORT_EXPLICIT" = "0" ]]; then
  # Exclude SESSION_PORT — session_http hasn't started yet, so ss won't
  # see it; without this exclude the viz can pick the same port and
  # silently die on bind.
  NEW_VIZ_PORT="$(_find_free_port "$VIZ_PORT" $((VIZ_PORT + 32)) "$SESSION_PORT")"
  if [[ "$NEW_VIZ_PORT" != "$VIZ_PORT" ]]; then
    echo "  (viz port $VIZ_PORT taken; using $NEW_VIZ_PORT)" >&2
    VIZ_PORT="$NEW_VIZ_PORT"
  fi
fi

# Per-session live-event log path so concurrent sessions don't clobber
# each other. Defaults under $TMPDIR (resolves on Linux/macOS/Windows).
# Caller may override by exporting SKETCH_LIVE_EVENTS before invoking run.sh.
SKETCH_LIVE_ROOT="${SKETCH_LIVE_ROOT:-${TMPDIR:-/tmp}/sketch_live}"
: "${SKETCH_LIVE_EVENTS:=$SKETCH_LIVE_ROOT/$SESSION_PORT/events.jsonl}"
export SKETCH_LIVE_EVENTS

# fresh event log
> "$OUT/events.jsonl"

VENV="$(bash "$SKILL_DIR/scripts/setup_venv.sh")"
INCR=$SKILL_DIR/engine

"$VENV/bin/python" "$INCR/session_http.py" --port "$SESSION_PORT" \
    --anchor "$ANCHOR" --pack "$PACK" --out "$OUT" >> "$OUT/_session.log" 2>&1 &
SESSION_PID=$!

"$VENV/bin/python" "$INCR/viz_web.py" --port "$VIZ_PORT" --session-port "$SESSION_PORT" \
    --follow-latest "$SKETCH_LIVE_ROOT" \
    >> "$OUT/_viz.log" 2>&1 &
VIZ_PID=$!

sleep 1.2
ANCHOR_COUNT=$(curl -s "http://127.0.0.1:$SESSION_PORT/status" | python3 -c "import json,sys; print(json.load(sys.stdin).get('placementCount', '?'))" 2>/dev/null || echo "?")

cat <<EOF
{
  "scale": "$SCALE",
  "mode": "$MODE",
  "intent": "$INTENT",
  "anchor": "$ANCHOR",
  "anchorPlacementCount": $ANCHOR_COUNT,
  "assetPack": "$PACK",
  "outDir": "$OUT",
  "addTarget": $ADD_TARGET,
  "session": {
    "pid": $SESSION_PID,
    "url": "http://127.0.0.1:$SESSION_PORT/"
  },
  "viz": {
    "pid": $VIZ_PID,
    "url": "http://localhost:$VIZ_PORT/"
  }
}
EOF
