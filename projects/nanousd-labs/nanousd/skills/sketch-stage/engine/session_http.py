# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP wrapper around Session — same tool surface as the MCP server.

Lets a non-MCP client (curl, the calling Claude Code session itself, a
shell script) drive the session while the viz tails events.jsonl. Stays
alive between tool calls so the in-memory PlacementIndex persists.

POST /tool/<name>  with JSON body  → JSON response
GET  /status                       → counts + footprint
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import socket
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Per-process defaults; use the OS temp dir so paths resolve correctly
# across platforms. Override via env / --out / --port.
_DEFAULT_SESSION_OUT = str(Path(tempfile.gettempdir()) / "sketch_session")
_DEFAULT_LIVE_ROOT = Path(tempfile.gettempdir()) / "sketch_live"

sys.path.insert(0, str(Path(__file__).parent))
from session import Session  # noqa: E402


_session: Session | None = None
_anchor: str = ""
_pack: str = ""
_out: str = ""


def _sess() -> Session:
    """Return the active Session, auto-opening from launch-time defaults if
    one hasn't been opened explicitly via the `open_session` tool. Raises
    if neither path is configured.
    """
    global _session
    if _session is not None:
        return _session
    if not (_anchor or _pack):
        raise RuntimeError(
            "no sketch session is open; POST /tool/open_session "
            "{\"anchor\":..., \"pack\":..., \"out\":...} first, or launch "
            "session_http.py with --anchor / --pack to set defaults.")
    _session = Session(Path(_anchor) if _anchor else None,
                        Path(_pack) if _pack else None,
                        Path(_out))
    return _session


def _open(anchor: str | None, pack: str | None, out: str | None,
          force: bool = False) -> dict:
    """Open (or replace) the active session with explicit paths. Used as
    the body of the `open_session` HTTP tool and as a helper if other
    code needs to swap session in-place.

    Safeguard: when an empty payload (anchor=pack=out=None) is sent and
    a session is already open, this is treated as a no-op and the current
    session is preserved. Wiping live placement state on an accidental
    `open_session({})` was a footgun. Pass `force=True` to re-open with
    the existing defaults (rare; use `reload_pack_semantics` instead if
    you only want pack.json to be re-read)."""
    global _session, _anchor, _pack, _out
    if (_session is not None and anchor is None and pack is None
            and out is None and not force):
        return {
            "ok": True,
            "note": "no-op: empty payload with an already-open session "
                    "(use reload_pack_semantics to refresh pack.json, or "
                    "pass force=true to recreate)",
            "anchor": _anchor, "pack": _pack, "out": _out,
            "placementCount": len(_session.index),
            "zoneCount": len(_session.zones),
            "archetypeCount": len(_session._archetype_meta),
            "surfaceCount": (len(_session.surface_registry)
                             if _session.surface_registry else 0),
        }
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None
    if anchor is not None:
        _anchor = anchor
    if pack is not None:
        _pack = pack
    if out is not None:
        _out = out
    _session = Session(Path(_anchor) if _anchor else None,
                        Path(_pack) if _pack else None,
                        Path(_out) if _out else Path(_DEFAULT_SESSION_OUT))
    return {
        "ok": True,
        "anchor": _anchor, "pack": _pack, "out": _out,
        "placementCount": len(_session.index),
        "zoneCount": len(_session.zones),
        "archetypeCount": len(_session._archetype_meta),
        "surfaceCount": (len(_session.surface_registry)
                         if _session.surface_registry else 0),
    }


def _reload_pack_semantics() -> dict:
    """Re-read the pack's pack.json and rebuild the surface registry
    from current placements WITHOUT dropping the spatial index. Use
    after editing pack.json (e.g. fixing surface localTopZ or adding new
    archetypes) to pick up the changes while a long-running session is
    live. Anchor and existing placements are preserved.
    """
    if _session is None:
        return {"ok": False, "error": "no session open"}
    if not _pack:
        return {"ok": False, "error": "no pack path configured"}
    # Rebuild the surface registry from the current spatial index.
    try:
        from surface_registry import SurfaceRegistry  # type: ignore
        placements = _session.index.all()
        _session.surface_registry = SurfaceRegistry.from_pack_and_placements(
            Path(_pack), placements)
        return {
            "ok": True,
            "placementCount": len(_session.index),
            "surfaceCount": (len(_session.surface_registry)
                             if _session.surface_registry else 0),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _close() -> dict:
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
    _session = None
    return {"ok": True}


def _shutdown(grace_seconds: float = 0.2) -> dict:
    """Tell the HTTP server process to exit cleanly. Used by the skill's
    teardown step after a task finishes — leaving the server running ties
    up the port forever otherwise. We schedule the exit in a background
    thread so the response gets sent before SIGTERM fires.
    """
    _close()
    import os
    import signal
    import threading
    pid = os.getpid()

    def _exit_soon():
        import time
        time.sleep(max(0.05, grace_seconds))
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            os._exit(0)

    threading.Thread(target=_exit_soon, daemon=True).start()
    return {"ok": True, "pid": pid, "graceSeconds": grace_seconds,
            "note": "server will exit shortly; this connection may be the last"}


TOOLS = {
    "query_stage_graph": lambda b: _sess().query_stage_graph(archetype=b.get("archetype")),
    "query_nearby": lambda b: _sess().query_nearby(b["posM"], b["radiusM"],
                                                    b.get("archetype"), b.get("limit", 50)),
    "query_collision": lambda b: _sess().query_collision(b["posM"], b["slotM"], b.get("yawDeg", 0.0)),
    "query_zones": lambda b: _sess().query_zones(),
    "query_source_hierarchy": lambda b: _sess().query_source_hierarchy(
        b.get("prefix"), b.get("maxDepth")),
    "create_zone": lambda b: _sess().create_zone(
        b["id"], b.get("boundsM"), b.get("originWorldM"),
        b.get("purpose"), b.get("allowedArchetypes"),
        b.get("attachFromSourcePathPrefix"),
        b.get("attachByArchetypePrefix"),
        b.get("attachPlacementIds")),
    "query_pack_archetypes": lambda b: _sess().query_pack_archetypes(),
    "place": lambda b: _sess().place(b["archetype"], b["posM"], b["slotM"],
                                      b.get("yawDeg", 0.0), b.get("id"),
                                      b.get("onCollision", "reject"),
                                      b.get("assetPath"), b.get("scaleM"),
                                      b.get("parentZoneId")),
    "query_pack_assets": lambda b: _sess().query_pack_assets(b.get("archetype")),
    "query_template_placements": lambda b: _sess().query_template_placements(
        b.get("archetype"), b.get("filledOnly", False), b.get("unfilledOnly", False)),
    "update_placement": lambda b: _sess().update_placement(
        b["id"], b.get("assetPath"), b.get("scaleM"), b.get("yawDeg")),
    "bind_archetype": lambda b: _sess().bind_archetype(
        b["archetype"], b["assetPath"], b.get("scaleM"),
        b.get("fitMode", "explicit")),
    "remove": lambda b: _sess().remove(b["id"]),
    "list_recent_events": lambda b: _sess().list_recent_events(b.get("n", 20)),
    "close": lambda b: _sess().close(),
    "realize": lambda b: _sess().realize(b.get("onMiss"),
                                         b.get("assetPackOverride"),
                                         b.get("compositionStrategy")),
    "realize_passthrough": lambda b: _sess().realize_passthrough(
        b.get("targetComposedPrimCount", 1000), b.get("absorbedTemplate")),
    "template_view": lambda b: _sess().template_view(),
    "query_archetype_semantics": lambda b: _sess().query_archetype_semantics(b.get("archetype")),
    "place_many": lambda b: _sess().place_many(b["placements"], b.get("onCollision", "reject")),
    "query_surfaces": lambda b: _sess().query_surfaces(
        b.get("regionId"), b.get("minFreeAreaM2", 0.0), b.get("label")),
    "place_on": lambda b: _sess().place_on(
        b["parentPlacementId"], b["surfaceLabel"], b["localXY"],
        b.get("yawDeg", 0.0), b["archetype"], b["archetypeSizeM"],
        b.get("regionId"), b.get("placementId"),
        b.get("surfaceIndex"), b.get("topWorldZ")),
    "open_session": lambda b: _open(b.get("anchor"), b.get("pack"), b.get("out"),
                                     bool(b.get("force", False))),
    "reload_pack_semantics": lambda b: _reload_pack_semantics(),
    "close_session": lambda b: _close(),
    "shutdown": lambda b: _shutdown(b.get("graceSeconds", 0.2)),
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a, **k): pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send_json(self, status: int, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        # Prevent any browser from MIME-sniffing this JSON as HTML/script —
        # closes the XSS-via-JSON-response surface (pythonsecurity:S5131).
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            s = _sess()
            items = s.index.all()
            xs = [p.posM[0] for p in items]
            ys = [p.posM[1] for p in items]
            anchor_count = s.anchor_placement_count
            self._send_json(200, {
                "placementCount": len(items),
                "anchorPlacementCount": anchor_count,
                "addedPlacementCount": len(items) - anchor_count,
                "zoneCount": len(s.zones),
                "archetypes": list(s._archetype_meta.keys()),
                "footprintM": ({"minX": min(xs), "maxX": max(xs),
                                "minY": min(ys), "maxY": max(ys)} if items else None),
                "outDir": str(s.out_dir),
            })
            return
        if self.path == "/template":
            self._send_json(200, _sess().template_view())
            return
        if self.path == "/tools":
            self._send_json(200, list(TOOLS.keys()))
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/tool/"):
            self._send_json(404, {"error": "not found"})
            return
        tool = self.path[len("/tool/"):]
        if tool not in TOOLS:
            self._send_json(404, {"error": f"unknown tool: {tool}",
                                  "available": list(TOOLS.keys())})
            return
        n = int(self.headers.get("Content-Length", "0") or 0)
        body_raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            body = json.loads(body_raw.decode("utf-8")) if body_raw.strip() else {}
        except Exception as e:
            self._send_json(400, {"error": f"bad JSON: {e}"})
            return
        try:
            result = TOOLS[tool](body)
        except KeyError as e:
            self._send_json(400, {"error": f"missing field: {e}"})
            return
        except Exception as e:
            self._send_json(500, {"error": f"{type(e).__name__}: {e}"})
            return
        self._send_json(200, result)


def main() -> int:
    global _anchor, _pack, _out
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.environ.get("SKETCH_ANCHOR", ""))
    ap.add_argument("--pack", default=os.environ.get("SKETCH_PACK",
        os.environ.get("SKETCH_STAGE_DEFAULT_PACK", "")))
    ap.add_argument("--out", default=os.environ.get("SKETCH_OUT_DIR",
        str(Path.home() / ".cache" / "sketch-stage" / "demo")))
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    _anchor, _pack, _out = args.anchor, args.pack, args.out
    if not os.environ.get("SKETCH_LIVE_EVENTS"):
        live_dir = _DEFAULT_LIVE_ROOT / str(args.port)
        live_dir.mkdir(parents=True, exist_ok=True)
        os.environ["SKETCH_LIVE_EVENTS"] = str(live_dir / "events.jsonl")
    # IPv6 dual-stack so browser JS that fetches http://localhost:8766 over
    # IPv6 (::1) doesn't stall. Bind to :: with V6ONLY=0 so the same socket
    # accepts both v4 and v6 connects (incl. IPv4-mapped 127.0.0.1). This
    # also makes the session reachable on the LAN, mirroring viz_web's bind
    # — keeps the demo flow simple at the cost of removing the prior
    # "localhost only" guard.
    class _DualStack(HTTPServer):
        address_family = socket.AF_INET6
        def server_bind(self):
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except (AttributeError, OSError):
                pass
            super().server_bind()
    try:
        httpd = _DualStack(("::", args.port), _Handler)
    except OSError:
        httpd = HTTPServer(("0.0.0.0", args.port), _Handler)
    print(f"sketch session HTTP at http://127.0.0.1:{args.port}/  "
          f"anchor={_anchor}  pack={_pack}  out={_out}  "
          f"live={os.environ['SKETCH_LIVE_EVENTS']}", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
