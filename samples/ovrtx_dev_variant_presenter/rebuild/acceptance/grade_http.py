# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Blind HTTP grader for a Dev Variant Presenter build.

Drives ONLY the public control contract (see
skills/omniverse-dev-variant-presenter/references/SPEC-FUNCTIONAL.md) — never the
app's internals. Scores the headless-checkable parts of the 9 feature areas plus
the USD-safety gate. The browser/pixel parts (area 1, pixel confirms, "no client
3D" gate) are graded separately by the sibling verify_browser.cjs, which drives a
real mouse/keyboard in headful Chrome against the same running server.

The app must already be running (preferred — manager owns GPU lifecycle):
    python grade_http.py --url http://127.0.0.1:8090 --usd <local .usd> [--render]
or let the grader launch + kill it (convenience / self-test) — the interpreter path is
`.venv/Scripts/python` on Windows, `.venv/bin/python` on Linux/macOS:
    python grade_http.py --launch ".venv/bin/python -m dev_variant_presenter --port 8090" \
        --cwd <repo> --usd <local .usd> [--render]

Design rules that keep grading safe + cheap:
  * count checks use 2-variant includes (queues at most 2-4 tiny renders), never
    a full set;
  * the explosion guard is tested WITHOUT confirm, so the huge job 409s and
    NOTHING is queued;
  * the real render-output check reuses the small one-at-a-time job's out_dir;
  * the remote-mirror probe runs LAST (it reopens to the tiny stage).
Stdlib only, so it runs under any interpreter.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import signal
import threading
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_mirror_probe import MirrorFixture  # noqa: E402


# ----------------------------- tiny HTTP client -----------------------------
def req(method, base, path, body=None, timeout=30):
    url = base.rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None), None
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            bd = json.loads(raw)
        except Exception:
            bd = raw.decode("utf-8", "replace")
        return e.code, bd, None
    except Exception as e:  # noqa: BLE001
        return None, None, str(e)


def fetch_raw(base, path, timeout=15):
    """GET raw bytes + status + content-type (for static assets, not JSON)."""
    url = base.rstrip("/") + (path if path.startswith("/") else "/" + path)
    r = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), (e.headers.get("Content-Type", "") if e.headers else "")
    except Exception as e:  # noqa: BLE001
        return None, str(e).encode(), ""


def grade_frontend_assets(base, card: Card):
    """Catch the DOA-frontend bug: index.html is served but its <script>/<link> assets 404
    because they're referenced at a different base than where they're mounted. An API-only
    grade (or 'the HTML contains app.js') passes while the whole UI is dead. Fetch every
    referenced asset and assert 200 + a JS/CSS content-type."""
    st_html, html_bytes, _ = fetch_raw(base, "/")
    html = html_bytes.decode("utf-8", "replace") if isinstance(html_bytes, (bytes, bytearray)) else ""
    srcs = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html, re.I)
    hrefs = re.findall(r'<link[^>]+href=["\']([^"\']+)["\']', html, re.I)
    assets = [a for a in (srcs + hrefs) if not a.startswith(("http://", "https://", "data:", "//"))]
    bad = []
    for a in assets:
        rel = a[1:] if a.startswith("./") else a            # ./app.js, /web/app.js, app.js all -> root-resolved
        sta, _raw, ct = fetch_raw(base, rel)
        stem = a.split("?")[0].lower()
        ctl = (ct or "").lower()
        ok_ct = (("javascript" in ctl or "ecmascript" in ctl) if stem.endswith(".js")
                 else ("css" in ctl if stem.endswith(".css") else True))
        if sta != 200 or not ok_ct:
            bad.append(f"{a}->{sta} ct={ct or '?'}")
    ok = st_html == 200 and bool(assets) and not bad
    detail = (f"{len(assets)} assets all 200" if ok
              else (f"NO <script> in / (st={st_html})" if not assets
                    else f"{len(bad)}/{len(assets)} BROKEN: " + " | ".join(bad)))
    card.add("frontend_assets", "every <script>/<link> in / serves 200 + right type", "full" if ok else "fail", detail)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grade_mirror_progress(base, area9_url, card: Card):
    """area9_mirror_progress — documents + checks the COLD-mirror progress contract.

    Contract: while the app mirrors a REMOTE closure for area9, it emits `mirror_progress`
    events over /events carrying a monotonic (non-decreasing) integer `downloaded` count, so a
    UI can show progress rather than a frozen window. To exercise a real DOWNLOAD we need the
    cache cold, so this best-effort deletes any `*.mirror_complete` marker under a known mirror
    dir before opening (env VS_MIRROR_DIR, if set). Strictly non-fatal: if no remote URL is
    given, or the remote/WS can't be reached, this marks `skip` (never `fail`) so it can't crash
    or false-fail a grader that simply has no network to the area9 source.
    """
    if not area9_url:
        card.add("area9_mirror_progress", "cold-mirror progress events", "skip",
                 "no --area9-url given (pass the remote area9 closure URL to exercise mirror_progress)")
        return
    try:
        # best-effort: clear a *.mirror_complete marker so the open re-downloads (cold mirror).
        import os
        mdir = os.environ.get("VS_MIRROR_DIR", "")
        cleared = 0
        if mdir and Path(mdir).is_dir():
            for m in Path(mdir).rglob("*.mirror_complete"):
                with contextlib.suppress(Exception):
                    m.unlink(); cleared += 1

        # collect mirror_progress events in a background thread while we kick the open.
        collected = {}

        def _match(ev):
            t = (ev.get("type") or ev.get("event") or "") if isinstance(ev, dict) else ""
            return "mirror_progress" in str(t) or ("downloaded" in ev if isinstance(ev, dict) else False)

        def _runner():
            evs, err = collect_events(base, duration=90, match=_match,
                                      stop_when=lambda e: len(e) >= 3)
            collected["events"], collected["err"] = evs, err

        th = threading.Thread(target=_runner, daemon=True)
        th.start()
        time.sleep(0.5)  # let the WS connect before the download starts
        st, _bd, err = req("POST", base, "/api/open", {"usd_path": area9_url}, timeout=300)
        th.join(timeout=95)
        evs = collected.get("events", [])
        werr = collected.get("err")

        if err or st is None:
            card.add("area9_mirror_progress", "cold-mirror progress events", "skip",
                     f"could not reach the remote area9 closure (open status={st}, err={err}); cleared={cleared}")
            return
        if werr and not evs:
            card.add("area9_mirror_progress", "cold-mirror progress events", "skip",
                     f"events WS unavailable ({werr}); open status={st}, cleared={cleared}")
            return

        # extract downloaded counts (accept top-level or nested under data/payload)
        def _dl(ev):
            for src in (ev, ev.get("data") if isinstance(ev.get("data"), dict) else None,
                        ev.get("payload") if isinstance(ev.get("payload"), dict) else None):
                if isinstance(src, dict) and isinstance(src.get("downloaded"), (int, float)):
                    return int(src["downloaded"])
            return None

        counts = [c for c in (_dl(e) for e in evs) if c is not None]
        nondec = all(counts[i] <= counts[i + 1] for i in range(len(counts) - 1)) if len(counts) >= 2 else True
        ok = len(counts) >= 1 and nondec
        card.add("area9_mirror_progress", "cold-mirror progress events",
                 "full" if ok else ("partial" if evs else "fail"),
                 f"open={st}, cleared_markers={cleared}, mirror_progress_events={len(evs)}, "
                 f"downloaded_seq={counts[:6]}, non_decreasing={nondec}")
    except Exception as e:  # noqa: BLE001 — never let this area crash the grader
        card.add("area9_mirror_progress", "cold-mirror progress events", "skip",
                 f"probe raised (non-fatal): {e}")


def ws_handshake(base, path="/events", timeout=10):
    """Confirm the /events WebSocket actually upgrades (HTTP 101) — NOT just that HTTP routes
    work. A missing websockets/uvicorn[standard] dep serves HTTP but fails the WS handshake,
    which polling /api/stage hides. Stdlib raw-socket handshake (no ws client lib needed)."""
    import base64
    import os
    import socket
    from urllib.parse import urlparse
    u = urlparse(base)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    key = base64.b64encode(os.urandom(16)).decode()
    req = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
           f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        try:
            s.sendall(req.encode())
            s.settimeout(timeout)
            line = s.recv(256).decode("latin1", "replace").split("\r\n")[0]
            return ("101" in line, line)
        finally:
            s.close()
    except Exception as e:  # noqa: BLE001
        return (False, str(e))


def collect_events(base, path="/events", duration=60, match=None, stop_when=None, connect_timeout=10):
    """Best-effort: open the /events WebSocket (stdlib raw-socket handshake + minimal frame
    decode), collect JSON events for up to `duration` seconds, return the list of parsed dicts.
    `match(ev)`-truthy events are kept; `stop_when(events)` (optional) ends early. Never raises —
    returns ([], err_str) on any failure so the caller can degrade to a skip. Handles only
    server→client text frames with <64KiB payloads (the app's small status JSON), unmasked."""
    import base64
    import json as _json
    import os
    import socket
    from urllib.parse import urlparse
    u = urlparse(base)
    host, port = u.hostname or "127.0.0.1", u.port or 80
    key = base64.b64encode(os.urandom(16)).decode()
    handshake = (f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\n"
                 f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
    events = []
    try:
        s = socket.create_connection((host, port), timeout=connect_timeout)
    except Exception as e:  # noqa: BLE001
        return events, f"connect failed: {e}"
    try:
        s.sendall(handshake.encode())
        s.settimeout(connect_timeout)
        hdr = b""
        while b"\r\n\r\n" not in hdr:
            chunk = s.recv(256)
            if not chunk:
                return events, "ws closed during handshake"
            hdr += chunk
        if b"101" not in hdr.split(b"\r\n", 1)[0]:
            return events, "no 101 upgrade"
        buf = hdr.split(b"\r\n\r\n", 1)[1]  # any payload bytes already read
        t0 = time.time()
        s.settimeout(2.0)
        while time.time() - t0 < duration:
            # ensure we have a frame header
            try:
                while len(buf) < 2:
                    buf += s.recv(4096)
            except socket.timeout:
                continue
            except Exception:  # noqa: BLE001
                break
            b0, b1 = buf[0], buf[1]
            opcode = b0 & 0x0F
            masked = (b1 & 0x80) != 0
            ln = b1 & 0x7F
            idx = 2
            try:
                if ln == 126:
                    while len(buf) < 4:
                        buf += s.recv(4096)
                    ln = int.from_bytes(buf[2:4], "big"); idx = 4
                elif ln == 127:
                    while len(buf) < 10:
                        buf += s.recv(4096)
                    ln = int.from_bytes(buf[2:10], "big"); idx = 10
                if masked:
                    idx += 4  # skip mask key (server frames are normally unmasked; tolerate anyway)
                while len(buf) < idx + ln:
                    buf += s.recv(4096)
            except socket.timeout:
                continue
            except Exception:  # noqa: BLE001
                break
            payload = buf[idx:idx + ln]
            buf = buf[idx + ln:]
            if opcode == 0x8:  # close
                break
            if opcode in (0x1, 0x2):  # text/binary
                try:
                    ev = _json.loads(payload.decode("utf-8", "replace"))
                except Exception:  # noqa: BLE001
                    continue
                if match is None or match(ev):
                    events.append(ev)
                if stop_when and stop_when(events):
                    break
        return events, None
    except Exception as e:  # noqa: BLE001
        return events, str(e)
    finally:
        with contextlib.suppress(Exception):
            s.close()


def wait_ready(base, timeout) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, bd, _ = req("GET", base, "/api/stage", timeout=10)
        if isinstance(bd, dict) and bd.get("ready"):
            return True
        time.sleep(2)
    return False


def _pick_busy(st, bd) -> bool:
    """Is this pick response a 'render thread is busy' fail-closed, rather than a real
    answer? A GPU pick needs the single render thread, which the app deliberately monopolizes
    (and freezes the viewport) while a batch/timeline renders — so a pick issued mid-render
    fails closed with 'pick timed out'. In the product no user picks then (the viewport is
    frozen); only this grader fires renders and picks back-to-back. Treat that as busy and
    retry, so the pick check reflects the real, idle-stage scenario."""
    if st is None:
        return True   # request-level timeout / refusal while the thread is under a heavy render
    if isinstance(bd, dict) and isinstance(bd.get("error"), str) and "tim" in bd["error"].lower():
        return True   # {"error": "pick timed out"}
    return False


def pick_when_idle(base, endpoint, deadline_s):
    """POST a centre pick, retrying ONLY while the render thread is busy with the throwaway
    batch/timeline jobs this grader fired. Returns the first non-busy (st, body) — the real
    answer once the stage is idle — or the last response at the deadline. A genuine hit or a
    clean miss stops immediately; only a busy renderer is retried."""
    t0 = time.time()
    st, bd = None, None
    while True:
        st, bd, _ = req("POST", base, endpoint, {"nx": 0.5, "ny": 0.45}, timeout=25)
        if not _pick_busy(st, bd) or (time.time() - t0) >= deadline_s:
            return st, bd
        time.sleep(5)


def wait_http(base, timeout=120) -> bool:
    """Poll until the control plane answers (the app serves HTTP before its renderer
    finishes warming, but uvicorn may not be up the instant the banner prints)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        st, _, err = req("GET", base, "/api/config", timeout=5)
        if st is not None:
            return True
        time.sleep(1)
    return False


# ----------------------------- scorecard -----------------------------
class Card:
    def __init__(self):
        self.rows = {}  # key -> {"name","score","evidence"}

    def add(self, key, name, score, evidence):
        self.rows[key] = {"name": name, "score": score, "evidence": str(evidence)[:400]}
        tag = {"full": "[FULL]", "partial": "[PART]", "fail": "[FAIL]", "skip": "[skip]"}.get(score, score)
        print(f"  {tag} {key}: {name} — {str(evidence)[:160]}")

    def dump(self, path=None):
        out = {"rows": self.rows}
        text = json.dumps(out, indent=2)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return out


def base_selection_from(info):
    return [{"prim_path": s["prim_path"], "set_name": s["set_name"], "variant": s.get("current") or s["variants"][0]}
            for s in info.get("variant_sets", [])]


def q():  # default quality body
    return {"mode": "RealTimePathTracing", "samples_per_pixel": 40, "max_bounces": 4, "resolution": [640, 360]}


# ----------------------------- the grade run -----------------------------
def grade(base, usd, do_render, render_timeout, card: Card, area9_url=None):
    root_sha0 = sha(usd) if Path(usd).is_file() else None

    # --- contract: /api/config ---
    st, bd, err = req("GET", base, "/api/config")
    ok_cfg = st == 200 and isinstance(bd, dict) and "signal_port" in bd and "stream_resolution" in bd
    card.add("contract", "GET /api/config", "full" if ok_cfg else "fail", bd if not err else err)

    # --- frontend assets must serve (catches the DOA static-mount bug; API-only grade can't) ---
    grade_frontend_assets(base, card)

    # --- gate G5: the /events WebSocket must actually upgrade (catches missing ws lib) ---
    ws_ok, ws_ev = ws_handshake(base)
    card.add("G5_events_ws", "GET /events WS handshake (101)", "full" if ws_ok else "fail",
             f"status line: {ws_ev}" + ("" if ws_ok else "  <- no WS lib? install uvicorn[standard]/websockets"))

    # --- open the local stage ---
    st, info, err = req("POST", base, "/api/open", {"usd_path": str(usd)}, timeout=120)
    if not (st == 200 and isinstance(info, dict) and info.get("variant_sets") is not None):
        card.add("open", "POST /api/open (local)", "fail", err or info)
        return  # nothing else is gradeable without an open stage
    sets = info.get("variant_sets", [])
    cams = info.get("cameras", [])
    base_sel = base_selection_from(info)
    multis = sorted([s for s in sets if len(s.get("variants", [])) >= 2], key=lambda s: -len(s["variants"]))
    card.add("open", "POST /api/open (local)", "full",
             f"{len(sets)} variant sets, {len(cams)} cameras, {len(multis)} multi-variant sets")

    # --- area 2: variant switch (HTTP accept; pixels = browser) ---
    if multis:
        s0 = multis[0]
        alt = next(v for v in s0["variants"] if v != (s0.get("current") or s0["variants"][0]))
        sel = [dict(c) for c in base_sel]
        for c in sel:
            if c["set_name"] == s0["set_name"]:
                c["variant"] = alt
        st, bd, err = req("POST", base, "/api/variant", {"selections": sel})
        card.add("area2", "variant switch (HTTP)", "partial" if st == 200 else "fail",
                 f"switch {s0['set_name']}->{alt}: {bd if not err else err} (pixel-change confirmed in live smoke)")
    else:
        card.add("area2", "variant switch", "fail", "no multi-variant set to switch")

    # --- area 3: cameras (list + snap + display accept) ---
    if cams:
        cam = cams[0]["path"]
        animated_flags = all("animated" in c for c in cams)
        st1, _, e1 = req("POST", base, "/api/camera/snap", {"camera_path": cam})
        st2, _, e2 = req("POST", base, "/api/display",
                         {"iso": 800.0, "focal_length": 50.0, "f_stop": 2.8, "focus_distance": 500.0})
        ok = st1 == 200 and st2 == 200 and animated_flags
        card.add("area3", "cameras (list/snap/display)", "partial" if ok else ("partial" if st1 == 200 else "fail"),
                 f"{len(cams)} cams, animated-flag={animated_flags}, snap={st1}, display={st2} (re-pose confirmed live)")
    else:
        card.add("area3", "cameras", "fail", "no cameras listed")

    # --- area 5: batch counts + explosion guard (+ queue one small real job) ---
    tmp_render = Path(tempfile.mkdtemp(prefix="vs_grade_render_"))
    if multis:
        s0 = multis[0]
        two = s0["variants"][:2]
        # one-at-a-time: count == sum of included variant counts
        st, bd, err = req("POST", base, "/api/batch", {"job": {
            "mode": "one_at_a_time", "base_selection": base_sel,
            "included": {s0["set_name"]: two}, "cameras": [cams[0]["path"]] if cams else [],
            "quality": q(), "frame_mode": "single", "out_dir": str(tmp_render), "confirm": False}})
        oaat_ok = st == 200 and isinstance(bd, dict) and bd.get("count") == 2
        card.add("area5_oaat", "batch one-at-a-time count", "full" if oaat_ok else "fail",
                 f"expected 2, got {bd}")
        # full cartesian: count == product
        if len(multis) >= 2:
            s1 = multis[1]
            st, bd, err = req("POST", base, "/api/batch", {"job": {
                "mode": "full_cartesian", "base_selection": base_sel,
                "included": {s0["set_name"]: s0["variants"][:2], s1["set_name"]: s1["variants"][:2]},
                "cameras": [], "quality": q(), "frame_mode": "single",
                "out_dir": str(Path(tempfile.mkdtemp(prefix="vs_grade_cart_"))), "confirm": False}})
            cart_ok = st == 200 and isinstance(bd, dict) and bd.get("count") == 4
            card.add("area5_cart", "batch cartesian count", "full" if cart_ok else "fail",
                     f"expected 4, got {bd}")
        # explosion guard: build a cartesian whose product > 500, NO confirm -> expect 409
        prod, included_big = 1, {}
        for s in multis:
            included_big[s["set_name"]] = s["variants"]
            prod *= len(s["variants"])
            if prod > 500:
                break
        if prod > 500:
            st, bd, err = req("POST", base, "/api/batch", {"job": {
                "mode": "full_cartesian", "base_selection": base_sel, "included": included_big,
                "cameras": [], "quality": q(), "frame_mode": "single",
                "out_dir": str(tmp_render), "confirm": False}})
            card.add("area5_guard", "explosion guard (409)", "full" if st == 409 else "fail",
                     f"product={prod}, status={st} (want 409)")
        else:
            card.add("area5_guard", "explosion guard", "skip", f"product only {prod}, can't exceed 500")
    else:
        card.add("area5_oaat", "batch counts", "fail", "no multi-variant set")

    # --- area 6: timeline frames (deterministic) ---
    if multis:
        s0 = multis[0]
        tl = {"duration_s": 4.0, "fps": 2.0, "tracks": [{
            "kind": "variant_set", "set_name": s0["set_name"], "prim_path": s0["prim_path"],
            "clips": [{"value": s0["variants"][0], "start_s": 0.0, "duration_s": 2.0},
                      {"value": s0["variants"][1], "start_s": 2.0, "duration_s": 2.0}]}]}
        st, bd, err = req("POST", base, "/api/timeline/render",
                          {"timeline": tl, "quality": q(), "out_dir": str(Path(tempfile.mkdtemp(prefix="vs_grade_tl_")))},
                          timeout=180)   # some builds BLOCK until the MP4 renders (~40s) before returning frames
        frames_ok = st == 200 and isinstance(bd, dict) and bd.get("frames") == 8
        card.add("area6", "timeline frames count", "full" if frames_ok else "fail",
                 f"4s @ 2fps -> expected 8 frames, got {bd}")
    else:
        card.add("area6", "timeline", "fail", "no set for a timeline track")

    # --- area 7: projects round-trip ---
    pname = "_grade_probe"
    st1, _, _ = req("POST", base, "/api/projects/save",
                    {"name": pname, "base_selection": base_sel, "display": {},
                     "camera": cams[0]["path"] if cams else ""})
    st2, lst, _ = req("GET", base, "/api/projects")
    listed = pname in json.dumps(lst)
    st3, rec, _ = req("GET", base, f"/api/projects/load?name={pname}")
    loaded = st3 == 200 and isinstance(rec, dict)
    req("POST", base, "/api/projects/delete", {"name": pname})  # cleanup, best-effort
    proj_ok = st1 == 200 and listed and loaded
    card.add("area7", "projects save/list/load", "full" if proj_ok else ("partial" if st1 == 200 else "fail"),
             f"save={st1}, listed={listed}, load={st3}")

    # ---------- warm/GPU-dependent checks ----------
    # The cold initial open_usd holds USD_LOCK for its whole duration, so turntable +
    # remote-mirror reopens (both pxr-authoring) can't run until the first frame exists.
    # Batch stills + post also need a real render. All of these are gated on readiness.
    if do_render:
        print(f"  ... waiting up to {render_timeout}s for the first frame ...")
        ready = wait_ready(base, timeout=render_timeout)

        # area 5 (files): the small one-at-a-time job queued above renders once warm
        pngs = []
        if ready:
            t0 = time.time()
            while time.time() - t0 < render_timeout:
                pngs = list(tmp_render.glob("*.png"))
                if pngs:
                    break
                time.sleep(3)
        card.add("area5_files", "batch writes PNG stills", "full" if pngs else ("partial" if ready else "fail"),
                 f"ready={ready}, {len(pngs)} png(s) in out_dir")

        # area 8: post on the rendered stills (originals must be untouched)
        if pngs:
            png0 = pngs[0]
            png0_sha = sha(png0)
            st, ov, _ = req("POST", base, "/api/post/overlay", {"out_dir": str(tmp_render)}, timeout=180)
            originals_ok = sha(png0) == png0_sha
            st2, cs, _ = req("POST", base, "/api/post/cutsheet", {"out_dir": str(tmp_render)}, timeout=180)
            cut_ok = st2 == 200 and isinstance(cs, dict) and cs.get("path") and Path(cs["path"]).is_file()
            st3, res, _ = req("GET", base, f"/api/results?dir={tmp_render}")
            res_ok = st3 == 200 and isinstance(res, dict) and res.get("permutations")
            ok8 = (st == 200) and originals_ok and cut_ok and res_ok
            card.add("area8", "results + overlay + cutsheet", "full" if ok8 else "partial",
                     f"overlay={st}/{ov}, originals_untouched={originals_ok}, cutsheet={cut_ok}, results={res_ok}")
        else:
            card.add("area8", "results + post", "fail", "no rendered stills to post-process")

        # area 4: turntable rig (needs the lock free — i.e. warm); reopens to the rig camera
        cams_before = len(cams)
        st, bd, err = req("POST", base, "/api/turntable",
                          {"pivot": [0.0, 0.0, 0.0], "radius": 400.0, "height": 120.0,
                           "frames": 48, "fps": 24.0, "focal_length": 35.0, "start_deg": 0.0}, timeout=120)
        rig_cam = False
        if st == 200 and isinstance(bd, dict):
            newcams = bd.get("cameras", [])
            rig_cam = len(newcams) > cams_before or any(
                re.search("turn", c.get("name", ""), re.I) or re.search("turn", c.get("path", ""), re.I)
                for c in newcams)
        root_ok_mid = (root_sha0 is None) or (sha(usd) == root_sha0)
        card.add("area4", "turntable rig (sidecar)",
                 "full" if (st == 200 and rig_cam and root_ok_mid) else ("partial" if st == 200 else "fail"),
                 f"status={st}, rig_camera_present={rig_cam}, source_root_unchanged={root_ok_mid} "
                 f"(quarter-lap motion confirmed in live smoke)")

        # picks (warm, IDLE stage): focus picker returns a DISTANCE; pivot pick returns a 3D POINT.
        # Wait out any throwaway batch/timeline render still in flight (the render thread is
        # single + frozen during a render, so a mid-render pick fails closed — a state no real
        # user is in). pick_when_idle retries only while busy, then asserts on the idle result.
        stf, fr = pick_when_idle(base, "/api/pick-focus", deadline_s=render_timeout)
        # accept either field name: the spec/builds use `focus_distance`; the reference app uses `distance`
        _fd = fr.get("focus_distance", fr.get("distance")) if isinstance(fr, dict) else None
        focus_ok = stf == 200 and isinstance(_fd, (int, float))
        card.add("area3_pickfocus", "focus picker returns a distance", "full" if focus_ok else "fail",
                 f"status={stf}, body={fr}")
        stp, pr = pick_when_idle(base, "/api/pick-point", deadline_s=render_timeout)
        # accept either field name: spec/builds use `world`; the reference app uses `point`
        world = (pr.get("world") if pr.get("world") is not None else pr.get("point")) if isinstance(pr, dict) else None
        # HIT must be a 3-float, NON-zero point AND not the fixed-up fallback [0,1,0] (a pick that
        # fabricates a constant point on every click passes a loose "3 floats" check — tighten it).
        _is3 = isinstance(world, (list, tuple)) and len(world) == 3
        _nonzero = _is3 and any(abs(float(x)) > 1e-6 for x in world)
        _not_up = _is3 and not (abs(float(world[0])) < 1e-6 and abs(float(world[1]) - 1.0) < 1e-6 and abs(float(world[2])) < 1e-6)
        point_ok = stp == 200 and _is3 and _nonzero and _not_up
        card.add("area4_pickpoint", "pivot pick returns a real on-car world point", "full" if point_ok else "fail",
                 f"status={stp} (405 = no endpoint), body={pr}, nonzero={_nonzero}, not_up={_not_up}")

        # NOTE: we deliberately do NOT gate a "corner pick must MISS" here. In a studio scene with a
        # backdrop DOME, an off-car corner click legitimately HITS the dome and returns a real
        # non-zero point — that is correct geometry resolution, NOT a fabricated fallback, so a
        # corner-miss assertion false-fails the (good) reference. The failure we DO care about (the pick
        # fabricates a CONSTANT point — world origin or the [0,1,0] up-fallback — on every click) is
        # already caught above by area4_pickpoint's `_nonzero` + `_not_up` tests on the on-car pick.

        # variant-survives-reload (the revert bug): switch a fast set, then a reload set, assert
        # the fast selection is still present afterward (full pixel-survival is the live runbook).
        if multis:
            fastset = multis[0]
            fastvar = next((v for v in fastset["variants"] if v != (fastset.get("current") or fastset["variants"][0])), None)
            relset = next((s for s in sets if s["set_name"].lower() in
                           ("doors", "wheel_turns", "headrests", "backdrops")
                           and len(s.get("variants", [])) >= 2), None)
            if fastvar and relset:
                sel = [dict(c) for c in base_sel]
                for c in sel:
                    if c["set_name"] == fastset["set_name"]:
                        c["variant"] = fastvar
                req("POST", base, "/api/variant", {"selections": sel}, timeout=30)
                time.sleep(3)
                relvar = next(v for v in relset["variants"] if v != (relset.get("current") or relset["variants"][0]))
                sel2 = [dict(c) for c in sel]
                for c in sel2:
                    if c["set_name"] == relset["set_name"]:
                        c["variant"] = relvar
                req("POST", base, "/api/variant", {"selections": sel2}, timeout=60)
                time.sleep(6)
                _, stg, _ = req("GET", base, "/api/stage", timeout=10)
                cur = {s["set_name"]: s["variant"] for s in (stg.get("selection", []) if isinstance(stg, dict) else [])}
                survived = cur.get(fastset["set_name"]) == fastvar and cur.get(relset["set_name"]) == relvar
                card.add("area2_reload_survive", "fast variant survives a reload (selection)",
                         "full" if survived else "fail",
                         f"after {fastset['set_name']}={fastvar} then {relset['set_name']}={relvar}: "
                         f"selection={cur.get(fastset['set_name'])}/{cur.get(relset['set_name'])} "
                         f"(NOTE: pixel-survival verified in the live runbook — selection alone isn't proof)")

        # area 9: remote mirror — LAST (reopens to the tiny stage, abandoning ConceptCar)
        with MirrorFixture() as fx:
            st, bd, err = req("POST", base, "/api/open", {"usd_path": fx.root_url}, timeout=120)
            probe_sets = bd.get("variant_sets", []) if isinstance(bd, dict) else []
            has_probe = any(s.get("set_name") == "probe" for s in probe_sets)
            src_url = (bd or {}).get("source_url", "") if isinstance(bd, dict) else ""
            unchanged = fx.source_unchanged()
            identity_ok = src_url == fx.root_url
            ok9 = st == 200 and has_probe and unchanged
            card.add("area9", "remote stage mirror",
                     "full" if (ok9 and identity_ok) else ("partial" if ok9 else "fail"),
                     f"status={st}, closure_composed(probe set seen)={has_probe}, source_unchanged={unchanged}, "
                     f"url_identity={identity_ok}")

        # area 9 (progress): COLD-mirror download progress contract. The reference streams
        # `mirror_progress` events (monotonic `downloaded` byte/file count) over /events while it
        # mirrors a remote closure, so the UI can show a progress bar instead of a frozen window.
        # This needs a real REMOTE area9 closure (--area9-url, e.g. the S3 ConceptCar). The local
        # MirrorFixture is too small/instant to download-mirror, so it is NOT used here. Best-effort
        # and strictly non-fatal: if no URL is given or the remote can't be reached, mark `skip`.
        grade_mirror_progress(base, area9_url, card)
    else:
        for k, n in (("area5_files", "batch writes PNG stills"), ("area8", "results + post"),
                     ("area4", "turntable rig"), ("area9", "remote stage mirror"),
                     ("area9_mirror_progress", "cold-mirror progress events")):
            card.add(k, n, "skip", "needs a warm stage — run with --render")

    # --- gate G3: source USD untouched across the whole session ---
    if root_sha0 is not None:
        unchanged = sha(usd) == root_sha0
        card.add("G3", "user USD never modified", "full" if unchanged else "fail",
                 f"root sha {'unchanged' if unchanged else 'CHANGED — gate FAIL'}")
    card.add("G1_G2_G4", "server-ovrtx / no-client-3D / single-step-owner", "skip",
             "graded via live smoke (browser) + builder code review")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="base URL of an already-running app, e.g. http://127.0.0.1:8090")
    ap.add_argument("--launch", help="shell command to launch the app (convenience/self-test)")
    ap.add_argument("--cwd", help="cwd for --launch")
    ap.add_argument("--usd", required=True, help="local .usd to open for grading")
    ap.add_argument("--render", action="store_true", help="also do GPU render-output checks (areas 5-files, 8)")
    ap.add_argument("--render-timeout", type=int, default=420)
    ap.add_argument("--area9-url", help="remote area9 closure URL to exercise mirror_progress (optional; "
                                        "skipped if absent or unreachable)")
    ap.add_argument("--json", help="write the scorecard JSON here")
    args = ap.parse_args()

    proc = None
    base = args.url
    try:
        if args.launch and not base:
            # POSIX: put the shell in its OWN session/process group so teardown can kill the
            # whole tree in one go (see the finally: block). No-op semantics on Windows.
            proc = subprocess.Popen(args.launch, shell=True, cwd=args.cwd,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    start_new_session=(os.name != "nt"))
            port = None
            t0 = time.time()
            while time.time() - t0 < 180:
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                print("   [app]", line.rstrip()[:140])
                m = re.search(r"127\.0\.0\.1:(\d+)", line)
                if m:
                    port = int(m.group(1))
                    break
            if not port:
                print("ERROR: never saw a 127.0.0.1:<port> banner from the app", file=sys.stderr)
                sys.exit(2)
            base = f"http://127.0.0.1:{port}"
            time.sleep(3)
        if not base:
            print("ERROR: give --url or --launch", file=sys.stderr)
            sys.exit(2)

        print(f"== grading {base} (usd={args.usd}, render={args.render}) ==")
        if not wait_http(base, timeout=120):
            print("ERROR: control plane never answered /api/config", file=sys.stderr)
            sys.exit(2)
        card = Card()
        grade(base, args.usd, args.render, args.render_timeout, card, area9_url=args.area9_url)
        result = card.dump(args.json)
        # crude rollup
        scores = [r["score"] for r in result["rows"].values()]
        print("\n== rollup ==")
        for s in ("full", "partial", "fail", "skip"):
            print(f"  {s}: {scores.count(s)}")
    finally:
        if proc and proc.poll() is None:
            # Clean teardown — never leave a stray ovstream holding the GPU + ports 8080/49100.
            # `shell=True` means proc.pid is the SHELL's pid, so a plain proc.kill() orphans the
            # real `python -m dev_variant_presenter` child. Kill the whole TREE instead:
            # Windows via taskkill /T, POSIX via the process group (the launch above used
            # start_new_session=True precisely so the group is ours alone to kill).
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True)
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
            with contextlib.suppress(Exception):
                proc.wait(timeout=15)   # reap, so the GPU is actually released before we exit


if __name__ == "__main__":
    main()
