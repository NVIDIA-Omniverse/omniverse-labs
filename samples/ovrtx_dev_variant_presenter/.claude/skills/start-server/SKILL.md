---
name: start-server
description: Use when asked to start, run, launch, restart, or bring up the OVRTX Dev Variant Presenter server, stream, or viewport in this repo — also called the "variant studio" in the walkthrough video and earlier releases — or after the stream / GPU / server was lost and needs recovering. RTX required; validated on Windows.
---

# Start the OVRTX Dev Variant Presenter server

## The one rule

Start the server ONLY through the watchdog script — `run_server.ps1` on Windows, `run_server.sh` on Linux/macOS:

```powershell
powershell -ExecutionPolicy Bypass -File run_server.ps1
```

```bash
./run_server.sh
```

From an agent / automation, launch it detached so it keeps running:

```powershell
Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-NoProfile','-File','run_server.ps1' -WorkingDirectory '<repo-root>' -WindowStyle Hidden
```

```bash
cd <repo-root> && mkdir -p logs && nohup ./run_server.sh > logs/watchdog.log 2>&1 &
```

## Forbidden launch methods

- ❌ `python -m dev_variant_presenter ...` (bare) — no watchdog
- ❌ A generic preview/dev-server launcher — bare process; its stdout is discarded when it exits, so a crash leaves no logs
- ❌ A second browser tab / client — the WebRTC stream is single-client; a second client steals the active session (e.g. an in-progress recording)

**Why:** `run_server.ps1` / `run_server.sh` are watchdogs. The ovrtx/ovstage/ovstream native layer hard-dies routinely (that's expected, not a bug). The watchdog (a) auto-relaunches on abnormal exit after 3s and (b) redirects stdout/stderr to `logs/server_out.log` / `server_err.log`. Launched WITHOUT it, a normal crash becomes a dead app with no recovery and no logs to diagnose from.

## Before starting

1. **Env exists:** `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` (Linux/macOS) must be present. If not → use the `setup-environment` skill first.
2. **No server already running:** check `http://127.0.0.1:8080/`. If one responds — or an orphan from an old session is holding the port — STOP it first. Orphaned renderers keep the GPU hot and force the new server onto shifted ports. To stop cleanly: kill the watchdog process (`run_server.ps1` / `run_server.sh`) FIRST (so it can't relaunch), THEN the `python … -m dev_variant_presenter` process(es). Watch for orphans on a *different* Python (e.g. system Python, not `.venv`).

## Find the URL — the port is NOT always 8080

The control port defaults to 8080 but **shifts to the next free port** when 8080 is busy (and the WebRTC signaling port shifts the same way). Never assume 8080 — resolve the actual URL first, or a healthy server looks like a failed one:

- The watchdog prints it: `*** OVRTX Dev Variant Presenter is up - open:  <url> ***` (in the console, or in the file the detached launch redirects to).
- The server also writes it to **`logs/server_url.txt`** on every launch (stale files are removed at launch, so the file appearing means *this* run).
- Fall back to `http://127.0.0.1:8080` only if neither is available yet.

Use that resolved URL — call it `$URL` — for every check below.

## Verify it came up

- Poll `$URL` until HTTP 200 — GPU warm-up can take ~30–90s. (`logs/server_url.txt` lands within ~1s of launch, well before the server answers.)
- Tail `logs/server_out.log`; wait for the stream to go live.
- Open `$URL` in ONE Chromium tab. Use `127.0.0.1`, not `localhost`.

## On crash

The watchdog relaunches automatically (3s) and the browser auto-reopens the last stage. Evidence: `logs/server_out.log` (+ `logs/server_out.prev.log`), exit lines in `server_crashes.log`. Avoid rapid kill/relaunch cycles — they can wedge the GPU/stream until a stage reopen.
