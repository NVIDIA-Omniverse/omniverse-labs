# Dev Variant Presenter — agent instructions

Live + batch + timeline variant-permutation renderer on **ovrtx** (renderer) + **ovstage**
(live stage host) + **ovstream** (WebRTC streamer). Developed and validated on **Windows + RTX**;
Linux is supported by the stack but not routinely validated here. FastAPI control plane; a single
render thread owns the `ovstage.Stage` and the attached `ovrtx.Renderer`.

**This file is the single source of truth for agent instructions in this repo.** `CLAUDE.md` and
`.cursor/rules/dev-variant-presenter.mdc` exist only so Claude Code and Cursor auto-load something
at session start; both point here rather than restating it, so there is one copy to keep correct.

## Skills — the full procedures

The two rules below are the summary. The step-by-step procedures live in skill files, which are
plain Markdown and worth opening directly whichever agent you are:

| Task | Read |
| --- | --- |
| Start the server and hand the user a URL | `.claude/skills/start-server/SKILL.md` |
| Set up a fresh clone / repair a broken env | `.claude/skills/setup-environment/SKILL.md` |

Claude Code discovers these automatically. Codex and Cursor do not, so read the file by path when
the task matches. Rebuilding the whole app from scratch is a separate, much larger package with its
own prompt and acceptance gates: see `rebuild/README.md`.

## Running the server — ALWAYS via the watchdog

Start it ONLY with the watchdog script — `run_server.ps1` on Windows, `run_server.sh` on
Linux/macOS:

```powershell
powershell -ExecutionPolicy Bypass -File run_server.ps1   # Windows
```

```bash
./run_server.sh                                           # Linux / macOS
```

NEVER start it bare (`python -m dev_variant_presenter …`) or via a generic "preview"/dev-server
launcher. Those skip the watchdog, so a routine ovrtx/ovstream native crash becomes a dead app with
no relaunch and no logs — an unrecoverable session with nothing to diagnose from. The watchdog
auto-relaunches on crash (3s) and writes `logs/server_out.log`. The
stream is **single-client** — never open a second browser/client (it steals an active session).

## Stopping the server

Kill the watchdog process (`run_server.ps1` / `run_server.sh`) FIRST (otherwise it relaunches in
3s), THEN the `python … -m dev_variant_presenter` process. Watch for ORPHANED servers from old sessions — a stray
`-m dev_variant_presenter` (sometimes on system Python, not `.venv`) holding 8080/49100 keeps the GPU hot
and forces the new server onto shifted ports.

## Setup (fresh clone)

`uv sync` from the repo root — creates `.venv`; everything (incl. `ovrtx` / `ovstage` / `ovstream`)
installs from PyPI, pinned by `uv.lock` (`ovrtx` 0.4.x, `ovstage` 0.1.x, `ovstream` 0.4.x,
Python 3.11). The environment is NOT committed (`.venv` is gitignored). Both watchdog scripts
also run the sync automatically on first launch.

## Tests

`node web/timeline-core.test.cjs` (frontend timeline logic) · `uv run pytest` (Python).
