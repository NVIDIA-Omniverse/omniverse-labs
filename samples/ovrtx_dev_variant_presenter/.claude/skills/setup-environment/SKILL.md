---
name: setup-environment
description: Use when setting up a fresh clone of this repo to run OVRTX Dev Variant Presenter (also called the "variant studio" in the walkthrough video and earlier releases) — no .venv yet, ovrtx/ovstage/ovstream not installed, ModuleNotFoundError on ovrtx/ovstage/ovstream, or "how do I install / get this running". RTX + uv; validated on Windows.
---

# Set up the environment to run OVRTX Dev Variant Presenter

The environment is NOT shipped — `.venv` is gitignored. `uv sync` rebuilds it from `pyproject.toml`.

## Prerequisites (README §Requirements)

- **An NVIDIA RTX GPU + recent driver.** Rendering is headless via Vulkan; no display server needed. This app does not run without an RTX GPU. Developed and validated on **Windows 11**; Linux is supported by the stack (all three packages ship manylinux wheels) but is not routinely validated here.
- **Python 3.11, via [`uv`](https://github.com/astral-sh/uv).** `uv` must already be installed (it is NOT auto-installed here).
- A modern Chromium browser (Chrome / Edge) for the viewer.
- A USD stage to open — none is bundled; the public NVIDIA **ConceptCar** URL in the README is the quickest one to try.

## Install

From the repo root:

```powershell
uv sync
```

This creates `.venv` and installs everything — including `ovrtx` (renderer), `ovstage`
(live stage host), and `ovstream` (WebRTC streamer) — from PyPI; `uv.lock` pins the
validated versions (`ovrtx` 0.4.x, `ovstage` 0.1.x, `ovstream` 0.4.x, Python 3.11). No
extra index, flag, or auth needs to be configured — but the install **does need network
access to `pypi.nvidia.com`**: the `ovrtx` package on PyPI is only a `wheel-stub` sdist
whose build backend downloads the real platform wheel from NVIDIA's index, so an environment
restricted to `pypi.org` alone cannot complete `uv sync`. (`run_server.ps1` /
`run_server.sh` also run this automatically on first launch if `.venv` is missing.)

One browser-side file is **not** in the repo and is **not** installed by `uv sync` (the wheels
ship no JavaScript): `web/omniverse-webrtc-streaming-library.js`, NVIDIA's StreamSDK WebRTC
client, which the viewer cannot stream without. It is NVIDIA's own software, so it is fetched
rather than redistributed: on first launch `run_server.ps1` / `run_server.sh` download it from
the public [`NVIDIA-Omniverse/ovstream`](https://github.com/NVIDIA-Omniverse/ovstream)
repository at a **pinned commit** and verify its SHA256, deleting the file and refusing to start
if the checksum does not match. This needs network access to `raw.githubusercontent.com`, once —
if the file is already on disk, no download happens.

## Verify

```powershell
uv run python -c "import ovrtx, ovstage, ovstream, warp; print('env OK')"
```

## Next

Start the server with the **`start-server`** skill — always via the watchdog (`run_server.ps1` on Windows, `run_server.sh` on Linux/macOS), never bare `python` or a generic preview/dev-server launcher.

## If it fails

- `uv: command not found` → install `uv` first (see its docs); not handled here.
- ovrtx / ovstage / ovstream won't resolve → confirm Python 3.11 (`ovrtx` ships no 3.12+ builds) and network access to **both** `pypi.org` **and** `pypi.nvidia.com` (the `ovrtx` PyPI package is a `wheel-stub` sdist that fetches the real wheel from NVIDIA's index at build time — a `pypi.org`-only mirror or an offline proxy fails here).
- Import fails at the verify step → wrong/no GPU+driver. An RTX GPU is required; the validated configuration is Windows + RTX.
- The launcher aborts on the stream-client fetch (network blocked / offline / SHA256 mismatch) → download `https://raw.githubusercontent.com/NVIDIA-Omniverse/ovstream/af7f1f9006d1037a3cc7b8eca73f39a6469b69c2/examples/webrtc_client/omniverse-webrtc-streaming-library.js` on any machine that can reach `raw.githubusercontent.com`, save it as `web/omniverse-webrtc-streaming-library.js` in the repo, and relaunch (the launcher skips the fetch when the file exists). The page also says so itself: if that file is missing the viewer renders an explanatory message instead of a dead page.
