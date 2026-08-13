# Third-Party Notices

This repository is licensed under Apache-2.0 (see [`LICENSE`](LICENSE)). It contains no
third-party code. The one third-party component the application needs is **fetched at setup**,
not redistributed here; it is described below so that its terms and its network behavior are
known before you run the viewer.

## NVIDIA Omniverse WebRTC Streaming Library

- **File:** `web/omniverse-webrtc-streaming-library.js` — present on disk after first launch,
  and **not** tracked by this repository (it is listed in `.gitignore`).
- **What it is:** a minified build of NVIDIA's StreamSDK WebRTC browser client. It implements
  the StreamSDK signaling flavor that the `ovstream` server speaks; off-the-shelf WebRTC tooling
  or a raw browser `RTCPeerConnection` will not interoperate with it.
- **Why it is needed:** the browser viewer requires it — `web/index.html` loads it directly, and
  without it the live stream cannot connect.
- **How it is obtained:** `run_server.ps1` / `run_server.sh` download it on first launch and
  verify its SHA256 (`447a74830162b91cb92b0a636f02c0b3e668d835e2a4496f560e31e2b48e5c71`). If the
  checksum does not match, the partial file is deleted and the launcher refuses to start.
- **Upstream:** [`NVIDIA-Omniverse/ovstream`](https://github.com/NVIDIA-Omniverse/ovstream),
  path `examples/webrtc_client/omniverse-webrtc-streaming-library.js`, pinned to commit
  `af7f1f9006d1037a3cc7b8eca73f39a6469b69c2` (pinned, not `HEAD`, so an upstream change cannot
  silently alter what is fetched).
- **Licensing:** this file is **not** covered by this repository's Apache-2.0 license. It is
  NVIDIA's own software and is covered by the terms of the `ovstream` project it is fetched
  from — see that repository's `LICENSE` and `THIRD_PARTY_NOTICES.md` for the conditions that
  apply to it and to its own dependencies.

### Outbound network endpoints

The bundle contains code that can contact NVIDIA-operated hosts. Telemetry endpoints:

- `telemetry.gfe.nvidia.com`
- `telemetry.gfestage.nvidia.com`
- `events.gfe.nvidia.com`
- `events.gfestage.nvidia.com`

It also references an NVIDIA **STUN** server, used for WebRTC ICE / NAT traversal rather than
telemetry (it is added to the ICE server list only when the bundle's `enableStunServer` option
is on):

- `s1.stun.gamestream.nvidia.com` (UDP 19308)

This is stated here as a factual property of the fetched bundle so that anyone deploying the
viewer is aware of the outbound connections it may make. For the local-first, single-machine
use this repository demonstrates (browser and server both on `127.0.0.1`), no STUN traversal is
needed.
