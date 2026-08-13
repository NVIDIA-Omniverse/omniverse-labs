# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry point: `python -m dev_variant_presenter [--host H] [--port P]`."""
from __future__ import annotations

import argparse
import os
from dataclasses import replace

from dev_variant_presenter.config import Settings, find_free_port


def _publish_url(url: str) -> None:
    """Write the resolved URL to the file named by ``DEV_VARIANT_PRESENTER_URL_FILE`` so a
    wrapping launcher can surface it. The watchdog (`run_server.ps1`) redirects our stdout
    to a log, so the banner we print isn't visible interactively — and the control port may
    have shifted off the default. Best-effort: never let discoverability block startup."""
    path = os.environ.get("DEV_VARIANT_PRESENTER_URL_FILE")
    if not path:
        return
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(url + "\n")
    except OSError:
        pass


def main() -> None:
    base = Settings()
    parser = argparse.ArgumentParser(
        prog="dev-variant-presenter",
        description="Dev Variant Presenter — live + batch + timeline variant rendering")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=base.control_port)
    args = parser.parse_args()

    # Pick free ports so multiple local web apps can coexist. The browser learns the
    # control port from its own URL; the signaling port is served via GET /api/config.
    control_port = find_free_port(args.port, host=args.host)
    signal_port = find_free_port(base.signal_port, host="127.0.0.1")
    settings = replace(base, signal_port=signal_port)

    if control_port != args.port:
        print(f"  [port] control {args.port} busy -> using {control_port}", flush=True)
    if signal_port != base.signal_port:
        print(f"  [port] signaling {base.signal_port} busy -> using {signal_port}", flush=True)

    # Advertise 127.0.0.1, not localhost, to dodge the Windows localhost->::1 IPv6 trap.
    shown = "127.0.0.1" if args.host in ("127.0.0.1", "0.0.0.0", "localhost") else args.host
    url = f"http://{shown}:{control_port}"
    print(f"\n  Dev Variant Presenter  ->  {url}"
          f"   (WebRTC signaling :{signal_port})\n", flush=True)
    _publish_url(url)

    import uvicorn  # imported after argparse so --help doesn't construct anything
    from dev_variant_presenter.app import build_app

    uvicorn.run(build_app(settings), host=args.host, port=control_port)


if __name__ == "__main__":
    main()
