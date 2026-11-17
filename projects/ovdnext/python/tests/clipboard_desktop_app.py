# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny real OVUI process for explicitly gated cross-application clipboard QA."""
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread

import omni.ui as ui
from ovd_ui.clipboard import copy_text

directory = Path(sys.argv[1])
commands = Queue()


def input_commands():
    for line in sys.stdin:
        commands.put(json.loads(line))
    commands.put({"id": "eof", "operation": "stop"})


def reply(identity, result):
    (directory / f"{identity}.json").write_text(json.dumps(result))


async def main():
    window = ui.Window("OS clipboard QA", width=320, height=120)
    with window.frame:
        ui.Label("Public GLFW clipboard — external X11 consumer")
    reply("ready", {"ready": True})
    running = True
    while running:
        try:
            command = commands.get_nowait()
        except Empty:
            command = None
        if command is not None:
            try:
                if command["operation"] == "stop":
                    running = False
                else:
                    copy_text(command["text"])
                result = {"ok": True}
            except Exception as error:
                result = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
            reply(command["id"], result)
        await ui.next_frame()


ui.init("OS clipboard QA", width=400, height=160, max_fps=60)
Thread(target=input_commands, daemon=True).start()
try:
    ui.run(main())
finally:
    ui.shutdown()
