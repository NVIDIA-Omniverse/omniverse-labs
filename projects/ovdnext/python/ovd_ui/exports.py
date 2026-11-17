# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native capture exports in an independent file session."""
from dataclasses import dataclass
from pathlib import Path
import tempfile

from _pvd_native import Session
from .recordings import copy_capture
from .ingestion import retain_capture


@dataclass(frozen=True)
class ExportRequest:
    source: Path
    destination: Path
    mode: str
    scene_id: int
    frame: int
    start: int=0
    stop: int=0
    instance_ids: tuple[int,...] | None=None
    overwrite: bool=False


def export_capture(request):
    """Preserve the displayed engine state and publish only a complete result."""
    source=Path(request.source).expanduser().resolve()
    destination=Path(request.destination).expanduser().absolute()
    if source==destination:
        raise ValueError("Choose an export destination different from the source capture")
    suffix=".pxd2" if request.mode=="clip" else ".repx"
    if destination.suffix.lower()!=suffix:
        raise ValueError(f"This export requires the {suffix} extension")
    if request.mode not in ("clip","repx"):
        raise ValueError("Unknown native export operation")
    with retain_capture(source), tempfile.TemporaryDirectory(prefix="pvd-export-") as temporary:
        before=source.stat()
        session=Session(str(source),temporary)
        try:
            if request.mode=="repx" and not session.seek(request.scene_id,request.frame):
                raise ValueError("The export frame could not be loaded")
            caps=session.export_capabilities()
            allowed,reason=(caps.clip,caps.clip_reason) if request.mode=="clip" else (caps.repx,caps.repx_reason)
            if not allowed:
                raise ValueError(reason)
            result=Path(temporary)/("export"+suffix)
            if request.mode=="clip":
                session.save_clip(str(result),request.scene_id,request.start,request.stop)
            else:
                session.export_repx(str(result),request.instance_ids)
            after=source.stat()
            if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(
                    after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError("Capture changed during export; use a completed recording")
            copy_capture(result,destination,overwrite=request.overwrite)
            return destination,destination.stat().st_size
        finally:
            del session
