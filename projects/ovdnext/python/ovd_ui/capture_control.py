# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connection-bound recording policy; independent of display/viewport selection."""


def _empty_page():
    return {"state": "busy", "generation": 0, "total": 0, "copied_bytes": 0, "entries": []}


class CaptureSelectionModel:
    def __init__(self):
        self.session = None
        self.info = {}
        self.page = _empty_page()
        self.generation = self.offset = self.parent = 0
        self.name_prefix = ""
        self.all = self.secondary = False
        self._checked = {}

    @property
    def selected_count(self):
        return len(self._checked)

    def is_selected(self, target):
        return target in self._checked

    def bind(self, session):
        if session is self.session:
            return
        self.__init__()
        self.session = session

    def poll(self):
        if self.session is None:
            return self.info
        old_revision = self.info.get("catalog_revision")
        self.info = self.session.capture_control_info()
        generation = self.info["generation"]
        if generation != self.generation:
            self._checked.clear()
            self.generation, self.offset = generation, 0
        if not self.info["connected"]:
            self._checked.clear()
            self.page = _empty_page()
        elif not self.info["catalog_ready"]:
            self.page = _empty_page()
        elif old_revision != self.info["catalog_revision"]:
            self.load_page(self.offset, self.parent, self.name_prefix)
        return self.info

    def load_page(self, offset=0, parent=0, name_prefix=""):
        if self.session is None:
            raise RuntimeError("No live producer is selected")
        self.page = self.session.capture_catalog(self.generation, offset, 128, parent, name_prefix)
        self.offset, self.parent, self.name_prefix = offset, parent, name_prefix
        return self.page

    def select(self, target, checked):
        row = next((r for r in self.page["entries"] if r["target_handle"] == target), None)
        if self.page["state"] != "ready" or row is None or not row["can_select"]:
            raise ValueError("Choose a selectable row from the current catalog page")
        if checked:
            if target not in self._checked and len(self._checked) >= 65536:
                raise ValueError("Capture selection exceeds 65536 selectors")
            self._checked[target] = row["type_name"] == "PxScene"
        else:
            self._checked.pop(target, None)

    def apply(self, supersede_unknown=False):
        self.poll()
        if not self.info.get("can_apply"):
            raise RuntimeError(self.info.get("reason") or "Selection is unavailable")
        if not self.all and not self._checked:
            raise ValueError("Select at least one catalog row or explicitly choose All permitted")
        scenes = [] if self.all else [target for target, scene in self._checked.items() if scene]
        objects = [] if self.all else [target for target, scene in self._checked.items() if not scene]
        request = self.session.request_capture(scenes, objects, self.secondary, self.all,
                                               self.generation, supersede_unknown=supersede_unknown)
        self.poll()
        return request

    def refresh_catalog(self, supersede_unknown=False):
        self.poll()
        if not self.info.get("can_refresh"):
            raise RuntimeError(self.info.get("reason") or "Catalog refresh is unavailable")
        request = self.session.request_capture_catalog(supersede_unknown=supersede_unknown)
        self.poll()
        return request

    @property
    def status(self):
        info, state = self.info, self.info.get("state", "idle")
        if state == "applied":
            if info["request"]["catalog_only"]:
                return "Catalog refresh acknowledged"
            waiting = (not info["has_completed_frame"] or
                       info["segment_transition_serial"] < info["metadata_boundary_serial"])
            return "Applied; recording restarted with the new selection" + ("; waiting for the first complete frame" if waiting else "")
        if state == "unknown":
            progress = "forward bytes are arriving" if info["forward_idle_seconds"] < 1 else "no recent forward bytes"
            return f"Result unknown; {progress}. The old request may still execute; a new request must be explicit."
        return {"sending": "Sending recording request", "awaiting_ack": "Waiting for producer acknowledgement",
                "invalid": "Producer rejected the selection", "busy": "Producer is busy", "denied": "Producer denied the request"}.get(
                    state, info.get("reason") or "Choose recording scope; requests are sent only on Apply or Refresh")
