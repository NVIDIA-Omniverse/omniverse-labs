# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded plain-text evidence details; imported judgments stay qualified."""
import omni.ui as ui
from ..style import layout as fl
from ..evidence import uint
from ..camera import normalize_key


class EvidenceView:
    def __init__(self, controller, surface, *, on_import, on_go, on_scene, on_clips, on_changed):
        self.controller, self.surface = controller, surface
        self.on_import, self.on_go, self.on_scene, self.on_clips, self.on_changed = on_import, on_go, on_scene, on_clips, on_changed
        self.window = self.content = None
        self._key = self.action_key = None
        self.go_reason, self.full_text = "", False

    def show(self):
        if self.window is None:
            self.window = ui.Window("Evidence", width=820, height=680, flags=ui.WINDOW_FLAGS_NO_DOCKING)
            self.window.set_key_pressed_fn(lambda key,_mods,pressed: self.hide() if pressed and normalize_key(key)==256 else None)
            with self.surface(self.window, "Evidence"):
                self.content = ui.Frame(build_fn=self._build)
        self.window.visible = True;self.window.focus()
        self.refresh(self.action_key, self.go_reason, force=True)

    def refresh(self, action_key=None, go_reason="", *, force=False):
        self.action_key, self.go_reason = action_key, go_reason
        key = action_key, go_reason, self.controller.error, self.controller.pending, self.full_text
        if force or key != self._key:
            self._key = key
            if self.content is not None:self.content.rebuild()

    def _changed(self):
        self.on_changed()

    def _select(self, finding, role=None, frame=None):
        self.controller.select(finding, role, frame);self._changed()

    def _text(self, text, *, muted=False):
        text = str(text)
        shown = text if self.full_text else text[:1500] + ("… [Show full text]" if len(text)>1500 else "")
        ui.Label(shown, name="muted" if muted else "", word_wrap=True, height=0)

    def _build(self):
        controller = self.controller
        self.finding_buttons, self.point_buttons = {}, {}
        with ui.VStack(spacing=8):
            with ui.HStack(height=32, spacing=8):
                self.import_button = ui.Button("Import evidence JSON", identifier="evidence_import", clicked_fn=self.on_import)
                ui.Button("Remove set", enabled=controller.selected_set is not None,
                          clicked_fn=lambda: (controller.remove(controller.selected_set), self._changed()))
                ui.Button("Show less" if self.full_text else "Show full text", clicked_fn=self._toggle_full)
            self._text("Markers verify displayed source samples. Imported comparisons and analytic warnings do not certify a whole capture.", muted=True)
            if controller.error:self._text("Import error: " + controller.error)
            if controller.pending:self._text("Reading and verifying evidence…", muted=True)
            keys = list(controller.sets)
            if not keys:
                self._text("Import SDK evidence v1/v2 or a viewer evidence set. Unknown spans keep the normal timeline.", muted=True)
                return
            with ui.HStack(height=30, spacing=8):
                self.set_combo = ui.ComboBox(keys.index(controller.selected_set), *[f"Set {i+1} · {key[:12]}" for i,key in enumerate(keys)])
                self.set_combo.model.add_item_changed_fn(lambda model,_: self._set(keys[model.get_item_value_model().as_int]))
            stored = controller.sets[controller.selected_set]
            binding = stored.binding
            label={"matching-both":"Capture bytes match both roles; choose a role for navigation.",
                "matching-left":"Capture bytes match the left role.","matching-right":"Capture bytes match the right role.",
                "pending":"Verifying capture samples…","no-capture":"No capture displayed.",
                "no-finalized-payload":"A finalized file is needed for navigation.","missing-binding":"Capture content binding is missing.",
                "mismatch":"Capture content does not match.","changed-payload":"Capture payload changed.",
                "unsupported-format/domain":"Capture format or frame domain is unsupported."}.get(binding.state,binding.state)
            self._text(label + (" " + binding.message if binding.message else ""), muted=True)
            with ui.ScrollingFrame(height=min(160,max(32,32*len(stored.evidence.findings)))):
                with ui.VStack(spacing=4):
                    for finding in stored.evidence.findings:
                        source = finding.data["evidence"] if finding.kind=="comparison" else finding.data["measure"]
                        label = f"{finding.id} · {source['status']} · {source.get('metric',source.get('name',''))}"
                        self.finding_buttons[finding.id] = ui.Button(label[:87]+("…" if len(label)>87 else ""), height=28,
                            clicked_fn=lambda f=finding.id: self._select(f), identifier="evidence_finding_"+finding.id)
            if controller.selection is None:
                self._text("Select a finding to inspect its provenance and choose a capture role.", muted=True)
                return
            finding_id, role, _ = controller.selection
            finding = next(f for f in stored.evidence.findings if f.id==finding_id)
            with ui.HStack(height=32, spacing=8):
                roles = [None, *stored.evidence.captures]
                self.role_combo = ui.ComboBox(roles.index(role) if role in roles else 0,
                                              "Choose capture role", *[r.title() for r in roles[1:]])
                self.role_combo.model.add_item_changed_fn(lambda model,_: self._select(finding_id,roles[model.get_item_value_model().as_int]))
                key = self.action_key
                self.go_button = ui.Button("Go to", identifier="evidence_go", enabled=not self.go_reason,
                    explicit_hover=True, tooltip=self.go_reason or "Pause at this exact verified frame and select its object",
                    clicked_fn=lambda k=key:self.on_go(k))
                self.scene_button = ui.Button("Select scene", identifier="evidence_scene", clicked_fn=self.on_scene)
                self.clips_button = ui.Button("Clips", clicked_fn=self.on_clips)
            self.reason_label = ui.Label(self.go_reason, name="muted", word_wrap=True, height=38)
            marker=controller.selected_marker
            if marker:self._text(f"Recorded {marker.role} sample verified: frame {marker.frame}, source segment {marker.identity['recording_segment']}, object UID {marker.identity['object_uid']}.",muted=True)
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    reasons = binding.reasons.get((finding_id,role),finding.reasons)
                    if reasons:self._text("; ".join(reasons))
                    if finding.kind == "comparison":self._comparison(finding.data["evidence"],role)
                    else:self._scalar(finding,role)

    def _comparison(self, sdk, role):
        self._text("Imported " + sdk["status"] + ": " + sdk["message"])
        self._text(f"{sdk['metric']} · units supplied: {sdk['units']} · distance {sdk['distance']} · tolerance {sdk['tolerance']}")
        self._text(f"Compared samples: {sdk['comparedFrames']}; only the retained pair is marked.", muted=True)
        if sdk["version"]==2:self._text(f"Attribute {sdk['measure']['attributeName']} · {sdk['measure']['distanceKind']} · {sdk['componentCount']} components",muted=True)
        if sdk["textWasReplaced"]:self._text("Producer text was replaced; names or explanatory text may be incomplete.")
        for side in ((role,) if role else ("left","right")):
            sample, capture = sdk[side+"Sample"], sdk[side+"Capture"]
            expected=self.controller.sets[self.controller.selected_set].evidence.captures[side]
            self._text(f"{side.title()} expected SHA256: {expected.sha256 or 'not supplied'} · bytes: {expected.byte_length if expected.byte_length is not None else 'unknown'}",muted=True)
            self._text(f"SDK source: {capture['sourcePath']} · SDK hash metadata: {capture['fileHash'] or 'not supplied'}", muted=True)
            self._text(f"Identity: {dict(sample['identity'])}")
            self._text(f"Requested: {dict(sample['requested'])} · source: {dict(sample['source']) if sample['source'] else 'unknown'}")
            self._text(f"{sample['state']} · raw context {sample['rawContext']} · raw frame {sample['rawFrame'] if sample['rawFrame'] is not None else 'unknown'} · stamp {sample['sampleTimestamp']} · age {sample['age'] if sample['age'] is not None else 'unknown'} · pre-membership {sample['preMembership']}")
            completed = sdk.get(side+"FrameCompleted")
            self._text(f"Retained frame completion: {'unknown' if completed is None else completed} · recorded value: {sdk.get(side+'Value',sdk.get(side+'Velocity'))}")
        self._text("Assumptions: " + sdk["assumptions"], muted=True)

    def _scalar(self, finding, role):
        data = finding.data
        self._text(f"Imported analytic evidence: {data['message']} · {dict(data['measure'])}")
        self._text("Phase labels and reference choices are supplied context; external work and solver correctness are not established.", muted=True)
        self._text(data["assumptions"], muted=True)
        for point in data["points"]:
            frame = uint(point["frame"]["recordedFrame"])
            self.point_buttons[frame] = ui.Button(f"Frame {frame} · {point['phase']} (supplied) · {point['valueJ']} J", height=28,
                clicked_fn=lambda n=frame:self._select(finding.id,role or data["role"],n))
            self._text(str(dict(point["components"])), muted=True)
            for reference in point["observations"]:
                observation = self.controller.sets[self.controller.selected_set].evidence.observations[reference].data
                self._text(f"{observation['attribute']}: {observation['value']} {observation['units']} · {observation['origin']} · {observation['sampleState']} · source {dict(observation['source']) if observation['source'] else 'unknown'}", muted=True)
        for reference in data["supporting"]:
            observation = self.controller.sets[self.controller.selected_set].evidence.observations[reference].data
            self._text(f"Setup {observation['attribute']}: {observation['value']} {observation['units']} · {observation['origin']} · {observation['note']}", muted=True)

    def _set(self, digest):
        self.controller.select_set(digest);self._changed()

    def _toggle_full(self):
        self.full_text = not self.full_text;self.refresh(self.action_key,self.go_reason,force=True)

    def destroy(self):
        if self.window is not None:self.window.destroy()
        self.window = self.content = None

    def hide(self):
        if self.window is not None:self.window.visible=False
