# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed analysis pages and independently owned, closable graph pages.

Tab changes are applied by poll, outside OVUI's draw callbacks. Hidden pages
remain alive; only explicit close or a source invalidation disposes their work.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import omni.ui as ui


FIXED_TABS = (("render", "Render"), ("profile", "Profile"),
              ("memory", "Memory View"), ("errors", "Error Stream"),
              ("threads", "Thread Profiler"))


@dataclass
class Page:
    key: str
    title: str
    host: object
    closable: bool = False
    view: object = None
    source: object = None
    enabled: bool = True
    applicable: bool = True
    reason: str = ""
    wait_key: object = None
    checkable: bool = False
    checking: bool = False
    controls: dict = field(default_factory=dict)
    description: str = ""

    @property
    def selectable(self):
        return self.applicable and (self.enabled or self.checkable)

    @property
    def tooltip(self):
        if self.checkable and not self.checking:
            return "Select to check recorded data for this frame"
        return self.reason or (f"{self.title}\n{self.description}" if self.description else self.title)


class PageHost:
    """The small window interface shared by embedded analysis views.

    Keyboard ownership is explicit. A hidden page never cancels a native job
    and never borrows the main window's navigation flags or keyboard focus.
    """
    embedded = True

    def __init__(self, owner, key, frame):
        self.owner, self.key, self.frame = owner, key, frame
        self.closed = False
        self.key_callback = self.focus_callback = None
        self.content = self.scroll = None
        self.keyboard_focus = False

    @property
    def visible(self):
        return not self.closed and self.owner.active == self.key and getattr(self.owner,"expanded",True)

    @property
    def focused(self):
        return self.visible and self.owner.window.focused and self.keyboard_focus

    @property
    def flags(self):
        return self.owner.window.flags

    @flags.setter
    def flags(self, value):
        self.owner.window.flags = value

    def set_key_pressed_fn(self, callback):
        self.key_callback = callback

    def set_focused_changed_fn(self, callback):
        self.focus_callback = callback

    def focus(self):
        self.owner.select(self.key)
        self.keyboard_focus = True
        self.owner.window.focus()

    def blur(self):
        self.keyboard_focus = False
        if self.focus_callback:
            self.focus_callback(False)

    def destroy(self):
        if self.closed:
            return
        self.blur()
        self.closed = True
        self.frame.visible = False
        self.frame.clear()
        self.key_callback = self.focus_callback = None
        self.content = self.scroll = None

    def resize(self):
        if self.content is not None and self.visible:
            # A narrow center scrolls its controls without pushing either
            # persisted side panel beyond its assigned width.
            self.content.width = ui.Pixel(max(getattr(self,"minimum_width",480), self.frame.computed_width - 24))


@contextmanager
def page_surface(host, title):
    with host.frame:
        with ui.ZStack():
            ui.Rectangle(name="island", opaque_for_mouse_events=False)
            with ui.HStack():
                ui.Spacer(width=12)
                with ui.VStack(spacing=8):
                    ui.Spacer(height=4)
                    with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF) as scroll:
                        host.scroll = scroll
                        with ui.VStack(width=480, spacing=8) as content:
                            host.content = content
                            yield
                    ui.Spacer(height=4)
                ui.Spacer(width=12)
    # Child controls receive their normal events; the page also records focus.
    host.frame.set_mouse_pressed_fn(lambda *_: setattr(host, "keyboard_focus", True))


class WorkspaceTabs:
    def __init__(self, window, on_activate, on_dispose, *, tabs=FIXED_TABS, initial="render"):
        self.window, self.on_activate, self.on_dispose = window, on_activate, on_dispose
        self.pages = {}
        self.initial = initial
        self.active = initial
        self.pending = None
        self.pending_close = []
        self.serial = 0
        self.source = None
        self.buttons = {}
        self.tab_scroll = None
        self.tab_extent = None
        self.reveal_active = False
        self.bar = ui.Frame(height=40, horizontal_clipping=True, vertical_clipping=True,
                            build_fn=self._build_bar)
        self.deck = ui.ZStack(content_clipping=True)
        with self.deck:
            for key, title in tabs:
                frame = ui.Frame(visible=key == initial, horizontal_clipping=True, vertical_clipping=True)
                self.pages[key] = Page(key, title, PageHost(self, key, frame), enabled=key == initial)

    @property
    def render_frame(self):
        return self.pages["render"].host.frame

    def _build_bar(self):
        self.buttons = {}
        with ui.ScrollingFrame(vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF) as scroll:
            self.tab_scroll = scroll
            self.reveal_active = True
            with ui.HStack(height=28, spacing=2):
                for page in self.pages.values():
                    if not page.applicable:
                        continue
                    width = max(62, min(174, len(page.title) * 7 + 16))
                    self.buttons[page.key] = ui.Button(page.title, width=width,
                        name="workspace_tab_active" if page.key == self.active else "workspace_tab",
                        enabled=page.selectable, explicit_hover=True, tooltip=page.tooltip,
                        clicked_fn=lambda key=page.key: self.select(key))
                    if page.closable:
                        ui.Button("×", width=22, name="workspace_tab", explicit_hover=True,
                            tooltip=f"Close {page.title}", clicked_fn=lambda key=page.key: self.close(key))
                ui.Spacer(width=1)

    def select(self, key):
        if key in self.pages and self.pages[key].selectable:
            if key != self.active or self.pages[key].view is None:
                self.pending = key
            return True
        return False

    def close(self, key):
        if key in self.pages and self.pages[key].closable:
            self.pending_close.append(key)
            return True
        return False

    def _dispose(self, page):
        if page.view is not None:
            view, page.view = page.view, None
            self.on_dispose(view)
        page.host.destroy()

    def add(self, title, factory, *, description=""):
        self.serial += 1
        key = f"graph-{self.serial}"
        with self.deck:
            host = PageHost(self, key, ui.Frame(visible=False,
                horizontal_clipping=True, vertical_clipping=True))
        page = Page(key, title, host, closable=True, description=description)
        try:
            page.view = factory(host)
        except BaseException:
            host.destroy()
            raise
        self.pages[key] = page
        self.select(key)
        self.bar.rebuild()
        return page.view

    def sync(self, source, states, *, paused=True, analysis_applicable=True):
        previous = self.source
        self.source = source
        same_timeline = previous is not None and source is not None and previous[:-1] == source[:-1]
        if previous is not None and (source is None or previous[:3] != source[:3]):
            # Capture / recording segment replacement invalidates every source.
            for key in tuple(self.pages):
                if self.pages[key].closable:
                    self.close(key)
            self.select("render")
        applicability_changed = False
        for key, page in self.pages.items():
            if key == "render" or page.closable:
                continue
            if page.applicable != analysis_applicable:
                page.applicable = analysis_applicable
                applicability_changed = True
            if not page.applicable and (self.active == key or self.pending == key):
                self.select("render")
            if not same_timeline:
                page.controls.clear()
            state = states["profile" if key == "threads" else key]
            page.enabled, page.reason = state.enabled, state.reason
            page.checkable = getattr(state,"requestable",False)
            page.checking = getattr(state,"checking",False)
            button = self.buttons.get(key)
            if button is not None:
                button.enabled, button.tooltip = page.selectable, page.tooltip
            if page.view is not None and page.source != source:
                # Keep only user controls within this capture/segment/scene.
                # Results and jobs still belong to one exact frame and are disposed.
                if same_timeline:
                    save = getattr(page.view, "get_workspace_state", None)
                    if save is not None:
                        page.controls = save()
                self._dispose(page)
                page.host.closed = False
                page.host.frame.visible = key == self.active
            wait_key = (source, page.enabled, page.checkable, page.reason, paused)
            if key == self.active and page.view is None and self.pending is None and page.wait_key != wait_key:
                self.pending = key
            page.wait_key = wait_key
        if applicability_changed:
            self.bar.rebuild()

    def blur(self):
        for page in self.pages.values():
            page.host.blur()

    def key(self, key, modifiers, pressed):
        host = self.pages[self.active].host
        if host.focused and host.key_callback:
            host.key_callback(key, modifiers, pressed)
            chord = modifiers & 15
            return ((chord == 0 and key in (257,335,256,268,262,263,264,265,45,333,334))
                    or (chord in (0,1) and key == 61) or (chord == 1 and key in (73,79)))
        return False

    @property
    def navigation_focused(self):
        page = self.pages[self.active]
        view = page.view
        return (view is not None and page.host.focused
                and not getattr(view,"editing_fields",())
                and not getattr(view,"search_editing",False)
                and ui.Menu.get_current() is None)

    def poll(self):
        for key in self.pending_close:
            page = self.pages.get(key)
            if page is None:
                continue
            keys = list(self.pages)
            adjacent = keys[max(0, keys.index(key) - 1)]
            self._dispose(page)
            del self.pages[key]
            if self.active == key and self.pending in (None, key):
                self.pending = adjacent if self.pages[adjacent].selectable else "render"
            self.bar.rebuild()
        self.pending_close.clear()
        key, self.pending = self.pending, None
        if key is not None and key in self.pages:
            self.blur()
            self.active = key
            self.reveal_active = True
            for candidate in self.pages.values():
                candidate.host.frame.visible = candidate.key == key
                button = self.buttons.get(candidate.key)
                if button is not None:
                    button.name = "workspace_tab_active" if candidate.key == key else "workspace_tab"
            page = self.pages[key]
            self.on_activate(page)
        for page in self.pages.values():
            page.host.resize()
        if self.tab_scroll is not None and self.tab_extent != self.tab_scroll.computed_width:
            self.tab_extent = self.tab_scroll.computed_width
            self.reveal_active = True
        if self.reveal_active and self.tab_scroll is not None:
            button = self.buttons.get(self.active)
            if button is not None and button.computed_width > 0:
                scroll = self.tab_scroll
                left = button.screen_position_x - scroll.screen_position_x
                right = left + button.computed_width + (24 if self.pages[self.active].closable else 0)
                if left < 0:
                    scroll.scroll_x = max(0, scroll.scroll_x + left)
                elif right > scroll.computed_width:
                    scroll.scroll_x += right - scroll.computed_width
                self.reveal_active = False

    def get_inspector_state(self):
        return {"active": self.active, "pages": [
            {"key": p.key, "title": p.title, "closable": p.closable,
             "enabled": p.enabled, "reason": p.reason, "loaded": p.view is not None,
             "selectable": p.selectable, "checkable": p.checkable,
             "checking": p.checking, "description": p.description,
             "ready": bool(p.view and getattr(p.view,"finished",False)),
             "source": [str(v) for v in p.source] if p.source else None}
            for p in self.pages.values()]}

    def destroy(self):
        for page in tuple(self.pages.values()):
            self._dispose(page)
        self.pages.clear()
