# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real About controller actions and disposable OS handler; no message is sent."""
import time
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from ovd_ui import __version__, desktop
from ovd_ui.app_info import parse_about_info, MANUAL_STATUS
from ovd_ui.clipboard import ClipboardError
from ovd_ui.widgets.about_view import AboutView
from desktop_test_handler import install_handler, missing_handler


@pytest.fixture(autouse=True)
def prevent_real_desktop_handler(monkeypatch):
    monkeypatch.setattr(desktop, '_launch', missing_handler)


def view(**kwargs):
    result = AboutView(parse_about_info(None), **kwargs)
    result.window = SimpleNamespace(visible=True, focus=lambda: None, destroy=lambda: None)
    result.status_label = SimpleNamespace(text="")
    result.support_button = SimpleNamespace(enabled=True)
    return result


def test_copy_uses_exact_visible_snapshot_and_reports_desktop_failure():
    copied, errors = [], []
    dialog = view(on_copy_text=copied.append, on_error=errors.append)
    assert dialog.copy_info()
    assert copied == [dialog.info.copy_text]
    assert dialog.status_label.text == "Displayed information copied."
    def fail(_): raise ClipboardError("No desktop clipboard")
    dialog.on_copy_text = fail
    assert not dialog.copy_info()
    assert errors == ["No desktop clipboard"] and dialog.window.visible
    assert "not included" in MANUAL_STATUS


def test_guide_closes_about_and_escape_does_not_trigger_an_action():
    opened = []
    dialog = view(on_workspace_guide=lambda: opened.append(True))
    dialog.key(256, 0, False)
    assert dialog.window.visible
    dialog.key(256, 0, True)
    assert not dialog.window.visible and not opened
    dialog.show()
    dialog.show_guide()
    assert opened == [True] and not dialog.window.visible


class Request:
    def __init__(self): self.done = self.closed = False
    def poll(self): return self.done
    def close(self): self.closed = True


def test_compose_is_single_pending_request_and_completion_is_reported_when_hidden():
    request = Request()
    calls = []
    dialog = view(compose_request=lambda: calls.append(True) or request)
    assert dialog.report_issue() and not dialog.report_issue() and calls == [True]
    assert not dialog.support_button.enabled
    dialog.hide()
    dialog.refresh()
    assert dialog.request is request
    request.done = True
    dialog.refresh()
    assert request.closed and dialog.request is None and dialog.support_button.enabled
    assert "No message has been sent" in dialog.status


def test_handler_errors_allow_retry_and_closed_view_cannot_start_work():
    errors = []
    def missing(): raise desktop.DesktopError("No desktop handler")
    dialog = view(compose_request=missing, on_error=errors.append)
    assert not dialog.report_issue() and dialog.request is None
    assert errors == ["No desktop handler"] and dialog.support_button.enabled
    request = Request()
    def fail(): raise desktop.DesktopError("No email application")
    request.poll = fail
    dialog.compose_request = lambda: request
    assert dialog.report_issue()
    dialog.refresh()
    assert request.closed and dialog.request is None and dialog.support_button.enabled
    dialog.compose_request = Request
    assert dialog.report_issue()
    owned = dialog.request
    dialog.destroy()
    assert owned.closed and dialog.window is None
    assert not dialog.report_issue() and not dialog.copy_info()
    dialog.refresh()


def test_desktop_request_uses_only_fixed_compose_uri_with_disposable_handler(tmp_path, monkeypatch):
    # This temporary executable replaces the desktop handler, records argv and
    # exits. It never invokes a browser, composer, network API or mail sender.
    log = install_handler(tmp_path, monkeypatch)
    request = desktop.SupportComposeRequest()
    deadline = time.monotonic() + 2
    while not request.poll():
        assert time.monotonic() < deadline
        time.sleep(.005)
    request.close()
    assert json.loads(log.read_text()) == [desktop.SUPPORT_URI]
    uri = urlparse(desktop.SUPPORT_URI)
    assert uri.scheme == "mailto" and uri.path == "pvdsupport@nvidia.com"
    assert parse_qs(uri.query) == {"subject": ["PVD Issue"]}


def test_missing_and_rejected_desktop_handler_are_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(desktop.DesktopError, match="No desktop email handler"):
        desktop.SupportComposeRequest()
    install_handler(tmp_path, monkeypatch, 'sys.exit(3)')
    request = desktop.SupportComposeRequest()
    deadline = time.monotonic() + 2
    with pytest.raises(desktop.DesktopError, match="exit 3"):
        while not request.poll():
            assert time.monotonic() < deadline
            time.sleep(.005)
    request.close()


def test_close_detaches_desktop_composer_instead_of_terminating_it(monkeypatch):
    calls = []
    process = SimpleNamespace(poll=lambda: None, wait=lambda: calls.append("wait"))
    request = desktop.SupportComposeRequest.__new__(desktop.SupportComposeRequest)
    request._process = process
    class Thread:
        def __init__(self, *, target, name, daemon):
            assert target == process.wait and daemon
            self.target = target
        def start(self): self.target()
    monkeypatch.setattr(desktop, "Thread", Thread)
    request.close()
    assert calls == ["wait"] and request._process is None


def test_application_reuses_about_and_guide_uses_canonical_version(monkeypatch):
    from ovd_ui import application
    shown, options = [], []
    def make(**kwargs):
        options.append(kwargs)
        return SimpleNamespace(show=lambda: shown.append(True))
    monkeypatch.setattr(application, "AboutView", make)
    monkeypatch.setattr(application, "WidgetCatalogue", make)
    app = application.Application.__new__(application.Application)
    app.show_about()
    app.show_about()
    assert len(options) == 1 and len(shown) == 2
    assert options[0]["on_workspace_guide"] == app.show_catalogue
    app.show_catalogue()
    assert options[1]["version"] == f"v{__version__} · migration preview"
