"""Selected runtime readiness must gate activation without a SIL fallback."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'addon'))
from ovrtx_blender_example import bundled_runtime, engine, public_sdk_install

@pytest.mark.parametrize('state', ['unconfigured', 'offline', 'failed', 'cancelled'])
def test_missing_public_runtime_is_actionable(monkeypatch, state):
    monkeypatch.setattr(bundled_runtime, 'public_sdk_selected', lambda: True)
    monkeypatch.setattr(bundled_runtime, 'public_sdk_python', lambda: None)
    monkeypatch.setattr(public_sdk_install, 'status', lambda: {'state': state, 'message': 'Select sidecar'})
    with pytest.raises(engine.RenderClientError, match='Select sidecar'):
        engine._require_viewport_runtime_ready()


def test_install_in_progress_defers_activation(monkeypatch):
    monkeypatch.setattr(bundled_runtime, 'public_sdk_selected', lambda: True)
    monkeypatch.setattr(bundled_runtime, 'public_sdk_python', lambda: None)
    monkeypatch.setattr(public_sdk_install, 'status', lambda: {'state': 'installing', 'message': 'Copying sidecar'})
    with pytest.raises(engine.RuntimeServicesPreparingError, match='Copying sidecar'):
        engine._require_viewport_runtime_ready()


def test_ready_public_runtime_does_not_wait_for_sil(monkeypatch):
    monkeypatch.setattr(bundled_runtime, 'public_sdk_selected', lambda: True)
    monkeypatch.setattr(bundled_runtime, 'public_sdk_python', lambda: '/ready/python')
    engine._require_viewport_runtime_ready()
