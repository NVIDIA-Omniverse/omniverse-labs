"""Native cancellation acknowledgements never release active stage readers."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from public_sdk_backend import RendererBackend


class Operation:
    ready = False
    def __init__(self): self.waits = []
    def wait(self, *, timeout_ns):
        self.waits.append(timeout_ns)
        return self if self.ready else None
    def fetch(self, *, timeout_ns):
        assert timeout_ns == 0
        return {} if self.ready else None


def test_cancel_ack_is_not_completion_and_wait_budget_is_bounded():
    backend = RendererBackend.__new__(RendererBackend)
    first, other = Operation(), Operation()
    backend.pending = {token: {'operation': op, 'cancelled': token == 'other'} for token, op in [('first', first), ('other', other)]}
    ack = backend.dispatch('cancel_render', {'ticket': 'first'})
    assert ack == {'cancelled': True, 'native_complete': False}
    with pytest.raises(ValueError, match='completed native samples'): backend._require_idle_native_render()
    ack = backend.dispatch('cancel_render', {'ticket': 'first', 'timeout_ns': 999_000_000})
    assert ack['native_complete'] is False
    assert max(first.waits) == 8_000_000 and set(other.waits) == {0}
    first.ready = True
    assert backend.dispatch('cancel_render', {'ticket': 'first'})['native_complete'] is True
    assert 'first' not in backend.pending and 'other' in backend.pending
    with pytest.raises(ValueError, match='completed native samples'): backend._require_idle_native_render()
    other.ready = True
    backend._require_idle_native_render()
    assert not backend.pending


def test_completed_products_cancel_without_native_wait_or_publication():
    backend = RendererBackend.__new__(RendererBackend)
    op = Operation()
    backend.pending = {'packed': {'operation': op, 'cancelled': False, 'products': {}}}
    assert backend.dispatch('cancel_render', {'ticket': 'packed'})['native_complete'] is True
    assert op.waits == [] and not backend.pending
