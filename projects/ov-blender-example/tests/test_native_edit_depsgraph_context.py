# SPDX-License-Identifier: Apache-2.0
"""Callback translation retains evaluated graph and geometry update evidence."""
from pathlib import Path
from types import SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "addon"))
from ovrtx_blender_example.blender_signal_translation import InteractiveEditTranslator
from ovrtx_blender_example.blender_signals import BlenderEditSignal, BlenderEditSignalSource
from ovrtx_blender_example.blender_callback_adapters import BlenderEditCallbackAdapter


def test_real_depsgraph_geometry_flags_and_evaluation_context_survive_translation():
    obj = object()
    update = SimpleNamespace(id=obj, is_updated_geometry=True, is_updated_transform=False)
    graph = SimpleNamespace(updates=(update,), scene_eval=object())
    captured = []

    def builder(received, **kwargs):
        captured.append(received)
        assert received.updates[0].is_updated_geometry
        assert not received.updates[0].is_updated_transform
        assert received.scene_eval is graph.scene_eval
        return []

    adapter = BlenderEditCallbackAdapter(edit_builder=builder)
    assert adapter.translate_depsgraph_edits(graph, selection_resolution={}) == ()
    assert captured == [graph]
