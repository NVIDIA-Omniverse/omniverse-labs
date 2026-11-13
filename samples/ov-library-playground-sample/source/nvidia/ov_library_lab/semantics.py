# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SimReady semantic labels for prims the application builds at run time.

Campus artwork is labeled by data/scene-overrides/r52-semantic-labels.usda. Models the
application references at run time get their labels here, at the reference site, using
the same taxonomy (config/semantic-taxonomy.json) and the SimReady convention:
SemanticsLabelsAPI:wikidata_qcode plus SemanticsLabelsAPI:wikidata_class.
"""
import json
from functools import lru_cache
from pathlib import Path

from pxr import UsdSemantics

TAXONOMY = Path(__file__).resolve().parents[3]/'config/semantic-taxonomy.json'


@lru_cache(maxsize=1)
def _classes():
    return json.loads(TAXONOMY.read_text(encoding='utf-8'))['classes']


def label(prim, name):
    """Label prim (and, by inheritance, its descendants) with a taxonomy class."""
    entry = _classes()[name]
    for instance, value in (('wikidata_qcode', entry['qcode']), ('wikidata_class', entry['wikidata_label'])):
        UsdSemantics.LabelsAPI.Apply(prim, instance).CreateLabelsAttr([value])
    return prim
