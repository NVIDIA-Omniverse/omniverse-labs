# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The F12 supersampling tooltip states the measured cost (build 29 review, line-and-render-fixes n-05-02).

The default stays 2x2. The tooltip used to say "+20% F12 time", which understates re-render cost
(Classroom, n-05-01: re-render ~2.2x, edit re-render ~1.4x, first render ~+5% against 1x).
"""

from __future__ import annotations

import ast
from pathlib import Path

PROPERTIES = Path(__file__).resolve().parents[1] / "addon/ovrtx_blender_example/properties.py"


def _final_supersample_keywords():
    for node in ast.walk(ast.parse(PROPERTIES.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "final_supersample"):
            return {kw.arg: ast.literal_eval(kw.value) for kw in node.annotation.keywords
                    if kw.arg in ("default", "description")}
    raise AssertionError("final_supersample property not found")


def test_tooltip_states_rerender_cost() -> None:
    keywords = _final_supersample_keywords()
    description = keywords["description"]
    assert keywords["default"] == "2"
    assert "+20%" not in description
    assert "re-render takes about 2.2x" in description
    assert "edit re-render about 1.4x" in description
    assert "+5%" in description
    assert "4x the pixels" in description
    assert "1x is fastest" in description
