#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create deterministic benchmark fixtures for the prototype."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent / "fixtures"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def large_flat_prim(count: int = 10000) -> str:
    lines = ["#usda 1.0", ""]
    for i in range(count):
        lines.append(f'def "Prim_{i:05d}"')
        lines.append("{")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def many_attributes(count: int = 10000) -> str:
    lines = ["#usda 1.0", "", 'def "Root"', "{"]
    for i in range(count):
        lines.append(f"    float attr_{i:05d} = {i}.0")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    write(ROOT / "tiny.usda", '#usda 1.0\n\ndef Xform "World"\n{\n    float size = 1.0\n}\n')
    write(ROOT / "large_flat_prim.usda", large_flat_prim())
    write(ROOT / "many_attributes.usda", many_attributes())
    print(f"Wrote fixtures under {ROOT}")


if __name__ == "__main__":
    main()
