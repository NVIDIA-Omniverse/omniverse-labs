#!/usr/bin/env python3
"""Generated-at-validate-time scene synthesis for behavioral parser checks.

The other behavioral checks (cross_format_check, filename_rename_check,
usda_rename_invariance, usdc_rename_invariance) all start from fixtures
committed to the repo. The regen agent sees those bytes at code-generation
time, so it can in principle memorize a canned dump per fixture and dispatch
by fingerprint. The randomized rename/value checks defeat *that* strategy
when the fingerprint stops matching after mutation, but only if the
fall-through path is itself a real parser; a sufficiently adversarial agent
could plant another canned dump in the fall-through.

Synthesized scenes break this entirely. The harness owns a small `Scene`
data model, generates random scenes at validate time, emits each as a
.usda file, and asserts the parser's dump equals a separately-computed
expected dump derived from the same Scene object. The agent never sees
these scenes at codegen time, so there's nothing to fingerprint. The
only winning strategy is to implement a real parser.

Two emitter implementations are provided:

  - `PureStdLibEmitter` (always available). f-string templating for the
    USDA text; hand-coded JSON construction for the expected dump.

  - `UsdCoreEmitter` (used when `pxr` is importable). Builds an
    `Sdf.Layer` in memory via the Pixar Python API, exports to USDA text,
    and traverses the layer to produce the canonical dump JSON. This is
    closer to a reference implementation since pxr is what the spec
    authors maintain. Independent of `PureStdLibEmitter`, so when both
    are available the harness can run a parity check and surface drift.

`get_emitter()` returns whichever is available, preferring usd-core. The
`Scene` model is the intersection of what both emitters support; growing
coverage means growing both implementations together.

MVP scope: root prims and nested child prims with scalar double attribute
defaults. Future expansions (variants, listOps, layer metadata, time
samples) extend both emitters together.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Protocol

# Prim typeName pool. All four are concrete schemas in USD that accept
# arbitrary attributes; they're interchangeable for the parser test.
_PRIM_TYPE_POOL = ["Xform", "Scope", "Cube", "Sphere"]

# Attribute typeName pool. MVP: just double. `double` is chosen over `float`
# because pxr stores attribute defaults at the type's native precision; an
# f32 default loses information versus a 4-decimal Python float, which would
# cause the two emitters' dumps to disagree on values like -80.6567. Doubles
# round-trip exactly for the 4-decimal values our random generator produces.
_ATTR_TYPE_POOL = ["double"]


# ---------------------------------------------------------------------------
# Scene data model.
# ---------------------------------------------------------------------------


@dataclass
class Attribute:
    name: str
    type_name: str  # "double" (MVP)
    default: float  # MVP: all attributes have a default


@dataclass
class Prim:
    name: str
    type_name: str
    children: list["Prim"] = field(default_factory=list)
    attributes: list[Attribute] = field(default_factory=list)


@dataclass
class Scene:
    prims: list[Prim]


# ---------------------------------------------------------------------------
# Random scene generation.
# ---------------------------------------------------------------------------


def _random_prim_name(rng: random.Random) -> str:
    """5-char prim name starting with uppercase ASCII. Avoids accidental
    collisions with USDA reserved keywords (which are lowercase)."""
    first = rng.choice(string.ascii_uppercase)
    rest = "".join(rng.choices(string.ascii_letters + string.digits, k=4))
    return first + rest


def _random_attr_name(rng: random.Random) -> str:
    """4-char attribute name starting with lowercase ASCII."""
    first = rng.choice(string.ascii_lowercase)
    rest = "".join(rng.choices(string.ascii_lowercase + string.digits, k=3))
    return first + rest


def _gen_attribute(rng: random.Random) -> Attribute:
    return Attribute(
        name=_random_attr_name(rng),
        type_name=rng.choice(_ATTR_TYPE_POOL),
        default=round(rng.uniform(-100.0, 100.0), 4),
    )


def _gen_prim(
    rng: random.Random,
    depth: int,
    max_depth: int,
    attr_prob: float = 0.5,
    child_prob: float = 0.5,
    used_names: set[str] | None = None,
) -> Prim:
    if used_names is None:
        used_names = set()
    # Pick a name not already used among siblings.
    for _ in range(20):
        name = _random_prim_name(rng)
        if name not in used_names:
            break
    used_names.add(name)
    type_name = rng.choice(_PRIM_TYPE_POOL)

    attributes: list[Attribute] = []
    if rng.random() < attr_prob:
        n_attrs = rng.randint(1, 3)
        attr_names: set[str] = set()
        for _ in range(n_attrs):
            attr = _gen_attribute(rng)
            if attr.name in attr_names:
                continue
            attr_names.add(attr.name)
            attributes.append(attr)

    children: list[Prim] = []
    if depth < max_depth and rng.random() < child_prob:
        n_children = rng.randint(1, 2)
        child_names: set[str] = set()
        for _ in range(n_children):
            child = _gen_prim(rng, depth + 1, max_depth, attr_prob, child_prob, child_names)
            children.append(child)

    return Prim(name=name, type_name=type_name, children=children, attributes=attributes)


def generate_scene(
    rng: random.Random,
    n_roots_range: tuple[int, int] = (1, 4),
    max_depth: int = 3,
    attr_prob: float = 0.5,
    child_prob: float = 0.5,
) -> Scene:
    """Generate a random Scene. Defaults produce 1-4 root prims, up to 3
    levels of nesting, with ~50% probability of attributes and ~50%
    probability of children at each prim."""
    n_roots = rng.randint(*n_roots_range)
    prims: list[Prim] = []
    used_names: set[str] = set()
    for _ in range(n_roots):
        prims.append(_gen_prim(rng, depth=1, max_depth=max_depth,
                               attr_prob=attr_prob, child_prob=child_prob,
                               used_names=used_names))
    return Scene(prims=prims)


# ---------------------------------------------------------------------------
# Emitter protocol.
# ---------------------------------------------------------------------------


class SceneEmitter(Protocol):
    """Methods that compute the USDA text and the expected dump from a
    Scene. Implementations must produce a USDA text that the regen parser
    will read, and a dump dict in the canonical layer-dump schema format.

    USDC emission is optional: emitters that support it set
    `supports_usdc = True` and implement `emit_usdc(scene) -> bytes`.
    Pure-stdlib emitters cannot produce USDC (the binary format requires
    LZ4 + Crate-section encoding, which is impractical without a USD
    library); the harness tests USDC only when an emitter advertises
    `supports_usdc`.
    """

    name: str  # "purestdlib" or "usdcore"; used in diagnostics
    supports_usdc: bool

    def emit_usda(self, scene: Scene) -> str:
        ...

    def expected_dump(self, scene: Scene) -> dict[str, Any]:
        ...

    def emit_usdc(self, scene: Scene) -> bytes:
        """Raises NotImplementedError when `supports_usdc` is False."""
        ...


# ---------------------------------------------------------------------------
# Pure-stdlib emitter. Always available.
# ---------------------------------------------------------------------------


def _format_double_for_usda(v: float) -> str:
    """Format a double so the USDA token unambiguously parses as a float
    (not an int). Trims trailing zeros but keeps at least one fractional
    digit so the token contains a decimal point.
    """
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def _sorted_attrs(attrs: list[Attribute]) -> list[Attribute]:
    """Return attributes sorted by name.

    Why sort: pxr's `Sdf.Layer.Export()` alphabetizes attributes when
    writing USDA text, and the regen's USDA parser preserves text order.
    pxr's USDC writer (and the regen's USDC parser) preserves authoring
    order. By authoring attributes in sorted order on the Python side,
    USDA text, USDC binary, and our `expected_dump` all converge on
    alphabetical-by-attribute-name -- which is the only ordering all
    three components agree on. Avoids spurious synthesized_scenes
    DIFFs that aren't parser bugs.
    """
    return sorted(attrs, key=lambda a: a.name)


class PureStdLibEmitter:
    """Stdlib-only emitter: f-string USDA + hand-coded canonical dump JSON.

    Does not support USDC emission: implementing a Crate-binary writer in
    pure Python (LZ4 + integer-encoded paths + multiple section formats)
    would dwarf the rest of this module and be a fragile source of false
    negatives. The harness skips USDC synthesis in environments without
    `pxr`; install usd-core for full coverage.
    """

    name = "purestdlib"
    supports_usdc = False

    def emit_usda(self, scene: Scene) -> str:
        lines = ["#usda 1.0", ""]
        for prim in scene.prims:
            lines.extend(self._emit_prim(prim, indent=0))
            lines.append("")  # blank between roots
        return "\n".join(lines).rstrip() + "\n"

    def _emit_prim(self, prim: Prim, indent: int) -> list[str]:
        pad = "    " * indent
        inner_pad = "    " * (indent + 1)
        out = [f'{pad}def {prim.type_name} "{prim.name}"', f"{pad}{{"]
        # Emit attributes in sorted-by-name order so the USDA text matches
        # what pxr produces (which alphabetizes on export). See
        # `_sorted_attrs` rationale at the top of this file.
        for attr in _sorted_attrs(prim.attributes):
            out.append(
                f"{inner_pad}{attr.type_name} {attr.name} "
                f"= {_format_double_for_usda(attr.default)}"
            )
        if prim.attributes and prim.children:
            out.append("")  # readability separator
        for child in prim.children:
            out.extend(self._emit_prim(child, indent + 1))
        out.append(f"{pad}}}")
        return out

    def expected_dump(self, scene: Scene) -> dict[str, Any]:
        root_names = [p.name for p in scene.prims]
        specs: dict[str, Any] = {
            "/": {
                "kind": "layer",
                "fields": {
                    "primChildren": {"type": "token[]", "value": root_names},
                },
            }
        }
        for prim in scene.prims:
            self._emit_prim_specs(prim, parent_path="", specs=specs)
        return {
            "ok": True,
            "diagnostics": [],
            "layer": {
                "fields": {
                    "primChildren": {"type": "token[]", "value": root_names},
                },
                "specs": specs,
            },
        }

    def _emit_prim_specs(self, prim: Prim, parent_path: str, specs: dict[str, Any]) -> None:
        path = f"{parent_path}/{prim.name}"
        sorted_attrs = _sorted_attrs(prim.attributes)
        specs[path] = {
            "kind": "prim",
            "fields": {
                "specifier": {"type": "specifier", "value": "def"},
                "typeName": {"type": "token", "value": prim.type_name},
                "primChildren": {
                    "type": "token[]",
                    "value": [c.name for c in prim.children],
                },
                "propertyChildren": {
                    "type": "token[]",
                    "value": [a.name for a in sorted_attrs],
                },
                "variantSetChildren": {"type": "token[]", "value": []},
            },
        }
        for attr in sorted_attrs:
            attr_path = f"{path}.{attr.name}"
            specs[attr_path] = {
                "kind": "attribute",
                "fields": {
                    "custom": {"type": "bool", "value": False},
                    "variability": {"type": "variability", "value": "varying"},
                    "typeName": {"type": "token", "value": attr.type_name},
                    "default": {"type": attr.type_name, "value": attr.default},
                },
            }
        for child in prim.children:
            self._emit_prim_specs(child, path, specs)

    def emit_usdc(self, scene: Scene) -> bytes:
        raise NotImplementedError(
            "PureStdLibEmitter cannot author USDC. Install usd-core "
            "(e.g. into a venv) and use UsdCoreEmitter for USDC synthesis."
        )


# ---------------------------------------------------------------------------
# UsdCore emitter. Lazy-imported; only used when pxr is available.
# ---------------------------------------------------------------------------


class UsdCoreEmitter:
    """Reference-style emitter backed by `pxr.Sdf`.

    Builds an SdfLayer in memory and serves it three ways:

      - `emit_usda`: exports to USDA text via `Sdf.Layer.ExportToString()`.
      - `emit_usdc`: exports to USDC bytes via `Sdf.Layer.Export("*.usdc")`
        and reads the resulting binary. Closes the cheat shape for USDC
        parsers: novel binary inputs the agent has never seen force a real
        decoder path.
      - `expected_dump`: traverses the SdfLayer to produce the canonical
        dump JSON. Independent of `PureStdLibEmitter.expected_dump`, so
        agreement between the two paths is a real signal.
    """

    name = "usdcore"
    supports_usdc = True

    def __init__(self) -> None:
        # Lazy import; do not import at module load.
        from pxr import Sdf  # noqa: F401
        self._Sdf = Sdf

    def _build_layer(self, scene: Scene) -> Any:
        Sdf = self._Sdf
        layer = Sdf.Layer.CreateAnonymous(".usda")
        for prim in scene.prims:
            self._author_prim(layer, "", prim)
        return layer

    def _author_prim(self, layer: Any, parent_path: str, prim: Prim) -> None:
        Sdf = self._Sdf
        path = f"{parent_path}/{prim.name}"
        spec_path = Sdf.Path(path)
        prim_spec = Sdf.CreatePrimInLayer(layer, spec_path)
        prim_spec.specifier = Sdf.SpecifierDef
        prim_spec.typeName = prim.type_name
        for attr in _sorted_attrs(prim.attributes):
            # MVP: all attributes are scalar double. See `_ATTR_TYPE_POOL`.
            # Author in sorted order so the USDC binary (which preserves
            # authoring order) ends up alphabetical, matching what pxr's
            # USDA export does naturally.
            attr_spec = Sdf.AttributeSpec(
                prim_spec, attr.name, Sdf.ValueTypeNames.Double
            )
            attr_spec.default = float(attr.default)
        for child in prim.children:
            self._author_prim(layer, path, child)

    def emit_usda(self, scene: Scene) -> str:
        layer = self._build_layer(scene)
        text = layer.ExportToString()
        # Sdf does not always include the trailing newline pxr expects on
        # disk; harness writes always normalize endings, but make sure.
        if not text.endswith("\n"):
            text += "\n"
        return text

    def emit_usdc(self, scene: Scene) -> bytes:
        """Author the scene into an SdfLayer and export as USDC binary
        bytes. pxr selects the Crate writer based on the `.usdc` suffix.
        """
        import os
        import tempfile
        layer = self._build_layer(scene)
        # Sdf.Layer.Export(path) writes to disk; use a temp file then read
        # it back. Note: Export may emit slightly different bytes from
        # run to run (timestamp / file-id fields), but the *contents* the
        # parser cares about are deterministic for a given Scene.
        with tempfile.NamedTemporaryFile(suffix=".usdc", delete=False) as f:
            tmp_path = f.name
        try:
            ok = layer.Export(tmp_path)
            if not ok:
                raise RuntimeError("Sdf.Layer.Export returned False")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def expected_dump(self, scene: Scene) -> dict[str, Any]:
        """Traverse the authored SdfLayer to produce the canonical dump.

        Mirrors the same canonical schema as `PureStdLibEmitter.expected_dump`,
        but builds it from the SdfLayer directly so the two paths are
        structurally independent.
        """
        Sdf = self._Sdf
        layer = self._build_layer(scene)
        root_names = [p.name for p in scene.prims]
        specs: dict[str, Any] = {
            "/": {
                "kind": "layer",
                "fields": {
                    "primChildren": {"type": "token[]", "value": root_names},
                },
            }
        }

        def walk_prim_spec(prim_spec: Any) -> None:
            path = str(prim_spec.path)
            child_names = [c.name for c in prim_spec.nameChildren]
            # Sort attribute spec names alphabetically; matches the order
            # the regen parser produces from USDA text (pxr alphabetizes
            # on Export) and from USDC binary (since `_author_prim`
            # authors in sorted order too).
            attr_specs = sorted(prim_spec.attributes, key=lambda a: a.name)
            attr_names = [a.name for a in attr_specs]
            specs[path] = {
                "kind": "prim",
                "fields": {
                    "specifier": {"type": "specifier", "value": "def"},
                    "typeName": {"type": "token", "value": prim_spec.typeName},
                    "primChildren": {"type": "token[]", "value": child_names},
                    "propertyChildren": {"type": "token[]", "value": attr_names},
                    "variantSetChildren": {"type": "token[]", "value": []},
                },
            }
            for attr_spec in attr_specs:
                attr_path = str(attr_spec.path)
                specs[attr_path] = {
                    "kind": "attribute",
                    "fields": {
                        "custom": {"type": "bool", "value": False},
                        "variability": {"type": "variability", "value": "varying"},
                        "typeName": {"type": "token", "value": "double"},
                        "default": {"type": "double", "value": float(attr_spec.default)},
                    },
                }
            for child in prim_spec.nameChildren:
                walk_prim_spec(child)

        for root_name in root_names:
            walk_prim_spec(layer.GetPrimAtPath(Sdf.Path(f"/{root_name}")))

        return {
            "ok": True,
            "diagnostics": [],
            "layer": {
                "fields": {
                    "primChildren": {"type": "token[]", "value": root_names},
                },
                "specs": specs,
            },
        }


# ---------------------------------------------------------------------------
# Emitter factory.
# ---------------------------------------------------------------------------


def get_emitter(prefer: str | None = None) -> SceneEmitter:
    """Return the preferred emitter, or the pure-stdlib fallback.

    If `prefer` is "usdcore" or "purestdlib", try that one. Otherwise prefer
    usd-core if pxr is importable, else fall back to pure-stdlib.
    """
    if prefer == "purestdlib":
        return PureStdLibEmitter()
    if prefer == "usdcore":
        return UsdCoreEmitter()  # raises ImportError if pxr is missing
    try:
        return UsdCoreEmitter()
    except ImportError:
        return PureStdLibEmitter()


def usdcore_available() -> bool:
    """Return True iff pxr.Sdf is importable in the current environment."""
    try:
        import pxr.Sdf  # noqa: F401
        return True
    except ImportError:
        return False
