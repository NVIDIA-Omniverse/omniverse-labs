# Canonical Layer Dump Contract

Generated single-layer targets must expose a command that accepts a single
layer resource path and writes one JSON document to stdout.

Current scopes may provide USDA text resources or USDC binary resources. The
resource path is still a single local layer input; package entries, sublayers,
references, payloads, and composition are outside this contract.

The JSON document must match `contracts/layer-dump.schema.json`.

## Successful Parse

```json
{
  "ok": true,
  "diagnostics": [],
  "layer": {
    "fields": {
      "primChildren": {"type": "token[]", "value": ["World"]}
    },
    "specs": {
      "/": {
        "kind": "layer",
        "fields": {
          "primChildren": {"type": "token[]", "value": ["World"]}
        }
      },
      "/World": {
        "kind": "prim",
        "fields": {
          "specifier": {"type": "specifier", "value": "def"},
          "typeName": {"type": "token", "value": "Xform"}
        }
      }
    }
  }
}
```

The layer spec must appear at path `/`. Its fields should also be mirrored in
`layer.fields` for convenience.

## Failed Parse

```json
{
  "ok": false,
  "diagnostics": [
    {
      "severity": "error",
      "code": "DuplicateSpecPath",
      "message": "duplicate spec path",
      "path": "/Foo"
    }
  ]
}
```

Failed parses must not include a partial `layer` object unless a future scope
explicitly adds partial-output semantics.

## Field Values

Every authored field is represented as:

```json
{"type": "token", "value": "Xform"}
```

Use the document-model field name, not the USDA syntax name, in field maps. For
example:

- `reorder rootPrims` maps to `primOrder`
- prim specifier maps to `specifier`
- attribute declaration type maps to `typeName`
- attribute default assignment maps to `default`
- relationship targets map to `targetPaths`

## Ordering

Object member ordering in JSON is ignored by the scorer. Array ordering is
semantic and must match exactly.

## Higher-level representation rules

The canonical dump JSON shape is defined by `contracts/layer-dump.schema.json`,
but the *content* — what fields each spec kind carries, how listOps are
subdivided, which field names are valid on which spec kind — is governed by
the format-neutral document-model productions:

- `contracts/document-model-productions/prim-spec.contract.json`
- `contracts/document-model-productions/attribute-spec.contract.json`
- `contracts/document-model-productions/relationship-spec.contract.json`
- `contracts/document-model-productions/variant-spec.contract.json`
- `contracts/document-model-productions/layer-spec.contract.json`
- `contracts/document-model-productions/common-metadata.contract.json`

These contracts apply to any source format (USDA, USDC, or any future format
that lands in `usd-document-model`). They pin the field set per spec kind,
listOp subfield semantics, alias mapping, the forbidden-fields rule per spec
kind (e.g. relationship specs reject variability), and the format-neutral
diagnostic codes (`DuplicateSpecPath`, `ForbiddenFieldOnSpecKind`,
`TypeMismatch`, `InvalidFieldAlias`).

Per-format mapping contracts (`contracts/usda-productions/*` and
`contracts/usdc-productions/*-mapping.contract.json`) cite the corresponding
format-neutral contract via `derives_from` and add only the format-specific
encoding / grammar dispatch. The dump JSON emitted from a `.usdc` fixture and
the equivalent `.usda` fixture is required to be identical (modulo object
member ordering) — this is enforced by the `cross_check` step on the
`usdc-single-layer` scope in `graph/skillgraph.json`.

See `docs/document-model-productions.md` for the full rule layout and how to
add a new spec-kind production.
