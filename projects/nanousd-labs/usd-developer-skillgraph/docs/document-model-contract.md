# Document Model Contract

`contracts/capabilities/document-data-model.json` captures the pinned spec's
bounded layer document model for this prototype: layer specs, prim specs,
attribute specs, relationship specs, field maps, child-list maintenance, and
duplicate path rejection.

`contracts/handles/document-model.handle.json` defines the standalone unit
surface used by generated targets. Each test case starts with a fresh layer and
drives it through JSON operations.

## Dependencies

The document model is intentionally a consumer layer:

- spec and field names come from `usd-identifiers-and-names`
- field names and token-valued/name-list metadata consume `usd-tokens`
- spec paths and parent/name construction come from `usd-paths`
- field values are canonical tagged values from `usd-foundational-values`
- list operation field values come from `usd-listops-authored`

The foundational data types spec defines the general list operation algebra and
eligible foundational element types. In this graph, the full authored listOp
structure is owned by `usd-listops-authored`; document-model fields only store
those values.

Field names are a separate identity surface from parser strings. The document
model contract now requires generated targets to predeclare the known core,
reserved, and deprecated metadata field names from the document data model as
tokens or equivalent field atoms. Extension field names can still be accepted,
but they are interned into the same field-name identity domain before durable
storage. Tagged `token` values and `token[]` name-list fields are likewise
token-backed internally, even though the adapter and canonical dump expose JSON
strings.

The canonical JSON dump is not the durable storage model. Generated targets
should have a native field value representation capable of storing foundational
and specialized document-model values, then convert to tagged JSON only for the
adapter/dump contract. In the C++ target this means `Layer` accepts token values
for token-typed concepts such as prim `typeName`, and `Spec` stores field values
in a typed field-value container rather than a `Json` map.

The backing container is not prescribed. A target may use `std::variant`, a
tagged union, enum-plus-payload storage, arena-owned values, or a backend-native
field value object. The observable requirement is that callers can inspect field
existence and stored type identity, then perform checked typed reads without
round-tripping through canonical JSON.

Numeric field payloads are not allowed to use decimal strings as authoritative
storage. A dump adapter may render JSON number text, but typed field reads and
composition helpers should consume native integer or floating payloads.

Field values also inherit the complete foundational typed payload catalog. A
document-model backend can use a compact shared container, including a
backend-native value type, but callers must still be able to inspect and read
`half`, `timecode`, vectors, quaternions, matrices, semantic aliases,
homogeneous arrays, dictionaries, and value blocks as distinct typed payloads.
Those categories must not degrade to untyped arrays or JSON object trees before
stage population or future nanousd-style backends consume them.

The generic fallback for not-yet-specialized field values should also avoid
retaining adapter JSON. It can be a native payload tree or another compact value
representation, with tagged JSON materialized only on demand for dumps.

The specialized field values currently in scope should likewise be native
domain values: `specifier`, `variability`, `Retiming[]`, `Relocates`,
`TimeSamples`, and `Spline`. Their dump representation remains tagged JSON, but
the layer model should not use strings, JSON objects, or dictionary/list payloads
as their durable representation.

Likewise, spec identity is path identity. Generated targets should store specs
by path handles from `usd-paths` and expose path handles on domain-facing
document-model APIs. String paths are parser/adapter inputs and dump/diagnostic
outputs. Parsing, stringification, and parent/name path construction remain
owned by `usd-paths`; `Layer` must not grow its own path text API.

## Scope

The current unit scope validates:

- the layer spec exists at `/`
- root and nested prim creation updates `primChildren`
- property creation updates `properties`
- prim, attribute, relationship, layer, and custom fields round-trip as tagged
  values
- known document-model field names are available through a token vocabulary
- child/name-list and token-valued fields are token-backed internally
- field values expose typed existence/type inspection and checked typed reads
- numeric field values use native numeric payloads internally
- foundational field payloads preserve catalog metadata for scalars, vectors,
  quaternions, matrices, semantic aliases, arrays, dictionaries, and value
  blocks
- duplicate spec paths are rejected
- invalid or missing parents are rejected without partially mutating the layer
- `active=false` is preserved and does not remove specs at the layer level

Variant set/spec storage and complete core metadata field validation remain
deferred.

```powershell
py harness/regen_graph.py --scope usd-document-model-unit --target python --validate
py harness/regen_graph.py --scope usd-document-model-unit --target cpp --validate
```
