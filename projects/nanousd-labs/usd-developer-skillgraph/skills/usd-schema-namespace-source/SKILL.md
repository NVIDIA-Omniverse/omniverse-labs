---
name: usd-schema-namespace-source
description: Use this skill when implementing or verifying schema-derived prim definitions, schema-defined properties, and schema fallback values over an already composed namespace source.
metadata:
  author: NVIDIA
---

# usd-schema-namespace-source

Use this skill when implementing or verifying schema-derived prim definitions,
schema-defined properties, and schema fallback values over an already composed
namespace source.

## Spec Sources

- `aousd/specifications-public@v1.0.1:specification/schemas/README.md`
- `aousd/specifications-public@v1.0.1:specification/stage_population/README.md`
  section `Ordered Property Children`
- `aousd/specifications-public@v1.0.1:specification/value_resolution/README.md`
  section `Fallback Values`

Pinned tag / commit: `v1.0.1`

## Provides

- A schema-aware `namespace_source` wrapper
- Explicit schema-registry input for implementation-defined schema definitions
- Typed schema property injection
- Typed schema base inheritance
- Single and multiple applied schema property injection
- `fallbackPrimTypes` typed schema substitution
- Built-in and auto-apply schema inclusion expansion from explicit registry data
- Multiple applied schema instance propagation through schema inclusions
- Override properties that sparsely strengthen included property fields
- Diagnostics for invalid schema inclusions and invalid override properties
- Schema-defined fallback fields for downstream value resolution

## Contract

This skill owns `contracts/handles/schema-namespace-source.handle.json` and
`contracts/spec-coverage/schema-fallbacks.coverage.json`. It also follows
`contracts/capabilities/schema-registry.json`,
`contracts/capabilities/namespace-source.json`, and
`contracts/capabilities/semantic-runtime-types.json`.

The source wraps a composed namespace source. It does not compose sublayers,
references, payloads, inherits, specializes, variants, or relocates itself.
Those arcs must already be represented by the consumed namespace source.

AOUSD v1.0.1 explicitly leaves schema definition ingestion up to the
implementation. Treat schema definitions as an explicit registry input at this
contract boundary. Do not consult OpenUSD registry behavior, fallback lists, or
plugin loading.

## Composition Rules

For each composed prim, read the composed `typeName` field from the wrapped
source. If it names a registered concrete typed schema, use that schema. If it
does not, consult composed `fallbackPrimTypes` and use the first registered
concrete typed schema in the preference list. If neither path succeeds, the
prim is typeless for schema purposes.

Read composed `apiSchemas` field data as the ordered applied schema list.
Single applied schemas contribute their properties directly. Multiple applied
schemas are recorded as `SchemaName:instanceName`; substitute the recorded
instance name into placeholder property names from the schema definition.

Build the prim definition in strong-to-weak schema order: typed schema
properties first, including typed base schemas as weaker definitions, then
applied schema properties in composed applied-schema order. Schema-defined
properties are weaker than authored composed property specs on the same prim.

When a registered schema lists built-ins or auto-applies, compose those schemas
as weaker inclusions in that schema's prim definition. Built-ins must name
applied schemas. Multiple applied schema inclusions substitute the recorded
instance name; when one multiple applied schema includes another by type, reuse
the owning instance name, and when it includes a named instance, prefix the
owning instance name to the included instance name.

Override properties do not define properties. Apply them only to properties
already supplied by included schemas, ignore override `typeName` and
`variability` fields, and emit diagnostics when the override target is missing
or its kind/typeName conflicts with the included property.

Property names exposed by this source are the union of authored/composed
property names and schema-defined prim-definition properties, sorted by
`usd-paths` path element ordering, then reordered by the strongest
`propertyOrder` field.

Schema fallback defaults remain schema-defined fallback data. They may appear
in schema property field maps so stage population can populate the property,
but they must not be reclassified as authored layer defaults.

## Boundary Guards

Do not perform schema definition discovery from files, plugins, process-global
registries, or OpenUSD.

Do not evaluate composition arcs, asset resolution, payload load policy,
stage population, interpolation, value clips, collections, color inheritance,
xform semantics, material binding, geometry behavior, or instancing.

## Test Obligations

- typed schema properties are injected into prim property names
- typed base schema properties are included as weaker schema properties
- schema fallback default fields are visible as schema-defined data
- authored property fields override weaker schema-defined fields
- single and multiple applied schemas inject properties
- multiple applied schemas substitute recorded instance names into property
  names
- fallbackPrimTypes substitutes the first registered concrete fallback type
- built-ins and auto-applies expand the prim definition
- multiple applied schema inclusions propagate or prefix instance names
- override properties sparsely strengthen included property fields
- invalid override properties emit schema-source diagnostics
- stage population preserves schema-injected properties from this source
