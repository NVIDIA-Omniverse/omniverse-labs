# Schema Namespace Source Contract

`contracts/handles/schema-namespace-source.handle.json` defines a schema-aware
namespace-source wrapper. It consumes a composed namespace source plus an
explicit schema registry and exposes schema-defined properties before stage
population.

AOUSD v1.0.1 is the only normative source. OpenUSD registry contents and
runtime behavior are not used to fill gaps. The schema specification leaves the
mechanism for defining and ingesting schemas implementation-defined, so this
contract treats registry data as an explicit conformance input.

## Covered Rules

- typed schema selection from composed `typeName`
- `fallbackPrimTypes` substitution for unregistered authored types
- typed schema base inheritance
- single and multiple applied schema properties from composed `apiSchemas`
- built-in and auto-apply schema inclusions supplied by the explicit registry
- multiple applied schema instance propagation through schema inclusions
- override properties as sparse stronger opinions over included property fields
- diagnostics for invalid schema inclusions and invalid override properties
- schema-defined property fields and fallback defaults
- authored property fields overriding weaker schema fields
- property child enumeration over authored plus schema-defined properties

Schema fallback defaults remain schema-defined fallback data for value
resolution. They are not authored layer opinions.

## Deferred

Schema family version selection and domain behavior for concrete schema families
are deferred to later contracts with independent goldens. Schema discovery from
files, plugins, or process-global registries remains outside this boundary
because AOUSD v1.0.1 leaves ingestion implementation-defined.
