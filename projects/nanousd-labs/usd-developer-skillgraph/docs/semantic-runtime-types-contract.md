# Semantic Runtime Types Contract

This contract captures a cross-cutting generation rule: strings are valid at
text, JSON, diagnostic, and C ABI boundaries, but they should not become the
authoritative runtime identity for USD paths, field keys, tokens, or names used
by composition and stage queries.

The contract lives in
`contracts/capabilities/semantic-runtime-types.json` and is referenced by the
document-model, namespace-source, stage-population, parser, and nanousd backend
contracts.

## Core Rule

- Values that are semantically USD paths should be represented internally as
  path handles/references or equivalent target-native path objects.
- Field keys should be field tokens or field atoms.
- USD token values should preserve token identity.
- Names used for identity, lookup, ordering, or path construction should be
  token/name handles or equivalent atoms.
- String spellings may be accepted or emitted by adapters, parsers,
  diagnostics, canonical dumps, reports, or C ABI functions.

## Type Preservation

Interning a name does not change the authored field type. For example,
`variantSetNames` is still a `listop<string>` field and `variantSelection`
remains a specialized map. A generator can intern those names for composition
records, but checked field reads and canonical dumps still report the authored
types.

Variant names are a useful example of why this is not simply "all names are
identifiers." AOUSD v1.0.1 defines a variant-name grammar that allows names
outside the prim/property identifier grammar. A target can still use an interned
name handle for selected variant names, but that handle must support the variant
name domain.

## Generator Implication

Composition records should look more like this:

```cpp
struct VariantRecord {
    PathRef prim_path;
    Token variant_set;
    Token selection;
    PathRef variant_path;
};
```

and less like this:

```cpp
struct VariantRecord {
    std::string prim_path;
    std::string variant_set;
    std::string selection;
    std::string variant_path;
};
```

Keeping cached text alongside semantic handles can be fine for diagnostics or
adapter output, but the semantic handle must remain the value used for
comparison, lookup, mapping, and composition.
