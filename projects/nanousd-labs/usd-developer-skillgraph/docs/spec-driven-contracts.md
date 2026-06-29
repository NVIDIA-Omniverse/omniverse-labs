# Spec-Driven Contracts

The pinned USD specification is the source of truth for behavior, grammar, and
document-model semantics. Skills, handles, contracts, and goldens are not a
replacement for the specification; they are an executable decomposition of it
plus the engineering constraints needed to generate a usable library.

Each skill should separate four concerns:

- Spec obligations: the exact pinned sections and productions owned by the skill.
- Mapping obligations: how valid spec productions become document-model data.
- Quality floor: representation and API constraints that avoid toy implementations.
- Performance model: workload and operation targets where the spec is abstract.

For parser skills, grammar recognition comes before storage. A generated parser
must first match a production valid in the current context, then map the result
to document-model fields. Document-model field names cannot make invalid syntax
valid. For example, `payload` is a USDA keyword and cannot be parsed as generic
attribute metadata merely because the document model has a `payload` field.

The coverage ledger for the USDA single-layer scope is
`contracts/spec-coverage/usda-single-layer.coverage.json`. It records production
ownership, implementation status, positive/negative goldens, storage mapping,
and quality constraints for the current scope.
