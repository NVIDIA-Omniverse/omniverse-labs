# AOUSD Core v1.0.1 Compliance Coverage

This document maps the backend-facing compliance suite to AOUSD Core v1.0.1
behavioral areas. The executable is intentionally pure C and only uses
`nanousdapi.h`, so alternate backends are measured through the same public API
surface as applications.

## Test Tiers

- Release: `compliance_test` is deterministic and fixture-backed. It is intended
  to run from a Release build for backend compliance decisions.
- Extended corpus: the current generated and adversarial cases are cheap enough
  to stay in the release suite. Larger randomized or corpus-mutation tests
  should use the same normalized JSON oracle and be added behind a separate CTest
  label or command-line tier before they are made mandatory.

## Normalized Fixture Oracle

Fixture comparisons flatten the source fixture through the public API, reopen
that flattened layer, and normalize the observable stage to JSON. The checked
JSON includes stage default prim metadata, prim path/type/defined/abstract/
instanceable state, child order, authored attributes, authored relationships,
defaults, connections, relationship targets, and scalar time samples.

The expected outputs are checked in as static JSON manifests under
`tests/compliance/usda/fixture_compare/*_expected.json`. These manifests do not
pass through the backend writer, so they catch squashed-layer and sampled-layer
behavioral drift independently of USDA serialization details.

## Coverage Matrix

| AOUSD Core area | Primary tests | Fixture/oracle coverage |
| --- | --- | --- |
| Stage lifecycle and resources | `test_stage_open_*`, `test_file_uri_resource_open`, diagnostics tests | Missing, invalid, masked, file URI, and package resource opens |
| Layer and stage metadata | `test_stage_metadata*`, `test_stage_default_prim`, `test_write_stage_metadata*`, kilograms-per-unit tests | Normalized fixture JSON records default prim metadata after flattening |
| Namespace and paths | `test_path_parse`, `test_path_operations`, `test_unicode_identifiers`, `test_usda_grammar_strictness`, ordering tests | Path element ordering and Unicode identifiers are exercised through parser/API paths |
| Prim population and traversal | traversal, hierarchy, child ordering, specifier, active/inactive, population mask tests | `mask_root.usda` checks mask ancestors, included prim, excluded siblings, descendants, and unrelated roots |
| Attributes and foundational values | scalar/vector/matrix/quaternion/string/array/bulk access tests, foundational type-name tests | Generated round-trip matrix covers scalar defaults, sampled float values, and bool/string/int defaults through USDA and USDC |
| Time-varying values | time sample tests, array time sample tests, value-resolution tests | `sampled_root.usda` compares resolved defaults, time samples, spline baking, and value clips against normalized JSON |
| Relationships and connections | relationship tests, creation tests, listOp target metadata tests | `remap_root.usda` validates connection and relationship target remapping across references and payloads |
| Collections | collection membership and containment tests | API-level checks cover lazy membership evaluation through public collection queries |
| Composition arcs and strength | sublayer/reference/payload/inherits/specializes/variant/relocates/LIVRPS tests | `squash_root.usda` validates squashed composition across sublayer, reference, payload, local opinions, children, and relationships |
| Schemas | schema query/registration/auto-apply tests, API schema listOp tests | Public API checks cover fallback prim types, auto-applied schemas, and abstract composed type handling |
| File formats | USDA read/write, USDC read/write, USDZ read/write, binary magic, file URI write tests | Fixture USDC metamorphic tests round-trip every flattened fixture and compare normalized JSON |
| Invalid input handling | null safety, diagnostics, generated USDA rejection cases | Adversarial USDA fixtures reject unterminated strings, invalid unsigned values, incomplete numbers, bad list syntax, and invalid relationship target syntax |
| Asset path resolution | asset resolver tests and write-fidelity tests | Cross-layer assets, arrays, time samples, metadata, URI passthrough, and self-contained flattening are covered |

## Remaining Blind Spots

- The normalized JSON oracle intentionally covers the compliance-relevant scalar
  surface used by the fixtures. Other value families are still covered by API
  tests, but not yet by independent fixture manifests.
- USDA and USDC malformed-input testing is representative, not exhaustive
  fuzzing.
- USDC metamorphic tests compare backend write/read behavior through normalized
  JSON. They catch observable corruption, but they are not a byte-level crate
  conformance proof.
- The suite exercises deterministic fixtures only. A future extended corpus
  should add generated composition graphs and grammar mutations while reusing
  this normalized JSON oracle.
