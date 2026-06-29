# First Python Target

This note records the first generated implementation for
`graph/scopes/usda-single-layer.yaml`.

## Generated Artifact

- Target entrypoint: `generated/python/dump_layer.py`
- Generated node modules:
  - `generated/python/usd_identifiers.py`
  - `generated/python/usd_values.py`
  - `generated/python/usd_paths.py`
  - `generated/python/usd_listops.py`
  - `generated/python/usd_document_model.py`
  - `generated/python/usda_lex.py`
  - `generated/python/usda_values.py`
  - `generated/python/usda_parser.py`
  - `generated/python/usda_layer_open.py`
- Status: ignored by git as disposable generated output
- Interface: accepts a USDA filepath as the final argument and writes canonical
  layer dump JSON to stdout

The target is intentionally narrow: one USDA layer, no composition operators or
asset resolution, and no stage population.

The first version of this target was monolithic. It was split into graph-node
modules after that proved too weak to exercise dependency-driven regeneration.

## Correctness

Command:

```powershell
py harness/score.py --goldens goldens/integration/usda-single-layer/basic.json --dump-cmd "{python} generated/python/dump_layer.py"
```

Result:

```text
OVERALL: 6/6 (100.0%)
```

The generated target passes the initial goldens for:

- empty layer
- root prim declaration
- nested prims and attributes
- relationship path targets
- layer metadata and authored child ordering
- duplicate spec path diagnostics

## Performance Finding

The first benchmark run exposed a measurement-model issue. The original
thresholds were parser-shaped, but the harness measured full CLI process
invocation.

Observed CLI timings before retuning the smoke thresholds:

```text
tiny_layer_parse:       107.329 ms
large_flat_prim_parse:  236.472 ms
many_attributes_parse:  310.973 ms
```

Observed in-process timings for the same generated parser, including JSON
serialization but excluding Python startup:

```text
tiny.usda:             0.045 ms
large_flat_prim.usda:  115.429 ms
many_attributes.usda:  191.517 ms
```

This supports separating benchmark contracts into at least two classes:

- CLI smoke checks, which include process startup and are useful for regression
  confidence
- library or persistent-driver checks, which measure parser and storage
  performance without startup noise

## Next Evolution

The next useful step is to add a language-neutral benchmark contract that names
the workflow, fixture shape, measurement model, and acceptable performance band.
For generated libraries, that probably means a tiny adapter per target language
instead of only a process-per-fixture dump command.
