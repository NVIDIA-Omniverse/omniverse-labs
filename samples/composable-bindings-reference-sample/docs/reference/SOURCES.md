# Add a second operational source

The [JSON source adapter](../../source/composable_bindings/reference/json_source.py) demonstrates a second integration without changing `Service` or `Edge`. It reads JSON Lines, maps field names, and posts the same source-neutral ingestion contract as the simulator. It does not import the simulator or evaluate alarm rules.

The supplied `line_gateway` records are synthetic fixtures, **not a Delta CL6 protocol implementation**. Actual CL6 field names, transport, units and alarm encodings must be verified against its payload contract. The example makes the adapter/configuration boundary concrete so that work can happen without service changes.

## Run alongside the simulator

Start `./run-services.sh` and a console or ovui client, then:

```sh
python3 reference.py configure --add examples/line_gateway
python3 reference.py adapt --mapping examples/line_gateway/adapter.json --input examples/line_gateway/readings.jsonl
```

A Gateway press 7 row appears. The fixture sends an alarm followed by a clear; the final display is NORMAL at 64.0 Cel. Its timestamps are fixed and are forwarded unchanged. In a fresh service session with this source configured, inspect the alarm alone by sending the first line before the second:

```sh
head -n 1 examples/line_gateway/readings.jsonl | python3 reference.py adapt --mapping examples/line_gateway/adapter.json
```

Use new occurrence IDs when changing fixture values; a repeated occurrence with a different payload is rejected. An identical retry is acknowledged with `accepted: false`. The source must issue occurrence IDs unique across its assets/event types. Adapters retry briefly with the same ID and refresh stale configuration revisions, but provide no durable outbox.

For an external-only demonstration:

```sh
python3 reference.py serve --config examples/line_gateway
# In another terminal:
python3 reference.py adapt --mapping examples/line_gateway/adapter.json --input examples/line_gateway/readings.jsonl
```

`serve` needs no `simulation.json`. Use a separate `--state` and `--port` if another service is already running. Pass the same `--state` to the adapter and readers.

## Contract and mapping

A source record in this example is:

```json
{"equipment":"press-7","occurrence":"gateway-reading-1",
 "observed_at":"2026-10-01T12:00:00Z",
 "readings":{"temperature_c":64.5},"conditions":{"overheat":true}}
```

`adapter.json` names the actual issuing system and declared event type, then maps `id_field`, `time_field`, `event_id_field`, `values` and `alarms` to dotted JSON object paths. The ID is taken verbatim from `equipment`; the catalog and USD must use the same value. Missing mapped values/alarms become null (unknown); identity and timestamp fields are required. Values must be finite numbers, alarm decisions booleans/null, and observation times timezone-qualified. Unit conversion and protocol decoding belong in the adapter, before posting declared units; the mapping supports no expressions or implicit numeric coercion.

The emitted request contains `source`, `source_id`, `event_id`, `event_type`, `time`, `values`, `alarms` and the current `config_revision`. `events.json` declares allowed assets and metrics independently of any generation method. `alarm: source` tells the service to preserve the source's decision; this metric has no entry in `rules.json`. A consumer receives the standard sample event plus catalog/display configuration and needs no source-specific transformation.

## Swap in a real feed

Use the source system's own name and inventory IDs in `events.json` and `bindings.json`; update metric names/units, alarm ownership and view configuration. Map the real JSON fields in `adapter.json`. A gateway emitting that shape can pipe one record per line into `reference.py adapt`. If CL6 uses another transport, type encoding or schema, implement that translation at the adapter boundary; do not add vendor branches to `Edge`.

For Kit, author `assetInfo.sourceIds.<issuing-system>.id` on the correct prim using the same source-issued ID. The default scene has only the simulator's IDs, so a new source appears in console/ovui before it has any Kit geometry binding. Adding a source does not guess a matching prim. Follow [USD authoring](USD_BINDINGS.md) to create that relationship.

Source history would be queried from the source using this ID and a time range. Feeding saved JSON fixtures or replaying local stream captures does not implement that history lookup.
