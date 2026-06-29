# Composed Stage Open Contract

`usd-composed-stage-open` is the runtime orchestration boundary between layer
opening, composition sources, `usd-composition-arbitrator`, and stage population.

The key tightening is that generated runtime open paths such as `nanousd_open`
must not build reference sources with empty layer maps when authored external
assets are discoverable. When the optional payload inclusion scenario is
requested, they also must not build payload sources with empty layer maps for
discoverable payload assets. They should call this node, which opens the root
layer stack, opens local dependencies, recursively opens reference dependencies
discovered inside already-opened referenced layer stacks, optionally opens
reference dependencies discovered inside loaded payload layer stacks, builds the
composition arbitrator, and populates a stage from that composed namespace.
The adapter exposes a `payloadInclusion` scenario control for testing whether
direct external payload assets are opened and supplied as loaded payload
opinions. This control is not an authored USD field, and it represents optional
support for the AOUSD population-mask payload flag rather than a baseline
compliance requirement.

This keeps responsibilities separate:

- `usd-layer-stack-open` opens the root and recursive local sublayers it owns.
- Reference and payload sources consume already-opened dependency layers.
- `usd-composition-arbitrator` evaluates LIVERPS over supplied sources.
- `usd-stage-population` builds traversal and query indexes.
- `usd-composed-stage-open` wires those pieces together for runtime stage open.

The first version remains partial. Recursive/chained references are opened only
through the bounded local policy and supplied to the reference source. As
optional support, when `payloadInclusion=include`, direct external payload
assets are opened through the same layer-stack-open boundary and supplied to the
payload source; loaded payload layer stacks may also contribute nested reference
dependencies to the payload source. When `payloadInclusion=omit`, payload assets
are not opened and the composition arbitrator receives no payload opinion source,
so `payload_loading=absent` without missing-payload diagnostics.

Recursive payload arcs, dependencies authored inside selected variants, payload
load masks and mutation, full asset resolver policy, schema fallbacks, full
instancing/prototypes, and value clips are still deferred.
