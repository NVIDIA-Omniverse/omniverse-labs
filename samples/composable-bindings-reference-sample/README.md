---
name: "Composable Bindings Reference Sample"

description: >
  Reference implementation of the live half of the Composable Bindings white paper:
  simulated machines publish measurements, an edge service preserves source identities
  and adds threshold alarms, and console, Kit and ovui clients consume the same results
  through one shared Python client.

version: "0.3.0"

type: example

complexity: intermediate

omniverse_libraries:
  - ovui  # optional standalone 2D client; the Kit client needs an installed Kit SDK

omniverse_library_version_min: "0.2.0"

tested_platforms:
  - Windows
  - Linux

tags:
  - usd
  - composable-bindings
  - source-identity
  - live-data
  - alarms
  - kit
  - recording

assets_included: true

skills_included: false

author: "Min Jiang"
maintainer: "minj@nvidia.com"
---

# Composable Bindings: two machines, one shared client

**Status:** Working

This reference sample implements the **live half** of the [Composable Bindings: Simplified System Integration white paper](https://aka.ms/ComposableBindings). It separates operational data production, event enrichment and visualization. Scene namespace and source identity are separate concerns described in [OpenUSD-proposals#105](https://github.com/PixarAnimationStudios/OpenUSD-proposals/pull/105): prims carry the issuing system's own ID under `assetInfo.sourceIds.<source>.id`.

Two simulated machines and their production line publish measurements; the edge preserves source identities and adds threshold alarms. A [second-source JSON adapter](docs/reference/SOURCES.md) demonstrates external readings and source-owned alarms without changing enrichment. Console, Kit and an optional 2D ovui dashboard consume the same results through a shared Python client.

Source history would be fetched from the issuing system through its ID and a time range. This sample does not implement that lookup. Kit's **Capture replay** only replays this sample's local capture of the live stream.

Kit also [colors the cylinders with a continuous pressure gradient](docs/reference/SCENE_COLORS.md), composing a temporary layer in memory while leaving the saved USD asset unchanged. Repeated readings with the same color do not cause USD writes.

**Service and console: Python 3.9+ · Linux or Windows · no pip install.** Kit visualization additionally needs a compatible GPU and an installed Kit SDK. The optional standalone ovui client uses Python 3.12 and its own UI dependencies; its guide covers Linux and Windows setup. All machine data and geometry are synthetic.

![Kit showing live cylinder colors, machine gauges and line throughput](artifacts/reference/kit-gradient.png)

## Part I — Architecture and contracts

Read the [architecture](docs/reference/ARCHITECTURE.md) for event production, enrichment, delivery, source identity, class diagrams and production boundaries. The [shared-client guide](docs/reference/CLIENT.md) shows the Python and HTTP APIs. [Configuration](docs/reference/CONFIGURATION.md) explains how event declarations, simulation, rules, views and source-ID labels remain separate. The [bus replacement contract](docs/reference/BUS.md) explains the snapshots and retained state required when scaling beyond the in-memory bus.

## Part II — Replaceable clients

The [console client](docs/reference/CONSOLE.md) logs live values and alarms. The [Kit extension](docs/reference/KIT.md) loads the prebuilt USD and adds selection, reusable gauges and a recording timeline. The [ovui client](docs/reference/OVUI.md) presents the same line and machine gauges in 2D, without Kit. Console and ovui are live-only; Kit exposes the shared recording/playback API. All three presentation implementations live under `source/clients/`, separate from `source/composable_bindings/`.

![Standalone ovui dashboard showing the same live metrics](artifacts/reference/ovui-live.png)

## Part III — Run the sample

Run these in separate terminals, from this project folder:

```sh
# Terminal 1: service and simulated producer
./run-services.sh
# Terminal 2: independent console reader
python3 reference.py console
# Terminal 3: Linux GPU with Kit already installed
./run-kit.sh
```

On Windows, use `python reference.py demo --no-console`, `python reference.py console`, and `.\run-kit.bat` in separate PowerShell terminals. The Kit launcher defaults to a sibling `kit` directory; override `KIT_ROOT` for another SDK location.

For the optional fourth terminal, follow the [ovui setup guide](docs/reference/OVUI.md) and run `./run-ovui.sh` with its Python environment. It subscribes to the already running service.

The [complete tutorial](docs/reference/TUTORIAL.md) covers setup, selection, adding voltage, changing values and thresholds, recording, playback and shutdown. For a one-terminal Python introduction, run `./run-default.sh`.

## Where to look

| Location | Responsibility |
| --- | --- |
| `source/clients/console_client/` | Standalone terminal client: prints live values and alarms |
| `source/clients/ovui_client/` | Optional standalone 2D ovui dashboard |
| `source/composable_bindings/reference/` | Shared Python: producer, edge, transport, configuration, client and recording/replay |
| `source/clients/extensions/utils/omni.example.reference_client/` | Kit adapter, scene loading, gauges, selection and timeline controls |
| `assets/production-line.usda` | Prebuilt scene with source-issued IDs in asset info |
| `apps/reference.kit` | Minimal Kit application and dependencies |
| `examples/default/`, `examples/voltage/` | Event declarations, separate simulation and reusable voltage addition |
| `examples/line_gateway/` | Second-source contract, field mapping and synthetic JSON Lines input |
| `config/bindings.json` | Source-issued IDs and display labels; no aliases |
| `docs/reference/` | Architecture, component guides, tutorial and diagrams |
| `artifacts/reference/` | Actual screenshots, emitted example event, console output |


## Scope

This is a small reference implementation, not a certified protocol stack or a production broker. The [architecture](docs/reference/ARCHITECTURE.md) describes its limits.

## Contributions

This project is currently not accepting contributions.

## License and notices

NVIDIA-authored sample code is licensed under the [Apache License 2.0](LICENSE).
See [third-party notices](THIRD_PARTY_NOTICES.md) for dependency copyrights and
license texts, including the separately licensed optional runtimes.
Maintainers must follow the [ongoing IP review procedure](docs/reference/IP_REVIEW.md)
before public distribution.
