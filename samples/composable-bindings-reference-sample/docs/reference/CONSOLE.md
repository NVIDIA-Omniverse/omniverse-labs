# Console client

The shell is the smallest complete consumer: a normal Python process that logs measured values, units and alarm results. It uses the same `Client` and `ViewModel` as Kit and ovui; it needs no USD, GPU or UI library.

## Start it

Start `./run-services.sh` first, then in another terminal:

```sh
python3 reference.py console
```

On Windows, start `python reference.py demo --no-console` and then `python reference.py console` in another PowerShell terminal. No WSL or extra Python packages are needed.

For a separate state directory, pass the same `--state /path/to/state` used by the service. For a quick combined producer/service/console demo, run `./run-default.sh`.

## Source layout

The terminal application is separate from the shared library:

- `source/clients/console_client/app.py`: terminal output loop, standalone argument parsing and shutdown.
- `source/clients/console_client/__main__.py`: module entry point.
- `source/composable_bindings/reference/client.py`: shared event subscription and log queue, also used by the UI clients.

The existing `python3 reference.py console` command forwards to this application. The combined `demo` command reuses the same loop; there is no duplicate terminal implementation.

To launch the application directly from the checkout on Linux:

```sh
PYTHONPATH=source python3 -m clients.console_client
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH = "$PWD/source"
python -m clients.console_client
```

Both entry points accept `--state`, `--view` and `--duration`. The direct module also accepts `--client-config` for an explicit reader credential file. It requires only the Python standard library.

## Read the output

Each line contains observation time, source/ID pair, display title, value/unit, optional text gauge, NORMAL/ALARM/UNKNOWN, and event occurrence ID. The hashes in a text gauge indicate position within the display range; the printed value remains the physical measurement.

[Captured console output](../../artifacts/reference/console-output.txt) comes from an actual isolated Python run. Event IDs differ each run. Compare matching IDs across clients to confirm they consumed the same occurrences.

```text
Temperature: 72.1 Cel [#########---] NORMAL
Pressure: 198.0 kPa [##########--] ALARM
```

This shortened example omits time and identity for readability. The linked capture contains complete lines. The configured Celsius unit is `Cel`; voltage uses `V` and throughput uses `parts/min`.

## Behavior and implementation

The `run()` loop in [source/clients/console_client/app.py](../../source/clients/console_client/app.py) drains the shared client’s log buffer every 100 ms and prints status transitions. File changes and new configured displays become visible without restarting the console. Ctrl-C stops this reader; an independently running service and Kit remain active.

The console is intentionally **live-only**. Recording in Kit does not pause it, and Kit scrubbing never changes its output. The shared replay API is available for another consumer, but this command does not expose it.

Both subscriber and console buffers are bounded. Slow consumers receive a gap notice rather than a false claim of complete history. Shell output is a demonstration aid, not a durable alarm audit or a production alerting service.

## Use it in production code

Replace `print()` with your application’s logging or alert sink after obtaining authorization and deduplication rules. Preserve occurrence IDs, quality and observation times. An alarm metric is evaluated for each reading; repeated ALARM lines are not automatically distinct incidents. Add transition grouping, acknowledgement and durable delivery in the operational application if those are required.
