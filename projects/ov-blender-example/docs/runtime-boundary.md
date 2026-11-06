---
keywords:
  - blender
  - ovrtx
  - ovphysx
  - runtime
---

# Runtime boundary

The public product is a Blender Python extension and a set of prebuilt binaries
for supported platforms. Blender owns the open scene, authoring values,
callbacks, and viewport presentation. The detailed add-on ownership and
lifecycle model is in
[Runtime and Authoring Architecture](architecture.md).

OVRTX and OVPhysX run outside the Blender process. The add-on starts or
connects to their local bridge services through bundled native clients, sends
typed operations over loopback gRPC, and presents OVRTX pixels in Blender.
OVPhysX supplies supported physics poses for OVRTX to render. The public
product does not embed the renderer or physics runtime in the Blender process.

`addon/blender_manifest.toml` configures the extension identity, supported host
platforms, required permissions, and package exclusions. The installed add-on
materializes a separate runtime from its runtime manifest. The public archive
therefore contains the Blender integration and runtime selection logic, while
the platform runtime components are installed beside it.

The public host contract, support matrix, licensing boundary, and test entry
points are owned by the [README](../README.md),
[support platforms](support-platforms.md), [licensing](licensing.md), and
[contribution guide](../CONTRIBUTING.md). This note does not name mutable
artifact versions or source revisions.

## Service endpoints

The bridge services listen on loopback. The OVPhysX endpoint defaults to
`127.0.0.1:50300`; OVRTX defaults to port `50051`.

`50300` is a deliberate choice, not an arbitrary one. Windows hands whole port
blocks to WinNAT/Hyper-V, and a bind inside one of those blocks fails with
`WinError 10013` — the same error code an ordinary permission failure produces,
so the symptom is a worker that exits before reporting `SERVING` with nothing
pointing at the port. On the reference Windows host the reserved blocks include
`50060-50159`, which covers the previous default. List them with:

```
netsh interface ipv4 show excludedportrange protocol=tcp
```

The reservations differ per machine and can change across reboots, so if the
default is reserved on your host, override it — and override **both** of these
together:

| Variable | Purpose |
| --- | --- |
| `OV_BLENDER_EXAMPLE_OVPHYSX_ADDRESS` | The endpoint the add-on connects to. |
| `OV_BLENDER_EXAMPLE_OVPHYSX_WORKER_COMMAND` | The command that launches the worker. **It bakes the port.** |

Setting only the address leaves the worker listening on the old port, and the
add-on then fails to connect to a service that is running perfectly well
somewhere else. When a bind is refused, the add-on's startup error names the
reserving range rather than leaving you to infer it.

The endpoint and its port are recorded in the runtime service diagnostics, so a
measurement taken on one host stays interpretable on another.
