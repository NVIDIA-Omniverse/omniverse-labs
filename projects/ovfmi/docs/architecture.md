<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovfmi

`ovfmi` is the repository's primary FMI/SSP simulation product. It owns model
discovery, fmpy lifecycle management, and
the routing between USD attribute identities and FMI variables. Rendering,
windowing, camera control, and ovphysx stepping remain application concerns.

The implementation is a Python package with fmpy behind a private
`MasterBackend` protocol. Applications use only the public types exported from
`ovfmi`; the app-local FMI modules provide compatibility imports for tests and
downstream scripts.

## Lifecycle

```python
from ovfmi import AttributeWrite, FmiHost

with FmiHost() as fmi:
    report = fmi.attach_ovstage(stage, source_asset=usd_path)
    fmi.update_from_ovstage(input_ordinal, output_ordinal)
    fmi.write([AttributeWrite(prim_paths, "physx:position", positions)])
    fmi.step_sync(dt)

    with fmi.read(attribute_names=["physx:force"]) as outputs:
        apply_forces(outputs.groups)

    fmi.write_to_ovstage(output_ordinal)
```

The public boundary names data in USD space: prim paths and USD attribute
names. FMI variable names, component mapping, FMU extraction, and SSP
coordination remain private.

The ovstage population used by the demo does not expose custom FMI schema
columns. `source_asset` supplies the USD source, which ovfmi parses in an
isolated USD process to discover those columns.

## Backend

The package is intentionally layered as follows:

```text
sample app -> public FmiHost API -> routing/lifecycle -> MasterBackend -> fmpy
```

FMPy implements the private backend protocol. Application-facing data flow is
expressed through the public `FmiHost` API.

## Development

From the repository root, after setting up the sample environment:

```powershell
.\demoapp\.venv\Scripts\python.exe -m pytest tests
```
