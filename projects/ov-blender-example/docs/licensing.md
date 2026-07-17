# Licensing and Distribution

Project-authored source and documentation in this repository are licensed under
the Apache License, Version 2.0 (`Apache-2.0`). The full license text is in
[`LICENSE`](../LICENSE). Python and Bash source files carry the following SPDX
tags near the top of each file:

```text
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

The Blender add-on runs inside Blender and uses Blender's Python API.

OVRTX, OVPhysX, their service processes, and their Python/native clients are
external runtime components. The add-on communicates with the service
processes over loopback gRPC. Those
components retain their own license terms; this repository does not relicense
them.

Committed fixture assets and bundled third-party source retain their own
licenses. Their provenance and required notices are recorded under
[`tests/fixtures`](../tests/fixtures/); they are not relicensed by the
repository's Apache-2.0 license.

Direct add-on dependency notices and runtime-evidence gaps are recorded in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Downloaded fixture assets and generated test artifacts are excluded from the
add-on archive. Their source and license records live in discovered
`tests/fixtures/<fixture-id>/spec.json` files. A fixture record documents
provenance and does not itself grant redistribution rights.

Repository licensing and process isolation do not represent OSRB or legal
approval.
