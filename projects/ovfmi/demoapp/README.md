<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ovfmi demo application

This sample combines the public `ovfmi` API with ovstage and ovrtx rendering
and optional ovphysx rigid-body simulation. It is deliberately separate from
the shipping library in the repository's `python/` directory.

Run `setup.ps1` on Windows or `setup.sh` on Linux from the repository root.
The scripts install the root ovfmi project and this demo in editable mode,
create the isolated USD parser environment, and build the models under `fmu/`
and `ssp/` into `usd/`.

For complete prerequisites and commands, see the repository root README.
