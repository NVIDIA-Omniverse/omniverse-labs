---
name: ovdnext-develop
description: Use when navigating OVD Viewer v0.2.21 source, preparing a source build, planning or implementing an OVD viewer or native inspection change, or choosing validation for the Labs snapshot.
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Develop OVD Viewer

Read [the development guide](references/development.md) for the architecture,
pinned inputs, prepared-environment build commands and validation matrix.
For a saved-capture investigation using a package, use `ovdnext-inspect`.

First identify the checkout, application commit and intended change. Read
its applicable `AGENTS.md` and existing implementation. In Labs,
`projects/ovdnext/` is an immutable upstream snapshot: application changes
belong in the development repository. Labs-owned documentation belongs
alongside this companion. Do not edit the snapshot, suppress its verifier
or regenerate its manifest to bless local changes.

Locate the responsible layer before proposing a fix: ingestion owns capture
preparation and lifetime; Session/Presentation own coherent native state and
bounded inspection; bindings expose that contract; adapters/rendering and
OvUI provide the frontend. Preserve availability and capture/scene/frame
identity. A missing projection is not proof of absent recorded data.

Verify exact dependencies and access before building. Use prepared inputs
and an external build directory. Report missing private dependencies rather
than substituting newer SDKs or claiming a clean public build. Respect
third-party licensing independently of the application's Apache license.

Select focused checks from the guide, then report actual passed, failed,
skipped and unexecuted coverage. Native tests do not qualify the viewer or
TCP interoperability. Follow upstream changelog rules for application work;
documentation alone does not require new application packages or a version
bump. Keep release publication and approval separate from code validation.
