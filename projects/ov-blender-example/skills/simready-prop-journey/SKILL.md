---
name: simready-prop-journey
description: "Take one Blender prop through a concise SimReady workflow: preserve the source, author and validate it with the official SimReady add-on, check the supported OVRTX conversion boundary, run a native OVPhysX drop/contact test when appropriate, and prepare a portable USD handoff. Use when an artist wants an end-to-end physics-ready prop workflow rather than an infrastructure or validation-report project."
---

# SimReady prop journey

Use this as the user-facing orchestrator. Keep each specialist gate independent;
later physics success does not repair failed authoring or export.

## Workflow

1. Preserve the supplied source and work on a caller-owned derivative. Use
   `blender-content-safety-and-privacy` for untrusted inputs.
2. Run `simready-addon-install-and-authoring`. Require the real add-on surface,
   its supported authoring structure, named validators, and a reopened USD.
3. Run the pinned OVRTX SimReady conversion test described by that skill. The
   current public example supports a deliberately narrow uni-body shape; stop
   on its structured unsupported cases rather than rewriting the asset silently.
4. When the prop needs a behavior check, run
   `ovphysx-drop-contact-acceptance` through the official
   `run_ovphysx_drop_probe.py --require-real` path. Authoring eligibility and
   runtime behavior remain separate results.
5. Use `usd-copy-and-flatten` only when the recipient needs localization or a
   flat derivative, then inspect the exact candidate with
   `usd-inspect-and-provenance`.

## Completion

Call the requested journey complete when the add-on validators pass, the USD
reopens, the current conversion boundary accepts the prop, and any requested
native behavior check passes. Summarize the prop, outputs, pass/blocker, and
important limitation. Produce a manifest, hashes, screenshots, or full reports
only when the caller needs a reproducible handoff or review artifact.
