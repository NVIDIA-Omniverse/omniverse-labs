---
name: simready-addon-install-and-authoring
description: Install or verify the official SimReady Blender add-on, author one physics-ready prop through its real operators and properties, run available validators, export and reopen USD, and verify compatibility with the pinned public OVRTX uni-body conversion. Use for SimReady asset preparation; never substitute plain Blender custom properties or generic USD export for add-on authoring.
---

# SimReady add-on install and authoring

Work on a caller-owned copy. The official SimReady add-on owns authoring,
validators, and export semantics. The pinned OVRTX example consumes supported
SimReady state but does not replace that add-on.

## Preflight

1. Confirm the Blender/add-on versions supported by the supplied distribution.
2. Run `scripts/probe_simready_surface.py` inside the target Blender. It reports
   the installed module and exact registered operator candidates without
   mutating the scene. A missing required authoring surface is `blocked`.
3. Read `references/simready-public-contract.md` before scripting. Probe
   operator RNA in the running version; never infer an operator from an old
   screenshot or invent equivalent custom properties.
4. From the `public/` distribution root, run:

```text
python3 -m pytest -q tests/test_simready_physics_conversion.py
```

This is the pinned public conversion contract, not proof that the external
SimReady authoring add-on is installed.

## Author and validate

1. Inventory the selected prop, materials, hierarchy, units, movable root, and
   contact surfaces. Separate independently movable pieces before authoring.
2. Use the installed add-on's registered collection, physics-property,
   material, collider, mass, and validation operations. Keep mass positive and
   finite; keep supports static; preserve functional openings in collider work.
3. Run every applicable named validator. Report focused and export-eligibility
   results separately.
4. Export with the installed SimReady hook/workflow, not plain USD plus invented
   metadata. Reopen the composed result and inspect default prim, dependencies,
   material bindings, body/mass/collider schemas, finite transforms, and unique
   paths.
5. Check the OVRTX conversion boundary from the reference. Unsupported joint,
   collider, material, or topology structures are actionable blockers; do not
   destructively coerce the source to manufacture a pass.

## Result

Summarize what was authored, validator/export/reopen status, conversion status,
and the first blocker. Add a source map, hashes, complete validator JSON, or UI
captures only when requested. A passing authoring result makes the USD eligible
for physics testing; it does not prove native simulation behavior.
