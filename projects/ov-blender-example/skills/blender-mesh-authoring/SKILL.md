---
name: blender-mesh-authoring
description: Create, modify, parent, and validate Blender mesh objects with stable names, direct data APIs, explicit modifier order, evaluated geometry checks, and reversible MCP transactions. Use for primitives, custom vertex/face meshes, transforms, parenting, bevel/solidify/subdivision modifiers, or topology repair when an out-of-box agent needs exact bpy mechanics rather than general modeling advice.
---

# Blender mesh authoring

Use `blender-python-execution` for live MCP calls. Prefer direct mesh/data APIs
for deterministic creation and reserve context-sensitive operators for edit-mode
tools that have no practical data API equivalent.

Read `references/bpy-mesh-recipes.md` before creating geometry, changing
parenting, or applying modifiers. Use caller-owned stable names and refuse to
replace an incompatible existing datablock.

## Authoring contract

1. Inspect the target object, mesh users, parent, collections, modifiers, mode,
   selection, and evaluated bounds before mutation.
2. Create or update only named caller-owned objects. Build custom topology with
   `Mesh.from_pydata` or `bmesh`; call `mesh.validate()` and `mesh.update()`.
3. Set transforms absolutely. When reparenting without a visible jump, preserve
   `matrix_world`, assign the parent, then restore `matrix_world`.
4. Name modifiers, reject type collisions, and set their order deliberately.
   Do not apply a modifier unless destructive topology conversion was requested.
5. Call `view_layer.update()`, inspect the evaluated object, and run
   `scripts/audit_mesh.py` on every primary target.
6. Inspect a current viewport or inexpensive render for visible tasks. Numeric
   validity does not establish silhouette, proportions, or shading quality.

For MCP, prepend
`MESH_AUDIT_REQUEST = {"objects": ["GEO-subject"]}` and append the complete
audit script. For a saved derivative:

```text
blender --background scene.blend --python scripts/audit_mesh.py -- \
  --objects GEO-subject GEO-trim
```

Require `ok: true`, finite evaluated coordinates and bounds, nonempty evaluated
faces, valid polygon indices, and no zero-area evaluated faces. Boundary and
non-manifold edge counts are reported for task-specific decisions: an open
cloth or plane may be correct, while a printable solid normally requires both
counts to be zero.

## Failure rules

- Stop on missing targets, linked non-editable data, incompatible name
  collisions, non-finite transforms, invalid face indices, or empty evaluated
  output.
- Preserve selection, active object, and mode around operators. Never use
  selection as object identity.
- Do not clear the scene, apply all transforms/modifiers, merge by distance, or
  recalculate normals globally as an incidental repair.
- Save only to an absolute caller-selected derivative path; never overwrite the
  source implicitly.
