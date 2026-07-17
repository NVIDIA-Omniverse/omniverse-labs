---
name: blender-asset-library-integration
description: Build a safe Blender Asset Browser integration for local or remote-backed `.blend` and USD assets, using a materialized cache, catalogs, previews, provenance metadata, and explicit import operators. Use when extending the OVRTX add-on with an artist-facing asset browser; do not assume arbitrary cloud URLs are native Blender libraries.
license: "Apache-2.0"
metadata:
  author: "Max Bickley"
  version: "0.1"
  team: "omniverse"
  domain: "physical-ai"
  tags:
    - blender
    - omniverse
    - ovrtx
    - assets
---
# Blender Asset Browser integration

Start with a local cache registered as a normal Blender Asset Library. This is
the portable path for artists and avoids binding the add-on to a provider SDK.
An HTTP remote-library experiment may be added only after the local path works
on the target Blender build.

## When to Use

Use when extending the OVRTX add-on with an artist-facing asset browser; do not assume arbitrary cloud URLs are native Blender libraries.

## Security

The add-on source is available; remote storage, signed URLs, and OVRTX/OVPhysX
runtime components are external dependencies. Do not place credentials,
bearer tokens, restricted URLs, runtime implementation details, or source-machine absolute paths in
`.blend` files, manifests, logs, or skill instructions. Use a user-configured
provider/cache adapter and treat download failures as visible errors.
Treat remote manifests, archives, `.blend` files, USD layers, and their
dependencies as untrusted content and apply
`blender-content-safety-and-privacy` before opening or importing them.

## Instructions

Materialize a cache such as:

```text
<user-cache>/ovrtx-blender/assets/
  blender_assets.cats.txt
  placeholders.blend
  previews/<asset-id>.png
  payloads/<asset-id>/...
  manifest.json
```

The manifest should map a stable asset ID to kind (`blend` or `usd`), relative
payload root, dependency list, source revision/hash, license, preview, and
import policy. Keep the payload closure complete; a USD root layer alone is
usually insufficient.

Before publishing an entry to the cache, download into a private temporary
location, enforce configured download-size, expanded-size, file-count, and
dependency-depth limits, and verify its declared digest (and signature when
supplied). Reject archive traversal, absolute output paths, symlinks, and any
entry that escapes its payload root. Publish the verified payload atomically.

Register the cache with Blender's normal Preferences → File Paths → Asset
Libraries flow or the documented `bpy.ops.preferences.asset_library_add` call.
Mark placeholder objects/assets with `asset_mark()`, catalogs, previews, and
non-secret metadata such as `source_asset_id`, `source_kind`, `source_sha256`,
and `import_policy`. Never treat a USD file as a native Blender datablock.

## Import behavior

Use explicit, reversible operators:

- `.blend`: materialize/verify the payload, then append or link the selected
  collection/object/material according to the user's choice; inspect untrusted
  files in an isolated profile with automatic script execution disabled;
- `.usd`: materialize and verify the root plus dependency closure, then import
  through Blender's USD workflow or open it through the OVRTX add-on;
- a placeholder that cannot resolve its manifest entry remains visible with an
  actionable error and is not silently replaced by an empty object.

Preserve source asset ID and hashes after import. Keep append/link/open choices
and destination collection explicit. Do not run arbitrary downloaded Python or
register unknown handlers from a payload; provider code belongs in the
add-on's reviewed source.

## Validation checklist

- Target Blender/add-on version and cache root are recorded.
- Catalog file, manifest, payload closure, and at least one preview survive a
  save/reopen in a clean user profile.
- Both `.blend` and `.usd` entries appear and have correct tags/description.
- Append/link/import/open operations resolve the declared payload and preserve
  identity; unresolved dependencies are reported with their relative path.
- Cache eviction or refresh cannot delete an asset currently in use without an
  explicit user action.
- Optional Blender remote HTTP-manifest support is tested separately and is
  never presented as direct `s3://`, `ovstorage://`, or arbitrary provider
  support. Signed URLs are fetched into the cache, not persisted in metadata.

Write `asset-library-report.json` with cache/manifest hashes, Blender/add-on
versions, asset IDs, import actions, dependency status, and
`status: pass|blocked|fail`. Pair this skill with
`blender-addon-extension-development` when changing registration or operators.
Use relative, sanitized paths in any shareable report and pass diagnostic
artifacts through `blender-sanitized-support-bundle` before sharing.
