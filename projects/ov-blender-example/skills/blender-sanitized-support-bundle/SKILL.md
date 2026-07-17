---
name: blender-sanitized-support-bundle
description: Collect Blender diagnostics locally and create a separate, sanitized support bundle for sharing. Use when preparing logs, reports, screenshots, scene metadata, or reproduction artifacts for troubleshooting or review.
---
# Blender sanitized support bundle

## Collect locally

1. Confirm the issue, intended recipient, and minimum evidence needed to reproduce or diagnose it.
2. Collect diagnostics into a local, access-restricted working directory. Keep the original collection private and unchanged.
3. Record collection time, tool and application versions, included file names, and a cryptographic hash for each original artifact. Do not collect unrelated scene content or user data.

## Create a shareable derivative

1. Copy only necessary artifacts into a separate staging directory. Never sanitize in place.
2. Remove or replace credentials, tokens, cookies, keys, restricted or signed URLs, environment-variable values, absolute home paths, private host names and addresses, unrelated UI, proprietary asset or project names, and personal information.
3. Review command lines, stack traces, metadata, manifests, filenames, embedded paths, thumbnails, screenshots, and scene-linked dependencies; visible text is not the only disclosure surface.
4. Crop or recreate screenshots when blurring would leave recoverable detail. Substitute minimal reproduction assets for proprietary scene content when possible.
5. Use stable placeholders such as `<HOME>`, `<HOST>`, and `<ASSET>` when their relationships matter to diagnosis. Do not invent replacement values that could be mistaken for real evidence.

## Preserve provenance

1. Generate a manifest for the sanitized derivative listing each included file, its hash, the source artifact it derives from, and the kind of redaction or substitution applied. Describe transformations without copying sensitive values into the manifest.
2. Hash the final archive after packaging. Keep the private original and sanitized derivative clearly named and physically separate.
3. Inspect the final archive from scratch: list every member, search text and metadata for sensitive values and path patterns, and visually review every included image or video.
4. Report what was omitted, redacted, or substituted and whether those changes limit diagnosis. Share only the sanitized derivative after the user confirms the destination when it is not already explicit.

If adequate evidence cannot be shared safely, provide a minimal written symptom and reproduction summary instead of the bundle.
