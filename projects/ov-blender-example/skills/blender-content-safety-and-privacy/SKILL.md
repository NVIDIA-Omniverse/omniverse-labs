---
name: blender-content-safety-and-privacy
description: Safely inspect, open, import, modify, and share externally supplied Blender, USD, texture, archive, and reference content. Use before handling content of unknown or third-party origin, downloading dependencies, or publishing scene evidence.
---
# Blender content safety and privacy

## Establish the boundary

1. Identify every supplied file, archive, URL, library, and referenced dependency. Treat unknown or third-party content as untrusted until the user confirms its origin and intended use.
2. Work on a copy in a dedicated output directory. Preserve the original bytes and record a cryptographic hash when provenance matters.
3. Keep generated files, logs, previews, and captures local by default. Share or upload them only when the user requests it and the destination is appropriate.

## Inspect safely

1. Inspect filenames, types, sizes, hashes, archive listings, and dependency metadata before opening content in an authoring application.
2. Open untrusted `.blend` files with automatic Python execution disabled. Do not enable embedded scripts, startup handlers, scripted drivers, add-ons, or referenced executable content without explicit trust and user approval.
3. Use an isolated Blender profile or disposable process for untrusted scenes. Avoid loading unrelated user preferences, credentials, extensions, or writable project locations.
4. Treat USD layers, textures, linked libraries, fonts, archives, and other dependencies as separate inputs. Do not assume a trusted container makes all referenced content trusted.

## Bound materialization

1. Set reasonable limits before downloading or extracting: total download size, expanded size, file count, nesting depth, dependency count, and processing time. Stop and report when a limit is exceeded.
2. Download into a temporary location, verify the expected origin and digest when one is provided, then publish to the working cache atomically.
3. Reject archive entries that escape the destination, use absolute paths, or traverse through links. Do not overwrite existing files during extraction.
4. Follow redirects only to expected HTTPS destinations. Never place credentials or restricted URLs in scene metadata, filenames, reports, or command output.

## Protect people and content

1. Confirm that the requested use and sharing of assets, textures, reference imagery, fonts, labels, and captured UI are permitted by their license and privacy terms. Record attribution or use restrictions when relevant.
2. Exclude unrelated windows, notifications, filenames, identities, and private project details from screenshots and recordings.
3. Make the smallest necessary edits, preserve reversibility, and do not replace the source artifact with a converted or sanitized derivative.
4. If safe handling cannot be established, stop before opening or publishing the content and explain the unresolved risk and a safer next step.
