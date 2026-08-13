---
name: usd-remote-stage-mirror
description: >
  Open a USD stage from an http(s):// URL by mirroring it and its full dependency closure
  (sublayers, references, payloads, textures) into a local cache, then opening the local
  copy — because pxr has no HTTP asset resolver. The remote source is never written; the
  stage's user-facing identity stays the URL. Use when a viewer must accept a pasted URL to a
  cloud-hosted USD (e.g. the NVIDIA ConceptCar S3 sample).
license: Apache-2.0
metadata:
  author: NVIDIA Customer Success
  tags: [usd, mirror, remote, s3, http]
  domain: ai-ml
  languages: [python]
---

# Remote stage mirroring

pxr cannot resolve `http(s)://` asset paths, and you must not write the remote source. So:
**download the closure into a local cache and open the local root.** The recipe below is the
PROVEN one from the production app — match it; the obvious "walk the USD arcs and fetch each ref"
approach is INSUFFICIENT and ships a **red car** (see the MDL note).

## `ensure_local(url, data_root="data", progress=None) -> local_root_path`

Mirror under `data_root/_mirror/<host>/<path>`, where `<host>` is the URL authority with `:`→`_`
(`host:port` is illegal on Windows). The local layout MIRRORS the host's layout so the stage's
relative refs resolve. A `<root>.mirror_complete` marker skips the (seconds-long) re-check on later
opens. Steps:

1. **Download the root** layer (stream to disk; `.usd` may be crate-binary — fetch bytes verbatim).
2. **BULK S3-prefix download FIRST (this is the red-car fix).** S3-style hosts allow anonymous prefix
   listing (`GET {scheme}://{netloc}/?list-type=2&prefix=<root-folder>/&max-keys=1000`, paged via
   `NextContinuationToken`). List the root's folder and download **EVERYTHING** under it. This is the
   ONLY thing that covers what pxr **cannot see**: the textures + `import ::Module` deps **inside `.mdl`
   files** (an `.mdl` is not USD; `UsdUtils`/`Sdf` do not look inside it). Miss them and ovrtx's MDL
   compiler fails → the car renders the **flat red error material** and no paint switch changes it.
3. **Fixpoint fallback for non-S3 hosts** (listing refused): loop
   `layers, assets, unresolved = UsdUtils.ComputeAllDependencies(local_root)` — the canonical closure
   API (covers sublayers/references/payloads/asset-attrs incl. textures+MDLs). `unresolved` come back
   anchored-ABSOLUTE under the mirror root; map each back to its URL by path-prefix, download, repeat
   (bounded ~64 rounds) until empty. (A complete S3 listing SUPERSEDES this — `ComputeAllDependencies`
   composes the whole stage and is slow, so only crawl when listing wasn't available.)
4. **Heal collected-nesting.** Collected-asset stages (the ConceptCar) LEGITIMATELY nest a copy of the
   host tree inside the stage folder (`<stage_dir>/<host>/…`) and reference into it; the published nested
   copy can be incomplete upstream while its MDLs use package-relative imports (`.::Module`) that need
   siblings present THERE (pxr's Composer survives via MDL search-path fallback; **ovrtx does not**).
   Hard-link (fallback copy) anything present in the top-level tree but missing at the same relative path
   in the nested copy — same bytes, no extra disk. (So a "doubled host path" is REAL structure, not a bug
   to de-double.)
5. **Windows MAX_PATH (260) junction.** Deep collected trees push absolute paths past 260 — Python
   (long-path aware) sees the files but **native code (pxr's Ar resolver, ovrtx's MDL entity resolver)
   CANNOT open them → materials silently fail → red car.** When the absolute mirror path is long
   (≳180 chars), create a short directory **junction** (`mklink /J C:\ovml\<sha1[:8]> <host_root>` — no
   admin needed, unlike symlinks) and return the stage path THROUGH the junction. This is essential, not
   optional, for the ConceptCar S3 stage on Windows.
6. Return the local root (through the junction if created). Open THAT; the composite only sublayers the
   local mirror — the source is never written. `progress(count, name)` → a `mirror_progress` WS event.

**Why the naive walk fails (the single most common mistake here):** hand-walking `Sdf` arcs +
`info:mdl:sourceAsset` gets the `.mdl` FILE (it even resolves as a USD asset, `resolvedPath=True`) — but
NOT the textures/modules referenced INSIDE the `.mdl`, and not through a MAX_PATH-safe path. Result: a
fully-downloaded closure whose car still renders red and whose paint switch moves zero pixels. Use the
bulk-listing + heal-nesting + junction recipe above.

## Don't let the S3 path crowd out the simple non-S3 closure (regression guard)
The bulk-S3-listing is for S3 hosts; a plain `http://127.0.0.1:<port>/root.usda` closure (no S3 listing)
MUST still mirror correctly via the `ComputeAllDependencies` fixpoint. **Verify BOTH:** (a) the REAL S3
ConceptCar car renders non-red + paint switches; AND (b) the canonical **2-file localhost closure** (root
references a child that defines a variant set) — after open, the child's variant set MUST appear in the
scan. Fixing S3 while breaking the localhost closure is a regression — keep the non-S3 fixpoint working.

## PREFER already-local dependencies — don't re-mirror a stage whose closure is already on disk
If the user opens a LOCAL stage whose asset refs already resolve to existing local files, **do not
re-download** — resolve in place. Only mirror genuinely remote (`http(s)://`) refs not already present.
The reference app opens a local closure directly and never re-downloads.

**This module uses `pxr`/`UsdUtils`**, so it must run wherever pxr lives (the single server process in
the supported design). If you ever isolate pxr in a subprocess, the mirror must run THERE too — calling
`pxr` from the ovrtx process after the renderer exists triggers the `ParticleField` clash → 500 on a URL open.

## Identity
Keep the **URL** as the stage's user-facing identity (`source_url` in `/api/open` and
`/api/stage`, and the `usd` field a project saves). The local junction path is an
implementation detail. A bare re-open of the URL resolves from the cache in ~1 s.

## Gotchas
- The remote source is read-only: only the local cache is written. Verify the served bytes
  are unchanged after an open (sha256) — a mirror that writes back to the source is a bug.
- Mirror the FULL closure: a viewer that fetches only the root and then fails to resolve a
  reference shows an empty/half stage. Test with a 2-file closure (root references a child
  that defines a variant set) and assert the scan reports the child's variant set.
- Large closures (the ConceptCar is hundreds of MB) download once; cache and reuse. Stream to
  disk; don't hold it all in memory.
