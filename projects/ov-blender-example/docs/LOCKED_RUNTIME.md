# Locked external runtime

`runtime/public-sdk-lock.json` is the canonical CPython 3.13 x64 tuple. The
requirements text is generated from it and checked for disagreement. Windows
artifacts pin OVRTX, OVStage, OVStream, OVPhysX, Warp, NumPy and packaging;
every wheel has its origin URL and SHA256. The Windows lock also pins a digest of
each complete wheel file inventory, including bundled native dependencies.
Linux x64 artifact metadata and layout contracts are retained; Linux runtime
qualification is deferred and no Linux runtime pass is implied.

Use `scripts/setup_public_sdk.py` with external CPython 3.13 to create a fresh
environment. It delegates to `runtime/materialize_sdk.py install --cache PATH`.
The materializer downloads only exact locked URLs, checks hashes, installs
offline with `--no-deps --no-index --require-hashes`, runs `pip check`, and checks
all installed package bytes against the trusted wheel inventories. It rejects
foreign import locations, untracked native libraries, startup hooks, wrong
versions and wrong interpreter ABI. Blender never imports these SDKs.

Standalone renderer diagnostics use a separate
[OpenUSD consumer environment](public-sdk-consumer.md) to retain immutable scene
and asset versions. Set up its hash-pinned packages and `consumer_python` before
running host validation. Blender uses its bundled OpenUSD; the locked SDK
environment and portable sidecar remain unchanged.

`runtime/materialize_sdk.py verify` checks an existing installed environment;
`resolve --cache PATH --output receipt.json` verifies the artifact selection in
a clean external Python without installing packages. Always supply an owned
cache location. The installed `public-sdk-integrity.json` contains the package
inventories, whose digests are independently pinned by the shipped lock. Worker
startup and physics admission use this same identity, including native hashes
and the actual interpreter version/executable digest.

`scripts/build_public_sdk_packages.py sidecar` uses the same installer, or checks
`--sdk-python` before copying its selected distributions. `--unpacked` retains
the portable sidecar directory instead of producing a ZIP. The sidecar roster
is checked against the trusted package inventories before running its Python;
altering a package and its self-declared file hash does not bypass the lock.

Installations create a new candidate and replace the ready/configuration pointer
only after verification. Failed downloads remove only their incomplete file;
failed candidates do not replace the previous usable runtime. On a cached hash
failure, remove only the named owned cache artifact and retry. On an installed
file/ABI mismatch, rebuild the complete tuple in a new candidate; do not upgrade
one SDK independently. Keep the previous environment until the replacement is
accepted. Temporary installer and packaging files stay in owned staging paths.

Lock maintenance is an explicit review operation done with the upstream build
repository's lock tooling, which is not part of this project: it refreshes
metadata for the fixed spec versions, verifies Windows downloads and freezes
actual wheel inventories. Changing the lock invalidates runtime and physics admission
identities and requires fresh qualification. Qualification results are external
evidence receipts, not declarations inferred from the metadata lock.
