# Standalone diagnostic consumer (Windows)

Blender uses its bundled OpenUSD to capture immutable accepted scene and texture
versions. Standalone renderer diagnostics require a separate consumer environment
with OpenUSD. Never install these packages into the locked NVIDIA SDK environment.
The consumer contains none of `ovrtx`, `ovstage`, `ovstream`, `ovphysx`, or `warp`;
those remain in the authenticated external worker.

Create the Windows x64 CPython 3.13 consumer with the repository script:

```powershell
py -3.13 scripts/setup_public_sdk_consumer.py --environment out/consumer-env --config out/public-sdk-config.json
out/consumer-env/Scripts/python.exe -I scripts/setup_public_sdk_consumer.py --verify
```

Use the actual configuration path created by `setup_public_sdk.py`. Creation
requires a new directory. The consumer wheels (`usd-core==25.11`,
`numpy==2.3.3`, and `pillow==11.3.0`) have pinned SHA-256 hashes. Pillow writes the
required recovery diagnostics. `consumer-identity.json` records the
interpreter and installed package/native-file hashes; verification rejects drift.
This lock currently covers Windows only. No Linux qualification is claimed.

`validate_public_sdk_host.py --config <config>` verifies both environments. Native
SDK probes use `python`; renderer async/recovery consumers use `consumer_python`.
For direct diagnostics, use the consumer interpreter and pass the SDK interpreter
as `--python`, preserving its venv launcher path:

```powershell
out/consumer-env/Scripts/python.exe scripts/run_public_sdk_first_frame_smoke.py --python <sdk-python> --gpu 0 --output <fresh-output>
out/consumer-env/Scripts/python.exe scripts/run_owned_frame_failure_smoke.py --python <sdk-python> --gpu 0 --output <fresh-output>
```

Keep each probe's existing image, identity, fault, and cleanup assertions enabled.
Missing OpenUSD is a setup error; immutable snapshot protection is never skipped.
