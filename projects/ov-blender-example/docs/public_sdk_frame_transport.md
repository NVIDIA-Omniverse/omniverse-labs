# Public SDK frame transport contract

The external runtime owns OVRTX, OVStage, OVStream, and Warp. Blender loads only
the original Python reader and framing modules. The wire layout follows the
[public OVStream architecture](https://github.com/NVIDIA-Omniverse/ovstream/blob/main/docs/ARCHITECTURE.md).

## Owned copies and acknowledgments

A publication returns a unique copy token and the SDK's actual stream, producer,
sequence, timestamp, and dimensions. The producer refuses another publication
until the consumer has copied the frame into owned bytes and acknowledged that
token. SDK slot sequence checks alone do not provide a lease.

The session owner holds its control transaction across copy and acknowledgment.
A copy error must discard/close the cached reader and release or cancel the
current token. A subsequent frame can attach a fresh reader to the same live
producer. Reader close detaches; it never unlinks producer resources.

Resize replaces only the transport stream. Worker death requires a new epoch.
Cleanup unlinks only names recorded in that worker's ownership registry, after
its process has been confirmed dead. It does not scan or clear unrelated SHM.

## Windows MVP output boundary

One selected RenderProduct with either LdrColor or HdrColor is supported.
Multiple selected products and additional AOVs (including semantic capture) are
future work. Client and worker entrypoints reject these requests before native
rendering; publication also validates the selection, so no first-product fallback
can silently satisfy a multi-product request. The semantic prefix RNA property
is retained for saved-file compatibility, labeled unavailable, and rejected when
nonempty. It is not offered by the render panel. Internal semantic decoding tests
preserve candidate-format coverage, not production admission.

## Raw LDR and HDR frames

Public OVStream SHM carries BGRA8. The explicit `raw_bytes_over_bgra8` application
encoding uses these byte positions as a container for typed, tightly packed
planes. It performs no quantization, channel swap, or color transform. The
descriptor declares little-endian formats and exact plane bounds.

LDR uses a typed `rgba8` plane at every resolution, including 720p. This avoids
the ordinary image route's GPU RGBA-to-BGRA conversion and the reader's CPU
BGRA-to-RGBA conversion and additional copies. HDR uses the same raw carrier.

OVStream LDR pixels arrive pre-corrected and bypass Blender color correction.
Only scene-linear HDR receives Blender's configured view/display transform.
Workers isolate Blender's inherited `OCIO` variable; the Blender process retains
its own display configuration. The raw carrier itself never color-corrects either
plane, and optional float32 HDR staging does not change binary16 payload bits.

The publisher copies borrowed renderer maps into one owned CUDA byte array,
then publishes chunks through an official two-slot ring. The default carrier
budget is 64 MiB on Windows and 4 MiB elsewhere; ring headers and pitch count
toward a separate 256 MiB allocation limit. The owned payload limit is 256 MiB.
The client copies each chunk into bounded owned memory and returns read-only
plane views retaining that allocation.

HDR float32 presentation staging defaults off. `OVRTX_HDR_GPU_FLOAT32=1`
explicitly enables it on Windows; `0` disables it. An enabled request supplies
the plane at every supported size, including 720p and 4K, or rejects the request
before allocation if the complete message exceeds 256 MiB. A 4K HDR message with
staging contains 199,065,600 bytes and uses three bounded Windows carriers; the
64 MiB carrier limit is not a limit on the entire message. The exact binary16
plane retains every bit, including NaNs and signed zero. The optional float32
plane converts RGB and sets presentation alpha to 1.0.

Each chunk identifies its transfer, index, total size, offset, sequence, and
producer epoch. `next_raw_chunk(token)` acknowledges the previous copied chunk
and publishes the next. Final `release(token)` frees the retained GPU payload;
partial transfers require `cancel_raw_transfer(token)`. Failed publication
poisons the transfer until cancellation. A worker restart cannot replay a
partial chunk request: discard the assembly and render a fresh frame.

Cancellation waits for the captured packing CUDA stream, including work queued
before a failed next-chunk event was recorded. A failed synchronization retains
the transfer for explicit teardown. A failed final ACK detaches the consumer
mapping and poisons the session; copied pixels are not returned as a frame.

`health.frame_resources` reports the publisher's active payload, one cached
payload, carrier GPU bytes, two-slot SHM ring bytes, pending chunks and held
leases separately. The ring includes row padding and headers, and can approach
128 MiB for a 64 MiB carrier. These counters exclude renderer-owned borrowed
maps and driver caches. The host assembler owns one whole-message allocation;
its plane views retain that allocation without duplicating it. Blender's
presentation buffer is a separate allocation and belongs in presentation-level
memory measurements. After successful ACK the active payload becomes the single
exact-size cache; cancellation never caches a partial transfer.

The worker registry uses a unique temporary file and atomic replacement. Windows
sharing/access-denied replacement failures retry for at most 310 ms. Persistent
failure remains a visible teardown error, preserves the previous complete
registry, and cleans only that attempt's temporary file. Readers never delete
producer resources; cleanup still requires authenticated, confirmed-dead owner
identity.

No raw-frame file fallback is part of this transport. The existing BGRA image
publisher and reader remain available for callers explicitly using that format.

## Focused acceptance scripts

- `run_ovstream_raw_frame_smoke.py`: verifies the locked SDK before imports; all
  binary16 bit patterns, odd widths, padded source/carrier rows, source overwrite,
  delayed consumer, duplicate ACK rejection, partial cancellation, 720p/1080p/4K
  LDR and HDR staging off/on, forced 4 KiB chunks, bounded ring and cleanup.
- `run_ovstream_receiver_recovery_smoke.py`: injected mapping failure after real
  attachment, fresh-reader recovery with the same PID/epoch/stream, delayed
  consumer, refused overwrite attempts, exact pixels, and cleanup.
- `run_public_sdk_recovery_smoke.py`: actual renderer failure/replay and resize.
- `run_public_sdk_first_frame_smoke.py --python <runtime> --output <fresh-dir>`:
  real rendered LDR/HDR, authenticated producer identity, mapping, owned copy,
  ACK, clean teardown and retained pixels after death. Add
  `--require-venv-child` to require a distinct Windows launcher and producer PID.
  Host validation runs this preflight before the broader suites.
- `run_owned_frame_failure_smoke.py`: real renderer frames with mapping/copy/
  disconnect faults and producer death before mapping, during copy, between
  chunks and before final ACK. A small-carrier wrapper changes only the test
  allocation size after production runtime validation. Every fault must reject
  its frame, preserve bounded ownership and recover a fresh epoch explicitly.
  `--faults <names...>` selects a scoped repeat; the default runs every boundary.

The first two use a separate consumer process that asserts no NVIDIA SDK or Warp
module was imported. Transport proofs do not establish viewport frame rate or
complete renderer feature coverage.

Windows technical completion currently uses representative checks below ten
minutes per run. The exhaustive three-resolution/staging benchmark and sustained
100-resize/20-restart/30-minute mixed stress campaign are deferred; exact-byte
transport coverage is distinct from viewport performance certification. Linux
runtime, installation and GPU execution remain pending after final human
acceptance and discussion.

Renderer first-frame, failure and recovery probes run with the separate
[pinned OpenUSD consumer](public-sdk-consumer.md), passing the locked SDK
interpreter via `--python`. Raw transport producers still run in SDK Python.
