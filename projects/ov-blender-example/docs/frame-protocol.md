# Application frame protocol

This contract is shared by the Blender-side reader and external worker through
`addon/ovrtx_blender_example/frame_protocol.py`. It imports only the standard
library. Native SDK imports remain outside Blender. Runtime installation identity
is separately enforced by `public_sdk_lock.py` before the worker loads any SDK.

## Three independent versions

The private bootstrap version is **2**, the JSON control protocol is **1**, the
typed payload version is **1.0**, and the OVStream SHM carrier ABI is **2** with
BGRA8 format code **1**. The carrier ABI does not imply application compatibility,
and application compatibility does not imply the installed SDK tuple is valid.
RGBA8, RGBA16F and optional RGBA32F are application encodings of raw bytes in the
BGRA8 carrier. They are not native OVStream HDR formats.

The bounded stdin bootstrap challenge contains `protocol`, fresh 64-hex `session`,
positive `epoch`, and `application`; Windows also requires the private loopback
`port`. The actual worker's hello repeats protocol/session/epoch and includes its
PID, `capabilities`, and the selected `application`. The parent authenticates
that PID against its owned process tree before duplicating the control socket.
A venv launcher PID cannot stand in for the actual producer.

Application offers contain `control`, `typed_major`, `typed_minor`, `carrier_abi`,
`required`, and `optional`. Exact control/major/carrier versions must match.
Required capabilities are `control=1`, `owned_frames=1`, `typed_frames=1`,
`contiguous_chunks=1`, and `rgba16f_bits=1`. Missing, changed, or unknown required
capabilities fail closed. Unknown optional capabilities are ignored. The worker
selects the lower minor version (currently 0), and all payloads must use it.
Higher minor versions may offer optional additions but cannot change existing
required meaning. All incompatibilities are rejected during bootstrap, before
native imports or frame allocation. Older schema-only payloads are deliberately
rejected; no implicit legacy downgrade is supported.

Control requests and replies keep the existing 16 MiB JSON line bound, ordered
request IDs, authenticated session/epoch, and reply producer PID. Bootstrap lines
are bounded to 4096 bytes. JSON numeric integer fields reject booleans, floats,
negative values and values outside their specified bounds.

## Typed descriptor and product identity

Every `raw_frame` requires `schema=1`, `encoding=raw_bytes_over_bgra8`,
`byte_order=little`, `version={major:1,minor:0}`, `carrier_abi=2`,
`required_capabilities`, `product_id`, `aov`, `storage_width`, `storage_height`,
`byte_count`, and `planes`. `product_id` is an absolute USD prim identifier of at
most 1024 characters; AOV is `LdrColor` or `HdrColor`. The client binds both to
the outstanding request and checks every plane against its requested dimensions.
The response dimensions must match the accepted render size, including a
successfully acknowledged resize.

The MVP accepts exactly one RenderProduct. LDR requires `rgba8` in RGBA8 format.
HDR requires `linear_rgba16f` in RGBA16F and may include `rgba8` and derived
`presentation_rgba32f`. Extra planes and wrong formats fail before owned
allocation. Multiple products and additional AOVs remain explicit future work;
the backend and client reject those requests. The low-level layout codec retains
R32F/R32U definitions for existing internal decoder coverage, which does not
admit additional products to the delivered runtime. Direct publisher probes use
the synthetic default product `/Render/Color`; production always supplies the
actual requested product and AOV.

Plane records require a unique identifier `name` (1–64 characters), `format`,
`width`, `height`, `offset_bytes`, `row_bytes`, and `byte_count`. Application planes
are tightly packed, ordered and contiguous: offset equals the preceding end,
row bytes equal width times format size (4/8/16 bytes for RGBA8/16F/32F; 4 for
R32F/U), and plane bytes equal rows times height. Overlaps, gaps, truncated planes
and trailing unclaimed payload bytes are errors. Dimensions are 1–16384 per
plane. There are 1–32 codec planes, with stricter MVP product constraints above.

Total owned payload is at most 256 MiB. Logical storage dimensions are 1–8192,
with `byte_count <= width*height*4 < byte_count + width*4`. Physical carrier
dimensions are independently 1–8192 and at most 64 MiB. Python bounded integers
are checked before products, offsets or sizes reach native allocation/copy APIs.

**Pitches are distinct.** The publisher reads supported padded source GPU rows
using the source stride and packs only active bytes. Application plane rows are
tight. OVStream carrier rows can be padded to native pitch; the reader validates
pitch and mapped-slot bounds, and copies only active row bytes into owned storage.
Odd dimensions, padded carrier rows and a partial last chunk are supported.
All original binary16 bit patterns, including signed zeros and NaN payloads,
remain exact. Float32 presentation is a separate derived plane.

## Ownership, chunks and validation order

`shm` identifies `stream_name`, `epoch` (the authenticated session nonce),
`producer_pid`, positive uint64 `sequence`, uint64 `capture_timestamp_ns`,
carrier `width`/`height`, and `copy_token`. Stream names are 1–63 safe characters;
epoch and token are bounded nonempty strings. Production uses random tokens.
The enclosing authenticated control session separately carries its increasing
numeric epoch. Producer, session, stream, dimensions, sequence and slot timestamp
must agree with the mapped SHM ABI. PID ownership checks are never bypassed.

Every chunk requires a 32-hex `transfer_id`, zero-based `index`, `count`,
`offset_bytes`, `byte_count`, and `total_bytes`. Count is the ceiling of full
payload/carrier capacity and is at most 65536. Chunks are contiguous and ordered;
all except the last occupy carrier capacity. Layout/product identity and stream
identity remain constant. Sequences increase, timestamps do not go backward,
and copy tokens cannot repeat within a transfer. The assembler validates the
first complete descriptor before allocating; later invalid chunks are rejected
before any read. A failed copy poisons and discards the assembly.

Only the serialized owner sends ACK/cancel. `next_raw_chunk` ACKs the current
token and publishes the next chunk. Final `release_frame` requires the final
chunk; early release, stale/duplicate tokens and out-of-order calls fail. Cancel
retires partial transfers. No new publication overwrites a held frame. The client
returns only complete owned bytes, never mapped SDK storage. SHM slot sequences
and timestamps are checked before and after copying. Producer GPU completion is
separately required before recycling packing storage.

Unknown descriptor fields fail closed except an optional `extensions` object.
Its keys carry ignorable annotations only; required semantics must be expressed
through negotiated capabilities or a new major version. Optional annotations
cannot change offsets, product identity, formats or allocation bounds.

## Errors and validation scope

`PROTOCOL_MISMATCH` identifies application control/typed version incompatibility;
`CARRIER_ABI_MISMATCH` identifies the advertised OVStream carrier ABI;
`CAPABILITY_MISMATCH` identifies required capability failures. `FRAME_DESCRIPTOR`,
`FRAME_BOUNDS`, and `FRAME_IDENTITY` distinguish malformed metadata, arithmetic
bounds, and ownership/product errors. SHM read errors additionally identify ring,
slot, sequence, lease or producer failures. SDK tuple/hash/ABI errors originate
from the separate runtime-lock policy and name the failed package/interpreter;
they are not protocol errors and cannot be repaired by protocol negotiation.

`test_frame_protocol.py` tests complete offers/layouts/chunks and rejection before
allocation/read, plus shared writer policy and worker bootstrap. Existing
`test_worker_identity.py` exercises actual Windows standalone/venv bootstrap and
owned teardown. `test_win32_shm.py` covers padded carrier rows and source mutation,
and raw-frame tests cover exact owned bytes. These CPU contracts do not substitute
for n-03-02's real locked-runtime first-frame, HDR-bit and lease qualification.
