# Live recording selection

Open **Live → Recording selection** while a compatible producer is connected.
The nonmodal panel always targets that live connection, including while an older
file is displayed. Selecting objects in the viewport never changes recorder scope.

1. Check catalog rows in **Selected**, or explicitly choose **All permitted**.
   Empty Selected is not All. Context-only rows cannot be selected.
2. Choose **Primary** or **+ Secondary**. These are the producer's two policies,
   not individual attribute switches. Whole articulations include their links and
   joints; other dependency/contact output follows the SDK's selection contract.
3. Press **Apply**. The producer may reject, deny, or narrow the request. Check the
   acknowledgement and **Actual** policy, not just the desired controls.

The catalog uses 128-row pages, name-prefix filtering, and parent navigation via
the row's name. **All rows** removes the parent filter; clear the prefix to remove
the name filter. **Refresh catalog** is explicit and does not restart recording.
Generation changes clear checked handles; reconnects also reset desired policy
and request history. Handles and requests never transfer between connections.

## Requests and frames

Sending is not success. The viewer waits for the matching producer acknowledgement.
After ten seconds without one, the result becomes **Unknown**. An explicit
supersession is required before another request: the earlier request may still
execute. Late acknowledgements cannot settle a different request. A reverse send
stalled for thirty seconds fails without stopping forward telemetry reception.

Applied means the producer restarted recording, not that a new frame is ready.
Discovery and acknowledgements never create placeholder viewport frames. The
panel separates producer capture epoch, catalog generation, and completed segment
identity. Follow-latest adopts a new segment only after a complete frame. Adopting
it clears object/inspector selection and selection history, while retaining camera,
render settings, and Pre/Post playback mode. Viewing an older segment is unaffected.

One TCP connection is one retained recording, even after multiple Applies. All
segments stay together in its saved OVD file; Apply never creates another history
item or triggers autosave. Connection completion/Detach/New finalizes that recording.
Automatic saving remains optional. Saving a displayed live capture freezes it;
saving an older displayed file leaves the live producer running.

## Compatibility and limits

Controls require advertised SDK capabilities: selection (0x4) and catalog (0x8)
are independent. Selection-only producers permit explicit All; unsupported actions
send nothing. Telemetry-only producers remain usable. The existing anonymous,
all-interface listener and deployment/firewall responsibilities are unchanged.

Catalogs and recordings remain bounded. Apply is refused when a recording budget
is at least 75% used or cannot reserve the previous segment's measured charge (with
minimum reserves). This is not a promise that an unexpectedly larger next snapshot
will fit. Refresh reserves catalog wire space, not another detailed snapshot. Hard
input limits still reject an oversized stream; a new connection starts fresh budgets.

The genuine interoperability fixture pins SDK `ad5c86f498025752261284367f0f1908d392a6b1`;
the viewer's existing SDK pin is unchanged. At that sender revision, a selected
rigid body contacting an unselected GPU articulation link has a known Direct GPU
counterpart limitation. CPU evidence does not establish GPU-family behavior.

The separate GPU fixture exercises PBD systems, base/diffuse buffers, and deformable
surfaces/volumes with actual simulation state and both recording policies. It uses
the same SDK pin and a separately verified CUDA 12.8 toolkit; no SDK source patch
or additional runtime dependency is shipped with the viewer. Set
`OVDNEXT_SDK_GPU_SENDER` to its executable and `OVDNEXT_REQUIRE_SDK_GPU_SENDER=1`
when running `python/tests/test_sdk_tcp_gpu.py` through `ci/test_ovui.py`. The CPU
equivalents are `OVDNEXT_SDK_TCP_SENDER` and `OVDNEXT_REQUIRE_SDK_TCP_SENDER`.
Ordinary environments without a genuine sender report an explicit skip; a required
lane fails if it is missing. Saved evidence includes independent SDK decoding,
recording/binary hashes, producer diagnostics, and GPU device/driver identity.
