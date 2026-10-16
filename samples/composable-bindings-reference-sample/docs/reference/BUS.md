# Replacing the live event bus

`Service(config, bindings, bus=bus, view=view, recordings=path)` accepts a replacement bus. The reference HTTP adapter and shared clients rely on more than publish/subscribe: they need a consistent initial state and configuration, explicit loss handling and stable ordering. NATS or Kafka would sit behind an adapter implementing this contract; neither is a tested dependency here.

## Python interface

| Member | Required behavior |
| --- | --- |
| `recorder` | Attribute initialized to `None`. When a recording directory is supplied, Service attaches its `FileRecorder` here. |
| `publish(event) -> bool` | Accept a normalized event, deduplicate `(source, id)` occurrence keys, and return true only for a new event. Identical retries return false; conflicting retry payloads raise `ValueError`. |
| `configure(context, clear=False, reason='configuration_changed')` | Order a new catalog/view/configuration revision against events and broadcast a snapshot. Clear retained readings when requested; otherwise preserve them. |
| `subscribe() -> subscription` | Register a reader and atomically enqueue its initial snapshot before any subsequent event. A capacity rejection raises `ValueError` (HTTP adapter returns 503). |
| `subscription.get(timeout=...)` | Return a packet, or raise `queue.Empty` on timeout so the HTTP adapter can send a heartbeat. A broker adapter can bridge its receiver to such a queue. |
| `unsubscribe(subscription)` | Stop delivery and release subscription resources. |
| `start_recording()` | If recording is configured, start the attached recorder with a snapshot at an atomic stream boundary; return recording metadata. No accepted event may fall between the snapshot and recorder attachment. |

The service no longer reaches into a bus lock or private snapshot method. A replacement may use broker transactions, cursors and a materialized state store instead of the reference bus's lock. `FileRecorder.offer(packet)` must receive accepted live packets and configuration snapshots in the same order as readers when capture is active; start-recording consistency is the adapter's responsibility. File replay remains separate from source history.

## Packets and retained state

```json
{"kind":"snapshot","reason":"initial","epoch":"stream-instance",
 "sequence":42,"events":[],"catalog":{"bindings":[],"metrics":[]},
 "view":{"version":1,"displays":[],"scene":[]},
 "config_revision":3,"config_error":null}
```

`events` contains the latest accepted event for every `(event.source, event.subject, event.type)` slot, selected by observation time. Do not retain only the last event globally, or collapse IDs from different issuing systems. Catalog entries carry the actual `source` and `id`, labels and applicable metrics. The snapshot's configuration and retained events must describe the same accepted revision. `config_error` also travels with the context, including configuration rejection/recovery notifications.

```json
{"kind":"event","epoch":"stream-instance","sequence":43,"event":{"...":"normalized CloudEvent"}}
```

Each accepted event or configuration boundary advances the stream sequence. `epoch` identifies the continuity of that ordered stream; change it when restarting without continuity. A new or reconnecting client always receives a fresh complete snapshot. The current HTTP adapter does not implement cursor-based catch-up or honor Last-Event-ID.

Registration and the snapshot boundary must be consistent: snapshot at cursor N, then deliver updates strictly after N, with no race that loses the first update. Current clients trust this ordering; they do not repair arbitrary reordering or detect all missing sequence numbers. A configuration snapshot replaces their catalog/view and clears/rebuilds their sample cache.

On subscriber overflow, the reference bus drops its queued deltas and sends a fresh complete snapshot with `reason: slow_consumer_gap`. Clients show a gap diagnostic; this restores current display state without pretending to recover every transition. A replacement must either retain/deliver all queued events within its contract or signal a reset with an explicit gap. Never silently drop deltas. Deduplication retention, late observations, conflicting retries, quotas and restart recovery must be specified for the deployment; the reference bus retains only 4,096 occurrence keys.

## A broker-backed design

1. Serialize accepted events and configuration changes to an ordered durable log. Keep the edge's revision check coupled to that publication boundary. Multiple service instances additionally need shared revision ownership and deduplication; injecting a bus alone does not provide those.
2. Maintain latest state by source/ID/event type and keep catalog, view, revision, error and the applied log cursor together in a durable materialized store.
3. On subscription, obtain a consistent snapshot at cursor N, then consume the log after N. Buffer updates while sending the snapshot, or use a transaction/cursor protocol that guarantees the same result.
4. If partitioning the log, translate partition cursors into a consistent snapshot barrier and ordered packet stream for these clients. A single integer sequence must not imply a total order the implementation does not provide.
5. Handle bounded queues, explicit gaps, reconnect snapshots and unsubscribe cleanup. Restore retained state and deduplication state on restart, or use a new epoch and a documented reset.

Test a late join after values already exist, concurrent publish/subscribe, reconnect, duplicate/conflicting retries, out-of-order observation timestamps, configuration changes, slow-reader overflow and restart recovery. These tests check semantics, not just throughput. A retained topic alone is insufficient unless it also handles the configuration boundary and snapshot-to-live transition.
