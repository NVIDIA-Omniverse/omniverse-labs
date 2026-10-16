# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ingestion, normalization, conditions and bounded pub/sub; no HTTP or Kit."""
import queue
import threading
import time
import uuid
from collections import OrderedDict
from .config import number, source_key, split_event_definitions, validate_configuration
from .model import SCHEMA, attrs, decode, timestamp


class Edge:
    def __init__(self, config, bindings):
        self.assets = {source_key(a) for a in bindings['bindings']}
        self.events = {e["type"]: e for e in config["events"]}

    def transform(self, raw):
        spec = self.events[raw["event_type"]]
        identity = raw['source'], raw['source_id']
        if identity not in self.assets or raw['source'] != spec['source'] or raw['source_id'] not in spec['assets']:
            raise ValueError('Event type is not applicable to this asset')
        observed = timestamp(raw["time"])
        if observed > time.time() + 30:
            raise ValueError("Future observation")
        event_id = raw['event_id']
        if not isinstance(event_id, str) or not 1 <= len(event_id) <= 256:
            raise ValueError('Expected a source occurrence ID')
        if set(raw["values"]) - {m["name"] for m in spec["metrics"]}:
            raise ValueError("Unknown source metric")
        alarms = raw.get('alarms', {})
        if not isinstance(alarms, dict) or set(alarms) - {m['name'] for m in spec['metrics'] if m.get('alarm') == 'source'}:
            raise ValueError('Only declared source-owned alarms may be supplied')
        if any(value is not None and type(value) is not bool for value in alarms.values()):
            raise ValueError('Source alarms must be true, false or null')
        metrics = []
        for m in spec["metrics"]:
            value = raw["values"].get(m["name"])
            if value is not None and not number(value):
                raise ValueError("Measurement must be finite or absent")
            p = {"timeUnixNano": str(round(observed * 1e9)),
                 "attributes": attrs(quality="missing" if value is None else "good")}
            p.update({"flags": 1} if value is None else {"asDouble": float(value)})
            metrics.append({"name": m["name"], "unit": m["unit"], "gauge": {"dataPoints": [p]}})
            if m.get('condition') or m.get('alarm') == 'source':
                if m.get('alarm') == 'source':
                    decision = alarms.get(m['name'])
                    provenance = attrs(kind='condition', metric=m['name'], rule='source', owner=raw['source'])
                else:
                    limit = m['condition']['above']
                    decision = None if value is None else value > limit
                    provenance = attrs(kind='condition', metric=m['name'], rule='above', threshold=limit)
                c = {"timeUnixNano": p["timeUnixNano"],
                     "attributes": provenance}
                c.update({'flags': 1} if decision is None else {'asDouble': float(decision)})
                metrics.append({"name": m["name"] + ".alarm", "unit": "1", "gauge": {"dataPoints": [c]}})
        event = {"specversion": "1.0", "id": event_id, "source": 'urn:example:source:' + raw['source'],
                 "subject": raw['source_id'], "type": spec["type"], "time": raw["time"],
                 "dataschema": SCHEMA, "datacontenttype": "application/json",
                 "data": {"resourceMetrics": [{"resource": {"attributes": attrs(**{'source.name': raw['source'], 'source.id': raw['source_id']})},
                          "scopeMetrics": [{"scope": {"name": "reference-edge"}, "metrics": metrics}]}]}}
        decode(event)
        return event


class EventBus:
    """Per-subscriber bounded queues; snapshots and subscription registration are atomic.

    In-memory only. Slow consumers get an explicit reset plus latest state, not a
    false claim that every transition survived. Source history is a separate API.
    """
    def __init__(self, capacity=128):
        self.lock = threading.Lock()
        self.capacity = capacity
        self.recorder = None
        self.subscribers = set()
        self.current = {}
        self.seen = OrderedDict()
        self.sequence = 0
        self.context = {}
        self.epoch = str(uuid.uuid4())

    def _snapshot(self, reason="initial"):
        return {"kind": "snapshot", "reason": reason, "epoch": self.epoch,
                "sequence": self.sequence, "events": list(self.current.values()), **self.context}

    def subscribe(self):
        q = queue.Queue(self.capacity)
        with self.lock:
            if len(self.subscribers) >= 12:
                raise ValueError("Subscriber limit reached")
            q.put(self._snapshot())
            self.subscribers.add(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.subscribers.discard(q)

    def publish(self, event):
        key = event["source"], event["id"]
        with self.lock:
            if key in self.seen:
                if self.seen[key] != event:
                    raise ValueError("Conflicting retry payload")
                return False
            self.seen[key] = event
            if len(self.seen) > 4096:
                self.seen.popitem(last=False)
            self.sequence += 1
            slot = event['source'], event['subject'], event['type']
            previous = self.current.get(slot)
            if previous is None or timestamp(event["time"]) >= timestamp(previous["time"]):
                self.current[slot] = event
            item = {"kind": "event", "epoch": self.epoch, "sequence": self.sequence, "event": event}
            if self.recorder:
                self.recorder.offer(item)
            for q in self.subscribers:
                try:
                    q.put_nowait(item)
                except queue.Full:
                    while not q.empty():
                        try:q.get_nowait()
                        except queue.Empty:break
                    q.put_nowait(self._snapshot("slow_consumer_gap"))
            return True

    def start_recording(self):
        """Capture the initial state and attach the recorder at one stream boundary."""
        with self.lock:
            return self.recorder.start(self._snapshot('recording_start'))


    def configure(self, context, clear=False, reason='configuration_changed'):
        with self.lock:
            self.context = context
            if clear:
                self.current.clear()
            self.sequence += 1
            packet = self._snapshot(reason)
            if self.recorder:
                self.recorder.offer(packet)
            for q in self.subscribers:
                while not q.empty():
                    try: q.get_nowait()
                    except queue.Empty: break
                q.put_nowait(packet)


class StaleConfiguration(ValueError):
    pass


class Service:
    def __init__(self, config, bindings, bus=None, view=None, recordings=None):
        self.lock = threading.RLock()
        self.bus = bus if bus is not None else EventBus()
        if recordings is not None:
            from .recording import FileRecorder
            self.bus.recorder = FileRecorder(recordings)
        self.recorder = self.bus.recorder
        self.source = None
        self.bindings = None
        self.view = None
        self.config_revision = 0
        self.error = None
        self.configure(config, bindings, view)

    def configure(self, config, bindings, view):
        validate_configuration(config, bindings, view)
        with self.lock:
            if config == self.source and bindings == self.bindings and view == self.view:
                self.report_error(None)
                return
            changed_source = config != self.source or bindings != self.bindings
            self.edge = Edge(config, bindings)
            self.bindings = bindings
            self.source, self.view = config, view
            self.config_revision += 1
            self.error = None
            self.catalog = {'profile': SCHEMA,
                            'bindings': [{'source': a['source'], 'id': a['id'], 'label': a['label'],
                                          'metrics': [m['name'] for e in config['events']
                                                      if a['source'] == e['source'] and a['id'] in e['assets']
                                                      for m in e['metrics']]}
                                         for a in bindings['bindings']],
                            'metrics': [{'name': m['name'], 'unit': m['unit'], 'event_type': e['type']}
                                        for e in config['events'] for m in e['metrics']]}
            self._broadcast(changed_source)

    def _broadcast(self, clear=False, reason='configuration_changed'):
        self.bus.configure({'catalog': self.catalog, 'view': self.view,
                            'config_revision': self.config_revision, 'config_error': self.error}, clear, reason)

    def report_error(self, error):
        with self.lock:
            if error != self.error:
                self.error = error
                self._broadcast(reason='configuration_rejected' if error else 'configuration_recovered')

    def start_recording(self):
        if not self.recorder:
            raise ValueError('Recording is not configured')
        return self.bus.start_recording()

    def producer_configuration(self):
        with self.lock:
            definitions, _ = split_event_definitions(self.source)
            return {'revision': self.config_revision, 'definitions': definitions}

    def ingest(self, raw):
        with self.lock:
            revision = raw.get('config_revision')
            if (self.view is not None or revision is not None) and revision != self.config_revision:
                raise StaleConfiguration('Refresh producer configuration')
            return self.bus.publish(self.edge.transform(raw))
