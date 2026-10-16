# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common event codec and view model used by shell and Kit; no rendering imports."""
from dataclasses import dataclass
from datetime import datetime, timezone
from .config import number, source_key, validate_identity

SCHEMA = "urn:example:composable:metrics:2"  # local OTLP gauge profile, not a standard schema


def attrs(**values):
    return [{"key": k, "value": {"stringValue": str(v)}} for k, v in values.items()]


def attributes(items):
    return {a["key"]: a["value"]["stringValue"] for a in items}


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp(value):
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise ValueError("Observation time requires a timezone")
    return d.timestamp()


@dataclass(frozen=True)
class Sample:
    source: str
    source_id: str
    event_id: str
    observed: float
    metric: str
    value: object
    unit: str
    quality: str
    alarm: object
    alarm_expected: bool

    @property
    def identity(self):
        return self.source, self.source_id


def decode(event):
    if event.get("specversion") != "1.0" or event.get("dataschema") != SCHEMA:
        raise ValueError("Unsupported CloudEvents/gauge profile")
    if event.get("datacontenttype") != "application/json":
        raise ValueError("Unsupported content type")
    for key in ("subject", "id", "source", "type"):
        if not isinstance(event.get(key), str) or not event[key]:
            raise ValueError("Missing event identity/type")
    observed = timestamp(event["time"])
    resources = event["data"]["resourceMetrics"]
    if len(resources) != 1:
        raise ValueError("Expected one resource")
    resource = resources[0]
    identity = attributes(resource['resource']['attributes'])
    source = identity['source.name']
    source_id = identity['source.id']
    validate_identity({'source': source, 'id': source_id})
    if source_id != event['subject'] or event['source'] != 'urn:example:source:' + source:
        raise ValueError('Resource source identity and envelope disagree')
    points, conditions = {}, {}
    for scope in resource["scopeMetrics"]:
        for m in scope["metrics"]:
            dp = m["gauge"]["dataPoints"]
            if len(dp) != 1 or abs(int(dp[0]["timeUnixNano"]) / 1e9 - observed) > .001:
                raise ValueError("Expected one point at event time")
            p = dp[0]; a = attributes(p.get("attributes", []))
            value = None if p.get("flags", 0) & 1 else p.get("asDouble")
            if value is not None and not number(value):
                raise ValueError("Expected a finite gauge")
            if a.get("kind") == "condition":
                if value not in (None, 0., 1.) or a["metric"] in conditions:
                    raise ValueError("Invalid/duplicate condition")
                conditions[a["metric"]] = None if value is None else bool(value)
            else:
                if m["name"] in points:
                    raise ValueError("Duplicate metric")
                points[m["name"]] = (value, m["unit"], a.get("quality", "good"))
    return [Sample(source, source_id, event['id'], observed, name, value, unit, quality,
                   conditions.get(name), name in conditions) for name, (value, unit, quality) in points.items()]


class ViewModel:
    def __init__(self):
        self.latest = {}

    def clear(self):
        self.latest.clear()

    def apply(self, event):
        changed = []
        for sample in decode(event):
            key = sample.identity, sample.metric
            previous = self.latest.get(key)
            if previous and (sample.observed < previous.observed or sample.event_id == previous.event_id):
                continue
            self.latest[key] = sample
            changed.append(sample)
        return changed

    def display(self, binding, spec):
        s = self.latest.get((binding, spec["metric"]))
        state = self.state(binding, spec['metric'])
        text = "—" if s is None or s.value is None else f"{s.value:.{spec.get('decimals', 1)}f} {s.unit}"
        fraction = 0.
        if s and s.value is not None and spec["kind"] == "gauge":
            low, high = spec["range"]
            fraction = max(0., min(1., (s.value - low) / (high - low)))
        return {"text": text, "state": state, "fraction": fraction, "sample": s}

    def state(self, binding, metric):
        sample = self.latest.get((binding, metric))
        return ('alarm' if sample and sample.alarm is True else
                'unknown' if sample is None or sample.value is None or
                (sample.alarm_expected and sample.alarm is None) else 'normal')

    def color(self, binding, spec):
        mode = spec.get('mode', 'state')
        palette = spec['colors']
        state = self.state(binding, spec['metric'])
        rgb = palette['unknown']
        if mode == 'gradient':
            sample = self.latest.get((binding, spec['metric']))
            if sample is not None and sample.value is not None:
                low, high = spec['range']
                fraction = max(0., min(1., (sample.value - low) / (high - low)))
                rgb = [a + fraction * (b - a) for a, b in zip(palette['low'], palette['high'])]
        else:
            rgb = palette[state]
        return {'mode': mode, 'state': state, 'rgb': rgb, 'unknown_rgb': palette['unknown']}

    def log_lines(self, changed, view):
        lines = []
        for sample in changed:
            for spec in view["displays"]:
                if spec["metric"] != sample.metric:
                    continue
                d = self.display(sample.identity, spec)
                gauge = ""
                if spec["kind"] == "gauge":
                    n = round(d["fraction"] * 12)
                    gauge = " [" + "#" * n + "-" * (12 - n) + "]"
                lines.append(f"{utc(sample.observed)} {sample.source}/{sample.source_id} {spec['title']}: "
                             f"{d['text']}{gauge} {d['state'].upper()} event={sample.event_id}")
        return lines


def presentation_rows(model, catalog, view):
    colors = {source_key(entry): entry['display_color'] for entry in view.get('scene', []) if 'display_color' in entry}
    rows = []
    for binding in catalog['bindings']:
        identity = source_key(binding)
        row = {'binding': binding, 'displays': [dict(spec=d, **model.display(identity, d))
               for d in view['displays'] if 'metrics' not in binding or d['metric'] in binding['metrics']]}
        if identity in colors:
            row['display_color'] = model.color(identity, colors[identity])
        rows.append(row)
    return rows
