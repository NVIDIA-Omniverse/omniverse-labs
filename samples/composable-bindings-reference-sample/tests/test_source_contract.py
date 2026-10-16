# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused regression coverage for source identity, adapters and live snapshots."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import threading
import unittest

from composable_bindings.reference.composition import compose
from composable_bindings.reference.config import load_configuration, read, validate_configuration
from composable_bindings.reference.json_source import adapt, validate_mapping
from composable_bindings.reference.model import ViewModel, decode, presentation_rows
from composable_bindings.reference.playback import PlaybackController, RecordingIndex
from composable_bindings.reference.scene_bindings import match_bindings
from composable_bindings.reference.service import Edge, EventBus, Service, StaleConfiguration
from composable_bindings.reference.simulation import compose_simulation
from composable_bindings.reference.transport import Server

ROOT = Path(__file__).resolve().parents[1]
GATEWAY = ROOT / 'examples/line_gateway'


class SourceContractTests(unittest.TestCase):
    def setUp(self):
        self.selection = {'base': str(ROOT / 'examples/default'), 'additions': [str(GATEWAY)]}
        self.config, self.bindings, self.view = compose(self.selection)
        self.mapping = validate_mapping(read(GATEWAY / 'adapter.json'))
        self.record = json.loads((GATEWAY / 'readings.jsonl').read_text().splitlines()[0])

    def test_external_source_without_simulation_and_source_alarm(self):
        raw = adapt(self.record, self.mapping)
        event = Edge(self.config, self.bindings).transform(raw)
        sample, = decode(event)
        self.assertEqual(sample.identity, ('line_gateway', 'press-7'))
        self.assertEqual(event['subject'], self.record['equipment'])
        self.assertEqual(event['id'], self.record['occurrence'])
        self.assertEqual(event['time'], self.record['observed_at'])
        self.assertTrue(sample.alarm)  # Source says alarm even though the reading is only 64.5.
        self.assertNotIn('simulation', json.dumps(self.config))
        sim = compose_simulation(self.selection, self.config)
        self.assertEqual(sim['source'], 'factory_simulator')

    def test_source_alarm_clear_missing_and_alarm_without_value(self):
        edge = Edge(self.config, self.bindings)
        model = ViewModel()
        spec = self.view['displays'][-1]
        for index, (alarm, value, expected) in enumerate([(True, None, 'alarm'), (False, 64, 'normal'), (None, 64, 'unknown')]):
            record = copy.deepcopy(self.record)
            record['occurrence'] = str(index)
            record['conditions']['overheat'] = alarm
            record['readings']['temperature_c'] = value
            model.apply(edge.transform(adapt(record, self.mapping)))
            self.assertEqual(model.display(('line_gateway', 'press-7'), spec)['state'], expected)

    def test_alarm_ownership_and_unknown_metrics_are_rejected(self):
        bad = copy.deepcopy(self.config)
        bad['events'][-1]['metrics'][0]['condition'] = {'above': 20}
        with self.assertRaises(ValueError):
            validate_configuration(bad, self.bindings, self.view)
        edge = Edge(self.config, self.bindings)
        for field, key, value in [('values', 'undeclared', 1), ('alarms', 'undeclared', True), ('alarms', 'gateway.temperature', 1)]:
            raw = adapt(self.record, self.mapping)
            raw[field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                edge.transform(raw)

    def test_source_scope_is_enforced(self):
        raw = adapt(self.record, self.mapping)
        raw['source'] = 'factory_simulator'
        with self.assertRaises(ValueError):
            Edge(self.config, self.bindings).transform(raw)

    def test_edge_threshold_stays_strict_and_declarations_reject_simulation(self):
        edge = Edge(self.config, self.bindings)
        for value, expected in [(100, False), (100.1, True), (None, None)]:
            raw = {'source': 'factory_simulator', 'source_id': 'mock-controller-1',
                   'event_type': 'org.example.measurements.v1', 'event_id': str(value),
                   'time': self.record['observed_at'], 'values': {'equipment.temperature': value}}
            self.assertIs(decode(edge.transform(raw))[0].alarm, expected)
        bad = copy.deepcopy(self.config)
        bad['events'][0]['metrics'][0]['simulation'] = {'center': 0}
        with self.assertRaises(ValueError):
            validate_configuration(bad, self.bindings, self.view)

    def test_same_id_and_occurrence_in_two_systems_do_not_collide(self):
        config, bindings, view = load_configuration(GATEWAY)
        other = copy.deepcopy(config['events'][0])
        other.update(source='other_gateway', type='org.example.other.v1')
        other['metrics'][0]['name'] = 'other.temperature'
        config['assets'].append({'source': 'other_gateway', 'id': 'press-7'})
        config['events'].append(other)
        bindings['bindings'].append({'source': 'other_gateway', 'id': 'press-7', 'label': 'Other press'})
        view['displays'].append(dict(view['displays'][0], id='other', metric='other.temperature'))
        service = Service(config, bindings, view=view)
        first = adapt(self.record, self.mapping)
        second = dict(first, source='other_gateway', event_type=other['type'],
                      values={'other.temperature': 12}, alarms={'other.temperature': False})
        for raw in (first, second):
            self.assertTrue(service.ingest(dict(raw, config_revision=service.config_revision)))
        packet = service.bus.subscribe().get_nowait()
        model = ViewModel()
        for event in packet['events']:
            model.apply(event)
        rows = presentation_rows(model, packet['catalog'], packet['view'])
        self.assertEqual([row['displays'][0]['sample'].value for row in rows], [64.5, 12])
        self.assertFalse(service.ingest(dict(first, config_revision=service.config_revision)))

    def test_late_join_gap_reset_and_configuration_clear(self):
        bus = EventBus(capacity=1)
        service = Service(self.config, self.bindings, bus=bus, view=self.view)
        raw = dict(adapt(self.record, self.mapping), config_revision=service.config_revision)
        service.ingest(raw)
        reader = bus.subscribe()
        initial = reader.get_nowait()
        self.assertEqual(initial['catalog'], service.catalog)
        self.assertEqual(initial['view'], self.view)
        self.assertEqual(len(initial['events']), 1)
        service.ingest(dict(raw, event_id='next'))
        service.ingest(dict(raw, event_id='overflow'))
        reset = reader.get_nowait()
        self.assertEqual(reset['reason'], 'slow_consumer_gap')
        self.assertEqual(reset['events'][0]['id'], 'overflow')
        self.assertGreater(reset['sequence'], initial['sequence'])
        changed = copy.deepcopy(self.config)
        changed['events'][-1]['metrics'][0]['unit'] = 'K'
        service.configure(changed, self.bindings, self.view)
        self.assertEqual(reader.get_nowait()['events'], [])
        with self.assertRaises(StaleConfiguration):
            service.ingest(raw)
        bus.unsubscribe(reader)

    def test_unmatched_unbound_and_multiple_scene_placements(self):
        def info(source, identity):
            return {'sourceIds': {source: {'id': identity}}}
        authored = [('/known', info('line_gateway', 'press-7')),
                    ('/instance', info('line_gateway', 'press-7')),
                    ('/wrong_source', info('different_system', 'press-7')),
                    ('/unknown', info('line_gateway', 'missing')),
                    ('/unbound', {})]
        matched, unmatched = match_bindings(authored, self.bindings['bindings'])
        self.assertEqual(matched[('line_gateway', 'press-7')], ['/known', '/instance'])
        self.assertEqual({entry['path'] for entry in unmatched}, {'/unknown', '/wrong_source'})
        self.assertTrue(all(entry['state'] == 'unmatched' for entry in unmatched))
        catalog = self.bindings['bindings'] + [{'source': 'line_gateway', 'id': 'missing'}]
        self.assertEqual(len(match_bindings(authored, catalog)[1]), 1)

    def test_capture_replay_preserves_source_identity_and_alarm(self):
        palette = {'normal': [0, 1, 0], 'alarm': [1, 0, 0], 'unknown': [.3, .3, .3]}
        self.view.setdefault('scene', []).append({'source': 'line_gateway', 'id': 'press-7',
            'display_color': {'metric': 'gateway.temperature', 'colors': palette}})
        with tempfile.TemporaryDirectory() as directory:
            service = Service(self.config, self.bindings, view=self.view, recordings=directory)
            raw = dict(adapt(self.record, self.mapping), config_revision=service.config_revision)
            service.ingest(raw)
            recording = service.start_recording()
            service.ingest(dict(raw, event_id='cleared', alarms={'gateway.temperature': False}))
            service.recorder.stop()
            with service.recorder.path(recording['id']).open() as stream:
                index = RecordingIndex(iter(stream))
            first = next(row for row in index.snapshot(0)['rows'] if row['binding']['source'] == 'line_gateway')
            last = next(row for row in index.snapshot(index.duration)['rows'] if row['binding']['source'] == 'line_gateway')
            self.assertEqual(first['displays'][0]['state'], 'alarm')
            self.assertEqual(last['displays'][0]['state'], 'normal')
            self.assertEqual(first['display_color']['rgb'], palette['alarm'])
            self.assertEqual(last['display_color']['rgb'], palette['normal'])
            self.assertIn('not source history', index.snapshot(0)['status'])

    def test_recording_panel_error_clears_after_credentials_recover(self):
        with tempfile.TemporaryDirectory() as directory:
            service = Service(self.config, self.bindings, view=self.view, recordings=directory)
            server = Server(0, service, 'reader', 'ingest')
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                credentials = Path(directory) / 'client.json'
                url = 'http://127.0.0.1:' + str(server.server_address[1])
                credentials.write_text(json.dumps({'url': url, 'token': 'stale'}))
                controller = PlaybackController(None, credentials)
                controller._poll()
                self.assertEqual(controller.state['error'], 'Read credential required')
                credentials.write_text(json.dumps({'url': url, 'token': 'reader'}))
                controller._poll()
                self.assertIsNone(controller.state['error'])
                controller._set(error='Record and stop a capture first')
                controller._poll()
                self.assertEqual(controller.state['error'], 'Record and stop a capture first')
            finally:
                server.shutdown()
                server.server_close()

    def test_color_policy_uses_shared_state_and_validates_source_metric(self):
        rows = presentation_rows(ViewModel(), {'bindings': self.bindings['bindings']}, self.view)
        self.assertEqual(rows[0]['display_color']['state'], 'unknown')
        for change in ('invalid_rgb', 'wrong_source_metric'):
            view = copy.deepcopy(self.view)
            color = view['scene'][1]['display_color']
            if change == 'invalid_rgb':
                color['colors']['low'] = [2, 0, 0]
            else:
                color['metric'] = 'gateway.temperature'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_configuration(self.config, self.bindings, view)

    def test_color_gradient_interpolates_clamps_and_preserves_alarm_decision(self):
        model = ViewModel()
        edge = Edge(self.config, self.bindings)
        identity = ('factory_simulator', 'mock-controller-1')
        spec = {'mode': 'gradient', 'metric': 'equipment.pressure', 'range': [160, 200],
                'colors': {'low': [0, 0, 1], 'high': [1, 0, 0], 'unknown': [.3, .3, .3]}}
        for index, (value, expected) in enumerate([(140, [0, 0, 1]), (160, [0, 0, 1]),
                (180, [.5, 0, .5]), (200, [1, 0, 0]), (220, [1, 0, 0]), (None, [.3, .3, .3])]):
            raw = dict(source=identity[0], source_id=identity[1], event_type='org.example.measurements.v1',
                       event_id=str(index), time=self.record['observed_at'], values={'equipment.pressure': value})
            model.apply(edge.transform(raw))
            color = model.color(identity, spec)
            self.assertEqual(color['rgb'], expected)
            self.assertEqual(color['state'], 'unknown' if value is None else 'alarm' if value > 195 else 'normal')
        bad = copy.deepcopy(self.view)
        bad['scene'][1]['display_color']['range'] = [200, 160]
        with self.assertRaises(ValueError):
            validate_configuration(self.config, self.bindings, bad)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'reads producer PID from /proc')
    def test_demo_exits_cleanly_when_ctrl_c_stops_producer_first(self):
        # Windows delivers Ctrl-C to the producer, which can exit before the demo's handler runs.
        with tempfile.TemporaryDirectory() as directory:
            demo = subprocess.Popen([sys.executable, str(ROOT / 'reference.py'), 'demo', '--no-console',
                                     '--port', '0', '--state', directory],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                children = Path(f'/proc/{demo.pid}/task/{demo.pid}/children')
                deadline = time.monotonic() + 15
                while not children.read_text().split() and time.monotonic() < deadline:
                    time.sleep(.05)
                producer = int(children.read_text().split()[0])
                time.sleep(1)
                os.kill(producer, signal.SIGINT)
                time.sleep(.3)
                demo.send_signal(signal.SIGINT)
                _, errors = demo.communicate(timeout=15)
            finally:
                if demo.poll() is None:
                    demo.kill()
                    demo.wait()
            self.assertEqual(demo.returncode, 0, errors)
            self.assertNotIn('Producer exited', errors)


if __name__ == '__main__':
    unittest.main()
