# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP adapters: protected ingestion and one server-pushed SSE stream per reader."""
import hmac
import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.parse import urlsplit
from .service import StaleConfiguration


def request(credentials, path, body=None):
    endpoint = credentials['url']
    address = urlsplit(endpoint)
    if address.scheme != 'http' or address.hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('Reference transport is loopback only; use an authenticated TLS adapter remotely')
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    return urlopen(Request(endpoint + path, data=data, headers={
        'Authorization': 'Bearer ' + credentials['token'], 'Content-Type': 'application/json'}), timeout=5)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    def __init__(self, port, service, read_token, ingest_token):
        self.service, self.read_token, self.ingest_token = service, read_token, ingest_token
        self.slots = threading.BoundedSemaphore(16)
        self.stopping = threading.Event()
        super().__init__(('127.0.0.1', port), Handler)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()

    def server_close(self):
        self.stopping.set()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args):
        pass

    def allowed(self, token):
        supplied = self.headers.get('Authorization', '')
        return hmac.compare_digest(supplied.encode(), ('Bearer ' + token).encode())

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path in ('/recordings/start','/recordings/stop'):
            if not self.allowed(self.server.read_token):
                return self.reply(403, {'error':'Reader credential required'})
            try:
                if not self.server.service.recorder:
                    raise ValueError('Recording is not configured')
                result = (self.server.service.start_recording() if self.path.endswith('/start')
                          else self.server.service.recorder.stop())
                return self.reply(200,result)
            except ValueError as error:
                return self.reply(409, {'error':str(error)})
        if not self.allowed(self.server.ingest_token):
            return self.reply(403, {'error': 'Ingest credential required'})
        if self.path != '/ingest':
            return self.reply(404, {'error': 'Unknown route'})
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536:
                raise ValueError('Body limit')
            raw = json.loads(self.rfile.read(size))
            accepted = self.server.service.ingest(raw)
        except StaleConfiguration:
            return self.reply(409, {"error": "Refresh producer configuration"})
        except (ValueError, KeyError, TypeError, OverflowError, AttributeError):
            return self.reply(400, {'error': 'Invalid source event'})
        self.reply(202, {'accepted': accepted})

    def do_GET(self):
        if self.path == '/producer-config':
            if not self.allowed(self.server.ingest_token):
                return self.reply(403, {'error': 'Ingest credential required'})
            return self.reply(200, self.server.service.producer_configuration())
        if not self.allowed(self.server.read_token):
            return self.reply(403, {'error': 'Read credential required'})
        if self.path == '/recordings':
            recorder = self.server.service.recorder
            return self.reply(200,recorder.catalog() if recorder else {'recordings':[],'active':None,'error':None})
        if self.path.startswith('/recordings/'):
            try:
                recorder = self.server.service.recorder
                if not recorder:
                    raise ValueError('Recording is not configured')
                path = recorder.path(self.path[len('/recordings/'):])
                with path.open('rb') as stream:
                    self.send_response(200)
                    self.send_header('Content-Type','application/x-ndjson')
                    self.send_header('Content-Length',str(path.stat().st_size))
                    self.end_headers()
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:break
                        self.wfile.write(chunk)
                return
            except ValueError as error:
                return self.reply(404, {'error':str(error)})
            except OSError:
                return
        if self.path == '/catalog':
            return self.reply(200, self.server.service.catalog)
        if self.path != '/events':
            return self.reply(404, {'error': 'Unknown route'})
        try:
            subscription = self.server.service.bus.subscribe()
        except ValueError:
            return self.reply(503, {'error': 'Subscriber limit'})
        try:
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            while not self.server.stopping.is_set():
                try:
                    packet = subscription.get(timeout=.25)
                    data = ('data: ' + json.dumps(packet, separators=(',', ':')) + '\n\n').encode()
                except queue.Empty:
                    data = b': heartbeat\n\n'
                self.wfile.write(data)
                self.wfile.flush()
        except (OSError, TimeoutError):
            pass
        finally:
            self.server.service.bus.unsubscribe(subscription)
