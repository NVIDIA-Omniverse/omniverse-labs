# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 3: ask a VLM to label the units the rules left unlabeled.

Reads a *.labels.json from enumerate_prims.py and updates it in place:
  labels        gains {label, source: "vlm", confidence, reason} for confident answers
  needs_review  lists units below --min-confidence or answered "unknown"

One request per instance group (identical assets / name stems), so repeated
objects are labeled once. Answers are cached in data/semantics/vlm-cache.json:
reruns only call the API for new or changed units. If render_crops.py has
written <crops>/<unit id>.png, the image is sent with the text context.

Any OpenAI-compatible chat-completions endpoint that accepts images works.
Configure it with environment variables (nothing is hardcoded):
  VLM_API_URL     full chat-completions URL, e.g. https://<host>/v1/chat/completions
  VLM_MODEL       model id served by that endpoint
  VLM_API_KEY     bearer token; read from the environment, never written to disk
  VLM_EXTRA_BODY  optional JSON merged into each request, e.g. to turn off a
                  reasoning model's thinking: {"chat_template_kwargs": {"enable_thinking": false}}
"""
import argparse
import base64
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import SEMANTICS_DIR, group_units, load_json, load_taxonomy, unit_id

CACHE_FILE = SEMANTICS_DIR/'vlm-cache.json'
SYSTEM = ('You label objects in a 3D USD scene of an indoor/outdoor technology campus for '
          'computer-vision datasets. Choose exactly one class from the list, or "unknown" when '
          'the evidence is insufficient. Prim names are artist names: they are strong hints but '
          'may be generic (Mesh0, Cube_12). Reply with JSON only: '
          '{"label": <class>, "confidence": <0..1>, "reason": <short phrase>}')


def build_messages(unit, members, classes, image_bytes):
    lines = ['Classes:']+['- %s: %s' % (k, v['description']) for k, v in classes.items()]
    lines += ['', 'Prim path: '+unit['path'], 'Prim name: '+unit['name'], 'Prim type: '+unit['type'],
              'Parent: '+unit['parent'],
              'Bounding box (m, x*y*z): %s' % ' x '.join('%.2f' % v for v in unit['size_m']),
              'Meshes in subtree: %d' % unit['meshes']]
    if unit.get('materials'):
        lines.append('Bound materials: '+', '.join(unit['materials']))
    if unit['assets']:
        lines.append('Referenced asset: '+', '.join(unit['assets']))
    if len(members) > 1:
        lines.append('This object appears %d times; other instances: %s' % (
            len(members), ', '.join(m['name'] for m in members[1:6])))
    content = [{'type': 'text', 'text': '\n'.join(lines)}]
    if image_bytes:
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,'+base64.b64encode(image_bytes).decode()}})
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]


def parse_answer(text, allowed):
    match = re.search(r'\{.*\}', text or '', re.S)
    if not match:
        raise ValueError('no JSON object in reply')
    data = json.loads(match.group(0))
    label = str(data.get('label', '')).strip().lower()
    if label not in allowed:
        raise ValueError('label outside taxonomy: '+label)
    return {'label': label, 'confidence': float(data.get('confidence', 0)),
            'reason': str(data.get('reason', ''))[:200]}


def endpoint():
    """(url, model, extra request fields) from the environment."""
    url, model = os.environ.get('VLM_API_URL'), os.environ.get('VLM_MODEL')
    if not url or not model:
        raise SystemExit('Set VLM_API_URL and VLM_MODEL (see this script\'s docstring).')
    extra = json.loads(os.environ.get('VLM_EXTRA_BODY') or '{}')
    return url, model, extra


def call(messages, url, model, extra, token, retries=6):
    body = dict({'model': model, 'messages': messages, 'temperature': 0, 'max_tokens': 300}, **extra)
    for attempt in range(retries):
        try:
            r = requests.post(url, json=body, timeout=120,
                              headers={'Authorization': 'Bearer '+token, 'Content-Type': 'application/json'})
        except requests.RequestException:
            if attempt == retries-1:
                raise
            time.sleep(2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(60, 5*2**attempt))
            continue
        r.raise_for_status()
        return r.json()['choices'][0]['message'].get('content')
    raise RuntimeError('API still throttled after %d attempts' % retries)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('labels', type=Path, help='*.labels.json from enumerate_prims.py (updated in place)')
    ap.add_argument('--crops', type=Path, default=SEMANTICS_DIR/'crops')
    ap.add_argument('--limit', type=int, help='Label at most N groups (smoke tests)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min-confidence', type=float, default=0.6)
    ap.add_argument('--dry-run', action='store_true', help='Print the first prompt and exit')
    args = ap.parse_args()

    doc = json.loads(args.labels.read_text(encoding='utf-8'))
    classes = load_taxonomy()['classes']
    allowed = set(classes) | {'unknown'}
    groups = group_units(doc['units'])
    keys = sorted(groups)[:args.limit] if args.limit else sorted(groups)

    url, model, extra = endpoint()
    jobs = []
    for key in keys:
        rep = groups[key][0]
        crop = args.crops/(unit_id(rep['path'])+'.png')
        image = crop.read_bytes() if crop.is_file() else None
        messages = build_messages(rep, groups[key], classes, image)
        digest = hashlib.sha256(json.dumps([model, messages], sort_keys=True).encode()).hexdigest()
        jobs.append((key, messages, digest, bool(image)))
    if args.dry_run:
        print(json.dumps(jobs[0][1], indent=2)[:4000] if jobs else 'nothing to label')
        print('%d groups, %d with images' % (len(jobs), sum(j[3] for j in jobs)))
        return

    token = os.environ.get('VLM_API_KEY')
    if not token:
        raise SystemExit('Set VLM_API_KEY to the bearer token for VLM_API_URL.')
    cache = load_json(CACHE_FILE, {})
    lock = threading.Lock()
    todo = [j for j in jobs if j[2] not in cache]
    print('%d groups: %d cached, %d to request' % (len(jobs), len(jobs)-len(todo), len(todo)))

    def work(job):
        key, messages, digest, _ = job
        last = None
        for _ in range(2):                          # one retry for malformed replies
            text = call(messages, url, model, extra, token)
            try:
                return digest, parse_answer(text, allowed)
            except (ValueError, json.JSONDecodeError) as error:
                last = '%s: %r' % (error, (text or '')[:120])
        return digest, {'label': 'unknown', 'confidence': 0.0, 'reason': 'unparseable: '+last}

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(work, j) for j in todo]):
            digest, answer = future.result()
            with lock:
                cache[digest] = answer
                done += 1
                if done % 20 == 0 or done == len(todo):
                    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    CACHE_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding='utf-8')
                    print('  %d/%d' % (done, len(todo)))

    review = []
    for key, _, digest, had_image in jobs:
        answer = cache[digest]
        for unit in groups[key]:
            entry = dict(answer, source='vlm', group=key, image=had_image)
            if answer['label'] != 'unknown' and answer['confidence'] >= args.min_confidence:
                doc['labels'][unit['path']] = entry
            else:
                review.append(dict(entry, path=unit['path']))
    doc['needs_review'] = sorted(review, key=lambda r: r['path'])
    args.labels.write_text(json.dumps(doc, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    vlm = sum(1 for v in doc['labels'].values() if v['source'] == 'vlm')
    print('vlm-labeled prims: %d | needs review: %d -> %s' % (vlm, len(review), args.labels))


if __name__ == '__main__':
    main()
