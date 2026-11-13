# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Confirm every taxonomy Q-code resolves on wikidata.org to its recorded English label.

SimReady SL.QCODE.001 validates only the Q-code format; this check covers the convention
that each code names a real entity. Run it after editing config/semantic-taxonomy.json.
Needs network access to wikidata.org; sends only the Q-codes.
"""
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _usd import load_taxonomy

API = 'https://www.wikidata.org/w/api.php'
HEADERS = {'User-Agent': 'ov-library-playground-semantic-labeler/1.0'}


def fetch(ids):
    params = {'action': 'wbgetentities', 'ids': '|'.join(ids), 'props': 'labels|descriptions',
              'languages': 'en', 'format': 'json'}
    for attempt in range(6):
        response = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if response.ok and response.text.startswith('{'):
            return response.json()['entities']
        time.sleep(5*(attempt+1))                 # wikidata.org throttles bursts
    raise SystemExit('wikidata.org did not answer (HTTP %d)' % response.status_code)


def main():
    classes = load_taxonomy()['classes']
    ids = sorted({c['qcode'] for c in classes.values()})
    entities = {}
    for start in range(0, len(ids), 50):          # API limit: 50 ids per request
        entities.update(fetch(ids[start:start+50]))
    errors = 0
    for name, entry in classes.items():
        entity = entities.get(entry['qcode'], {})
        label = entity.get('labels', {}).get('en', {}).get('value')
        ok = 'missing' not in entity and label == entry['wikidata_label']
        errors += not ok
        print('%s %-14s %-10s %-22s %s' % ('ok ' if ok else 'BAD', name, entry['qcode'], label,
                                          entity.get('descriptions', {}).get('en', {}).get('value', '')[:60]))
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
