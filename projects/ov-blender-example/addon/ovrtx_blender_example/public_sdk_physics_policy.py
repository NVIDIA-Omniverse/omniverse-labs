# SPDX-License-Identifier: Apache-2.0
"""Measured candidates are local qualification inputs, never shipped admission."""
import re


def admits(row, identity, device, family, *, local=False):
    if not isinstance(row.get('devices'), list) or not isinstance(row.get('families'), list):
        return False
    if row.get('identity') != identity or device not in row.get('devices', ()) or family not in row.get('families', ()):
        return False
    if not isinstance(identity.get('runtime_lock'), dict) or not identity['runtime_lock']:
        return False
    if not isinstance(identity.get('adapter_components'), dict) or not identity['adapter_components']:
        return False
    status = row.get('status')
    if status != 'production' and not (local and status == 'candidate'):
        return False
    cells = row.get('evidence')
    if not isinstance(cells, dict):
        return False
    evidence = cells.get(f'{family}/{device}', {})
    if not isinstance(evidence, dict):
        return False
    fields = ['fixture_sha256', 'results_sha256']
    if status == 'production':
        fields.append('production_results_sha256')
        if evidence.get('production_measurement_only') is not False:
            return False
    return all(isinstance(evidence.get(key), str) and re.fullmatch('[0-9a-f]{64}', evidence[key])
               for key in fields)
