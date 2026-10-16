# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operational-side simulation and aggregation; no USD or Kit.

All inputs for a batch use the same observation instant. Aggregates consume the
actual member values generated in that batch; absent inputs remain unknown.
"""
import math


def measurements(config, elapsed):
    values = {}
    indices = {a['id']: i for i, a in enumerate(config['assets'])}
    for event in config['events']:
        for native in event.get('assets', indices):
            for metric in event['metrics']:
                if 'simulation' in metric:
                    wave = metric['simulation']
                    value = wave['center'] + wave['amplitude'] * math.sin(
                        elapsed * 2 * math.pi / wave['period'] + indices[native] * 1.7)
                else:
                    spec = metric['aggregate']
                    inputs = [values.get((member, spec['metric'])) for member in spec['members']]
                    value = None if any(v is None for v in inputs) else sum(inputs)
                values[native, metric['name']] = value
    return values
