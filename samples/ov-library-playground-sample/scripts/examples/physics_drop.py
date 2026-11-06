# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare two actual ovphysx gravity settings in a small, reproducible scene."""
import json
from _common import scene, output_folder, run_example


def main():
    output = output_folder()
    runtime, stage = scene()
    results = {}
    try:
        for gravity in (9.81, 1.62):
            runtime.reset_physics()
            stage.GetPrimAtPath('/World/Physics').GetAttribute('physics:gravityMagnitude').Set(gravity)
            runtime.start_physics(['/World/Carton'])
            # Native PhysX steps; poses below come from native read sessions.
            runtime.physx.step_n_sync(60, 1/120)
            position = runtime.read_bodies('position')['/World/Carton']
            results[str(gravity)] = float(position[1])
        assert results['1.62'] > results['9.81']+.5, results
        (output/'gravity.json').write_text(json.dumps(results, indent=2))
        print('Carton height after 0.5 simulated seconds:', results)
    finally:
        runtime.close()


if __name__ == '__main__':
    run_example(main)
