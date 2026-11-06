# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run with <runtime>/newton-env/Scripts/python.exe; CPU, no gallery required."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/nvidia/ov_library_lab'))
from newton_engine import GantrySimulation

if __name__ == '__main__':
    simulation = GantrySimulation()
    for profile in ('gentle','fast'):
        result = simulation.run(profile)
        print(f"{profile}: {result['peak_sway']:.1f} degrees peak sway; "
              f"{len(result['frames'])} native pose samples; sealed ordinal {result['sealed_ordinal']}")

# Inspect newton_model.py for the USD rigid bodies, prismatic drive and pivot.
# newton_engine.py owns the solver loop and calls the real bridge:
# population.open_usd_from_string(stage, GANTRY_USDA, domains=ALL)
# binding = ovnewton.attach_ovstage(stage)
# binding.update_from_ovstage(state, control=control, ordinal=ordinal)
# solver.step(state, next_state, control, None, 1/120.)
# binding.update_to_ovstage(state, ordinal=ordinal)
