# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native ovnewton experiment, imported only by the isolated worker.

The host supplies a target trajectory, never body poses. ovnewton imports USD
through ovstage, consumes drive targets and publishes Newton solver results.
"""
import math
import time
import numpy as np
import ovstage
from ovstage import population, PopulationDomain, make_dltensor
import ovnewton
import newton
import warp as wp
from newton_model import GANTRY_USDA


class GantrySimulation:
    def __init__(self):
        wp.set_device('cpu')
        # Register Newton and PhysX USD schemas before population so joint state
        # and drive channels are available to the OVNewton stage adapter.
        ovnewton.register_usd_schemas()

    def run(self, mode='gentle', warm=False):
        if mode not in ('gentle', 'fast'):
            raise ValueError('Unknown motion profile')
        started = time.perf_counter()
        with ovstage.Stage('lab-gantry') as stage, ovstage.PathDictionary(stage) as pd:
            population.open_usd_from_string(stage, GANTRY_USDA, ordinal=1, domains=PopulationDomain.ALL)
            stage.advance_write_floor(1).wait()
            binding = ovnewton.attach_ovstage(stage)
            try:
                model = binding.model
                solver = newton.solvers.SolverXPBD(model, iterations=8)
                state, next_state = model.state(), model.state()
                control, contacts = model.control(), model.contacts()
                paths = pd.create_path_list_from_strings(['/Gantry/Travel'])
                try:
                    with stage.query_from_path_list(paths) as drive:
                        drive.wait()
                        frames, ordinal = [], 1
                        duration = 4.5 if mode == 'gentle' else 1.25
                        frame_count = 2 if warm else 241
                        for frame in range(frame_count):
                            t = frame / 30.
                            u = min(1., t / duration)
                            # Quintic position command: zero velocity/acceleration at both ends.
                            target = -1.6 + 3.2 * (10*u**3-15*u**4+6*u**5)
                            ordinal += 1
                            value = np.asarray([target], dtype=np.float32)
                            stage.write_attribute(drive, 'drive:linear:physics:targetPosition', ordinal,
                                                  make_dltensor(value), is_array=False).wait()
                            stage.advance_write_floor(ordinal).wait()
                            binding.update_from_ovstage(state, control=control, ordinal=ordinal)
                            for _ in range(4):
                                state.clear_forces()
                                # Suspended articulated rig: no environmental contacts.
                                solver.step(state, next_state, control, None, 1/120.)
                                state, next_state = next_state, state
                            ordinal += 1
                            binding.update_to_ovstage(state, ordinal=ordinal)
                            stage.advance_write_floor(ordinal).wait()
                            poses = state.body_q.numpy()
                            if not np.isfinite(poses).all():
                                raise RuntimeError('Newton returned a non-finite body pose')
                            frames.append(poses.tolist())
                finally:
                    pd.destroy_path_list(paths)
                labels = list(model.body_label)
                load = labels.index('/Gantry/Load')
                trolley = labels.index('/Gantry/Trolley')
                angles = [math.degrees(math.atan2(f[trolley][0]-f[load][0],
                                                   f[trolley][1]-f[load][1])) for f in frames]
                return {'frames': frames, 'labels': labels, 'angles': angles, 'fps': 30,
                        'mode': mode, 'travel_seconds': duration,
                        'peak_sway': max(abs(a) for a in angles),
                        'compute_seconds': time.perf_counter()-started,
                        'source': 'ovnewton / ovstage / Newton XPBD', 'sealed_ordinal': ordinal}
            finally:
                # StageBinding owns path lists through weakref.finalize.
                del binding
