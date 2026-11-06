# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability stories grounded in the available library integrations."""
from .extensions import value

CATALOG_URL = 'https://developer.nvidia.com/omniverse'
ROOMS = {'physics': {'number': '01',
             'name': 'ovphysx',
             'title': 'Explore motion, contact and friction.',
             'subtitle': 'RIGID BODIES / CONTACT / FRICTION',
             'color': (0.62, 0.89, 0.23),
             'center': -11.0,
             'summary': 'Drop cartons, settle a stack and compare sliding friction on a receiving conveyor.',
             'use': 'Explore physical behavior in robotics, digital twins and interactive worlds. Gravity, contacts and '
                    'material friction determine how objects move.',
             'url': 'https://github.com/NVIDIA-Omniverse/PhysX/tree/main/ovphysx',
             'implementation': 'OVPhysX: a native simulation attached to ovstage, stepped at '
                               '120 Hz. Rigid-body poses are read from ovphysx and shown through ovrtx. '
                               'Nudges use native velocity write sessions. Six rigid test loads use '
                               'authored mass, cuboid inertia, contact geometry and friction. The belt '
                               'contact surface is visible. Material values are illustrative, not measured '
                               'customer data; carton deformation is not simulated.'},
 'rendering': {'number': '02',
               'name': 'ovrtx',
               'title': 'Explore rendering and sensor simulation.',
               'subtitle': 'MATERIALS / LIGHTS / CAMERAS',
               'color': (0.25, 0.78, 0.98),
               'center': 0.0,
               'summary': 'Explore robot sensing, automotive finishes and robot materials. '
                          'See how OVRTX turns one scene into images and sensor measurements.',
               'use': 'Visualize designs, explore digital twins and develop perception systems. '
                      'Camera outputs and LiDAR returns help you test what a machine can see.',
               'url': 'https://github.com/NVIDIA-Omniverse/ovrtx',
               'implementation': 'OVRTX renders ovstage into camera render products. Materials, '
                                 'lights and cameras update through ovstage tensor writes. Car Concept by '
                                 'Eric Chadwick / DGG (CC BY 4.0) and ABB IRB 4600 geometry from '
                                 'ROS-Industrial (Apache-2.0) retain source metre dimensions at 1:1 scale. '
                                 'The arms are static material studies. Distance, normals and semantic '
                                 'classes come from native camera tensors; display colors are separate '
                                 'from the raw measurements. The mobile robot uses native LiDAR PointCloud '
                                 'returns and opposing fisheye camera products at the same scene revision, '
                                 'with a scripted route. Sensor profiles and fisheye calibration '
                                 'are illustrative, not calibrated to physical hardware.'},
 'scene': {'number': '03',
           'name': 'ovstage',
           'title': 'Build and update shared worlds.',
           'subtitle': 'INSTANCING / SHARED WORLD STATE',
           'color': (0.74, 0.55, 0.98),
           'center': 11.0,
           'summary': 'Grow one AMR into a fleet, update shared cargo and watch both camera views stay in '
                      'sync.',
           'use': 'A shared scene-data layer connects simulation, rendering and tools. Reusable assets let '
                  'applications construct and update larger worlds.',
           'url': 'https://github.com/NVIDIA-Omniverse/ovstage',
           'implementation': 'OVStage owns the rendered scene. OpenUSD authors reusable assets; '
                             'ovstage population loads them and ordinal-based tensor writes update '
                             'transforms, materials and visibility. Both cameras consume that same ovstage. '
                             'Fleet travel is application-scripted, not autonomous navigation.'},
 'storage': {'number': '05',
             'name': 'ovstorage',
             'title': 'Explore designs. Keep revisions.',
             'subtitle': 'ASSET FILES / REVISIONS / RESTORE',
             'color': (0.98, 0.76, 0.25),
             'center': 0.0,
             'summary': 'Edit rack capacity, save multiple USD revisions and bring an earlier design back '
                        'into the scene.',
             'use': 'Reliable asset access lets teams iterate on a digital twin. ovstorage provides a common '
                    'interface across storage backends.',
             'url': 'https://github.com/NVIDIA-Omniverse/ovstorage',
             'implementation': 'OVStorage Stack and FileBackend perform every revision write, '
                               'listing and read. Saved bytes are read back through ovstorage before the UI '
                               'confirms success. Numbered USD revisions are managed by this application '
                               'under its runtime storage directory. Remote backends are not part of this '
                               'exhibit.'},
 'interface': {'number': '06',
               'name': 'ovui',
               'title': 'Build controls for your digital twin.',
               'lobby_title': 'Build your application controls.',
               'subtitle': 'WIDGETS / LAYOUTS / DATA MODELS',
               'color': (0.94, 0.42, 0.7),
               'center': 0.0,
               'summary': 'Ready-made buttons, sliders, toggles and menus turn scene data into a working '
                          'operator panel.',
               'use': 'Compose a UI for your application: move equipment, set operating state and show live '
                      'feedback. Multiple widgets can share one data model.',
               'url': 'https://github.com/NVIDIA-Omniverse/ovui',
               'implementation': 'Standalone OVUI hosts the whole application through its '
                                 'omni.ui Python API. Buttons, sliders, layouts and shared value models '
                                 'control the ovstage scene. The pallet stacker follows the shared height model '
                                 'within its mast travel range. This is kinematic '
                                 'geometry, not a hydraulic or load-capacity simulation. QC status is '
                                 'user-authored.'},
 'streaming': {'number': '07',
               'name': 'ovstream',
               'title': 'Bring your scene to a client.',
               'subtitle': 'RENDERED FRAMES / CLIENT COMMANDS',
               'color': (0.22, 0.88, 0.79),
               'center': 0.0,
               'summary': 'Watch a scene camera delivered through ovstream, then send a command back from '
                          'the receiving client.',
               'use': 'Connect rendered experiences to other processes and devices. Pixels travel out; user '
                      'actions travel back to the application.',
               'url': 'https://github.com/NVIDIA-Omniverse/ovstream',
               'implementation': 'OVStream server and client deliver ovrtx camera frames over '
                                 'local shared memory, with CUDA frame upload and bidirectional messages. '
                                 'Both endpoints run inside this app. This demonstrates local transport, not '
                                 'a deployed remote service or WebRTC.'},
 'newton': {'number': '10',
            'name': 'ovnewton',
            'title': 'Explore articulated motion.',
            'subtitle': 'ARTICULATIONS / DRIVES / MOTION',
            'color': (0.99, 0.68, 0.3),
            'center': 0.0,
            'summary': 'Move a suspended load with an articulated gantry. Compare gentle and fast drive '
                       'commands and measure the resulting sway.',
            'use': 'Connect robots and articulated mechanisms to Newton through scene data. '
                   'Tune joint drives and inspect their physical response.',
            'url': 'https://github.com/NVIDIA-Omniverse/ovnewton',
            'implementation': 'ovnewton imports an original USD gantry through ovstage, consumes prismatic '
                              'drive targets and publishes Newton XPBD body states at sealed ordinals. The '
                              'application owns the 120 Hz solver loop. Each run computes a fresh trajectory '
                              'in an isolated CPU worker; the UI plays those results at simulation speed and '
                              'interpolates poses for display. A 35 kg rigid load hangs on a rigid revolute '
                              'link, not a flexible cable. Motion targets and mass/inertia are illustrative. '
                              'OVNewton is paired with OVStage in its own environment; '
                              'the newer main-stage metadata schema is incompatible with this release.'},
 'package': {'number': '11',
             'name': 'ovpackage',
             'title': 'Deliver the right asset revision.',
             'subtitle': 'PUBLISH / INSTALL / VERIFY',
             'color': (0.52, 0.78, 0.99),
             'center': 0.0,
             'summary': 'Publish a robot asset, install a selected revision and verify its files. Detect a '
                        'changed working copy, then restore the published asset.',
             'use': 'Hand off a repeatable asset revision between teams or workstations. Keep published '
                    'packages separate from working files and detect when the delivered content differs.',
             'url': 'https://github.com/NVIDIA-Omniverse/ovpackage',
             'implementation': 'ovpackage create, install, catalog(local_hash=True) and diff_catalogs '
                               'perform real file operations and content verification in a local repository. '
                               'Packages contain authored robot USD geometry, materials and their texture '
                               'dependencies. The receiving display applies transform and visibility values '
                               'read from the installed USD. The application supplies revision names, UI and '
                               "the deliberate working-copy edit. Only ovpackage's own install receipt is "
                               'excluded from payload comparison. Each visit owns a separate workspace; '
                               'published files are preserved. No cloud service is represented by this local '
                               'demo.'}}
ROOMS.update(value('ROOMS', {}))
ROOMS = dict(sorted(ROOMS.items(), key=lambda item: int(item[1]['number'])))
MORE_LIBRARIES = []
