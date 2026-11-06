# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit NPC conversation choices grounded in the lab's actual exhibits.

No free-text matching: every visible choice leads to its own prepared topic.
"""
from dataclasses import dataclass
from .content import ROOMS
from .experience import ROOM_RESTRICTIONS
from .extensions import value, call
from .campus_assets import CAMPUS_ENABLED, CAMPUS_HIDDEN_ROOMS, CAMPUS_TEN_GALLERIES, CAMPUS_VISITOR_POLISH


@dataclass(frozen=True)
class Choice:
    target: str
    label: str


@dataclass(frozen=True)
class Page:
    title: str
    text: str
    choices: tuple[Choice, ...] = ()
    room: str | None = None


DIRECTIONS = {'rendering': 'the left room on the front wall',
 'scene': 'the third room from the left on the front wall',
 'newton': 'the far-right room on the front wall',
 'package': 'the second bay from the left on the rear wall',
 'physics': 'the front half of the left wall',
 'interface': 'the rear half of the left wall',
 'storage': 'the rear half of the right wall',
 'streaming': 'the left room on the rear wall'}
DIRECTIONS.update(value('DIRECTIONS', {}))
if not value('ROOMS'):
    DIRECTIONS.update(scene='the second room on the front wall', newton='the third room on the front wall',
                      package='the far-right room on the front wall', storage='the front half of the right wall')
if CAMPUS_ENABLED:
    DIRECTIONS.update({
        'scene': 'the right gallery on the front wall',
        'newton': 'the right gallery on the rear wall',
        'package': 'the rear gallery on the right wall, beyond OVStorage',
        **value('CAMPUS_DIRECTIONS', {}),
    })
    if CAMPUS_TEN_GALLERIES:
        DIRECTIONS.update({'package': 'the center gallery on the rear wall', **value('TEN_DIRECTIONS', {})})
        if CAMPUS_VISITOR_POLISH:
            DIRECTIONS.update({'rendering': 'the center gallery on the front wall', **value('FLAGSHIP_DIRECTIONS', {})})

OVERVIEWS = {'newton': 'ovnewton connects scene data to Newton physics. This gantry uses a driven sliding joint and a '
           'pivoted load. Compare transfer profiles to see how acceleration changes the load’s sway.',
 'package': 'ovpackage publishes and installs repeatable asset packages. This room hands a robot asset from '
            'a source station to a receiving workspace, then checks that its files match the published '
            'revision.',
 'physics': 'ovphysx simulates rigid-body motion and contact. In this room, gravity pulls cartons onto a '
            'receiving belt, contacts support a stack, and friction changes how far a load slides.',
 'rendering': 'ovrtx turns scene data into images and sensor measurements. Start with a moving robot: '
              'LiDAR and two fisheye cameras show the same world in different ways. The car and robot '
              'arms let you explore surface finishes and illumination.',
 'scene': 'ovstage is the shared scene-data layer behind this lab. The fleet room shows reusable assets and '
          'live scene updates, with two camera views reading the same world state.',
 'storage': 'ovstorage provides a common interface for reading and writing asset data across storage '
            'backends. This room uses its local file backend to save and restore warehouse layout revisions.',
 'interface': 'ovui provides ready-made buttons, sliders, menus, layouts and data models for building '
              'application interfaces. Here, those widgets form an operator panel that controls equipment in '
              'the scene. This guide is also built with ovui.',
 'streaming': 'ovstream carries rendered frames to a client and commands back to the application. This '
              'exhibit uses a local server and client to show that two-way connection in action.'}
OVERVIEWS.update(value('OVERVIEWS', {}))

EXPERIMENTS = {'newton': 'Run the gentle transfer, then run the fast transfer. Watch the suspended load swing and compare '
           'the measured peak sway. You can pause or replay each computed motion.\n'
           '\n'
           'Both runs move the same rigid load across the same distance. The joint-drive command changes; '
           'Newton computes the physical response.',
 'package': 'Choose a logistics load or inspection rig and publish it. Install the revision to see it on the '
            'receiving station. Publish the other configuration and use the revision selector to switch '
            'between them.\n'
            '\n'
            'Remove the installed payload to make a working-copy edit. Verification detects the changed USD; '
            'Restore selected puts the published asset back.',
 'physics': 'In Drop, choose a transfer height and run the experiment. Reset and compare Earth gravity with '
            'low gravity.\n'
            '\n'
            'In Stack, settle the cartons and nudge the top layer. In Slide, run both friction conditions '
            'and compare the measured travel distances.',
 'rendering': 'Start with Robot sensors. Pause the robot and move the pallet; compare the LiDAR point '
              'cloud with the front and rear fisheye images.\n'
              '\n'
              'Switch to Concept car to compare finishes, move lights and explore camera outputs. '
              'Robot arms offers another material study.',
 'scene': 'Grow one AMR into a fleet, change the shared cargo, and dispatch the vehicles. Watch both camera '
          'views reflect the same changes.\n'
          '\n'
          'The fleet motion is scripted by the lab; the exhibit demonstrates scene data and instancing, not '
          'robot navigation.',
 'storage': 'Change the rack configuration and save a USD revision. Make another change and save again.\n'
            '\n'
            'Select an earlier revision and restore it to see that layout return. The lab reads the saved '
            'bytes back through ovstorage before confirming a successful save.',
 'interface': 'In Operate, raise the forks, change the pallet QC state and switch the work light. In Compose, choose '
              'which controls appear.\n'
              '\n'
              'In Data models, edit the slider or number field: both widgets and the scene update from the '
              'same value model.',
 'streaming': 'Connect the camera feed and dispatch an AMR from the receiving client. Pause delivery: the '
              'client holds its last image while the scene keeps moving.\n'
              '\n'
              'Resume to receive fresh frames, or send a load/unload command back through the connection.'}
EXPERIMENTS.update(value('EXPERIMENTS', {}))

HOW_IT_WORKS = {'newton': 'ovnewton builds a Newton model from the gantry’s USD data populated into ovstage. The lab writes '
           'drive targets into the stage, runs the solver and publishes the resulting body states through '
           'ovnewton.\n'
           '\n'
           'The UI displays those computed poses at simulation speed. The load is rigid and connected by a '
           'pivoted rigid link; flexible cables are not part of this experiment.',
 'package': 'ovpackage copies the source USD, material networks and textures into a versioned local '
            'repository. Installing a revision creates a receiving workspace. The lab reads that installed '
            'USD to update the robot you see.\n'
            '\n'
            'ovpackage builds file catalogs with content hashes and compares them with the published source. '
            'The deliberate payload removal is a lab-authored edit, which the library detects and repairs on '
            'restore.',
 'physics': 'ovphysx advances the simulation in small steps, applying gravity and resolving contacts between '
            'the cartons and conveyor. Friction affects sliding, while the resulting body positions update '
            'the scene you see.\n'
            '\n'
            'The lab uses rigid cartons with illustrative material values. It does not simulate cardboard '
            'bending or crushing.',
 'rendering': "ovrtx renders camera images from the scene's geometry, materials and lights. Changing a "
              'finish or moving a light changes how that surface appears in the next image.\n'
              '\n'
              'The robot produces native LiDAR returns and two camera images from the same scene revision. '
              'Its route is scripted and the sensor calibration is illustrative. The car and arms offer '
              'material and lighting controls, plus an inspection camera for comparisons.',
 'scene': 'ovstage holds the shared scene that rendering and the other libraries use. The lab loads reusable '
          'USD assets, creates instances and updates their transforms, materials and visibility. Both camera '
          'views read those same changes.\n'
          '\n'
          'The fleet follows motion authored by the lab; this is not autonomous navigation.',
 'storage': 'ovstorage writes, lists and reads the saved USD files through its local file backend. When you '
            'save, the lab reads the bytes back before confirming success; restoring loads the selected '
            'revision into the scene.\n'
            '\n'
            'The lab manages the revision names. This exhibit uses local files rather than a remote storage '
            'service.',
 'interface': 'ovui connects widgets to values in your application. Here, a slider and number field share a '
              'data model, so editing either updates both controls and the pallet stacker in the scene. Buttons and '
              'menus use the same pattern for lighting and operating state.\n'
              '\n'
              "The lab authors the stacker's movement; the exhibit demonstrates interface behavior rather than "
              'hydraulic simulation.',
 'streaming': 'ovrtx produces the camera images, and ovstream sends them from a server to a receiving '
              'client. Commands such as dispatch and load/unload travel back to the application.\n'
              '\n'
              'This exhibit uses local transport through shared memory, with both endpoints in the lab. It '
              'does not represent a deployed remote streaming service.'}
HOW_IT_WORKS.update(value('HOW_IT_WORKS', {}))

TOPICS = (
    ('overview', 'What does it do?'),
    ('try', 'What can I try?'),
    ('use', 'Where is it useful?'),
    ('details', 'How does it work?'),
    ('where', 'Where is the room?'),
)


def room_choices(keys, allowed):
    return tuple(Choice(f'room:{key}:overview', ROOMS[key]['name']) for key in keys if allowed(key) and key not in CAMPUS_HIDDEN_ROOMS)


def page(topic, allowed=lambda key: True):
    if topic == 'home':
        return Page('Welcome to the Playground',
            "Hi! I'm your guide. Explore Omniverse libraries, try their features, "
            'and discover how they can help with your own use cases.\n\n'
            'Choose a topic below. I can point you to an experiment, explain what a library does, '
            'or help you find a room.',
            (Choice('start', 'Where should I start?'), Choice('libraries', 'Explore a library'),
             Choice('factory', 'Use case demonstrations'), Choice('navigation', 'Getting around')))
    if topic == 'libraries':
        return Page('Choose a library', 'Pick a library, then choose what you would like to explore.',
            tuple(Choice(f'room:{key}:overview', room['name']+('' if allowed(key) else ' / unavailable'))
                  for key, room in ROOMS.items() if key not in CAMPUS_HIDDEN_ROOMS))
    if topic == 'start':
        third = call('suggested_room', allowed) or 'scene'
        return Page('Three good places to start',
            'Start with ovrtx to explore a moving robot\'s LiDAR and fisheye cameras, '
            'or switch to the car to compare finishes and lighting. Try ovphysx to see cartons drop, '
            'settle and slide.\n\n'+('Then explore '+ROOMS[third]['name']+': '+ROOMS[third]['summary'])+
            '\n\nPick a library to see what to try. There is no required order.',
            room_choices(('rendering', 'physics', third), allowed))
    if topic == 'factory':
        choices = [('physics', 'Physical simulation'), ('rendering', 'Visualization & sensing'),
                   ('scene', 'Shared scene data'),
                   ('newton', 'Articulated motion'), ('package', 'Asset handoff')]
        choices.extend(value('FACTORY_CHOICES', ()))
        return Page('Start with your use case',
            'Choose a workflow to find a relevant experiment. Each room connects a library to a '
            'small example you can adapt to your own application.',
            tuple(Choice(f'room:{key}:use', label) for key, label in choices if allowed(key) and key not in CAMPUS_HIDDEN_ROOMS))
    if topic == 'navigation':
        return Page('Explore at your own pace',
            'Use WASD to walk, drag the scene or use the arrow keys to look around, '
            'and hold Shift to move faster. The panel on the right holds each room\'s experiments.\n\n'
            'Home returns to the atrium; F1 opens settings. Press Esc, use the close icon, '
            'or click the scene to leave this conversation. Come back and press E to talk again.',
            (Choice('libraries', 'Show me the libraries'),))
    parts = topic.split(':')
    if len(parts) != 3 or parts[0] != 'room' or parts[1] not in ROOMS or parts[2] not in dict(TOPICS):
        raise ValueError('Unknown guide topic: '+topic)
    _, key, subject = parts
    room = ROOMS[key]
    if not allowed(key):
        return Page(room['name']+' / unavailable',
            'This experiment is unavailable in the current profile. '+
            ('It requires a separately installed internal library.' if key in ROOM_RESTRICTIONS
             else 'Its required library is not enabled.')+
            '\n\nUse Back to choose another library.', room=key)
    titles = {'overview': 'What it does', 'try': 'Try it in the lab',
              'use': 'Where you can use it', 'details': 'Inside the demo', 'where': 'Find the room'}
    texts = {'overview': OVERVIEWS[key], 'try': EXPERIMENTS[key], 'use': room['use'],
             'details': HOW_IT_WORKS[key],
             'where': f"Find {room['number']} / {room['name']} in {DIRECTIONS[key]}.\n\n"
                      'These directions are from the entrance, facing the NVIDIA sculpture. '
                      'You can also use the room buttons at the top of the screen.'}
    return Page(room['name']+' / '+titles[subject], texts[subject],
                tuple(Choice(f'room:{key}:{next_topic}', label) for next_topic, label in TOPICS
                      if next_topic != subject), key)


class Conversation:
    """Keep branch context and history; only visible choices can be followed."""
    def __init__(self, allowed=lambda key: True):
        self.allowed = allowed
        self.topic = 'home'
        self.history = []

    @property
    def current(self):
        return page(self.topic, self.allowed)

    def choose(self, target):
        if target not in {choice.target for choice in self.current.choices}:
            raise ValueError('Choice is not available on this page: '+target)
        self.history.append(self.topic)
        self.topic = target

    def back(self):
        if self.history:
            self.topic = self.history.pop()

    def home(self):
        self.topic = 'home'
        self.history.clear()
