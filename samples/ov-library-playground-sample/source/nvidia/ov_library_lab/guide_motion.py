# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A settled atrium host with occasional gestures and visitor attention."""
import math
from .layout import LOBBY_POSE
from .campus_assets import CAMPUS_ENABLED, CAMPUS_TEN_GALLERIES, CAMPUS_VISITOR_POLISH

# Beside the sculpture, clear of its plinth and all room entrances. The robot
# faces the visitor's starting position rather than choosing random headings.
HOME = ((-2.4 if CAMPUS_TEN_GALLERIES else -1.8), 4.5) if CAMPUS_ENABLED else (2.8, 2.45)
if CAMPUS_ENABLED and CAMPUS_VISITOR_POLISH:
    HOME = (3.5, 3.4)
HOME_YAW = math.atan2(LOBBY_POSE[0]-HOME[0], LOBBY_POSE[1]-HOME[1])
RADIUS = .55
TALK_DISTANCE = 2.8
HOVER_HEIGHT = .24
IDLE_PERIOD = 20.


def angle_delta(target, current):
    return (target-current+math.pi) % (2*math.pi)-math.pi


def smooth(value):
    value = max(0.,min(1.,value))
    return value**3*(value*(value*6.-15.)+10.)


def gesture(time, start, end, ease=2.):
    return smooth((time-start)/ease)*smooth((end-time)/ease)


class GuideMotion:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.z = HOME
        self.yaw = HOME_YAW
        self.clock = self.idle_clock = 0.
        self.clipboard = self.writing = 0.
        self.state = 'idle'
        self.attending = False

    def update(self, dt, visitor=None, talking=False, frozen=False):
        remaining = min(.1, max(0., float(dt)))
        if frozen:return
        while remaining > 1e-8:
            step = min(remaining, 1/60)
            remaining -= step
            self._step(step, visitor, talking)

    def _step(self, dt, visitor, talking):
        self.clock += dt
        separation = math.dist((self.x,self.z),visitor) if visitor else math.inf
        was_attending = self.attending
        self.attending = talking or separation < (3.15 if self.attending else TALK_DISTANCE)
        heading = HOME_YAW
        if self.attending and visitor and separation > .01:
            heading = math.atan2(visitor[0]-self.x,visitor[1]-self.z)
        # Watching from nearby is not a conversation. Keep the idle gestures
        # visible until the visitor actually opens the guide's dialogue.
        if talking:
            self.idle_clock = 0.
            self.clipboard = self.writing = 0.
            self.state = 'talking'
        else:
            # A visitor arriving during a quiet pause should see a gesture
            # soon. Never restart one already in progress; proximity uses
            # hysteresis above so standing at the threshold cannot retrigger it.
            if self.attending and not was_attending and self.clipboard < .001:
                self.idle_clock = 1.
            self.idle_clock += dt
            phase = self.idle_clock % IDLE_PERIOD
            # Raise, write, lower, then rest. Keep more than half the cycle
            # quiet without the old long wait for someone standing nearby.
            self.clipboard = gesture(phase,2.,10.,1.5)
            self.writing = gesture(phase,5.,7.5,.6)
            self.state = 'writing' if self.writing > .001 else 'clipboard' if self.clipboard > .001 else 'idle'
        error = angle_delta(heading,self.yaw)
        turn = error*(1.-math.exp(-dt*1.8))
        self.yaw += max(-dt*.55,min(dt*.55,turn))

    def blocks(self, x, z, radius):
        return math.hypot(x-self.x,z-self.z) < RADIUS+radius

    def in_reach(self, walker):
        dx,dz = self.x-walker.x,self.z-walker.z
        distance = math.hypot(dx,dz)
        return (distance <= TALK_DISTANCE and
                (distance < .5 or (dx*math.sin(walker.yaw)-dz*math.cos(walker.yaw))/distance > .25))
