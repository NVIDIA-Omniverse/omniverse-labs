# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kit controls only: all file, HTTP, indexing and replay work is shared Python."""
from datetime import datetime
import omni.ui as ui

TIME_FIELD_STYLE = {"background_color": 0xff535354, "color": 0xffdddddd}


def clock(created, offset=0.):
    return datetime.fromtimestamp(created+offset).strftime('%H:%M:%S.%f')[:-3] if created else '--:--:--'


class TimelinePanel:
    def __init__(self, controller):
        self.controller = controller
        self.syncing = False
        self.record_target = None
        self.mode_target = None
        self.recordings = []
        self.list_signature = None
        self.window = ui.Window('Recording and Playback', width=900, height=174)
        with self.window.frame:
            with ui.VStack(spacing=6, margin=10):
                with ui.HStack(height=26, spacing=8):
                    ui.Label('Mode',width=40)
                    self.mode_selector = ui.ComboBox(0,'Live','Capture replay',width=150)
                    self.mode_selector.model.add_item_changed_fn(self._mode_changed)
                    ui.Spacer(width=ui.Fraction(1))
                    self.record_button = ui.Button('Record',width=142,clicked_fn=self._toggle_recording)
                self.playback_controls = ui.VStack(height=0,spacing=6,visible=False)
                with self.playback_controls:
                    with ui.HStack(height=24,spacing=8):
                        ui.Label('Capture',width=62)
                        self.chooser_frame = ui.Frame()
                        self._choices([],None)
                    with ui.HStack(height=24,spacing=10):
                        self.start = ui.StringField(read_only=True,width=154,tooltip='Recording start time',
                            style=TIME_FIELD_STYLE)
                        self.play_button = ui.Button('Play',width=88,clicked_fn=self._toggle_playback)
                        self.slider = ui.FloatSlider(min=0.,max=1.,step=.01,format='--:--:--',style=TIME_FIELD_STYLE)
                        self.slider.model.add_value_changed_fn(self._scrub)
                        self.end = ui.StringField(read_only=True,width=154,tooltip='Recording end time',
                            style=TIME_FIELD_STYLE)
                self.info = ui.Label('Record live events, then stop and choose Capture replay (not source history).',height=22,elided_text=True)

    def _toggle_recording(self):
        if self.record_target is not None:
            return
        active = bool(self.controller.snapshot()['timeline']['active'])
        self.record_target = not active
        self.record_button.enabled = False
        if active:
            self.controller.stop_recording()
        else:
            self.controller.record()

    def _toggle_playback(self):
        if self.controller.snapshot()['timeline']['playing']:
            self.controller.pause()
        else:
            self.controller.play()

    def _mode_changed(self, model, _item):
        if self.syncing:
            return
        self.mode_target = model.get_item_value_model().as_int
        if self.mode_target == 0:
            self.controller.live()
        else:
            self._load()

    def _capture_changed(self, _model, _item):
        if not self.syncing and self.mode_selector.model.get_item_value_model().as_int == 1:
            self._load()

    def _choices(self, recordings, selected):
        self.recordings = recordings
        index = next((i for i,r in enumerate(recordings) if r['id']==selected),0)
        names = [datetime.fromtimestamp(r['created']).strftime('%Y-%m-%d %H:%M:%S')+
                 '  |  '+str(round(r['duration'],1))+' s'+('  (incomplete)' if r.get('error') else '') for r in recordings]
        with self.chooser_frame:
            self.chooser = ui.ComboBox(index,*(names or ['No finished recordings']))
            self.chooser.model.add_item_changed_fn(self._capture_changed)

    def _load(self):
        index = self.chooser.model.get_item_value_model().as_int
        self.controller.load(self.recordings[index]['id'] if self.recordings else None)

    def _scrub(self, model):
        if not self.syncing:
            self.controller.seek(model.as_float)

    def update(self, state):
        signature = tuple(r['id'] for r in state['recordings'])
        if signature != self.list_signature:
            old = self.chooser.model.get_item_value_model().as_int
            selected = self.recordings[old]['id'] if old<len(self.recordings) else None
            self._choices(state['recordings'],selected)
            self.list_signature = signature
        recorded = state['mode']=='recorded'
        self.playback_controls.visible = recorded
        active = state['active']
        if self.record_target == bool(active) or state['error']:
            self.record_target = None
        self.record_button.enabled = self.record_target is None and not state['busy']
        self.record_button.text = 'Stop recording' if active else 'Record'
        self.record_button.style = ({
            'Button': {'background_color': 0xff3434b5, 'color': 0xffffffff},
            'Button:hovered': {'background_color': 0xff4545d0},
            'Button:pressed': {'background_color': 0xff26268e},
        } if active else {})
        self.mode_selector.enabled = (bool(state['recordings']) or recorded) and not state['busy']
        self.chooser.enabled = bool(state['recordings']) and not state['busy']
        if self.mode_target == int(recorded) or state['error']:
            self.mode_target = None
        self.play_button.enabled = recorded and not state['busy']
        self.play_button.text = 'Pause' if state['playing'] else 'Play'
        self.slider.enabled = recorded and not state['busy']
        self.syncing = True
        self.mode_selector.model.get_item_value_model().set_value(
            int(recorded) if self.mode_target is None else self.mode_target)
        self.slider.min = 0.
        self.slider.max = max(.001,state['duration'])
        self.slider.model.set_value(state['position'])
        self.syncing = False
        # Keep the numeric model in elapsed seconds; format only its visible text.
        self.slider.format = clock(state['created'],state['position'])
        self.start.model.set_value('Start '+clock(state['created']))
        self.end.model.set_value('End '+clock(state['created'],state['duration']))
        self.info.text = (state['error'] or ('Loading recording...' if state['busy'] else
            ('RECORDING live feed: '+str(round(active['duration'],1))+' s / '+str(active['packets'])+' packets' if active else
             ('File: '+state['file']+' | Live collection continues' if recorded else
              'Record live events, then stop and choose Capture replay (not source history).'))))

    def close(self):
        self.window.destroy()
