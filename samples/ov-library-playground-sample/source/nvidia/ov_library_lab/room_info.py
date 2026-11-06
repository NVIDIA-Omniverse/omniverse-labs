# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A focused learning page in the existing sidebar, separate from controls."""
import omni.ui as ui
from .content import ROOMS
from .style import MUTED, INK, GREEN
from .extensions import value


DETAILS = {
    'rendering': [
        ('Robot sensing', 'One native OVRTX step renders LiDAR and both mounted fisheye cameras from the same scene revision. Each point is a measured surface return; the camera images are native color outputs. Pausing stops the scripted route while all sensors stay live.'),
        ('Front and rear', 'Camera names describe their fixed mounting directions. The robot drives backwards on its return trip. Both cameras use an illustrative OpenCV fisheye calibration: 480 × 270, fx = fy = 165 px, principal point (240, 135), and zero distortion coefficients.'),
        ('LiDAR profile', 'The 360° sensor uses a 12 m range, 1° firing steps and full-scan snapshots. Only valid returns are displayed. XYZ, intensity and timing come from the library. These are illustrative settings, not a calibrated physical sensor or an autonomous-navigation system.'),
        ('Materials and camera outputs', 'Select Concept car or Robot arms to explore materials and lights. The Camera tab offers color, distance, normals and object classes. Display colors help interpret the raw measurements. Keep reference freezes an image for comparison; Save raw data preserves the native tensor and its metadata.'),
        ('Inspection lens', 'The car/arm inspection camera uses an illustrative 640 × 360 fisheye calibration with fx = fy = 220 px. Supply measured lens parameters when matching physical hardware.'),
    ],
    'physics': [
        ('Contacts and mass', 'The detailed visible cartons use simpler collision meshes sent to PhysX as convex hulls. Mass changes inertia and contact response; under the same gravity, heavier cartons do not fall faster.'),
        ('Reading the comparison', 'Low- and high-friction runs start the same six cartons at 1.15 m/s. Average travel comes from native body positions after two simulated seconds. Surface values are illustrative; real packaging needs measured contact properties.'),
    ],
    'interface': [
        ('One model, several controls', 'The slider, number field and equipment-attached card share an OVUI value model. Editing any control updates the other controls and the equipment. The automatic cycle also writes to this model.'),
        ('Controls attached to the scene', 'In scene opens a screen-sized card anchored to the pallet stacker. It stays readable as you approach and stays still when the forks move. Closing it or leaving the room hides it. This is a projected OVUI overlay; it does not implement 3D occlusion or Kit viewport APIs.'),
    ],
    'newton': [
        ('From command to motion', 'The application sends joint-drive targets through ovnewton. Newton solves the articulated motion, and ovnewton publishes body states to ovstage. Each run computes a fresh trajectory in a worker, then displays it at simulation speed.'),
        ('Rig assumptions', 'Both drive profiles move the same 35 kg load. The suspension uses a rigid pivoted link, not a flexible cable.'),
    ],
    'package': [
        ('Publish, install, verify', 'Publish creates a reusable versioned asset. Install reads it into the receiving workspace, and Verify checks its files against the publication. Selecting an older revision does not install it until you press the action button.'),
        ('A safe place to experiment', 'This demonstration uses a real local repository and install workspace. The receiving view reads the installed USD. Each visit has its own files; resetting the room preserves published revisions. Optional repair controls demonstrate detecting and restoring an edited payload.'),
    ],
    'streaming': [
        ('The connection', 'Real OVRTX frames travel through an OVStream local shared-memory server/client connection. Commands return through that same connection. Pausing the incoming feed holds the last client image while the source scene keeps moving; this is not a remote deployment example.'),
    ],
}


def make_button(lab):
    """Use OVUI's native pixel provider; standalone Button URLs don't draw."""
    if not getattr(lab,'info_icon_provider',None):
        import numpy as np
        from PIL import Image
        with Image.open(lab.gallery.data_dir/'icons/info.png') as image:
            pixels=np.asarray(image.convert('RGBA')).copy()
        lab.info_icon_provider=ui.ByteImageProvider()
        lab.info_icon_provider.set_data_array(pixels.reshape(-1),[pixels.shape[1],pixels.shape[0]])
    scale=lab.settings.text_scale
    def hover(entered):
        lab.info_icon.style={'color':GREEN if getattr(lab,'showing_room_info',False) else INK if entered else MUTED}
    with ui.ZStack(width=36*scale,height=36*scale):
        # Standalone OVUI collapses a completely empty button to its padding.
        # A blank label reserves a line box behind the icon and its hit target.
        button=ui.Button(' ',width=36*scale,height=36*scale,name='Info',tooltip='How this library works',
            style={
                'Button::Info':{'padding':7*scale},'Button::InfoActive':{'padding':7*scale},
                'Button.Label::Info':{'font_size':22*scale},'Button.Label::InfoActive':{'font_size':22*scale}},
            clicked_fn=lambda:show(lab,not lab.showing_room_info),mouse_hovered_fn=hover)
        with ui.VStack():
            ui.Spacer(height=7*scale)
            with ui.HStack(height=22*scale):
                ui.Spacer(width=7*scale)
                lab.info_icon=ui.ImageWithProvider(lab.info_icon_provider,width=22*scale,height=22*scale,
                    opaque_for_mouse_events=False,style={'color':MUTED})
    return button


def build(lab, key):
    from .panels import show_example
    import webbrowser
    room=ROOMS[key]
    with ui.VStack(spacing=13):
        lab.info_back=ui.Button('Back to controls',height=36,name='Utility',clicked_fn=lambda:show(lab,False))
        lab._label('How it works',27)
        lab._label(room['summary'],16)
        lab._section('WHAT YOU CAN BUILD')
        lab._label(room['use'],15)
        for heading,body in (*DETAILS.get(key,()),*value('ROOM_DETAILS',{}).get(key,())):
            lab._label(heading,18)
            lab._label(body,15,MUTED)
        lab._section('ABOUT THIS DEMONSTRATION')
        lab._label(room['implementation'],15,MUTED)
        ui.Separator(height=1)
        with ui.HStack(height=36,spacing=7):
            ui.Button('Library on GitHub',clicked_fn=lambda:webbrowser.open(room['url']))
            ui.Button('Example code',clicked_fn=lambda:lab.queue_action('Opening example',lambda:show_example(lab,key)))
        ui.Button('Back to controls',height=36,name='Utility',clicked_fn=lambda:show(lab,False))


def show(lab, opened):
    """Swap sidebar pages without destroying active controls or experiment state."""
    was_open=getattr(lab,'showing_room_info',False)
    if opened and not was_open:
        lab.controls_scroll=lab.panel_scroll.scroll_y
    lab.showing_room_info=bool(opened)
    lab.room_info_key=lab.current_room if opened else None
    if opened:
        lab.info_frame.clear()
        with lab.info_frame:build(lab,lab.current_room)
    lab.info_frame.visible=bool(opened)
    lab.controls_frame.visible=not opened
    lab.info_button.name='InfoActive' if opened else 'Info'
    lab.info_icon.style={'color':GREEN if opened else MUTED}
    lab.info_button.tooltip='Back to controls' if opened else 'How this library works'
    for view in [lab.observer_view,*getattr(lab,'robot_camera_views',{}).values()]:
        if view:view.updates_enabled=not opened
    lab.panel_scroll.scroll_y=0 if opened else getattr(lab,'controls_scroll',0)
    lab._clear_input()
