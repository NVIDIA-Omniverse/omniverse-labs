# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visitor workflow derived from published revisions and verified SDK receipts."""
PROFILES = {'logistics': 'Logistics load', 'inspection': 'Inspection rig'}


def describe(room):
    selected = room.versions[room.selected] if room.versions else None
    latest = room.versions[-1] if room.versions else None
    published = bool(latest and room.published_profiles.get(latest) == room.profile)
    modified = bool(room.installed and room.receipt and not room.receipt['verified'])
    installed = bool(selected and selected == room.installed and room.receipt and room.receipt['verified'])
    next_action = 'restore' if modified else 'publish' if not published else 'install' if not installed else None
    return {
        'selected': selected, 'modified': modified, 'installed': installed, 'next': next_action,
        'source': f"Published: revision {latest}" if published else 'This source configuration has not been published.',
        'target': ('Publish a source first to make a revision available here.' if not selected else
                   f"Selected: {selected} / {PROFILES.get(room.published_profiles.get(selected), 'Robot asset')}\n"
                   + (f"Installed at right: {room.installed}" if room.installed else 'Nothing installed at the receiving station yet.')),
        'result': ('Waiting for an installation. Files are checked automatically after install.' if not room.installed else
                   'Changed files detected. Restore a published revision to repair the working copy.' if modified else
                   f"Verified: {room.receipt['file_count']} files match revision {room.installed}."),
        'hint': ('Next: restore the selected revision to repair the edit.' if modified else
                 'Next: publish the source.' if not published else
                 'Next: install the selected revision on the right.' if not installed else
                 'Handoff complete. The installed files are verified.'),
    }
