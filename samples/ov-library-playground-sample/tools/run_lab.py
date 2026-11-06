# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the standalone app using its installed, pinned library environment."""
from pathlib import Path
import argparse
import os
import sys
_selection = argparse.ArgumentParser(add_help=False)
_selection.add_argument('--profile', default='public')
_profile, _ = _selection.parse_known_args()
if _profile.profile == 'public':
    # Select the catalog before importing the app or constructing any exhibits.
    os.environ['OVLAB_RELEASE_PREVIEW'] = '1'
else:
    os.environ.pop('OVLAB_RELEASE_PREVIEW', None)
if '--verify' in sys.argv:
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(300,repeat=True)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source'))
from nvidia.ov_library_lab.app import main
try:
    sys.exit(main() or 0)
finally:
    if '--verify' in sys.argv:
        faulthandler.cancel_dump_traceback_later()
