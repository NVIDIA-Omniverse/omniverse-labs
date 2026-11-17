# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
from .runtime_security import configure_runtime_security

configure_runtime_security()

from .application import Application

parser=argparse.ArgumentParser(description="PVD on OVUI — native physics capture workspace")
parser.add_argument("capture",nargs="?")
parser.add_argument("--theme",choices=("dark","light"),default=None)
parser.add_argument("--width",type=int,default=1600)
parser.add_argument("--height",type=int,default=1000)
parser.add_argument("--live","--connect",dest="live",nargs="?",const="5425",help="Listen on an OVD port (default 5425); takes precedence over a file")
args=parser.parse_args()
Application(args.capture,args.width,args.height,args.theme,args.live).run()
