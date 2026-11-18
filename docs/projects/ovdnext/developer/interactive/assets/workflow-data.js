// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Recorded integration and packaged formatter results; see DATA.md.
window.OVD_WORKFLOW_DATA = {
  "tcp": {
    "test": "test_authentic_selected_then_all_ack_frames_and_saved_payloads",
    "platform": "Linux",
    "captureSha256": "f14133f6775d4e34c1b8b9bb85f7724efca2bccf38947464ee62a77257b285d5",
    "evidenceSha256": "84a47857c975c79f82cc33783daa92b5a4791c8bb53fa1947c55d8a18291caec",
    "senderSha256": "1fa7e77bc3e1b5b6c98d98e1918d993d403cf2d508c9a494ae360aab3872b5c8",
    "senderRevision": "ad5c86f498025752261284367f0f1908d392a6b1",
    "sourceKind": "retained SDK reader output and receiver observations",
    "catalog": [
      {
        "name": "box",
        "class": "PxRigidDynamic"
      },
      {
        "name": "ground",
        "class": "PxRigidStatic"
      },
      {
        "name": "selection fixture",
        "class": "PxScene"
      }
    ],
    "states": [
      {
        "label": "Discovery",
        "requestId": null,
        "captureEpoch": null,
        "ackStatus": "catalog ready",
        "boxPresent": false,
        "groundPresent": false,
        "lastPose": null,
        "lastVelocity": null,
        "mass": null
      },
      {
        "label": "Selected box \u00b7 Primary",
        "requestId": 1,
        "captureEpoch": 1,
        "ackStatus": "applied",
        "boxPresent": true,
        "groundPresent": false,
        "lastPose": [
          0.0,
          0.0,
          0.0,
          1.0,
          0.0,
          1.9972749948501587,
          0.0
        ],
        "lastVelocity": [
          0.0,
          -0.16350001096725464,
          0.0
        ],
        "mass": null
      },
      {
        "label": "All permitted \u00b7 + Secondary",
        "requestId": 2,
        "captureEpoch": 2,
        "ackStatus": "applied",
        "boxPresent": true,
        "groundPresent": true,
        "lastPose": [
          0.0,
          0.0,
          0.0,
          1.0,
          0.0,
          1.991824984550476,
          0.0
        ],
        "lastVelocity": [
          0.0,
          -0.3270000219345093,
          0.0
        ],
        "mass": [
          1.0
        ]
      }
    ],
    "stopFrames": 5,
    "readerEof": true
  },
  "contactOffsets": {
    "capture": "multi_geom.ovd",
    "captureSha256": "8e75574bd535dba0ea06494af81055c9aa21c3ad63ba731fe880b1dc262fa0de",
    "scene": 3,
    "frame": 2,
    "shapes": [
      {
        "id": 10,
        "contactOffset": 0.019999999552965164,
        "restOffset": 0.0
      },
      {
        "id": 7,
        "contactOffset": 0.019999999552965164,
        "restOffset": 0.0
      }
    ],
    "gapSource": "Illustrative user-controlled gap; not a recorded trajectory or solver result."
  }
};
