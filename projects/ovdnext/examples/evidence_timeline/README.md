# Qualified CPU recording control

This asset-free client demonstrates recording and inspecting the box-scene
parameters discussed in [PhysX issue 584](https://github.com/NVIDIA-Omniverse/PhysX/issues/584).
It runs CPU TGS with CPU SDF cooking. The original GPU workload, contact-reduction
behavior, fix and root cause remain unverified by this example.

The exercised host was WSL Ubuntu 22.04 with GCC 11.4.0. Build the exact public
SDK in **checked** configuration: its Release configuration disables OmniPVD
sampling. Use a new build/install directory and record the actual checkout.

```sh
git clone https://github.com/NVIDIA-Omniverse/PhysX.git physx-control
git -C physx-control checkout b97365c6ddb32fda8469bc788ec18024862b0368
git -C physx-control rev-parse HEAD
cmake -S physx-control/physx -B sdk-control \
  -DCMAKE_BUILD_TYPE=checked -DPX_GENERATE_GPU_PROJECTS=OFF \
  -DPX_GENERATE_STATIC_LIBRARIES=ON -DPX_BUILDPVDRUNTIME=ON -DPX_BUILDSNIPPETS=OFF
cmake --build sdk-control --parallel 4
cmake --install sdk-control --config checked --prefix "$PWD/sdk-installed"
cmake -S examples/evidence_timeline -B rocking-client -DCMAKE_BUILD_TYPE=checked \
  -DPhysX_DIR="$PWD/sdk-installed/lib/cmake/PhysX" \
  -DSHOWCASE_SDK_REVISION=b97365c6ddb32fda8469bc788ec18024862b0368
cmake --build rocking-client --parallel 4
```

Run from the OvdNext repository root. Both capture and log destinations must be
new files. Check every command's exit status; keep incomplete files and any
`.failed` marker as failed evidence, and choose new names for a retry.

```sh
mkdir control-runs
rocking-client/rocking-capture --capture control-runs/cpu240-3.ovd \
  --log control-runs/cpu240-3.json --steps 2400 --hz 240 --tilt 3
rocking-client/rocking-capture --capture control-runs/cpu240-1p5.ovd \
  --log control-runs/cpu240-1p5.json --steps 2400 --hz 240 --tilt 1.5
rocking-client/rocking-capture --capture control-runs/cpu480-3.ovd \
  --log control-runs/cpu480-3.json --steps 4800 --hz 480 --tilt 3
```

The client attaches OmniPVD at physics creation, starts sampling at an idle
boundary, records before/after live SDK observations for every step, and checks
sampling/writer/close status. Logs include actual float timestep, SDK/header
version, source revision, core-library/client hashes, compiler and argv. The
generator checks those observations against actual capture samples; the declared
source revision alone is not binary attestation.

Parameters are length 0.1 m, width 0.0157 m, height 0.1177 m, density 1300 kg/m³,
gravity (0,0,-9.81) m/s², material (static friction 0.5, dynamic friction 0.4,
restitution 0), one CPU dispatcher, TGS, 0.005 m mesh grid and 0.001 m SDF.
Linear/angular damping and sleep threshold are zero; initial drop is 0.0001 m.
`buildGPUData=false` deliberately differs from the reporter's GPU-compatible
cooking. Tilts 1.5° and 3°, rates 240/480 Hz and at most 4800 steps bound this
recipe. No GPU manager is created.

See [the evidence walkthrough](../../docs/EVIDENCE_SHOWCASE.md) to verify and
inspect the captures. Keep captures and full logs outside the source tree.
