"""Measure public SDK kinematic contact, roster, velocity and pose-write behavior."""
import sys, math, json, faulthandler, argparse
from pathlib import Path
faulthandler.dump_traceback_later(60, repeat=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from public_sdk_backend import PhysicsBackend
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--device', choices=('cpu', 'gpu'), default='gpu')
args = parser.parse_args()
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=True)
fixture = Path(__file__).resolve().parents[1] / 'tests/fixtures/public_sdk_kinematic.usda'
results = {}
for moving in (False, True):
    backend = PhysicsBackend(root)
    try:
        backend.dispatch('load',dict(scene_path=str(fixture),device=args.device,body_prims=['/World/Dynamic']))
        for step in range(1,121):
            backend.dispatch('write_poses',dict(poses=[dict(prim_path='/World/Kinematic',translate=[-1+(step/120 if moving else 0),0,1],orient=[0,0,math.sqrt(.5),math.sqrt(.5)])]))
            state = backend.dispatch('step',dict(time_ns=step*8333333,dt_ns=8333333))['states']
            assert len(state)==1 and state[0]['prim_path']=='/World/Dynamic'
        results['moving' if moving else 'held'] = state[0]['translate']['x']
        if moving:
            assert results['moving']>.3,results
        else:
            assert abs(results['held'])<.001,results
        kin = backend._columns('position')['/World/Kinematic']
        quat = backend._columns('orientation')['/World/Kinematic']
        assert max(abs(a-b) for a,b in zip(kin,[0 if moving else -1,0,1]))<1e-5
        assert max(abs(a-b) for a,b in zip(quat,[0,0,math.sqrt(.5),math.sqrt(.5)]))<1e-5
        backend.dispatch('write_velocities',dict(velocities=[dict(prim_path='/World/Dynamic',linear=[2,0,0],angular=[0,0,1])]))
        old=backend._columns('position')['/World/Dynamic'][0]
        backend.dispatch('step',dict(dt_ns=16666667))
        new=backend._columns('position')['/World/Dynamic'][0]
        angular=backend._columns('angularVelocity')['/World/Dynamic'][2]
        assert abs(angular-1)<.02,angular
        assert new-old>.02,(old,new)
        results['velocity_displacement_'+str(moving)] = new-old
        results['angular_velocity_'+str(moving)] = angular
        backend.dispatch('write_poses',dict(reset=True,poses=[dict(prim_path='/World/Dynamic',translate=[3,0,2],orient=[0,0,0,1])]))
        pose=backend._columns('position')['/World/Dynamic']
        assert max(abs(a-b) for a,b in zip(pose,[3,0,2]))<1e-5,pose
    finally:
        backend.unload()
print(json.dumps(results),flush=True)
(root/'physics-writes.json').write_text(json.dumps(results,indent=2)+'\n')
