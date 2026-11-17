# Inspector fixtures

The original nine JSON files are unedited stdout from `ovd-inspect` SDK 5.13.0 using
the repository's `test/data/test_scene.ovd` and `resample_two_segment.ovd`.
Capture arguments were relative to the repository root. Cases named
`different-offset` actually return `Incomparable` because an exact sample is
missing; labels preserve the probe identity. The reused-handle cases compare
object 11 in source segment 0 with object 22 in segment 1 at recorded frame 2.
Their shared API handle does not identify an incarnation. Bare hashes are empty.
Generation was checked against archived stdout with only sourcePath replacement.

`v2-cpu-control-orientation.json` is additional unedited SDK 5.13.0 inspector
stdout from the independently recorded +3°/+1.5° CPU controls in
`docs/EVIDENCE_SHOWCASE.md`. Its nonzero quaternion distance exercises the
producer's normalized, sign-invariant `4*atan2(norm(a-b),norm(a+b))` convention.
`left.ovd` and `right.ovd` were relative staged inputs; full captures, command
records and envelope content hashes are generated locally by the walkthrough.
This bare SDK fixture has no capture hashes and supplies no navigation proof.
