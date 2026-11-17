# Current native CI inventory

Captured from the Qt/Boost-free native test graph at `864d38bec29a93d622df7f7d4a53e62966cec529`.
Both platforms discover 293 cases: 288 enabled, five existing disabled diagnostics.
Only three named fixture-dependent runtime skips are permitted; no platform exclusions.

The schema's historical `legacy` key now identifies the 289 native core cases,
not a desktop frontend. Four standalone data tests form the remainder.
The six original CI responsibilities map to the current native/data checks,
version check, two OvUI builds, two fresh-package checks and guarded publisher.
The four root/data blocking IDs remain unchanged; other branch-protection and
publication settings are not changed by this inventory.

The full internal fixture set is not included in the public source selection.
See [source-distribution test limits](../../../SOURCE_DISTRIBUTION.md).
Parent-directory baselines are historical evidence and remain unchanged.
