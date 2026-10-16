# USD source identifiers

The shipped [production-line.usda](../../assets/production-line.usda) contains the source-issued IDs of the line and machines. Kit loads them and looks up the same `(source, id)` pairs in the service catalog. No ID is minted by `bindings.json`.

The separation of scene namespace and external identity is described by [OpenUSD-proposals#105](https://github.com/PixarAnimationStudios/OpenUSD-proposals/pull/105). This sample uses `assetInfo.sourceIds.<source>.id` as a dictionary convention within the standard asset-info API; it is not claiming that the proposal is an accepted schema. The key under `sourceIds` names the system that issued the ID. “Composable” names a pattern, not such a system.

## What belongs in the USD

```usda
def Xform "Machine_0" (
    assetInfo = {
        dictionary sourceIds = {
            dictionary factory_simulator = {
                string id = "mock-controller-1"
            }
        }
    }
)
{
}
```

`factory_simulator` is the issuing source in this runnable example. Its contract and producer use `mock-controller-1` unchanged. An external integration must use its actual issuing system and ID. Two systems may use the same ID text, so both parts of the pair are required for matching.

Author this metadata on the assembly. A picked child can resolve to a bound ancestor; meshes do not need a duplicate. A prim may have IDs from several systems, and several prim placements may refer to one source entity. The client resolves every authored pair and presents all matching placements. Use per-source-ID offsets if multiple gauge stacks share a prim.

Keep credentials, endpoints and changing values outside the USD. Source IDs can disclose operational identity; publishing them is a deployment decision, not an anonymization mechanism.

## Author a source ID

Obtain the ID from the source's asset inventory, not from a new UUID generator in the scene pipeline. The reader-authenticated `/catalog` lists the registered source IDs, labels and metrics but does not infer which geometry represents an asset.

```python
from pxr import Usd

stage = Usd.Stage.Open("factory.usda")
if stage is None:
    raise RuntimeError("Cannot open factory.usda")
prim = stage.GetPrimAtPath("/World/ProductionLine/Machine_0")
if not prim:
    raise ValueError("Expected assembly prim is missing")

source = "factory_simulator"
source_id = "mock-controller-1"  # Issued and recognized by this source.
key = f"sourceIds:{source}:id"
existing = prim.GetAssetInfoByKey(key)
if existing is None:
    prim.SetAssetInfoByKey(key, source_id)
elif existing != source_id:
    raise ValueError("Existing source ID differs; resolve the conflict explicitly")

if not stage.Export("factory-bound.usda"):
    raise RuntimeError("Cannot export bound USD")
```

`SetAssetInfoByKey` preserves unrelated asset-info keys and writes to the current edit target; reads return composed metadata. Choose the intended layer in a production workflow. This example exports a separate flattened copy. See the [OpenUSD asset-info API](https://openusd.org/25.11/api/class_usd_prim.html).

## Matched, unmatched and unbound

A valid pair in the active catalog gets telemetry displays. An authored ID absent from the catalog gets an **UNMATCHED** card showing its source and ID, with a button to select the prim; the status line counts unmatched IDs. A prim without a source ID is simply unbound. The sample never guesses a match from a prim name or silently substitutes another system's ID. Scene and catalog changes trigger rematching.

Renames and reparenting can preserve this metadata; replacing or decomposing assets still needs publisher-owned identity decisions. Source history would be fetched from the issuing system using this same ID and a time range. This sample implements only live delivery and local capture replay, not that historical query.
