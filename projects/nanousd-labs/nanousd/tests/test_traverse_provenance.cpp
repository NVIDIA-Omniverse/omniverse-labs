// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Tests for PR #124: cached populated-prim provenance for Stage::Traverse().
//
// Each test exercises one write-side correctness contract documented in the
// PR description. Run via the `nanousd_tests` binary; see entries in
// test_main.cpp's main() for the call sites.

#include "nanousd/nanousd.h"
#include "nanousd/schema.h"

#include <cassert>
#include <iostream>

using namespace nanousd;

// Case 0 (baseline): no writes — verify the cached provenance returns the
// correct strongest spec on a multi-layer fixture. Carried over from the
// original prim_provenance commit; serves as the regression line for the
// fast-path read.
void TestStageTraversePreservesStrongestSpecProvenance() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    bool found = false;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        found = true;
        assert(prim.IsDefined());

        auto order = prim.GetPropertyOrder();
        assert(order.size() == 2);
        assert(order[0] == "overValue");
        assert(order[1] == "baseValue");

        auto overValue = prim.GetAttribute("overValue");
        assert(overValue.IsValid());
        assert(overValue.GetDefault());
        assert(*overValue.GetDefault()->Get<Float>() == 10.0f);

        auto baseValue = prim.GetAttribute("baseValue");
        assert(baseValue.IsValid());
        assert(baseValue.GetDefault());
        assert(*baseValue.GetDefault()->Get<Float>() == 5.0f);
    }
    assert(found);

    std::cout << "  Stage traversal preserves strongest spec provenance: OK\n";
}

// Case 1: in-place field edit on the existing strongest spec. The cache
// records the Spec's location, not its field values, so a mutation through
// the same Spec is picked up on the next read without an explicit Recompose.
void TestStageTraverseSeesInPlaceSpecValueEdit() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    bool seen = false;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        seen = true;
        auto v = prim.GetAttribute("overValue").GetDefault();
        assert(v && *v->Get<Float>() == 10.0f);
    }
    assert(seen);

    Layer& rootLayer = stage.GetMutableLayer();
    Spec* attrSpec = rootLayer.GetSpec(Path::Parse("/GainsDef.overValue"));
    assert(attrSpec != nullptr);
    attrSpec->SetField(FieldNames::defaultValue, Value(Float(99.0f)));

    int updated = 0;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        updated++;
        auto v = prim.GetAttribute("overValue").GetDefault();
        assert(v && *v->Get<Float>() == 99.0f);
    }
    assert(updated == 1);

    std::cout << "  Stage traversal sees in-place value edits without recompose: OK\n";
}

// Case 2: structural writes that change which prims exist mark the population
// cache dirty (populateDirty_) so the next Traverse rebuilds populatedPrims_
// and surfaces the new prims.
void TestStageTraverseRebuildsAfterDefinePrim() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    bool sawNewBeforeDefine = false;
    int initialCount = 0;
    for (const auto& prim : stage.Traverse()) {
        initialCount++;
        if (prim.GetPath() == Path::Parse("/AddedAfterTraverse"))
            sawNewBeforeDefine = true;
    }
    assert(!sawNewBeforeDefine);
    assert(initialCount > 0);

    auto added = stage.DefinePrim(Path::Parse("/AddedAfterTraverse"), Token("Xform"));
    assert(added.IsValid());

    bool sawNewAfterDefine = false;
    int newCount = 0;
    for (const auto& prim : stage.Traverse()) {
        newCount++;
        if (prim.GetPath() == Path::Parse("/AddedAfterTraverse"))
            sawNewAfterDefine = true;
    }
    assert(sawNewAfterDefine);
    assert(newCount == initialCount + 1);

    std::cout << "  Stage traversal rebuilds populatedPrims_ after DefinePrim: OK\n";
}

// Case 3: the cached strongest-spec provenance is a fast path, not the
// source of truth. When a write removes the spec the cache points at,
// Traverse() must fall back to the full opinion walk via
// GetStrongestPrimSpec and return the next-strongest spec rather than a
// stale or empty handle.
void TestStageTraverseFallsBackWhenCachedSpecRemoved() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    bool initial = false;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        initial = true;
        auto overValue = prim.GetAttribute("overValue").GetDefault();
        assert(overValue && *overValue->Get<Float>() == 10.0f);
    }
    assert(initial);

    Layer& rootLayer = stage.GetMutableLayer();
    Path gainsPath = Path::Parse("/GainsDef");
    assert(rootLayer.HasSpec(gainsPath));
    assert(rootLayer.RemoveSpec(gainsPath));
    assert(!rootLayer.HasSpec(gainsPath));

    int fallback = 0;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        fallback++;
        assert(prim.IsValid());
        auto baseValue = prim.GetAttribute("baseValue").GetDefault();
        assert(baseValue && *baseValue->Get<Float>() == 5.0f);
    }
    assert(fallback == 1);

    std::cout << "  Stage traversal falls back when cached spec removed: OK\n";
}

// Case 4: typeName mutation on an existing prim. The cached provenance keeps
// pointing at the same Spec (good — its location hasn't moved), and the
// PrimDefinition cache from PR #123 is keyed by typeName so the new
// typeName looks up a different entry on the next read.
void TestStageTraverseSeesTypeNameChange() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    // Sublayer defines GainsDef as Xform; root has an `over` with no type.
    bool seenAsXform = false;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        seenAsXform = true;
        assert(prim.GetTypeName() == Token("Xform"));
    }
    assert(seenAsXform);

    // Author a stronger typeName opinion on the root layer's over-spec.
    Layer& rootLayer = stage.GetMutableLayer();
    Spec* primSpec = rootLayer.GetSpec(Path::Parse("/GainsDef"));
    assert(primSpec != nullptr);
    primSpec->SetTypeName(Token("Scope"));

    int seenAsScope = 0;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        seenAsScope++;
        // Cached provenance still points at the same root-layer Spec; the
        // typeName field on that Spec is now "Scope", so GetTypeName picks
        // it up. The PrimDefinition cache will look up "Scope" on first
        // access — fresh build, distinct from the "Xform" entry.
        assert(prim.GetTypeName() == Token("Scope"));
    }
    assert(seenAsScope == 1);

    std::cout << "  Stage traversal sees in-place typeName change: OK\n";
}

// Case 5: adding a new attribute spec under an existing prim. The
// populatedPrims_ record is unaffected (the prim itself hasn't moved), and
// the per-layer authored-name index from PR #123 is maintained inline by
// Layer::SetSpec — so the new attribute is visible in GetAttributeNames on
// the next read without an explicit Recompose.
void TestStageTraverseSeesNewAuthoredAttribute() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    // Sanity: the new attribute name isn't there yet.
    bool initial = false;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        initial = true;
        auto names = prim.GetAttributeNames();
        for (const auto& n : names) assert(n != Token("addedLater"));
    }
    assert(initial);

    // Author a new attribute spec on the root layer's over-spec for
    // /GainsDef. SetSpec maintains primAttributes_ inline.
    Layer& rootLayer = stage.GetMutableLayer();
    Spec attrSpec(SpecType::Attribute);
    attrSpec.SetTypeName(Token("float"));
    attrSpec.SetField(FieldNames::defaultValue, Value(Float(7.0f)));
    rootLayer.SetSpec(Path::Parse("/GainsDef.addedLater"), std::move(attrSpec));

    int observed = 0;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/GainsDef")) continue;
        observed++;
        auto names = prim.GetAttributeNames();
        bool found = false;
        for (const auto& n : names) if (n == Token("addedLater")) found = true;
        assert(found);
        auto v = prim.GetAttribute("addedLater").GetDefault();
        assert(v && *v->Get<Float>() == 7.0f);
    }
    assert(observed == 1);

    std::cout << "  Stage traversal sees newly authored attribute spec: OK\n";
}

// Case 6: public CreateAttribute writes through Layer::SetSpec after
// Traverse() has populated prim/provenance caches. The layer property indexes
// must refresh immediately, and the existing populated prim must see the value.
void TestStageCreateAttributeAfterTraverse() {
    auto stage = Stage::CreateInMemory();
    auto root = stage.DefinePrim(Path::Parse("/Root"), Token("Xform"));
    assert(root.IsValid());

    auto initialPrims = stage.Traverse();
    assert(initialPrims.size() == 1);

    root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());
    for (const auto& name : root.GetAttributeNames()) {
        assert(name != Token("lateFloat"));
    }

    auto attr = root.CreateAttribute(Token("lateFloat"), Token("float"));
    assert(attr.IsValid());
    assert(attr.Set(Value(Float(3.5f))));

    auto refreshed = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(refreshed.IsValid());
    bool foundName = false;
    for (const auto& name : refreshed.GetAttributeNames()) {
        if (name == Token("lateFloat")) foundName = true;
    }
    assert(foundName);

    auto value = refreshed.GetAttribute(Token("lateFloat")).GetDefault();
    assert(value && value->Get<Float>() && *value->Get<Float>() == 3.5f);

    int observed = 0;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() != Path::Parse("/Root")) continue;
        observed++;
        auto traversedValue = prim.GetAttribute(Token("lateFloat")).GetDefault();
        assert(traversedValue && traversedValue->Get<Float>() &&
               *traversedValue->Get<Float>() == 3.5f);
    }
    assert(observed == 1);

    std::cout << "  Stage CreateAttribute after Traverse refreshes property cache: OK\n";
}

// Case 7: DefinePrim updates the composed child index and population cache
// even if callers already queried children/traversal before the edit.
void TestStageChildrenAfterNestedDefine() {
    auto stage = Stage::CreateInMemory();
    auto world = stage.DefinePrim(Path::Parse("/World"), Token("Xform"));
    assert(world.IsValid());

    auto roots = stage.GetRootPrims();
    assert(roots.size() == 1);
    assert(roots[0].GetPath() == Path::Parse("/World"));

    auto cachedWorld = stage.GetPrimAtPath(Path::Parse("/World"));
    assert(cachedWorld.IsValid());
    assert(cachedWorld.GetChildren().empty());
    assert(stage.Traverse().size() == 1);

    auto nested = stage.DefinePrim(Path::Parse("/World/A/B/C"), Token("Xform"));
    assert(nested.IsValid());
    auto sibling = stage.DefinePrim(Path::Parse("/World/D"), Token("Scope"));
    assert(sibling.IsValid());

    auto refreshedWorld = stage.GetPrimAtPath(Path::Parse("/World"));
    assert(refreshedWorld.IsValid());
    auto worldChildren = refreshedWorld.GetChildren();
    assert(worldChildren.size() == 2);
    assert(worldChildren[0].GetPath() == Path::Parse("/World/A"));
    assert(worldChildren[1].GetPath() == Path::Parse("/World/D"));

    auto a = stage.GetPrimAtPath(Path::Parse("/World/A"));
    assert(a.IsValid());
    auto aChildren = a.GetChildren();
    assert(aChildren.size() == 1);
    assert(aChildren[0].GetPath() == Path::Parse("/World/A/B"));

    bool sawNested = false;
    size_t traverseCount = 0;
    for (const auto& prim : stage.Traverse()) {
        traverseCount++;
        if (prim.GetPath() == Path::Parse("/World/A/B/C")) sawNested = true;
    }
    assert(sawNested);
    assert(traverseCount == 5);

    std::cout << "  Stage children refresh after nested DefinePrim: OK\n";
}

// Case 8: a UsdPrim captured before a write must keep its original spec
// pointer + prim-index pointer functional. This PR doesn't change UsdPrim's
// internal caching, but documenting the contract under a write keeps a
// regression-shield against future refactors that might.
void TestUsdPrimHandleSurvivesWriteOnUnrelatedSpec() {
    auto stage = Stage::Open("tests/compliance/usda/specifiers_composition.usda");
    assert(stage.IsValid());

    UsdPrim heldGainsDef;
    for (const auto& prim : stage.Traverse()) {
        if (prim.GetPath() == Path::Parse("/GainsDef")) {
            heldGainsDef = prim;
        }
    }
    assert(heldGainsDef.IsValid());

    // Write that adds a new prim — does not touch /GainsDef's spec storage.
    auto unrelated = stage.DefinePrim(
        Path::Parse("/UnrelatedNewPrim"), Token("Xform"));
    assert(unrelated.IsValid());

    // The handle captured before the write still answers correctly. We're
    // not asserting that the handle picks up *new* edits — only that the
    // pre-write reads it provides remain consistent.
    assert(heldGainsDef.IsValid());
    assert(heldGainsDef.GetPath() == Path::Parse("/GainsDef"));
    assert(heldGainsDef.GetTypeName() == Token("Xform"));
    auto v = heldGainsDef.GetAttribute("baseValue").GetDefault();
    assert(v && *v->Get<Float>() == 5.0f);

    std::cout << "  UsdPrim handle survives unrelated structural write: OK\n";
}
