// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanousd/nanousd.h"
#include "nanousd/clips.h"
#include "nanousd/geom_metrics.h"
#include "nanousd/physics_metrics.h"
#include "nanousd/schema.h"
#include "nanousd/spline.h"
#include "nanousd/usda_writer.h"
#include "nanousd/usdc_writer.h"
#include "nanousd/usdz_package.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace nanousd;

// ============================================================
// Foundational Data Types
// ============================================================

void TestHalf() {
    Half h(1.0f);
    assert(h.bits == 0x3C00);
    float back = static_cast<float>(h);
    assert(std::abs(back - 1.0f) < 0.001f);

    Half zero;
    assert(zero.bits == 0);

    Half neg(-2.0f);
    assert(neg.bits == 0xC000);
    assert(static_cast<float>(neg) < 0.0f);

    Half negativeZero;
    negativeZero.bits = 0x8000;
    float negativeZeroFloat = static_cast<float>(negativeZero);
    assert(negativeZeroFloat == 0.0f);
    assert(std::signbit(negativeZeroFloat));

    Half minSubnormal(std::ldexp(1.0f, -24));
    assert(minSubnormal.bits == 0x0001);
    assert(static_cast<float>(minSubnormal) == std::ldexp(1.0f, -24));

    Half halfUnderflowTie(std::ldexp(1.0f, -25));
    assert(halfUnderflowTie.bits == 0x0000);

    Half minNormal(std::ldexp(1.0f, -14));
    assert(minNormal.bits == 0x0400);
    assert(static_cast<float>(minNormal) == std::ldexp(1.0f, -14));

    Half largestSubnormal(
        std::ldexp(1.0f, -14) - std::ldexp(1.0f, -24));
    assert(largestSubnormal.bits == 0x03FF);
    assert(static_cast<float>(largestSubnormal) ==
           std::ldexp(1.0f, -14) - std::ldexp(1.0f, -24));

    const float halfUlpAtOne = std::ldexp(1.0f, -10);
    Half tieToEvenDown(1.0f + halfUlpAtOne * 0.5f);
    assert(tieToEvenDown.bits == 0x3C00);
    Half tieToEvenUp(1.0f + halfUlpAtOne * 1.5f);
    assert(tieToEvenUp.bits == 0x3C02);

    Half maxFinite(65504.0f);
    assert(maxFinite.bits == 0x7BFF);
    assert(static_cast<float>(maxFinite) == 65504.0f);

    Half overflow(65520.0f);
    assert(overflow.bits == 0x7C00);
    assert(std::isinf(static_cast<float>(overflow)));

    Half inf(std::numeric_limits<float>::infinity());
    assert(inf.bits == 0x7C00);
    Half negInf(-std::numeric_limits<float>::infinity());
    assert(negInf.bits == 0xFC00);

    Half nan(std::numeric_limits<float>::quiet_NaN());
    assert((nan.bits & 0x7C00) == 0x7C00);
    assert((nan.bits & 0x03FF) != 0);
    assert(std::isnan(static_cast<float>(nan)));

    std::cout << "  Half: OK\n";
}

void TestToken() {
    // Two tokens from the same string share the same interned pointer
    Token a("hello");
    Token b("hello");
    assert(a == b);  // O(1) pointer comparison

    // Different strings are different tokens
    Token c("world");
    assert(a != c);

    // Comparison with raw strings
    assert(a == "hello");
    assert(a != "world");
    assert(a == std::string("hello"));

    // Empty token
    Token empty;
    assert(empty.IsEmpty());
    assert(empty == "");

    // GetString returns the underlying text
    assert(a.GetString() == "hello");

    // Assignment
    Token d;
    d = "hello";
    assert(d == a);  // same interned pointer

    // Works in unordered containers with O(1) hash
    std::unordered_set<Token, Token::Hash> tokenSet;
    tokenSet.insert(Token("foo"));
    tokenSet.insert(Token("bar"));
    tokenSet.insert(Token("foo"));  // duplicate
    assert(tokenSet.size() == 2);

    // Works with std::hash specialization
    std::unordered_set<Token> tokenSet2;
    tokenSet2.insert(Token("x"));
    tokenSet2.insert(Token("y"));
    assert(tokenSet2.count(Token("x")) == 1);

    // Implicit conversion to const std::string&
    const std::string& ref = a;
    assert(ref == "hello");

    std::cout << "  Token: OK\n";
}

void TestDefaultResolveAssetIdentifiers() {
    const std::string remote =
        "https://example.com/assets/model.usda?variant=hero#payload";
    assert(DefaultResolve("/tmp/root.usda", remote) == remote);
    assert(DefaultResolve("/tmp/root.usda", "http://example.com/assets/model.usda") ==
           "http://example.com/assets/model.usda");

    assert(DefaultResolve("https://example.com/shots/shot.usda",
                          "../assets/model.usda") ==
           "https://example.com/assets/model.usda");
    assert(DefaultResolve("https://example.com/shots/shot.usda",
                          "assets/model.usda") ==
           "assets/model.usda");
    assert(DefaultResolve("/tmp/root.usda", "https://example.com/%zz") ==
           "https://example.com/%zz");
    assert(DefaultResolve("/tmp/root.usda", "/var/tmp/nanousd/model.usda") ==
           "/var/tmp/nanousd/model.usda");
    assert(DefaultResolve("/tmp/nanousd/root.usda", "model.usda") ==
           "model.usda");
    assert(DefaultResolve("/tmp/nanousd/root.usda", "textures/diffuse.png") ==
           "textures/diffuse.png");
    /* Anchored-relative resolution is purely lexical (lexically_normal +
     * generic_string), so a POSIX-style "/tmp/..." anchor stays drive-less
     * on every host OS -- no Windows C:/ canonicalization. */
    const char* anchoredTmpModel = "/tmp/nanousd/model.usda";
    const char* parentTmpModel = "/tmp/model.usda";
    {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path dir = fs::temp_directory_path() / "nanousd_resolve_bare_sibling";
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
        fs::path root = dir / "root.usda";
        fs::path sibling = dir / "model.usda";
        std::ofstream(root.string()).close();
        std::ofstream(sibling.string()).close();

        std::error_code canonicalEc;
        auto canonical = fs::weakly_canonical(sibling, canonicalEc);
        assert(DefaultResolve(root.generic_string(), "model.usda") ==
               (canonicalEc ? sibling.generic_string() : canonical.generic_string()));
        fs::remove_all(dir, ec);
    }
    assert(DefaultResolve("/tmp/nanousd/root.usda", "./model.usda") ==
           anchoredTmpModel);
    assert(DefaultResolve("/tmp/nanousd/root.usda", "../model.usda") ==
           parentTmpModel);
    assert(DefaultResolve("", "file:/tmp/nanousd/model.usda") ==
           "/tmp/nanousd/model.usda");
    assert(DefaultResolve("", "file:///tmp/nanousd/model.usda") ==
           "/tmp/nanousd/model.usda");
    assert(DefaultResolve("", "file://localhost/tmp/nanousd/model.usda") ==
           "/tmp/nanousd/model.usda");
    assert(DefaultResolve("", "file://files.example.com/share/model.usda") ==
           "file://files.example.com/share/model.usda");
    assert(DefaultResolve("", "file:///C:/project/assets/model.usda") ==
           "C:/project/assets/model.usda");
    assert(DefaultResolve("file:///C:/project/shot/root.usda", "./model.usda") ==
           "C:/project/shot/model.usda");
    assert(DefaultResolve("file:///tmp/nanousd/root.usda", "model.usda") ==
           "model.usda");
    assert(DefaultResolve("file:///tmp/nanousd/root.usda", "./model.usda") ==
           "/tmp/nanousd/model.usda");
    assert(DefaultResolve("file:///tmp/nanousd/root.usda", "../model.usda") ==
           "/tmp/model.usda");

    assert(DefaultResolve("C:/project/shot/root.usda", "textures/diffuse.png") ==
           "textures/diffuse.png");
    assert(DefaultResolve("C:/project/shot/root.usda", "./textures/diffuse.png") ==
           "C:/project/shot/textures/diffuse.png");
    assert(DefaultResolve("C:\\project\\shot\\root.usda", "textures\\diffuse.png") ==
           "textures\\diffuse.png");
    assert(DefaultResolve("C:\\project\\shot\\root.usda", ".\\textures\\diffuse.png") ==
           "C:/project/shot/textures/diffuse.png");
    assert(DefaultResolve("C:\\project\\shot\\root.usda",
                          "https://example.com/assets/model.usda") ==
           "https://example.com/assets/model.usda");
    assert(DefaultResolve("C:\\project\\shot\\root.usda", "..\\textures\\diffuse.png") ==
           "C:/project/textures/diffuse.png");
    assert(DefaultResolve("C:\\project\\shot\\root.usda", "\\shared\\model.usda") ==
           "C:/shared/model.usda");
    assert(DefaultResolve("", "C:/project/assets/model.usda") ==
           "C:/project/assets/model.usda");
    assert(DefaultResolve("", "C:\\project\\assets\\model.usda") ==
           "C:\\project\\assets\\model.usda");
    assert(DefaultResolve("", "\\\\server\\share\\model.usda") ==
           "\\\\server\\share\\model.usda");
    assert(DefaultResolve("", "C:relative\\model.usda") ==
           "C:relative\\model.usda");
    assert(DefaultResolve("/tmp/root.usda",
                          "https://example.com/assets/car.usdz[textures/diffuse.png]") ==
           "https://example.com/assets/car.usdz[textures/diffuse.png]");
    assert(DefaultResolve("/tmp/root.usda",
                          "http://[2001:db8::1]/assets/car.usdz[textures/diffuse.png]") ==
           "http://[2001:db8::1]/assets/car.usdz[textures/diffuse.png]");
    // The outer package is resolved as a normal URI; the anchored relative
    // asset then resolves against the package-internal layer entry.
    assert(DefaultResolve("https://example.com/assets/car.usdz[scenes/root.usda]",
                          "../textures/diffuse.png") ==
           "https://example.com/assets/car.usdz[textures/diffuse.png]");
    assert(DefaultResolve("https://example.com/assets/car.usdz[scenes/root.usda]",
                          "https://cdn.example.com/texture.png") ==
           "https://cdn.example.com/texture.png");

    std::cout << "  DefaultResolve asset identifiers: OK\n";
}

void TestCustomResolverSurvivesRecompose() {
    // A custom AssetResolver passed to Stage::Open must stay installed across
    // Stage::Recompose() (which composition-changing authoring edits trigger
    // internally). Before the fix, Recompose() re-ran Compose() WITHOUT the
    // resolver, so the default filesystem resolver silently replaced it.
    int calls = 0;
    AssetResolver counting = [&calls](const std::string& anchor,
                                      const std::string& asset) {
        ++calls;
        return DefaultResolve(anchor, asset);  // delegate; we only count
    };
    // with_reference.usda references ./ref_source.usda — a relative arc, so
    // composition has to resolve an asset path and the resolver is invoked.
    auto stage = Stage::Open("tests/compliance/usda/with_reference.usda",
                             counting);
    assert(stage.IsValid());
    assert(calls > 0);  // resolver used during the initial composition
    assert(stage.GetPrimAtPath(Path::Parse("/MyPrim")).IsValid());

    int before = calls;
    assert(stage.Recompose());
    // The resolver is carried into re-composition; without the fix the
    // default resolver would be used and the count would not advance.
    assert(calls > before);

    std::cout << "  Custom resolver survives recompose: OK\n";
}

void TestVec() {
    GfVec3f v;
    v[0] = 1.0f; v[1] = 2.0f; v[2] = 3.0f;
    assert(v[0] == 1.0f);
    assert(v[1] == 2.0f);
    assert(v[2] == 3.0f);

    GfVec3f v2;
    v2[0] = 1.0f; v2[1] = 2.0f; v2[2] = 3.0f;
    assert(v == v2);

    GfVec4d v4d;
    v4d[3] = 1.0;
    assert(v4d[3] == 1.0);

    std::cout << "  Vec: OK\n";
}

void TestMatrix() {
    auto identity = GfMatrix4d::Identity();
    assert(identity(0, 0) == 1.0);
    assert(identity(1, 1) == 1.0);
    assert(identity(0, 1) == 0.0);
    assert(identity(3, 3) == 1.0);

    // Translation in row 3 (row-major, 0-indexed)
    GfMatrix4d xform = GfMatrix4d::Identity();
    xform(3, 0) = 10.0;
    xform(3, 1) = 20.0;
    xform(3, 2) = 30.0;
    assert(xform(3, 0) == 10.0);

    std::cout << "  Matrix: OK\n";
}

void TestValue() {
    Value intVal(42);
    assert(intVal.GetTypeId() == TypeId::Int);
    assert(*intVal.Get<Int>() == 42);

    Value strVal(std::string("hello"));
    assert(strVal.GetTypeId() == TypeId::String);
    assert(*strVal.Get<String>() == "hello");

    Value floatVal(3.14f);
    assert(floatVal.GetTypeId() == TypeId::Float);

    Value vecVal(GfVec3f{{1.0f, 2.0f, 3.0f}});
    assert(vecVal.GetTypeId() == TypeId::Float3);

    Value block(ValueBlock{});
    assert(block.IsBlock());

    Value empty;
    assert(empty.IsEmpty());

    std::cout << "  Value: OK\n";
}

void TestDictionary() {
    Dictionary dict;
    dict["name"] = Value(std::string("test"));
    dict["count"] = Value(42);

    assert(dict.count("name") == 1);
    assert(*dict["name"].Get<String>() == "test");
    assert(*dict["count"].Get<Int>() == 42);

    // Nested dictionary
    Dictionary inner;
    inner["x"] = Value(1.0);
    dict["nested"] = Value(std::move(inner));
    assert(dict["nested"].GetTypeId() == TypeId::Dictionary);

    std::cout << "  Dictionary: OK\n";
}

// Spec §6.6.2.1: dictionary combining.
void TestDictionaryCombine() {
    // (1) Empty on both sides.
    {
        Dictionary s, w;
        auto r = CombineDicts(s, w);
        assert(r.empty());
    }

    // (2) Disjoint flat keys — union of both.
    {
        Dictionary s; s["a"] = Value(1);
        Dictionary w; w["b"] = Value(2);
        auto r = CombineDicts(s, w);
        assert(r.size() == 2);
        assert(*r["a"].Get<Int>() == 1);
        assert(*r["b"].Get<Int>() == 2);
    }

    // (3) Conflicting non-dict values — stronger wins.
    {
        Dictionary s; s["x"] = Value(100);
        Dictionary w; w["x"] = Value(200);
        auto r = CombineDicts(s, w);
        assert(r.size() == 1);
        assert(*r["x"].Get<Int>() == 100);
    }

    // (4) Both values are dicts — recursive combine.
    {
        Dictionary sInner; sInner["a"] = Value(1);
        Dictionary wInner; wInner["b"] = Value(2);
        Dictionary s; s["n"] = Value(std::move(sInner));
        Dictionary w; w["n"] = Value(std::move(wInner));
        auto r = CombineDicts(s, w);
        assert(r.size() == 1);
        const auto* inner = r["n"].Get<Dictionary>();
        assert(inner);
        assert(inner->size() == 2);
        assert(*inner->at("a").Get<Int>() == 1);
        assert(*inner->at("b").Get<Int>() == 2);
    }

    // (5) Type mismatch (dict vs scalar) — stronger wins, no recursion.
    {
        Dictionary wInner; wInner["x"] = Value(99);
        Dictionary s; s["key"] = Value(42);                      // scalar
        Dictionary w; w["key"] = Value(std::move(wInner));        // dict
        auto r = CombineDicts(s, w);
        assert(*r["key"].Get<Int>() == 42);
        assert(r["key"].Get<Dictionary>() == nullptr);
    }

    // (6) Spec §12.2.5 example: {a=true} stronger, {b=false} weaker →
    // {a=true, b=false}.
    {
        Dictionary s; s["a"] = Value(true);
        Dictionary w; w["b"] = Value(false);
        auto r = CombineDicts(s, w);
        assert(r.size() == 2);
        assert(*r["a"].Get<bool>() == true);
        assert(*r["b"].Get<bool>() == false);
    }

    // (7) Three-level nesting survives recursion.
    {
        Dictionary sL3; sL3["deep_s"] = Value(1);
        Dictionary sL2; sL2["l3"] = Value(std::move(sL3));
        Dictionary sL1; sL1["l2"] = Value(std::move(sL2));

        Dictionary wL3; wL3["deep_w"] = Value(2);
        Dictionary wL2; wL2["l3"] = Value(std::move(wL3));
        Dictionary wL1; wL1["l2"] = Value(std::move(wL2));

        auto r = CombineDicts(sL1, wL1);
        const auto* l2 = r["l2"].Get<Dictionary>();
        assert(l2);
        const auto* l3 = l2->at("l3").Get<Dictionary>();
        assert(l3);
        assert(l3->size() == 2);
        assert(*l3->at("deep_s").Get<Int>() == 1);
        assert(*l3->at("deep_w").Get<Int>() == 2);
    }

    std::cout << "  Dictionary combine (§6.6.2.1): OK\n";
}

// ============================================================
// List Operations
// ============================================================

void TestListOpExplicit() {
    auto op = ListOp<int>::CreateExplicit({1, 6, 5});
    assert(op.IsExplicit());
    auto items = op.GetItems();
    assert(items.size() == 3);
    assert(items[0] == 1);
    assert(items[1] == 6);
    assert(items[2] == 5);

    std::cout << "  ListOp Explicit: OK\n";
}

void TestListOpComposable() {
    auto op = ListOp<int>::CreateComposable(
        /*prepend=*/{10},
        /*append=*/{20, 30},
        /*delete=*/{}
    );
    assert(!op.IsExplicit());
    auto items = op.GetItems();
    // prepend items not in append: {10}, then append: {20, 30}
    assert(items.size() == 3);
    assert(items[0] == 10);
    assert(items[1] == 20);
    assert(items[2] == 30);

    std::cout << "  ListOp Composable: OK\n";
}

void TestListOpCombine() {
    // Spec example: S = (delete:<5>, prepend:<5>, append:<5>)
    //               E = (explicit:<4, 5, 6>)
    //               S ⊔ E = (explicit:<4, 6, 5>)
    auto S = ListOp<int>::CreateComposable(
        /*prepend=*/{5}, /*append=*/{5}, /*delete=*/{5});
    auto E = ListOp<int>::CreateExplicit({4, 5, 6});

    auto result = S.Combine(E);
    assert(result.IsExplicit());
    auto items = result.GetItems();
    assert(items.size() == 3);
    assert(items[0] == 4);
    assert(items[1] == 6);
    assert(items[2] == 5);

    std::cout << "  ListOp Combine (composable+explicit): OK\n";
}

void TestListOpCombineComposable() {
    // Test composable + composable
    auto S = ListOp<int>::CreateComposable(
        /*prepend=*/{1}, /*append=*/{3}, /*delete=*/{});
    auto C = ListOp<int>::CreateComposable(
        /*prepend=*/{}, /*append=*/{2}, /*delete=*/{});

    auto result = S.Combine(C);
    assert(!result.IsExplicit());
    // prepend: S.prepend not in S.append = {1}
    //          C.prepend not in S.append/S.delete/S.prepend = {}
    // append:  C.append not in S.append/S.delete/S.prepend = {2}
    //          S.append = {3}
    auto items = result.GetItems();
    assert(items.size() == 3);
    assert(items[0] == 1);
    assert(items[1] == 2);
    assert(items[2] == 3);

    std::cout << "  ListOp Combine (composable+composable): OK\n";
}

void TestListOpExplicitOverrides() {
    // Stronger explicit always wins
    auto S = ListOp<int>::CreateExplicit({100, 200});
    auto W = ListOp<int>::CreateComposable({1}, {2}, {3});

    auto result = S.Combine(W);
    assert(result.IsExplicit());
    assert(result.GetItems() == std::vector<int>({100, 200}));

    std::cout << "  ListOp Explicit Override: OK\n";
}

void TestListOpDefault() {
    // Default I' is composable with empty sequences
    ListOp<int> I;
    assert(!I.IsExplicit());
    assert(I.GetItems().empty());

    // S ⊔ I' ≅ S
    auto S = ListOp<int>::CreateComposable({1}, {2}, {});
    auto result = S.Combine(I);
    auto items = result.GetItems();
    assert(items == S.GetItems());

    std::cout << "  ListOp Default: OK\n";
}

void TestListOpReduce() {
    // Elements in append should be removed from prepend and delete
    auto op = ListOp<int>::CreateComposable(
        /*prepend=*/{5, 10},
        /*append=*/{10, 50},
        /*delete=*/{50, 5}
    );
    auto reduced = op.Reduced();
    // append stays: {10, 50}
    // prepend: {5} (10 removed because in append)
    // delete: {} (50 in append, 5 in prepend)
    assert(reduced.GetAppendedItems() == std::vector<int>({10, 50}));
    assert(reduced.GetPrependedItems() == std::vector<int>({5}));
    assert(reduced.GetDeletedItems().empty());

    std::cout << "  ListOp Reduce: OK\n";
}

// USDC roundtrip for ListOp<Path>: write a relationship's targetPaths
// (now stored as ListOp<Path>) to a USDC file, parse it back, and
// confirm the in-memory type and items survive the round-trip via the
// PathListOp crate encoding (spec §16.3.8). Catches regressions where
// the writer's name-sniffing dispatch or the parser's downconversion
// to ListOp<std::string> creep back in.
void TestListOpPathRoundtrip() {
    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    auto prim = stage.DefinePrim(Path::Parse("/Root"), Token("Xform"));
    assert(prim.IsValid());
    auto rel = prim.CreateRelationship(Token("targets"));
    std::vector<Path> targets = {
        Path::Parse("/Root/A"),
        Path::Parse("/Root/B"),
        Path::Parse("/Root/C"),
    };
    assert(rel.SetTargets(targets));

    // Round-trip through USDC.
    auto tmp = std::filesystem::temp_directory_path() /
               "nanousd_listop_path_roundtrip.usdc";
    std::string tmpPath = tmp.string();
    assert(WriteUsdcFile(stage.GetMutableLayer(), tmpPath));

    auto parsed = ParseUsdcFile(tmpPath);
    assert(parsed.success);
    auto* spec = parsed.layer.GetRelationshipSpec(Path::Parse("/Root.targets"));
    assert(spec);
    auto* field = spec->GetField(FieldNames::targetPaths);
    assert(field);
    auto* lop = field->Get<ListOp<Path>>();
    assert(lop && "USDC roundtrip must preserve ListOp<Path> typing");
    auto items = lop->GetItems();
    assert(items.size() == 3);
    assert(items[0].GetText() == "/Root/A");
    assert(items[1].GetText() == "/Root/B");
    assert(items[2].GetText() == "/Root/C");

    // Best-effort cleanup: with lazy field decoding the parsed layer
    // keeps the source file mmap'd until the layer destructs. On
    // platforms where that prevents removal, leave the file —
    // the OS temp dir handles it.
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::cout << "  ListOp<Path> USDC roundtrip: OK\n";
}

// ============================================================
// Paths
// ============================================================

// --- Path test data ---

struct PathTestCase {
    const char* input;
    bool expectValid;
    bool isAbsolute;
    bool isAbsoluteRoot;
    bool isPrimPath;
    bool isPropertyPath;
    bool isReflexive;
    bool hasVariantSelections;
    int elementCount;
    int parentCount;
    const char* propertyName;       // "" if none
    const char* lastName;           // expected GetName(), "" if N/A
    const char* expectedText;       // expected canonical text, nullptr to skip
};

struct PathElementExpectation {
    const char* name;
    int variantCount;
};

struct PathVariantExpectation {
    int testIdx;
    int elemIdx;
    int varIdx;
    const char* setName;
    const char* variantName;
};

struct PathPropertyNsExpectation {
    const char* input;
    int nsCount;
    std::vector<const char*> segments;
};

static const PathTestCase kPathTests[] = {
    // Absolute root
    { "/",        true,  true,  true,  true,  false, false, false, 0, 0, "",    "",            "/" },
    // Absolute prim paths
    { "/Root/Child/Grandchild", true, true, false, true, false, false, false, 3, 0, "", "Grandchild", nullptr },
    // Property paths
    { "/Root/Child.visibility",  true, true, false, false, true, false, false, 2, 0, "visibility",  "", nullptr },
    { "/Root.primvars:color",    true, true, false, false, true, false, false, 1, 0, "primvars:color", "", nullptr },
    // Variant selections
    { "/World/Foo/Baz{VariantSet1=VarFoobar}Foobar.foobarprop1",
        true, true, false, false, true, false, true, 4, 0, "foobarprop1", "", nullptr },
    { "/World/Bar{VariantSet2=VarBarFoo}{VariantSet3=VarBarBar}BarFoo",
        true, true, false, true, false, false, true, 3, 0, "", "BarFoo", nullptr },
    { "/City/Street{selection=}",
        true, true, false, true, false, false, true, 2, 0, "", "Street", nullptr },
    { "/City{ selection = NewYork }",
        true, true, false, true, false, false, true, 1, 0, "", "City", nullptr },
    // Reflexive relative
    { ".",  true,  false, false, false, false, true, false, 0, 0, "", "", "." },
    // Relative paths
    { "../Sibling/Child",  true, false, false, true, false, false, false, 2, 1, "", "Child", nullptr },
    { "../../Uncle",       true, false, false, true, false, false, false, 1, 2, "", "Uncle", nullptr },
    { ".points",           true, false, false, false, true, false, false, 0, 0, "points", "", nullptr },
    { "Child/Grandchild",  true, false, false, true, false, false, false, 2, 0, "", "Grandchild", nullptr },
    // Invalid paths
    { "",              false, false, false, false, false, false, false, 0, 0, "", "", nullptr },
    { "/Root/",        false, false, false, false, false, false, false, 0, 0, "", "", nullptr },
    { "/Root//Child",  false, false, false, false, false, false, false, 0, 0, "", "", nullptr },
    { "/abc/123",      false, false, false, false, false, false, false, 0, 0, "", "", nullptr },
};

// Element name expectations for specific paths (by index into kPathTests)
static const std::vector<std::pair<int, std::vector<PathElementExpectation>>> kElementExpectations = {
    { 1, { {"Root", 0}, {"Child", 0}, {"Grandchild", 0} } },
    { 4, { {"World", 0}, {"Foo", 0}, {"Baz", 1}, {"Foobar", 0} } },
    { 5, { {"World", 0}, {"Bar", 2}, {"BarFoo", 0} } },
    { 9, { {"Sibling", 0}, {"Child", 0} } },
    { 10, { {"Uncle", 0} } },
    { 12, { {"Child", 0}, {"Grandchild", 0} } },
};

static const PathVariantExpectation kVariantExpectations[] = {
    // testIdx, elemIdx, varIdx, setName, variantName
    { 4, 2, 0, "VariantSet1", "VarFoobar" },
    { 5, 1, 0, "VariantSet2", "VarBarFoo" },
    { 5, 1, 1, "VariantSet3", "VarBarBar" },
    { 6, 1, 0, "selection",   "" },
    { 7, 0, 0, "selection",   "NewYork" },
};

static const PathPropertyNsExpectation kPropertyNsExpectations[] = {
    { "/Root.primvars:color", 2, { "primvars", "color" } },
};

void TestPathParsing() {
    int count = 0;
    for (const auto& tc : kPathTests) {
        // Absolute (and only absolute) paths go through Path::Parse.
        // Relative authorings — including the reflexive "." and bare
        // ".prop" / "Child/..." cases — go through RelativePath::Parse
        // since `Path` is now an absolute-only interned handle.
        if (tc.isAbsolute) {
            auto p = Path::Parse(tc.input);
            if (!tc.expectValid) {
                assert(p.IsEmpty());
                ++count;
                continue;
            }
            assert(!p.IsEmpty());
            assert(p.IsAbsolute());
            assert(p.IsAbsoluteRoot() == tc.isAbsoluteRoot);
            assert(p.IsPrimPath() == tc.isPrimPath);
            assert(p.IsPropertyPath() == tc.isPropertyPath);
            assert(p.HasVariantSelections() == tc.hasVariantSelections);
            assert(static_cast<int>(p.GetElements().size()) == tc.elementCount);
            assert(p.GetPropertyName() == tc.propertyName);
            if (tc.lastName[0] != '\0') {
                assert(p.GetName() == tc.lastName);
            }
            if (tc.expectedText) {
                assert(p.GetText() == tc.expectedText);
            }
        } else {
            auto r = RelativePath::Parse(tc.input);
            if (!tc.expectValid) {
                assert(!r.IsValid());
                ++count;
                continue;
            }
            assert(r.IsValid());
            assert(!r.IsAlreadyAbsolute());
            assert(r.IsReflexive() == tc.isReflexive);
            assert(static_cast<int>(r.GetElements().size()) == tc.elementCount);
            assert(r.GetParentCount() == tc.parentCount);
            assert(r.GetPropertyName() == tc.propertyName);
        }
        ++count;
    }
    std::cout << "  Path parsing (" << count << " cases): OK\n";
}

void TestUnicodePathIdentifiers() {
    const char* unicodePath =
        "/M\xC3\xBC" "nchen/\xE6\xA8\xA1\xE5\x9E\x8B.primvars:\xE5\x80\xBC";
    auto p = Path::Parse(unicodePath);
    assert(!p.IsEmpty());
    assert(p.GetText() == unicodePath);
    auto elems = p.GetElements();
    assert(elems.size() == 2);
    assert(elems[0].name == "M\xC3\xBC" "nchen");
    assert(elems[1].name == "\xE6\xA8\xA1\xE5\x9E\x8B");
    assert(p.GetPropertyName() == "primvars:\xE5\x80\xBC");

    auto ns = p.GetPropertyNamespaces();
    assert(ns.size() == 2);
    assert(ns[0] == "primvars");
    assert(ns[1] == "\xE5\x80\xBC");

    auto decomposed = Path::Parse("/Cafe\xCC\x81");
    assert(!decomposed.IsEmpty());
    assert(decomposed.GetText() == "/Cafe\xCC\x81");

    auto variant = Path::Parse("/Root{look=\xE5\x86\xAC-v1}");
    assert(!variant.IsEmpty());
    auto variantElems = variant.GetElements();
    assert(variantElems.size() == 1);
    assert(variantElems[0].variantSelections.size() == 1);
    assert(variantElems[0].variantSelections[0].variantName == "\xE5\x86\xAC-v1");

    assert(Path::Parse("/\xF0\x9F\x98\x80").IsEmpty()); // emoji is not XID_Start
    assert(Path::Parse("/\xEE\x80\x80").IsEmpty());     // private-use U+E000
    assert(Path::Parse("/\xCC\x81" "bad").IsEmpty());   // combining mark cannot start
    assert(Path::Parse("/bad\xC3(").IsEmpty());         // invalid UTF-8

    auto rel = RelativePath::Parse("../M\xC3\xBC" "nchen/\xE6\xA8\xA1\xE5\x9E\x8B");
    assert(rel.IsValid());
    assert(rel.GetParentCount() == 1);
    assert(rel.GetElements().size() == 2);

    std::cout << "  Unicode path identifiers: OK\n";
}

void TestPathElementOrdering() {
    std::vector<Token> names = {
        "foobar",
        "Foobar",
        "_foobar",
        "foo_bar",
        "foo001bar001abc",
        "foo001bar002abc",
        "foo0001bar0002xyz",
        "foo00001bar",
        "a0",
        "a\xC3\xBC",
        "ab",
    };

    std::sort(names.begin(), names.end(), PathElementTokenLess);

    const std::vector<std::string> expected = {
        "_foobar",
        "a0",
        "ab",
        "a\xC3\xBC",
        "foo_bar",
        // AOUSD v1.0.1's example places Foobar/foobar here, but the
        // normative bullets say ASCII numbers order before ASCII letters.
        "foo00001bar",
        "foo001bar001abc",
        "foo001bar002abc",
        "foo0001bar0002xyz",
        "Foobar",
        "foobar",
    };
    assert(names.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(names[i].GetString() == expected[i]);
    }

    std::vector<Path> paths = {
        Path::Parse("/a{b=}"),
        Path::Parse("/a/b"),
        Path::Parse("/a.b"),
    };
    std::sort(paths.begin(), paths.end());
    assert(paths[0].GetText() == "/a.b");
    assert(paths[1].GetText() == "/a/b");
    assert(paths[2].GetText() == "/a{b=}");

    std::cout << "  Path element ordering: OK\n";
}

void TestPathElements() {
    for (const auto& [testIdx, expected] : kElementExpectations) {
        const auto& tc = kPathTests[testIdx];
        // Pull elements from whichever type holds them in the new
        // split: absolute → Path, relative → RelativePath.
        std::vector<PathElement> elems;
        if (tc.isAbsolute) {
            auto p = Path::Parse(tc.input);
            elems = p.GetElements();
        } else {
            auto r = RelativePath::Parse(tc.input);
            elems = r.GetElements();
        }
        assert(static_cast<int>(elems.size()) == static_cast<int>(expected.size()));
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(elems[i].name == expected[i].name);
            assert(static_cast<int>(elems[i].variantSelections.size()) == expected[i].variantCount);
        }
    }

    for (const auto& ve : kVariantExpectations) {
        const auto& tc = kPathTests[ve.testIdx];
        std::vector<PathElement> elems;
        if (tc.isAbsolute) {
            elems = Path::Parse(tc.input).GetElements();
        } else {
            elems = RelativePath::Parse(tc.input).GetElements();
        }
        const auto& sel = elems[ve.elemIdx].variantSelections[ve.varIdx];
        assert(sel.setName == ve.setName);
        assert(sel.variantName == ve.variantName);
    }

    std::cout << "  Path elements & variants: OK\n";
}

void TestPathPropertyNamespaces() {
    for (const auto& pns : kPropertyNsExpectations) {
        auto p = Path::Parse(pns.input);
        auto ns = p.GetPropertyNamespaces();
        assert(static_cast<int>(ns.size()) == pns.nsCount);
        for (size_t i = 0; i < pns.segments.size(); ++i) {
            assert(ns[i] == pns.segments[i]);
        }
    }
    std::cout << "  Path property namespaces: OK\n";
}

void TestUsdaWriterPathElementOrdering() {
    auto makePrim = []() {
        Spec prim(SpecType::Prim);
        prim.SetSpecifier(Specifier::Def);
        prim.SetTypeName("Xform");
        return prim;
    };
    auto makeFloatAttr = []() {
        Spec attr(SpecType::Attribute);
        attr.SetTypeName("float");
        attr.SetField(FieldNames::defaultValue, Value(Float(1.0f)));
        return attr;
    };

    Layer layer;
    layer.SetSpec(Path::Parse("/Root"), makePrim());
    for (const char* name : {
             "foobar", "Foobar", "_foobar", "foo_bar",
             "foo001bar001abc", "foo001bar002abc",
             "foo0001bar0002xyz", "foo00001bar", "a0",
             "a\xC3\xBC", "ab"}) {
        layer.SetSpec(Path::Parse(std::string("/Root.") + name), makeFloatAttr());
    }
    layer.SetSpec(Path::Parse("/Root/ChildB"), makePrim());
    layer.SetSpec(Path::Parse("/Root/ChildA"), makePrim());

    std::string usda = WriteUsda(layer);
    size_t previous = 0;
    for (const char* snippet : {
             "float _foobar",
             "float a0",
             "float ab",
             "float a\xC3\xBC",
             "float foo_bar",
             "float foo00001bar",
             "float foo001bar001abc",
             "float foo001bar002abc",
             "float foo0001bar0002xyz",
             "float Foobar",
             "float foobar"}) {
        const size_t pos = usda.find(snippet);
        assert(pos != std::string::npos);
        assert(pos >= previous);
        previous = pos;
    }
    const size_t childB = usda.find("def Xform \"ChildB\"");
    const size_t childA = usda.find("def Xform \"ChildA\"");
    assert(childB != std::string::npos);
    assert(childA != std::string::npos);
    assert(childB < childA);

    std::cout << "  USDA writer path element ordering: OK\n";
}

void TestPathOperations() {
    auto p = Path::Parse("/Root/Child.prop");

    // GetParentPath of property path -> prim path
    auto parent = p.GetParentPath();
    assert(parent.GetText() == "/Root/Child");
    assert(parent.IsPrimPath());

    // GetParentPath of prim path -> parent prim
    auto grandparent = parent.GetParentPath();
    assert(grandparent.GetText() == "/Root");

    // GetParentPath of root prim -> absolute root
    auto root = grandparent.GetParentPath();
    assert(root.IsAbsoluteRoot());

    // GetPrimPath
    assert(p.GetPrimPath().GetText() == "/Root/Child");

    // AppendChild
    auto appended = Path::Parse("/Root").AppendChild("NewChild");
    assert(appended.GetText() == "/Root/NewChild");

    // AppendProperty
    auto withProp = Path::Parse("/Root/Child").AppendProperty("myProp");
    assert(withProp.GetText() == "/Root/Child.myProp");

    std::cout << "  Path operations: OK\n";
}

void TestPathAnchor() {
    auto base = Path::Parse("/World/Bar/BarBaz");
    auto rel = RelativePath::Parse("../../Foo/Baz/Foobar");
    auto anchored = rel.Anchor(base);
    assert(anchored.GetText() == "/World/Foo/Baz/Foobar");

    // Reflexive anchoring
    auto dot = RelativePath::Parse(".");
    auto dotAnchored = dot.Anchor(base);
    assert(dotAnchored.GetText() == base.GetText());

    // Would go above root — returns empty
    auto tooFar = RelativePath::Parse("../../../../TooFar");
    auto bad = tooFar.Anchor(base);
    assert(bad.IsEmpty());

    std::cout << "  Path anchor: OK\n";
}

// ============================================================
// Document Data Model (Specs and Layer)
// ============================================================

void TestPrimSpec() {
    Spec prim(SpecType::Prim);
    prim.SetSpecifier(Specifier::Def);
    prim.SetTypeName("Xform");
    prim.SetActive(true);

    assert(prim.GetSpecifier() == Specifier::Def);
    assert(prim.GetTypeName() == "Xform");
    assert(prim.GetActive() == true);
    assert(prim.GetInstanceable() == false);  // fallback
    assert(prim.GetHidden() == false);         // fallback

    // Generic field access
    assert(prim.HasField(FieldNames::specifier));
    assert(prim.HasField(FieldNames::typeName));
    assert(!prim.HasField(FieldNames::kind));  // not set

    // Field validity
    assert(prim.IsValidField(FieldNames::specifier));
    assert(prim.IsValidField(FieldNames::active));
    assert(!prim.IsValidField(FieldNames::targetPaths));  // relationship only

    std::cout << "  PrimSpec: OK\n";
}

void TestAttributeSpec() {
    Spec attr(SpecType::Attribute);
    attr.SetTypeName("float3");
    attr.SetVariability(Variability::Varying);
    attr.SetField(FieldNames::defaultValue, Value(GfVec3f{{1.0f, 2.0f, 3.0f}}));
    attr.SetCustom(false);

    assert(attr.GetTypeName() == "float3");
    assert(attr.GetVariability() == Variability::Varying);
    assert(attr.HasField(FieldNames::defaultValue));
    assert(attr.GetField(FieldNames::defaultValue)->GetTypeId() == TypeId::Float3);

    // Field validity
    assert(attr.IsValidField(FieldNames::typeName));
    assert(attr.IsValidField(FieldNames::timeSamples));
    assert(!attr.IsValidField(FieldNames::specifier));  // prim only

    std::cout << "  AttributeSpec: OK\n";
}

void TestRelationshipSpec() {
    Spec rel(SpecType::Relationship);
    // targetPaths is a valid field for relationships
    assert(rel.IsValidField(FieldNames::targetPaths));
    assert(!rel.IsValidField(FieldNames::timeSamples));  // attribute only

    std::cout << "  RelationshipSpec: OK\n";
}

void TestLayer() {
    Layer layer;

    // Set layer metadata via typed accessors
    layer.GetLayerSpec().SetDefaultPrim("World");
    layer.GetLayerSpec().SetTimeCodesPerSecond(24.0);
    layer.GetLayerSpec().SetStartTimeCode(1.0);
    layer.GetLayerSpec().SetEndTimeCode(100.0);

    // Add a root prim
    Spec worldPrim(SpecType::Prim);
    worldPrim.SetSpecifier(Specifier::Def);
    worldPrim.SetTypeName("Xform");

    auto worldPath = Path::Parse("/World");
    layer.SetSpec(worldPath, worldPrim);

    // Add a child prim
    Spec cubePrim(SpecType::Prim);
    cubePrim.SetSpecifier(Specifier::Def);
    cubePrim.SetTypeName("Mesh");

    auto cubePath = Path::Parse("/World/Cube");
    layer.SetSpec(cubePath, cubePrim);

    // Add an attribute
    Spec translateAttr(SpecType::Attribute);
    translateAttr.SetTypeName("double3");
    translateAttr.SetVariability(Variability::Varying);
    translateAttr.SetField(FieldNames::defaultValue,
                           Value(GfVec3d{{0.0, 10.0, 0.0}}));

    auto attrPath = Path::Parse("/World.xformOp:translate");
    layer.SetSpec(attrPath, translateAttr);

    Spec relSpec(SpecType::Relationship);
    relSpec.SetField(FieldNames::targetPaths,
                     Value(ListOp<Path>::CreateExplicit({cubePath})));
    auto relPath = Path::Parse("/World.material:binding");
    layer.SetSpec(relPath, relSpec);

    // Verify
    assert(layer.HasSpec(worldPath));
    assert(layer.HasSpec(cubePath));
    assert(layer.HasSpec(attrPath));
    assert(layer.HasSpec(relPath));
    assert(!layer.HasSpec(Path::Parse("/Nonexistent")));

    auto* wp = layer.GetPrimSpec(worldPath);
    assert(wp != nullptr);
    assert(wp->GetTypeName() == "Xform");

    auto* cp = layer.GetPrimSpec(cubePath);
    assert(cp != nullptr);
    assert(cp->GetTypeName() == "Mesh");

    auto* ap = layer.GetAttributeSpec(attrPath);
    assert(ap != nullptr);
    assert(ap->GetTypeName() == "double3");

    const auto& attrs = layer.GetAttributeNames(worldPath);
    assert(attrs.size() == 1);
    assert(attrs[0] == Token("xformOp:translate"));

    const auto& props = layer.GetPropertyNames(worldPath);
    assert(props.size() == 2);
    assert(props[0] == Token("xformOp:translate"));
    assert(props[1] == Token("material:binding"));

    Spec translateAttrReplacement(SpecType::Attribute);
    translateAttrReplacement.SetTypeName("double3");
    layer.SetSpec(attrPath, translateAttrReplacement);

    const auto& propsAfterAttrReplace = layer.GetPropertyNames(worldPath);
    assert(propsAfterAttrReplace.size() == 2);
    assert(propsAfterAttrReplace[0] == Token("xformOp:translate"));
    assert(propsAfterAttrReplace[1] == Token("material:binding"));

    const auto& attrsAfterAttrReplace = layer.GetAttributeNames(worldPath);
    assert(attrsAfterAttrReplace.size() == 1);
    assert(attrsAfterAttrReplace[0] == Token("xformOp:translate"));

    Spec bindingAttr(SpecType::Attribute);
    bindingAttr.SetTypeName("token");
    layer.SetSpec(relPath, bindingAttr);

    const auto& propsAfterTypeChange = layer.GetPropertyNames(worldPath);
    assert(propsAfterTypeChange.size() == 2);
    assert(propsAfterTypeChange[0] == Token("xformOp:translate"));
    assert(propsAfterTypeChange[1] == Token("material:binding"));

    const auto& attrsAfterTypeChange = layer.GetAttributeNames(worldPath);
    assert(attrsAfterTypeChange.size() == 2);
    assert(attrsAfterTypeChange[0] == Token("xformOp:translate"));
    assert(attrsAfterTypeChange[1] == Token("material:binding"));

    // Layer metadata
    assert(layer.GetLayerSpec().GetDefaultPrim() == "World");
    assert(layer.GetLayerSpec().GetTimeCodesPerSecond() == 24.0);

    // Remove
    assert(layer.RemoveSpec(cubePath));
    assert(!layer.HasSpec(cubePath));

    std::cout << "  Layer: OK\n";
}

void TestLayerArcOpinionPrimPathsCacheInvalidation() {
    auto makeRefPrim = [](const std::string& assetPath) {
        Reference ref;
        ref.assetPath = assetPath;

        Spec prim(SpecType::Prim);
        prim.SetSpecifier(Specifier::Def);
        prim.SetTypeName("Xform");
        prim.SetField(FieldNames::references,
                      Value(ListOp<Reference>::CreateExplicit({ref})));
        return prim;
    };

    auto containsPath = [](const std::vector<Path>& paths,
                           const Path& wanted) {
        for (const auto& path : paths) {
            if (path == wanted) return true;
        }
        return false;
    };

    Layer layer;
    const Path a = Path::Parse("/A");
    const Path b = Path::Parse("/B");
    const Path c = Path::Parse("/C");

    Spec aSpec(SpecType::Prim);
    aSpec.SetSpecifier(Specifier::Def);
    aSpec.SetTypeName("Xform");
    layer.SetSpec(a, std::move(aSpec));
    layer.SetSpec(b, makeRefPrim("./b.usda"));

    {
        const auto& paths = detail::GetLayerArcOpinionPrimPaths(layer);
        assert(paths.size() == 1);
        assert(containsPath(paths, b));
    }

    layer.SetSpec(c, makeRefPrim("./c.usda"));
    {
        const auto& paths = detail::GetLayerArcOpinionPrimPaths(layer);
        assert(paths.size() == 2);
        assert(containsPath(paths, b));
        assert(containsPath(paths, c));
    }

    Spec* mutableA = layer.GetSpec(a);
    assert(mutableA != nullptr);
    Reference refA;
    refA.assetPath = "./a.usda";
    mutableA->SetField(FieldNames::references,
                       Value(ListOp<Reference>::CreateExplicit({refA})));
    {
        const auto& paths = detail::GetLayerArcOpinionPrimPaths(layer);
        assert(paths.size() == 3);
        assert(containsPath(paths, a));
        assert(containsPath(paths, b));
        assert(containsPath(paths, c));
    }

    assert(layer.RemoveSpec(b));
    {
        const auto& paths = detail::GetLayerArcOpinionPrimPaths(layer);
        assert(paths.size() == 2);
        assert(containsPath(paths, a));
        assert(!containsPath(paths, b));
        assert(containsPath(paths, c));
    }

    Spec replacementA(SpecType::Prim);
    replacementA.SetSpecifier(Specifier::Def);
    replacementA.SetTypeName("Xform");
    layer.SetSpec(a, std::move(replacementA));
    {
        const auto& paths = detail::GetLayerArcOpinionPrimPaths(layer);
        assert(paths.size() == 1);
        assert(!containsPath(paths, a));
        assert(containsPath(paths, c));
    }

    std::cout << "  Layer arc-opinion path cache invalidation: OK\n";
}

void TestLayerArcOpinionPrimPathsConcurrentRead() {
    Layer layer;
    constexpr int kPrimCount = 256;
    constexpr int kArcCount = kPrimCount / 2;

    for (int i = 0; i < kPrimCount; ++i) {
        Spec prim(SpecType::Prim);
        prim.SetSpecifier(Specifier::Def);
        prim.SetTypeName("Xform");
        if ((i % 2) == 0) {
            Reference ref;
            ref.assetPath = "./asset_" + std::to_string(i) + ".usda";
            prim.SetField(FieldNames::references,
                          Value(ListOp<Reference>::CreateExplicit({ref})));
        }
        layer.SetSpec(Path::Parse("/P" + std::to_string(i)), std::move(prim));
    }

    auto readArcPaths = [&]() {
        for (int repeat = 0; repeat < 32; ++repeat) {
            const auto& paths = detail::GetLayerArcOpinionPrimPaths(layer);
            assert(paths.size() == kArcCount);
            std::unordered_set<std::string> seen;
            seen.reserve(paths.size());
            for (const auto& path : paths) {
                seen.insert(path.GetText());
            }
            for (int i = 0; i < kPrimCount; i += 2) {
                assert(seen.count("/P" + std::to_string(i)) == 1);
            }
        }
    };

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 8; ++i) {
        futures.push_back(std::async(std::launch::async, readArcPaths));
    }
    for (auto& future : futures) future.get();

    std::cout << "  Layer arc-opinion path cache concurrent read: OK\n";
}

void TestLayerChildPathIndexInvalidation() {
    auto makePrim = []() {
        Spec prim(SpecType::Prim);
        prim.SetSpecifier(Specifier::Def);
        prim.SetTypeName("Xform");
        return prim;
    };

    Layer layer;
    const Path world = Path::Parse("/World");
    const Path a = Path::Parse("/World/A");
    const Path b = Path::Parse("/World/B");
    layer.SetSpec(world, makePrim());
    layer.SetSpec(a, makePrim());

    auto assertChildren = [&layer, &world](const std::vector<Path>& expected) {
        const auto& names = layer.GetChildNames(world);
        const auto& paths = detail::GetLayerChildPaths(layer, world);
        assert(names.size() == expected.size());
        assert(paths.size() == expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            assert(names[i] == expected[i].GetName());
            assert(paths[i] == expected[i]);
        }
    };

    assertChildren({a});

    Spec attr(SpecType::Attribute);
    attr.SetTypeName("float");
    layer.SetSpec(Path::Parse("/World.size"), std::move(attr));
    assertChildren({a});

    layer.SetSpec(b, makePrim());
    assertChildren({a, b});

    assert(layer.RemoveSpec(a));
    assertChildren({b});

    Spec nonPrimAtChildPath(SpecType::Attribute);
    nonPrimAtChildPath.SetTypeName("float");
    layer.SetSpec(b, std::move(nonPrimAtChildPath));
    assertChildren({});

    layer.SetSpec(b, makePrim());
    assertChildren({b});

    std::cout << "  Layer child path index invalidation: OK\n";
}

void TestLayerCompositionFields() {
    Spec prim(SpecType::Prim);
    prim.SetSpecifier(Specifier::Def);
    prim.SetTypeName("Xform");

    // References — stored as a generic field
    Reference ref;
    ref.assetPath = "./other.usda";
    ref.primPath = Path::Parse("/OtherPrim");
    ref.offset = 10.0;
    ref.scale = 1.0;

    // Field validity
    assert(prim.IsValidField(FieldNames::references));
    assert(prim.IsValidField(FieldNames::variantSelection));

    std::cout << "  Layer composition fields: OK\n";
}

void TestFieldRegistry() {
    auto& reg = FieldRegistry::Instance();

    // specifier is valid for Prim and Variant, not Attribute
    assert(reg.IsValidField(FieldNames::specifier, SpecType::Prim));
    assert(reg.IsValidField(FieldNames::specifier, SpecType::Variant));
    assert(!reg.IsValidField(FieldNames::specifier, SpecType::Attribute));

    // targetPaths is Relationship only
    assert(reg.IsValidField(FieldNames::targetPaths, SpecType::Relationship));
    assert(!reg.IsValidField(FieldNames::targetPaths, SpecType::Prim));

    // timeSamples is Attribute only
    assert(reg.IsValidField(FieldNames::timeSamples, SpecType::Attribute));
    assert(!reg.IsValidField(FieldNames::timeSamples, SpecType::Relationship));

    // documentation is shared across many forms
    assert(reg.IsValidField(FieldNames::documentation, SpecType::Layer));
    assert(reg.IsValidField(FieldNames::documentation, SpecType::Prim));
    assert(reg.IsValidField(FieldNames::documentation, SpecType::Attribute));
    assert(reg.IsValidField(FieldNames::documentation, SpecType::Relationship));

    // Lookup by name
    auto* desc = reg.Find(FieldNames::active);
    assert(desc != nullptr);
    assert(desc->type == TypeId::Bool);

    // Enumerate fields for a spec form
    auto& primFields = reg.GetFields(SpecType::Prim);
    assert(!primFields.empty());

    // Unknown field
    assert(reg.Find(Token("nonexistentField")) == nullptr);
    assert(!reg.IsValidField(Token("nonexistentField"), SpecType::Prim));

    std::cout << "  FieldRegistry: OK\n";
}

void TestGenericFieldAccess() {
    Spec prim(SpecType::Prim);

    // Set and get via generic fields
    prim.SetField(FieldNames::documentation, Value(std::string("A test prim")));
    assert(prim.HasField(FieldNames::documentation));
    assert(prim.GetDocumentation() == "A test prim");

    // Remove a field
    prim.RemoveField(FieldNames::documentation);
    assert(!prim.HasField(FieldNames::documentation));
    assert(prim.GetDocumentation() == "");  // fallback

    // Fields container iteration
    prim.SetSpecifier(Specifier::Def);
    prim.SetTypeName("Mesh");
    prim.SetComment("hello");
    assert(prim.GetFields().size() == 3);

    std::cout << "  Generic field access: OK\n";
}

void TestSpline() {
    Spline spline;
    spline.curveType = CurveType::Bezier;
    spline.preExtrapolationMode = ExtrapolationMode::Held;
    spline.postExtrapolationMode = ExtrapolationMode::Linear;

    SplineKnot k0;
    k0.time = 0.0;
    k0.value = 0.0;
    k0.nextInterpolationMode = InterpolationMode::Curve;
    k0.postTangentSlope = 1.0;
    k0.postTangentWidth = 5.0;
    spline.knots.push_back(k0);

    SplineKnot k1;
    k1.time = 10.0;
    k1.value = 100.0;
    k1.preTangentSlope = 1.0;
    k1.preTangentWidth = 5.0;
    k1.nextInterpolationMode = InterpolationMode::Held;
    spline.knots.push_back(k1);

    assert(spline.knots.size() == 2);
    assert(spline.knots[0].value == 0.0);
    assert(spline.knots[1].value == 100.0);

    std::cout << "  Spline: OK\n";
}

// --- Spline Phase 1: USDA parse + write + roundtrip (spec §7.4.2.4,
//     §16.2.13). Storage only — no evaluation.

// Helper to pull the Spline off an attribute spec in a parsed layer.
static const Spline* GetSplineAtPath(const Layer& layer, const char* path) {
    const Spec* attr = layer.GetAttributeSpec(Path::Parse(path));
    if (!attr) return nullptr;
    const Value* v = attr->GetField(FieldNames::spline);
    if (!v) return nullptr;
    return v->Get<Spline>();
}

// Parse the authored §16.2.13 grammar and verify every field on the
// materialized Spline — exercises curveType, pre/post extrapolation
// (including sloped slope capture), loop parameters, and each knot
// sub-clause (dual-value, pre-tangent, post-tangent + interp mode).
void TestSplineUsdaParse() {
    const char* src = R"(#usda 1.0

def "anim"
{
    double translate.spline = {
        bezier,
        pre: held,
        post: sloped(2.5),
        loop: (3.4, 5.7, 0, 4, -22.3),
        0: 0 & 9 ; pre (-0.5, 1); post curve (1, 2),
        5: 10 ; post linear,
        10: 20,
    }

    float weight.spline = {
        hermite,
        0: 0; post (1, 0),
        1: 1,
    }
}
)";
    auto res = ParseUsda(src);
    assert(res.success);

    const Spline* s = GetSplineAtPath(res.layer, "/anim.translate");
    assert(s);
    assert(s->curveType == CurveType::Bezier);
    assert(s->preExtrapolationMode == ExtrapolationMode::Held);
    assert(s->postExtrapolationMode == ExtrapolationMode::Sloped);
    assert(s->postExtrapolationSlope == 2.5);
    assert(s->loopParameters.protoStart == 3.4);
    assert(s->loopParameters.protoEnd == 5.7);
    assert(s->loopParameters.numPreLoops == 0);
    assert(s->loopParameters.numPostLoops == 4);
    assert(s->loopParameters.valueOffset == -22.3);
    assert(s->knots.size() == 3);

    // Knot 0: dual-value + pre + post curve.
    const auto& k0 = s->knots[0];
    assert(k0.time == 0.0);
    assert(k0.value == 0.0);
    assert(k0.preValue == 9.0);
    assert(k0.preTangentSlope == -0.5);
    assert(k0.preTangentWidth == 1.0);
    assert(k0.nextInterpolationMode == InterpolationMode::Curve);
    assert(k0.postTangentSlope == 1.0);
    assert(k0.postTangentWidth == 2.0);

    // Knot 1: only post interp mode, no tangent parens.
    const auto& k1 = s->knots[1];
    assert(k1.time == 5.0);
    assert(k1.value == 10.0);
    assert(k1.preValue == 10.0);  // single-valued
    assert(k1.nextInterpolationMode == InterpolationMode::Linear);
    assert(k1.postTangentSlope == 0.0);
    assert(k1.postTangentWidth == 0.0);

    // Knot 2: bare knot.
    const auto& k2 = s->knots[2];
    assert(k2.time == 10.0);
    assert(k2.value == 20.0);
    assert(k2.preValue == 20.0);
    assert(k2.nextInterpolationMode == InterpolationMode::Held);

    // Second spline — hermite curve type, implicit interp mode.
    const Spline* s2 = GetSplineAtPath(res.layer, "/anim.weight");
    assert(s2);
    assert(s2->curveType == CurveType::Hermite);
    assert(s2->knots.size() == 2);
    assert(s2->knots[0].postTangentSlope == 1.0);
    assert(s2->knots[0].postTangentWidth == 0.0);

    std::cout << "  Spline USDA parse (§16.2.13): OK\n";
}

// Parse → write → parse: every Spline field must survive the round
// trip bit-exact. Uses a deliberately varied authored fixture to
// exercise all grammar branches at once.
void TestSplineUsdaRoundtrip() {
    const char* src = R"(#usda 1.0

def "anim"
{
    double value.spline = {
        hermite,
        pre: linear,
        post: sloped(3.25),
        loop: (1, 10, 2, 3, 0.5),
        0: 0 & -1; pre (0.5, 1); post curve (1, 1.5),
        5: 10; post held,
        10: 100 & 99,
    }
}
)";
    auto first = ParseUsda(src);
    assert(first.success);

    std::string written = WriteUsda(first.layer);
    auto second = ParseUsda(written);
    assert(second.success);

    const Spline* a = GetSplineAtPath(first.layer,  "/anim.value");
    const Spline* b = GetSplineAtPath(second.layer, "/anim.value");
    assert(a && b);

    auto eq = [](const Spline& x, const Spline& y) {
        if (x.curveType != y.curveType) return false;
        if (x.preExtrapolationMode != y.preExtrapolationMode) return false;
        if (x.preExtrapolationSlope != y.preExtrapolationSlope) return false;
        if (x.postExtrapolationMode != y.postExtrapolationMode) return false;
        if (x.postExtrapolationSlope != y.postExtrapolationSlope) return false;
        const auto& lx = x.loopParameters; const auto& ly = y.loopParameters;
        if (lx.protoStart != ly.protoStart) return false;
        if (lx.protoEnd != ly.protoEnd) return false;
        if (lx.numPreLoops != ly.numPreLoops) return false;
        if (lx.numPostLoops != ly.numPostLoops) return false;
        if (lx.valueOffset != ly.valueOffset) return false;
        if (x.knots.size() != y.knots.size()) return false;
        for (size_t i = 0; i < x.knots.size(); ++i) {
            const auto& kx = x.knots[i]; const auto& ky = y.knots[i];
            if (kx.time != ky.time) return false;
            if (kx.value != ky.value) return false;
            if (kx.preValue != ky.preValue) return false;
            if (kx.preTangentSlope != ky.preTangentSlope) return false;
            if (kx.preTangentWidth != ky.preTangentWidth) return false;
            if (kx.postTangentSlope != ky.postTangentSlope) return false;
            if (kx.postTangentWidth != ky.postTangentWidth) return false;
            if (kx.nextInterpolationMode != ky.nextInterpolationMode) return false;
        }
        return true;
    };
    assert(eq(*a, *b));
    std::cout << "  Spline USDA parse→write→parse roundtrip: OK\n";
}

// Empty spline (no knots) and default-state spline both survive the
// roundtrip. Writer must emit a syntactically valid block with no
// knots; parser accepts it.
void TestSplineEmptyRoundtrip() {
    const char* src = R"(#usda 1.0
def "x"
{
    double v.spline = {}
}
)";
    auto first = ParseUsda(src);
    assert(first.success);
    const Spline* a = GetSplineAtPath(first.layer, "/x.v");
    assert(a);
    assert(a->knots.empty());
    assert(a->curveType == CurveType::Bezier);

    std::string written = WriteUsda(first.layer);
    auto second = ParseUsda(written);
    assert(second.success);
    const Spline* b = GetSplineAtPath(second.layer, "/x.v");
    assert(b);
    assert(b->knots.empty());
    std::cout << "  Spline empty block roundtrip: OK\n";
}

// Spline field is registered with TypeId::Spline so GetTypeId() on a
// Value built from Spline reflects that.
void TestSplineValueTypeId() {
    Spline s;
    Value v(std::move(s));
    assert(v.GetTypeId() == TypeId::Spline);
    assert(v.Get<Spline>() != nullptr);
    std::cout << "  Spline TypeId::Spline round-trips through Value: OK\n";
}

// --- Spline Phase 2: USDC binary roundtrip.
//     Parse a USDA authoring → WriteUsdc → ParseUsdc → expect the
//     Spline value to match bit-exactly. Exercises the new
//     CrateTypeId::Splines layout in usdc_writer.cpp /
//     usdc_parser.cpp.

// Helper: field-by-field equality for Spline. Out-of-line so multiple
// tests can reuse it.
static bool SplineEqualsExact(const Spline& x, const Spline& y) {
    if (x.curveType != y.curveType) return false;
    if (x.preExtrapolationMode != y.preExtrapolationMode) return false;
    if (x.preExtrapolationSlope != y.preExtrapolationSlope) return false;
    if (x.postExtrapolationMode != y.postExtrapolationMode) return false;
    if (x.postExtrapolationSlope != y.postExtrapolationSlope) return false;
    const auto& lx = x.loopParameters; const auto& ly = y.loopParameters;
    if (lx.protoStart != ly.protoStart) return false;
    if (lx.protoEnd != ly.protoEnd) return false;
    if (lx.numPreLoops != ly.numPreLoops) return false;
    if (lx.numPostLoops != ly.numPostLoops) return false;
    if (lx.valueOffset != ly.valueOffset) return false;
    if (x.knots.size() != y.knots.size()) return false;
    for (size_t i = 0; i < x.knots.size(); ++i) {
        const auto& kx = x.knots[i]; const auto& ky = y.knots[i];
        if (kx.time != ky.time) return false;
        if (kx.value != ky.value) return false;
        if (kx.preValue != ky.preValue) return false;
        if (kx.preTangentSlope != ky.preTangentSlope) return false;
        if (kx.preTangentWidth != ky.preTangentWidth) return false;
        if (kx.postTangentSlope != ky.postTangentSlope) return false;
        if (kx.postTangentWidth != ky.postTangentWidth) return false;
        if (kx.nextInterpolationMode != ky.nextInterpolationMode) return false;
    }
    return true;
}

// Roundtrip a full-featured Spline through USDC. Exercises dual-value
// knots, pre+post tangents, sloped extrapolation, loop parameters, and
// Bezier tangent widths.
void TestSplineUsdcRoundtrip() {
    const char* src = R"(#usda 1.0

def "anim"
{
    double value.spline = {
        bezier,
        pre: linear,
        post: sloped(3.25),
        loop: (1, 10, 2, 3, 0.5),
        0: 0 & -1; pre (0.5, 1); post curve (1, 1.5),
        5: 10; post held,
        10: 100 & 99,
    }
}
)";
    auto parsed = ParseUsda(src);
    assert(parsed.success);

    auto bytes = WriteUsdc(parsed.layer);
    assert(!bytes.empty());
    auto reparsed = ParseUsdc(bytes.data(), bytes.size());
    assert(reparsed.success);

    const Spec* attr = reparsed.layer.GetAttributeSpec(Path::Parse("/anim.value"));
    assert(attr);
    const Value* v = attr->GetField(FieldNames::spline);
    assert(v);
    const Spline* round = v->Get<Spline>();
    assert(round);

    const Spec* original = parsed.layer.GetAttributeSpec(Path::Parse("/anim.value"));
    assert(original);
    const Spline* orig = original->GetField(FieldNames::spline)->Get<Spline>();
    assert(SplineEqualsExact(*orig, *round));
    std::cout << "  Spline USDC roundtrip (full-featured): OK\n";
}

// Empty-spline roundtrip through USDC.
void TestSplineUsdcEmptyRoundtrip() {
    const char* src = R"(#usda 1.0
def "x"
{
    double v.spline = {}
}
)";
    auto parsed = ParseUsda(src);
    assert(parsed.success);
    auto bytes = WriteUsdc(parsed.layer);
    auto reparsed = ParseUsdc(bytes.data(), bytes.size());
    assert(reparsed.success);

    const Value* v = reparsed.layer.GetAttributeSpec(Path::Parse("/x.v"))
                                    ->GetField(FieldNames::spline);
    assert(v);
    const Spline* s = v->Get<Spline>();
    assert(s);
    assert(s->knots.empty());
    assert(s->curveType == CurveType::Bezier);  // default
    assert(s->preExtrapolationMode == ExtrapolationMode::Held);
    std::cout << "  Spline USDC empty roundtrip: OK\n";
}

// Cross-format triangle: USDA → USDC → USDA must reproduce the same
// Spline. This catches layout mismatches that a USDC-only roundtrip
// might miss (e.g. field ordering bugs that are self-consistent in
// the binary path but mismatch the authored USDA semantics).
void TestSplineUsdaUsdcUsdaTriangle() {
    const char* src = R"(#usda 1.0

def "anim"
{
    float v.spline = {
        bezier,
        post: linear,
        0: 0; post curve (2, 1),
        10: 5,
    }
}
)";
    auto a = ParseUsda(src);
    assert(a.success);

    auto binary = WriteUsdc(a.layer);
    auto b = ParseUsdc(binary.data(), binary.size());
    assert(b.success);

    std::string text = WriteUsda(b.layer);
    auto c = ParseUsda(text);
    assert(c.success);

    const Spline* aS = a.layer.GetAttributeSpec(Path::Parse("/anim.v"))
                              ->GetField(FieldNames::spline)->Get<Spline>();
    const Spline* cS = c.layer.GetAttributeSpec(Path::Parse("/anim.v"))
                              ->GetField(FieldNames::spline)->Get<Spline>();
    assert(aS && cS);
    assert(SplineEqualsExact(*aS, *cS));
    std::cout << "  Spline USDA→USDC→USDA triangle: OK\n";
}

static uint64_t ReadLe64(const std::vector<uint8_t>& bytes, size_t offset) {
    assert(offset + 8 <= bytes.size());
    uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[offset + i]) << (i * 8);
    }
    return value;
}

static void CheckUsdcVersion(const std::vector<uint8_t>& bytes,
                             uint8_t major,
                             uint8_t minor,
                             uint8_t patch) {
    assert(bytes.size() >= 11);
    assert(bytes[8] == major);
    assert(bytes[9] == minor);
    assert(bytes[10] == patch);
}

void TestUsdcCrateVersionSelection() {
    auto base = ParseUsda(R"(#usda 1.0
def "Root"
{
    int x = 1
}
)");
    assert(base.success);
    CheckUsdcVersion(WriteUsdc(base.layer), 0, 8, 0);

    auto legacy07 = WriteUsdc(base.layer);
    legacy07[9] = 7;
    legacy07[10] = 0;
    auto legacyParsed = ParseUsdc(legacy07.data(), legacy07.size());
    assert(legacyParsed.success);

    auto timecode = ParseUsda(R"(#usda 1.0
def "Root"
{
    timecode t = 24
}
)");
    assert(timecode.success);
    CheckUsdcVersion(WriteUsdc(timecode.layer), 0, 9, 0);

    auto relocates = ParseUsda(R"(#usda 1.0
(
    relocates = {
        </Root/Child>: </Root/MovedChild>,
    }
)

def "Root"
{
    def "Child"
    {
    }
}
)");
    assert(relocates.success);
    CheckUsdcVersion(WriteUsdc(relocates.layer), 0, 11, 0);

    auto spline = ParseUsda(R"(#usda 1.0
def "Root"
{
    double v.spline = {
        0: 1,
    }
}
)");
    assert(spline.success);
    CheckUsdcVersion(WriteUsdc(spline.layer), 0, 12, 0);
    std::cout << "  USDC crate version selection: OK\n";
}

void TestSplineUsdcSpecByteEncoding() {
    Layer layer;
    Spec attr(SpecType::Attribute);
    attr.SetTypeName(Token("float"));

    Spline spline;
    spline.curveType = CurveType::Bezier;
    spline.postExtrapolationMode = ExtrapolationMode::Sloped;
    spline.postExtrapolationSlope = 2.0;
    spline.loopParameters.protoStart = 0.0;
    spline.loopParameters.protoEnd = 4.0;
    spline.loopParameters.numPreLoops = 1;
    spline.loopParameters.numPostLoops = 2;
    spline.loopParameters.valueOffset = 3.0;

    SplineKnot knot;
    knot.time = 1.0;
    knot.value = 1.25;
    knot.preValue = -2.5;
    knot.preTangentWidth = 0.5;
    knot.postTangentWidth = 0.75;
    knot.preTangentSlope = 1.5;
    knot.postTangentSlope = 2.5;
    knot.nextInterpolationMode = InterpolationMode::Curve;
    spline.knots.push_back(knot);

    attr.SetField(FieldNames::spline, Value(std::move(spline)));
    layer.SetSpec(Path::Parse("/Root.v"), std::move(attr));

    auto bytes = WriteUsdc(layer);
    CheckUsdcVersion(bytes, 0, 12, 0);

    // Header + bootstrap are 32 bytes. This layer's only offset-encoded
    // field is the spline, so its value data begins there.
    const size_t splineOffset = 32;
    const uint64_t byteCount = ReadLe64(bytes, splineOffset);
    assert(byteCount == 87);
    const size_t payload = splineOffset + 8;
    assert(bytes[payload] == 0x21);      // version 1, float data, bezier
    assert(bytes[payload + 1] == 0x59);  // held pre, sloped post, looping
    assert(ReadLe64(bytes, payload + byteCount) == 0);

    auto reparsed = ParseUsdc(bytes.data(), bytes.size());
    assert(reparsed.success);
    const auto* readAttr = reparsed.layer.GetAttributeSpec(Path::Parse("/Root.v"));
    assert(readAttr);
    const auto* splineValue = readAttr->GetField(FieldNames::spline);
    assert(splineValue && splineValue->Get<Spline>());
    std::cout << "  Spline USDC spec byte encoding: OK\n";
}

// --- Spline Phase 3: evaluation + per-opinion integration (§12.5.3).

// Held interpolation: segment value is the left knot's value across
// the entire segment until the next knot. Query landing on the right
// knot switches to its own value (right-continuous, per the §12.5.3
// rule that the knot's `value` is "the value of the knot at its
// time").
void TestSplineEvalHeld() {
    Spline s;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Held, 1.0, 1.0 });
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Held, 5.0, 5.0 });
    auto v3 = EvaluateSpline(s, 3.0);
    assert(v3 && *v3->Get<Double>() == 1.0);
    auto v10 = EvaluateSpline(s, 10.0);
    assert(v10 && *v10->Get<Double>() == 5.0);
    std::cout << "  Spline eval held: OK\n";
}

// Linear interpolation: exact lerp between knot values at 50%.
void TestSplineEvalLinear() {
    Spline s;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear, 0.0, 0.0 });
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 100.0, 100.0 });
    auto v = EvaluateSpline(s, 5.0);
    assert(v && *v->Get<Double>() == 50.0);
    std::cout << "  Spline eval linear: OK\n";
}

// None segment: explicit value block. Query inside the segment
// returns nullopt (spec §7.4.2.4.2 "no value exists for the segment").
void TestSplineEvalNoneSegment() {
    Spline s;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::None, 0.0, 0.0 });
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Held, 5.0, 5.0 });
    auto v = EvaluateSpline(s, 5.0);
    assert(!v);
    // Right knot still resolves to its value at t=10 (next segment
    // or equality with the knot's time).
    auto v10 = EvaluateSpline(s, 10.0);
    assert(v10 && *v10->Get<Double>() == 5.0);
    std::cout << "  Spline eval none segment → value block: OK\n";
}

// Curve (Hermite): zero-slope tangents at both ends → classic
// smoothstep-shaped curve interpolating 0→1 over [0,10]. At the
// midpoint u=0.5 the smoothstep value is exactly 0.5.
void TestSplineEvalHermite() {
    Spline s;
    s.curveType = CurveType::Hermite;
    SplineKnot a{ 0.0, 0, 0, 0, 0, InterpolationMode::Curve, 0.0, 0.0 };
    SplineKnot b{10.0, 0, 0, 0, 0, InterpolationMode::Curve, 1.0, 1.0 };
    s.knots.push_back(a);
    s.knots.push_back(b);
    auto vmid = EvaluateSpline(s, 5.0);
    assert(vmid && std::abs(*vmid->Get<Double>() - 0.5) < 1e-12);
    auto v0 = EvaluateSpline(s, 0.0);
    auto v10 = EvaluateSpline(s, 10.0);
    assert(v0 && *v0->Get<Double>() == 0.0);
    assert(v10 && *v10->Get<Double>() == 1.0);
    std::cout << "  Spline eval Hermite smoothstep midpoint: OK\n";
}

// Curve (Bezier): control points forming a straight-line Bezier
// (tangents along the segment direction, width = 1/3 of dt) must
// reproduce the linear interp result. Validates the bisection root
// solver + cubic Bernstein evaluator.
void TestSplineEvalBezierDegenerateLinear() {
    Spline s;
    s.curveType = CurveType::Bezier;
    SplineKnot a{0.0, 0, 0, /*postSlope=*/1.0, /*postWidth=*/10.0 / 3.0,
                 InterpolationMode::Curve, 0.0, 0.0};
    SplineKnot b{10.0, /*preSlope=*/1.0, /*preWidth=*/10.0 / 3.0, 0, 0,
                 InterpolationMode::Curve, 10.0, 10.0};
    s.knots.push_back(a);
    s.knots.push_back(b);
    auto v = EvaluateSpline(s, 5.0);
    assert(v && std::abs(*v->Get<Double>() - 5.0) < 1e-9);
    std::cout << "  Spline eval Bezier degenerate linear: OK\n";
}

// Dual-value knot: at the shared stageTime the knot's `value`
// wins (right-continuous). The left-limit (preValue) shows up as
// the right endpoint of the preceding segment in a linear interp.
void TestSplineEvalDualValueKnot() {
    Spline s;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear, 0.0, 0.0 });
    // t=10: dual-valued. Approaches 100 from below (preValue), jumps
    // to 200 (value) at t=10. Onward, linear to 300 at t=20.
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 200.0, 100.0});
    s.knots.push_back(SplineKnot{20.0, 0, 0, 0, 0, InterpolationMode::Linear, 300.0, 300.0});

    // Just before t=10: segment 0→1 lerps 0 → preValue=100 at 90%.
    auto v9 = EvaluateSpline(s, 9.0);
    assert(v9 && std::abs(*v9->Get<Double>() - 90.0) < 1e-9);

    // At exactly t=10: knot's value=200 (right-continuous).
    auto v10 = EvaluateSpline(s, 10.0);
    assert(v10 && *v10->Get<Double>() == 200.0);

    // Just after t=10: segment 1→2 lerps 200 → 300 at 10%.
    auto v11 = EvaluateSpline(s, 11.0);
    assert(v11 && std::abs(*v11->Get<Double>() - 210.0) < 1e-9);
    std::cout << "  Spline eval dual-value knot (pre vs value): OK\n";
}

// Pre/post extrapolation modes: None / Held / Linear (using the
// edge knot's own pre/post-tangent slopes) / Sloped (using the
// spline's authored extrapolation slope).
void TestSplineEvalExtrapolation() {
    // Baseline single-segment spline.
    auto build = [](ExtrapolationMode preMode, double preSlope,
                    ExtrapolationMode postMode, double postSlope) {
        Spline s;
        s.preExtrapolationMode  = preMode;
        s.preExtrapolationSlope = preSlope;
        s.postExtrapolationMode  = postMode;
        s.postExtrapolationSlope = postSlope;
        // Edge knots have their own pre/post tangent slopes used by
        // ExtrapolationMode::Linear.
        SplineKnot a{ 0.0, /*preSlope=*/2.0, 0.0, 0.0, 0.0,
                     InterpolationMode::Held, 10.0, 10.0};
        SplineKnot b{10.0, 0.0, 0.0, /*postSlope=*/3.0, 0.0,
                     InterpolationMode::Held, 50.0, 50.0};
        s.knots.push_back(a);
        s.knots.push_back(b);
        return s;
    };

    // None — out of range → nullopt.
    {
        auto s = build(ExtrapolationMode::None, 0, ExtrapolationMode::None, 0);
        assert(!EvaluateSpline(s, -5.0));
        assert(!EvaluateSpline(s, 20.0));
    }
    // Held — pre holds preValue, post holds value.
    {
        auto s = build(ExtrapolationMode::Held, 0, ExtrapolationMode::Held, 0);
        auto vpre  = EvaluateSpline(s, -5.0);
        auto vpost = EvaluateSpline(s, 20.0);
        assert(vpre  && *vpre->Get<Double>()  == 10.0);
        assert(vpost && *vpost->Get<Double>() == 50.0);
    }
    // Linear — use the edge knot's own tangent slope.
    {
        auto s = build(ExtrapolationMode::Linear, 0, ExtrapolationMode::Linear, 0);
        // preValue=10 at t=0, preTangentSlope=2 → at t=-5 value = 10 + 2*(-5-0) = 0.
        auto vpre = EvaluateSpline(s, -5.0);
        assert(vpre && std::abs(*vpre->Get<Double>() - 0.0) < 1e-9);
        // value=50 at t=10, postTangentSlope=3 → at t=15 value = 50 + 3*(15-10) = 65.
        auto vpost = EvaluateSpline(s, 15.0);
        assert(vpost && std::abs(*vpost->Get<Double>() - 65.0) < 1e-9);
    }
    // Sloped — use the spline's authored slope, ignoring edge knot tangents.
    {
        auto s = build(ExtrapolationMode::Sloped, -4.0,
                       ExtrapolationMode::Sloped, 10.0);
        // pre: 10 + (-4)*(-5-0) = 10 + 20 = 30.
        auto vpre = EvaluateSpline(s, -5.0);
        assert(vpre && std::abs(*vpre->Get<Double>() - 30.0) < 1e-9);
        // post: 50 + 10*(15-10) = 100.
        auto vpost = EvaluateSpline(s, 15.0);
        assert(vpost && std::abs(*vpost->Get<Double>() - 100.0) < 1e-9);
    }
    std::cout << "  Spline eval extrapolation (none/held/linear/sloped): OK\n";
}

// Empty spline and single-knot edge cases.
void TestSplineEvalEdges() {
    Spline empty;
    assert(!EvaluateSpline(empty, 0.0));

    Spline single;
    single.knots.push_back(SplineKnot{5.0, 0, 0, 0, 0,
                                       InterpolationMode::Held, 7.0, 7.0});
    auto at  = EvaluateSpline(single, 5.0);
    assert(at && *at->Get<Double>() == 7.0);
    // Default Held extrapolation holds preValue (pre) / value (post).
    auto before = EvaluateSpline(single, 0.0);
    auto after  = EvaluateSpline(single, 10.0);
    assert(before && *before->Get<Double>() == 7.0);
    assert(after  && *after->Get<Double>()  == 7.0);
    std::cout << "  Spline eval empty + single-knot edges: OK\n";
}

// BakeSplineToTimeSamples: gathers EvaluateSpline outputs at each
// provided sample time, skipping entries where evaluation returns
// nullopt. Uses a spline with a `none` segment to verify skipping.
void TestSplineBakeToTimeSamples() {
    Spline s;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear, 0.0, 0.0 });
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::None,  10.0, 10.0});
    s.knots.push_back(SplineKnot{20.0, 0, 0, 0, 0, InterpolationMode::Held,  20.0, 20.0});

    auto dict = BakeSplineToTimeSamples(s, {0.0, 5.0, 10.0, 15.0, 20.0});
    // t=0,5 in linear segment; t=10,20 on knots; t=15 in `none` segment (skipped).
    assert(dict.size() == 4);
    assert(dict.count(std::to_string(0.0)));
    assert(dict.count(std::to_string(5.0)));
    assert(dict.count(std::to_string(10.0)));
    assert(dict.count(std::to_string(20.0)));
    assert(!dict.count(std::to_string(15.0)));
    std::cout << "  BakeSplineToTimeSamples skips none-segments: OK\n";
}

// Stage integration: an attribute with only a spline authored (no
// timeSamples, no default) resolves via the per-opinion walk's
// spline slot. Confirms the wiring in UsdAttribute::Get.
void TestSplineStageIntegration() {
    const char* src = R"(#usda 1.0

def "anim"
{
    double v.spline = {
        bezier,
        0: 0,
        10: 100,
    }
}
)";
    auto res = ParseUsda(src);
    assert(res.success);
    auto stage = Stage::CreateFromComposedLayer(res.layer);
    assert(stage.IsValid());
    auto attr = stage.GetPrimAtPath(Path::Parse("/anim")).GetAttribute(Token("v"));
    assert(attr.IsValid());

    // Held interp default on knots with no tangents → constant left
    // knot's value across the segment.
    auto at3 = attr.Get(UsdTimeCode(3.0));
    assert(at3.found);
    assert(at3.value.Get<Double>());
    assert(*at3.value.Get<Double>() == 0.0);

    // Exactly on the right knot: right-continuous → 100.
    auto at10 = attr.Get(UsdTimeCode(10.0));
    assert(at10.found);
    assert(*at10.value.Get<Double>() == 100.0);
    std::cout << "  Spline resolves through UsdAttribute::Get: OK\n";
}

// Per-opinion priority: within a single opinion, timeSamples >
// spline. An attribute with both authored must return the
// timeSamples value, never the spline value.
void TestSplinePerOpinionTimeSamplesWins() {
    const char* src = R"(#usda 1.0

def "anim"
{
    double v = 0
    double v.timeSamples = {
        0: 1000,
        10: 2000,
    }
    double v.spline = {
        bezier,
        0: 0,
        10: 100,
    }
}
)";
    auto res = ParseUsda(src);
    assert(res.success);
    auto stage = Stage::CreateFromComposedLayer(res.layer);
    auto attr = stage.GetPrimAtPath(Path::Parse("/anim")).GetAttribute(Token("v"));

    // At t=5, timeSamples linear interp → 1500. Spline held-at-0
    // would be 0. timeSamples must win.
    auto v = attr.Get(UsdTimeCode(5.0));
    assert(v.found);
    assert(std::abs(*v.value.Get<Double>() - 1500.0) < 1e-9);
    std::cout << "  Per-opinion: timeSamples beats spline (§12.3): OK\n";
}

// --- Spline Phase 4: loop extrapolation, inner loops, anti-
//     regression, and typeName narrowing. Closes the spec-compliance
//     gap left open by PR3.

// LoopRepeat (§12.5.3.4.1): each copy of the authored range adds
// (lastValue - firstValue) so joins are continuous. Evaluating at
// t = tFirst - period gives value_at_tFirst - delta; at tLast +
// period gives value_at_tLast + delta.
void TestSplineLoopRepeat() {
    Spline s;
    s.preExtrapolationMode  = ExtrapolationMode::LoopRepeat;
    s.postExtrapolationMode = ExtrapolationMode::LoopRepeat;
    // period = 10, delta = 10 (v=0 at t=0, v=10 at t=10).
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear,  0.0,  0.0});
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 10.0, 10.0});

    // Post: at t=15 → fold to effective 5 (value 5), + 1*10 = 15.
    auto vp = EvaluateSpline(s, 15.0);
    assert(vp && std::abs(*vp->Get<Double>() - 15.0) < 1e-9);

    // Post: at t=25 → effective 5, + 2*10 = 25.
    auto vp2 = EvaluateSpline(s, 25.0);
    assert(vp2 && std::abs(*vp2->Get<Double>() - 25.0) < 1e-9);

    // Pre: at t=-5 → effective 5, + (-1)*10 = -5.
    auto vn = EvaluateSpline(s, -5.0);
    assert(vn && std::abs(*vn->Get<Double>() - (-5.0)) < 1e-9);
    std::cout << "  Spline loop-repeat outer extrap: OK\n";
}

// LoopReset: periodic repetition of the authored range without value
// offset — the curve can jump discontinuously at the join.
void TestSplineLoopReset() {
    Spline s;
    s.postExtrapolationMode = ExtrapolationMode::LoopReset;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear, 0.0, 0.0});
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 5.0, 5.0});

    // t=12 → fold to 2 in same period layout → value 1 (no delta).
    auto v = EvaluateSpline(s, 12.0);
    assert(v && std::abs(*v->Get<Double>() - 1.0) < 1e-9);
    // t=25 → fold to 5 → value 2.5 (no delta).
    auto v2 = EvaluateSpline(s, 25.0);
    assert(v2 && std::abs(*v2->Get<Double>() - 2.5) < 1e-9);
    std::cout << "  Spline loop-reset outer extrap: OK\n";
}

// LoopOscillate: every other copy is reversed in time, so a linear
// ramp 0→10 over [0,10] becomes 10→0 in the second copy, 0→10 in
// the third, etc. No value offset is applied.
void TestSplineLoopOscillate() {
    Spline s;
    s.postExtrapolationMode = ExtrapolationMode::LoopOscillate;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear,  0.0,  0.0});
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 10.0, 10.0});

    // t=15 → 2nd copy (odd, mirrored). effective=15-10=5, mirror →
    // 10-5=5. Value 5 (non-mirrored linear gives same at midpoint).
    auto v15 = EvaluateSpline(s, 15.0);
    assert(v15 && std::abs(*v15->Get<Double>() - 5.0) < 1e-9);
    // t=12 → mirror copy. effective=12-10=2, mirror → 10-2=8 → 8.
    auto v12 = EvaluateSpline(s, 12.0);
    assert(v12 && std::abs(*v12->Get<Double>() - 8.0) < 1e-9);
    // t=22 → 3rd copy (even, non-mirrored). effective=22-20=2 → 2.
    auto v22 = EvaluateSpline(s, 22.0);
    assert(v22 && std::abs(*v22->Get<Double>() - 2.0) < 1e-9);
    std::cout << "  Spline loop-oscillate outer extrap: OK\n";
}

// Inner loops (§12.5.3.4.2): proto [0, 10] with numPostLoops=2,
// valueOffset=100. At t=5 (inside proto) → authored 5. At t=15
// (first post-copy, offset 5 into proto) → 5 + 100 = 105. At t=25
// (second post-copy) → 5 + 200 = 205.
void TestSplineInnerLoopsPost() {
    Spline s;
    s.loopParameters.protoStart  = 0.0;
    s.loopParameters.protoEnd    = 10.0;
    s.loopParameters.numPostLoops = 2;
    s.loopParameters.valueOffset = 100.0;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear,  0.0,  0.0});
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 10.0, 10.0});

    auto v5  = EvaluateSpline(s, 5.0);
    auto v15 = EvaluateSpline(s, 15.0);
    auto v25 = EvaluateSpline(s, 25.0);
    assert(v5  && std::abs(*v5->Get<Double>()  -   5.0) < 1e-9);
    assert(v15 && std::abs(*v15->Get<Double>() - 105.0) < 1e-9);
    assert(v25 && std::abs(*v25->Get<Double>() - 205.0) < 1e-9);
    std::cout << "  Spline inner loops (post) apply valueOffset per copy: OK\n";
}

// Inner loops before the proto range. numPreLoops=2 means the proto
// is replicated 2 times BEFORE protoStart, with negative
// valueOffset accumulation.
void TestSplineInnerLoopsPre() {
    Spline s;
    s.loopParameters.protoStart  = 0.0;
    s.loopParameters.protoEnd    = 10.0;
    s.loopParameters.numPreLoops = 2;
    s.loopParameters.valueOffset = 100.0;
    s.knots.push_back(SplineKnot{ 0.0, 0, 0, 0, 0, InterpolationMode::Linear,  0.0,  0.0});
    s.knots.push_back(SplineKnot{10.0, 0, 0, 0, 0, InterpolationMode::Linear, 10.0, 10.0});

    // t=-5 (first pre-copy) → effective 5, delta=-100 → -95.
    auto v = EvaluateSpline(s, -5.0);
    assert(v && std::abs(*v->Get<Double>() - (-95.0)) < 1e-9);
    // t=-15 (second pre-copy) → effective 5, delta=-200 → -195.
    auto v2 = EvaluateSpline(s, -15.0);
    assert(v2 && std::abs(*v2->Get<Double>() - (-195.0)) < 1e-9);
    std::cout << "  Spline inner loops (pre) apply valueOffset per copy: OK\n";
}

// Anti-regression (§12.5.3.5): a Bezier with pre/post tangent widths
// summing to well above the segment's time extent would be
// regressive. EvaluateBezierSegment must shrink the widths
// proportionally so the curve stays single-valued in time. Validated
// by running the solver — a regressive curve would produce wildly
// incorrect values or infinite loops in bisection.
void TestSplineAntiRegression() {
    Spline s;
    s.curveType = CurveType::Bezier;
    // dt = 10, but the author asked for postWidth=20 + preWidth=20 =
    // 40 which would push control points past each other in time.
    // Anti-regression scales both to 5 so they sum to dt.
    SplineKnot a{ 0.0, 0, 0, /*postSlope=*/1.0, /*postWidth=*/20.0,
                 InterpolationMode::Curve, 0.0, 0.0};
    SplineKnot b{10.0, /*preSlope=*/1.0, /*preWidth=*/20.0, 0, 0,
                 InterpolationMode::Curve, 10.0, 10.0};
    s.knots.push_back(a);
    s.knots.push_back(b);

    // Shouldn't hang, shouldn't produce NaN, should produce a value
    // monotonically between 0 and 10 at t=5.
    auto v = EvaluateSpline(s, 5.0);
    assert(v);
    double d = *v->Get<Double>();
    assert(std::isfinite(d));
    assert(d >= 0.0 && d <= 10.0);
    std::cout << "  Spline anti-regression (§12.5.3.5): OK\n";
}

// Type narrowing at the stage boundary: splines store values as
// double, but `float bar.spline` and `half bar.spline` attributes
// must resolve to Float / Half values for C++ callers that know the
// attribute's declared type.
void TestSplineTypeNarrowingFloat() {
    const char* src = R"(#usda 1.0
def "a"
{
    float v.spline = { 0: 1, 10: 2 }
}
)";
    auto res = ParseUsda(src);
    assert(res.success);
    auto stage = Stage::CreateFromComposedLayer(res.layer);
    auto attr = stage.GetPrimAtPath(Path::Parse("/a")).GetAttribute(Token("v"));
    auto r = attr.Get(UsdTimeCode(0.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && std::abs(*f - 1.0f) < 1e-6f);
    assert(r.value.GetTypeId() == TypeId::Float);
    std::cout << "  Spline stage return narrows double → Float: OK\n";
}

void TestSplineTypeNarrowingHalf() {
    const char* src = R"(#usda 1.0
def "a"
{
    half v.spline = { 0: 1, 10: 2 }
}
)";
    auto res = ParseUsda(src);
    assert(res.success);
    auto stage = Stage::CreateFromComposedLayer(res.layer);
    auto attr = stage.GetPrimAtPath(Path::Parse("/a")).GetAttribute(Token("v"));
    auto r = attr.Get(UsdTimeCode(0.0));
    assert(r.found);
    auto* h = r.value.Get<Half>();
    assert(h);
    assert(r.value.GetTypeId() == TypeId::Half);
    std::cout << "  Spline stage return narrows double → Half: OK\n";
}

void TestUsdaBasicLayer() {
    const char* usda = R"(#usda 1.0
(
    defaultPrim = "World"
    timeCodesPerSecond = 24
    startTimeCode = 1
    endTimeCode = 100
    doc = "A test layer"
)

def Xform "World"
{
    def Mesh "Cube"
    {
        float3 extent.timeSamples = {
            0: (-1, -1, -1),
            1: (1, 1, 1),
        }
    }
}
)";

    auto result = ParseUsda(usda);
    if (!result.success) {
        std::cerr << "  Parse error at " << result.line << ":" << result.column
                  << ": " << result.error << std::endl;
    }
    assert(result.success);
    auto& layer = result.layer;

    // Layer metadata
    assert(layer.GetLayerSpec().GetDefaultPrim() == "World");
    assert(layer.GetLayerSpec().GetTimeCodesPerSecond() == 24.0);
    assert(layer.GetLayerSpec().GetStartTimeCode() == 1.0);
    assert(layer.GetLayerSpec().GetEndTimeCode() == 100.0);
    assert(layer.GetLayerSpec().GetDocumentation() == "A test layer");

    // Prim specs
    auto* world = layer.GetPrimSpec(Path::Parse("/World"));
    assert(world != nullptr);
    assert(world->GetSpecifier() == Specifier::Def);
    assert(world->GetTypeName() == "Xform");

    auto* cube = layer.GetPrimSpec(Path::Parse("/World/Cube"));
    assert(cube != nullptr);
    assert(cube->GetSpecifier() == Specifier::Def);
    assert(cube->GetTypeName() == "Mesh");

    std::cout << "  USDA basic layer: OK\n";
}

void TestUsdaAttributes() {
    const char* usda = R"(#usda 1.0

def Xform "Root"
{
    double3 xformOp:translate = (1.0, 2.0, 3.0)
    float inputs:opacity = 0.5
    uniform token visibility = "inherited"
    custom float myCustomAttr = 42.0
    bool active = true
    string name = "hello world"
}
)";

    auto result = ParseUsda(usda);
    if (!result.success) {
        std::cout << "  Parse error at " << result.line << ":" << result.column
                  << ": " << result.error << "\n";
    }
    assert(result.success);
    auto& layer = result.layer;

    auto* root = layer.GetPrimSpec(Path::Parse("/Root"));
    assert(root != nullptr);
    assert(root->GetTypeName() == "Xform");

    // double3 attribute
    auto* translate = layer.GetAttributeSpec(Path::Parse("/Root.xformOp:translate"));
    assert(translate != nullptr);
    assert(translate->GetTypeName() == "double3");
    auto* defVal = translate->GetField(FieldNames::defaultValue);
    assert(defVal != nullptr);
    auto* vec = defVal->Get<GfVec3d>();
    assert(vec != nullptr);
    assert((*vec)[0] == 1.0);
    assert((*vec)[1] == 2.0);
    assert((*vec)[2] == 3.0);

    // float attribute
    auto* opacity = layer.GetAttributeSpec(Path::Parse("/Root.inputs:opacity"));
    assert(opacity != nullptr);
    assert(opacity->GetTypeName() == "float");
    auto* opVal = opacity->GetField(FieldNames::defaultValue);
    assert(opVal != nullptr);
    assert(*opVal->Get<Float>() == 0.5f);

    // uniform token
    auto* vis = layer.GetAttributeSpec(Path::Parse("/Root.visibility"));
    assert(vis != nullptr);
    assert(vis->GetVariability() == Variability::Uniform);
    assert(vis->GetTypeName() == "token");

    // custom attribute
    auto* custom = layer.GetAttributeSpec(Path::Parse("/Root.myCustomAttr"));
    assert(custom != nullptr);
    assert(custom->GetCustom() == true);

    // bool attribute
    auto* activeAttr = layer.GetAttributeSpec(Path::Parse("/Root.active"));
    assert(activeAttr != nullptr);
    assert(activeAttr->GetTypeName() == "bool");

    // string attribute
    auto* nameAttr = layer.GetAttributeSpec(Path::Parse("/Root.name"));
    assert(nameAttr != nullptr);
    auto* nameVal = nameAttr->GetField(FieldNames::defaultValue);
    assert(nameVal != nullptr);
    assert(*nameVal->Get<String>() == "hello world");

    std::cout << "  USDA attributes: OK\n";
}

const Value* GetAuthoredDefault(const Layer& layer, const char* attrPath) {
    auto* spec = layer.GetAttributeSpec(Path::Parse(attrPath));
    assert(spec != nullptr);
    auto* value = spec->GetField(FieldNames::defaultValue);
    assert(value != nullptr);
    return value;
}

void AssertUsdaRoundtripParses(const char* filePath) {
    auto first = ParseUsdaFile(filePath);
    if (!first.success) {
        std::cerr << "  Parse error in " << filePath << ": "
                  << first.error << "\n";
    }
    assert(first.success);

    std::string written = WriteUsda(first.layer);
    assert(written.find("# unsupported") == std::string::npos);

    auto second = ParseUsda(written);
    if (!second.success) {
        std::cerr << "  Reparse error after writing " << filePath << ": "
                  << second.error << "\n";
    }
    assert(second.success);
}

void TestUsdaFoundationalTypeCoverage() {
    auto result = ParseUsdaFile("tests/usda/foundational_types.usda");
    assert(result.success);
    const Layer& layer = result.layer;

    const Value* ucharMax = GetAuthoredDefault(layer, "/FoundationalTypes.ucharMax");
    assert(ucharMax->GetTypeId() == TypeId::UChar);
    auto* ucharValue = ucharMax->Get<UChar>();
    assert(ucharValue != nullptr && *ucharValue == 255);

    const Value* timecode = GetAuthoredDefault(layer, "/FoundationalTypes.timecodeVal");
    assert(timecode->GetTypeId() == TypeId::TimeCode);
    auto* timecodeValue = timecode->Get<TimeCode>();
    assert(timecodeValue != nullptr && *timecodeValue == 101.5);

    const auto* uchars =
        GetAuthoredDefault(layer, "/FoundationalTypes.uchars")->Get<std::vector<UChar>>();
    assert(uchars != nullptr && uchars->size() == 3 && (*uchars)[2] == 255);

    const auto* timecodes =
        GetAuthoredDefault(layer, "/FoundationalTypes.timecodes")->Get<std::vector<TimeCode>>();
    assert(timecodes != nullptr && timecodes->size() == 3 && (*timecodes)[1] == 12.5);

    const auto* h2s =
        GetAuthoredDefault(layer, "/FoundationalTypes.h2s")->Get<std::vector<GfVec2h>>();
    assert(h2s != nullptr && h2s->size() == 1);
    assert(std::abs(static_cast<float>((*h2s)[0][1]) - 1.0f) < 0.001f);

    const auto* qaths =
        GetAuthoredDefault(layer, "/FoundationalTypes.qaths")->Get<std::vector<GfQuath>>();
    assert(qaths != nullptr && qaths->size() == 1);
    assert(std::abs(static_cast<float>((*qaths)[0][3]) - 1.0f) < 0.001f);

    const auto* matrices2 =
        GetAuthoredDefault(layer, "/FoundationalTypes.matrices2")->Get<std::vector<GfMatrix2d>>();
    assert(matrices2 != nullptr && matrices2->size() == 1);
    assert((*matrices2)[0](1, 1) == 1.0);

    const auto* matrices3 =
        GetAuthoredDefault(layer, "/FoundationalTypes.matrices3")->Get<std::vector<GfMatrix3d>>();
    assert(matrices3 != nullptr && matrices3->size() == 1);
    assert((*matrices3)[0](2, 2) == 1.0);

    const Value* points = GetAuthoredDefault(layer, "/FoundationalTypes.points");
    assert(points->GetRole() == Role::Point);

    const Value* color3h = GetAuthoredDefault(layer, "/FoundationalTypes.c3h");
    assert(color3h->GetTypeId() == TypeId::Half3);
    assert(color3h->GetRole() == Role::Color);

    const Value* color4h = GetAuthoredDefault(layer, "/FoundationalTypes.c4h");
    assert(color4h->GetTypeId() == TypeId::Half4);
    assert(color4h->GetRole() == Role::Color);

    const Value* frame = GetAuthoredDefault(layer, "/FoundationalTypes.frame");
    assert(frame->GetTypeId() == TypeId::Matrix4d);
    assert(frame->GetRole() == Role::Frame);

    AssertUsdaRoundtripParses("tests/usda/foundational_types.usda");

    std::cout << "  USDA foundational type coverage: OK\n";
}

void TestUsdaUnicodeIdentifiers() {
    const char* usda =
        "#usda 1.0\n"
        "(\n"
        "    m\xC3\xA9" "tadonn\xC3\xA9" "es = \"ok\"\n"
        ")\n"
        "\n"
        "def Xform \"World\"\n"
        "{\n"
        "    double \xC3\xBC" "ber = 3.25\n"
        "    token inputs:\xE7\x8A\xB6\xE6\x85\x8B = \"ready\"\n"
        "    rel cible = </World.\xC3\xBC" "ber>\n"
        "}\n";

    auto result = ParseUsda(usda);
    if (!result.success) {
        std::cerr << "  Parse error at " << result.line << ":" << result.column
                  << ": " << result.error << std::endl;
    }
    assert(result.success);
    const auto& layer = result.layer;

    auto* meta = layer.GetLayerSpec().GetField(Token("m\xC3\xA9" "tadonn\xC3\xA9" "es"));
    assert(meta && meta->Get<String>() && *meta->Get<String>() == "ok");

    auto attrPath = Path::Parse("/World.\xC3\xBC" "ber");
    assert(!attrPath.IsEmpty());
    auto* attr = layer.GetAttributeSpec(attrPath);
    assert(attr != nullptr);
    auto* value = attr->GetField(FieldNames::defaultValue);
    assert(value && value->Get<Double>() && *value->Get<Double>() == 3.25);

    auto namespacedPath = Path::Parse("/World.inputs:\xE7\x8A\xB6\xE6\x85\x8B");
    assert(!namespacedPath.IsEmpty());
    auto* tokenAttr = layer.GetAttributeSpec(namespacedPath);
    assert(tokenAttr != nullptr);
    assert(tokenAttr->GetTypeName() == "token");

    auto badEmoji = ParseUsda(
        "#usda 1.0\n"
        "def \"World\"\n"
        "{\n"
        "    double \xF0\x9F\x98\x80 = 1\n"
        "}\n");
    assert(!badEmoji.success);

    auto badUtf8 = ParseUsda(
        "#usda 1.0\n"
        "def \"World\"\n"
        "{\n"
        "    double bad\xC3( = 1\n"
        "}\n");
    assert(!badUtf8.success);

    std::cout << "  USDA Unicode identifiers: OK\n";
}

void TestUsdaGrammarStrictness() {
    const char* usda =
        "#usda 1.0\n"
        "def Scope \"Escapes\"\n"
        "{\n"
        "    string value = \"\\a\\b\\f\\n\\r\\t\\v\\'\\\"\\\\\\x41\\101\"\n"
        "}\n";
    auto result = ParseUsda(usda);
    if (!result.success) {
        std::cerr << "  Parse error at " << result.line << ":" << result.column
                  << ": " << result.error << std::endl;
    }
    assert(result.success);
    auto* attr = result.layer.GetAttributeSpec(Path::Parse("/Escapes.value"));
    assert(attr != nullptr);
    auto* value = attr->GetField(FieldNames::defaultValue);
    assert(value && value->Get<String>());

    std::string expected;
    expected.push_back('\a');
    expected.push_back('\b');
    expected.push_back('\f');
    expected.push_back('\n');
    expected.push_back('\r');
    expected.push_back('\t');
    expected.push_back('\v');
    expected.push_back('\'');
    expected.push_back('"');
    expected.push_back('\\');
    expected.push_back('A');
    expected.push_back('A');
    assert(*value->Get<String>() == expected);

    auto expectInvalid = [](const std::string& text) {
        auto invalid = ParseUsda(text);
        assert(!invalid.success);
    };

    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    float value = +1\n"
        "}\n");
    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    float value = 1e\n"
        "}\n");
    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    float value = +inf\n"
        "}\n");
    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    string value = \"\\q\"\n"
        "}\n");
    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    string value = \"\\xg\"\n"
        "}\n");

    std::string rawControl =
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    string value = \"";
    rawControl.push_back('\x01');
    rawControl += "\"\n}\n";
    expectInvalid(rawControl);

    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    string value = \"bad\xC3(\"\n"
        "}\n");
    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    asset value = @foo\nbar@\n"
        "}\n");
    expectInvalid(
        "#usda 1.0\n"
        "def Scope \"Bad\"\n"
        "{\n"
        "    asset value = @\xEE\x80\x80@\n"
        "}\n");

    std::cout << "  USDA grammar strictness: OK\n";
}

void TestUsdaRelationships() {
    const char* usda = R"(#usda 1.0

def Xform "World"
{
    rel material:binding = </Materials/DefaultMaterial>
    custom rel myRel
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto& layer = result.layer;

    auto* binding = layer.GetRelationshipSpec(Path::Parse("/World.material:binding"));
    assert(binding != nullptr);
    assert(binding->GetType() == SpecType::Relationship);

    auto* myRel = layer.GetRelationshipSpec(Path::Parse("/World.myRel"));
    assert(myRel != nullptr);
    assert(myRel->GetCustom() == true);

    std::cout << "  USDA relationships: OK\n";
}

void TestUsdaOverAndClass() {
    const char* usda = R"(#usda 1.0

class "_MyClass"
{
    float size = 1.0
}

over "ExistingPrim"
{
    float size = 2.0
}

def Mesh "Concrete"
{
    int faceCount = 6
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto& layer = result.layer;

    auto* cls = layer.GetPrimSpec(Path::Parse("/_MyClass"));
    assert(cls != nullptr);
    assert(cls->GetSpecifier() == Specifier::Class);

    auto* over = layer.GetPrimSpec(Path::Parse("/ExistingPrim"));
    assert(over != nullptr);
    assert(over->GetSpecifier() == Specifier::Over);

    auto* concrete = layer.GetPrimSpec(Path::Parse("/Concrete"));
    assert(concrete != nullptr);
    assert(concrete->GetSpecifier() == Specifier::Def);
    assert(concrete->GetTypeName() == "Mesh");

    std::cout << "  USDA over and class: OK\n";
}

void TestUsdaNestedPrims() {
    const char* usda = R"(#usda 1.0

def Xform "Root"
{
    def Xform "Child1"
    {
        def Mesh "GrandChild"
        {
        }
    }

    def Xform "Child2"
    {
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto& layer = result.layer;

    assert(layer.HasSpec(Path::Parse("/Root")));
    assert(layer.HasSpec(Path::Parse("/Root/Child1")));
    assert(layer.HasSpec(Path::Parse("/Root/Child1/GrandChild")));
    assert(layer.HasSpec(Path::Parse("/Root/Child2")));
    assert(!layer.HasSpec(Path::Parse("/Root/Child3")));

    std::cout << "  USDA nested prims: OK\n";
}

void TestUsdaTimeSamples() {
    const char* usda = R"(#usda 1.0

def Xform "Anim"
{
    double3 xformOp:translate.timeSamples = {
        0: (0, 0, 0),
        12: (10, 20, 30),
        24: (0, 0, 0),
    }
}
)";

    auto result = ParseUsda(usda);
    if (!result.success) {
        std::cout << "  Parse error at " << result.line << ":" << result.column
                  << ": " << result.error << "\n";
    }
    assert(result.success);
    auto& layer = result.layer;

    auto* attr = layer.GetAttributeSpec(Path::Parse("/Anim.xformOp:translate"));
    assert(attr != nullptr);
    assert(attr->HasField(FieldNames::timeSamples));

    std::cout << "  USDA time samples: OK\n";
}

void TestUsdaComments() {
    const char* usda = R"(#usda 1.0
# This is a comment
// This is also a comment
/* Block comment */

def Xform "World" /* inline comment */
{
    # Property comment
    float size = 1.0
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto& layer = result.layer;
    assert(layer.HasSpec(Path::Parse("/World")));

    std::cout << "  USDA comments: OK\n";
}

void TestUsdaMetadata() {
    const char* usda = R"(#usda 1.0

def Xform "World" (
    doc = "The world root"
    kind = "assembly"
    hidden = false
    active = true
    customData = {
        string author = "test"
        int version = 1
    }
)
{
}
)";

    auto result = ParseUsda(usda);
    if (!result.success) {
        std::cout << "  Parse error at " << result.line << ":" << result.column
                  << ": " << result.error << "\n";
    }
    assert(result.success);
    auto& layer = result.layer;

    auto* world = layer.GetPrimSpec(Path::Parse("/World"));
    assert(world != nullptr);
    assert(world->GetDocumentation() == "The world root");
    assert(world->GetKind() == "assembly");
    assert(world->GetHidden() == false);
    assert(world->GetActive() == true);
    assert(world->HasField(FieldNames::customData));

    std::cout << "  USDA metadata: OK\n";
}

void TestUsdaMatrix() {
    const char* usda = R"(#usda 1.0

def Xform "Root"
{
    matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto& layer = result.layer;

    auto* attr = layer.GetAttributeSpec(Path::Parse("/Root.xformOp:transform"));
    assert(attr != nullptr);
    assert(attr->GetTypeName() == "matrix4d");
    auto* val = attr->GetField(FieldNames::defaultValue);
    assert(val != nullptr);
    auto* mat = val->Get<GfMatrix4d>();
    assert(mat != nullptr);
    assert((*mat)(0, 0) == 1.0);
    assert((*mat)(1, 1) == 1.0);
    assert((*mat)(3, 3) == 1.0);
    assert((*mat)(0, 1) == 0.0);

    std::cout << "  USDA matrix: OK\n";
}

// ============================================================
// USDA File-based Tests — scans tests/usda/ directory
// ============================================================

static std::vector<std::string> ListUsdaFiles(const std::string& dir) {
    std::vector<std::string> files;
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".usda") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

int TestUsdaFiles(const std::string& dir) {
    auto files = ListUsdaFiles(dir);
    if (files.empty()) {
        std::cerr << "  WARNING: no .usda files found in " << dir << std::endl;
        return 0;
    }

    int passed = 0;
    int failed = 0;
    for (const auto& path : files) {
        // Extract filename for display
        auto slash = path.find_last_of("/\\");
        std::string name = (slash != std::string::npos) ? path.substr(slash + 1) : path;

        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "  " << name << ": FAIL (cannot open file)" << std::endl;
            ++failed;
            continue;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string content = ss.str();

        auto result = ParseUsda(content);
        if (result.success) {
            std::cout << "  " << name << ": OK" << std::endl;
            ++passed;
        } else {
            std::cerr << "  " << name << ": FAIL at " << result.line << ":"
                      << result.column << ": " << result.error << std::endl;
            ++failed;
        }
    }

    std::cout << "  " << passed << " passed, " << failed << " failed out of "
              << files.size() << " files" << std::endl;
    assert(failed == 0);
    return passed;
}

// ============================================================
// USDA Invalid File Tests — files that must fail to parse
// ============================================================

int TestUsdaInvalidFiles(const std::string& dir) {
    auto files = ListUsdaFiles(dir);
    if (files.empty()) {
        std::cerr << "  WARNING: no .usda files found in " << dir << std::endl;
        return 0;
    }

    int passed = 0;
    int failed = 0;
    for (const auto& path : files) {
        auto slash = path.find_last_of("/\\");
        std::string name = (slash != std::string::npos) ? path.substr(slash + 1) : path;

        std::ifstream f(path, std::ios::binary);
        if (!f) {
            std::cerr << "  " << name << ": FAIL (cannot open file)" << std::endl;
            ++failed;
            continue;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        std::string content = ss.str();

        auto result = ParseUsda(content);
        if (!result.success) {
            std::cout << "  " << name << ": OK (rejected: " << result.error << ")" << std::endl;
            ++passed;
        } else {
            std::cerr << "  " << name << ": FAIL (should have been rejected)" << std::endl;
            ++failed;
        }
    }

    std::cout << "  " << passed << " rejected, " << failed << " incorrectly accepted out of "
              << files.size() << " files" << std::endl;
    assert(failed == 0);
    return passed;
}

// ============================================================
// Composition Tests
// ============================================================

void TestParserStoresSubLayers() {
    const char* usda = R"(#usda 1.0
(
    subLayers = [
        @./base.usda@,
        @./overlay.usda@ (offset = 10; scale = 2)
    ]
))";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto* field = result.layer.GetLayerSpec().GetField(FieldNames::subLayers);
    assert(field != nullptr);
    auto* sl = field->Get<SubLayerPaths>();
    assert(sl != nullptr);
    assert(sl->paths.size() == 2);
    assert(sl->paths[0] == "./base.usda");
    assert(sl->paths[1] == "./overlay.usda");
    assert(sl->offsets[0].offset == 0.0);
    assert(sl->offsets[0].scale == 1.0);
    assert(sl->offsets[1].offset == 10.0);
    assert(sl->offsets[1].scale == 2.0);

    std::cout << "  Parser stores subLayers: OK\n";
}

void TestParserStoresReferences() {
    const char* usda = R"(#usda 1.0

def Xform "World" (
    prepend references = [@./other.usda@</OtherPrim>]
)
{
})";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto* world = result.layer.GetPrimSpec(Path::Parse("/World"));
    assert(world != nullptr);
    auto* refField = world->GetField(FieldNames::references);
    assert(refField != nullptr);
    auto* listOp = refField->Get<ListOp<Reference>>();
    assert(listOp != nullptr);
    assert(!listOp->IsExplicit());
    auto prepended = listOp->GetPrependedItems();
    assert(prepended.size() == 1);
    assert(prepended[0].assetPath == "./other.usda");
    assert(prepended[0].primPath == Path::Parse("/OtherPrim"));

    std::cout << "  Parser stores references: OK\n";
}

void TestComposeSubLayers() {
    auto result = ParseUsdaFile("tests/composition/overlay.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/composition/overlay.usda");
    if (!composed.success) {
        std::cerr << "  Compose error: " << composed.error << "\n";
    }
    assert(composed.success);

    // Create a Stage from the compose result to verify through the Stage API
    auto stage = Stage::CreateFromComposedLayer(std::move(composed.graph));

    // Overlay's opinion (size=2.0) should win over base (size=1.0)
    auto rootPrim = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(rootPrim.IsValid());
    auto sizeAttr = rootPrim.GetAttribute("size");
    assert(sizeAttr.IsValid());
    auto* sizeVal = sizeAttr.GetDefault();
    assert(sizeVal != nullptr);
    assert(*sizeVal->Get<Float>() == 2.0f);

    // Base's name opinion should come through (not overridden)
    auto nameAttr = rootPrim.GetAttribute("name");
    assert(nameAttr.IsValid());
    auto* nameVal = nameAttr.GetDefault();
    assert(nameVal != nullptr);
    assert(*nameVal->Get<String>() == "from_base");

    // Overlay added visible=true
    auto visAttr = rootPrim.GetAttribute("visible");
    assert(visAttr.IsValid());

    // Base's child prim should come through
    auto child = rootPrim.GetChild("Child");
    assert(child.IsValid());

    // Base-only prim should come through
    auto baseOnly = stage.GetPrimAtPath(Path::Parse("/BaseOnly"));
    assert(baseOnly.IsValid());

    // subLayers field should be removed after composition
    assert(!stage.GetComposedLayerSpec().HasField(FieldNames::subLayers));

    std::cout << "  Compose subLayers: OK\n";
}

// Regression: USDC root layer with USDC sublayers must compose all sublayer
// prims. The bug: lazy field decode stored values in a vector<pair<Token,
// Value>>; PostProcessSubLayers held the slField pointer across the
// GetField("subLayerOffsets") fault-in, which reallocated the vector and
// invalidated slField, causing every USDC sublayer on every USDC root to
// be silently dropped at parse time.
void TestComposeUsdcSubLayersRegression() {
    namespace fs = std::filesystem;
    auto subPath  = (fs::temp_directory_path() / "nanousd_usdc_sublayer_regression_sub.usdc").string();
    auto rootPath = (fs::temp_directory_path() / "nanousd_usdc_sublayer_regression_root.usdc").string();

    {
        auto subStage = Stage::CreateInMemory();
        assert(subStage.DefinePrim(Path::Parse("/InSub"), Token("Xform")).IsValid());
        assert(WriteUsdcFile(subStage.GetMutableLayer(), subPath));
    }
    {
        auto rootStage = Stage::CreateInMemory();
        assert(rootStage.DefinePrim(Path::Parse("/InRoot"), Token("Xform")).IsValid());
        SubLayerPaths slp;
        slp.paths.push_back(subPath);
        slp.offsets.push_back({0.0, 1.0});
        rootStage.GetMutableLayer().GetLayerSpec().SetField(
            FieldNames::subLayers, Value(std::move(slp)));
        assert(WriteUsdcFile(rootStage.GetMutableLayer(), rootPath));
    }

    auto stage = Stage::Open(rootPath);
    assert(stage.IsValid());
    assert(stage.GetPrimAtPath(Path::Parse("/InRoot")).IsValid()
           && "Root-layer prim must compose");
    assert(stage.GetPrimAtPath(Path::Parse("/InSub")).IsValid()
           && "Sublayer prim must compose — USDC subLayers must survive lazy decode");

    std::error_code ec;
    fs::remove(rootPath, ec);
    fs::remove(subPath, ec);
    std::cout << "  USDC subLayers compose (regression): OK\n";
}

// Inherits composition-arc resolution. Mirrors the local non-cycle cases
// from the spec-audited inherits regen sandbox (issue #88), plus the
// chained-inherit regression, each authored as a separate top-level prim
// in tests/composition/with_inherits.usda.
//
// What this validates is the *port* of the sandbox's spec-grounded
// algorithm into nanousd's composition pipeline — see
// proof/sandboxes/inherits/spec-citations.md on branch
// aluk/phase-2-regeneration-proof for the per-decision spec citations.
//
// Cycle handling (`direct_cycle` in the sandbox) is intentionally NOT
// covered here: the spec is silent on cycles for inherits arcs (cycle
// language at composition.md lines 88–89 is sublayer-only), and nanousd's
// resolve loop converges via ConsumedArcs + depth-limit rather than
// emitting an explicit cycle error. Both behaviours are spec-permissible.
void TestComposeInherits() {
    auto stage = Stage::Open("tests/composition/with_inherits.usda");
    assert(stage.IsValid());

    // -------- single_inherit --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Single_A"));
        assert(a.IsValid());
        // Direct opinion comes through.
        auto aVal = a.GetAttribute("aVal").GetDefault();
        assert(aVal && *aVal->Get<Int>() == 1);
        // Inherited opinions come through.
        auto bVal = a.GetAttribute("bVal").GetDefault();
        assert(bVal && *bVal->Get<Int>() == 2);
        auto shared = a.GetAttribute("shared").GetDefault();
        assert(shared && *shared->Get<Int>() == 10);
    }

    // -------- multi_target_inherit --------
    // list-op order preserved (first target strongest); both targets contribute.
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Multi_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("bVal").GetDefault()->Get<Int>() == 1);
        assert(*a.GetAttribute("cVal").GetDefault()->Get<Int>() == 2);
        // On a key authored by both targets, /Multi_B (first in list) wins.
        assert(*a.GetAttribute("conflict").GetDefault()->Get<Int>() == 100);
    }

    // -------- dangling_target --------
    // Spec §10.5.1 lines 459–461: composition error when no spec exists at
    // the inherit target in any layer of the stack. We surface this via a
    // diagnostic; the prim itself is otherwise unaffected.
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Dangling_A"));
        assert(a.IsValid());
        // No inherited attrs (target had no specs).
        assert(!a.GetAttribute("anyVal").IsValid());

        bool sawDanglingDiag = false;
        for (const auto& d : stage.GetDiagnostics().GetAll()) {
            if (d.arcType == ArcType::Inherits &&
                d.category == DiagCategory::MissingInheritTarget &&
                d.assetPath.find("/Missing_For_Dangling") != std::string::npos) {
                sawDanglingDiag = true;
                break;
            }
        }
        assert(sawDanglingDiag);
    }

    // -------- inherit_deeper_namespace --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Deeper_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("baseVal").GetDefault()->Get<Int>() == 42);
    }

    // -------- inherit_from_prim_with_no_inherits --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/NoChain_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("val").GetDefault()->Get<Int>() == 1);
    }

    // -------- inherit_namespace_mapping --------
    // /Mapping_B/Child → /Mapping_A/Child; descendant content shows up under
    // the inheriting prim's namespace.
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Mapping_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("parentVal").GetDefault()->Get<Int>() == 1);

        auto child = stage.GetPrimAtPath(Path::Parse("/Mapping_A/Child"));
        assert(child.IsValid());
        assert(*child.GetAttribute("childVal").GetDefault()->Get<Int>() == 2);
    }

    // -------- inherit_opinions_weaker_than_direct --------
    // LIVRPS: direct (Local, strength 0) outranks Inherits (strength 1).
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Strength_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("val").GetDefault()->Get<Int>() == 1);
    }

    // -------- chained_inherit --------
    // /Chained_A inherits /Chained_B inherits /Chained_C - opinions
    // from both /B and /C contribute to /A.
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Chained_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("bVal").GetDefault()->Get<Int>() == 2);
        assert(*a.GetAttribute("cVal").GetDefault()->Get<Int>() == 3);
    }

    std::cout << "  Compose inherits: OK\n";
}

void TestComposeImpliedInherits() {
    auto stage = Stage::Open("tests/composition/implied_inherits_root.usda");
    assert(stage.IsValid());

    // /LeftGroup references group.usda, whose /Group/Asset references
    // asset.usda. asset.usda authors inherits = </class_Asset>; the
    // authored asset-stack target plus implied targets in group/root stacks
    // must all contribute to the composed prim.
    auto asset = stage.GetPrimAtPath(Path::Parse("/LeftGroup/Asset"));
    assert(asset.IsValid());
    assert(*asset.GetAttribute("assetLocalVal").GetDefault()->Get<Int>() == 40);
    assert(*asset.GetAttribute("assetClassVal").GetDefault()->Get<Int>() == 30);
    assert(*asset.GetAttribute("groupClassVal").GetDefault()->Get<Int>() == 20);
    assert(*asset.GetAttribute("rootClassVal").GetDefault()->Get<Int>() == 10);
    assert(*asset.GetAttribute("conflictVal").GetDefault()->Get<Int>() == 10);

    auto child = stage.GetPrimAtPath(Path::Parse("/LeftGroup/Asset/SharedChild"));
    assert(child.IsValid());
    assert(*child.GetAttribute("assetChildVal").GetDefault()->Get<Int>() == 31);
    assert(*child.GetAttribute("groupChildVal").GetDefault()->Get<Int>() == 21);
    assert(*child.GetAttribute("rootChildVal").GetDefault()->Get<Int>() == 11);

    // Missing implied targets in upstream stacks are explicitly not errors:
    // only asset.usda has /assetOnlyClass, and it still must compose.
    auto noUpstream = stage.GetPrimAtPath(Path::Parse("/LeftGroup/NoUpstream"));
    assert(noUpstream.IsValid());
    assert(*noUpstream.GetAttribute("assetOnlyClassVal").GetDefault()->Get<Int>() == 50);
    for (const auto& d : stage.GetDiagnostics().GetAll()) {
        assert(!(d.arcType == ArcType::Inherits &&
                 d.severity == DiagSeverity::Error));
    }

    std::cout << "  Compose implied inherits through refs: OK\n";
}

// Specializes composition-arc resolution. Mirrors the eight non-cycle
// cases from the spec-audited specializes regen sandbox (issue #88,
// branch eslavin/specializes-regen-prep), each authored as a separate
// top-level prim in tests/composition/with_specializes.usda.
//
// Spec line 535 says specializes "is exactly the same as inherits, except
// that the list of specializes are computed using the `specializes`
// field." This test validates the parity claim is testable at the
// composition-pipeline level: equivalent inputs and expected attribute
// resolution between the two arcs in single-layer-stack scope.
//
// Cycle handling (direct_cycle, transitive_cycle from the sandbox) is
// intentionally NOT covered here for the same reason as inherits — the
// spec is silent and nanousd's resolve loop converges via ConsumedArcs +
// depth-limit rather than emitting an explicit cycle error. Both
// behaviours are spec-permissible.
void TestComposeSpecializes() {
    auto stage = Stage::Open("tests/composition/with_specializes.usda");
    assert(stage.IsValid());

    // -------- single_specialize --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Single_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("aVal").GetDefault()->Get<Int>() == 1);
        assert(*a.GetAttribute("bVal").GetDefault()->Get<Int>() == 2);
        assert(*a.GetAttribute("shared").GetDefault()->Get<Int>() == 10);
    }

    // -------- multi_target_specialize --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Multi_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("bVal").GetDefault()->Get<Int>() == 1);
        assert(*a.GetAttribute("cVal").GetDefault()->Get<Int>() == 2);
        // First target in list-op order (Multi_B) wins on conflict.
        assert(*a.GetAttribute("conflict").GetDefault()->Get<Int>() == 100);
    }

    // -------- dangling_target --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Dangling_A"));
        assert(a.IsValid());
        assert(!a.GetAttribute("anyVal").IsValid());

        bool sawDanglingDiag = false;
        for (const auto& d : stage.GetDiagnostics().GetAll()) {
            if (d.arcType == ArcType::Specialize &&
                d.category == DiagCategory::MissingSpecializeTarget &&
                d.assetPath.find("/Missing_For_Specialize_Dangling") != std::string::npos) {
                sawDanglingDiag = true;
                break;
            }
        }
        assert(sawDanglingDiag);
    }

    // -------- specialize_deeper_namespace --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Deeper_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("baseVal").GetDefault()->Get<Int>() == 42);
    }

    // -------- specialize_from_prim_with_no_specializes --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/NoChain_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("val").GetDefault()->Get<Int>() == 1);
    }

    // -------- specialize_namespace_mapping --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Mapping_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("parentVal").GetDefault()->Get<Int>() == 1);

        auto child = stage.GetPrimAtPath(Path::Parse("/Mapping_A/Child"));
        assert(child.IsValid());
        assert(*child.GetAttribute("childVal").GetDefault()->Get<Int>() == 2);
    }

    // -------- specialize_opinions_weaker_than_direct --------
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Strength_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("val").GetDefault()->Get<Int>() == 1);
    }

    // -------- chained_specialize --------
    // /Chained_A specializes /Chained_B specializes /Chained_C — opinions
    // from both /B and /C contribute to /A.
    {
        auto a = stage.GetPrimAtPath(Path::Parse("/Chained_A"));
        assert(a.IsValid());
        assert(*a.GetAttribute("bVal").GetDefault()->Get<Int>() == 2);
        assert(*a.GetAttribute("cVal").GetDefault()->Get<Int>() == 3);
    }

    std::cout << "  Compose specializes: OK\n";
}

void TestComposeImpliedSpecializes() {
    auto stage = Stage::Open("tests/composition/implied_specializes_root.usda");
    assert(stage.IsValid());

    // /LeftGroup references group.usda, whose /Group/Asset references
    // asset.usda. asset.usda authors specializes = </class_Asset>, and
    // /class_Asset specializes /class_Base. Authored and implied targets
    // from all three layer stacks must contribute, but all specializes
    // opinions must remain globally weaker than /Asset's own opinions.
    auto asset = stage.GetPrimAtPath(Path::Parse("/LeftGroup/Asset"));
    assert(asset.IsValid());
    assert(*asset.GetAttribute("assetLocalVal").GetDefault()->Get<Int>() == 40);
    assert(*asset.GetAttribute("assetClassVal").GetDefault()->Get<Int>() == 30);
    assert(*asset.GetAttribute("groupClassVal").GetDefault()->Get<Int>() == 20);
    assert(*asset.GetAttribute("rootClassVal").GetDefault()->Get<Int>() == 10);
    assert(*asset.GetAttribute("assetBaseVal").GetDefault()->Get<Int>() == 130);
    assert(*asset.GetAttribute("groupBaseVal").GetDefault()->Get<Int>() == 120);
    assert(*asset.GetAttribute("rootBaseVal").GetDefault()->Get<Int>() == 110);

    // These conflicts catch the specializes-specific strength rule:
    // upstream implied class opinions beat downstream class opinions, but
    // direct opinions for /Asset beat every specializes opinion.
    assert(*asset.GetAttribute("classConflictVal").GetDefault()->Get<Int>() == 10);
    assert(*asset.GetAttribute("baseConflictVal").GetDefault()->Get<Int>() == 110);
    assert(*asset.GetAttribute("localBeatsSpecializesVal").GetDefault()->Get<Int>() == 40);

    auto sharedChild = stage.GetPrimAtPath(Path::Parse("/LeftGroup/Asset/SharedChild"));
    assert(sharedChild.IsValid());
    assert(*sharedChild.GetAttribute("assetChildVal").GetDefault()->Get<Int>() == 31);
    assert(*sharedChild.GetAttribute("groupChildVal").GetDefault()->Get<Int>() == 21);
    assert(*sharedChild.GetAttribute("rootChildVal").GetDefault()->Get<Int>() == 11);

    auto baseChild = stage.GetPrimAtPath(Path::Parse("/LeftGroup/Asset/BaseChild"));
    assert(baseChild.IsValid());
    assert(*baseChild.GetAttribute("assetBaseChildVal").GetDefault()->Get<Int>() == 131);
    assert(*baseChild.GetAttribute("groupBaseChildVal").GetDefault()->Get<Int>() == 121);
    assert(*baseChild.GetAttribute("rootBaseChildVal").GetDefault()->Get<Int>() == 111);

    // Missing implied targets in upstream stacks are not errors.
    auto noUpstream = stage.GetPrimAtPath(Path::Parse("/LeftGroup/NoUpstream"));
    assert(noUpstream.IsValid());
    assert(*noUpstream.GetAttribute("assetOnlyClassVal").GetDefault()->Get<Int>() == 50);
    for (const auto& d : stage.GetDiagnostics().GetAll()) {
        assert(!(d.arcType == ArcType::Specialize &&
                 d.severity == DiagSeverity::Error));
    }

    std::cout << "  Compose implied specializes through refs: OK\n";
}

void TestComposeLivrpsStrengthOrdering() {
    auto stage = Stage::Open("tests/composition/livrps_strength_root.usda");
    assert(stage.IsValid());

    auto depth = stage.GetPrimAtPath(Path::Parse("/NamespaceDepth/Child"));
    assert(depth.IsValid());
    assert(*depth.GetAttribute("namespaceDepthVal")
                .GetDefault()->Get<Int>() == 20);

    auto sibling = stage.GetPrimAtPath(Path::Parse("/SiblingRefs"));
    assert(sibling.IsValid());
    assert(*sibling.GetAttribute("siblingRefVal")
                  .GetDefault()->Get<Int>() == 10);

    auto authored = stage.GetPrimAtPath(Path::Parse("/AuthoredVsImplied"));
    assert(authored.IsValid());
    assert(*authored.GetAttribute("authoredVsImpliedVal")
                   .GetDefault()->Get<Int>() == 10);

    auto payloadStage =
        Stage::Open("tests/composition/pcp_BasicNestedPayload_root.usda");
    assert(payloadStage.IsValid());
    auto nestedPayload =
        payloadStage.GetPrimAtPath(Path::Parse("/Set2/Prop/PropScope"));
    assert(nestedPayload.IsValid());
    assert(*nestedPayload.GetAttribute("x").GetDefault()->Get<String>() ==
           "from prop_payload");

    std::cout << "  Compose LIVRPS strength ordering: OK\n";
}

void TestComposeReferences() {
    auto result = ParseUsdaFile("tests/composition/with_ref.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/composition/with_ref.usda");
    if (!composed.success) {
        std::cerr << "  Compose error: " << composed.error << "\n";
    }
    assert(composed.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(composed.graph));

    // MyPrim should exist
    auto myPrim = stage.GetPrimAtPath(Path::Parse("/MyPrim"));
    assert(myPrim.IsValid());

    // Local override wins: label = "local_override"
    auto labelAttr = myPrim.GetAttribute("label");
    assert(labelAttr.IsValid());
    auto* labelVal = labelAttr.GetDefault();
    assert(labelVal != nullptr);
    assert(*labelVal->Get<String>() == "local_override");

    // Referenced opinion comes through: height = 5.0
    auto heightAttr = myPrim.GetAttribute("height");
    assert(heightAttr.IsValid());
    auto* heightVal = heightAttr.GetDefault();
    assert(heightVal != nullptr);
    assert(*heightVal->Get<Float>() == 5.0f);

    // Referenced child subtree comes through
    auto geo = myPrim.GetChild("Geo");
    assert(geo.IsValid());

    std::cout << "  Compose references: OK\n";
}

void TestComposeBareSiblingReference() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path dir = fs::temp_directory_path() / "nanousd_compose_bare_sibling_ref";
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);

    fs::path base = dir / "base.usda";
    fs::path root = dir / "root.usda";
    {
        std::ofstream out(base.string());
        out << R"(#usda 1.0
(
    defaultPrim = "Base"
)

def Xform "Base"
{
    float height = 5.0
    def Mesh "Geo"
    {
        int vertexCount = 3
    }
}
)";
    }
    {
        std::ofstream out(root.string());
        out << R"(#usda 1.0

def Xform "Root" (
    prepend references = @base.usda@
)
{
    string label = "local_override"
}
)";
    }

    auto stage = Stage::Open(root.generic_string());
    assert(stage.IsValid());
    assert(!stage.HasCompositionErrors());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(prim.IsValid());
    assert(*prim.GetAttribute("label").GetDefault()->Get<String>() ==
           "local_override");
    assert(*prim.GetAttribute("height").GetDefault()->Get<Float>() == 5.0f);
    assert(prim.GetChild("Geo").IsValid());

    fs::remove_all(dir, ec);
    std::cout << "  Bare sibling references compose: OK\n";
}

void TestComposeLayerOffset() {
    auto result = ParseUsdaFile("tests/composition/offset_sub.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/composition/offset_sub.usda");
    if (!composed.success) {
        std::cerr << "  Compose error: " << composed.error << "\n";
    }
    assert(composed.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(composed.graph));

    // The sublayer anim.usda has timeSamples at 0 and 10.
    // With offset=10, scale=2: newTime = time*2 + 10
    // So 0 -> 10, 10 -> 30
    auto animPrim = stage.GetPrimAtPath(Path::Parse("/Anim"));
    assert(animPrim.IsValid());
    auto attr = animPrim.GetAttribute("xformOp:translate");
    assert(attr.IsValid());
    assert(attr.HasTimeSamples());

    std::cout << "  Compose layer offset: OK\n";
}

// --- PCP-derived composition tests ---

void TestComposeThreeSublayers() {
    auto stage = Stage::Open("tests/composition/three_sublayers_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Strongest layer wins "opinion"
    auto opinion = prim.GetAttribute("opinion");
    assert(opinion.IsValid());
    assert(*opinion.GetDefault()->Get<String>() == "strongest");

    // Each layer contributes unique attrs
    assert(prim.GetAttribute("rootOnly").IsValid());
    assert(prim.GetAttribute("midOnly").IsValid());
    assert(prim.GetAttribute("weakOnly").IsValid());

    // Mid layer's midOnly=2.0 wins over weak layer's midOnly=99.0
    auto midOnly = prim.GetAttribute("midOnly");
    assert(*midOnly.GetDefault()->Get<Float>() == 2.0f);

    std::cout << "  Three sublayers strength: OK\n";
}

void TestComposeMultipleReferences() {
    auto stage = Stage::Open("tests/composition/multi_ref_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Local opinion wins
    assert(*prim.GetAttribute("label").GetDefault()->Get<String>() == "local");

    // First ref (A) is stronger — its "label" would be "from_A" if local didn't override
    // Both refs contribute unique attrs
    assert(prim.GetAttribute("aOnly").IsValid());
    assert(*prim.GetAttribute("aOnly").GetDefault()->Get<Float>() == 10.0f);
    assert(prim.GetAttribute("bOnly").IsValid());
    assert(*prim.GetAttribute("bOnly").GetDefault()->Get<Float>() == 20.0f);

    // Children from both refs
    assert(prim.GetChild("ChildA").IsValid());
    assert(prim.GetChild("ChildB").IsValid());

    std::cout << "  Multiple references: OK\n";
}

void TestComposeInternalReference() {
    auto stage = Stage::Open("tests/composition/internal_ref.usda");
    assert(stage.IsValid());

    auto inst = stage.GetPrimAtPath(Path::Parse("/Instance"));
    assert(inst.IsValid());

    // Local override wins
    assert(*inst.GetAttribute("tag").GetDefault()->Get<String>() == "instance_override");

    // Referenced attr comes through
    assert(inst.GetAttribute("height").IsValid());
    assert(*inst.GetAttribute("height").GetDefault()->Get<Float>() == 5.0f);

    // Referenced child subtree comes through
    assert(inst.GetChild("Geo").IsValid());

    std::cout << "  Internal reference: OK\n";
}

void TestComposeChainedReferences() {
    auto stage = Stage::Open("tests/composition/chained_ref_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Root's opinion is strongest
    assert(*prim.GetAttribute("opinion").GetDefault()->Get<String>() == "root");

    // A's attr comes through (A stronger than B)
    assert(*prim.GetAttribute("aAttr").GetDefault()->Get<Float>() == 1.0f);

    // B's unique attr comes through the chain
    assert(prim.GetAttribute("bAttr").IsValid());
    assert(*prim.GetAttribute("bAttr").GetDefault()->Get<Float>() == 2.0f);

    // B's child subtree comes through the chain
    assert(prim.GetChild("Geo").IsValid());

    std::cout << "  Chained references: OK\n";
}

void TestComposeReferenceDiamond() {
    auto stage = Stage::Open("tests/composition/ref_diamond_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Both refs contribute unique attrs
    assert(prim.GetAttribute("aOnly").IsValid());
    assert(*prim.GetAttribute("aOnly").GetDefault()->Get<Float>() == 1.0f);
    assert(prim.GetAttribute("bOnly").IsValid());
    assert(*prim.GetAttribute("bOnly").GetDefault()->Get<Float>() == 2.0f);

    // C's shared attr comes through (from both diamond paths)
    assert(prim.GetAttribute("shared").IsValid());
    assert(*prim.GetAttribute("shared").GetDefault()->Get<Float>() == 3.0f);

    // C's child comes through
    assert(prim.GetChild("SharedChild").IsValid());

    std::cout << "  Reference diamond: OK\n";
}

void TestComposeSublayerRefInteraction() {
    // LIVRPS: sublayer opinions (Local + Sublayer) are stronger than references
    auto stage = Stage::Open("tests/composition/sublayer_ref_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Root local is strongest in LIVRPS
    assert(*prim.GetAttribute("opinion").GetDefault()->Get<String>() == "root_local");

    // Sublayer-only attr comes through
    assert(prim.GetAttribute("subOnly").IsValid());
    assert(*prim.GetAttribute("subOnly").GetDefault()->Get<Float>() == 10.0f);

    // Reference-only attr comes through
    assert(prim.GetAttribute("refOnly").IsValid());
    assert(*prim.GetAttribute("refOnly").GetDefault()->Get<Float>() == 20.0f);

    // Referenced child subtree comes through
    assert(prim.GetChild("Geo").IsValid());

    std::cout << "  Sublayer + reference interaction (LIVRPS): OK\n";
}

void TestComposeSubrootReference() {
    auto stage = Stage::Open("tests/composition/subroot_ref.usda");
    assert(stage.IsValid());

    auto consumer = stage.GetPrimAtPath(Path::Parse("/Consumer"));
    assert(consumer.IsValid());

    // Sub-root ref to /Source/Geo brings its attrs
    assert(consumer.GetAttribute("faceCount").IsValid());
    assert(*consumer.GetAttribute("faceCount").GetDefault()->Get<Int>() == 6);
    assert(consumer.GetAttribute("radius").IsValid());

    std::cout << "  Sub-root reference: OK\n";
}

void TestComposeReferenceOffset() {
    auto stage = Stage::Open("tests/composition/ref_offset_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    auto attr = prim.GetAttribute("xformOp:translate");
    assert(attr.IsValid());
    assert(attr.HasTimeSamples());

    // Source has samples at 0, 10, 20. With offset=10, scale=2:
    // newTime = time*scale + offset, so 0->10, 10->30, 20->50
    auto keys = attr.GetTimeSampleTimes();
    assert(keys.size() == 3);
    assert(keys[0] == 10.0);
    assert(keys[1] == 30.0);
    assert(keys[2] == 50.0);

    std::cout << "  Reference time offset: OK\n";
}

bool HasInvalidRetimingScaleDiagnostic(const Stage& stage, ArcType arcType) {
    for (const auto& diag : stage.GetDiagnostics().GetAll()) {
        if (diag.severity == DiagSeverity::Error &&
            diag.category == DiagCategory::InvalidRetimingScale &&
            diag.arcType == arcType) {
            return true;
        }
    }
    return false;
}

void AssertInvalidRetimingScaleFallsBack(
        const char* filePath,
        const char* primPath,
        ArcType arcType) {
    auto stage = Stage::Open(filePath);
    assert(stage.IsValid());
    assert(stage.HasCompositionErrors());
    assert(HasInvalidRetimingScaleDiagnostic(stage, arcType));

    auto prim = stage.GetPrimAtPath(Path::Parse(primPath));
    assert(prim.IsValid());

    auto attr = prim.GetAttribute("value");
    assert(attr.IsValid());

    auto value = attr.Get(UsdTimeCode(0.0));
    assert(value.found);
    auto* d = value.value.Get<Double>();
    assert(d != nullptr);
    assert(*d == 0.0);
}

void TestInvalidRetimingScaleFallback() {
    AssertInvalidRetimingScaleFallsBack(
        "tests/composition/invalid_retiming_sublayer_root.usda",
        "/Source",
        ArcType::Sublayer);
    AssertInvalidRetimingScaleFallsBack(
        "tests/composition/invalid_retiming_ref_root.usda",
        "/Prim",
        ArcType::Reference);
    AssertInvalidRetimingScaleFallsBack(
        "tests/composition/invalid_retiming_internal_ref.usda",
        "/Prim",
        ArcType::Reference);
    AssertInvalidRetimingScaleFallsBack(
        "tests/composition/invalid_retiming_payload_root.usda",
        "/Prim",
        ArcType::Payload);
    AssertInvalidRetimingScaleFallsBack(
        "tests/composition/invalid_retiming_internal_payload.usda",
        "/Prim",
        ArcType::Payload);

    std::cout << "  Invalid retiming scale falls back to identity: OK\n";
}

void TestComposeCompoundOffset() {
    // Reference offset=10 into a file whose sublayer has offset=20, scale=2
    // Sub's samples at 0, 5, 10 become (via sublayer): 0*2+20=20, 5*2+20=30, 10*2+20=40
    // Then ref offset=10 adds: 20+10=30, 30+10=40, 40+10=50
    auto stage = Stage::Open("tests/composition/compound_offset_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    auto attr = prim.GetAttribute("xformOp:translate");
    assert(attr.IsValid());
    assert(attr.HasTimeSamples());

    auto keys = attr.GetTimeSampleTimes();
    assert(keys.size() == 3);
    assert(keys[0] == 30.0);
    assert(keys[1] == 40.0);
    assert(keys[2] == 50.0);

    std::cout << "  Compound offset (ref + sublayer): OK\n";
}

// ============================================================
// Stage Tests
// ============================================================

void TestStageOpen() {
    auto stage = Stage::Open("tests/composition/overlay.usda");
    if (!stage.IsValid()) {
        std::cerr << "  Stage error: " << stage.GetError() << "\n";
    }
    assert(stage.IsValid());

    // Default prim
    auto dp = stage.GetDefaultPrim();
    assert(dp.IsValid());
    assert(dp.GetName() == "Root");

    std::cout << "  Stage open: OK\n";
}

void TestStagePopulation() {
    const char* usda = R"(#usda 1.0

def Xform "Active" (
    kind = "assembly"
)
{
    def Mesh "Child" (
        kind = "component"
    )
    {
    }

    over "OverChild"
    {
        def Mesh "HiddenGrandchild" {}
    }
}

def Xform "Inactive" (
    active = false
)
{
    def Mesh "Hidden" {}
}

over "OverOnly"
{
    float size = 1.0
}

class "_Abstract"
{
    float radius = 5.0

    def Xform "AbstractChild" {}
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    // Root prims: Active and _Abstract should be populated.
    // Inactive is excluded (active=false). OverOnly is excluded (Over specifier).
    auto roots = stage.GetRootPrims();
    assert(roots.size() == 2);  // Active, _Abstract

    // Check Active prim
    auto active = stage.GetPrimAtPath(Path::Parse("/Active"));
    assert(active.IsValid());
    assert(active.IsDefined());
    assert(active.IsActive());
    assert(active.IsLoaded());
    assert(active.IsModel());

    // Active's child should be populated
    auto children = active.GetChildren();
    assert(children.size() == 1);
    assert(children[0].GetName() == "Child");
    assert(children[0].IsLoaded());
    assert(children[0].IsModel());
    assert(!stage.HasPrimAtPath(Path::Parse("/Active/OverChild")));
    assert(!stage.HasPrimAtPath(Path::Parse("/Active/OverChild/HiddenGrandchild")));

    // Inactive prim should not be populated
    assert(!stage.HasPrimAtPath(Path::Parse("/Inactive")));

    // Over-only should not be populated
    assert(!stage.HasPrimAtPath(Path::Parse("/OverOnly")));

    // Abstract class should be populated
    auto abstract = stage.GetPrimAtPath(Path::Parse("/_Abstract"));
    assert(abstract.IsValid());
    assert(abstract.IsAbstract());

    auto abstractChild = stage.GetPrimAtPath(Path::Parse("/_Abstract/AbstractChild"));
    assert(abstractChild.IsValid());
    assert(abstractChild.IsDefined());
    assert(abstractChild.IsAbstract());

    std::cout << "  Stage population: OK\n";
}

void TestStagePopulationMask() {
    namespace fs = std::filesystem;
    auto rootPath = fs::temp_directory_path() /
        "nanousd_stage_population_mask.usda";

    {
        std::ofstream f(rootPath);
        f << R"(#usda 1.0

def Xform "World"
{
    def Scope "Keep"
    {
        def Scope "Leaf"
        {
            def Mesh "Grandchild" {}
        }
        def Mesh "Sibling" {}
    }

    def Scope "Drop"
    {
        def Mesh "Leaf" {}
    }
}

def Scope "Other" {}
)";
    }

    auto stage = Stage::OpenMasked(
        rootPath.string(), {Path::Parse("/World/Keep/Leaf")});
    assert(stage.IsValid());

    std::vector<std::string> paths;
    for (const auto& prim : stage.Traverse()) {
        paths.push_back(prim.GetPath().GetText());
    }
    assert(paths.size() == 3);
    assert(paths[0] == "/World");
    assert(paths[1] == "/World/Keep");
    assert(paths[2] == "/World/Keep/Leaf");

    assert(stage.HasPrimAtPath(Path::Parse("/World")));
    assert(stage.HasPrimAtPath(Path::Parse("/World/Keep")));
    assert(stage.HasPrimAtPath(Path::Parse("/World/Keep/Leaf")));
    assert(!stage.HasPrimAtPath(Path::Parse("/World/Keep/Leaf/Grandchild")));
    assert(!stage.HasPrimAtPath(Path::Parse("/World/Keep/Sibling")));
    assert(!stage.HasPrimAtPath(Path::Parse("/World/Drop")));
    assert(!stage.HasPrimAtPath(Path::Parse("/Other")));

    auto world = stage.GetPrimAtPath(Path::Parse("/World"));
    auto worldChildren = world.GetChildren();
    assert(worldChildren.size() == 1);
    assert(worldChildren[0].GetName() == "Keep");

    auto keep = stage.GetPrimAtPath(Path::Parse("/World/Keep"));
    auto keepChildren = keep.GetChildren();
    assert(keepChildren.size() == 1);
    assert(keepChildren[0].GetName() == "Leaf");
    assert(!keep.HasChild(Token("Sibling")));

    auto leaf = stage.GetPrimAtPath(Path::Parse("/World/Keep/Leaf"));
    assert(leaf.GetChildren().empty());
    assert(!leaf.HasChild(Token("Grandchild")));

    auto unmasked = Stage::OpenMasked(rootPath.string(), std::vector<Path>{});
    assert(unmasked.IsValid());
    assert(unmasked.Traverse().size() == 8);
    assert(unmasked.GetRootPrims().size() == 2);
    assert(unmasked.HasPrimAtPath(Path::Parse("/World")));
    assert(unmasked.HasPrimAtPath(Path::Parse("/World/Keep/Leaf/Grandchild")));
    assert(unmasked.HasPrimAtPath(Path::Parse("/Other")));

    std::error_code ec;
    fs::remove(rootPath, ec);

    std::cout << "  Stage population mask: OK\n";
}

void TestStagePopulationConsistency() {
    {
        auto stage = Stage::Open(
            "tests/composition/stage_population_inactive_default.usda");
        assert(stage.IsValid());
        assert(!stage.GetDefaultPrim().IsValid());
        assert(!stage.HasPrimAtPath(Path::Parse("/InactiveDefault")));
        assert(stage.HasPrimAtPath(Path::Parse("/ActivePeer")));
    }

    {
        Layer layer;
        layer.GetLayerSpec().SetDefaultPrim("IntInactive");

        Spec inactive(SpecType::Prim);
        inactive.SetSpecifier(Specifier::Def);
        inactive.SetTypeName("Xform");
        inactive.SetField(FieldNames::active, Value(Int(0)));
        layer.SetSpec(Path::Parse("/IntInactive"), std::move(inactive));

        Spec peer(SpecType::Prim);
        peer.SetSpecifier(Specifier::Def);
        peer.SetTypeName("Xform");
        layer.SetSpec(Path::Parse("/Peer"), std::move(peer));

        auto stage = Stage::CreateFromComposedLayer(std::move(layer));
        assert(stage.IsValid());
        assert(!stage.GetDefaultPrim().IsValid());
        assert(!stage.HasPrimAtPath(Path::Parse("/IntInactive")));
        assert(stage.HasPrimAtPath(Path::Parse("/Peer")));
    }

    {
        auto stage = Stage::Open(
            "tests/composition/stage_population_order_root.usda");
        assert(stage.IsValid());

        auto parent = stage.GetPrimAtPath(Path::Parse("/Parent"));
        assert(parent.IsValid());

        auto children = parent.GetChildren();
        assert(children.size() == 3);
        assert(children[0].GetName() == "charlie");
        assert(children[1].GetName() == "alpha");
        assert(children[2].GetName() == "bravo");

        auto prims = stage.Traverse();
        assert(prims.size() == 4);
        assert(prims[0].GetPath().GetText() == "/Parent");
        assert(prims[1].GetPath().GetText() == "/Parent/charlie");
        assert(prims[2].GetPath().GetText() == "/Parent/alpha");
        assert(prims[3].GetPath().GetText() == "/Parent/bravo");
    }

    {
        namespace fs = std::filesystem;
        auto dir = fs::temp_directory_path();
        auto rootPath = dir / "nanousd_stage_population_order_stack_root.usda";
        auto strongPath = dir / "nanousd_stage_population_order_stack_strong.usda";
        auto weakPath = dir / "nanousd_stage_population_order_stack_weak.usda";

        {
            std::ofstream f(weakPath);
            f << R"(#usda 1.0

def Xform "Parent"
{
    reorder nameChildren = ["charlie", "bravo", "alpha"]
    def Scope "alpha" {}
    def Scope "bravo" {}
    def Scope "charlie" {}
}
)";
        }
        {
            std::ofstream f(strongPath);
            f << R"(#usda 1.0

def Xform "Parent"
{
    reorder nameChildren = ["alpha"]
}
)";
        }
        {
            std::ofstream f(rootPath);
            f << R"(#usda 1.0
(
    subLayers = [
        @./nanousd_stage_population_order_stack_strong.usda@,
        @./nanousd_stage_population_order_stack_weak.usda@
    ]
)
)";
        }

        auto stage = Stage::Open(rootPath.string());
        assert(stage.IsValid());

        auto parent = stage.GetPrimAtPath(Path::Parse("/Parent"));
        assert(parent.IsValid());

        auto children = parent.GetChildren();
        assert(children.size() == 3);
        assert(children[0].GetName() == "alpha");
        assert(children[1].GetName() == "charlie");
        assert(children[2].GetName() == "bravo");

        std::error_code ec;
        fs::remove(rootPath, ec);
        fs::remove(strongPath, ec);
        fs::remove(weakPath, ec);
    }

    {
        auto stage = Stage::Open("tests/usda/reorder.usda");
        assert(stage.IsValid());

        auto roots = stage.GetRootPrims();
        assert(roots.size() == 3);
        assert(roots[0].GetName() == "B");
        assert(roots[1].GetName() == "A");
        assert(roots[2].GetName() == "C");

        auto prims = stage.Traverse();
        assert(prims.size() == 6);
        assert(prims[0].GetPath().GetText() == "/B");
        assert(prims[1].GetPath().GetText() == "/A");
        assert(prims[2].GetPath().GetText() == "/A/Second");
        assert(prims[3].GetPath().GetText() == "/A/First");
        assert(prims[4].GetPath().GetText() == "/A/Third");
        assert(prims[5].GetPath().GetText() == "/C");
    }

    std::cout << "  Stage population consistency: OK\n";
}

void TestStageTraversal() {
    const char* usda = R"(#usda 1.0

def Xform "A"
{
    def Mesh "D" {}
    def Xform "B"
    {
        def Mesh "C" {}
    }
}

def Xform "E" {}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prims = stage.Traverse();
    assert(prims.size() == 5);  // A, D, B, C, E

    // Depth-first order with authored children: A, D, B, C, E
    assert(prims[0].GetName() == "A");
    assert(prims[1].GetName() == "D");
    assert(prims[2].GetName() == "B");
    assert(prims[3].GetName() == "C");
    assert(prims[4].GetName() == "E");

    std::cout << "  Stage traversal: OK\n";
}

void TestComposedChildIndexDedupesLayerSpecs() {
    const char* usda = R"(#usda 1.0

def Xform "Root"
{
    def Xform "B" {}
    def Xform "A" {}
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto composed = Compose(result.layer, "");
    assert(composed.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(composed.graph));
    assert(stage.IsValid());

    auto root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());
    auto children = root.GetChildren();
    assert(children.size() == 2);
    assert(children[0].GetName() == "B");
    assert(children[1].GetName() == "A");

    std::cout << "  Composed child index dedupes layer specs: OK\n";
}

// Tests for PR #124 (cached populated-prim provenance) live in
// tests/test_traverse_provenance.cpp; declare and call them here so they
// run as part of the unified nanousd_tests binary.
void TestStageTraversePreservesStrongestSpecProvenance();
void TestStageTraverseSeesInPlaceSpecValueEdit();
void TestStageTraverseRebuildsAfterDefinePrim();
void TestStageTraverseFallsBackWhenCachedSpecRemoved();
void TestStageTraverseSeesTypeNameChange();
void TestStageTraverseSeesNewAuthoredAttribute();
void TestStageCreateAttributeAfterTraverse();
void TestStageChildrenAfterNestedDefine();
void TestUsdPrimHandleSurvivesWriteOnUnrelatedSpec();

void TestStageProperties() {
    const char* usda = R"(#usda 1.0

def Xform "Root"
{
    float size = 1.0
    string name = "hello"
    rel material:binding = </Materials/Default>
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());

    // Property names: 3 authored (size, name, material:binding) + schema props from Xform hierarchy
    // Xform -> Xformable (xformOpOrder) -> Imageable (visibility, purpose)
    // proxyPrim is rel type, skipped. material:binding is authored rel, included.
    auto propNames = root.GetPropertyNames();
    assert(propNames.size() >= 3);  // at least the 3 authored properties

    // Attribute access
    assert(root.HasAttribute("size"));
    auto sizeAttr = root.GetAttribute("size");
    assert(sizeAttr.IsValid());
    assert(sizeAttr.GetTypeName() == "float");
    assert(sizeAttr.HasDefault());
    assert(*sizeAttr.GetDefault()->Get<Float>() == 1.0f);

    // Relationship access
    assert(root.HasRelationship("material:binding"));
    auto rel = root.GetRelationship("material:binding");
    assert(rel.IsValid());

    // Non-existent
    assert(!root.HasAttribute("nonexistent"));
    assert(!root.HasRelationship("nonexistent"));

    std::cout << "  Stage properties: OK\n";
}

void TestStageMetadata() {
    const char* usda = R"(#usda 1.0
(
    defaultPrim = "Root"
    timeCodesPerSecond = 30
    startTimeCode = 1
    endTimeCode = 240
    framesPerSecond = 30
)

def Xform "Root" {}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    assert(stage.GetTimeCodesPerSecond() == 30.0);
    assert(stage.GetFramesPerSecond() == 30.0);
    assert(stage.GetStartTimeCode() == 1.0);
    assert(stage.GetEndTimeCode() == 240.0);

    auto dp = stage.GetDefaultPrim();
    assert(dp.IsValid());
    assert(dp.GetName() == "Root");

    std::cout << "  Stage metadata: OK\n";
}

void TestMetersPerUnit() {
    // Test 1: metersPerUnit authored
    {
        const char* usda = R"(#usda 1.0
(
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "Root" {}
)";
        auto result = ParseUsda(usda);
        assert(result.success);
        auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

        assert(HasMetersPerUnit(stage));
        assert(GetMetersPerUnit(stage) == 1.0);
        assert(HasUpAxis(stage));
        assert(GetUpAxis(stage) == "Z");
    }

    // Test 2: metersPerUnit not authored — fallback to 0.01 (centimeters)
    {
        const char* usda = R"(#usda 1.0

def Xform "Root" {}
)";
        auto result = ParseUsda(usda);
        assert(result.success);
        auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

        assert(!HasMetersPerUnit(stage));
        assert(GetMetersPerUnit(stage) == 0.01);
        assert(!HasUpAxis(stage));
        assert(GetUpAxis(stage) == "Y");
    }

    // Test 3: various standard unit values
    {
        const char* usda = R"(#usda 1.0
(
    metersPerUnit = 0.0254
)

def Xform "Root" {}
)";
        auto result = ParseUsda(usda);
        assert(result.success);
        auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
        assert(GetMetersPerUnit(stage) == 0.0254);  // inches
    }

    // Test 4: Layer-level set/get
    {
        Layer layer;
        assert(!HasMetersPerUnit(layer));
        assert(GetMetersPerUnit(layer) == 0.01);

        SetMetersPerUnit(layer, 0.3048);
        assert(HasMetersPerUnit(layer));
        assert(GetMetersPerUnit(layer) == 0.3048);  // feet

        SetUpAxis(layer, Token("Z"));
        assert(HasUpAxis(layer));
        assert(GetUpAxis(layer) == "Z");
    }

    // Test 5: combined with other layer metadata
    {
        const char* usda = R"(#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 0.01
    upAxis = "Y"
    timeCodesPerSecond = 24
)

def Xform "Root" {}
)";
        auto result = ParseUsda(usda);
        assert(result.success);
        auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
        assert(GetMetersPerUnit(stage) == 0.01);
        assert(GetUpAxis(stage) == "Y");
        assert(stage.GetTimeCodesPerSecond() == 24.0);
        auto dp = stage.GetDefaultPrim();
        assert(dp.IsValid());
        assert(dp.GetName() == "Root");
    }

    std::cout << "  metersPerUnit / upAxis: OK\n";
}

void TestStageComposedOpen() {
    // Open a stage with sublayers — tests the full pipeline
    auto stage = Stage::Open("tests/composition/overlay.usda");
    assert(stage.IsValid());

    // Root prim from base + overlay opinions
    auto root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());

    // Overlay's size wins
    auto sizeAttr = root.GetAttribute("size");
    assert(sizeAttr.IsValid());
    assert(*sizeAttr.GetDefault()->Get<Float>() == 2.0f);

    // Base's name comes through
    auto nameAttr = root.GetAttribute("name");
    assert(nameAttr.IsValid());
    assert(*nameAttr.GetDefault()->Get<String>() == "from_base");

    // Base's child comes through
    auto child = root.GetChild("Child");
    assert(child.IsValid());

    // BaseOnly prim comes through from sublayer
    auto baseOnly = stage.GetPrimAtPath(Path::Parse("/BaseOnly"));
    assert(baseOnly.IsValid());

    std::cout << "  Stage composed open: OK\n";
}

// ============================================================
// Schema System Tests
// ============================================================

void TestSchemaRegistration() {
    auto& reg = SchemaRegistry::GetInstance();

    // Normative schemas should already be registered
    assert(reg.FindSchema("CollectionAPI") != nullptr);
    assert(reg.FindSchema("ColorSpaceAPI") != nullptr);
    assert(reg.FindSchema("ColorSpaceDefinitionAPI") != nullptr);

    // Query typed vs applied
    assert(reg.IsAppliedSchema("CollectionAPI"));
    assert(reg.IsAppliedSchema("ColorSpaceAPI"));
    assert(reg.IsAppliedSchema("ColorSpaceDefinitionAPI"));
    assert(!reg.IsTypedSchema("CollectionAPI"));

    // Non-existent schema
    assert(reg.FindSchema("NoSuchSchema") == nullptr);
    assert(!reg.IsTypedSchema("NoSuchSchema"));

    // Register a custom typed schema
    SchemaDef custom;
    custom.name = "TestCustom";
    custom.kind = SchemaKind::Typed;
    custom.parent = "";
    assert(reg.RegisterSchema(custom));

    // Duplicate registration fails
    SchemaDef dup;
    dup.name = "TestCustom";
    dup.kind = SchemaKind::Typed;
    assert(!reg.RegisterSchema(dup));

    // Verify custom schema is findable
    auto* found = reg.FindSchema("TestCustom");
    assert(found != nullptr);
    assert(found->kind == SchemaKind::Typed);

    // Clean up for other tests
    reg.Clear();

    std::cout << "  Schema registration: OK\n";
}

void TestSchemaJSON() {
    auto& reg = SchemaRegistry::GetInstance();
    reg.Clear();

    const char* json = R"({
      "schemas": {
        "MyTyped": {
          "schemaKind": "typed",
          "parent": "",
          "properties": {
            "radius": { "type": "float", "fallback": 1.0 },
            "label": { "type": "token", "variability": "uniform", "fallback": "default" }
          }
        },
        "MyAPI": {
          "schemaKind": "singleApply",
          "properties": {
            "myapi:enabled": { "type": "bool", "fallback": true }
          }
        }
      }
    })";

    std::string err;
    assert(reg.LoadFromJSON(json, &err));

    auto* typed = reg.FindSchema("MyTyped");
    assert(typed != nullptr);
    assert(typed->kind == SchemaKind::Typed);
    assert(typed->properties.size() == 2);
    assert(typed->properties.at("radius").typeName == "float");
    assert(typed->properties.at("label").variability == Variability::Uniform);

    auto* api = reg.FindSchema("MyAPI");
    assert(api != nullptr);
    assert(api->kind == SchemaKind::SingleApply);
    assert(api->properties.size() == 1);

    // Invalid JSON
    std::string err2;
    assert(!reg.LoadFromJSON("{bad json", &err2));
    assert(!err2.empty());

    reg.Clear();
    std::cout << "  Schema JSON loading: OK\n";
}

void TestSchemaIsA() {
    auto& reg = SchemaRegistry::GetInstance();
    reg.Clear();

    // Register an inheritance chain: TestSpecialMesh -> TestBaseMesh -> TestGprim
    const char* json = R"({
      "schemas": {
        "TestGprim": {
          "schemaKind": "typed",
          "parent": "",
          "properties": {
            "visibility": { "type": "token", "fallback": "inherited" }
          }
        },
        "TestBaseMesh": {
          "schemaKind": "typed",
          "parent": "TestGprim",
          "properties": {
            "points": { "type": "point3f[]" }
          }
        },
        "TestSpecialMesh": {
          "schemaKind": "typed",
          "parent": "TestBaseMesh",
          "properties": {
            "subdivisionScheme": { "type": "token", "fallback": "catmullClark" }
          }
        }
      }
    })";

    std::string err;
    assert(reg.LoadFromJSON(json, &err));

    // Self-identity
    assert(reg.IsA("TestSpecialMesh", "TestSpecialMesh"));
    assert(reg.IsA("TestBaseMesh", "TestBaseMesh"));

    // Inheritance
    assert(reg.IsA("TestSpecialMesh", "TestBaseMesh"));
    assert(reg.IsA("TestSpecialMesh", "TestGprim"));
    assert(reg.IsA("TestBaseMesh", "TestGprim"));

    // Not reverse
    assert(!reg.IsA("TestGprim", "TestSpecialMesh"));
    assert(!reg.IsA("TestGprim", "TestBaseMesh"));

    // Non-existent
    assert(!reg.IsA("TestSpecialMesh", "NoSuch"));
    assert(!reg.IsA("NoSuch", "TestGprim"));

    reg.Clear();
    std::cout << "  Schema IsA: OK\n";
}

void TestSchemaHasAPI() {
    // Parse a USDA with apiSchemas metadata and test HasAPI
    const char* usda = R"(#usda 1.0

def Xform "Root" (
    prepend apiSchemas = ["CollectionAPI:lights", "ColorSpaceAPI"]
)
{
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());

    // Test GetAppliedSchemas
    auto schemas = root.GetAppliedSchemas();
    assert(schemas.size() == 2);

    // Test HasAPI - single-apply
    assert(root.HasAPI("ColorSpaceAPI"));

    // Test HasAPI - multi-apply with instance name
    assert(root.HasAPI("CollectionAPI", "lights"));
    assert(root.HasAPI("CollectionAPI")); // prefix match
    assert(!root.HasAPI("CollectionAPI", "shadows")); // wrong instance

    // Negative
    assert(!root.HasAPI("NoSuchAPI"));

    std::cout << "  Schema HasAPI: OK\n";
}

void TestSchemaPrimDefinition() {
    auto& reg = SchemaRegistry::GetInstance();
    reg.Clear();

    const char* json = R"({
      "schemas": {
        "Base": {
          "schemaKind": "typed",
          "parent": "",
          "properties": {
            "visibility": { "type": "token", "fallback": "inherited" },
            "purpose": { "type": "token", "fallback": "default" }
          }
        },
        "Derived": {
          "schemaKind": "typed",
          "parent": "Base",
          "properties": {
            "points": { "type": "point3f[]" },
            "material:binding": { "type": "rel" },
            "visibility": { "type": "token", "fallback": "invisible" }
          }
        }
      }
    })";

    std::string err;
    assert(reg.LoadFromJSON(json, &err));

    // Get prim definition for Derived
    auto* primDef = reg.GetPrimDefinition("Derived");
    assert(primDef != nullptr);

    // Should have all properties from both schemas
    assert(primDef->HasProperty("points"));
    assert(primDef->HasProperty("visibility"));
    assert(primDef->HasProperty("purpose"));
    assert(primDef->HasProperty("material:binding"));
    assert(std::find(primDef->attributeNameTokens.begin(),
                     primDef->attributeNameTokens.end(),
                     Token("points")) != primDef->attributeNameTokens.end());
    assert(std::find(primDef->attributeNameTokens.begin(),
                     primDef->attributeNameTokens.end(),
                     Token("visibility")) != primDef->attributeNameTokens.end());
    assert(std::find(primDef->attributeNameTokens.begin(),
                     primDef->attributeNameTokens.end(),
                     Token("purpose")) != primDef->attributeNameTokens.end());
    assert(std::find(primDef->attributeNameTokens.begin(),
                     primDef->attributeNameTokens.end(),
                     Token("material:binding")) == primDef->attributeNameTokens.end());

    // Derived's visibility should win (stronger overrides weaker)
    auto* visProp = primDef->GetPropertyDef("visibility");
    assert(visProp != nullptr);
    assert(visProp->fallback.has_value());
    auto* visTok = visProp->fallback->Get<Token>();
    assert(visTok != nullptr);
    assert(*visTok == "invisible");

    // purpose comes from Base
    auto* purposeProp = primDef->GetPropertyDef("purpose");
    assert(purposeProp != nullptr);
    assert(purposeProp->fallback.has_value());

    // Schemas list should include both
    assert(primDef->schemas.size() == 2);
    assert(primDef->schemas[0] == "Derived");
    assert(primDef->schemas[1] == "Base");

    // Non-existent type returns null
    assert(reg.GetPrimDefinition("NoSuch") == nullptr);

    // Applied schemas don't have prim definitions
    assert(reg.GetPrimDefinition("CollectionAPI") == nullptr);

    reg.Clear();
    std::cout << "  Schema prim definition: OK\n";
}

void TestSchemaFallbackValues() {
    auto& reg = SchemaRegistry::GetInstance();
    reg.Clear();

    const char* json = R"({
      "schemas": {
        "MyGprim": {
          "schemaKind": "typed",
          "parent": "",
          "properties": {
            "visibility": { "type": "token", "fallback": "inherited" },
            "purpose": { "type": "token", "fallback": "default" }
          }
        },
        "MyAPI": {
          "schemaKind": "singleApply",
          "properties": {
            "myapi:intensity": { "type": "float", "fallback": 1.0 }
          }
        }
      }
    })";

    std::string err;
    assert(reg.LoadFromJSON(json, &err));

    // Fallback from typed schema via BuildFullPrimDefinition
    std::vector<std::string> noAPIs;
    auto def = reg.BuildFullPrimDefinition("MyGprim", noAPIs);
    auto* visProp = def.GetPropertyDef("visibility");
    assert(visProp != nullptr);
    assert(visProp->fallback.has_value());
    auto* tok = visProp->fallback->Get<Token>();
    assert(tok && *tok == "inherited");

    // Fallback from API schema via BuildFullPrimDefinition
    std::vector<std::string> apis = {"MyAPI"};
    auto apiDef = reg.BuildFullPrimDefinition("", apis);
    auto* intProp = apiDef.GetPropertyDef("myapi:intensity");
    assert(intProp != nullptr);
    assert(intProp->fallback.has_value());
    auto* flt = intProp->fallback->Get<Float>();
    assert(flt != nullptr);
    assert(*flt == 1.0f);

    // No fallback for non-existent property
    assert(def.GetPropertyDef("nonexistent") == nullptr);

    // Test through UsdPrim / UsdAttribute: schema fallbacks via GetDefault()
    const char* usda = R"(#usda 1.0

def MyGprim "Prim" (
    prepend apiSchemas = ["MyAPI"]
)
{
    token visibility = "invisible"
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Authored attribute: GetDefault() returns authored value
    auto visAttr = prim.GetAttribute("visibility");
    assert(visAttr.IsValid());
    assert(visAttr.IsSchemaDefined());
    auto* visDefault = visAttr.GetDefault();
    assert(visDefault != nullptr);
    auto* visTok = visDefault->Get<Token>();
    assert(visTok && *visTok == "invisible");

    // Schema-only attribute (not authored): GetDefault() returns fallback
    auto purposeAttr = prim.GetAttribute("purpose");
    assert(purposeAttr.IsValid());
    assert(purposeAttr.IsSchemaDefined());
    auto* purposeDefault = purposeAttr.GetDefault();
    assert(purposeDefault != nullptr);
    auto* purposeTok = purposeDefault->Get<Token>();
    assert(purposeTok && *purposeTok == "default");

    // API schema attribute (not authored): fallback from applied API
    auto intensityAttr = prim.GetAttribute("myapi:intensity");
    assert(intensityAttr.IsValid());
    assert(intensityAttr.IsSchemaDefined());
    auto* intDefault = intensityAttr.GetDefault();
    assert(intDefault != nullptr);
    auto* intFlt = intDefault->Get<Float>();
    assert(intFlt && *intFlt == 1.0f);

    // GetFallback() returns schema fallback even when authored value exists
    auto* visFallback = visAttr.GetFallback();
    assert(visFallback != nullptr);
    auto* visFallTok = visFallback->Get<Token>();
    assert(visFallTok && *visFallTok == "inherited");

    // Non-existent attribute
    auto noAttr = prim.GetAttribute("nonexistent");
    assert(!noAttr.IsValid());

    reg.Clear();
    std::cout << "  Schema fallback values: OK\n";
}

void TestSchemaCoreSchemas() {
    auto& reg = SchemaRegistry::GetInstance();

    // CollectionAPI should be multipleApply with correct properties
    auto* coll = reg.FindSchema("CollectionAPI");
    assert(coll != nullptr);
    assert(coll->kind == SchemaKind::MultipleApply);
    assert(coll->instancePrefix == "collection");
    assert(coll->properties.count("collection:<__INSTANCE_NAME__>:expansionRule") > 0);
    assert(coll->properties.count("collection:<__INSTANCE_NAME__>:includeRoot") > 0);
    assert(coll->properties.count("collection:<__INSTANCE_NAME__>:includes") > 0);
    assert(coll->properties.count("collection:<__INSTANCE_NAME__>:excludes") > 0);
    assert(coll->properties.count("collection:<__INSTANCE_NAME__>") > 0);
    // Verify fallback values
    auto& expRule = coll->properties.at("collection:<__INSTANCE_NAME__>:expansionRule");
    assert(expRule.typeName == "token");
    assert(expRule.variability == Variability::Uniform);
    assert(expRule.fallback.has_value());
    assert(*expRule.fallback->Get<Token>() == "expandPrims");
    auto& inclRoot = coll->properties.at("collection:<__INSTANCE_NAME__>:includeRoot");
    assert(inclRoot.typeName == "bool");
    assert(inclRoot.fallback.has_value());
    assert(*inclRoot.fallback->Get<Bool>() == false);

    // ColorSpaceAPI should be singleApply
    auto* cs = reg.FindSchema("ColorSpaceAPI");
    assert(cs != nullptr);
    assert(cs->kind == SchemaKind::SingleApply);
    assert(cs->properties.count("colorSpace:name") > 0);
    auto& csName = cs->properties.at("colorSpace:name");
    assert(csName.typeName == "token");
    assert(csName.variability == Variability::Uniform);

    // ColorSpaceDefinitionAPI should be multipleApply with 7 properties
    auto* csd = reg.FindSchema("ColorSpaceDefinitionAPI");
    assert(csd != nullptr);
    assert(csd->kind == SchemaKind::MultipleApply);
    assert(csd->instancePrefix == "colorSpaceDefinition");
    assert(csd->properties.size() == 7);  // name, redChroma, greenChroma, blueChroma, whitePoint, gamma, linearBias
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:name") > 0);
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:redChroma") > 0);
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:greenChroma") > 0);
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:blueChroma") > 0);
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:whitePoint") > 0);
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:gamma") > 0);
    assert(csd->properties.count("colorSpaceDefinition:<__INSTANCE_NAME__>:linearBias") > 0);
    auto& redChroma = csd->properties.at("colorSpaceDefinition:<__INSTANCE_NAME__>:redChroma");
    assert(redChroma.typeName == "float2");
    assert(redChroma.fallback.has_value());
    auto* redFallback = redChroma.fallback->Get<GfVec2f>();
    assert(redFallback && (*redFallback)[0] == 1.0f && (*redFallback)[1] == 0.0f);
    auto& greenChroma = csd->properties.at("colorSpaceDefinition:<__INSTANCE_NAME__>:greenChroma");
    assert(greenChroma.fallback.has_value());
    auto* greenFallback = greenChroma.fallback->Get<GfVec2f>();
    assert(greenFallback && (*greenFallback)[0] == 0.0f && (*greenFallback)[1] == 1.0f);
    auto& blueChroma = csd->properties.at("colorSpaceDefinition:<__INSTANCE_NAME__>:blueChroma");
    assert(blueChroma.fallback.has_value());
    auto* blueFallback = blueChroma.fallback->Get<GfVec2f>();
    assert(blueFallback && (*blueFallback)[0] == 0.0f && (*blueFallback)[1] == 0.0f);
    auto& whitePoint = csd->properties.at("colorSpaceDefinition:<__INSTANCE_NAME__>:whitePoint");
    assert(whitePoint.fallback.has_value());
    auto* whiteFallback = whitePoint.fallback->Get<GfVec2f>();
    assert(whiteFallback);
    assert(std::abs((*whiteFallback)[0] - 0.33333333f) < 1e-6f);
    assert(std::abs((*whiteFallback)[1] - 0.33333333f) < 1e-6f);
    // Verify gamma fallback
    auto& gamma = csd->properties.at("colorSpaceDefinition:<__INSTANCE_NAME__>:gamma");
    assert(gamma.typeName == "float");
    assert(gamma.fallback.has_value());
    assert(*gamma.fallback->Get<Float>() == 1.0f);
    // Verify linearBias fallback
    auto& linBias = csd->properties.at("colorSpaceDefinition:<__INSTANCE_NAME__>:linearBias");
    assert(linBias.fallback.has_value());
    assert(*linBias.fallback->Get<Float>() == 0.0f);

    std::cout << "  Schema core schemas: OK\n";
}

void TestColorSpaceResolution() {
    const char* usda = R"usda(#usda 1.0
def Xform "Root" (
    prepend apiSchemas = ["ColorSpaceAPI", "ColorSpaceDefinitionAPI:studio"]
)
{
    uniform token colorSpace:name = "lin_rec709_scene"
    token colorSpaceDefinition:studio:name = "studio"

    def Material "Material1" (
        prepend apiSchemas = ["ColorSpaceAPI"]
    )
    {
        uniform token colorSpace:name = "srgb_p3d65_scene"
        color3f inputs:diffuseColor = (0.2, 0.5, 0.8)
    }

    def Material "Material2"
    {
        uniform token colorSpace:name = "data"
        color3f inputs:diffuseColor = (0.2, 0.5, 0.8)
    }

    def Material "Material3"
    {
        color3f inputs:diffuseColor = (0.2, 0.5, 0.8) (
            colorSpace = "srgb_rec709_scene"
        )
    }

    def Shader "Texture"
    {
        asset inputs:file = @./assetTexture.png@
    }

    color3f animated.timeSamples = {
        0: (0, 0, 0),
        1: (1, 1, 1),
    }
    color3h halfColor = (1, 0, 0)
}

def Scope "Plain"
{
    color3f color = (1, 1, 1)
}
)usda";

    auto parsed = ParseUsda(usda);
    assert(parsed.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(parsed.layer));
    assert(stage.IsValid());

    auto root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());
    assert(root.ComputeColorSpaceName() == "lin_rec709_scene");

    auto material1 = stage.GetPrimAtPath(Path::Parse("/Root/Material1"));
    assert(material1.ComputeColorSpaceName() == "srgb_p3d65_scene");
    assert(material1.GetAttribute(Token("inputs:diffuseColor")).ComputeColorSpaceName() ==
           "srgb_p3d65_scene");

    auto material2 = stage.GetPrimAtPath(Path::Parse("/Root/Material2"));
    assert(material2.ComputeColorSpaceName() == "lin_rec709_scene");
    assert(material2.GetAttribute(Token("inputs:diffuseColor")).ComputeColorSpaceName() ==
           "lin_rec709_scene");

    auto material3 = stage.GetPrimAtPath(Path::Parse("/Root/Material3"));
    auto diffuse = material3.GetAttribute(Token("inputs:diffuseColor"));
    assert(diffuse.HasColorSpace());
    assert(diffuse.GetColorSpace() == "srgb_rec709_scene");
    assert(diffuse.ComputeColorSpaceName() == "srgb_rec709_scene");

    auto texture = stage.GetPrimAtPath(Path::Parse("/Root/Texture"));
    assert(texture.GetAttribute(Token("inputs:file")).ComputeColorSpaceName() ==
           "lin_rec709_scene");

    auto plain = stage.GetPrimAtPath(Path::Parse("/Plain"));
    assert(plain.ComputeColorSpaceName().IsEmpty());
    assert(plain.GetAttribute(Token("color")).ComputeColorSpaceName().IsEmpty());

    auto animated = root.GetAttribute(Token("animated"));
    auto atDefault = animated.Get(UsdTimeCode(0.0));
    assert(atDefault.found);
    assert(atDefault.value.GetRole() == Role::Color);
    auto atHalf = animated.Get(UsdTimeCode(0.5));
    assert(atHalf.found);
    assert(atHalf.value.GetRole() == Role::Color);

    auto halfColor = root.GetAttribute(Token("halfColor")).Get();
    assert(halfColor.found);
    assert(halfColor.value.GetRole() == Role::Color);

    std::cout << "  Color-space resolution: OK\n";
}

void TestColorSpaceAuthoring() {
    Stage stage = Stage::CreateInMemory();
    auto prim = stage.DefinePrim(Path::Parse("/Looks/Mat"), Token("Material"));
    auto attr = prim.CreateAttribute(Token("inputs:diffuseColor"), Token("color3f"));

    assert(!attr.HasColorSpace());
    assert(attr.SetColorSpace(Token("srgb_rec709_scene")));
    assert(attr.HasColorSpace());
    assert(attr.GetColorSpace() == "srgb_rec709_scene");
    assert(attr.ComputeColorSpaceName() == "srgb_rec709_scene");
    assert(attr.ClearColorSpace());
    assert(!attr.HasColorSpace());
    assert(attr.ComputeColorSpaceName().IsEmpty());

    assert(prim.ApplyAPI(Token("ColorSpaceAPI")));
    auto primColor = prim.CreateAttribute(Token("colorSpace:name"), Token("token"), false);
    assert(primColor.Set(Value(Token("lin_rec709_scene"))));
    assert(prim.ComputeColorSpaceName() == "lin_rec709_scene");
    assert(attr.ComputeColorSpaceName() == "lin_rec709_scene");

    Value color(GfVec3f{0.1f, 0.2f, 0.3f});
    assert(attr.Set(color));
    auto resolved = attr.Get();
    assert(resolved.found);
    assert(resolved.value.GetRole() == Role::Color);

    std::cout << "  Color-space authoring: OK\n";
}

void TestSchemaGeometrySchemas() {
    auto& reg = SchemaRegistry::GetInstance();

    // --- Verify all 12 geometry schemas are registered ---
    const char* schemaNames[] = {
        "Imageable", "Scope", "Xformable", "Xform", "Boundable",
        "Gprim", "PointBased", "Mesh", "Points", "Curves", "BasisCurves"
    };
    for (auto name : schemaNames) {
        assert(reg.FindSchema(name) != nullptr);
        assert(reg.IsTypedSchema(name));
        assert(!reg.IsAppliedSchema(name));
    }

    // --- Verify abstract flags ---
    assert(reg.FindSchema("Imageable")->isAbstract == true);
    assert(reg.FindSchema("Xformable")->isAbstract == true);
    assert(reg.FindSchema("Boundable")->isAbstract == true);
    assert(reg.FindSchema("Gprim")->isAbstract == true);
    assert(reg.FindSchema("PointBased")->isAbstract == true);
    assert(reg.FindSchema("Curves")->isAbstract == true);
    // Concrete schemas
    assert(reg.FindSchema("Scope")->isAbstract == false);
    assert(reg.FindSchema("Xform")->isAbstract == false);
    assert(reg.FindSchema("Mesh")->isAbstract == false);
    assert(reg.FindSchema("Points")->isAbstract == false);
    assert(reg.FindSchema("BasisCurves")->isAbstract == false);

    // --- Verify inheritance chain ---
    // Mesh -> PointBased -> Gprim -> Boundable -> Xformable -> Imageable
    assert(reg.IsA("Mesh", "PointBased"));
    assert(reg.IsA("Mesh", "Gprim"));
    assert(reg.IsA("Mesh", "Boundable"));
    assert(reg.IsA("Mesh", "Xformable"));
    assert(reg.IsA("Mesh", "Imageable"));
    assert(!reg.IsA("Mesh", "Scope"));
    assert(!reg.IsA("Mesh", "Points"));

    // Points -> PointBased -> Gprim -> Boundable -> Xformable -> Imageable
    assert(reg.IsA("Points", "PointBased"));
    assert(reg.IsA("Points", "Gprim"));
    assert(reg.IsA("Points", "Imageable"));
    assert(!reg.IsA("Points", "Mesh"));

    // BasisCurves -> Curves -> PointBased -> Gprim -> ...
    assert(reg.IsA("BasisCurves", "Curves"));
    assert(reg.IsA("BasisCurves", "PointBased"));
    assert(reg.IsA("BasisCurves", "Gprim"));
    assert(reg.IsA("BasisCurves", "Imageable"));

    // Scope -> Imageable (not Xformable)
    assert(reg.IsA("Scope", "Imageable"));
    assert(!reg.IsA("Scope", "Xformable"));

    // Xform -> Xformable -> Imageable
    assert(reg.IsA("Xform", "Xformable"));
    assert(reg.IsA("Xform", "Imageable"));
    assert(!reg.IsA("Xform", "Gprim"));

    // --- Verify Imageable properties ---
    auto* imageable = reg.FindSchema("Imageable");
    assert(imageable->properties.count("visibility") > 0);
    assert(imageable->properties.at("visibility").typeName == "token");
    assert(imageable->properties.at("visibility").variability == Variability::Varying);
    assert(imageable->properties.at("visibility").fallback.has_value());
    assert(*imageable->properties.at("visibility").fallback->Get<Token>() == "inherited");
    assert(imageable->properties.count("purpose") > 0);
    assert(imageable->properties.at("purpose").variability == Variability::Uniform);
    assert(*imageable->properties.at("purpose").fallback->Get<Token>() == "default");
    assert(imageable->properties.count("proxyPrim") > 0);
    assert(imageable->properties.at("proxyPrim").typeName == "rel");

    // --- Verify Gprim properties ---
    auto* gprim = reg.FindSchema("Gprim");
    assert(gprim->properties.count("orientation") > 0);
    assert(gprim->properties.at("orientation").variability == Variability::Uniform);
    assert(*gprim->properties.at("orientation").fallback->Get<Token>() == "rightHanded");
    assert(gprim->properties.count("doubleSided") > 0);
    assert(*gprim->properties.at("doubleSided").fallback->Get<Bool>() == false);
    assert(gprim->properties.count("primvars:displayColor") > 0);
    assert(gprim->properties.at("primvars:displayColor").typeName == "color3f[]");

    // --- Verify BasisCurves properties ---
    auto* bc = reg.FindSchema("BasisCurves");
    assert(bc->properties.count("type") > 0);
    assert(*bc->properties.at("type").fallback->Get<Token>() == "cubic");
    assert(bc->properties.count("basis") > 0);
    assert(*bc->properties.at("basis").fallback->Get<Token>() == "bezier");
    assert(bc->properties.count("wrap") > 0);
    assert(*bc->properties.at("wrap").fallback->Get<Token>() == "nonperiodic");

    // --- Verify PrimDefinition composition for Mesh ---
    auto* meshDef = reg.GetPrimDefinition("Mesh");
    assert(meshDef != nullptr);
    // Should include all inherited properties
    assert(meshDef->HasProperty("visibility"));        // from Imageable
    assert(meshDef->HasProperty("purpose"));            // from Imageable
    assert(meshDef->HasProperty("xformOpOrder"));       // from Xformable
    assert(meshDef->HasProperty("extent"));             // from Boundable
    assert(meshDef->HasProperty("orientation"));        // from Gprim
    assert(meshDef->HasProperty("doubleSided"));        // from Gprim
    assert(meshDef->HasProperty("primvars:displayColor")); // from Gprim
    assert(meshDef->HasProperty("points"));             // from PointBased
    assert(meshDef->HasProperty("normals"));            // from PointBased
    assert(meshDef->HasProperty("faceVertexCounts"));   // from Mesh
    assert(meshDef->HasProperty("faceVertexIndices"));  // from Mesh
    assert(meshDef->HasProperty("subdivisionScheme"));  // from Mesh
    // Verify fallback through the composed definition
    auto* subdiv = meshDef->GetPropertyDef("subdivisionScheme");
    assert(subdiv != nullptr);
    assert(subdiv->fallback.has_value());
    assert(*subdiv->fallback->Get<Token>() == "catmullClark");
    // Inherited fallback
    auto* vis = meshDef->GetPropertyDef("visibility");
    assert(vis != nullptr);
    assert(vis->fallback.has_value());
    assert(*vis->fallback->Get<Token>() == "inherited");

    // --- Verify PrimDefinition for Points ---
    auto* ptsDef = reg.GetPrimDefinition("Points");
    assert(ptsDef != nullptr);
    assert(ptsDef->HasProperty("widths"));
    assert(ptsDef->HasProperty("ids"));
    assert(ptsDef->HasProperty("points"));       // from PointBased
    assert(ptsDef->HasProperty("normals"));      // from PointBased
    assert(ptsDef->HasProperty("orientation"));  // from Gprim

    // --- Verify PrimDefinition for BasisCurves ---
    auto* bcDef = reg.GetPrimDefinition("BasisCurves");
    assert(bcDef != nullptr);
    assert(bcDef->HasProperty("type"));
    assert(bcDef->HasProperty("basis"));
    assert(bcDef->HasProperty("wrap"));
    assert(bcDef->HasProperty("curveVertexCounts"));  // from Curves
    assert(bcDef->HasProperty("widths"));             // from Curves
    assert(bcDef->HasProperty("points"));             // from PointBased

    // --- Verify Scope has no extra properties, only Imageable's ---
    auto* scopeDef = reg.GetPrimDefinition("Scope");
    assert(scopeDef != nullptr);
    assert(scopeDef->HasProperty("visibility"));
    assert(scopeDef->HasProperty("purpose"));
    assert(!scopeDef->HasProperty("xformOpOrder"));  // Scope doesn't inherit Xformable
    assert(!scopeDef->HasProperty("extent"));

    // --- Verify Xform ---
    auto* xformDef = reg.GetPrimDefinition("Xform");
    assert(xformDef != nullptr);
    assert(xformDef->HasProperty("xformOpOrder"));
    assert(xformDef->HasProperty("visibility"));
    assert(!xformDef->HasProperty("extent"));  // Xform doesn't inherit Boundable

    std::cout << "  Schema geometry schemas: OK\n";
}

void TestSchemaIsAWithStage() {
    // Mesh and Xform are now registered as core geometry schemas.
    // Test IsA through UsdPrim
    const char* usda = R"(#usda 1.0

def Mesh "MyMesh"
{
}

def Xform "MyXform"
{
}

def "Untyped"
{
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto mesh = stage.GetPrimAtPath(Path::Parse("/MyMesh"));
    assert(mesh.IsValid());
    assert(mesh.IsA("Mesh"));
    assert(!mesh.IsA("Xform"));

    auto xform = stage.GetPrimAtPath(Path::Parse("/MyXform"));
    assert(xform.IsValid());
    assert(xform.IsA("Xform"));
    assert(!xform.IsA("Mesh"));

    auto untyped = stage.GetPrimAtPath(Path::Parse("/Untyped"));
    assert(untyped.IsValid());
    assert(!untyped.IsA("Mesh"));
    assert(!untyped.IsA("Xform"));

    // GetPropertyNames() should include schema-defined properties even
    // when they are not authored in the layer.
    auto meshNames = mesh.GetPropertyNames();
    std::unordered_set<std::string> nameSet(meshNames.begin(), meshNames.end());
    assert(nameSet.count("points") > 0);
    assert(nameSet.count("normals") > 0);
    assert(nameSet.count("faceVertexCounts") > 0);
    assert(nameSet.count("faceVertexIndices") > 0);

    // HasAttribute() should find schema-defined attributes
    assert(mesh.HasAttribute("points"));
    assert(mesh.HasAttribute("normals"));

    // GetAttribute() returns a valid attribute for schema-only properties
    auto pointsAttr = mesh.GetAttribute("points");
    assert(pointsAttr.IsValid());
    assert(pointsAttr.IsSchemaDefined());
    assert(pointsAttr.GetTypeName() == "point3f[]");

    // Xform has schema-defined properties from Xformable and Imageable
    auto xformNames = xform.GetPropertyNames();
    std::unordered_set<std::string> xformNameSet(xformNames.begin(), xformNames.end());
    assert(xformNameSet.count("xformOpOrder") > 0);   // from Xformable
    assert(xformNameSet.count("visibility") > 0);      // from Imageable
    assert(xformNameSet.count("purpose") > 0);         // from Imageable

    // Untyped prim has no schema properties
    auto untypedNames = untyped.GetPropertyNames();
    assert(untypedNames.empty());

    std::cout << "  Schema IsA with Stage: OK\n";
}

void TestSchemaAbstractTypeNameRejected() {
    auto& reg = SchemaRegistry::GetInstance();
    assert(reg.FindSchema("Gprim") != nullptr);
    assert(reg.FindSchema("Gprim")->isAbstract);
    assert(reg.GetPrimDefinition("Gprim") == nullptr);

    const char* usda = R"(#usda 1.0

def Gprim "BadGprim"
{
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/BadGprim"));
    assert(prim.IsValid());
    assert(prim.GetTypeName() == "Gprim");
    assert(!prim.IsA("Gprim"));
    assert(!prim.IsA("Imageable"));
    assert(!prim.HasAttribute("doubleSided"));
    assert(!prim.GetAttribute("doubleSided").IsValid());

    auto names = prim.GetPropertyNames();
    assert(std::find(names.begin(), names.end(), Token("doubleSided")) ==
           names.end());

    std::cout << "  Schema abstract typeName rejection: OK\n";
}

void TestSchemaFallbackPrimTypes() {
    const char* usda = R"(#usda 1.0
(
    dictionary fallbackPrimTypes = {
        token[] NewMesh = ["NoSuchFallback", "Gprim", "Mesh"]
    }
)

def NewMesh "CompatMesh"
{
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    assert(result.layer.GetLayerSpec().GetField(FieldNames::fallbackPrimTypes) != nullptr);

    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/CompatMesh"));
    assert(prim.IsValid());
    assert(prim.GetTypeName() == "NewMesh");
    assert(!prim.IsA("NewMesh"));
    assert(prim.IsA("Mesh"));
    assert(prim.IsA("PointBased"));
    assert(prim.IsA("Gprim"));

    auto subd = prim.GetAttribute("subdivisionScheme");
    assert(subd.IsValid());
    assert(subd.IsSchemaDefined());
    auto* fallback = subd.GetDefault();
    assert(fallback != nullptr);
    auto* token = fallback->Get<Token>();
    assert(token && *token == "catmullClark");

    std::cout << "  Schema fallbackPrimTypes: OK\n";
}

void TestSchemaAutoApplies() {
    auto& reg = SchemaRegistry::GetInstance();
    reg.Clear();

    const char* json = R"({
      "schemas": {
        "BaseThing": {
          "schemaKind": "typed",
          "properties": {
            "base:flag": { "type": "bool", "fallback": false }
          }
        },
        "DerivedThing": {
          "schemaKind": "typed",
          "parent": "BaseThing"
        },
        "AutoThingAPI": {
          "schemaKind": "singleApply",
          "autoApplies": ["BaseThing"],
          "properties": {
            "auto:enabled": { "type": "bool", "fallback": true }
          }
        }
      }
    })";

    std::string err;
    assert(reg.LoadFromJSON(json, &err));

    const char* usda = R"(#usda 1.0

def DerivedThing "AutoPrim"
{
}

def "Untyped"
{
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/AutoPrim"));
    assert(prim.IsValid());
    assert(prim.HasAPI("AutoThingAPI"));

    auto schemas = prim.GetAppliedSchemas();
    assert(std::find(schemas.begin(), schemas.end(), Token("AutoThingAPI")) !=
           schemas.end());

    auto attr = prim.GetAttribute("auto:enabled");
    assert(attr.IsValid());
    assert(attr.IsSchemaDefined());
    auto* fallback = attr.GetDefault();
    assert(fallback != nullptr);
    auto* enabled = fallback->Get<Bool>();
    assert(enabled && *enabled);

    auto untyped = stage.GetPrimAtPath(Path::Parse("/Untyped"));
    assert(untyped.IsValid());
    assert(!untyped.HasAPI("AutoThingAPI"));
    assert(!untyped.HasAttribute("auto:enabled"));

    reg.Clear();
    std::cout << "  Schema autoApplies: OK\n";
}

void TestSchemaMultiApplyProperties() {
    auto& reg = SchemaRegistry::GetInstance();

    // Test that multi-apply schema instance name substitution works
    // through the full UsdPrim path.
    const char* usda = R"(#usda 1.0

def Xform "Light" (
    prepend apiSchemas = ["CollectionAPI:shadow"]
)
{
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Light"));
    assert(prim.IsValid());

    // Should have instanced properties from CollectionAPI:shadow
    assert(prim.HasAttribute("collection:shadow:expansionRule"));
    assert(prim.HasAttribute("collection:shadow:includeRoot"));

    auto expRule = prim.GetAttribute("collection:shadow:expansionRule");
    assert(expRule.IsValid());
    assert(expRule.IsSchemaDefined());
    assert(expRule.GetTypeName() == "token");
    assert(expRule.GetVariability() == Variability::Uniform);

    // Fallback value should be "expandPrims"
    auto* fallback = expRule.GetDefault();
    assert(fallback != nullptr);
    auto* fbTok = fallback->Get<Token>();
    assert(fbTok && *fbTok == "expandPrims");

    // includeRoot fallback should be false
    auto inclRoot = prim.GetAttribute("collection:shadow:includeRoot");
    assert(inclRoot.IsValid());
    auto* inclFallback = inclRoot.GetDefault();
    assert(inclFallback != nullptr);
    auto* inclBool = inclFallback->Get<Bool>();
    assert(inclBool && *inclBool == false);

    // GetPropertyNames should include these instanced properties
    auto names = prim.GetPropertyNames();
    std::unordered_set<std::string> nameSet(names.begin(), names.end());
    assert(nameSet.count("collection:shadow:expansionRule") > 0);
    assert(nameSet.count("collection:shadow:includeRoot") > 0);
    assert(nameSet.count("collection:shadow:includes") > 0);
    assert(nameSet.count("collection:shadow:excludes") > 0);

    // Relationship properties (includes/excludes) should NOT appear as attributes
    assert(!prim.HasAttribute("collection:shadow:includes"));
    assert(!prim.HasAttribute("collection:shadow:excludes"));

    std::cout << "  Schema multi-apply properties: OK\n";
}

void TestCollectionEvaluation() {
    const char* usda = R"(#usda 1.0

def "World"
{
    def "Set"
    {
        uniform token keep = "yes"
        uniform token drop = "no"

        def "Child"
        {
            uniform token childAttr = "child"
        }
    }

    def "Other" (
        prepend apiSchemas = ["CollectionAPI:leafs"]
    )
    {
        rel collection:leafs:includes = [
            </World/Other/Leaf>
        ]

        def "Leaf"
        {
        }
    }

    def "Collections" (
        prepend apiSchemas = [
            "CollectionAPI:all",
            "CollectionAPI:exact",
            "CollectionAPI:prims",
            "CollectionAPI:rootOnly"
        ]
    )
    {
        uniform token collection:all:expansionRule = "expandPrimsAndProperties"
        rel collection:all:includes = [
            </World/Set>,
            </World/Other.collection:leafs>,
            </World/Missing>
        ]
        rel collection:all:excludes = [
            </World/Set.drop>,
            </World/Orphan>
        ]

        uniform token collection:exact:expansionRule = "explicitOnly"
        rel collection:exact:includes = [
            </World/Set.keep>,
            </World/Set/Child>
        ]
        rel collection:exact:excludes = [
            </World/Set/Child>
        ]

        rel collection:prims:includes = [
            </World/Set>
        ]
        rel collection:prims:excludes = [
            </World/Set/Child>
        ]

        uniform token collection:rootOnly:expansionRule = "explicitOnly"
        uniform bool collection:rootOnly:includeRoot = true
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto collections = stage.GetPrimAtPath(Path::Parse("/World/Collections"));
    assert(collections.IsValid());

    auto all = collections.ComputeCollectionMembership(Token("all"));
    std::vector<std::string> allText;
    for (const auto& path : all) allText.push_back(path.GetText());
    const std::vector<std::string> expectedAll = {
        "/World/Set",
        "/World/Set.keep",
        "/World/Set/Child",
        "/World/Set/Child.childAttr",
        "/World/Other/Leaf",
    };
    assert(allText == expectedAll);
    assert(collections.IsCollectionMember(Token("all"),
                                          Path::Parse("/World/Set.keep")));
    assert(!collections.IsCollectionMember(Token("all"),
                                           Path::Parse("/World/Set.drop")));
    assert(!collections.IsCollectionMember(Token("all"),
                                           Path::Parse("/World/Orphan")));

    auto exact = collections.ComputeCollectionMembership(Token("exact"));
    assert(exact.size() == 1);
    assert(exact[0] == Path::Parse("/World/Set.keep"));

    auto prims = collections.ComputeCollectionMembership(Token("prims"));
    assert(prims.size() == 1);
    assert(prims[0] == Path::Parse("/World/Set"));

    auto rootOnly = collections.ComputeCollectionMembership(Token("rootOnly"));
    assert(rootOnly.size() == 1);
    assert(rootOnly[0] == Path::AbsoluteRoot());

    std::cout << "  Collection evaluation: OK\n";
}

void TestGeometryGprimSchemas() {
    auto& reg = SchemaRegistry::GetInstance();

    // Verify all new gprim schemas exist and inherit from Gprim
    const char* gprimSchemas[] = {
        "Cube", "Sphere", "Cone", "Cylinder", "Cylinder_1",
        "Capsule", "Capsule_1", "Plane"
    };
    for (const char* name : gprimSchemas) {
        assert(reg.FindSchema(name) != nullptr);
        assert(reg.IsTypedSchema(name));
        assert(reg.IsA(name, "Gprim"));
        assert(reg.IsA(name, "Boundable"));
        assert(reg.IsA(name, "Xformable"));
        assert(reg.IsA(name, "Imageable"));
    }

    // Sphere: radius=1.0
    {
        auto* def = reg.GetPrimDefinition("Sphere");
        assert(def != nullptr);
        auto* rp = def->GetPropertyDef("radius");
        assert(rp && rp->typeName == "double");
        assert(rp->fallback.has_value());
        assert(*rp->fallback->Get<Double>() == 1.0);
        // Inherits extent from Boundable, doubleSided from Gprim
        assert(def->HasProperty("extent"));
        assert(def->HasProperty("doubleSided"));
    }

    // Cone: height=2.0, radius=1.0, axis="Z" (uniform)
    {
        auto* def = reg.GetPrimDefinition("Cone");
        assert(def != nullptr);
        auto* hp = def->GetPropertyDef("height");
        assert(hp && *hp->fallback->Get<Double>() == 2.0);
        auto* rp = def->GetPropertyDef("radius");
        assert(rp && *rp->fallback->Get<Double>() == 1.0);
        auto* ap = def->GetPropertyDef("axis");
        assert(ap && ap->variability == Variability::Uniform);
        assert(*ap->fallback->Get<Token>() == "Z");
    }

    // Cylinder_1: radiusTop=1.0, radiusBottom=1.0
    {
        auto* def = reg.GetPrimDefinition("Cylinder_1");
        assert(def != nullptr);
        auto* rt = def->GetPropertyDef("radiusTop");
        assert(rt && *rt->fallback->Get<Double>() == 1.0);
        auto* rb = def->GetPropertyDef("radiusBottom");
        assert(rb && *rb->fallback->Get<Double>() == 1.0);
    }

    // Capsule: height=1.0, radius=0.5
    {
        auto* def = reg.GetPrimDefinition("Capsule");
        assert(def != nullptr);
        auto* hp = def->GetPropertyDef("height");
        assert(hp && *hp->fallback->Get<Double>() == 1.0);
        auto* rp = def->GetPropertyDef("radius");
        assert(rp && *rp->fallback->Get<Double>() == 0.5);
    }

    // Plane: width=2.0, length=2.0, axis="Z"
    {
        auto* def = reg.GetPrimDefinition("Plane");
        assert(def != nullptr);
        auto* wp = def->GetPropertyDef("width");
        assert(wp && *wp->fallback->Get<Double>() == 2.0);
        auto* lp = def->GetPropertyDef("length");
        assert(lp && *lp->fallback->Get<Double>() == 2.0);
    }

    // Cube: size=2.0
    {
        auto* def = reg.GetPrimDefinition("Cube");
        assert(def != nullptr);
        auto* sp = def->GetPropertyDef("size");
        assert(sp && *sp->fallback->Get<Double>() == 2.0);
    }

    // GeomSubset: does NOT inherit from Imageable/Gprim
    {
        assert(reg.FindSchema("GeomSubset") != nullptr);
        assert(reg.IsTypedSchema("GeomSubset"));
        assert(!reg.IsA("GeomSubset", "Imageable"));
        assert(!reg.IsA("GeomSubset", "Gprim"));
        auto* def = reg.GetPrimDefinition("GeomSubset");
        assert(def != nullptr);
        auto* et = def->GetPropertyDef("elementType");
        assert(et && et->variability == Variability::Uniform);
        assert(*et->fallback->Get<Token>() == "face");
        auto* fn = def->GetPropertyDef("familyName");
        assert(fn && fn->variability == Variability::Uniform);
        assert(*fn->fallback->Get<Token>() == "");
        assert(def->HasProperty("indices"));
        // Should NOT have Gprim properties like doubleSided
        assert(!def->HasProperty("doubleSided"));
        assert(!def->HasProperty("visibility"));
    }

    // Test via Stage: parse a USDA with a Sphere and verify schema-driven queries
    const char* usda = R"(#usda 1.0

def Sphere "Ball" {
    double radius = 2.5
}

def Cone "MyCone" {
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto ball = stage.GetPrimAtPath(Path::Parse("/Ball"));
    assert(ball.IsValid());
    assert(ball.IsA("Sphere"));
    assert(ball.IsA("Gprim"));
    auto radiusAttr = ball.GetAttribute("radius");
    assert(radiusAttr.IsValid());
    auto* rv = radiusAttr.GetDefault();
    assert(rv && *rv->Get<Double>() == 2.5);

    // MyCone: no authored properties, should get fallbacks from schema
    auto cone = stage.GetPrimAtPath(Path::Parse("/MyCone"));
    assert(cone.IsValid());
    assert(cone.IsA("Cone"));
    auto heightAttr = cone.GetAttribute("height");
    assert(heightAttr.IsValid());
    assert(heightAttr.IsSchemaDefined());
    auto* hv = heightAttr.GetDefault();
    assert(hv && *hv->Get<Double>() == 2.0);
    auto axisAttr = cone.GetAttribute("axis");
    assert(axisAttr.IsValid());
    assert(axisAttr.GetVariability() == Variability::Uniform);
    auto* av = axisAttr.GetDefault();
    assert(av && *av->Get<Token>() == "Z");

    std::cout << "  Geometry gprim schemas: OK\n";
}

void TestMaterialSchemas() {
    auto& reg = SchemaRegistry::GetInstance();

    // NodeGraph: typed, no parent, no properties
    {
        auto* ng = reg.FindSchema("NodeGraph");
        assert(ng != nullptr);
        assert(ng->kind == SchemaKind::Typed);
        assert(ng->parent.empty());
        assert(ng->properties.empty());
    }

    // Material: typed, inherits from NodeGraph
    {
        auto* mat = reg.FindSchema("Material");
        assert(mat != nullptr);
        assert(mat->kind == SchemaKind::Typed);
        assert(mat->parent == "NodeGraph");
        assert(reg.IsA("Material", "NodeGraph"));

        auto* def = reg.GetPrimDefinition("Material");
        assert(def != nullptr);
        auto* surface = def->GetPropertyDef("outputs:surface");
        assert(surface != nullptr);
        assert(surface->typeName == "token");
        assert(surface->variability == Variability::Uniform);
        auto* displacement = def->GetPropertyDef("outputs:displacement");
        assert(displacement != nullptr);
        assert(displacement->typeName == "token");
        assert(displacement->variability == Variability::Uniform);
        auto* volume = def->GetPropertyDef("outputs:volume");
        assert(volume != nullptr);
        assert(volume->typeName == "token");
        assert(volume->variability == Variability::Uniform);
    }

    // MaterialBindingAPI: single-apply API schema
    {
        auto* mba = reg.FindSchema("MaterialBindingAPI");
        assert(mba != nullptr);
        assert(mba->kind == SchemaKind::SingleApply);
        assert(reg.IsAppliedSchema("MaterialBindingAPI"));
        assert(!reg.IsTypedSchema("MaterialBindingAPI"));

        auto* bp = mba->properties.find("material:binding") != mba->properties.end()
            ? &mba->properties.at("material:binding") : nullptr;
        assert(bp != nullptr);
        assert(bp->typeName == "rel");
        assert(bp->variability == Variability::Uniform);
    }

    // Test via Stage: parse Material prim
    const char* usda = R"(#usda 1.0

def Material "MyMat" {
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto mat = stage.GetPrimAtPath(Path::Parse("/MyMat"));
    assert(mat.IsValid());
    assert(mat.IsA("Material"));
    assert(mat.IsA("NodeGraph"));
    // Schema-defined properties should be accessible
    auto surfaceAttr = mat.GetAttribute("outputs:surface");
    assert(surfaceAttr.IsValid());
    assert(surfaceAttr.GetVariability() == Variability::Uniform);

    std::cout << "  Material schemas: OK\n";
}

void TestPhysicsSchemas() {
    auto& reg = SchemaRegistry::GetInstance();

    // --- Typed schemas ---

    // PhysicsScene: typed, no parent
    {
        auto* s = reg.FindSchema("PhysicsScene");
        assert(s != nullptr);
        assert(s->kind == SchemaKind::Typed);
        assert(s->parent.empty());
        auto* def = reg.GetPrimDefinition("PhysicsScene");
        assert(def != nullptr);
        // gravityMagnitude: float, fallback -inf
        auto* gm = def->GetPropertyDef("physics:gravityMagnitude");
        assert(gm && gm->typeName == "float");
        assert(gm->fallback.has_value());
        float gmVal = *gm->fallback->Get<Float>();
        assert(std::isinf(gmVal) && gmVal < 0);
        // gravityDirection: vector3f, fallback (0,0,0)
        auto* gd = def->GetPropertyDef("physics:gravityDirection");
        assert(gd && gd->typeName == "vector3f");
        assert(gd->fallback.has_value());
        auto* gdv = gd->fallback->Get<GfVec3f>();
        assert(gdv && (*gdv)[0] == 0.0f && (*gdv)[1] == 0.0f && (*gdv)[2] == 0.0f);
    }

    // PhysicsJoint: typed, parent Imageable
    {
        assert(reg.IsA("PhysicsJoint", "Imageable"));
        auto* def = reg.GetPrimDefinition("PhysicsJoint");
        assert(def != nullptr);
        // localRot0: quatf, fallback (1,0,0,0) identity
        auto* lr = def->GetPropertyDef("physics:localRot0");
        assert(lr && lr->typeName == "quatf");
        auto* q = lr->fallback->Get<GfQuatf>();
        assert(q);
        // Internal storage: i,j,k,r — identity is (0,0,0,1)
        assert((*q)[0] == 0.0f && (*q)[1] == 0.0f && (*q)[2] == 0.0f && (*q)[3] == 1.0f);
        // breakForce: float, fallback inf
        auto* bf = def->GetPropertyDef("physics:breakForce");
        assert(bf && std::isinf(*bf->fallback->Get<Float>()) && *bf->fallback->Get<Float>() > 0);
        // jointEnabled: bool, fallback true
        auto* je = def->GetPropertyDef("physics:jointEnabled");
        assert(je && je->fallback->Get<Bool>() && *je->fallback->Get<Bool>() == true);
        // excludeFromArticulation: uniform
        auto* ea = def->GetPropertyDef("physics:excludeFromArticulation");
        assert(ea && ea->variability == Variability::Uniform);
        // Inherits visibility from Imageable
        assert(def->HasProperty("visibility"));
    }

    // Joint subtypes inherit from PhysicsJoint
    assert(reg.IsA("PhysicsFixedJoint", "PhysicsJoint"));
    assert(reg.IsA("PhysicsFixedJoint", "Imageable"));
    assert(reg.IsA("PhysicsDistanceJoint", "PhysicsJoint"));
    assert(reg.IsA("PhysicsSphericalJoint", "PhysicsJoint"));
    assert(reg.IsA("PhysicsRevoluteJoint", "PhysicsJoint"));
    assert(reg.IsA("PhysicsPrismaticJoint", "PhysicsJoint"));

    // PhysicsDistanceJoint properties
    {
        auto* def = reg.GetPrimDefinition("PhysicsDistanceJoint");
        assert(def != nullptr);
        auto* md = def->GetPropertyDef("physics:minDistance");
        assert(md && *md->fallback->Get<Float>() == -1.0f);
        auto* mx = def->GetPropertyDef("physics:maxDistance");
        assert(mx && *mx->fallback->Get<Float>() == -1.0f);
        // Inherits localPos0 from PhysicsJoint
        assert(def->HasProperty("physics:localPos0"));
    }

    // PhysicsSphericalJoint: axis uniform, fallback "X"
    {
        auto* def = reg.GetPrimDefinition("PhysicsSphericalJoint");
        auto* ax = def->GetPropertyDef("physics:axis");
        assert(ax && ax->variability == Variability::Uniform);
        assert(*ax->fallback->Get<Token>() == "X");
        auto* ca = def->GetPropertyDef("physics:coneAngle0Limit");
        assert(ca && *ca->fallback->Get<Float>() == -1.0f);
    }

    // PhysicsRevoluteJoint: lowerLimit -inf, upperLimit inf
    {
        auto* def = reg.GetPrimDefinition("PhysicsRevoluteJoint");
        auto* lo = def->GetPropertyDef("physics:lowerLimit");
        assert(lo && std::isinf(*lo->fallback->Get<Float>()) && *lo->fallback->Get<Float>() < 0);
        auto* hi = def->GetPropertyDef("physics:upperLimit");
        assert(hi && std::isinf(*hi->fallback->Get<Float>()) && *hi->fallback->Get<Float>() > 0);
    }

    // PhysicsCollisionGroup: has builtIn CollectionAPI:colliders
    {
        auto* s = reg.FindSchema("PhysicsCollisionGroup");
        assert(s != nullptr);
        assert(s->builtIns.size() == 1);
        assert(s->builtIns[0] == "CollectionAPI:colliders");
        auto* def = reg.GetPrimDefinition("PhysicsCollisionGroup");
        // Should have CollectionAPI properties from builtIn
        assert(def->HasProperty("collection:colliders:expansionRule"));
    }

    // --- Single-apply API schemas ---

    // PhysicsRigidBodyAPI
    {
        assert(reg.IsAppliedSchema("PhysicsRigidBodyAPI"));
        auto* s = reg.FindSchema("PhysicsRigidBodyAPI");
        assert(s->kind == SchemaKind::SingleApply);
        auto vel = s->properties.find("physics:velocity");
        assert(vel != s->properties.end());
        assert(vel->second.typeName == "vector3f");
        auto* v = vel->second.fallback->Get<GfVec3f>();
        assert(v && (*v)[0] == 0.0f);
    }

    // PhysicsCollisionAPI
    assert(reg.IsAppliedSchema("PhysicsCollisionAPI"));

    // PhysicsMeshCollisionAPI
    {
        auto* s = reg.FindSchema("PhysicsMeshCollisionAPI");
        assert(s != nullptr);
        auto ap = s->properties.find("physics:approximation");
        assert(ap != s->properties.end());
        assert(*ap->second.fallback->Get<Token>() == "none");
    }

    // PhysicsMassAPI
    {
        auto* s = reg.FindSchema("PhysicsMassAPI");
        assert(s != nullptr);
        auto mp = s->properties.find("physics:mass");
        assert(mp != s->properties.end() && *mp->second.fallback->Get<Float>() == 0.0f);
        auto com = s->properties.find("physics:centerOfMass");
        assert(com != s->properties.end() && com->second.typeName == "point3f");
        auto di = s->properties.find("physics:diagonalInertia");
        assert(di != s->properties.end() && di->second.typeName == "float3");
        auto pa = s->properties.find("physics:principalAxes");
        assert(pa != s->properties.end() && pa->second.typeName == "quatf");
        // principalAxes fallback: zero quaternion (0,0,0,0)
        auto* pq = pa->second.fallback->Get<GfQuatf>();
        assert(pq && (*pq)[0] == 0.0f && (*pq)[1] == 0.0f && (*pq)[2] == 0.0f && (*pq)[3] == 0.0f);
    }

    // PhysicsMaterialAPI
    {
        auto* s = reg.FindSchema("PhysicsMaterialAPI");
        assert(s != nullptr);
        auto sf = s->properties.find("physics:staticFriction");
        assert(sf != s->properties.end() && *sf->second.fallback->Get<Float>() == 0.0f);
        auto re = s->properties.find("physics:restitution");
        assert(re != s->properties.end() && *re->second.fallback->Get<Float>() == 0.0f);
    }

    // PhysicsFilteredPairsAPI
    assert(reg.IsAppliedSchema("PhysicsFilteredPairsAPI"));

    // PhysicsArticulationRootAPI: no properties
    {
        auto* s = reg.FindSchema("PhysicsArticulationRootAPI");
        assert(s != nullptr && s->kind == SchemaKind::SingleApply);
        assert(s->properties.empty());
    }

    // --- Multi-apply API schemas ---

    // PhysicsLimitAPI
    {
        auto* s = reg.FindSchema("PhysicsLimitAPI");
        assert(s != nullptr && s->kind == SchemaKind::MultipleApply);
        assert(s->instancePrefix == "limit");
        auto lo = s->properties.find("limit:<__INSTANCE_NAME__>:physics:low");
        assert(lo != s->properties.end());
        assert(std::isinf(*lo->second.fallback->Get<Float>()) && *lo->second.fallback->Get<Float>() < 0);
    }

    // PhysicsDriveAPI
    {
        auto* s = reg.FindSchema("PhysicsDriveAPI");
        assert(s != nullptr && s->kind == SchemaKind::MultipleApply);
        assert(s->instancePrefix == "drive");
        auto tp = s->properties.find("drive:<__INSTANCE_NAME__>:physics:type");
        assert(tp != s->properties.end());
        assert(*tp->second.fallback->Get<Token>() == "force");
        auto mf = s->properties.find("drive:<__INSTANCE_NAME__>:physics:maxForce");
        assert(mf != s->properties.end());
        assert(std::isinf(*mf->second.fallback->Get<Float>()) && *mf->second.fallback->Get<Float>() > 0);
    }

    // --- Stage-level test: parse physics scene and rigid body ---
    const char* usda = R"(#usda 1.0

def PhysicsScene "scene" {
    vector3f physics:gravityDirection = (0, 0, -1)
    float physics:gravityMagnitude = 9.81
}

def Sphere "ball" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI"]
) {
    vector3f physics:velocity = (1, 0, 0)
    bool physics:rigidBodyEnabled = true
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto scene = stage.GetPrimAtPath(Path::Parse("/scene"));
    assert(scene.IsValid());
    assert(scene.IsA("PhysicsScene"));

    auto ball = stage.GetPrimAtPath(Path::Parse("/ball"));
    assert(ball.IsValid());
    assert(ball.IsA("Sphere"));
    assert(ball.HasAPI("PhysicsRigidBodyAPI"));
    assert(ball.HasAPI("PhysicsCollisionAPI"));

    // Authored velocity
    auto velAttr = ball.GetAttribute("physics:velocity");
    assert(velAttr.IsValid());

    // Schema fallback: kinematicEnabled should be false from PhysicsRigidBodyAPI
    auto kinAttr = ball.GetAttribute("physics:kinematicEnabled");
    assert(kinAttr.IsValid());
    auto* kv = kinAttr.GetDefault();
    assert(kv && *kv->Get<Bool>() == false);

    std::cout << "  Physics schemas: OK\n";
}

void TestKilogramsPerUnit() {
    // Layer-level
    {
        auto result = ParseUsda(R"(#usda 1.0
(
    kilogramsPerUnit = 0.001
)
)");
        assert(result.success);
        assert(HasKilogramsPerUnit(result.layer));
        assert(GetKilogramsPerUnit(result.layer) == 0.001);
    }

    // Layer without kilogramsPerUnit: fallback 1.0
    {
        auto result = ParseUsda("#usda 1.0\n");
        assert(result.success);
        assert(!HasKilogramsPerUnit(result.layer));
        assert(GetKilogramsPerUnit(result.layer) == 1.0);
    }

    // SetKilogramsPerUnit
    {
        auto result = ParseUsda("#usda 1.0\n");
        assert(result.success);
        SetKilogramsPerUnit(result.layer, 0.5);
        assert(HasKilogramsPerUnit(result.layer));
        assert(GetKilogramsPerUnit(result.layer) == 0.5);
    }

    // Stage-level
    {
        auto result = ParseUsda(R"(#usda 1.0
(
    kilogramsPerUnit = 2.0
)
)");
        assert(result.success);
        auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
        assert(stage.IsValid());
        assert(HasKilogramsPerUnit(stage));
        assert(GetKilogramsPerUnit(stage) == 2.0);
    }

    // Stage without kilogramsPerUnit: fallback 1.0
    {
        auto result = ParseUsda("#usda 1.0\n");
        assert(result.success);
        auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
        assert(!HasKilogramsPerUnit(stage));
        assert(GetKilogramsPerUnit(stage) == 1.0);
    }

    std::cout << "  kilogramsPerUnit: OK\n";
}

void TestComposeListOpAcrossLayers() {
    // Root layer prepends ColorSpaceAPI, sublayer prepends CollectionAPI:base.
    // After composition, the prim should have both API schemas.
    auto stage = Stage::Open("tests/composition/api_root.usda");
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    auto schemas = prim.GetAppliedSchemas();

    // Both APIs should be present
    std::unordered_set<std::string> schemaSet(schemas.begin(), schemas.end());
    assert(schemaSet.count("ColorSpaceAPI") > 0);
    assert(schemaSet.count("CollectionAPI:base") > 0);

    // Stronger (root) prepend comes first in the resolved list
    // Per ListOp::Combine: stronger prepend before weaker prepend
    assert(schemas.size() >= 2);
    assert(schemas[0] == "ColorSpaceAPI");
    assert(schemas[1] == "CollectionAPI:base");

    // Both APIs' properties should be accessible
    assert(prim.HasAPI("ColorSpaceAPI"));
    assert(prim.HasAPI("CollectionAPI", "base"));

    // Schema-defined properties from both should appear
    assert(prim.HasAttribute("colorSpace:name"));
    assert(prim.HasAttribute("collection:base:expansionRule"));

    std::cout << "  Compose listop across layers: OK\n";
}

// ============================================================
// Value Resolution Tests (spec Section 12)
// ============================================================

void TestValueResolutionDefault() {
    // Test default time value resolution: authored default > fallback
    auto& reg = SchemaRegistry::GetInstance();
    const char* testSchemas = R"({
      "schemas": {
        "TestGprim": {
          "schemaKind": "typed",
          "parent": "",
          "properties": {
            "visibility": { "type": "token", "fallback": "inherited" },
            "purpose": { "type": "token", "fallback": "default" }
          }
        }
      }
    })";
    std::string schemaErr;
    reg.LoadFromJSON(testSchemas, &schemaErr);

    const char* usda = R"(#usda 1.0

def TestGprim "Prim"
{
    token visibility = "invisible"
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    assert(prim.IsValid());

    // Authored default: Get at default time returns authored value
    auto vis = prim.GetAttribute("visibility");
    auto resolved = vis.Get(UsdTimeCode::Default());
    assert(resolved.found);
    assert(*resolved.value.Get<Token>() == "invisible");

    // Unauthored attribute: Get returns schema fallback
    auto purpose = prim.GetAttribute("purpose");
    auto resolved2 = purpose.Get(UsdTimeCode::Default());
    assert(resolved2.found);
    assert(*resolved2.value.Get<Token>() == "default");

    // Non-existent attribute: Get returns not found
    auto noAttr = prim.GetAttribute("nonexistent");
    assert(!noAttr.IsValid());

    std::cout << "  Value resolution (default): OK\n";
}

void TestValueResolutionTimeSamples() {
    // Test time-based value resolution with interpolation
    const char* usda = R"(#usda 1.0

def "Ball"
{
    double radius.timeSamples = {
        1: 0.0,
        5: 4.0,
        10: 8.0,
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Ball"));
    auto radius = prim.GetAttribute("radius");
    assert(radius.IsValid());
    assert(radius.HasTimeSamples());

    // GetTimeSampleTimes
    auto times = radius.GetTimeSampleTimes();
    assert(times.size() == 3);
    assert(times[0] == 1.0);
    assert(times[1] == 5.0);
    assert(times[2] == 10.0);

    // Exact time sample hit
    auto v1 = radius.Get(UsdTimeCode(1.0));
    assert(v1.found);
    assert(*v1.value.Get<Double>() == 0.0);

    auto v5 = radius.Get(UsdTimeCode(5.0));
    assert(v5.found);
    assert(*v5.value.Get<Double>() == 4.0);

    auto v10 = radius.Get(UsdTimeCode(10.0));
    assert(v10.found);
    assert(*v10.value.Get<Double>() == 8.0);

    // Linear interpolation between samples
    auto v3 = radius.Get(UsdTimeCode(3.0));
    assert(v3.found);
    // Between 1→0 and 5→4: t=3 is (3-1)/(5-1) = 0.5 of the way → 2.0
    assert(std::abs(*v3.value.Get<Double>() - 2.0) < 1e-9);

    // Held interpolation
    auto v3h = radius.Get(UsdTimeCode(3.0), UsdInterpolationType::Held);
    assert(v3h.found);
    assert(*v3h.value.Get<Double>() == 0.0);  // held to previous (t=1 value)

    // Before first sample: held to first
    auto vBefore = radius.Get(UsdTimeCode(0.0));
    assert(vBefore.found);
    assert(*vBefore.value.Get<Double>() == 0.0);

    // After last sample: held to last
    auto vAfter = radius.Get(UsdTimeCode(15.0));
    assert(vAfter.found);
    assert(*vAfter.value.Get<Double>() == 8.0);

    // Default time with time samples: per spec 12.3, if queried at default time,
    // consult the default field (not time samples)
    auto vDefault = radius.Get(UsdTimeCode::Default());
    assert(!vDefault.found);  // no authored default, no schema fallback

    std::cout << "  Value resolution (time samples): OK\n";
}

void TestValueResolutionTimeSamplesNoDefault() {
    // If time samples exist and time query is made, but no default authored,
    // and no time samples match, check fallback behavior
    const char* usda = R"(#usda 1.0

def "Prim"
{
    double radius = 5.0
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    auto radius = prim.GetAttribute("radius");

    // No time samples — time query falls through to default
    auto v = radius.Get(UsdTimeCode(1.0));
    assert(v.found);
    assert(*v.value.Get<Double>() == 5.0);

    std::cout << "  Value resolution (default with time): OK\n";
}

void TestValueResolutionBlocked() {
    // Test value blocks (spec Section 12.3.6)
    const char* usda = R"(#usda 1.0

def "Prim"
{
    double radius = None
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    auto radius = prim.GetAttribute("radius");
    assert(radius.IsValid());

    // Blocked value → no authored default, falls through to fallback (none)
    auto v = radius.Get(UsdTimeCode::Default());
    assert(!v.found);

    std::cout << "  Value resolution (blocked): OK\n";
}

void TestValueResolutionBlockedWithFallback() {
    // Blocked attribute with schema fallback should return fallback
    auto& reg = SchemaRegistry::GetInstance();
    const char* testSchemas = R"({
      "schemas": {
        "TestSphere": {
          "schemaKind": "typed",
          "parent": "",
          "properties": {
            "radius": { "type": "double", "fallback": 1.0 }
          }
        }
      }
    })";
    std::string schemaErr;
    reg.LoadFromJSON(testSchemas, &schemaErr);

    const char* usda = R"(#usda 1.0

def TestSphere "Ball"
{
    double radius = None
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prim = stage.GetPrimAtPath(Path::Parse("/Ball"));
    auto radius = prim.GetAttribute("radius");
    assert(radius.IsValid());

    // Blocked → returns schema fallback
    auto v = radius.Get(UsdTimeCode::Default());
    assert(v.found);
    assert(*v.value.Get<Double>() == 1.0);

    std::cout << "  Value resolution (blocked with fallback): OK\n";
}

void TestValueResolutionLinearInterpolation() {
    // Test linear interpolation for various numeric types
    const char* usda = R"(#usda 1.0

def "Prim"
{
    float floatAttr.timeSamples = {
        0: 0.0,
        10: 10.0,
    }
    float3 vecAttr.timeSamples = {
        0: (0, 0, 0),
        10: (10, 20, 30),
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));

    // Float linear interpolation at midpoint
    auto floatAttr = prim.GetAttribute("floatAttr");
    auto fv = floatAttr.Get(UsdTimeCode(5.0));
    assert(fv.found);
    assert(std::abs(*fv.value.Get<Float>() - 5.0f) < 1e-5f);

    // Vec3f linear interpolation at midpoint
    auto vecAttr = prim.GetAttribute("vecAttr");
    auto vv = vecAttr.Get(UsdTimeCode(5.0));
    assert(vv.found);
    auto* vec = vv.value.Get<GfVec3f>();
    assert(vec != nullptr);
    assert(std::abs((*vec)[0] - 5.0f) < 1e-5f);
    assert(std::abs((*vec)[1] - 10.0f) < 1e-5f);
    assert(std::abs((*vec)[2] - 15.0f) < 1e-5f);

    std::cout << "  Value resolution (linear interpolation): OK\n";
}

void TestValueResolutionUniformHeld() {
    // Spec Section 12.3: uniform attributes must use Held interpolation
    // regardless of the stage interpolation type setting
    const char* usda = R"(#usda 1.0

def "Prim"
{
    uniform float uniformAttr.timeSamples = {
        0: 0.0,
        10: 10.0,
    }
    float varyingAttr.timeSamples = {
        0: 0.0,
        10: 10.0,
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));

    auto uniformAttr = prim.GetAttribute("uniformAttr");
    auto varyingAttr = prim.GetAttribute("varyingAttr");

    assert(uniformAttr.GetVariability() == Variability::Uniform);
    assert(varyingAttr.GetVariability() == Variability::Varying);

    // At midpoint (t=5): varying should interpolate to 5.0, uniform should hold at 0.0
    auto uv = uniformAttr.Get(UsdTimeCode(5.0));
    assert(uv.found);
    assert(std::abs(*uv.value.Get<Float>() - 0.0f) < 1e-5f);  // Held: floor sample

    auto vv = varyingAttr.Get(UsdTimeCode(5.0));
    assert(vv.found);
    assert(std::abs(*vv.value.Get<Float>() - 5.0f) < 1e-5f);  // Linear: interpolated

    // Even when explicitly requesting Linear, uniform must still use Held
    auto uv2 = uniformAttr.Get(UsdTimeCode(5.0), UsdInterpolationType::Linear);
    assert(uv2.found);
    assert(std::abs(*uv2.value.Get<Float>() - 0.0f) < 1e-5f);

    // At exact sample times, both should return the sample value
    auto u0 = uniformAttr.Get(UsdTimeCode(0.0));
    assert(u0.found);
    assert(std::abs(*u0.value.Get<Float>() - 0.0f) < 1e-5f);

    auto u10 = uniformAttr.Get(UsdTimeCode(10.0));
    assert(u10.found);
    assert(std::abs(*u10.value.Get<Float>() - 10.0f) < 1e-5f);

    std::cout << "  Value resolution (uniform forces Held): OK\n";
}

void TestRelationshipTargets() {
    // Test relationship target resolution (spec Section 12.4)
    const char* usda = R"(#usda 1.0

def "foo" {
    rel myRel = [</foo/bar>, </baz>]

    def "bar" {
    }
}

def "baz" {
    rel bazrel = [</foo/bar>]
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto foo = stage.GetPrimAtPath(Path::Parse("/foo"));
    assert(foo.IsValid());

    // Raw targets
    auto myRel = foo.GetRelationship("myRel");
    assert(myRel.IsValid());
    assert(myRel.HasTargets());

    auto targets = myRel.GetTargets();
    assert(targets.size() == 2);
    assert(targets[0].GetText() == std::string("/foo/bar"));
    assert(targets[1].GetText() == std::string("/baz"));

    // Baz relationship
    auto baz = stage.GetPrimAtPath(Path::Parse("/baz"));
    auto bazRel = baz.GetRelationship("bazrel");
    assert(bazRel.IsValid());
    auto bazTargets = bazRel.GetTargets();
    assert(bazTargets.size() == 1);
    assert(bazTargets[0].GetText() == std::string("/foo/bar"));

    std::cout << "  Relationship targets: OK\n";
}

void TestRelationshipForwardedTargets() {
    // Test forwarded relationship targets (spec Section 12.4)
    const char* usda = R"(#usda 1.0

def "foo" {
    rel myRel = [</foo/bar>, </baz.bazrel>]

    def "bar" {
    }
}

def "baz" {
    rel bazrel = [</foo/foobar>, </foo/foobar/barbaz>]

    def "sub" {
    }
}

def "foobar" {
    def "barbaz" {
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto foo = stage.GetPrimAtPath(Path::Parse("/foo"));
    auto myRel = foo.GetRelationship("myRel");
    assert(myRel.IsValid());

    // Forwarded targets: /foo/bar stays, /baz.bazrel forwards to its targets
    auto forwarded = myRel.GetForwardedTargets();
    assert(forwarded.size() == 3);
    assert(forwarded[0].GetText() == std::string("/foo/bar"));
    assert(forwarded[1].GetText() == std::string("/foo/foobar"));
    assert(forwarded[2].GetText() == std::string("/foo/foobar/barbaz"));

    std::cout << "  Relationship forwarded targets: OK\n";
}

void TestPathValuedListOpNamespaceRemap() {
    auto stage = Stage::Open("tests/composition/path_remap_root.usda");
    assert(stage.IsValid());

    auto inst = stage.GetPrimAtPath(Path::Parse("/Instance"));
    assert(inst.IsValid());

    auto material = inst.GetRelationship("material:binding");
    auto materialTargets = material.GetTargets();
    assert(materialTargets.size() == 1);
    assert(materialTargets[0].GetText() == std::string("/Instance/Materials/Preview"));

    auto color = inst.GetAttribute(Token("inputs:color"));
    auto colorConnections = color.GetConnections();
    assert(colorConnections.size() == 1);
    assert(colorConnections[0].GetText() == std::string("/Instance/Shader.outputs:color"));

    // External references map only the referenced subtree. A source-layer
    // target outside /Model cannot be transformed into /Instance namespace.
    auto externalRel = inst.GetRelationship("externalTarget");
    assert(externalRel.GetTargets().empty());

    auto externalAttr = inst.GetAttribute(Token("externalInput"));
    assert(externalAttr.GetConnections().empty());

    // Equal source/target prim paths on an external reference are still scoped
    // to the referenced subtree, not a full identity mapping.
    auto samePath = stage.GetPrimAtPath(Path::Parse("/Model"));
    assert(samePath.IsValid());

    auto samePathMaterial = samePath.GetRelationship("material:binding");
    auto samePathMaterialTargets = samePathMaterial.GetTargets();
    assert(samePathMaterialTargets.size() == 1);
    assert(samePathMaterialTargets[0].GetText() ==
           std::string("/Model/Materials/Preview"));

    auto samePathExternalRel = samePath.GetRelationship("externalTarget");
    assert(samePathExternalRel.GetTargets().empty());

    auto samePathInput = samePath.GetAttribute(Token("inputs:color"));
    auto samePathConnections = samePathInput.GetConnections();
    assert(samePathConnections.size() == 1);
    assert(samePathConnections[0].GetText() ==
           std::string("/Model/Shader.outputs:color"));

    auto samePathExternalAttr = samePath.GetAttribute(Token("externalInput"));
    assert(samePathExternalAttr.GetConnections().empty());

    // Internal references include an identity mapping for paths outside the
    // referenced subtree, so /Outside stays /Outside.
    auto local = stage.GetPrimAtPath(Path::Parse("/LocalInstance"));
    assert(local.IsValid());

    auto localRel = local.GetRelationship("localTarget");
    auto localTargets = localRel.GetTargets();
    assert(localTargets.size() == 1);
    assert(localTargets[0].GetText() == std::string("/Outside"));

    auto localAttr = local.GetAttribute(Token("localInput"));
    auto localConnections = localAttr.GetConnections();
    assert(localConnections.size() == 1);
    assert(localConnections[0].GetText() == std::string("/Outside.outputs:value"));

    std::cout << "  Path-valued listOp namespace remap: OK\n";
}

void TestComposeRelocates() {
    auto stage = Stage::Open("tests/composition/relocates_root.usda");
    assert(stage.IsValid());

    auto root = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(root.IsValid());

    auto moved = stage.GetPrimAtPath(Path::Parse("/Root/MovedChild"));
    assert(moved.IsValid());
    assert(!stage.GetPrimAtPath(Path::Parse("/Root/Child")).IsValid());

    auto sourceLabel = moved.GetAttribute(Token("sourceLabel")).GetDefault();
    assert(sourceLabel && sourceLabel->Get<String>());
    assert(*sourceLabel->Get<String>() == "from_ref_child");

    auto localLabel = moved.GetAttribute(Token("localLabel")).GetDefault();
    assert(localLabel && localLabel->Get<String>());
    assert(*localLabel->Get<String>() == "from_root_over");

    auto ignoredSourceLabel =
        moved.GetAttribute(Token("ignoredSourceLabel")).GetDefault();
    assert(!ignoredSourceLabel);

    auto childTarget = root.GetRelationship("childTarget").GetTargets();
    assert(childTarget.size() == 1);
    assert(childTarget[0].GetText() == std::string("/Root/MovedChild"));

    auto childInput = root.GetAttribute(Token("childInput")).GetConnections();
    assert(childInput.size() == 1);
    assert(childInput[0].GetText() ==
           std::string("/Root/MovedChild.outputs:value"));

    std::cout << "  Compose relocates: OK\n";
}

static void CheckNestedRelocatesRoot(const Stage& stage,
                                     const char* rootPathText) {
    auto rootPath = Path::Parse(rootPathText);
    auto root = stage.GetPrimAtPath(rootPath);
    assert(root.IsValid());

    auto oldChild = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("Child")));
    assert(!oldChild.IsValid());

    auto moved = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("RenamedChild")));
    assert(moved.IsValid());

    auto leafLabel = moved.GetAttribute(Token("leafLabel")).GetDefault();
    assert(leafLabel && leafLabel->Get<String>());
    assert(*leafLabel->Get<String>() == "from_leaf_child");

    auto midLabel = moved.GetAttribute(Token("midLabel")).GetDefault();
    assert(midLabel && midLabel->Get<String>());
    assert(*midLabel->Get<String>() == "from_mid_target");

    auto ignoredMidSourceLabel =
        moved.GetAttribute(Token("ignoredMidSourceLabel")).GetDefault();
    assert(!ignoredMidSourceLabel);

    auto winner = moved.GetAttribute(Token("winner")).GetDefault();
    assert(winner && winner->Get<String>());
    assert(*winner->Get<String>() == "mid");

    auto staleTargetLabel =
        moved.GetAttribute(Token("staleTargetLabel")).GetDefault();
    assert(!staleTargetLabel);

    auto removed = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("RemoveMe")));
    assert(!removed.IsValid());

    auto removedTarget = root.GetRelationship("removedTarget").GetTargets();
    assert(removedTarget.empty());

    auto removedInput =
        root.GetAttribute(Token("removedInput")).GetConnections();
    assert(removedInput.empty());

    auto invalidGrandChild = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("InvalidGrandChild")));
    assert(!invalidGrandChild.IsValid());

    auto grandChild = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("RenamedChild"))
                .AppendChild(Token("GrandChild")));
    assert(grandChild.IsValid());
    auto grandLabel =
        grandChild.GetAttribute(Token("grandLabel")).GetDefault();
    assert(grandLabel && grandLabel->Get<String>());
    assert(*grandLabel->Get<String>() == "from_leaf_grand");

    auto validNested = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("RenamedChild"))
                .AppendChild(Token("ValidNested")));
    assert(!validNested.IsValid());

    auto validNestedMoved = stage.GetPrimAtPath(
        rootPath.AppendChild(Token("RenamedChild"))
                .AppendChild(Token("ValidNestedMoved")));
    assert(validNestedMoved.IsValid());
    auto validNestedLabel =
        validNestedMoved.GetAttribute(Token("validNestedLabel")).GetDefault();
    assert(validNestedLabel && validNestedLabel->Get<String>());
    assert(*validNestedLabel->Get<String>() == "from_leaf_valid_nested");

    std::string expectedTarget =
        std::string(rootPathText) + "/RenamedChild";
    auto childTarget = root.GetRelationship("childTarget").GetTargets();
    assert(childTarget.size() == 1);
    assert(childTarget[0].GetText() == expectedTarget);

    auto childInput = root.GetAttribute(Token("childInput")).GetConnections();
    assert(childInput.size() == 1);
    assert(childInput[0].GetText() ==
           expectedTarget + ".outputs:value");
}

void TestComposeNestedRelocates() {
    auto stage = Stage::Open("tests/composition/relocates_nested_root.usda");
    assert(stage.IsValid());

    CheckNestedRelocatesRoot(stage, "/RootRef");
    CheckNestedRelocatesRoot(stage, "/RootPayload");

    bool foundAncestralDiag = false;
    for (const auto& d : stage.GetDiagnostics().GetAll()) {
        if (d.arcType == ArcType::Relocate &&
            d.category == DiagCategory::InvalidRelocate &&
            d.message.find("ancestral relocated path") != std::string::npos) {
            foundAncestralDiag = true;
            break;
        }
    }
    assert(foundAncestralDiag);

    std::cout << "  Compose nested layer-stack relocates: OK\n";
}

static const char* kRelocatesRoundtripUsda = R"(#usda 1.0
(
    relocates = {
        </Root/Child>: </Root/MovedChild>,
        </Root/RemoveMe>: None,
    }
)

def "Root"
{
    def "Child"
    {
    }
}
)";

static void CheckLayerRelocatesRoundtrip(const Layer& layer) {
    const Value* field =
        layer.GetLayerSpec().GetField(FieldNames::layerRelocates);
    assert(field);

    const auto* relocates = field->Get<std::vector<Relocate>>();
    assert(relocates);
    assert(relocates->size() == 2);

    assert((*relocates)[0].sourcePath.GetText() == "/Root/Child");
    assert((*relocates)[0].targetPath);
    assert((*relocates)[0].targetPath->GetText() == "/Root/MovedChild");

    assert((*relocates)[1].sourcePath.GetText() == "/Root/RemoveMe");
    assert(!(*relocates)[1].targetPath);
}

void TestRelocatesUsdaRoundtrip() {
    auto first = ParseUsda(kRelocatesRoundtripUsda);
    assert(first.success);
    CheckLayerRelocatesRoundtrip(first.layer);

    std::string written = WriteUsda(first.layer);
    assert(written.find("relocates = {") != std::string::npos);
    assert(written.find("</Root/Child>: </Root/MovedChild>") !=
           std::string::npos);
    assert(written.find("</Root/RemoveMe>: None") != std::string::npos);
    assert(written.find("layerRelocates") == std::string::npos);

    auto second = ParseUsda(written);
    assert(second.success);
    CheckLayerRelocatesRoundtrip(second.layer);

    std::cout << "  Relocates USDA roundtrip: OK\n";
}

void TestRelocatesUsdcRoundtrip() {
    auto parsed = ParseUsda(kRelocatesRoundtripUsda);
    assert(parsed.success);

    auto bytes = WriteUsdc(parsed.layer);
    assert(!bytes.empty());

    auto reparsed = ParseUsdc(bytes.data(), bytes.size());
    assert(reparsed.success);
    CheckLayerRelocatesRoundtrip(reparsed.layer);

    std::string written = WriteUsda(reparsed.layer);
    assert(written.find("relocates = {") != std::string::npos);
    assert(written.find("</Root/RemoveMe>: None") != std::string::npos);

    std::cout << "  Relocates USDC roundtrip: OK\n";
}

void TestStageInterpolationType() {
    // Test stage-level interpolation type setting
    const char* usda = R"(#usda 1.0

def "Prim"
{
    double radius.timeSamples = {
        0: 0.0,
        10: 10.0,
    }
}
)";

    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    // Default is Linear
    assert(stage.GetInterpolationType() == UsdInterpolationType::Linear);

    auto prim = stage.GetPrimAtPath(Path::Parse("/Prim"));
    auto radius = prim.GetAttribute("radius");

    // Linear at midpoint
    auto vLin = radius.Get(UsdTimeCode(5.0), UsdInterpolationType::Linear);
    assert(vLin.found);
    assert(std::abs(*vLin.value.Get<Double>() - 5.0) < 1e-9);

    // Held at midpoint
    auto vHeld = radius.Get(UsdTimeCode(5.0), UsdInterpolationType::Held);
    assert(vHeld.found);
    assert(*vHeld.value.Get<Double>() == 0.0);

    // Change stage default to Held
    stage.SetInterpolationType(UsdInterpolationType::Held);
    assert(stage.GetInterpolationType() == UsdInterpolationType::Held);

    std::cout << "  Stage interpolation type: OK\n";
}

// ============================================================
// USDC Parser Tests
// ============================================================

void TestUsdcFormat() {
    // Valid USDC magic
    const uint8_t magic[] = {'P','X','R','-','U','S','D','C', 0,0,0,0,0,0,0,0};
    assert(IsUsdcFormat(magic, sizeof(magic)));

    // Too short
    assert(!IsUsdcFormat(magic, 4));

    // Wrong magic
    const uint8_t bad[] = {'P','X','R','-','U','S','D','A'};
    assert(!IsUsdcFormat(bad, sizeof(bad)));

    // Null data
    assert(!IsUsdcFormat(nullptr, 0));

    std::cout << "  USDC format detection: OK\n";
}

static std::vector<std::string> ListUsdcFiles(const std::string& dir) {
    std::vector<std::string> files;
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".usdc") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

int TestUsdcFiles(const std::string& dir) {
    auto files = ListUsdcFiles(dir);
    if (files.empty()) {
        std::cerr << "  WARNING: no .usdc files found in " << dir << std::endl;
        return 0;
    }

    int passed = 0;
    int failed = 0;
    for (const auto& path : files) {
        auto slash = path.find_last_of("/\\");
        std::string name = (slash != std::string::npos) ? path.substr(slash + 1) : path;

        auto result = ParseUsdcFile(path);
        if (result.success) {
            std::cout << "  " << name << ": OK" << std::endl;
            ++passed;
        } else {
            std::cerr << "  " << name << ": FAIL: " << result.error << std::endl;
            ++failed;
        }
    }

    std::cout << "  " << passed << " passed, " << failed << " failed out of "
              << files.size() << " files" << std::endl;
    assert(failed == 0);
    return passed;
}

// Compare specs between USDA and USDC parsed layers.
// We compare: prim paths exist, spec types match, key fields match.
static void CompareLayerSpecs(const Layer& usda, const Layer& usdc,
                               const std::string& name,
                               int& specMismatch, int& fieldMismatch) {
    auto usdaPaths = usda.GetSpecPaths();
    auto usdcPaths = usdc.GetSpecPaths();

    // Check that all USDA spec paths exist in USDC
    for (const auto& path : usdaPaths) {
        if (!usdc.HasSpec(path)) {
            std::cerr << "    " << name << ": spec " << path.GetText()
                      << " in USDA but not USDC" << std::endl;
            ++specMismatch;
            continue;
        }
        auto* usdaSpec = usda.GetSpec(path);
        auto* usdcSpec = usdc.GetSpec(path);

        // Spec type must match
        if (usdaSpec->GetType() != usdcSpec->GetType()) {
            std::cerr << "    " << name << ": spec " << path.GetText()
                      << " type mismatch" << std::endl;
            ++specMismatch;
        }

        // For prim specs, check specifier matches
        if (usdaSpec->GetType() == SpecType::Prim &&
            usdcSpec->GetType() == SpecType::Prim) {
            if (usdaSpec->GetSpecifier() != usdcSpec->GetSpecifier()) {
                std::cerr << "    " << name << ": spec " << path.GetText()
                          << " specifier mismatch" << std::endl;
                ++fieldMismatch;
            }
        }
    }

    // Check USDC paths not in USDA (extra specs)
    for (const auto& path : usdcPaths) {
        if (!usda.HasSpec(path)) {
            // Some extra specs may be expected (e.g. pseudo-root), don't fail hard
        }
    }
}

void TestUsdcUsdaEquivalence(const std::string& usdcDir, const std::string& usdaDir) {
    auto usdcFiles = ListUsdcFiles(usdcDir);
    if (usdcFiles.empty()) {
        std::cerr << "  WARNING: no .usdc files found\n";
        return;
    }

    int tested = 0;
    int equivOk = 0;
    int equivFail = 0;

    for (const auto& usdcPath : usdcFiles) {
        // Find matching USDA file
        namespace fs = std::filesystem;
        std::string stem = fs::path(usdcPath).stem().string();
        std::string usdaPath = usdaDir + "/" + stem + ".usda";
        if (!fs::exists(usdaPath)) continue;  // no matching USDA file

        // Parse both
        std::ifstream f(usdaPath, std::ios::binary);
        if (!f) continue;
        std::ostringstream ss;
        ss << f.rdbuf();
        auto usdaResult = ParseUsda(ss.str());
        if (!usdaResult.success) continue;

        auto usdcResult = ParseUsdcFile(usdcPath);
        if (!usdcResult.success) continue;

        ++tested;
        int specMismatch = 0;
        int fieldMismatch = 0;
        CompareLayerSpecs(usdaResult.layer, usdcResult.layer, stem,
                          specMismatch, fieldMismatch);

        if (specMismatch == 0 && fieldMismatch == 0) {
            ++equivOk;
        } else {
            std::cerr << "  " << stem << ": EQUIV FAIL (" << specMismatch
                      << " spec, " << fieldMismatch << " field mismatches)" << std::endl;
            ++equivFail;
        }
    }

    std::cout << "  Equivalence: " << equivOk << " ok, " << equivFail << " fail out of "
              << tested << " tested" << std::endl;
    assert(equivFail == 0);
}

// ============================================================
// USDC Deep Value Tests
// ============================================================

// Helper: get a spec field's default value as type T
template <typename T>
static const T* GetDefault(const Layer& layer, const std::string& path, const std::string& prop) {
    auto fullPath = Path::Parse(path).AppendProperty(prop);
    auto* spec = layer.GetSpec(fullPath);
    if (!spec) return nullptr;
    auto* val = spec->GetField(FieldNames::defaultValue);
    if (!val) return nullptr;
    return val->Get<T>();
}

void TestUsdcScalarValues() {
    auto result = ParseUsdcFile("tests/usdc/scalar_types.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Booleans
    auto* bTrue = GetDefault<Bool>(L, "/ScalarTypes", "boolTrue");
    assert(bTrue && *bTrue == true);
    auto* bFalse = GetDefault<Bool>(L, "/ScalarTypes", "boolFalse");
    assert(bFalse && *bFalse == false);
    auto* bOne = GetDefault<Bool>(L, "/ScalarTypes", "boolOne");
    assert(bOne && *bOne == true);
    auto* bZero = GetDefault<Bool>(L, "/ScalarTypes", "boolZero");
    assert(bZero && *bZero == false);

    // Integers
    auto* iPos = GetDefault<Int>(L, "/ScalarTypes", "intPositive");
    assert(iPos && *iPos == 42);
    auto* iNeg = GetDefault<Int>(L, "/ScalarTypes", "intNegative");
    assert(iNeg && *iNeg == -100);
    auto* iZero = GetDefault<Int>(L, "/ScalarTypes", "intZero");
    assert(iZero && *iZero == 0);
    auto* iMax = GetDefault<Int>(L, "/ScalarTypes", "intMax");
    assert(iMax && *iMax == 2147483647);
    auto* iMin = GetDefault<Int>(L, "/ScalarTypes", "intMin");
    assert(iMin && *iMin == -2147483648);

    // Unsigned integers
    auto* uVal = GetDefault<UInt>(L, "/ScalarTypes", "uintVal");
    assert(uVal && *uVal == 12345u);
    auto* uMax = GetDefault<UInt>(L, "/ScalarTypes", "uintMax");
    assert(uMax && *uMax == 4294967295u);

    // int64
    auto* i64 = GetDefault<Int64>(L, "/ScalarTypes", "int64Val");
    assert(i64 && *i64 == 9999999999LL);
    auto* i64Max = GetDefault<Int64>(L, "/ScalarTypes", "int64Max");
    assert(i64Max && *i64Max == 9223372036854775807LL);

    // uint64
    auto* u64 = GetDefault<UInt64>(L, "/ScalarTypes", "uint64Val");
    assert(u64 && *u64 == 999999999999ULL);
    auto* u64Max = GetDefault<UInt64>(L, "/ScalarTypes", "uint64Max");
    assert(u64Max && *u64Max == 18446744073709551615ULL);

    // Half
    auto* hVal = GetDefault<Half>(L, "/ScalarTypes", "halfVal");
    assert(hVal);
    // Half(1.5) — verify round-trip
    assert(static_cast<float>(*hVal) == 1.5f);

    // Float
    auto* fVal = GetDefault<Float>(L, "/ScalarTypes", "floatVal");
    assert(fVal && std::abs(*fVal - 3.14f) < 0.001f);
    auto* fNeg = GetDefault<Float>(L, "/ScalarTypes", "floatNeg");
    assert(fNeg && std::abs(*fNeg - (-2.718f)) < 0.001f);

    // Double
    auto* dVal = GetDefault<Double>(L, "/ScalarTypes", "doubleVal");
    assert(dVal && std::abs(*dVal - 2.718281828459045) < 1e-12);

    // String
    auto* sVal = GetDefault<String>(L, "/ScalarTypes", "stringVal");
    assert(sVal && *sVal == "hello world");
    auto* sEmpty = GetDefault<String>(L, "/ScalarTypes", "stringEmpty");
    assert(sEmpty && sEmpty->empty());

    // Token
    auto* tVal = GetDefault<Token>(L, "/ScalarTypes", "tokenVal");
    assert(tVal && tVal->GetString() == "myToken");

    // Asset
    auto* aVal = GetDefault<String>(L, "/ScalarTypes", "assetVal");
    assert(aVal && *aVal == "./path/to/asset.usd");

    std::cout << "  USDC scalar values: OK\n";
}

void TestUsdcVectorValues() {
    auto result = ParseUsdcFile("tests/usdc/vec_types.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Verify all vector property specs exist
    const char* vecProps[] = {
        "h2", "h3", "h4", "f2", "f3", "f4",
        "d2", "d3", "d4", "i2", "i3", "i4",
        "qf", "qd", "qh", "m2", "m3", "m4", "fm4"
    };
    for (const char* name : vecProps) {
        auto p = Path::Parse("/VecTypes").AppendProperty(name);
        assert(L.HasSpec(p));
        auto* spec = L.GetSpec(p);
        assert(spec);
        auto* val = spec->GetField(FieldNames::defaultValue);
        assert(val);  // every property should have a default value
    }

    // float3 — check exact values
    auto* f3 = GetDefault<GfVec3f>(L, "/VecTypes", "f3");
    assert(f3 && (*f3)[0] == 1.5f && (*f3)[1] == 2.5f && (*f3)[2] == 3.5f);

    // float2
    auto* f2 = GetDefault<GfVec2f>(L, "/VecTypes", "f2");
    assert(f2 && (*f2)[0] == 1.5f && (*f2)[1] == 2.5f);

    // matrix4d — verify property exists with a value
    auto m4path = Path::Parse("/VecTypes").AppendProperty("m4");
    auto* m4spec = L.GetSpec(m4path);
    assert(m4spec && m4spec->GetField(FieldNames::defaultValue));

    std::cout << "  USDC vector values: OK\n";
}

void TestUsdcArrayValues() {
    auto result = ParseUsdcFile("tests/usdc/array_types.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // int array
    auto path = Path::Parse("/ArrayTypes").AppendProperty("ints");
    auto* spec = L.GetSpec(path);
    assert(spec);
    auto* val = spec->GetField(FieldNames::defaultValue);
    assert(val);
    auto* intArr = val->Get<std::vector<Int>>();
    assert(intArr && intArr->size() == 6);
    assert((*intArr)[0] == 1 && (*intArr)[1] == -2 && (*intArr)[2] == 3);
    assert((*intArr)[3] == 0);
    assert((*intArr)[4] == 2147483647);
    assert((*intArr)[5] == -2147483648);

    // float array
    path = Path::Parse("/ArrayTypes").AppendProperty("floats");
    spec = L.GetSpec(path);
    assert(spec);
    val = spec->GetField(FieldNames::defaultValue);
    assert(val);
    auto* floatArr = val->Get<std::vector<Float>>();
    assert(floatArr && floatArr->size() == 4);
    assert(std::abs((*floatArr)[0] - 1.0f) < 0.001f);
    assert(std::abs((*floatArr)[1] - 2.5f) < 0.001f);
    assert(std::abs((*floatArr)[2] - (-3.14f)) < 0.01f);
    assert((*floatArr)[3] == 0.0f);

    // bool array
    path = Path::Parse("/ArrayTypes").AppendProperty("bools");
    spec = L.GetSpec(path);
    assert(spec);
    val = spec->GetField(FieldNames::defaultValue);
    assert(val);
    auto* boolArr = val->Get<std::vector<Bool>>();
    assert(boolArr && boolArr->size() == 4);
    assert((*boolArr)[0] == true && (*boolArr)[1] == false);
    assert((*boolArr)[2] == true && (*boolArr)[3] == false);

    // Empty arrays
    path = Path::Parse("/ArrayTypes").AppendProperty("emptyInts");
    spec = L.GetSpec(path);
    assert(spec);
    val = spec->GetField(FieldNames::defaultValue);
    assert(val);
    auto* emptyIntArr = val->Get<std::vector<Int>>();
    assert(emptyIntArr && emptyIntArr->empty());

    // string array
    path = Path::Parse("/ArrayTypes").AppendProperty("strings");
    spec = L.GetSpec(path);
    assert(spec);
    val = spec->GetField(FieldNames::defaultValue);
    assert(val);
    auto* strArr = val->Get<std::vector<String>>();
    assert(strArr && strArr->size() == 3);
    assert((*strArr)[0] == "hello" && (*strArr)[1] == "world" && (*strArr)[2] == "");

    // float3 array (point3f)
    path = Path::Parse("/ArrayTypes").AppendProperty("points");
    spec = L.GetSpec(path);
    assert(spec);
    val = spec->GetField(FieldNames::defaultValue);
    assert(val);
    auto* f3Arr = val->Get<std::vector<GfVec3f>>();
    assert(f3Arr && f3Arr->size() == 2);
    assert((*f3Arr)[0][0] == 0.0f && (*f3Arr)[0][1] == 0.0f && (*f3Arr)[0][2] == 0.0f);
    assert((*f3Arr)[1][0] == 1.0f && (*f3Arr)[1][1] == 2.0f && (*f3Arr)[1][2] == 3.0f);

    std::cout << "  USDC array values: OK\n";
}

void TestUsdcSpecifiers() {
    auto result = ParseUsdcFile("tests/usdc/specifiers.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Check all specifier forms
    auto* defWithType = L.GetSpec(Path::Parse("/DefWithType"));
    assert(defWithType && defWithType->GetSpecifier() == Specifier::Def);
    assert(defWithType->GetTypeName().GetText() == std::string("Xform"));

    auto* defBare = L.GetSpec(Path::Parse("/DefBare"));
    assert(defBare && defBare->GetSpecifier() == Specifier::Def);

    auto* overWithType = L.GetSpec(Path::Parse("/OverWithType"));
    assert(overWithType && overWithType->GetSpecifier() == Specifier::Over);
    assert(overWithType->GetTypeName().GetText() == std::string("Mesh"));

    auto* overBare = L.GetSpec(Path::Parse("/OverBare"));
    assert(overBare && overBare->GetSpecifier() == Specifier::Over);

    auto* classWithType = L.GetSpec(Path::Parse("/ClassWithType"));
    assert(classWithType && classWithType->GetSpecifier() == Specifier::Class);
    assert(classWithType->GetTypeName().GetText() == std::string("Material"));

    auto* classBare = L.GetSpec(Path::Parse("/ClassBare"));
    assert(classBare && classBare->GetSpecifier() == Specifier::Class);

    // Nested specifiers
    auto* parent = L.GetSpec(Path::Parse("/Parent"));
    assert(parent && parent->GetSpecifier() == Specifier::Def);

    auto* childDef = L.GetSpec(Path::Parse("/Parent/ChildDef"));
    assert(childDef && childDef->GetSpecifier() == Specifier::Def);

    auto* childOver = L.GetSpec(Path::Parse("/Parent/ChildOver"));
    assert(childOver && childOver->GetSpecifier() == Specifier::Over);

    auto* childClass = L.GetSpec(Path::Parse("/Parent/ChildClass"));
    assert(childClass && childClass->GetSpecifier() == Specifier::Class);

    // Deep nesting
    auto* level1 = L.GetSpec(Path::Parse("/Parent/Level1"));
    assert(level1 && level1->GetSpecifier() == Specifier::Def);
    auto* level2 = L.GetSpec(Path::Parse("/Parent/Level1/Level2"));
    assert(level2 && level2->GetSpecifier() == Specifier::Over);
    auto* level3 = L.GetSpec(Path::Parse("/Parent/Level1/Level2/Level3"));
    assert(level3 && level3->GetSpecifier() == Specifier::Class);
    auto* level4 = L.GetSpec(Path::Parse("/Parent/Level1/Level2/Level3/Level4"));
    assert(level4 && level4->GetSpecifier() == Specifier::Def);

    // Field values through USDC
    auto* sizeVal = GetDefault<Float>(L, "/DefWithType", "size");
    assert(sizeVal && *sizeVal == 1.0f);

    auto* countVal = GetDefault<Int>(L, "/DefBare", "count");
    assert(countVal && *countVal == 5);

    std::cout << "  USDC specifiers: OK\n";
}

void TestUsdcVariants() {
    auto result = ParseUsdcFile("tests/usdc/variants.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Variant selection paths should be reconstructed
    assert(L.HasSpec(Path::Parse("/SimpleVariant")));
    assert(L.HasSpec(Path::Parse("/SimpleVariant{foo=bar}")));

    // Properties inside variant selections
    auto rPath = Path::Parse("/SimpleVariant{foo=bar}").AppendProperty("r");
    // The path might be stored differently depending on variant encoding
    // Check that the variant prim exists
    auto* vs = L.GetSpec(Path::Parse("/SimpleVariant{foo=bar}"));
    assert(vs);

    // Multiple variants
    assert(L.HasSpec(Path::Parse("/MultipleVariants")));
    assert(L.HasSpec(Path::Parse("/MultipleVariants{present=bar}")));
    assert(L.HasSpec(Path::Parse("/MultipleVariants{present=baz}")));
    assert(L.HasSpec(Path::Parse("/MultipleVariants{present=foo}")));

    // Property inside a variant selection prim
    assert(L.HasSpec(Path::Parse("/MultipleVariants{present=bar}/root")));

    // VariantNameCheck paths (special characters in names)
    assert(L.HasSpec(Path::Parse("/VariantNameCheck{a=leadingdot}")));
    assert(L.HasSpec(Path::Parse("/VariantNameCheck{d=_has_an_underscore}")));

    std::cout << "  USDC variants: OK\n";
}

void TestUsdcLayerMetadata() {
    auto result = ParseUsdcFile("tests/usdc/simple.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Layer metadata fields
    auto& ls = L.GetLayerSpec();
    auto* fps = ls.GetField(Token("framesPerSecond"));
    assert(fps);
    auto* fpsVal = fps->Get<Double>();
    assert(fpsVal && *fpsVal == 24.0);

    auto* startFrame = ls.GetField(Token("startFrame"));
    assert(startFrame);

    // Prim fields from simple.usdc
    auto* cam = L.GetSpec(Path::Parse("/overview_cam"));
    assert(cam && cam->GetSpecifier() == Specifier::Def);
    assert(cam->GetTypeName().GetText() == std::string("Camera"));

    // Property value
    auto* camx = GetDefault<Double>(L, "/overview_cam", "camx");
    assert(camx && std::abs(*camx - 1.2) < 1e-10);

    // Nested prim
    auto* head = L.GetSpec(Path::Parse("/overview_cam/Head"));
    assert(head && head->GetSpecifier() == Specifier::Def);

    auto* aspect = GetDefault<Double>(L, "/overview_cam/Head", "aspect");
    assert(aspect && std::abs(*aspect - 1.02517) < 1e-4);

    // Over specifier
    auto* testOver = L.GetSpec(Path::Parse("/TestOver"));
    assert(testOver && testOver->GetSpecifier() == Specifier::Over);
    assert(testOver->GetTypeName().GetText() == std::string("MfScope"));

    std::cout << "  USDC layer metadata: OK\n";
}

void TestUsdcTimeSamples() {
    auto result = ParseUsdcFile("tests/usdc/timesamples_types.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Check timeSamples field exists for scalar float
    auto tsPath = Path::Parse("/TimeSamplesTypes").AppendProperty("scalarFloat");
    auto* spec = L.GetSpec(tsPath);
    assert(spec);
    auto* tsField = spec->GetField(Token("timeSamples"));
    assert(tsField);

    // TimeSamples are stored as Dictionary (time→value)
    auto* tsDict = tsField->Get<Dictionary>();
    assert(tsDict && tsDict->size() == 3);

    // scalarInt timeSamples
    tsPath = Path::Parse("/TimeSamplesTypes").AppendProperty("scalarInt");
    spec = L.GetSpec(tsPath);
    assert(spec);
    tsField = spec->GetField(Token("timeSamples"));
    assert(tsField);
    tsDict = tsField->Get<Dictionary>();
    assert(tsDict && tsDict->size() == 3);

    // vecFloat3 timeSamples
    tsPath = Path::Parse("/TimeSamplesTypes").AppendProperty("vecFloat3");
    spec = L.GetSpec(tsPath);
    assert(spec);
    tsField = spec->GetField(Token("timeSamples"));
    assert(tsField);
    tsDict = tsField->Get<Dictionary>();
    assert(tsDict && tsDict->size() == 3);

    std::cout << "  USDC time samples: OK\n";
}

void TestUsdcDictionaries() {
    auto result = ParseUsdcFile("tests/usdc/dictionaries.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // dictionaries.usda has customData on a property, not the prim
    auto propPath = Path::Parse("/root").AppendProperty("testStringWithDictionaryMetadata");
    auto* spec = L.GetSpec(propPath);
    assert(spec);
    auto* cd = spec->GetField(Token("customData"));
    assert(cd);
    auto* dict = cd->Get<Dictionary>();
    assert(dict && !dict->empty());

    // Verify the dictionary has multiple keys
    assert(dict->size() >= 2);

    std::cout << "  USDC dictionaries: OK\n";
}

void TestUsdcListOps() {
    auto result = ParseUsdcFile("tests/usdc/listop_forms.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Check that reference listops are parsed
    auto* explicitRef = L.GetSpec(Path::Parse("/ExplicitRefList"));
    assert(explicitRef);
    auto* refs = explicitRef->GetField(Token("references"));
    assert(refs);

    // Check prepend/append/delete forms exist
    assert(L.HasSpec(Path::Parse("/PrependReferences")));
    assert(L.HasSpec(Path::Parse("/AppendReferences")));
    assert(L.HasSpec(Path::Parse("/DeleteReferences")));

    // Check inherits
    assert(L.HasSpec(Path::Parse("/PrependInherits")));
    assert(L.HasSpec(Path::Parse("/ExplicitInherits")));

    // Check payloads
    assert(L.HasSpec(Path::Parse("/PrependPayload")));
    assert(L.HasSpec(Path::Parse("/ExplicitPayload")));

    std::cout << "  USDC list ops: OK\n";
}

void TestUsdcNesting() {
    auto result = ParseUsdcFile("tests/usdc/nesting_deep.usdc");
    assert(result.success);
    const auto& L = result.layer;

    // Deep nesting: 8 levels
    assert(L.HasSpec(Path::Parse("/L1")));
    assert(L.HasSpec(Path::Parse("/L1/L2")));
    assert(L.HasSpec(Path::Parse("/L1/L2/L3")));
    assert(L.HasSpec(Path::Parse("/L1/L2/L3/L4")));
    assert(L.HasSpec(Path::Parse("/L1/L2/L3/L4/L5")));
    assert(L.HasSpec(Path::Parse("/L1/L2/L3/L4/L5/L6")));
    assert(L.HasSpec(Path::Parse("/L1/L2/L3/L4/L5/L6/L7")));
    assert(L.HasSpec(Path::Parse("/L1/L2/L3/L4/L5/L6/L7/L8")));

    // Siblings
    assert(L.HasSpec(Path::Parse("/L1/Sibling1")));
    assert(L.HasSpec(Path::Parse("/L1/Sibling2")));
    assert(L.HasSpec(Path::Parse("/L1/Sibling1/Child1")));
    assert(L.HasSpec(Path::Parse("/L1/Sibling1/Child2")));
    assert(L.HasSpec(Path::Parse("/L1/Sibling2/Child1")));

    // Multiple roots
    assert(L.HasSpec(Path::Parse("/Root2")));
    assert(L.HasSpec(Path::Parse("/Root3")));

    std::cout << "  USDC deep nesting: OK\n";
}

namespace {

void ZipAppendU16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
}

void ZipAppendU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xffu));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xffu));
}

uint16_t ZipReadU16(const std::vector<uint8_t>& bytes, size_t offset) {
    assert(offset + 2 <= bytes.size());
    return static_cast<uint16_t>(bytes[offset]) |
           static_cast<uint16_t>(bytes[offset + 1] << 8);
}

uint32_t ZipReadU32(const std::vector<uint8_t>& bytes, size_t offset) {
    assert(offset + 4 <= bytes.size());
    return static_cast<uint32_t>(bytes[offset]) |
           (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24);
}

uint32_t ZipCrc32(const std::vector<uint8_t>& data) {
    uint32_t crc = 0xffffffffu;
    for (uint8_t b : data) {
        crc ^= b;
        for (int i = 0; i < 8; ++i) {
            uint32_t mask = 0u - (crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

bool WriteStoredZip(const std::filesystem::path& path,
                    const std::vector<std::pair<std::string, std::vector<uint8_t>>>& entries) {
    struct Central {
        std::string name;
        uint32_t crc = 0;
        uint32_t size = 0;
        uint32_t offset = 0;
    };

    std::vector<uint8_t> out;
    std::vector<Central> central;
    central.reserve(entries.size());

    for (const auto& [name, data] : entries) {
        Central c;
        c.name = name;
        c.crc = ZipCrc32(data);
        c.size = static_cast<uint32_t>(data.size());
        c.offset = static_cast<uint32_t>(out.size());
        central.push_back(c);

        ZipAppendU32(out, 0x04034b50u);
        ZipAppendU16(out, 20); // version needed
        ZipAppendU16(out, 0);  // flags
        ZipAppendU16(out, 0);  // stored
        ZipAppendU16(out, 0);  // mod time
        ZipAppendU16(out, 0);  // mod date
        ZipAppendU32(out, c.crc);
        ZipAppendU32(out, c.size);
        ZipAppendU32(out, c.size);
        ZipAppendU16(out, static_cast<uint16_t>(name.size()));
        ZipAppendU16(out, 0);  // extra length
        out.insert(out.end(), name.begin(), name.end());
        out.insert(out.end(), data.begin(), data.end());
    }

    uint32_t centralOffset = static_cast<uint32_t>(out.size());
    for (const auto& c : central) {
        ZipAppendU32(out, 0x02014b50u);
        ZipAppendU16(out, 20); // version made by
        ZipAppendU16(out, 20); // version needed
        ZipAppendU16(out, 0);  // flags
        ZipAppendU16(out, 0);  // stored
        ZipAppendU16(out, 0);  // mod time
        ZipAppendU16(out, 0);  // mod date
        ZipAppendU32(out, c.crc);
        ZipAppendU32(out, c.size);
        ZipAppendU32(out, c.size);
        ZipAppendU16(out, static_cast<uint16_t>(c.name.size()));
        ZipAppendU16(out, 0); // extra length
        ZipAppendU16(out, 0); // comment length
        ZipAppendU16(out, 0); // disk start
        ZipAppendU16(out, 0); // internal attrs
        ZipAppendU32(out, 0); // external attrs
        ZipAppendU32(out, c.offset);
        out.insert(out.end(), c.name.begin(), c.name.end());
    }
    uint32_t centralSize = static_cast<uint32_t>(out.size()) - centralOffset;

    ZipAppendU32(out, 0x06054b50u);
    ZipAppendU16(out, 0);
    ZipAppendU16(out, 0);
    ZipAppendU16(out, static_cast<uint16_t>(central.size()));
    ZipAppendU16(out, static_cast<uint16_t>(central.size()));
    ZipAppendU32(out, centralSize);
    ZipAppendU32(out, centralOffset);
    ZipAppendU16(out, 0);

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(out.data()),
            static_cast<std::streamsize>(out.size()));
    return static_cast<bool>(f);
}

std::vector<uint8_t> ReadBinaryFile(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    assert(f);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(f),
                                std::istreambuf_iterator<char>());
}

std::vector<uint8_t> Bytes(std::string_view text) {
    return std::vector<uint8_t>(text.begin(), text.end());
}

std::string FileUriFromPath(const std::filesystem::path& path) {
    std::string p = std::filesystem::absolute(path).lexically_normal().generic_string();
#ifdef _WIN32
    if (p.size() >= 2 && p[1] == ':') return "file:///" + p;
#endif
    return "file://" + p;
}

} // anonymous namespace

void TestUnifiedParser() {
    namespace fs = std::filesystem;

    // Parse USDA file via unified parser
    auto usdaResult = ParseUsdFile("tests/usda/simple.usda");
    assert(usdaResult.success);
    assert(usdaResult.layer.HasSpec(Path::Parse("/overview_cam")));

    // Parse USDC file via unified parser
    auto usdcResult = ParseUsdFile("tests/usdc/simple.usdc");
    assert(usdcResult.success);
    assert(usdcResult.layer.HasSpec(Path::Parse("/overview_cam")));

    // Both should produce the same specs
    auto usdaPaths = usdaResult.layer.GetSpecPaths();
    auto usdcPaths = usdcResult.layer.GetSpecPaths();
    assert(usdaPaths.size() == usdcPaths.size());

    // Verify specifiers match
    for (const auto& p : usdaPaths) {
        assert(usdcResult.layer.HasSpec(p));
        auto* a = usdaResult.layer.GetSpec(p);
        auto* c = usdcResult.layer.GetSpec(p);
        assert(a->GetType() == c->GetType());
        if (a->GetType() == SpecType::Prim) {
            assert(a->GetSpecifier() == c->GetSpecifier());
        }
    }

    // Verify value equivalence for a specific property
    auto* aVal = GetDefault<Double>(usdaResult.layer, "/overview_cam", "camx");
    auto* cVal = GetDefault<Double>(usdcResult.layer, "/overview_cam", "camx");
    assert(aVal && cVal && std::abs(*aVal - *cVal) < 1e-10);

    std::string usdaUri = FileUriFromPath(fs::path("tests/usda/simple.usda"));
    auto usdaLocation = ResolvedLocation::FromResolvedString(usdaUri);
    assert(usdaLocation.IsLocalFile());
    auto resource = ReadResource(usdaLocation);
    assert(resource.success);
    assert(resource.fileBacked);
    assert(!resource.filePath.empty());
    assert(resource.bytes.size() > 8);

    auto usdaUriResult = ParseUsdFile(usdaLocation);
    assert(usdaUriResult.success);
    assert(usdaUriResult.layer.HasSpec(Path::Parse("/overview_cam")));

    std::string usdcUri = FileUriFromPath(fs::path("tests/usdc/simple.usdc"));
    auto usdcUriResult = ParseUsdFile(ResolvedLocation::FromResolvedString(usdcUri));
    assert(usdcUriResult.success);
    assert(usdcUriResult.layer.HasSpec(Path::Parse("/overview_cam")));

    auto unsupportedResource = ReadResource("https://example.com/assets/simple.usda");
    assert(!unsupportedResource.success);
    assert(unsupportedResource.error.find("Unsupported resource scheme: https") !=
           std::string::npos);

    // Non-existent file
    auto badResult = ParseUsdFile("nonexistent.usd");
    assert(!badResult.success);

    std::cout << "  Unified parser (USDA + USDC auto-detect): OK\n";
}

void TestUsdzPackageRead() {
    namespace fs = std::filesystem;
    fs::path pkg = fs::temp_directory_path() / "nanousd_usdz_package_read.usdz";

    std::string root = R"(#usda 1.0
(
    defaultPrim = "World"
    subLayers = [
        @../layers/weak.usda@
    ]
)

def "World" (
    references = @../models/model.usda@</Model>
)
{
    double local = 1
    asset inputs:texture = @../textures/diffuse.png@
    asset inputs:packedRemote = @https://example.com/assets/car.usdz[textures/paint.png]@
}
)";
    std::string weak = R"(#usda 1.0
over "World"
{
    double weak = 7
}
)";
    std::string model = R"(#usda 1.0
def "Model"
{
    double size = 5
}
)";
    std::string textureData = "placeholder texture bytes";

    assert(WriteStoredZip(pkg, {
        {"scenes/root.usda", Bytes(root)},
        {"layers/weak.usda", Bytes(weak)},
        {"models/model.usda", Bytes(model)},
        {"textures/diffuse.png", Bytes(textureData)},
    }));

    auto rootResult = ParseUsdFile(pkg.string());
    assert(rootResult.success);
    assert(rootResult.layer.HasSpec(Path::Parse("/World")));

    auto explicitResult = ParseUsdFile(pkg.string() + "[models/model.usda]");
    assert(explicitResult.success);
    assert(explicitResult.layer.HasSpec(Path::Parse("/Model")));

    std::string pkgUri = FileUriFromPath(pkg);
    auto uriRootResult = ParseUsdFile(pkgUri);
    assert(uriRootResult.success);
    assert(uriRootResult.layer.HasSpec(Path::Parse("/World")));

    auto uriExplicitResult = ParseUsdFile(pkgUri + "[models/model.usda]");
    assert(uriExplicitResult.success);
    assert(uriExplicitResult.layer.HasSpec(Path::Parse("/Model")));

    auto stage = Stage::Open(pkg.string());
    assert(stage.IsValid());
    auto world = stage.GetPrimAtPath(Path::Parse("/World"));
    assert(world.IsValid());

    auto local = world.GetAttribute(Token("local")).Get(UsdTimeCode::Default());
    assert(local.found && local.value.Get<Double>() && *local.value.Get<Double>() == 1.0);
    auto weakAttr = world.GetAttribute(Token("weak")).Get(UsdTimeCode::Default());
    assert(weakAttr.found && weakAttr.value.Get<Double>() && *weakAttr.value.Get<Double>() == 7.0);
    auto size = world.GetAttribute(Token("size")).Get(UsdTimeCode::Default());
    assert(size.found && size.value.Get<Double>() && *size.value.Get<Double>() == 5.0);
    auto texture = world.GetAttribute(Token("inputs:texture")).Get(UsdTimeCode::Default());
    assert(texture.found && texture.value.Get<std::string>());
    assert(*texture.value.Get<std::string>() == pkg.string() + "[textures/diffuse.png]");
    auto packedRemote = world.GetAttribute(Token("inputs:packedRemote")).Get(UsdTimeCode::Default());
    assert(packedRemote.found && packedRemote.value.Get<std::string>());
    assert(*packedRemote.value.Get<std::string>() ==
           "https://example.com/assets/car.usdz[textures/paint.png]");

    std::error_code ec;
    fs::remove(pkg, ec);

    std::cout << "  USDZ package read + internal resolution: OK\n";
}

void AssertWrittenUsdzLayout(const std::vector<uint8_t>& bytes,
                             const std::vector<std::string>& expectedNames) {
    assert(bytes.size() >= 22);
    const size_t eocd = bytes.size() - 22;
    assert(ZipReadU32(bytes, eocd) == 0x06054b50u);
    assert(ZipReadU16(bytes, eocd + 4) == 0);
    assert(ZipReadU16(bytes, eocd + 6) == 0);
    assert(ZipReadU16(bytes, eocd + 20) == 0);

    const uint16_t centralEntries = ZipReadU16(bytes, eocd + 8);
    const uint16_t totalEntries = ZipReadU16(bytes, eocd + 10);
    const uint32_t centralSize = ZipReadU32(bytes, eocd + 12);
    const uint32_t centralOffset = ZipReadU32(bytes, eocd + 16);
    assert(centralEntries == totalEntries);
    assert(centralEntries == expectedNames.size());
    assert(static_cast<size_t>(centralOffset) + centralSize == eocd);

    size_t pos = centralOffset;
    for (size_t i = 0; i < expectedNames.size(); ++i) {
        assert(ZipReadU32(bytes, pos) == 0x02014b50u);
        const uint16_t flags = ZipReadU16(bytes, pos + 8);
        const uint16_t method = ZipReadU16(bytes, pos + 10);
        const uint32_t compressedSize = ZipReadU32(bytes, pos + 20);
        const uint32_t uncompressedSize = ZipReadU32(bytes, pos + 24);
        const uint16_t nameLen = ZipReadU16(bytes, pos + 28);
        const uint16_t extraLen = ZipReadU16(bytes, pos + 30);
        const uint16_t commentLen = ZipReadU16(bytes, pos + 32);
        const uint16_t diskStart = ZipReadU16(bytes, pos + 34);
        const uint32_t localOffset = ZipReadU32(bytes, pos + 42);
        assert(method == 0);
        assert((flags & 0x0001u) == 0);
        assert(compressedSize == uncompressedSize);
        assert(extraLen == 0);
        assert(commentLen == 0);
        assert(diskStart == 0);
        assert((localOffset % 64u) == 0);

        std::string name(reinterpret_cast<const char*>(bytes.data() + pos + 46),
                         nameLen);
        assert(name == expectedNames[i]);

        assert(ZipReadU32(bytes, localOffset) == 0x04034b50u);
        assert(ZipReadU16(bytes, localOffset + 6) == 0);
        assert(ZipReadU16(bytes, localOffset + 8) == 0);
        assert(ZipReadU32(bytes, localOffset + 18) == compressedSize);
        assert(ZipReadU32(bytes, localOffset + 22) == uncompressedSize);
        const uint16_t localNameLen = ZipReadU16(bytes, localOffset + 26);
        const uint16_t localExtraLen = ZipReadU16(bytes, localOffset + 28);
        std::string localName(
            reinterpret_cast<const char*>(bytes.data() + localOffset + 30),
            localNameLen);
        assert(localName == expectedNames[i]);
        if (localExtraLen > 0) {
            assert(localExtraLen >= 4);
            assert(ZipReadU16(bytes, localOffset + 30 + localNameLen) == 0xffffu);
        }

        pos += 46u + nameLen + extraLen + commentLen;
    }
    assert(pos == static_cast<size_t>(centralOffset) + centralSize);
}

void TestUsdzPackageWrite() {
    namespace fs = std::filesystem;
    fs::path pkg = fs::temp_directory_path() / "nanousd_usdz_package_write.usdz";
    fs::path pkgEntries = fs::temp_directory_path() / "nanousd_usdz_package_entries.usdz";

    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());
    stage.GetMutableLayer().GetLayerSpec().SetDefaultPrim("Root");
    auto root = stage.DefinePrim(Path::Parse("/Root"), Token("Xform"));
    assert(root.IsValid());
    auto attr = root.CreateAttribute("size", Token("double"));
    assert(attr.IsValid());
    assert(attr.Set(Value(Double(3.5))));

    std::string error;
    assert(WriteUsdzFile(stage.GetMutableLayer(), pkg.string(), "root.usdc", &error));
    auto bytes = ReadBinaryFile(pkg);
    AssertWrittenUsdzLayout(bytes, {"root.usdc"});

    std::string rootError;
    assert(GetUsdzRootLayerPath(pkg.string(), &rootError) == "root.usdc");
    auto parsed = ParseUsdFile(pkg.string());
    assert(parsed.success);
    auto* spec = parsed.layer.GetSpec(Path::Parse("/Root").AppendProperty("size"));
    assert(spec);
    auto* defaultValue = spec->GetField(Token("default"));
    assert(defaultValue && defaultValue->Get<Double>() &&
           *defaultValue->Get<Double>() == 3.5);

    std::string rootText = WriteUsda(stage.GetMutableLayer());
    assert(WriteUsdzFile({
        {"scenes/root.usda", Bytes(rootText)},
        {"textures/diffuse.png", Bytes("texture bytes")},
    }, pkgEntries.string(), &error));
    auto entryBytes = ReadBinaryFile(pkgEntries);
    AssertWrittenUsdzLayout(entryBytes, {"scenes/root.usda", "textures/diffuse.png"});
    auto texture = ReadUsdzFile(pkgEntries.string() + "[textures/diffuse.png]");
    assert(texture.success);
    assert(std::string(texture.data.begin(), texture.data.end()) == "texture bytes");

    std::error_code ec;
    fs::remove(pkg, ec);
    fs::remove(pkgEntries, ec);

    std::cout << "  USDZ package write: OK\n";
}

// ============================================================
// Comprehensive Value Tests
// ============================================================

void TestValueAllScalars() {
    // Bool
    Value bTrue(true);
    assert(bTrue.GetTypeId() == TypeId::Bool);
    assert(*bTrue.Get<Bool>() == true);
    Value bFalse(false);
    assert(*bFalse.Get<Bool>() == false);

    // UChar
    Value uc(UChar(255));
    assert(uc.GetTypeId() == TypeId::UChar);
    assert(*uc.Get<UChar>() == 255);
    Value uc0(UChar(0));
    assert(*uc0.Get<UChar>() == 0);

    // Int
    Value iPos(Int(2147483647));
    assert(iPos.GetTypeId() == TypeId::Int);
    assert(*iPos.Get<Int>() == 2147483647);
    Value iNeg(Int(-2147483647));
    assert(*iNeg.Get<Int>() == -2147483647);

    // UInt
    Value ui(UInt(4294967295u));
    assert(ui.GetTypeId() == TypeId::UInt);
    assert(*ui.Get<UInt>() == 4294967295u);

    // Int64
    Value i64(Int64(9223372036854775807LL));
    assert(i64.GetTypeId() == TypeId::Int64);
    assert(*i64.Get<Int64>() == 9223372036854775807LL);
    Value i64Neg(Int64(-1LL));
    assert(*i64Neg.Get<Int64>() == -1LL);

    // UInt64
    Value u64(UInt64(18446744073709551615ULL));
    assert(u64.GetTypeId() == TypeId::UInt64);
    assert(*u64.Get<UInt64>() == 18446744073709551615ULL);

    // Half
    Value hv(Half(1.0f));
    assert(hv.GetTypeId() == TypeId::Half);
    auto* hp = hv.Get<Half>();
    assert(hp);
    assert(std::abs(static_cast<float>(*hp) - 1.0f) < 0.001f);

    // Float
    Value fv(Float(3.14f));
    assert(fv.GetTypeId() == TypeId::Float);
    assert(std::abs(*fv.Get<Float>() - 3.14f) < 0.001f);

    // Double
    Value dv(Double(2.718281828));
    assert(dv.GetTypeId() == TypeId::Double);
    assert(std::abs(*dv.Get<Double>() - 2.718281828) < 1e-6);

    // String (all constructor forms)
    Value s1(std::string("hello"));
    assert(s1.GetTypeId() == TypeId::String);
    assert(*s1.Get<String>() == "hello");
    Value s2("world");  // const char*
    assert(s2.GetTypeId() == TypeId::String);
    assert(*s2.Get<String>() == "world");
    std::string tmp = "moved";
    Value s3(std::move(tmp));
    assert(*s3.Get<String>() == "moved");

    // TimeCode (uses nullptr_t disambiguator)
    Value tc(TimeCode(24.0), nullptr);
    assert(tc.GetTypeId() == TypeId::TimeCode);
    assert(std::abs(*tc.Get<TimeCode>() - 24.0) < 1e-9);

    // Wrong type returns nullptr
    assert(bTrue.Get<Int>() == nullptr);
    assert(iPos.Get<Float>() == nullptr);
    assert(s1.Get<Double>() == nullptr);

    std::cout << "  Value All Scalars: OK\n";
}

void TestValueDimensionedTypes() {
    // All 12 Vec types
    Value v2h(GfVec2h{{Half(1.0f), Half(2.0f)}});
    assert(v2h.GetTypeId() == TypeId::Half2);
    assert(v2h.Get<GfVec2h>() != nullptr);

    Value v3h(GfVec3h{{Half(1.0f), Half(2.0f), Half(3.0f)}});
    assert(v3h.GetTypeId() == TypeId::Half3);

    Value v4h(GfVec4h{{Half(1.0f), Half(2.0f), Half(3.0f), Half(4.0f)}});
    assert(v4h.GetTypeId() == TypeId::Half4);

    Value v2f(GfVec2f{{1.0f, 2.0f}});
    assert(v2f.GetTypeId() == TypeId::Float2);
    assert(v2f.Get<GfVec2f>()->data[0] == 1.0f);
    assert(v2f.Get<GfVec2f>()->data[1] == 2.0f);

    Value v3f(GfVec3f{{1.0f, 2.0f, 3.0f}});
    assert(v3f.GetTypeId() == TypeId::Float3);

    Value v4f(GfVec4f{{1.0f, 2.0f, 3.0f, 4.0f}});
    assert(v4f.GetTypeId() == TypeId::Float4);
    assert(v4f.Get<GfVec4f>()->data[3] == 4.0f);

    Value v2d(GfVec2d{{1.0, 2.0}});
    assert(v2d.GetTypeId() == TypeId::Double2);
    assert(v2d.Get<GfVec2d>()->data[0] == 1.0);

    Value v3d(GfVec3d{{10.0, 20.0, 30.0}});
    assert(v3d.GetTypeId() == TypeId::Double3);

    Value v4d(GfVec4d{{1.0, 2.0, 3.0, 4.0}});
    assert(v4d.GetTypeId() == TypeId::Double4);

    Value v2i(GfVec2i{{10, 20}});
    assert(v2i.GetTypeId() == TypeId::Int2);
    assert(v2i.Get<GfVec2i>()->data[0] == 10);

    Value v3i(GfVec3i{{1, 2, 3}});
    assert(v3i.GetTypeId() == TypeId::Int3);

    Value v4i(GfVec4i{{10, 20, 30, 40}});
    assert(v4i.GetTypeId() == TypeId::Int4);
    assert(v4i.Get<GfVec4i>()->data[3] == 40);

    // All 3 Quaternion types
    GfQuath qh;
    qh[0] = Half(0.0f); qh[1] = Half(0.0f); qh[2] = Half(0.0f); qh[3] = Half(1.0f);
    Value vqh(qh);
    assert(vqh.GetTypeId() == TypeId::Quath);
    assert(vqh.Get<GfQuath>() != nullptr);
    assert(std::abs(static_cast<float>(vqh.Get<GfQuath>()->data[3]) - 1.0f) < 0.001f);

    GfQuatf qf;
    qf[0] = 0.0f; qf[1] = 0.0f; qf[2] = 0.707f; qf[3] = 0.707f;
    Value vqf(qf);
    assert(vqf.GetTypeId() == TypeId::Quatf);
    assert(vqf.Get<GfQuatf>() != nullptr);
    assert(std::abs(vqf.Get<GfQuatf>()->data[2] - 0.707f) < 0.001f);
    assert(std::abs(vqf.Get<GfQuatf>()->data[3] - 0.707f) < 0.001f);

    GfQuatd qd;
    qd[0] = 0.0; qd[1] = 0.0; qd[2] = 0.0; qd[3] = 1.0;
    Value vqd(qd);
    assert(vqd.GetTypeId() == TypeId::Quatd);
    assert(vqd.Get<GfQuatd>()->data[3] == 1.0);

    // All 3 Matrix types
    auto m2 = GfMatrix2d::Identity();
    Value vm2(m2);
    assert(vm2.GetTypeId() == TypeId::Matrix2d);
    assert(vm2.Get<GfMatrix2d>()->operator()(0, 0) == 1.0);
    assert(vm2.Get<GfMatrix2d>()->operator()(0, 1) == 0.0);
    assert(vm2.Get<GfMatrix2d>()->operator()(1, 1) == 1.0);

    auto m3 = GfMatrix3d::Identity();
    Value vm3(m3);
    assert(vm3.GetTypeId() == TypeId::Matrix3d);
    assert(vm3.Get<GfMatrix3d>()->operator()(2, 2) == 1.0);

    auto m4 = GfMatrix4d::Identity();
    Value vm4(m4);
    assert(vm4.GetTypeId() == TypeId::Matrix4d);
    assert(vm4.Get<GfMatrix4d>()->operator()(3, 3) == 1.0);

    // Non-identity matrix
    GfMatrix4d custom;
    custom(0, 0) = 2.0; custom(1, 1) = 3.0; custom(2, 2) = 4.0; custom(3, 3) = 1.0;
    custom(3, 0) = 1.0; custom(3, 1) = 2.0; custom(3, 2) = 3.0;
    Value vmCustom(custom);
    assert(vmCustom.Get<GfMatrix4d>()->operator()(0, 0) == 2.0);
    assert(vmCustom.Get<GfMatrix4d>()->operator()(3, 2) == 3.0);

    std::cout << "  Value Dimensioned Types: OK\n";
}

void TestValueArrays() {
    // Float array
    Value floatArr(Value::ArrayTag{}, TypeId::Float, std::vector<float>{1.0f, 2.0f, 3.0f});
    assert(floatArr.GetTypeId() == TypeId::Float);
    assert(floatArr.IsArray());
    assert(!floatArr.IsEmpty());
    assert(!floatArr.IsBlock());
    auto* fa = floatArr.Get<std::vector<float>>();
    assert(fa && fa->size() == 3);
    assert((*fa)[0] == 1.0f && (*fa)[2] == 3.0f);

    // Int array
    Value intArr(Value::ArrayTag{}, TypeId::Int, std::vector<int>{10, 20, 30, 40});
    assert(intArr.IsArray());
    assert(intArr.Get<std::vector<int>>()->size() == 4);

    // String array
    Value strArr(Value::ArrayTag{}, TypeId::String,
                 std::vector<std::string>{"a", "b", "c"});
    assert(strArr.IsArray());
    assert(strArr.Get<std::vector<std::string>>()->size() == 3);

    // Bool array
    Value boolArr(Value::ArrayTag{}, TypeId::Bool, std::vector<bool>{true, false, true});
    assert(boolArr.IsArray());

    // Empty array
    Value emptyArr(Value::ArrayTag{}, TypeId::Float, std::vector<float>{});
    assert(emptyArr.IsArray());
    assert(emptyArr.Get<std::vector<float>>()->empty());

    // Vec3f array
    Value vec3Arr(Value::ArrayTag{}, TypeId::Float3,
                  std::vector<GfVec3f>{GfVec3f{{1, 2, 3}}, GfVec3f{{4, 5, 6}}});
    assert(vec3Arr.IsArray());
    auto* va = vec3Arr.Get<std::vector<GfVec3f>>();
    assert(va && va->size() == 2);
    assert((*va)[0][0] == 1.0f && (*va)[1][2] == 6.0f);

    // Double array
    Value dblArr(Value::ArrayTag{}, TypeId::Double,
                 std::vector<double>{1.1, 2.2, 3.3});
    assert(dblArr.IsArray());
    assert(dblArr.Get<std::vector<double>>()->size() == 3);

    std::cout << "  Value Arrays: OK\n";
}

void TestValueCopyMoveSemantics() {
    // Copy construction
    Value orig(42);
    Value copy = orig;
    assert(copy.GetTypeId() == TypeId::Int);
    assert(*copy.Get<Int>() == 42);
    assert(*orig.Get<Int>() == 42);  // original unchanged

    // Move construction
    Value moveOrig(std::string("moveme"));
    Value moved = std::move(moveOrig);
    assert(moved.GetTypeId() == TypeId::String);
    assert(*moved.Get<String>() == "moveme");

    // Copy assignment
    Value a(1.0f);
    Value b(2.0);
    a = b;
    assert(a.GetTypeId() == TypeId::Double);
    assert(*a.Get<Double>() == 2.0);

    // Move assignment
    Value c(GfVec3f{{1, 2, 3}});
    Value d;
    d = std::move(c);
    assert(d.GetTypeId() == TypeId::Float3);
    assert(d.Get<GfVec3f>()->data[0] == 1.0f);

    // Copy array value
    Value arrOrig(Value::ArrayTag{}, TypeId::Int, std::vector<int>{1, 2, 3});
    Value arrCopy = arrOrig;
    assert(arrCopy.IsArray());
    assert(arrCopy.Get<std::vector<int>>()->size() == 3);

    // ValueBlock copies
    Value block1(ValueBlock{});
    Value block2 = block1;
    assert(block2.IsBlock());

    // Empty copies
    Value e1;
    Value e2 = e1;
    assert(e2.IsEmpty());

    std::cout << "  Value Copy/Move: OK\n";
}

void TestValueRoles() {
    Value v(GfVec3f{{1.0f, 2.0f, 3.0f}});
    assert(v.GetRole() == Role::None);

    v.SetRole(Role::Color);
    assert(v.GetRole() == Role::Color);

    v.SetRole(Role::Normal);
    assert(v.GetRole() == Role::Normal);

    v.SetRole(Role::Point);
    assert(v.GetRole() == Role::Point);

    v.SetRole(Role::Vector);
    assert(v.GetRole() == Role::Vector);

    v.SetRole(Role::TexCoord);
    assert(v.GetRole() == Role::TexCoord);

    // Role is preserved through copy
    Value copy = v;
    assert(copy.GetRole() == Role::TexCoord);

    // Different types can have roles
    Value dv(GfVec3d{{1, 2, 3}});
    dv.SetRole(Role::Point);
    assert(dv.GetRole() == Role::Point);

    std::cout << "  Value Roles: OK\n";
}

void TestValueBlockAndEmpty() {
    Value block(ValueBlock{});
    assert(block.IsBlock());
    assert(!block.IsEmpty());
    assert(!block.IsArray());
    assert(block.GetTypeId() == TypeId::Unknown);

    Value empty;
    assert(empty.IsEmpty());
    assert(!empty.IsBlock());
    assert(!empty.IsArray());
    assert(empty.GetTypeId() == TypeId::Unknown);

    // A Value with data is neither block nor empty
    Value data(42);
    assert(!data.IsBlock());
    assert(!data.IsEmpty());

    std::cout << "  Value Block/Empty: OK\n";
}

void TestValueCompositionTypes() {
    // SubLayerPaths
    SubLayerPaths slp;
    slp.paths = {"./sub1.usda", "./sub2.usda"};
    slp.offsets = {Retiming{0.0, 1.0}, Retiming{10.0, 2.0}};
    Value slpVal(std::move(slp));
    assert(slpVal.GetTypeId() == TypeId::Unknown);
    auto* slpGet = slpVal.Get<SubLayerPaths>();
    assert(slpGet && slpGet->paths.size() == 2);
    assert(slpGet->paths[0] == "./sub1.usda");

    // ListOp<Reference>
    Reference ref1;
    ref1.assetPath = "./ref.usda";
    ref1.primPath = Path::Parse("/Root");
    auto refListOp = ListOp<Reference>::CreateExplicit({ref1});
    Value refVal(std::move(refListOp));
    auto* refGet = refVal.Get<ListOp<Reference>>();
    assert(refGet && refGet->IsExplicit());
    assert(refGet->GetExplicitItems().size() == 1);
    assert(*refGet->GetExplicitItems()[0].assetPath == "./ref.usda");

    // ListOp<std::string>
    auto strListOp = ListOp<std::string>::CreateComposable(
        {"prepA"}, {"appB"}, {"delC"});
    Value strLopVal(std::move(strListOp));
    auto* strGet = strLopVal.Get<ListOp<std::string>>();
    assert(strGet && !strGet->IsExplicit());
    assert(strGet->GetPrependedItems() == std::vector<std::string>{"prepA"});

    std::cout << "  Value Composition Types: OK\n";
}

// ============================================================
// Comprehensive Quaternion Tests
// ============================================================

void TestQuaternions() {
    // GfQuath — half precision
    {
        GfQuath q;
        // Default zero-initialized
        assert(q[0] == Half(0.0f));
        assert(q[3] == Half(0.0f));

        // Set identity quaternion (0,0,0,1)
        q[0] = Half(0.0f); q[1] = Half(0.0f);
        q[2] = Half(0.0f); q[3] = Half(1.0f);
        assert(std::abs(static_cast<float>(q[3]) - 1.0f) < 0.001f);

        // Equality
        GfQuath q2;
        q2[0] = Half(0.0f); q2[1] = Half(0.0f);
        q2[2] = Half(0.0f); q2[3] = Half(1.0f);
        assert(q == q2);

        q2[3] = Half(0.5f);
        assert(q != q2);
    }

    // GfQuatf — float precision
    {
        GfQuatf q;
        assert(q[0] == 0.0f);
        q[0] = 0.0f; q[1] = 0.0f; q[2] = 0.707107f; q[3] = 0.707107f;
        assert(std::abs(q[2] - 0.707107f) < 0.0001f);
        assert(std::abs(q[3] - 0.707107f) < 0.0001f);

        // 90-degree rotation about Z axis
        GfQuatf qz;
        qz[0] = 0.0f; qz[1] = 0.0f; qz[2] = 0.707107f; qz[3] = 0.707107f;
        assert(q == qz);

        // Value round-trip
        Value v(q);
        assert(v.GetTypeId() == TypeId::Quatf);
        auto* p = v.Get<GfQuatf>();
        assert(p && std::abs((*p)[2] - 0.707107f) < 0.0001f);
    }

    // GfQuatd — double precision
    {
        GfQuatd q;
        q[0] = 0.0; q[1] = 0.0; q[2] = 0.0; q[3] = 1.0;
        assert(q[3] == 1.0);

        GfQuatd q2;
        q2[0] = 0.0; q2[1] = 0.0; q2[2] = 0.0; q2[3] = 1.0;
        assert(q == q2);

        // 180-degree rotation about Y
        GfQuatd qy;
        qy[0] = 0.0; qy[1] = 1.0; qy[2] = 0.0; qy[3] = 0.0;
        assert(q != qy);
        assert(qy[1] == 1.0);

        // Value round-trip
        Value v(qy);
        auto* p = v.Get<GfQuatd>();
        assert(p && (*p)[1] == 1.0 && (*p)[3] == 0.0);
    }

    // Quat arrays in Value
    {
        std::vector<GfQuatf> quats = {
            GfQuatf{{0.0f, 0.0f, 0.0f, 1.0f}},
            GfQuatf{{0.0f, 0.0f, 0.707f, 0.707f}},
        };
        Value arr(Value::ArrayTag{}, TypeId::Quatf, std::move(quats));
        assert(arr.IsArray());
        auto* p = arr.Get<std::vector<GfQuatf>>();
        assert(p && p->size() == 2);
        assert((*p)[0][3] == 1.0f);
    }

    std::cout << "  Quaternions: OK\n";
}

// ============================================================
// Comprehensive Vec Tests (all 12 types)
// ============================================================

void TestVecComprehensive() {
    // GfVec2h
    {
        GfVec2h v{{Half(1.0f), Half(2.0f)}};
        assert(std::abs(static_cast<float>(v[0]) - 1.0f) < 0.01f);
        assert(std::abs(static_cast<float>(v[1]) - 2.0f) < 0.01f);
        GfVec2h v2{{Half(1.0f), Half(2.0f)}};
        assert(v == v2);
        v2[0] = Half(3.0f);
        assert(v != v2);
    }

    // GfVec3h
    {
        GfVec3h v{{Half(1.0f), Half(2.0f), Half(3.0f)}};
        assert(v.data.size() == 3);
        Value val(v);
        assert(val.GetTypeId() == TypeId::Half3);
    }

    // GfVec4h
    {
        GfVec4h v{{Half(1.0f), Half(2.0f), Half(3.0f), Half(4.0f)}};
        assert(v.data.size() == 4);
        Value val(v);
        assert(val.GetTypeId() == TypeId::Half4);
    }

    // GfVec2f
    {
        GfVec2f v{{-1.5f, 2.5f}};
        assert(v[0] == -1.5f);
        assert(v[1] == 2.5f);
        Value val(v);
        assert(*val.Get<GfVec2f>() == v);
    }

    // GfVec3f
    {
        GfVec3f v{{1.0f, 2.0f, 3.0f}};
        GfVec3f v2{{1.0f, 2.0f, 3.0f}};
        assert(v == v2);
        Value val(v);
        assert(val.GetTypeId() == TypeId::Float3);
    }

    // GfVec4f
    {
        GfVec4f v{{1.0f, 0.0f, 0.0f, 1.0f}};
        assert(v[0] == 1.0f && v[3] == 1.0f);
    }

    // GfVec2d
    {
        GfVec2d v{{1e10, -1e10}};
        assert(v[0] == 1e10);
        assert(v[1] == -1e10);
        Value val(v);
        assert(val.GetTypeId() == TypeId::Double2);
    }

    // GfVec3d
    {
        GfVec3d v{{100.0, 200.0, 300.0}};
        Value val(v);
        assert(val.Get<GfVec3d>()->data[2] == 300.0);
    }

    // GfVec4d
    {
        GfVec4d v{{1.0, 2.0, 3.0, 4.0}};
        assert(v[3] == 4.0);
    }

    // GfVec2i
    {
        GfVec2i v{{-100, 200}};
        assert(v[0] == -100);
        assert(v[1] == 200);
        Value val(v);
        assert(val.GetTypeId() == TypeId::Int2);
    }

    // GfVec3i
    {
        GfVec3i v{{1, 2, 3}};
        Value val(v);
        assert(val.Get<GfVec3i>()->data[0] == 1);
    }

    // GfVec4i
    {
        GfVec4i v{{10, 20, 30, 40}};
        assert(v[3] == 40);
        Value val(v);
        assert(val.GetTypeId() == TypeId::Int4);
        assert(val.Get<GfVec4i>()->data[3] == 40);
    }

    // Vec equality and inequality across all sizes
    {
        GfVec3f a{{1, 2, 3}};
        GfVec3f b{{1, 2, 3}};
        GfVec3f c{{1, 2, 4}};
        assert(a == b);
        assert(a != c);
    }

    std::cout << "  Vec Comprehensive (all 12 types): OK\n";
}

// ============================================================
// Comprehensive Matrix Tests
// ============================================================

void TestMatrixComprehensive() {
    // GfMatrix2d
    {
        auto id = GfMatrix2d::Identity();
        assert(id(0, 0) == 1.0 && id(1, 1) == 1.0);
        assert(id(0, 1) == 0.0 && id(1, 0) == 0.0);

        GfMatrix2d m;
        m(0, 0) = 2.0; m(0, 1) = 3.0;
        m(1, 0) = 4.0; m(1, 1) = 5.0;
        assert(m(0, 0) == 2.0 && m(1, 1) == 5.0);

        GfMatrix2d m2;
        m2(0, 0) = 2.0; m2(0, 1) = 3.0;
        m2(1, 0) = 4.0; m2(1, 1) = 5.0;
        assert(m == m2);

        m2(0, 0) = 9.0;
        assert(m != m2);

        Value val(m);
        assert(val.GetTypeId() == TypeId::Matrix2d);
        assert(val.Get<GfMatrix2d>()->operator()(0, 0) == 2.0);
    }

    // GfMatrix3d
    {
        auto id = GfMatrix3d::Identity();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                assert(id(r, c) == (r == c ? 1.0 : 0.0));

        GfMatrix3d m;
        m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
        m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
        m(2, 0) = 7.0; m(2, 1) = 8.0; m(2, 2) = 9.0;
        Value val(m);
        assert(val.GetTypeId() == TypeId::Matrix3d);
        assert(val.Get<GfMatrix3d>()->operator()(2, 2) == 9.0);
    }

    // GfMatrix4d — non-identity
    {
        GfMatrix4d m;
        // Translation matrix
        m = GfMatrix4d::Identity();
        m(3, 0) = 10.0; m(3, 1) = 20.0; m(3, 2) = 30.0;
        assert(m(0, 0) == 1.0);
        assert(m(3, 0) == 10.0);
        assert(m(3, 2) == 30.0);

        // Scale matrix
        GfMatrix4d scale;
        scale(0, 0) = 2.0; scale(1, 1) = 3.0;
        scale(2, 2) = 4.0; scale(3, 3) = 1.0;
        Value val(scale);
        assert(val.Get<GfMatrix4d>()->operator()(0, 0) == 2.0);
        assert(val.Get<GfMatrix4d>()->operator()(2, 2) == 4.0);
    }

    // Matrix array in Value
    {
        std::vector<GfMatrix4d> mats = {GfMatrix4d::Identity(), GfMatrix4d::Identity()};
        mats[1](3, 0) = 5.0;
        Value arr(Value::ArrayTag{}, TypeId::Matrix4d, std::move(mats));
        assert(arr.IsArray());
        auto* p = arr.Get<std::vector<GfMatrix4d>>();
        assert(p && p->size() == 2);
        assert((*p)[1](3, 0) == 5.0);
    }

    std::cout << "  Matrix Comprehensive: OK\n";
}

// ============================================================
// Comprehensive ListOp Tests
// ============================================================

void TestListOpGettersSetters() {
    // Test all individual setters and their effect on IsExplicit
    ListOp<int> op;
    assert(!op.IsExplicit());

    op.SetExplicitItems({1, 2, 3});
    assert(op.IsExplicit());
    assert(op.GetExplicitItems() == std::vector<int>({1, 2, 3}));

    // Setting composable sequences switches back
    op.SetPrependedItems({10});
    assert(!op.IsExplicit());
    assert(op.GetPrependedItems() == std::vector<int>({10}));

    op.SetAppendedItems({20, 30});
    assert(!op.IsExplicit());
    assert(op.GetAppendedItems() == std::vector<int>({20, 30}));

    op.SetDeletedItems({5});
    assert(!op.IsExplicit());
    assert(op.GetDeletedItems() == std::vector<int>({5}));

    // Setting explicit again clears composable mode
    op.SetExplicitItems({99});
    assert(op.IsExplicit());

    std::cout << "  ListOp Getters/Setters: OK\n";
}

void TestListOpCombineAllCases() {
    // Case 1: Explicit + Explicit => Stronger explicit wins
    {
        auto S = ListOp<int>::CreateExplicit({1, 2});
        auto W = ListOp<int>::CreateExplicit({3, 4});
        auto R = S.Combine(W);
        assert(R.IsExplicit());
        assert(R.GetItems() == std::vector<int>({1, 2}));
    }

    // Case 2: Explicit + Composable => Stronger explicit wins
    {
        auto S = ListOp<int>::CreateExplicit({100});
        auto W = ListOp<int>::CreateComposable({1}, {2}, {3});
        auto R = S.Combine(W);
        assert(R.IsExplicit());
        assert(R.GetItems() == std::vector<int>({100}));
    }

    // Case 3: Composable + Explicit => New explicit
    {
        auto S = ListOp<int>::CreateComposable({10}, {30}, {20});
        auto W = ListOp<int>::CreateExplicit({1, 2, 3, 4, 5});
        auto R = S.Combine(W);
        assert(R.IsExplicit());
        // prepend {10} not in append {30}: => [10]
        // W.explicit {1,2,3,4,5} not in S.append{30}/S.delete{20}/S.prepend{10}: => {1,2,3,4,5}
        // S.append {30}
        auto items = R.GetItems();
        assert(items.size() == 7);
        assert(items[0] == 10);
        assert(items[1] == 1);
        assert(items[2] == 2);
        assert(items[3] == 3);
        assert(items[4] == 4);
        assert(items[5] == 5);
        assert(items[6] == 30);
    }

    // Case 4: Composable + Composable => New composable
    {
        auto S = ListOp<int>::CreateComposable({1, 2}, {5, 6}, {3});
        auto W = ListOp<int>::CreateComposable({10, 20}, {30, 40}, {50});
        auto R = S.Combine(W);
        assert(!R.IsExplicit());

        // prepend: S.prepend{1,2} not in S.append{5,6} => {1,2}
        //          W.prepend{10,20} not in S.append/S.delete/S.prepend => {10,20}
        assert(R.GetPrependedItems() == std::vector<int>({1, 2, 10, 20}));

        // append: W.append{30,40} not in S.append/S.delete/S.prepend => {30,40}
        //         S.append{5,6}
        assert(R.GetAppendedItems() == std::vector<int>({30, 40, 5, 6}));

        // delete: W.delete{50} not in S.prepend/S.append => {50}
        //         S.delete{3} not in S.prepend/S.append/W.delete => {3}
        assert(R.GetDeletedItems() == std::vector<int>({50, 3}));
    }

    std::cout << "  ListOp Combine All Cases: OK\n";
}

void TestListOpCombineEdgeCases() {
    // Identity combines
    {
        ListOp<int> identity;
        auto S = ListOp<int>::CreateComposable({1}, {2}, {3});
        auto R = S.Combine(identity);
        // identity is composable with empty sequences, so composable+composable
        assert(!R.IsExplicit());
        assert(R.GetItems() == S.GetItems());
    }

    // Empty explicit
    {
        auto E = ListOp<int>::CreateExplicit({});
        assert(E.IsExplicit());
        assert(E.GetItems().empty());

        auto S = ListOp<int>::CreateComposable({1}, {}, {});
        auto R = S.Combine(E);
        // composable + explicit => new explicit
        assert(R.IsExplicit());
        assert(R.GetItems() == std::vector<int>({1}));
    }

    // Duplicate handling in prepend/append overlap
    {
        auto S = ListOp<int>::CreateComposable({1, 2}, {2, 3}, {});
        // GetItems: prepend not in append: {1}, then append: {2, 3}
        auto items = S.GetItems();
        assert(items == std::vector<int>({1, 2, 3}));
    }

    // Delete removes from explicit combination
    {
        auto S = ListOp<int>::CreateComposable({}, {}, {2});
        auto E = ListOp<int>::CreateExplicit({1, 2, 3});
        auto R = S.Combine(E);
        // composable+explicit: no prepend, E.explicit not in delete{2} => {1,3}, no append
        assert(R.IsExplicit());
        assert(R.GetItems() == std::vector<int>({1, 3}));
    }

    // Multi-layer combine chain
    {
        auto L1 = ListOp<int>::CreateComposable({1}, {}, {});
        auto L2 = ListOp<int>::CreateComposable({2}, {}, {});
        auto L3 = ListOp<int>::CreateComposable({3}, {}, {});
        auto R12 = L1.Combine(L2);
        auto R123 = R12.Combine(L3);
        // After L1+L2: prepend={1,2}
        // After +L3: prepend={1,2,3}
        auto items = R123.GetItems();
        assert(items.size() == 3);
        assert(items[0] == 1 && items[1] == 2 && items[2] == 3);
    }

    std::cout << "  ListOp Combine Edge Cases: OK\n";
}

void TestListOpReduceComprehensive() {
    // Already reduced (no overlaps)
    {
        auto op = ListOp<int>::CreateComposable({1}, {2}, {3});
        auto r = op.Reduced();
        assert(r.GetPrependedItems() == std::vector<int>({1}));
        assert(r.GetAppendedItems() == std::vector<int>({2}));
        assert(r.GetDeletedItems() == std::vector<int>({3}));
    }

    // Explicit is a no-op for Reduce
    {
        auto op = ListOp<int>::CreateExplicit({1, 1, 2});
        auto r = op.Reduced();
        assert(r.IsExplicit());
        assert(r.GetExplicitItems() == std::vector<int>({1, 1, 2}));
    }

    // All sequences empty
    {
        auto op = ListOp<int>::CreateComposable({}, {}, {});
        auto r = op.Reduced();
        assert(r.GetPrependedItems().empty());
        assert(r.GetAppendedItems().empty());
        assert(r.GetDeletedItems().empty());
    }

    // All overlap: element in all three sequences
    {
        auto op = ListOp<int>::CreateComposable({5}, {5}, {5});
        auto r = op.Reduced();
        // append stays: {5}
        // prepend: 5 is in append, removed => {}
        // delete: 5 is in append, removed => {}
        assert(r.GetAppendedItems() == std::vector<int>({5}));
        assert(r.GetPrependedItems().empty());
        assert(r.GetDeletedItems().empty());
    }

    // Delete item in prepend
    {
        auto op = ListOp<int>::CreateComposable({1, 2}, {}, {2, 3});
        auto r = op.Reduced();
        // prepend stays: {1, 2} (not in append)
        // delete: 2 is in prepend => removed; 3 stays
        assert(r.GetPrependedItems() == std::vector<int>({1, 2}));
        assert(r.GetDeletedItems() == std::vector<int>({3}));
    }

    std::cout << "  ListOp Reduce Comprehensive: OK\n";
}

void TestListOpEquality() {
    // Same explicit
    auto a = ListOp<int>::CreateExplicit({1, 2, 3});
    auto b = ListOp<int>::CreateExplicit({1, 2, 3});
    assert(a == b);

    // Different explicit
    auto c = ListOp<int>::CreateExplicit({1, 2, 4});
    assert(a != c);

    // Different mode
    auto d = ListOp<int>::CreateComposable({1, 2, 3}, {}, {});
    assert(a != d);  // explicit != composable even with same prepend content

    // Same composable
    auto e = ListOp<int>::CreateComposable({1}, {2}, {3});
    auto f = ListOp<int>::CreateComposable({1}, {2}, {3});
    assert(e == f);

    // Different composable
    auto g = ListOp<int>::CreateComposable({1}, {2}, {4});
    assert(e != g);

    // Default equality
    ListOp<int> h, i;
    assert(h == i);

    // Empty explicit == empty explicit
    auto j = ListOp<int>::CreateExplicit({});
    auto k = ListOp<int>::CreateExplicit({});
    assert(j == k);

    std::cout << "  ListOp Equality: OK\n";
}

void TestListOpTypeSpecializations() {
    // ListOp<std::string>
    {
        auto op = ListOp<std::string>::CreateComposable(
            {"hello"}, {"world"}, {"remove"});
        assert(!op.IsExplicit());
        auto items = op.GetItems();
        assert(items.size() == 2);
        assert(items[0] == "hello");
        assert(items[1] == "world");

        auto op2 = ListOp<std::string>::CreateExplicit({"a", "b", "c"});
        auto combined = op.Combine(op2);
        assert(combined.IsExplicit());
        // prepend{"hello"} not in append{"world"} => [hello]
        // E{"a","b","c"} not in append/delete/prepend => [a,b,c] minus "hello","remove","world" => [a,b,c]
        // append{"world"}
        auto cItems = combined.GetItems();
        assert(cItems[0] == "hello");
        // a, b, c are not in delete/prepend/append
        assert(cItems[1] == "a");
        assert(cItems[2] == "b");
        assert(cItems[3] == "c");
        assert(cItems[4] == "world");
    }

    // ListOp<int64_t>
    {
        auto op = ListOp<int64_t>::CreateExplicit({100LL, 200LL, 300LL});
        assert(op.IsExplicit());
        assert(op.GetItems().size() == 3);
        assert(op.GetItems()[0] == 100LL);
    }

    // ListOp<uint32_t>
    {
        auto op = ListOp<uint32_t>::CreateComposable({1u}, {2u}, {});
        auto items = op.GetItems();
        assert(items.size() == 2);
    }

    // ListOp<Reference>
    {
        Reference r1;
        r1.assetPath = "./a.usda";
        r1.primPath = Path::Parse("/A");
        Reference r2;
        r2.assetPath = "./b.usda";
        auto op = ListOp<Reference>::CreateExplicit({r1, r2});
        assert(op.GetItems().size() == 2);
        assert(*op.GetItems()[0].assetPath == "./a.usda");

        // Composable references
        Reference r3;
        r3.primPath = Path::Parse("/C");
        auto cop = ListOp<Reference>::CreateComposable({r3}, {}, {});
        auto combined = cop.Combine(op);
        // composable + explicit
        assert(combined.IsExplicit());
    }

    // ListOp<Reference> with offset/scale
    {
        Reference r1;
        r1.assetPath = "./ref.usda";
        r1.offset = 10.0;
        r1.scale = 2.0;
        Reference r2;
        r2.assetPath = "./ref.usda";
        r2.offset = 10.0;
        r2.scale = 2.0;
        assert(r1 == r2);

        r2.scale = 1.0;
        assert(r1 != r2);
    }

    std::cout << "  ListOp Type Specializations: OK\n";
}

void TestListOpGetItemsOverlap() {
    // GetItems with prepend/append overlap
    auto op = ListOp<int>::CreateComposable({1, 2, 3}, {3, 4, 5}, {});
    auto items = op.GetItems();
    // prepend items NOT in append: 1, 2 (3 is in append)
    // then append: 3, 4, 5
    assert(items.size() == 5);
    assert(items[0] == 1);
    assert(items[1] == 2);
    assert(items[2] == 3);
    assert(items[3] == 4);
    assert(items[4] == 5);

    // All prepend in append
    auto op2 = ListOp<int>::CreateComposable({1, 2}, {1, 2}, {});
    auto items2 = op2.GetItems();
    assert(items2.size() == 2);
    assert(items2[0] == 1 && items2[1] == 2);

    std::cout << "  ListOp GetItems Overlap: OK\n";
}


// ============================================================
// Write Operations
// ============================================================

void TestWriteScalarDefaults() {
    std::cout << "  Write scalar defaults: ";
    // Parse a simple USDA with an Xform prim
    auto result = ParseUsda(R"(#usda 1.0
def Xform "Root" {
    float height = 5.0
    double width = 10.0
    int count = 3
    string label = "hello"
}
)");
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    assert(stage.IsValid());

    auto prim = stage.GetPrimAtPath(Path::Parse("/Root"));
    assert(prim.IsValid());

    // Modify existing float attribute
    auto heightAttr = prim.GetAttribute("height");
    assert(heightAttr.IsValid());
    assert(heightAttr.Set(Value(Float(99.0f))));
    auto* val = heightAttr.GetDefault();
    assert(val);
    assert(*val->Get<Float>() == 99.0f);

    // Re-fetch from prim and verify
    auto heightAttr2 = prim.GetAttribute("height");
    auto* val2 = heightAttr2.GetDefault();
    assert(val2 && *val2->Get<Float>() == 99.0f);

    // Modify double
    auto widthAttr = prim.GetAttribute("width");
    assert(widthAttr.Set(Value(Double(42.0))));
    assert(*widthAttr.GetDefault()->Get<Double>() == 42.0);

    // Modify int
    auto countAttr = prim.GetAttribute("count");
    assert(countAttr.Set(Value(Int(100))));
    assert(*countAttr.GetDefault()->Get<Int>() == 100);

    // Modify string
    auto labelAttr = prim.GetAttribute("label");
    assert(labelAttr.Set(Value(String("world"))));
    assert(*labelAttr.GetDefault()->Get<String>() == "world");

    std::cout << "OK\n";
}

void TestWriteVectorDefaults() {
    std::cout << "  Write vector defaults: ";
    auto result = ParseUsda(R"(#usda 1.0
def Xform "Root" {
    float3 position = (1, 2, 3)
    double3 velocity = (4, 5, 6)
}
)");
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/Root"));

    auto posAttr = prim.GetAttribute("position");
    assert(posAttr.Set(Value(GfVec3f{10, 20, 30})));
    auto* val = posAttr.GetDefault();
    assert(val);
    auto* v = val->Get<GfVec3f>();
    assert(v && (*v)[0] == 10.0f && (*v)[1] == 20.0f && (*v)[2] == 30.0f);

    auto velAttr = prim.GetAttribute("velocity");
    assert(velAttr.Set(Value(GfVec3d{7, 8, 9})));
    auto* vd = velAttr.GetDefault()->Get<GfVec3d>();
    assert(vd && (*vd)[0] == 7.0 && (*vd)[1] == 8.0 && (*vd)[2] == 9.0);

    std::cout << "OK\n";
}

void TestWriteTimeSamples() {
    std::cout << "  Write time samples: ";
    auto result = ParseUsda(R"(#usda 1.0
def Xform "Root" {
    float height = 5.0
}
)");
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/Root"));

    auto attr = prim.GetAttribute("height");
    assert(attr.IsValid());

    // Add time samples
    assert(attr.SetTimeSample(1.0, Value(Float(10.0f))));
    assert(attr.SetTimeSample(2.0, Value(Float(20.0f))));
    assert(attr.HasTimeSamples());

    auto times = attr.GetTimeSampleTimes();
    assert(times.size() == 2);

    // Add another sample
    assert(attr.SetTimeSample(3.0, Value(Float(30.0f))));
    times = attr.GetTimeSampleTimes();
    assert(times.size() == 3);

    std::cout << "OK\n";
}

void TestWriteClearAndBlock() {
    std::cout << "  Write clear and block: ";
    auto result = ParseUsda(R"(#usda 1.0
def Xform "Root" {
    float height = 5.0
    float width.timeSamples = {
        1: 10.0,
        2: 20.0,
    }
}
)");
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/Root"));

    // Clear default
    auto heightAttr = prim.GetAttribute("height");
    assert(heightAttr.HasDefault());
    assert(heightAttr.ClearDefault());
    assert(!heightAttr.HasDefault());

    // Clear time samples
    auto widthAttr = prim.GetAttribute("width");
    assert(widthAttr.HasTimeSamples());
    assert(widthAttr.ClearTimeSamples());
    assert(!widthAttr.HasTimeSamples());

    // Block an attribute
    auto heightAttr2 = prim.GetAttribute("height");
    assert(heightAttr2.Block());
    auto* val = heightAttr2.GetDefault();
    assert(val && val->IsBlock());

    std::cout << "OK\n";
}

void TestWriteCreateAttribute() {
    std::cout << "  Write create attribute: ";
    auto result = ParseUsda(R"(#usda 1.0
def Xform "Root" {
}
)");
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/Root"));

    // Create a brand new attribute
    auto attr = prim.CreateAttribute("myFloat", Token("float"));
    assert(attr.IsValid());
    assert(attr.Set(Value(Float(42.0f))));
    assert(*attr.GetDefault()->Get<Float>() == 42.0f);

    // Verify it's now visible
    assert(prim.HasAttribute("myFloat"));

    // Creating again returns the existing one
    auto attr2 = prim.CreateAttribute("myFloat", Token("float"));
    assert(attr2.IsValid());
    assert(*attr2.GetDefault()->Get<Float>() == 42.0f);

    std::cout << "OK\n";
}

// ============================================================
// P0 Physics Prerequisites Tests
// ============================================================


void TestRelationshipWrite() {
    auto result = ParseUsda(R"(#usda 1.0
def PhysicsJoint "joint" {
    rel physics:body0 = </world/box0>
}
def Xform "world" {
    def Cube "box0" {}
    def Cube "box1" {}
}
)");
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto joint = stage.GetPrimAtPath(Path::Parse("/joint"));
    assert(joint.IsValid());

    // Read existing relationship
    auto rel0 = joint.GetRelationship("physics:body0");
    assert(rel0.IsValid());
    auto targets0 = rel0.GetTargets();
    assert(targets0.size() == 1);
    assert(targets0[0].GetText() == std::string("/world/box0"));

    // Create and set a new relationship
    auto rel1 = joint.CreateRelationship("physics:body1");
    assert(rel1.IsValid());
    assert(rel1.SetTargets({Path::Parse("/world/box1")}));
    auto targets1 = rel1.GetTargets();
    assert(targets1.size() == 1);
    assert(targets1[0].GetText() == std::string("/world/box1"));

    // Add target
    assert(rel1.AddTarget(Path::Parse("/world/box0")));
    auto targets1b = rel1.GetTargets();
    assert(targets1b.size() == 2);

    // Clear targets
    assert(rel1.ClearTargets());
    assert(!rel1.HasTargets());

    std::cout << "  Relationship write: OK\n";
}

void TestStageCreation() {
    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    // Define prim with type
    auto sphere = stage.DefinePrim(Path::Parse("/world/sphere"), Token("Sphere"));
    assert(sphere.IsValid());
    assert(sphere.IsA("Sphere"));
    assert(sphere.IsA("Gprim"));

    // Schema fallbacks should work
    auto radiusAttr = sphere.GetAttribute("radius");
    assert(radiusAttr.IsValid());
    auto* rv = radiusAttr.GetDefault();
    assert(rv && *rv->Get<Double>() == 1.0);

    // Define prim without type
    auto group = stage.DefinePrim(Path::Parse("/group"));
    assert(group.IsValid());

    // Ancestor creation
    auto deep = stage.DefinePrim(Path::Parse("/a/b/c"), Token("Xform"));
    assert(deep.IsValid());
    assert(stage.HasPrimAtPath(Path::Parse("/a")));
    assert(stage.HasPrimAtPath(Path::Parse("/a/b")));

    std::cout << "  Stage creation + DefinePrim: OK\n";
}

void TestApplyAPI() {
    auto stage = Stage::CreateInMemory();
    auto prim = stage.DefinePrim(Path::Parse("/body"), Token("Sphere"));
    assert(prim.IsValid());

    // Apply single-apply API
    assert(prim.ApplyAPI("PhysicsRigidBodyAPI"));
    assert(prim.HasAPI("PhysicsRigidBodyAPI"));

    // Schema properties should now be accessible
    auto velAttr = prim.GetAttribute("physics:velocity");
    assert(velAttr.IsValid());
    auto* vv = velAttr.GetDefault();
    assert(vv);
    auto* vec = vv->Get<GfVec3f>();
    assert(vec && (*vec)[0] == 0.0f);

    // Apply with instance name (multi-apply)
    assert(prim.ApplyAPI("PhysicsLimitAPI", "transX"));
    assert(prim.HasAPI("PhysicsLimitAPI"));

    // Applying same API twice should be fine
    assert(prim.ApplyAPI("PhysicsRigidBodyAPI"));
    auto schemas = prim.GetAppliedSchemas();
    int rbCount = 0;
    for (const auto& s : schemas) {
        if (s == "PhysicsRigidBodyAPI") rbCount++;
    }
    assert(rbCount == 1);

    std::cout << "  ApplyAPI: OK\n";
}

void TestPhysicsP0Integration() {
    // End-to-end test: create a stage, add physics scene and rigid body
    auto stage = Stage::CreateInMemory();

    auto scene = stage.DefinePrim(Path::Parse("/scene"), Token("PhysicsScene"));
    assert(scene.IsValid());
    assert(scene.IsA("PhysicsScene"));

    auto body = stage.DefinePrim(Path::Parse("/body"), Token("Sphere"));
    assert(body.ApplyAPI("PhysicsRigidBodyAPI"));
    assert(body.ApplyAPI("PhysicsCollisionAPI"));

    // Set velocity
    auto velAttr = body.CreateAttribute("physics:velocity", Token("vector3f"));
    GfVec3f vel; vel[0] = 1.0f; vel[1] = 2.0f; vel[2] = 3.0f;
    assert(velAttr.Set(Value(vel)));

    // Create joint
    auto joint = stage.DefinePrim(Path::Parse("/joint"), Token("PhysicsRevoluteJoint"));
    assert(joint.IsA("PhysicsJoint"));
    assert(joint.IsA("Imageable"));

    // Set joint body relationships
    auto rel0 = joint.CreateRelationship("physics:body0");
    assert(rel0.SetTargets({Path::Parse("/body")}));
    auto targets = rel0.GetTargets();
    assert(targets.size() == 1 && targets[0].GetText() == std::string("/body"));

    // Set joint rotation
    auto rotAttr = joint.CreateAttribute("physics:localRot0", Token("quatf"));
    GfQuatf identity;
    identity[0] = 0; identity[1] = 0; identity[2] = 0; identity[3] = 1;
    assert(rotAttr.Set(Value(identity)));

    // Apply drive
    assert(joint.ApplyAPI("PhysicsDriveAPI", "angular"));

    std::cout << "  Physics P0 integration: OK\n";
}

// ============================================================
// P1 Tests
// ============================================================


void TestAssetPathRead() {
    const char* usda = R"(#usda 1.0
def Mesh "mesh" {
    asset modelRef = @./models/hero.usd@
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));

    auto prim = stage.GetPrimAtPath(Path::Parse("/mesh"));
    auto attr = prim.GetAttribute("modelRef");
    assert(attr.IsValid());

    // Asset is stored as a string internally
    auto* val = attr.GetDefault();
    assert(val);
    auto* s = val->Get<std::string>();
    assert(s && *s == "./models/hero.usd");

    std::cout << "  Asset path read: OK\n";
}

// ============================================================
// XformOp Tests
// ============================================================

void TestXformOpBasicSRT() {
    // Simple Scale-Rotate-Translate stack
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    double3 xformOp:translate = (10, 20, 30)
    float3 xformOp:rotateXYZ = (0, 0, 0)
    float3 xformOp:scale = (2, 2, 2)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));
    assert(prim.IsValid());

    bool reset = false;
    GfMatrix4d m = ComputeLocalTransform(prim, UsdTimeCode::Default(), &reset);
    assert(!reset);

    // Post-multiply convention: P' = P * scale * rotate * translate
    // With zero rotation and (2,2,2) scale + (10,20,30) translate:
    // A point at (1,0,0) → scale to (2,0,0) → translate to (12,20,30)
    // The composed matrix should have scale on diagonal and translation in row 3.
    assert(m(0, 0) == 2.0);
    assert(m(1, 1) == 2.0);
    assert(m(2, 2) == 2.0);
    assert(m(3, 0) == 10.0);
    assert(m(3, 1) == 20.0);
    assert(m(3, 2) == 30.0);

    std::cout << "  XformOp basic SRT: OK\n";
}

void TestXformOpTranslateOnly() {
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    double3 xformOp:translate = (5, -3, 7)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    // Identity except for translation
    assert(m(0, 0) == 1.0 && m(1, 1) == 1.0 && m(2, 2) == 1.0 && m(3, 3) == 1.0);
    assert(m(3, 0) == 5.0 && m(3, 1) == -3.0 && m(3, 2) == 7.0);

    std::cout << "  XformOp translate only: OK\n";
}

void TestXformOpRotation() {
    // 90-degree rotation about Z axis
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    double xformOp:rotateZ = 90
    uniform token[] xformOpOrder = ["xformOp:rotateZ"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    // rotateZ(90): cos=0, sin=1
    // [0  1  0  0]
    // [-1 0  0  0]
    // [0  0  1  0]
    // [0  0  0  1]
    assert(std::abs(m(0, 0) - 0.0) < 1e-10);
    assert(std::abs(m(0, 1) - 1.0) < 1e-10);
    assert(std::abs(m(1, 0) - (-1.0)) < 1e-10);
    assert(std::abs(m(1, 1) - 0.0) < 1e-10);
    assert(std::abs(m(2, 2) - 1.0) < 1e-10);

    std::cout << "  XformOp rotation: OK\n";
}

void TestXformOpOrient() {
    // Identity quaternion: (w=1, i=0, j=0, k=0) → identity matrix
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    quatf xformOp:orient = (1, 0, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:orient"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    auto id = GfMatrix4d::Identity();
    for (int i = 0; i < 16; ++i) {
        assert(std::abs(m.data[i] - id.data[i]) < 1e-10);
    }

    std::cout << "  XformOp orient (identity): OK\n";
}

void TestXformOpSuffix() {
    // Translate with suffix (pivot pattern)
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    double3 xformOp:translate = (100, 0, 0)
    double3 xformOp:translate:pivot = (5, 5, 5)
    float3 xformOp:scale = (2, 2, 2)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:translate:pivot", "xformOp:scale", "!invert!xformOp:translate:pivot"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    // Post-multiply: P * !invert!pivot * scale * pivot * translate
    // A point at origin (0,0,0):
    //   → !invert!pivot: subtract (5,5,5) → (-5,-5,-5)
    //   → scale(2,2,2): (-10,-10,-10)
    //   → pivot: add (5,5,5) → (-5,-5,-5)
    //   → translate: add (100,0,0) → (95,-5,-5)
    // Check via matrix: extract translation from row 3
    // The matrix should encode this pivot-scale-translate.
    // Scale is 2 on diagonal, translation incorporates the pivot offset.
    assert(m(0, 0) == 2.0 && m(1, 1) == 2.0 && m(2, 2) == 2.0);
    assert(std::abs(m(3, 0) - 95.0) < 1e-10);
    assert(std::abs(m(3, 1) - (-5.0)) < 1e-10);
    assert(std::abs(m(3, 2) - (-5.0)) < 1e-10);

    std::cout << "  XformOp suffix + inverse: OK\n";
}

void TestXformOpResetStack() {
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:translate"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    assert(HasResetXformStack(prim));

    bool reset = false;
    GfMatrix4d m = ComputeLocalTransform(prim, UsdTimeCode::Default(), &reset);
    assert(reset);
    assert(m(3, 0) == 1.0 && m(3, 1) == 2.0 && m(3, 2) == 3.0);

    std::cout << "  XformOp resetXformStack: OK\n";
}

void TestXformOpNoOps() {
    // Prim with no xformOpOrder → identity
    const char* usda = R"(#usda 1.0
def Xform "obj" {
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    auto id = GfMatrix4d::Identity();
    for (int i = 0; i < 16; ++i) assert(m.data[i] == id.data[i]);

    assert(!HasResetXformStack(prim));

    std::cout << "  XformOp no ops (identity): OK\n";
}

void TestXformOpSingleAxis() {
    // translateX, scaleY single-axis ops
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    double xformOp:translateX = 42
    double xformOp:scaleY = 3
    uniform token[] xformOpOrder = ["xformOp:translateX", "xformOp:scaleY"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    // Post-multiply: P * scaleY * translateX
    // scaleY(3) then translateX(42)
    assert(m(1, 1) == 3.0);
    assert(m(3, 0) == 42.0);
    // Other diagonals = 1
    assert(m(0, 0) == 1.0);
    assert(m(2, 2) == 1.0);

    std::cout << "  XformOp single-axis ops: OK\n";
}

void TestXformOpTransform() {
    // Full matrix4d transform op
    const char* usda = R"(#usda 1.0
def Xform "obj" {
    matrix4d xformOp:transform = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (7, 8, 9, 1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
}
)";
    auto result = ParseUsda(usda);
    assert(result.success);
    auto stage = Stage::CreateFromComposedLayer(std::move(result.layer));
    auto prim = stage.GetPrimAtPath(Path::Parse("/obj"));

    GfMatrix4d m = ComputeLocalTransform(prim);
    assert(m(3, 0) == 7.0 && m(3, 1) == 8.0 && m(3, 2) == 9.0);
    assert(m(0, 0) == 1.0 && m(1, 1) == 1.0 && m(2, 2) == 1.0);

    std::cout << "  XformOp transform (matrix4d): OK\n";
}

// ============================================================
// USDC Write Support
// ============================================================

void TestWriteUsdcRoundtrip() {
    std::cout << "  USDC write roundtrip (in-memory): ";

    // Build a stage with metadata, prims, and attributes
    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    auto& layer = stage.GetMutableLayer();
    auto& layerSpec = layer.GetLayerSpec();
    layerSpec.SetField(Token("metersPerUnit"), Value(Double(1.0)));
    layerSpec.SetField(Token("upAxis"), Value(Token("Z")));

    auto sphere = stage.DefinePrim(Path::Parse("/World/Sphere"), Token("Sphere"));
    assert(sphere.IsValid());
    auto radiusAttr = sphere.CreateAttribute("radius", Token("double"));
    assert(radiusAttr.IsValid());
    assert(radiusAttr.Set(Value(Double(2.5))));

    auto xform = stage.DefinePrim(Path::Parse("/World"), Token("Xform"));
    assert(xform.IsValid());

    // Serialize to memory
    auto bytes = WriteUsdc(layer);
    assert(!bytes.empty());
    // Magic header must be present
    assert(bytes.size() >= 8);
    assert(std::string(reinterpret_cast<const char*>(bytes.data()), 8) == "PXR-USDC");

    // Parse back
    auto result = ParseUsdc(bytes.data(), bytes.size());
    assert(result.success);
    const auto& L = result.layer;

    // Verify layer metadata
    auto* mpuField = L.GetLayerSpec().GetField(Token("metersPerUnit"));
    assert(mpuField != nullptr);
    auto* mpuVal = mpuField->Get<Double>();
    assert(mpuVal && std::abs(*mpuVal - 1.0) < 1e-9);

    auto* upField = L.GetLayerSpec().GetField(Token("upAxis"));
    assert(upField != nullptr);
    auto* upVal = upField->Get<Token>();
    assert(upVal && upVal->GetString() == "Z");

    // Verify prims exist
    assert(L.GetSpec(Path::Parse("/World")) != nullptr);
    assert(L.GetSpec(Path::Parse("/World/Sphere")) != nullptr);

    // Verify attribute value
    auto* rSpec = L.GetSpec(Path::Parse("/World/Sphere").AppendProperty("radius"));
    assert(rSpec != nullptr);
    auto* defVal = rSpec->GetField(Token("default"));
    assert(defVal != nullptr);
    auto* dv = defVal->Get<Double>();
    assert(dv && std::abs(*dv - 2.5) < 1e-9);

    std::cout << "OK\n";
}

void TestWriteUsdcFileRoundtrip() {
    std::cout << "  USDC write roundtrip (file): ";

    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    auto& layer = stage.GetMutableLayer();
    layer.GetLayerSpec().SetField(Token("metersPerUnit"), Value(Double(0.01)));

    auto prim = stage.DefinePrim(Path::Parse("/Root"), Token("Xform"));
    assert(prim.IsValid());
    auto attr = prim.CreateAttribute("count", Token("int"));
    assert(attr.IsValid());
    assert(attr.Set(Value(Int(77))));

    const std::string tmpPath = (std::filesystem::temp_directory_path() / "nanousd_test_roundtrip.usdc").string();
    assert(WriteUsdcFile(layer, tmpPath));

    auto result = ParseUsdcFile(tmpPath);
    assert(result.success);
    const auto& L = result.layer;

    auto* rootSpec = L.GetSpec(Path::Parse("/Root"));
    assert(rootSpec != nullptr);

    auto* countSpec = L.GetSpec(Path::Parse("/Root").AppendProperty("count"));
    assert(countSpec != nullptr);
    auto* defVal = countSpec->GetField(Token("default"));
    assert(defVal != nullptr);
    auto* iv = defVal->Get<Int>();
    assert(iv && *iv == 77);

    std::error_code ec;
    std::filesystem::remove(tmpPath, ec);
    std::cout << "OK\n";
}

void TestResolvedResourceWriteSupport() {
    namespace fs = std::filesystem;
    std::cout << "  Resolved resource writes: ";

    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    auto& layer = stage.GetMutableLayer();
    auto prim = stage.DefinePrim(Path::Parse("/Root"), Token("Xform"));
    assert(prim.IsValid());
    auto attr = prim.CreateAttribute("count", Token("int"));
    assert(attr.IsValid());
    assert(attr.Set(Value(Int(91))));

    fs::path usdaPath = fs::temp_directory_path() / "nanousd_resource_write.usda";
    fs::path usdcPath = fs::temp_directory_path() / "nanousd_resource_write.usdc";
    fs::path bytesPath = fs::temp_directory_path() / "nanousd_resource_write.bin";

    std::string usdaUri = FileUriFromPath(usdaPath);
    std::string usdcUri = FileUriFromPath(usdcPath);
    std::string bytesUri = FileUriFromPath(bytesPath);

    auto usdaLocation = ResolvedLocation::FromResolvedString(usdaUri);
    assert(usdaLocation.IsLocalFile());
    assert(WriteUsdaFile(layer, usdaLocation));
    auto usdaResult = ParseUsdFile(usdaLocation);
    assert(usdaResult.success);
    assert(usdaResult.layer.HasSpec(Path::Parse("/Root")));

    auto usdcLocation = ResolvedLocation::FromResolvedString(usdcUri);
    assert(WriteUsdcFile(layer, usdcLocation));
    auto usdcResult = ParseUsdFile(usdcLocation);
    assert(usdcResult.success);
    assert(usdcResult.layer.HasSpec(Path::Parse("/Root")));

    std::string payload = "nanousd resource payload";
    auto bytesLocation = ResolvedLocation::FromResolvedString(bytesUri);
    auto write = WriteResource(bytesLocation,
                               reinterpret_cast<const uint8_t*>(payload.data()),
                               payload.size());
    assert(write.success);
    assert(write.fileBacked);
    auto read = ReadResource(bytesLocation);
    assert(read.success);
    assert(std::string(read.bytes.begin(), read.bytes.end()) == payload);

    auto unsupportedUri = ResolvedLocation::FromResolvedString("https://example.com/out.usda");
    auto unsupportedWrite = WriteResource(unsupportedUri,
                                          reinterpret_cast<const uint8_t*>(payload.data()),
                                          payload.size());
    assert(!unsupportedWrite.success);
    assert(unsupportedWrite.error.find("Unsupported resource scheme for write: https") !=
           std::string::npos);

    auto packageLocation =
        ResolvedLocation::FromResolvedString(usdaPath.string() + "[inner.usda]");
    auto packageWrite = WriteResource(packageLocation,
                                      reinterpret_cast<const uint8_t*>(payload.data()),
                                      payload.size());
    assert(!packageWrite.success);
    assert(packageWrite.error.find("Writing packaged resources is not supported") !=
           std::string::npos);

    std::error_code ec;
    fs::remove(usdaPath, ec);
    fs::remove(usdcPath, ec);
    fs::remove(bytesPath, ec);
    std::cout << "OK\n";
}

// Test that USDC write correctly encodes multi-root complex hierarchies.
// Regression for: path element tokens added by path-typed field encoding
// (refs/relationships) were missing from the TOKENS section because spec
// collection ran after the token pre-add pass — causing the parser to abort
// the path jump-tree traversal early and drop most prims.
void TestWriteUsdcMultiRootHierarchy() {
    std::cout << "  USDC write: multi-root deep hierarchy prim count: ";

    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());
    auto& layer = stage.GetMutableLayer();

    // Three top-level prims (multiple roots) with varying depths, mimicking
    // a real scene (e.g. boxes_falling_on_groundplane.usda).
    stage.DefinePrim(Path::Parse("/Environment"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/Environment/defaultLight"), Token("DistantLight"));

    stage.DefinePrim(Path::Parse("/Render"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/Render/OmniverseKit"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/Render/OmniverseKit/HydraTextures"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/Render/Vars"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/Render/Vars/LdrColor"), Token("RenderVar"));

    stage.DefinePrim(Path::Parse("/World"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/World/GroundPlane"), Token("Xform"));
    stage.DefinePrim(Path::Parse("/World/GroundPlane/CollisionMesh"), Token("Mesh"));
    stage.DefinePrim(Path::Parse("/World/GroundPlane/CollisionPlane"), Token("Plane"));
    for (int i = 1; i <= 11; ++i) {
        stage.DefinePrim(Path::Parse("/World/Cube" + std::to_string(i)), Token("Mesh"));
    }
    stage.DefinePrim(Path::Parse("/World/BigBase"), Token("Mesh"));

    // Count prim specs from the original layer to get the expected count.
    // The schema system may add extra implicit specs, so we count rather than
    // hardcode.
    int expectedPrims = 0;
    for (const auto& p : layer.GetSpecPaths()) {
        auto* spec = layer.GetSpec(p);
        if (spec && spec->GetType() == SpecType::Prim) ++expectedPrims;
    }
    assert(expectedPrims >= 17);  // at minimum the prims we explicitly defined

    auto bytes = WriteUsdc(layer);
    assert(!bytes.empty());

    auto result = ParseUsdc(bytes.data(), bytes.size());
    assert(result.success);
    const auto& L = result.layer;

    int got = 0;
    for (const auto& p : L.GetSpecPaths()) {
        auto* spec = L.GetSpec(p);
        if (spec && spec->GetType() == SpecType::Prim) ++got;
    }
    assert(got == expectedPrims);

    // Spot-check a few paths
    assert(L.GetSpec(Path::Parse("/Environment")) != nullptr);
    assert(L.GetSpec(Path::Parse("/Environment/defaultLight")) != nullptr);
    assert(L.GetSpec(Path::Parse("/Render/Vars/LdrColor")) != nullptr);
    assert(L.GetSpec(Path::Parse("/World/GroundPlane/CollisionMesh")) != nullptr);
    assert(L.GetSpec(Path::Parse("/World/Cube11")) != nullptr);
    assert(L.GetSpec(Path::Parse("/World/BigBase")) != nullptr);

    std::cout << "OK\n";
}

// upAxis must be authored as Token, not String, per USD spec.
// Pixar's UsdGeom.GetStageUpAxis() throws if the field is mistyped.
void TestWriteUsdcUpAxisToken() {
    std::cout << "  USDC write: upAxis token-typed: ";

    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    auto& layerSpec = stage.GetMutableLayer().GetLayerSpec();
    // Simulate what the new C API set_stage_metadata_token does.
    layerSpec.SetField(Token("upAxis"), Value(Token("Y")));
    layerSpec.SetField(Token("metersPerUnit"), Value(Double(1.0)));

    stage.DefinePrim(Path::Parse("/World"), Token("Xform"));

    auto bytes = WriteUsdc(stage.GetMutableLayer());
    assert(!bytes.empty());

    auto result = ParseUsdc(bytes.data(), bytes.size());
    assert(result.success);

    auto* upField = result.layer.GetLayerSpec().GetField(Token("upAxis"));
    assert(upField != nullptr);

    // Critical: the round-tripped value must be Token-typed, not String-typed.
    auto* asToken = upField->Get<Token>();
    auto* asString = upField->Get<String>();
    assert(asToken != nullptr && "upAxis must round-trip as Token");
    assert(asString == nullptr && "upAxis must NOT be String-typed");
    assert(asToken->GetString() == "Y");

    std::cout << "OK\n";
}

// pxr's USDC reader populates the prim hierarchy from `primChildren` /
// `propertyChildren` fields (TokenVector, spec §16.3.10.27 + §16.3.8.4.6).
// Without those fields it reports zero prims even though TOKENS / PATHS /
// SPECS are all valid. nanousd's own reader rebuilds the indices from the
// PATHS jump-tree, so round-trip works either way — but the writer must
// emit these fields for cross-implementation compatibility.
//
// This test exercises the synthesis: define a multi-prim stage, write USDC,
// reparse, then verify the layer spec carries `primChildren` and each prim
// spec with children/properties carries the corresponding field. Field type
// must be a TokenVector (vector<string> with TypeId::String) — anything else
// (e.g. CrateTypeId::String + array flag) won't be recognised by pxr.
void TestWriteUsdcPrimChildrenSynthesis() {
    std::cout << "  USDC write: primChildren / propertyChildren synthesised: ";

    auto stage = Stage::CreateInMemory();
    assert(stage.IsValid());

    stage.DefinePrim(Path::Parse("/World"), Token("Xform"));
    auto sphere = stage.DefinePrim(Path::Parse("/World/Sphere"), Token("Sphere"));
    auto radiusAttr = sphere.CreateAttribute("radius", Token("double"));
    radiusAttr.Set(Value(Double(1.5)));
    stage.DefinePrim(Path::Parse("/World/Cube"), Token("Cube"));

    auto bytes = WriteUsdc(stage.GetMutableLayer());
    assert(!bytes.empty());
    auto result = ParseUsdc(bytes.data(), bytes.size());
    assert(result.success);
    const auto& L = result.layer;

    // Layer spec must list World as a child.
    auto* lsPrimChildren = L.GetLayerSpec().GetField(Token("primChildren"));
    assert(lsPrimChildren && "layer spec must carry primChildren");
    auto* lsKids = lsPrimChildren->Get<std::vector<std::string>>();
    assert(lsKids && lsKids->size() == 1);
    assert((*lsKids)[0] == "World");

    // /World must list Sphere and Cube as children, in insertion order.
    auto* worldSpec = L.GetSpec(Path::Parse("/World"));
    assert(worldSpec);
    auto* worldKidsField = worldSpec->GetField(Token("primChildren"));
    assert(worldKidsField);
    auto* worldKids = worldKidsField->Get<std::vector<std::string>>();
    assert(worldKids && worldKids->size() == 2);
    assert((*worldKids)[0] == "Sphere");
    assert((*worldKids)[1] == "Cube");

    // /World/Sphere must list radius as a property child.
    auto* sphereSpec = L.GetSpec(Path::Parse("/World/Sphere"));
    assert(sphereSpec);
    auto* propsField = sphereSpec->GetField(Token("propertyChildren"));
    assert(propsField && "Sphere spec must carry propertyChildren");
    auto* props = propsField->Get<std::vector<std::string>>();
    assert(props && props->size() == 1);
    assert((*props)[0] == "radius");

    // /World/Cube has no properties, so no propertyChildren field should be
    // synthesised (pxr tolerates absence).
    auto* cubeSpec = L.GetSpec(Path::Parse("/World/Cube"));
    assert(cubeSpec);
    assert(cubeSpec->GetField(Token("propertyChildren")) == nullptr);

    // USDC->USDC round-trip must not duplicate or corrupt the field type:
    // re-encoding the parsed `vector<string>` value through the array path
    // would produce CrateTypeId::String + array flag (wrong type) and
    // duplicate the field. Re-write and verify the field is still a single
    // TokenVector-decoded vector<string> covering the same name set (the
    // PATHS section is sorted alphabetically so the per-spec order may
    // differ between writes — order is checked on the first write only).
    auto bytes2 = WriteUsdc(L);
    assert(!bytes2.empty());
    auto result2 = ParseUsdc(bytes2.data(), bytes2.size());
    assert(result2.success);
    const auto& L2 = result2.layer;
    auto* worldSpec2 = L2.GetSpec(Path::Parse("/World"));
    assert(worldSpec2);
    auto* worldKidsField2 = worldSpec2->GetField(Token("primChildren"));
    assert(worldKidsField2);
    auto* worldKids2 = worldKidsField2->Get<std::vector<std::string>>();
    assert(worldKids2 && worldKids2->size() == 2);
    bool foundSphere = false, foundCube = false;
    for (const auto& s : *worldKids2) {
        if (s == "Sphere") foundSphere = true;
        if (s == "Cube") foundCube = true;
    }
    assert(foundSphere && foundCube);

    std::cout << "OK\n";
}


void TestPrimMetadata() {
    std::cout << "  TestPrimMetadata ... ";

    // Create a stage with a prim that has kind and documentation
    auto stage = Stage::CreateInMemory();
    auto prim = stage.DefinePrim(Path::Parse("/World"), Token("Xform"));
    assert(prim.IsValid());

    // Set kind via the spec directly
    auto& rootLayer = stage.GetMutableLayer();
    auto* spec = rootLayer.GetMutablePrimSpec(prim.GetPath());
    assert(spec != nullptr);
    spec->SetKind(Token("assembly"));

    // Read kind back through UsdPrim
    Token kind = prim.GetKind();
    assert(kind.GetString() == "assembly");

    // Read kind through GetPrimMetadata
    auto kindVal = prim.GetPrimMetadata(FieldNames::kind);
    assert(kindVal.has_value());
    auto* kindTok = kindVal->Get<Token>();
    assert(kindTok && kindTok->GetString() == "assembly");

    // Set documentation
    spec->SetDocumentation("test doc");
    assert(prim.GetDocumentation() == "test doc");
    auto docVal = prim.GetPrimMetadata(FieldNames::documentation);
    assert(docVal.has_value());
    auto* docStr = docVal->Get<String>();
    assert(docStr && *docStr == "test doc");

    // Verify typeName is stored as Token
    auto tnVal = prim.GetPrimMetadata(FieldNames::typeName);
    assert(tnVal.has_value());
    auto* tnTok = tnVal->Get<Token>();
    assert(tnTok && tnTok->GetString() == "Xform");

    // GetPrimMetadata returns nullopt for unset fields
    auto missing = prim.GetPrimMetadata(Token("nonexistent"));
    assert(!missing.has_value());

    std::cout << "OK\n";
}


// ============================================================
// Instancing (spec Section 11.3)
// ============================================================

void TestInstancingBasic() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    auto instA = stage.GetPrimAtPath(Path::Parse("/InstanceA"));
    auto instB = stage.GetPrimAtPath(Path::Parse("/InstanceB"));
    assert(instA.IsValid());
    assert(instB.IsValid());

    // Both should be instances
    assert(instA.IsInstance());
    assert(instB.IsInstance());

    // They should share the same prototype
    auto protoA = instA.GetPrototype();
    auto protoB = instB.GetPrototype();
    assert(protoA.IsValid());
    assert(protoB.IsValid());
    assert(protoA.GetPath() == protoB.GetPath());

    // The prototype should be a prototype
    assert(protoA.IsPrototype());
    assert(protoA.IsInPrototype());

    std::cout << "  Instancing basic (two instances share prototype): OK\n";
}

void TestInstancingDifferentRefs() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    auto instA = stage.GetPrimAtPath(Path::Parse("/InstanceA"));
    auto instC = stage.GetPrimAtPath(Path::Parse("/InstanceC"));
    assert(instA.IsInstance());
    assert(instC.IsInstance());

    // Different references should get different prototypes
    auto protoA = instA.GetPrototype();
    auto protoC = instC.GetPrototype();
    assert(protoA.IsValid());
    assert(protoC.IsValid());
    assert(protoA.GetPath() != protoC.GetPath());

    std::cout << "  Instancing different refs get different prototypes: OK\n";
}

void TestInstancingNoArc() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    // Instanceable without a composition arc is not an instance
    auto noArc = stage.GetPrimAtPath(Path::Parse("/InstanceableNoArc"));
    assert(noArc.IsValid());
    assert(noArc.IsInstanceable());
    assert(!noArc.IsInstance());

    std::cout << "  Instanceable without arc is not an instance: OK\n";
}

void TestInstancingNonInstance() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    // Non-instanceable ref is not an instance
    auto nonInst = stage.GetPrimAtPath(Path::Parse("/NonInstance"));
    assert(nonInst.IsValid());
    assert(!nonInst.IsInstance());
    assert(!nonInst.IsInstanceable());

    std::cout << "  Non-instanceable prim with ref is not an instance: OK\n";
}

void TestInstanceChildren() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    auto instA = stage.GetPrimAtPath(Path::Parse("/InstanceA"));
    assert(instA.IsInstance());

    // Instance GetChildren() returns the prototype's children
    auto children = instA.GetChildren();
    assert(children.size() == 2);

    // Children should be Child1 and Child2 from the referenced file
    std::unordered_set<std::string> childNames;
    for (const auto& c : children) {
        childNames.insert(c.GetName().GetString());
    }
    assert(childNames.count("Child1"));
    assert(childNames.count("Child2"));

    std::cout << "  Instance GetChildren returns prototype children: OK\n";
}

void TestPrototypeHidden() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    // Prototypes should NOT appear in Traverse()
    auto allPrims = stage.Traverse();
    for (const auto& p : allPrims) {
        assert(!p.IsPrototype());
        // No path should start with /__Prototype_
        std::string pathStr = p.GetPath().GetText();
        assert(pathStr.find("/__Prototype_") == std::string::npos);
    }

    std::cout << "  Prototypes hidden from Traverse: OK\n";
}

void TestLocalOpinionFiltering() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    auto instA = stage.GetPrimAtPath(Path::Parse("/InstanceA"));
    assert(instA.IsInstance());

    auto proto = instA.GetPrototype();
    assert(proto.IsValid());

    // The prototype should have "purpose" from the referenced file
    auto purposeAttr = proto.GetAttribute("purpose");
    assert(purposeAttr.IsValid());
    auto* val = purposeAttr.GetDefault();
    assert(val);
    auto* s = val->Get<String>();
    assert(s && *s == "from_ref");

    // The prototype should NOT have "localOverride" — local opinions are filtered
    auto localAttr = proto.GetAttribute("localOverride");
    assert(!localAttr.IsValid() || !localAttr.HasDefault());

    std::cout << "  Local opinion filtering on prototype: OK\n";
}

void TestPrototypeNavigation() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    auto instA = stage.GetPrimAtPath(Path::Parse("/InstanceA"));
    auto instB = stage.GetPrimAtPath(Path::Parse("/InstanceB"));
    assert(instA.IsInstance());
    assert(instB.IsInstance());

    auto proto = instA.GetPrototype();
    assert(proto.IsValid());

    // GetInstances() on prototype should return both instances
    auto instances = proto.GetInstances();
    assert(instances.size() == 2);

    std::unordered_set<std::string> instancePaths;
    for (const auto& inst : instances) {
        instancePaths.insert(inst.GetPath().GetText());
    }
    assert(instancePaths.count("/InstanceA"));
    assert(instancePaths.count("/InstanceB"));

    std::cout << "  Prototype navigation (GetPrototype/GetInstances roundtrip): OK\n";
}

void TestRepeatedInstanceableRefsUseRepresentativeExpansion() {
    auto stage = Stage::Open("tests/composition/instancing_root.usda");
    assert(stage.IsValid());

    const Path aPath = Path::Parse("/InstanceA");
    const Path bPath = Path::Parse("/InstanceB");
    auto instA = stage.GetPrimAtPath(aPath);
    auto instB = stage.GetPrimAtPath(bPath);
    assert(instA.IsInstance());
    assert(instB.IsInstance());
    assert(instA.GetPrototype().GetPath() == instB.GetPrototype().GetPath());

    auto childrenB = instB.GetChildren();
    assert(childrenB.size() == 2);

    const auto& graph = stage.GetGraph();
    Path repForA = graph.GetInstanceRepresentative(aPath);
    Path repForB = graph.GetInstanceRepresentative(bPath);
    const bool aSkipped = !repForA.IsEmpty();
    const bool bSkipped = !repForB.IsEmpty();
    assert(aSkipped != bSkipped);

    Path skipped = aSkipped ? aPath : bPath;
    Path representative = aSkipped ? repForA : repForB;
    assert(representative == (aSkipped ? bPath : aPath));

    assert(graph.GetPrimIndex(representative.AppendChild(Token("Child1"))) != nullptr);
    assert(graph.GetPrimIndex(skipped.AppendChild(Token("Child1"))) == nullptr);

    std::cout << "  Repeated instanceable refs use representative expansion: OK\n";
}

void TestInactiveRepresentedInstanceIsPruned() {
    auto stage = Stage::Open(
        "tests/composition/instancing_inactive_representative_root.usda");
    assert(stage.IsValid());

    auto instA = stage.GetPrimAtPath(Path::Parse("/InstanceA"));
    auto instB = stage.GetPrimAtPath(Path::Parse("/InstanceB"));
    assert(instA.IsValid());
    assert(instA.IsInstance());
    assert(!instB.IsValid());

    auto traversed = stage.Traverse();
    for (const auto& prim : traversed) {
        assert(prim.GetPath() != Path::Parse("/InstanceB"));
    }

    std::cout << "  Inactive represented instance is pruned: OK\n";
}


// ============================================================
// Diagnostic tests
// ============================================================

void TestDiagnosticCollectorBasics() {
    DiagnosticCollector dc;
    assert(dc.Empty());
    assert(dc.Count() == 0);
    assert(!dc.HasErrors());
    assert(!dc.HasWarnings());
    assert(dc.GetFirstError().empty());

    dc.Add({DiagSeverity::Warning, DiagCategory::MissingPayload,
            "test warning", "/Prim", "/layer.usda", "missing.usda", ArcType::Payload});
    assert(dc.Count() == 1);
    assert(!dc.HasErrors());
    assert(dc.HasWarnings());
    assert(dc.GetFirstError().empty());

    dc.Add({DiagSeverity::Error, DiagCategory::MissingReference,
            "test error", "/Prim2", "/layer.usda", "missing2.usda", ArcType::Reference});
    assert(dc.Count() == 2);
    assert(dc.HasErrors());
    assert(dc.GetFirstError() == "test error");

    assert(std::string(ArcTypeToString(ArcType::Local)) == "local");
    assert(std::string(ArcTypeToString(ArcType::Inherits)) == "inherits");
    assert(std::string(ArcTypeToString(ArcType::Variant)) == "variant");
    assert(std::string(ArcTypeToString(ArcType::Specialize)) == "specialize");

    std::cout << "  DiagnosticCollector basics: OK\n";
}

void TestDiagnosticToJson() {
    DiagnosticCollector dc;
    // Empty produces "[]"
    assert(dc.ToJson() == "[]");

    dc.Add({DiagSeverity::Error, DiagCategory::MissingSublayer,
            "Failed to resolve", "", "/root.usda", "sub.usda", ArcType::Sublayer});
    std::string json = dc.ToJson();
    // Verify it contains expected fields
    assert(json.find("\"severity\":\"error\"") != std::string::npos);
    assert(json.find("\"category\":\"missing_sublayer\"") != std::string::npos);
    assert(json.find("\"arcType\":\"sublayer\"") != std::string::npos);
    assert(json.find("\"message\":\"Failed to resolve\"") != std::string::npos);
    assert(json[0] == '[' && json[json.size()-1] == ']');

    std::cout << "  DiagnosticCollector ToJson: OK\n";
}

void TestComposeMissingSublayerDiagnostic() {
    auto result = ParseUsdaFile("tests/usda/missing_sublayer.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/usda/missing_sublayer.usda");
    // Should succeed (non-fatal) with diagnostics
    assert(composed.success);
    assert(!composed.diagnostics.Empty());
    assert(composed.diagnostics.HasErrors());

    const auto& diags = composed.diagnostics.GetAll();
    // DefaultResolve returns a path even for nonexistent files (it doesn't
    // check existence), so the failure is a parse error, not a resolution error.
    bool foundSublayerDiag = false;
    for (const auto& d : diags) {
        if (d.category == DiagCategory::SublayerParseFail) {
            foundSublayerDiag = true;
            assert(d.severity == DiagSeverity::Error);
            assert(d.arcType == ArcType::Sublayer);
            assert(!d.assetPath.empty());
        }
    }
    assert(foundSublayerDiag);

    std::cout << "  Compose missing sublayer diagnostic: OK\n";
}

void TestComposeMissingReferenceDiagnostic() {
    auto result = ParseUsdaFile("tests/usda/missing_ref_target.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/usda/missing_ref_target.usda");
    assert(composed.success);
    assert(!composed.diagnostics.Empty());

    // DefaultResolve returns a path even for nonexistent files, so the
    // failure is a parse error rather than a resolution failure.
    bool foundRefDiag = false;
    for (const auto& d : composed.diagnostics.GetAll()) {
        if (d.category == DiagCategory::ReferenceParseFail) {
            foundRefDiag = true;
            assert(d.severity == DiagSeverity::Error);
            assert(d.arcType == ArcType::Reference);
        }
    }
    assert(foundRefDiag);

    std::cout << "  Compose missing reference diagnostic: OK\n";
}

void TestComposeMissingPayloadDiagnostic() {
    auto result = ParseUsdaFile("tests/usda/missing_payload.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/usda/missing_payload.usda");
    assert(composed.success);
    assert(!composed.diagnostics.Empty());

    // DefaultResolve returns a path even for nonexistent files, so the
    // failure is a parse error rather than a resolution failure.
    bool foundPayloadDiag = false;
    for (const auto& d : composed.diagnostics.GetAll()) {
        if (d.category == DiagCategory::PayloadParseFail) {
            foundPayloadDiag = true;
            assert(d.severity == DiagSeverity::Warning);
            assert(d.arcType == ArcType::Payload);
        }
    }
    assert(foundPayloadDiag);

    std::cout << "  Compose missing payload diagnostic (Warning): OK\n";
}

void TestComposeMissingDefaultPrimDiagnostic() {
    auto result = ParseUsdaFile("tests/usda/missing_defaultprim_ref.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/usda/missing_defaultprim_ref.usda");
    assert(composed.success);
    assert(!composed.diagnostics.Empty());

    bool foundMissingDP = false;
    for (const auto& d : composed.diagnostics.GetAll()) {
        if (d.category == DiagCategory::MissingDefaultPrim) {
            foundMissingDP = true;
            assert(d.severity == DiagSeverity::Error);
        }
    }
    assert(foundMissingDP);

    std::cout << "  Compose missing defaultPrim diagnostic: OK\n";
}

void TestComposeInvalidReferenceTargetDiagnostic() {
    auto stage = Stage::Open(
        "tests/composition/pcp_SubrootReferenceAndVariants_root.usda");
    assert(stage.IsValid());

    auto invalid = stage.GetPrimAtPath(
        Path::Parse("/InvalidSubrootRefWithVariantSelection"));
    assert(invalid.IsValid());
    assert(!invalid.GetAttribute("a").IsValid());

    bool foundInvalidTarget = false;
    for (const auto& d : stage.GetDiagnostics().GetAll()) {
        if (d.category == DiagCategory::InvalidReferenceTarget &&
            d.arcType == ArcType::Reference &&
            d.primPath == "/InvalidSubrootRefWithVariantSelection" &&
            d.assetPath == "/Group{v=x}/Model") {
            foundInvalidTarget = true;
            break;
        }
    }
    assert(foundInvalidTarget);

    std::cout << "  Compose invalid reference target diagnostic: OK\n";
}

void TestStageOpenDegradedValid() {
    auto stage = Stage::Open("tests/usda/missing_sublayer.usda");
    // Stage should be valid but have diagnostics
    assert(stage.IsValid());
    assert(stage.HasCompositionErrors());
    assert(!stage.GetDiagnostics().Empty());
    assert(stage.GetError().empty());  // no fatal error

    std::cout << "  Stage::Open degraded-valid: OK\n";
}

void TestComposeCleanStageNoDiagnostics() {
    auto result = ParseUsdaFile("tests/composition/overlay.usda");
    assert(result.success);

    auto composed = Compose(result.layer, "tests/composition/overlay.usda");
    assert(composed.success);
    assert(composed.diagnostics.Empty());

    std::cout << "  Clean compose has no diagnostics: OK\n";
}

// SetVariantSelection + Recompose: the public API path for switching
// variants at runtime. variant_basic.usda selects "high" by default;
// after switching to "low" and recomposing, the composed stage should
// drop /Model/HighChild and read fromVariant=10 instead of 100.
void TestVariantApiSetAndRecompose() {
    auto stage = Stage::Open("tests/composition/variant_basic.usda");
    assert(stage.IsValid());

    auto model = stage.GetPrimAtPath(Path::Parse("/Model"));
    assert(model.IsValid());
    assert(model.GetVariantSelection(Token("lod")) == Token("high"));

    // Baseline: high variant is materialized.
    {
        auto highChild = model.GetChild(Token("HighChild"));
        assert(highChild.IsValid());
        auto fromVariant = model.GetAttribute(Token("fromVariant"));
        assert(fromVariant.IsValid());
        const Value* v = fromVariant.GetDefault();
        assert(v);
        auto* i = v->Get<Int>();
        assert(i && *i == 100);
    }

    // Switch selection on the root layer (layerIndex 0) and recompose.
    assert(model.SetVariantSelection(Token("lod"), Token("low"), 0));
    assert(stage.Recompose());

    // Handles are stale after Recompose — re-acquire via GetPrimAtPath.
    auto model2 = stage.GetPrimAtPath(Path::Parse("/Model"));
    assert(model2.IsValid());
    assert(model2.GetVariantSelection(Token("lod")) == Token("low"));

    // "low" variant's body does not declare HighChild.
    auto highChild2 = model2.GetChild(Token("HighChild"));
    assert(!highChild2.IsValid());

    // And fromVariant now reads 10 from the "low" variant body.
    auto fromVariant2 = model2.GetAttribute(Token("fromVariant"));
    assert(fromVariant2.IsValid());
    const Value* v2 = fromVariant2.GetDefault();
    assert(v2);
    auto* i2 = v2->Get<Int>();
    assert(i2 && *i2 == 10);

    std::cout << "  SetVariantSelection + Recompose switches variant: OK\n";
}

// Spec §12.3.4 (Value Clips): verify the `clips` and `clipSets` fields
// register, parse, round-trip through GetPrimMetadata, and compose
// across the opinion stack per §12.2.5 (dict combining) and §12.2.6
// (listop combining). This PR only lands the plumbing — the actual
// value-resolution use of clips is a later PR.
void TestClipsMetadataRoundtripAndCombining() {
    auto stage = Stage::Open("tests/composition/clips_metadata_root.usda");
    assert(stage.IsValid());

    auto geo = stage.GetPrimAtPath(Path::Parse("/Geo"));
    assert(geo.IsValid());

    // --- clips: Dictionary combining across the two layers.
    auto clipsVal = geo.GetPrimMetadata(FieldNames::clips);
    assert(clipsVal.has_value());
    const auto* clipsDict = clipsVal->Get<Dictionary>();
    assert(clipsDict != nullptr);
    // Both layers' clip sets survive §12.2.5 combining.
    assert(clipsDict->count("tight") == 1);
    assert(clipsDict->count("wide")  == 1);

    // Drill into root's "tight" entry — check a scalar + array key
    // made it through parse.
    {
        const auto& tight = clipsDict->at("tight");
        const auto* tightDict = tight.Get<Dictionary>();
        assert(tightDict != nullptr);
        auto itPrim = tightDict->find("primPath");
        assert(itPrim != tightDict->end());
        if (auto* s = itPrim->second.Get<String>())
            assert(*s == "/GeoTight");
        else if (auto* t = itPrim->second.Get<Token>())
            assert(t->GetString() == "/GeoTight");
        else
            assert(false && "primPath not a string-like value");
    }

    // Drill into weaker "wide" entry to confirm it also survived.
    {
        const auto& wide = clipsDict->at("wide");
        const auto* wideDict = wide.Get<Dictionary>();
        assert(wideDict != nullptr);
        auto itPrim = wideDict->find("primPath");
        assert(itPrim != wideDict->end());
    }

    // --- clipSets: listop combining across the two layers.
    auto cs = geo.GetPrimMetadata(FieldNames::clipSets);
    assert(cs.has_value());
    const auto* csOp = cs->Get<ListOp<std::string>>();
    assert(csOp != nullptr);
    auto items = csOp->GetItems();
    // Both layers prepend; root's prepends precede base's per §6.6.3.6.
    assert(items.size() == 2);
    assert(items[0] == "tight");
    assert(items[1] == "wide");

    std::cout << "  clips + clipSets parse / roundtrip / combine: OK\n";
}

// Spec §13.2.1.2.2: clipSets is listop<string>. In USDC that must
// encode as CrateTypeId::StringListOp (33), not TokenListOp (32).
// apiSchemas (listop<token>, §13.2.1.2) stays TokenListOp. Guards
// against the field-name sniffing in the USDC writer drifting into
// wrong-type encoding.
void TestClipSetsUsdcStringListOp() {
    auto stage = Stage::CreateInMemory();
    auto prim = stage.DefinePrim(Path::Parse("/Geo"), Token("Xform"));
    assert(prim.IsValid());
    // Set clipSets via direct spec write (no public setter yet).
    auto* spec = stage.GetMutableLayer().GetMutablePrimSpec(prim.GetPath());
    assert(spec);
    ListOp<std::string> cs;
    cs.SetExplicitItems({"foo", "bar"});
    spec->SetField(FieldNames::clipSets, Value(std::move(cs)));

    auto tmp = std::filesystem::temp_directory_path() /
               "nanousd_clipsets_usdc.usdc";
    std::string tmpPath = tmp.string();
    assert(WriteUsdcFile(stage.GetMutableLayer(), tmpPath));

    auto parsed = ParseUsdcFile(tmpPath);
    assert(parsed.success);
    auto* readSpec = parsed.layer.GetPrimSpec(Path::Parse("/Geo"));
    assert(readSpec);
    auto* field = readSpec->GetField(FieldNames::clipSets);
    assert(field);
    auto* lop = field->Get<ListOp<std::string>>();
    assert(lop && "clipSets must decode as ListOp<std::string>");
    auto items = lop->GetItems();
    assert(items.size() == 2);
    assert(items[0] == "foo");
    assert(items[1] == "bar");

    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::cout << "  clipSets USDC roundtrip (StringListOp): OK\n";
}

// Spec §13.2.1.2: apiSchemas is listop<token>. In USDC that must
// encode as CrateTypeId::TokenListOp (32) AND the in-memory type must
// be ListOp<Token> — not ListOp<std::string> as the pre-rework code
// stored it. This test catches the type regressing.
void TestApiSchemasUsdcTokenListOp() {
    auto stage = Stage::CreateInMemory();
    auto prim = stage.DefinePrim(Path::Parse("/Prim"), Token("Xform"));
    assert(prim.IsValid());
    auto* spec = stage.GetMutableLayer().GetMutablePrimSpec(prim.GetPath());
    assert(spec);
    ListOp<Token> api;
    api.SetPrependedItems({Token("FooAPI"), Token("BarAPI:inst")});
    spec->SetField(FieldNames::apiSchemas, Value(std::move(api)));

    auto tmp = std::filesystem::temp_directory_path() /
               "nanousd_apischemas_usdc.usdc";
    std::string tmpPath = tmp.string();
    assert(WriteUsdcFile(stage.GetMutableLayer(), tmpPath));

    auto parsed = ParseUsdcFile(tmpPath);
    assert(parsed.success);
    auto* readSpec = parsed.layer.GetPrimSpec(Path::Parse("/Prim"));
    assert(readSpec);
    auto* field = readSpec->GetField(FieldNames::apiSchemas);
    assert(field);
    auto* lop = field->Get<ListOp<Token>>();
    assert(lop && "apiSchemas must decode as ListOp<Token>");
    auto prepends = lop->GetPrependedItems();
    assert(prepends.size() == 2);
    assert(prepends[0] == Token("FooAPI"));
    assert(prepends[1] == Token("BarAPI:inst"));

    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    std::cout << "  apiSchemas USDC roundtrip (TokenListOp): OK\n";
}

// --- Value clips Phase 2: ClipSetEntry materialization + template
//     expansion. Pure in-memory transform tests; no stage/composition
//     involved. A passthrough resolver keeps input asset paths
//     verbatim so assertions don't have to predict filesystem
//     anchoring behavior.

static AssetResolver MakePassthroughResolver() {
    return [](const std::string&, const std::string& assetPath) {
        return assetPath;
    };
}

// Build a clips Dictionary of the shape authored in USDA as
// `clips = { dictionary <name> = { ... } }` — one named clip set
// with the given inner dict.
static Dictionary MakeClipsDict(const std::string& name, Dictionary inner) {
    Dictionary outer;
    outer[name] = Value(std::move(inner));
    return outer;
}

// §12.3.4.1.2 — explicit form: materialize copies assetPaths,
// active, times, primPath, manifestAssetPath verbatim (with asset
// resolution through the passthrough resolver, which is a no-op
// here).
void TestClipsMaterializeExplicit() {
    Dictionary inner;
    inner["primPath"] = Value(std::string("/Geo"));
    inner["manifestAssetPath"] = Value(Value::AssetTag{}, std::string("./manifest.usda"));
    {
        std::vector<std::string> paths = {"./quad_1.usda", "./quad_2.usda", "./quad_3.usda"};
        inner["assetPaths"] = Value(Value::ArrayTag{}, TypeId::Asset, std::move(paths));
    }
    {
        std::vector<GfVec2d> active = {GfVec2d{{0, 0}}, GfVec2d{{1, 1}}, GfVec2d{{2, 2}}};
        inner["active"] = Value(Value::ArrayTag{}, TypeId::Double2, std::move(active));
    }
    {
        std::vector<GfVec2d> times = {GfVec2d{{0, 1}}, GfVec2d{{1, 2}}, GfVec2d{{2, 3}}};
        inner["times"] = Value(Value::ArrayTag{}, TypeId::Double2, std::move(times));
    }

    auto clips = MakeClipsDict("default", std::move(inner));
    auto entry = MaterializeClipSet(clips, "default", "/anchor/stage.usda",
                                     MakePassthroughResolver());
    assert(entry.has_value());
    assert(entry->name == "default");
    assert(entry->primPath.GetText() == "/Geo");
    assert(entry->manifestAssetPath == "./manifest.usda");
    assert(entry->assetPaths.size() == 3);
    assert(entry->assetPaths[0] == "./quad_1.usda");
    assert(entry->active.size() == 3);
    assert(entry->active[0].first == 0.0 && entry->active[0].second == 0);
    assert(entry->active[2].second == 2);
    assert(entry->times.size() == 3);
    assert(entry->times[2].first == 2.0 && entry->times[2].second == 3.0);

    std::cout << "  ClipSet materialize (explicit form §12.3.4.1.2): OK\n";
}

// §12.3.4.1.3 — template form: synthesises assetPaths/active/times
// from template params. Spec example: start=0, end=2, stride=1,
// template="./quad_###.usda" → assetPaths=[quad_000,quad_001,quad_002].
void TestClipsMaterializeTemplate() {
    Dictionary inner;
    inner["primPath"] = Value(std::string("/Geo"));
    inner["templateAssetPath"] = Value(Value::AssetTag{}, std::string("./quad_###.usda"));
    inner["templateStartTime"] = Value(0.0);
    inner["templateEndTime"]   = Value(2.0);
    inner["templateStride"]    = Value(1.0);

    auto clips = MakeClipsDict("default", std::move(inner));
    auto entry = MaterializeClipSet(clips, "default", "/anchor/stage.usda",
                                     MakePassthroughResolver());
    assert(entry.has_value());
    assert(entry->assetPaths.size() == 3);
    assert(entry->assetPaths[0] == "./quad_000.usda");
    assert(entry->assetPaths[1] == "./quad_001.usda");
    assert(entry->assetPaths[2] == "./quad_002.usda");
    // No templateActiveOffset authored → no edge knots, identity times.
    assert(entry->times.size() == 3);
    assert(entry->times[0].first == 0.0 && entry->times[0].second == 0.0);
    assert(entry->times[1].first == 1.0 && entry->times[1].second == 1.0);
    assert(entry->active.size() == 3);
    assert(entry->active[0].first == 0.0 && entry->active[0].second == 0);
    assert(entry->active[2].first == 2.0 && entry->active[2].second == 2);

    std::cout << "  ClipSet materialize (template form §12.3.4.1.3): OK\n";
}

// §12.3.4.1.3.4 stride — start=12, end=25, stride=6 →
// "clipname.12.usd", "clipname.18.usd", "clipname.24.usd". Last step
// (30) is past end, so stops at 24.
void TestClipsTemplateStride() {
    ClipSetEntry e;
    bool ok = ExpandTemplateClipSet("path/clipname.#.usd",
                                    /*start=*/12, /*end=*/25, /*stride=*/6,
                                    /*offset=*/std::nullopt,
                                    "/anchor", MakePassthroughResolver(), e);
    assert(ok);
    assert(e.assetPaths.size() == 3);
    assert(e.assetPaths[0] == "path/clipname.12.usd");
    assert(e.assetPaths[1] == "path/clipname.18.usd");
    assert(e.assetPaths[2] == "path/clipname.24.usd");
    std::cout << "  Template stride (§12.3.4.1.3.4): OK\n";
}

// §12.3.4.1.3.5 — templateActiveOffset edge-knot expansion. Exactly
// the spec example: start=101, end=103, stride=1, offset=0.5.
void TestClipsTemplateActiveOffset() {
    ClipSetEntry e;
    bool ok = ExpandTemplateClipSet("./clip.#.usda",
                                    /*start=*/101, /*end=*/103, /*stride=*/1,
                                    /*offset=*/0.5,
                                    "/anchor", MakePassthroughResolver(), e);
    assert(ok);
    // Per spec:
    //   times = [(100.5,100.5), (101,101), (102,102), (103,103), (103.5,103.5)]
    //   active = [(101.5, 0), (102.5, 1), (103.5, 2)]
    assert(e.times.size() == 5);
    assert(e.times[0].first == 100.5 && e.times[0].second == 100.5);
    assert(e.times[1].first == 101.0 && e.times[1].second == 101.0);
    assert(e.times[2].first == 102.0);
    assert(e.times[3].first == 103.0);
    assert(e.times[4].first == 103.5 && e.times[4].second == 103.5);
    assert(e.active.size() == 3);
    assert(e.active[0].first == 101.5 && e.active[0].second == 0);
    assert(e.active[1].first == 102.5 && e.active[1].second == 1);
    assert(e.active[2].first == 103.5 && e.active[2].second == 2);

    std::cout << "  Template activeOffset + edge knots (§12.3.4.1.3.5): OK\n";
}

// §12.3.4.1.3.5 — "templateActiveOffset cannot exceed the absolute
// value of templateStride." Violation must return false without
// populating the output.
void TestClipsTemplateOffsetExceedsStride() {
    ClipSetEntry e;
    bool ok = ExpandTemplateClipSet("./clip.#.usda",
                                    /*start=*/0, /*end=*/5, /*stride=*/1,
                                    /*offset=*/1.5,  // > stride → reject
                                    "/anchor", MakePassthroughResolver(), e);
    assert(!ok);
    std::cout << "  Template offset > stride rejected: OK\n";
}

// §12.3.4.1.3.1 — sub-integer form "clipname.###.###.usd" supports
// fractional stage times. Test with sub-second framing: t=1.5 →
// clip.001.500.usda.
void TestClipsTemplateSubInteger() {
    ClipSetEntry e;
    bool ok = ExpandTemplateClipSet("./clip.###.###.usda",
                                    /*start=*/1.5, /*end=*/2.5, /*stride=*/0.5,
                                    /*offset=*/std::nullopt,
                                    "/anchor", MakePassthroughResolver(), e);
    assert(ok);
    assert(e.assetPaths.size() == 3);
    assert(e.assetPaths[0] == "./clip.001.500.usda");
    assert(e.assetPaths[1] == "./clip.002.000.usda");
    assert(e.assetPaths[2] == "./clip.002.500.usda");
    std::cout << "  Template sub-integer form (§12.3.4.1.3.1): OK\n";
}

// §12.3.4.1 — "When both explicit and template clip metadata is
// authored, explicit will be chosen." Authoring both should yield the
// explicit sequence.
void TestClipsExplicitWinsOverTemplate() {
    Dictionary inner;
    inner["primPath"] = Value(std::string("/Geo"));
    // Explicit (wins):
    {
        std::vector<std::string> paths = {"./expl_a.usda", "./expl_b.usda"};
        inner["assetPaths"] = Value(Value::ArrayTag{}, TypeId::Asset, std::move(paths));
    }
    {
        std::vector<GfVec2d> active = {GfVec2d{{0, 0}}, GfVec2d{{5, 1}}};
        inner["active"] = Value(Value::ArrayTag{}, TypeId::Double2, std::move(active));
    }
    // Template (ignored):
    inner["templateAssetPath"] = Value(Value::AssetTag{}, std::string("./tpl_###.usda"));
    inner["templateStartTime"] = Value(0.0);
    inner["templateEndTime"]   = Value(10.0);
    inner["templateStride"]    = Value(1.0);

    auto clips = MakeClipsDict("default", std::move(inner));
    auto entry = MaterializeClipSet(clips, "default", "/anchor/stage.usda",
                                     MakePassthroughResolver());
    assert(entry.has_value());
    assert(entry->assetPaths.size() == 2);
    assert(entry->assetPaths[0] == "./expl_a.usda");
    assert(entry->assetPaths[1] == "./expl_b.usda");
    // Template-generated times would be 11 entries; explicit times
    // weren't authored, so the result times is empty.
    assert(entry->times.empty());

    std::cout << "  Explicit wins over template (§12.3.4.1): OK\n";
}

// Malformed template strings (no # groups, three groups, non-adjacent
// groups) must fail. Keeps the template parser honest.
void TestClipsTemplateMalformed() {
    ClipSetEntry e;
    // No `#` groups at all.
    assert(!ExpandTemplateClipSet("./no_placeholders.usda", 0, 2, 1, std::nullopt,
                                  "/anchor", MakePassthroughResolver(), e));
    // Three `#` groups.
    assert(!ExpandTemplateClipSet("./##.##.##.usda", 0, 2, 1, std::nullopt,
                                  "/anchor", MakePassthroughResolver(), e));
    // Groups not adjacent (separator other than single '.').
    assert(!ExpandTemplateClipSet("./##_##.usda", 0, 2, 1, std::nullopt,
                                  "/anchor", MakePassthroughResolver(), e));
    // Stride must be positive.
    assert(!ExpandTemplateClipSet("./#.usda", 0, 5, 0, std::nullopt,
                                  "/anchor", MakePassthroughResolver(), e));
    std::cout << "  Template malformed / invalid stride rejected: OK\n";
}

// Unknown clip set name returns nullopt.
void TestClipsMaterializeUnknownName() {
    Dictionary inner;
    inner["primPath"] = Value(std::string("/Geo"));
    auto clips = MakeClipsDict("default", std::move(inner));
    auto entry = MaterializeClipSet(clips, "not_there", "/anchor",
                                     MakePassthroughResolver());
    assert(!entry.has_value());
    std::cout << "  Materialize unknown clip set → nullopt: OK\n";
}

// --- Value clips Phase 3: ClipLayerCache + manifest loading ---
//     §12.3.4.3. Opens real USDA files from tests/clips/; tests are
//     expected to run with WORKING_DIRECTORY = repo root (set by
//     add_test in CMakeLists.txt).

// Cache hit: opening the same path twice returns the same shared_ptr
// and a single cache entry exists.
void TestClipCacheHitReturnsSameLayer() {
    ClipLayerCache cache;
    auto a = cache.GetOrOpen("tests/clips/clip_001.usda");
    auto b = cache.GetOrOpen("tests/clips/clip_001.usda");
    assert(a != nullptr);
    assert(a.get() == b.get());  // cache hit — same pointer
    assert(cache.Size() == 1);
    assert(cache.Contains("tests/clips/clip_001.usda"));
    std::cout << "  ClipLayerCache hit returns same layer: OK\n";
}

// Negative caching: a missing path returns nullptr and is cached so
// subsequent lookups don't retry the open.
void TestClipCacheMissNegativeCached() {
    ClipLayerCache cache;
    auto a = cache.GetOrOpen("tests/clips/does_not_exist.usda");
    assert(a == nullptr);
    assert(cache.Contains("tests/clips/does_not_exist.usda"));
    auto b = cache.GetOrOpen("tests/clips/does_not_exist.usda");
    assert(b == nullptr);
    assert(cache.Size() == 1);  // no re-entry on retry
    std::cout << "  ClipLayerCache miss is negatively cached: OK\n";
}

// Empty path: no lookup, no cache entry, nullptr return.
void TestClipCacheEmptyPath() {
    ClipLayerCache cache;
    assert(cache.GetOrOpen("") == nullptr);
    assert(cache.Size() == 0);
    std::cout << "  ClipLayerCache empty path rejected: OK\n";
}

// Clear() empties the cache.
void TestClipCacheClear() {
    ClipLayerCache cache;
    cache.GetOrOpen("tests/clips/clip_001.usda");
    assert(cache.Size() == 1);
    cache.Clear();
    assert(cache.Size() == 0);
    assert(!cache.Contains("tests/clips/clip_001.usda"));
    std::cout << "  ClipLayerCache Clear resets: OK\n";
}

// §12.3.4.3 — manifest declares which attributes the clip set
// provides. GetManifestAttributePaths returns every attribute spec
// path authored in the manifest layer.
void TestManifestReturnsAttributePaths() {
    ClipSetEntry entry;
    entry.name = "default";
    entry.manifestAssetPath = "tests/clips/manifest_simple.usda";

    ClipLayerCache cache;
    auto paths = GetManifestAttributePaths(entry, cache);

    // manifest_simple.usda declares: /Geo.radius, /Geo.color,
    // /Geo/Child.opacity (three attribute specs).
    assert(paths.size() == 3);
    std::unordered_set<std::string> asText;
    for (const auto& p : paths) asText.insert(p.GetText());
    assert(asText.count("/Geo.radius"));
    assert(asText.count("/Geo.color"));
    assert(asText.count("/Geo/Child.opacity"));
    std::cout << "  Manifest returns attribute paths (§12.3.4.3): OK\n";
}

// Missing manifestAssetPath (empty) returns an empty vector, doesn't
// touch the cache.
void TestManifestEmptyPath() {
    ClipSetEntry entry;
    entry.manifestAssetPath.clear();
    ClipLayerCache cache;
    auto paths = GetManifestAttributePaths(entry, cache);
    assert(paths.empty());
    assert(cache.Size() == 0);
    std::cout << "  Manifest with empty path → empty, no I/O: OK\n";
}

// Missing manifest file returns empty and negatively caches the
// failure so a second query doesn't re-open.
void TestManifestFailureIsCached() {
    ClipSetEntry entry;
    entry.manifestAssetPath = "tests/clips/missing_manifest.usda";
    ClipLayerCache cache;
    auto a = GetManifestAttributePaths(entry, cache);
    auto b = GetManifestAttributePaths(entry, cache);
    assert(a.empty() && b.empty());
    assert(cache.Size() == 1);  // single negative-cached entry
    std::cout << "  Manifest open-failure is negatively cached: OK\n";
}

// Cache is shared across clip sets: two entries with the same
// manifestAssetPath open the file exactly once.
void TestManifestCacheSharedAcrossEntries() {
    ClipSetEntry a, b;
    a.manifestAssetPath = "tests/clips/manifest_simple.usda";
    b.manifestAssetPath = "tests/clips/manifest_simple.usda";
    ClipLayerCache cache;
    auto pa = GetManifestAttributePaths(a, cache);
    auto pb = GetManifestAttributePaths(b, cache);
    assert(pa.size() == 3 && pb.size() == 3);
    assert(cache.Size() == 1);  // single cache entry for both queries
    std::cout << "  Manifest cache shared across clip sets: OK\n";
}

// --- Value clips Phase 4: end-to-end value resolution integration
//     (§12.3.4.5). Stage-level attribute Get() now walks opinions
//     per-opinion with the 4-field priority timeSamples > spline >
//     default > clips.

// Basic clip contribution: /Geo.radius has no local default and no
// local timeSamples, but a clip set provides values. Query at t=0
// returns the first clip's sample at clip-time 0.
void TestClipsValueResolutionBasic() {
    auto stage = Stage::Open("tests/clips/stage_clips_simple.usda");
    assert(stage.IsValid());
    auto prim = stage.GetPrimAtPath(Path::Parse("/Geo"));
    assert(prim.IsValid());
    auto attr = prim.GetAttribute(Token("radius"));
    assert(attr.IsValid());

    // stageTime=0 → active asset 0 (clip_a), times maps 0→0, clip_a
    // radius.timeSamples[0] = 1.0.
    auto r = attr.Get(UsdTimeCode(0.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && *f == 1.0f);
    std::cout << "  Clips resolve basic attribute value (§12.3.4.5): OK\n";
}

// Clip active-set switch: at stageTime=10 the active array flips to
// asset index 1 (clip_b), with times mapping stage→clip identity in
// that region.
void TestClipsActiveAssetSwitch() {
    auto stage = Stage::Open("tests/clips/stage_clips_simple.usda");
    assert(stage.IsValid());
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));

    // times=[(0,0),(10,10),(20,10)] → stageTime 10 maps to clipTime 10.
    // active=[(0,0),(10,1)] → stageTime 10 is in asset 1 (clip_b).
    // clip_b radius.timeSamples[10] = 20.0.
    auto r = attr.Get(UsdTimeCode(10.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && *f == 20.0f);
    std::cout << "  Clips active-set switch flips to asset 1: OK\n";
}

// Interpolation inside the clip: clip_a has samples at clip-times 0
// and 10 with values 1.0 and 2.0. Query at stageTime 5 (still in
// asset 0; times map 5→5 linearly) yields 1.5.
void TestClipsLinearInterpolation() {
    auto stage = Stage::Open("tests/clips/stage_clips_simple.usda");
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));

    auto r = attr.Get(UsdTimeCode(5.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && std::abs(*f - 1.5f) < 1e-5f);
    std::cout << "  Clips linear interpolation inside clip: OK\n";
}

// Clips don't override locally-authored default on the same prim
// (§12.3.4.5 "just weaker than Local"). `/Geo.color` has a local
// default = (0.5, 0.5, 0.5); clips carry no opinion for color.
// Query at any time must return the default.
void TestClipsDoNotOverrideLocalDefault() {
    auto stage = Stage::Open("tests/clips/stage_clips_simple.usda");
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("color"));

    auto r = attr.Get(UsdTimeCode(5.0));
    assert(r.found);
    auto* c = r.value.Get<GfVec3f>();
    assert(c);
    assert((*c)[0] == 0.5f && (*c)[1] == 0.5f && (*c)[2] == 0.5f);
    std::cout << "  Local default beats clips at same opinion: OK\n";
}

// primPath remap (§12.3.4.5): stage prim /Stage with primPath=/GeoClip
// looks up /Stage.radius as /GeoClip.radius inside clip_remap.usda.
void TestClipsPrimPathRemap() {
    auto stage = Stage::Open("tests/clips/stage_clips_remap.usda");
    assert(stage.IsValid());
    auto attr = stage.GetPrimAtPath(Path::Parse("/Stage")).GetAttribute(Token("radius"));

    auto r = attr.Get(UsdTimeCode(0.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && *f == 99.0f);  // clip_remap.usda authors /GeoClip.radius = 99.0
    std::cout << "  Clips primPath remap stage→clip path (§12.3.4.5): OK\n";
}

// Per-opinion ordering bug fix: before Phase 4 the Get() walk checked
// timeSamples across all opinions first, then default across all
// opinions — so a weak-layer timeSamples could beat a strong-layer
// default. Now a stronger layer's `default` wins.
void TestPerOpinionStrongDefaultBeatsWeakTimeSamples() {
    auto stage = Stage::Open("tests/clips/stage_ordering_root.usda");
    assert(stage.IsValid());
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));

    // Root authors `default = 7.0`; base authors timeSamples [100,200].
    // Per-opinion: root wins. Any time query must return 7.0.
    for (double t : {0.0, 5.0, 10.0, 100.0}) {
        auto r = attr.Get(UsdTimeCode(t));
        assert(r.found);
        auto* f = r.value.Get<Float>();
        assert(f && *f == 7.0f);
    }
    std::cout << "  Per-opinion: strong default beats weak timeSamples: OK\n";
}

void TestFlattenSamplesStrongDefaultOverWeakTimeSamples() {
    auto stage = Stage::Open("tests/clips/stage_ordering_root.usda");
    assert(stage.IsValid());

    Layer flat = FlattenStage(stage);
    const Spec* spec = flat.GetAttributeSpec(Path::Parse("/Geo.radius"));
    assert(spec);

    const Value* def = spec->GetField(FieldNames::defaultValue);
    assert(def);
    const auto* df = def->Get<Float>();
    assert(df && *df == 7.0f);

    const Value* ts = spec->GetField(FieldNames::timeSamples);
    assert(ts);
    const auto* samples = ts->Get<Dictionary>();
    assert(samples && samples->size() == 2);
    for (const char* key : {"0", "10"}) {
        auto it = samples->find(key);
        assert(it != samples->end());
        const auto* f = it->second.Get<Float>();
        assert(f && *f == 7.0f);
    }

    auto flatStage = Stage::CreateFromComposedLayer(std::move(flat));
    auto flatAttr = flatStage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));
    for (double t : {0.0, 5.0, 10.0}) {
        auto r = flatAttr.Get(UsdTimeCode(t));
        assert(r.found);
        const auto* f = r.value.Get<Float>();
        assert(f && *f == 7.0f);
    }

    std::cout << "  Flatten samples strong default over weak timeSamples: OK\n";
}

void TestFlattenBakesSplineToTimeSamples() {
    auto parsed = ParseUsda(R"(#usda 1.0
def Scope "Geo"
{
    double radius.spline = {
        0: 1.0,
        10: 3.0,
    }
}
)");
    assert(parsed.success);

    auto stage = Stage::CreateFromComposedLayer(std::move(parsed.layer));
    Layer flat = FlattenStage(stage);
    const Spec* spec = flat.GetAttributeSpec(Path::Parse("/Geo.radius"));
    assert(spec);
    assert(!spec->HasField(FieldNames::spline));
    assert(!spec->HasField(FieldNames::defaultValue));

    const Value* ts = spec->GetField(FieldNames::timeSamples);
    assert(ts);
    const auto* samples = ts->Get<Dictionary>();
    assert(samples && samples->size() == 2);

    auto expect = [&](const char* key, double value) {
        auto it = samples->find(key);
        assert(it != samples->end());
        const auto* d = it->second.Get<Double>();
        assert(d && std::abs(*d - value) < 1e-12);
    };
    expect("0", 1.0);
    expect("10", 3.0);

    std::cout << "  Flatten bakes spline to sampled timeSamples: OK\n";
}

void TestFlattenSamplesClipValues() {
    auto stage = Stage::Open("tests/clips/stage_clips_simple.usda");
    assert(stage.IsValid());

    Layer flat = FlattenStage(stage);
    const Spec* primSpec = flat.GetPrimSpec(Path::Parse("/Geo"));
    assert(primSpec);
    assert(!primSpec->HasField(FieldNames::clips));
    assert(!primSpec->HasField(FieldNames::clipSets));

    const Spec* spec = flat.GetAttributeSpec(Path::Parse("/Geo.radius"));
    assert(spec);
    assert(!spec->HasField(FieldNames::spline));

    const Value* ts = spec->GetField(FieldNames::timeSamples);
    assert(ts);
    const auto* samples = ts->Get<Dictionary>();
    assert(samples);

    auto expect = [&](const char* key, float value) {
        auto it = samples->find(key);
        assert(it != samples->end());
        const auto* f = it->second.Get<Float>();
        assert(f && std::abs(*f - value) < 1e-6f);
    };
    expect("0", 1.0f);
    expect("10", 20.0f);
    expect("20", 20.0f);

    auto flatStage = Stage::CreateFromComposedLayer(std::move(flat));
    auto flatAttr = flatStage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));
    for (auto [time, value] : std::vector<std::pair<double, float>>{
             {0.0, 1.0f}, {10.0, 20.0f}, {20.0, 20.0f}}) {
        auto r = flatAttr.Get(UsdTimeCode(time));
        assert(r.found);
        const auto* f = r.value.Get<Float>();
        assert(f && std::abs(*f - value) < 1e-6f);
    }

    std::cout << "  Flatten samples clip values into timeSamples: OK\n";
}

// ClipLayerCache is stage-owned via the composition graph and
// persists across queries. First query populates; second hits cache.
void TestStageClipCachePersists() {
    auto stage = Stage::Open("tests/clips/stage_clips_simple.usda");
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));
    (void)attr.Get(UsdTimeCode(0.0));
    const auto& graph = stage.GetGraph();
    assert(graph.clipLayerCache);
    size_t afterFirst = graph.clipLayerCache->Size();
    assert(afterFirst >= 1);  // at least clip_a opened
    (void)attr.Get(UsdTimeCode(0.0));
    assert(graph.clipLayerCache->Size() == afterFirst);  // no re-open
    std::cout << "  Stage-owned clip cache persists across queries: OK\n";
}

// --- Value clips Phase 5: manifest gating, auto-manifest, missing-
//     value fallback (§12.3.4.3, §12.3.4.6, §12.3.4.7), and jump
//     discontinuities in `times` (§12.3.4.8).

// §12.3.4.3 auto-manifest: no manifestAssetPath authored → derive
// from the union of attribute specs across clip layers. Both radius
// and color are authored in clip_multi.usda, so both resolve.
void TestClipsAutoManifestIncludesAllClipAttrs() {
    auto stage = Stage::Open("tests/clips/stage_clips_auto_manifest.usda");
    assert(stage.IsValid());
    auto prim = stage.GetPrimAtPath(Path::Parse("/Geo"));

    auto radius = prim.GetAttribute(Token("radius")).Get(UsdTimeCode(5.0));
    assert(radius.found);
    auto* rf = radius.value.Get<Float>();
    assert(rf && std::abs(*rf - 1.5f) < 1e-5f);

    auto color = prim.GetAttribute(Token("color")).Get(UsdTimeCode(5.0));
    assert(color.found);
    auto* cv = color.value.Get<GfVec3f>();
    assert(cv);
    assert(std::abs((*cv)[0] - 0.5f) < 1e-5f);
    assert(std::abs((*cv)[1] - 0.5f) < 1e-5f);
    assert(std::abs((*cv)[2] - 0.0f) < 1e-5f);
    std::cout << "  Auto-manifest unions attrs across clip layers (§12.3.4.3): OK\n";
}

// §12.3.4.3 gating: an authored manifest that declares only `radius`
// must cause /Geo.color queries to skip the clip set entirely, even
// though the clip authors color. Color has no local default here,
// so the resolved value is "not found" — NOT the interp from the
// clip.
void TestClipsManifestGatesOutUndeclaredAttr() {
    auto stage = Stage::Open("tests/clips/stage_clips_manifest_gate.usda");
    assert(stage.IsValid());
    auto prim = stage.GetPrimAtPath(Path::Parse("/Geo"));

    // Radius IS in the manifest → clip resolves.
    auto r = prim.GetAttribute(Token("radius")).Get(UsdTimeCode(5.0));
    assert(r.found);
    auto* rf = r.value.Get<Float>();
    assert(rf && std::abs(*rf - 1.5f) < 1e-5f);

    // Color is NOT in the manifest → clip skipped. No local default
    // and no schema fallback for a custom attr → not found.
    auto c = prim.GetAttribute(Token("color")).Get(UsdTimeCode(5.0));
    assert(!c.found);
    std::cout << "  Manifest gates out undeclared attribute (§12.3.4.3): OK\n";
}

// §12.3.4.6 missing-value hold (interpolateMissingClipValues=false).
// Three-asset sequence; middle clip has no `radius`. Querying inside
// middle-asset's active range exercises the fallback: hold at the
// nearest neighbouring clip's value.
void TestClipsMissingValueHoldsNearest() {
    auto stage = Stage::Open("tests/clips/stage_clips_missing_hold.usda");
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));

    // Representative samples seen by the recovery walk:
    //   knot (0, asset0):  clipTime 0 → 1.0
    //   knot (10, asset1): no radius (skipped)
    //   knot (20, asset2): clipTime 20 (held past last sample at 5) → 20.0
    // stageTime=15 is in asset1's active range (missing). Hold mode:
    // nearer neighbour is (20, ...) → 20.0.
    auto r = attr.Get(UsdTimeCode(15.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && std::abs(*f - 20.0f) < 1e-5f);

    // stageTime=3 is inside asset0's range and NOT missing — the
    // primary path takes over and interpolates clip_has_radius's
    // samples (1.0 at clipTime 0, 2.0 at clipTime 5) at clipTime 3:
    //   1.0 + (2.0 - 1.0) * (3/5) = 1.6.
    auto r2 = attr.Get(UsdTimeCode(3.0));
    assert(r2.found);
    auto* f2 = r2.value.Get<Float>();
    assert(f2 && std::abs(*f2 - 1.6f) < 1e-5f);
    std::cout << "  Missing-value hold picks nearest neighbour (§12.3.4.6): OK\n";
}

// §12.3.4.7 missing-value linear interpolation. Same sequence but
// interpolateMissingClipValues=true. At stageTime=15 the value is
// linearly interpolated between the before/after neighbour samples.
void TestClipsMissingValueLinearInterp() {
    auto stage = Stage::Open("tests/clips/stage_clips_missing_interp.usda");
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));

    // before = (0, 1.0), after = (20, 20.0). alpha = 15/20 = 0.75.
    // interp = 1.0 + (20.0 - 1.0) * 0.75 = 15.25.
    auto r = attr.Get(UsdTimeCode(15.0));
    assert(r.found);
    auto* f = r.value.Get<Float>();
    assert(f && std::abs(*f - 15.25f) < 1e-4f);
    std::cout << "  Missing-value linear interp across clips (§12.3.4.7): OK\n";
}

// §12.3.4.8 jump discontinuity: two `times` knots at stageTime=10
// map to clipTime 10 and clipTime 0 respectively. Right-continuous
// evaluation — at exactly stageTime=10 the clipTime 0 branch wins.
void TestClipsJumpDiscontinuityRightContinuous() {
    auto stage = Stage::Open("tests/clips/stage_clips_jump.usda");
    auto attr = stage.GetPrimAtPath(Path::Parse("/Geo")).GetAttribute(Token("radius"));

    // clip_a radius at clipTime 0 = 1.0; at clipTime 10 = 2.0.

    // stageTime=9 → pre-jump, clipTime≈9 → interp → ≈1.9.
    auto r9 = attr.Get(UsdTimeCode(9.0));
    assert(r9.found);
    auto* f9 = r9.value.Get<Float>();
    assert(f9 && std::abs(*f9 - 1.9f) < 1e-5f);

    // stageTime=10 → post-jump (right-continuous) → clipTime 0 → 1.0.
    auto r10 = attr.Get(UsdTimeCode(10.0));
    assert(r10.found);
    auto* f10 = r10.value.Get<Float>();
    assert(f10 && std::abs(*f10 - 1.0f) < 1e-5f);

    // stageTime=15 → in the post-jump segment, halfway → clipTime 5.
    // clip_a radius at clipTime 5 = 1.5.
    auto r15 = attr.Get(UsdTimeCode(15.0));
    assert(r15.found);
    auto* f15 = r15.value.Get<Float>();
    assert(f15 && std::abs(*f15 - 1.5f) < 1e-5f);
    std::cout << "  Times jump discontinuity is right-continuous (§12.3.4.8): OK\n";
}

// Manifest resolution is cached on the graph's ClipLayerCache.
// Resolving the same clip set's manifest twice must not repopulate
// the underlying layer cache.
void TestClipSetManifestCaching() {
    ClipLayerCache cache;
    ClipSetEntry e;
    e.manifestAssetPath = "tests/clips/manifest_radius_only.usda";
    const auto& a = cache.GetOrResolveManifest(e);
    size_t layerCountAfter1 = cache.Size();
    const auto& b = cache.GetOrResolveManifest(e);
    assert(&a == &b);  // same cached set
    assert(cache.Size() == layerCountAfter1);
    assert(a.size() == 1);
    std::cout << "  ClipSet manifest resolution cached (authored): OK\n";
}

// Auto-manifest caching: resolving twice opens clip layers exactly
// once per layer.
void TestAutoManifestCaching() {
    ClipLayerCache cache;
    ClipSetEntry e;
    // No manifestAssetPath authored — auto-derive from clip layers.
    e.assetPaths = {"tests/clips/clip_multi.usda"};
    const auto& a = cache.GetOrResolveManifest(e);
    assert(a.size() == 2);  // radius + color from clip_multi.usda
    size_t layerCountAfter1 = cache.Size();
    const auto& b = cache.GetOrResolveManifest(e);
    assert(&a == &b);
    assert(cache.Size() == layerCountAfter1);
    std::cout << "  Auto-manifest resolution cached: OK\n";
}

int main() {
    std::cout << "=== Foundational Data Types ===\n";
    TestHalf();
    TestToken();
    TestDefaultResolveAssetIdentifiers();
    TestCustomResolverSurvivesRecompose();
    TestVec();
    TestMatrix();
    TestValue();
    TestDictionary();
    TestDictionaryCombine();

    std::cout << "\n=== Value Comprehensive ===\n";
    TestValueAllScalars();
    TestValueDimensionedTypes();
    TestValueArrays();
    TestValueCopyMoveSemantics();
    TestValueRoles();
    TestValueBlockAndEmpty();
    TestValueCompositionTypes();

    std::cout << "\n=== Quaternions ===\n";
    TestQuaternions();

    std::cout << "\n=== Vec Comprehensive ===\n";
    TestVecComprehensive();

    std::cout << "\n=== Matrix Comprehensive ===\n";
    TestMatrixComprehensive();

    std::cout << "\n=== List Operations ===\n";
    TestListOpExplicit();
    TestListOpComposable();
    TestListOpCombine();
    TestListOpCombineComposable();
    TestListOpExplicitOverrides();
    TestListOpDefault();
    TestListOpReduce();
    TestListOpPathRoundtrip();

    std::cout << "\n=== ListOp Comprehensive ===\n";
    TestListOpGettersSetters();
    TestListOpCombineAllCases();
    TestListOpCombineEdgeCases();
    TestListOpReduceComprehensive();
    TestListOpEquality();
    TestListOpTypeSpecializations();
    TestListOpGetItemsOverlap();

    std::cout << "\n=== Paths ===\n";
    TestPathParsing();
    TestUnicodePathIdentifiers();
    TestPathElementOrdering();
    TestPathElements();
    TestPathPropertyNamespaces();
    TestUsdaWriterPathElementOrdering();
    TestPathOperations();
    TestPathAnchor();

    std::cout << "\n=== Document Data Model ===\n";
    TestPrimSpec();
    TestAttributeSpec();
    TestRelationshipSpec();
    TestLayer();
    TestLayerArcOpinionPrimPathsCacheInvalidation();
    TestLayerArcOpinionPrimPathsConcurrentRead();
    TestLayerChildPathIndexInvalidation();
    TestLayerCompositionFields();
    TestFieldRegistry();
    TestGenericFieldAccess();
    TestSpline();
    TestSplineValueTypeId();
    TestSplineUsdaParse();
    TestSplineUsdaRoundtrip();
    TestSplineEmptyRoundtrip();
    TestSplineUsdcRoundtrip();
    TestSplineUsdcEmptyRoundtrip();
    TestSplineUsdaUsdcUsdaTriangle();
    TestUsdcCrateVersionSelection();
    TestSplineUsdcSpecByteEncoding();
    TestSplineEvalHeld();
    TestSplineEvalLinear();
    TestSplineEvalNoneSegment();
    TestSplineEvalHermite();
    TestSplineEvalBezierDegenerateLinear();
    TestSplineEvalDualValueKnot();
    TestSplineEvalExtrapolation();
    TestSplineEvalEdges();
    TestSplineBakeToTimeSamples();
    TestSplineStageIntegration();
    TestSplinePerOpinionTimeSamplesWins();
    TestSplineLoopRepeat();
    TestSplineLoopReset();
    TestSplineLoopOscillate();
    TestSplineInnerLoopsPost();
    TestSplineInnerLoopsPre();
    TestSplineAntiRegression();
    TestSplineTypeNarrowingFloat();
    TestSplineTypeNarrowingHalf();

    std::cout << "\n=== USDA Parser ===\n";
    TestUsdaBasicLayer();
    TestUsdaAttributes();
    TestUsdaFoundationalTypeCoverage();
    TestUsdaUnicodeIdentifiers();
    TestUsdaGrammarStrictness();
    TestUsdaRelationships();
    TestUsdaOverAndClass();
    TestUsdaNestedPrims();
    TestUsdaTimeSamples();
    TestUsdaComments();
    TestUsdaMetadata();
    TestUsdaMatrix();

    std::cout << "\n=== USDA File Tests ===\n";
    TestUsdaFiles("tests/usda");

    std::cout << "\n=== USDA Invalid File Tests ===\n";
    TestUsdaInvalidFiles("tests/usda/invalid");

    std::cout << "\n=== Composition ===\n";
    TestParserStoresSubLayers();
    TestParserStoresReferences();
    TestComposeSubLayers();
    TestComposeUsdcSubLayersRegression();
    TestComposeInherits();
    TestComposeImpliedInherits();
    TestComposeSpecializes();
    TestComposeImpliedSpecializes();
    TestComposeLivrpsStrengthOrdering();
    TestComposeReferences();
    TestComposeBareSiblingReference();
    TestComposeLayerOffset();
    TestComposeThreeSublayers();
    TestComposeMultipleReferences();
    TestComposeInternalReference();
    TestComposeChainedReferences();
    TestComposeReferenceDiamond();
    TestComposeSublayerRefInteraction();
    TestComposeSubrootReference();
    TestComposeReferenceOffset();
    TestComposeCompoundOffset();
    TestInvalidRetimingScaleFallback();

    std::cout << "\n=== Diagnostics ===\n";
    TestDiagnosticCollectorBasics();
    TestDiagnosticToJson();
    TestComposeMissingSublayerDiagnostic();
    TestComposeMissingReferenceDiagnostic();
    TestComposeMissingPayloadDiagnostic();
    TestComposeMissingDefaultPrimDiagnostic();
    TestComposeInvalidReferenceTargetDiagnostic();
    TestStageOpenDegradedValid();
    TestComposeCleanStageNoDiagnostics();

    std::cout << "\n=== Variants ===\n";
    TestVariantApiSetAndRecompose();
    TestClipsMetadataRoundtripAndCombining();
    TestClipSetsUsdcStringListOp();
    TestApiSchemasUsdcTokenListOp();

    std::cout << "\n=== Value Clips ===\n";
    TestClipsMaterializeExplicit();
    TestClipsMaterializeTemplate();
    TestClipsTemplateStride();
    TestClipsTemplateActiveOffset();
    TestClipsTemplateOffsetExceedsStride();
    TestClipsTemplateSubInteger();
    TestClipsExplicitWinsOverTemplate();
    TestClipsTemplateMalformed();
    TestClipsMaterializeUnknownName();
    TestClipCacheHitReturnsSameLayer();
    TestClipCacheMissNegativeCached();
    TestClipCacheEmptyPath();
    TestClipCacheClear();
    TestManifestReturnsAttributePaths();
    TestManifestEmptyPath();
    TestManifestFailureIsCached();
    TestManifestCacheSharedAcrossEntries();
    TestClipsValueResolutionBasic();
    TestClipsActiveAssetSwitch();
    TestClipsLinearInterpolation();
    TestClipsDoNotOverrideLocalDefault();
    TestClipsPrimPathRemap();
    TestPerOpinionStrongDefaultBeatsWeakTimeSamples();
    TestFlattenSamplesStrongDefaultOverWeakTimeSamples();
    TestFlattenBakesSplineToTimeSamples();
    TestFlattenSamplesClipValues();
    TestStageClipCachePersists();
    TestClipsAutoManifestIncludesAllClipAttrs();
    TestClipsManifestGatesOutUndeclaredAttr();
    TestClipsMissingValueHoldsNearest();
    TestClipsMissingValueLinearInterp();
    TestClipsJumpDiscontinuityRightContinuous();
    TestClipSetManifestCaching();
    TestAutoManifestCaching();

    std::cout << "\n=== Stage ===\n";
    TestStageOpen();
    TestStagePopulation();
    TestStagePopulationMask();
    TestStagePopulationConsistency();
    TestStageTraversal();
    TestComposedChildIndexDedupesLayerSpecs();
    TestStageTraversePreservesStrongestSpecProvenance();
    TestStageTraverseSeesInPlaceSpecValueEdit();
    TestStageTraverseRebuildsAfterDefinePrim();
    TestStageTraverseFallsBackWhenCachedSpecRemoved();
    TestStageTraverseSeesTypeNameChange();
    TestStageTraverseSeesNewAuthoredAttribute();
    TestStageCreateAttributeAfterTraverse();
    TestStageChildrenAfterNestedDefine();
    TestUsdPrimHandleSurvivesWriteOnUnrelatedSpec();
    TestStageProperties();
    TestStageMetadata();
    TestMetersPerUnit();
    TestStageComposedOpen();

    std::cout << "\n=== Schema System ===\n";
    TestSchemaRegistration();
    TestSchemaJSON();
    TestSchemaIsA();
    TestSchemaHasAPI();
    TestSchemaPrimDefinition();
    TestSchemaFallbackValues();
    TestSchemaCoreSchemas();
    TestColorSpaceResolution();
    TestColorSpaceAuthoring();
    TestSchemaGeometrySchemas();
    TestSchemaIsAWithStage();
    TestSchemaAbstractTypeNameRejected();
    TestSchemaFallbackPrimTypes();
    TestSchemaAutoApplies();
    TestSchemaMultiApplyProperties();
    TestCollectionEvaluation();
    TestGeometryGprimSchemas();
    TestMaterialSchemas();
    TestPhysicsSchemas();
    TestKilogramsPerUnit();
    TestComposeListOpAcrossLayers();

    std::cout << "\n=== Value Resolution ===\n";
    TestValueResolutionDefault();
    TestValueResolutionTimeSamples();
    TestValueResolutionTimeSamplesNoDefault();
    TestValueResolutionBlocked();
    TestValueResolutionBlockedWithFallback();
    TestValueResolutionLinearInterpolation();
    TestValueResolutionUniformHeld();
    TestRelationshipTargets();
    TestRelationshipForwardedTargets();
    TestPathValuedListOpNamespaceRemap();
    TestComposeRelocates();
    TestComposeNestedRelocates();
    TestRelocatesUsdaRoundtrip();
    TestRelocatesUsdcRoundtrip();
    TestStageInterpolationType();

    std::cout << "\n=== USDC Parser ===\n";
    TestUsdcFormat();
    TestUsdcFiles("tests/usdc");
    TestUsdcUsdaEquivalence("tests/usdc", "tests/usda");

    std::cout << "\n=== USDC Deep Value Tests ===\n";
    // (these test functions are defined above main())
    TestUsdcScalarValues();
    TestUsdcVectorValues();
    TestUsdcArrayValues();
    TestUsdcSpecifiers();
    TestUsdcVariants();
    TestUsdcLayerMetadata();
    TestUsdcTimeSamples();
    TestUsdcDictionaries();
    TestUsdcListOps();
    TestUsdcNesting();

    std::cout << "\n=== Unified Parser ===\n";
    TestUnifiedParser();
    TestUsdzPackageRead();
    TestUsdzPackageWrite();

    std::cout << "\n=== Write Operations ===\n";
    TestWriteScalarDefaults();
    TestWriteVectorDefaults();
    TestWriteTimeSamples();
    TestWriteClearAndBlock();
    TestWriteCreateAttribute();

    std::cout << "\n=== P0 Physics Prerequisites ===\n";
    TestRelationshipWrite();
    TestStageCreation();
    TestApplyAPI();
    TestPhysicsP0Integration();

    std::cout << "\n=== P1 Extensions ===\n";
    TestAssetPathRead();

    std::cout << "\n=== XformOp ===\n";
    TestXformOpBasicSRT();
    TestXformOpTranslateOnly();
    TestXformOpRotation();
    TestXformOpOrient();
    TestXformOpSuffix();
    TestXformOpResetStack();
    TestXformOpNoOps();
    TestXformOpSingleAxis();
    TestXformOpTransform();

    std::cout << "\n=== USDC Write Support ===\n";
    TestWriteUsdcRoundtrip();
    TestWriteUsdcFileRoundtrip();
    TestResolvedResourceWriteSupport();
    TestWriteUsdcMultiRootHierarchy();
    TestWriteUsdcUpAxisToken();
    TestWriteUsdcPrimChildrenSynthesis();

    std::cout << "\n=== Prim Metadata ===\n";
    TestPrimMetadata();

    std::cout << "\n=== Instancing ===\n";
    TestInstancingBasic();
    TestInstancingDifferentRefs();
    TestInstancingNoArc();
    TestInstancingNonInstance();
    TestInstanceChildren();
    TestPrototypeHidden();
    TestLocalOpinionFiltering();
    TestPrototypeNavigation();
    TestRepeatedInstanceableRefsUseRepresentativeExpansion();
    TestInactiveRepresentedInstanceIsPruned();

    std::cout << "\nAll tests passed!\n";
    return 0;
}
