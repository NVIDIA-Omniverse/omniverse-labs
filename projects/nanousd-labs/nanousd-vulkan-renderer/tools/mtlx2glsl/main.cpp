// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// mtlx2glsl: load a .mtlx, run MaterialXGenGlsl, print the generated GLSL.
//
// Phase 1 PoC: with stock MaterialXGenGlsl, emits a forward-shaded fragment
// shader that includes a hard-coded direct-light loop.
// Phase 2a (B-closure, --rt flag): swaps the surface-shader implementation
// for nusd::RtSurfaceNodeGlsl, which emits a closure-shaped GLSL function
// (`SurfaceClosure evalSurface_<name>(...)`) appropriate for our Vulkan
// ray-tracer hit shader.
//
// Usage:
//     mtlx2glsl [--rt] <input.mtlx> [<libraries-path>]
//
// Defaults the libraries path to MaterialX's source-tree libraries/.
//
// Build (linked against a MaterialX install at /tmp/mtlx-install):
//     g++ -std=c++17 -I/tmp/mtlx-install/include main.cpp \
//         RtSurfaceNodeGlsl.cpp \
//         -L/tmp/mtlx-install/lib \
//         -lMaterialXGenGlsl -lMaterialXGenShader -lMaterialXFormat \
//         -lMaterialXCore -o mtlx2glsl

#include "RtSurfaceNodeGlsl.h"

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace mx = MaterialX;

static int usage(const char* argv0)
{
    std::fprintf(stderr,
        "Usage: %s [--rt] <input.mtlx> [<libraries-path>]\n"
        "  --rt  Use closure-based RtSurfaceNodeGlsl (Phase 2a) instead of\n"
        "        the stock forward-shaded SurfaceNodeGlsl.\n"
        "Default libraries path: /tmp/mtlx-src/libraries\n",
        argv0);
    return 2;
}

int main(int argc, const char** argv)
{
    bool useRt = false;
    int argIdx = 1;
    while (argIdx < argc && argv[argIdx][0] == '-')
    {
        if (std::strcmp(argv[argIdx], "--rt") == 0)
        {
            useRt = true;
            ++argIdx;
        }
        else if (std::strcmp(argv[argIdx], "-h") == 0 ||
                 std::strcmp(argv[argIdx], "--help") == 0)
        {
            return usage(argv[0]);
        }
        else
        {
            std::fprintf(stderr, "Unknown flag: %s\n", argv[argIdx]);
            return usage(argv[0]);
        }
    }
    if (argIdx >= argc) return usage(argv[0]);

    const char* inputPath = argv[argIdx++];
    // Default: searchPath is the MaterialX source root that contains a
    // `libraries/` subdirectory; pass an extra arg to override.
    const std::string libsRoot = (argIdx < argc) ? argv[argIdx] : "/tmp/mtlx-src";

    try {
        // 1. Load the standard libraries — stdlib + pbrlib + bxdf + targets.
        // Convention from MaterialXView/Viewer.cpp:1806 and tests:
        //   searchPath points at the parent of `libraries/`; the folder list
        //   is `{ "libraries" }`.
        mx::FileSearchPath searchPath(libsRoot);
        mx::DocumentPtr stdLib = mx::createDocument();
        mx::loadLibraries({ "libraries" }, searchPath, stdLib);

        // 2. Load the input document and attach the data library.
        mx::DocumentPtr doc = mx::createDocument();
        doc->setDataLibrary(stdLib);
        mx::readFromXmlFile(doc, inputPath);

        std::string validateMsg;
        if (!doc->validate(&validateMsg)) {
            std::fprintf(stderr, "MaterialX validate failed:\n%s\n", validateMsg.c_str());
            return 1;
        }

        // 3. Create the GLSL shader generator + context.
        mx::ShaderGeneratorPtr generator = mx::GlslShaderGenerator::create();

        // 3a. Optionally override the stock surface-node implementation. This
        // must happen BEFORE any shader is generated (so the ImplFactory
        // picks up the new creator on first lookup), but the ImplFactory map
        // is just an unordered_map<string, fn> — re-registration overwrites.
        // The registration key matches GlslShaderGenerator's stock binding:
        //   "IM_surface_" + GlslShaderGenerator::TARGET  →  "IM_surface_genglsl"
        if (useRt)
        {
            const std::string key = "IM_surface_" + mx::GlslShaderGenerator::TARGET;
            generator->registerImplementation(key, nusd::RtSurfaceNodeGlsl::create);
            std::fprintf(stderr,
                "[mtlx2glsl] Closure-based RtSurfaceNodeGlsl registered for '%s'\n",
                key.c_str());
        }

        mx::GenContext context(generator);
        context.registerSourceCodeSearchPath(searchPath);

        // 3b. In RT mode there is no GLSL direct-light loop; suppress the
        // emission of the (unused) light-sampling helper functions
        // numActiveLightSources() and sampleLightSource() that the
        // GlslShaderGenerator would otherwise inject. Both of those are
        // gated on hwMaxActiveLightSources > 0 in
        // GlslShaderGenerator::emitLightFunctionDefinitions.
        if (useRt)
        {
            context.getOptions().hwMaxActiveLightSources = 0;
        }
        // Default color management — sRGB texture decode, etc.
        mx::ColorManagementSystemPtr cms =
            mx::DefaultColorManagementSystem::create(generator->getTarget());
        cms->loadLibrary(stdLib);
        generator->setColorManagementSystem(cms);

        // 4. Find renderable elements (surface materials, displacement nodes).
        std::vector<mx::TypedElementPtr> renderables = mx::findRenderableElements(doc);
        if (renderables.empty()) {
            std::fprintf(stderr, "No renderable elements in %s\n", inputPath);
            return 1;
        }

        // 5. For each, run the generator and dump GLSL.
        for (auto& elem : renderables) {
            std::fprintf(stderr, "==== %s ====\n", elem->getName().c_str());

            // Phase 2b: in --rt mode, expose the renderable element's name
            // to RtSurfaceNodeGlsl so it can suffix the closure function
            // (e.g. evalSurface_M_King_B). Without this push the emitter
            // falls back to the inner ShaderNode's name, which on
            // standard_surface compounds is always "shader_constructor"
            // and would cause name collisions when multiple materials are
            // emitted into the same translation unit.
            if (useRt)
            {
                context.pushUserData(
                    nusd::RtNamingUserData::NAME,
                    nusd::RtNamingUserData::create(elem->getName()));
            }

            mx::ShaderPtr shader = generator->generate(elem->getName(), elem, context);

            if (useRt)
            {
                context.popUserData(nusd::RtNamingUserData::NAME);
            }

            if (!shader) {
                std::fprintf(stderr, "  [generate returned null]\n");
                continue;
            }
            const std::string vsSource = shader->getSourceCode(mx::Stage::VERTEX);
            const std::string psSource = shader->getSourceCode(mx::Stage::PIXEL);
            std::fprintf(stderr, "  vertex: %zu bytes\n", vsSource.size());
            std::fprintf(stderr, "  pixel:  %zu bytes\n", psSource.size());

            std::printf("// ============= %s — VERTEX STAGE =============\n%s\n",
                        elem->getName().c_str(), vsSource.c_str());
            std::printf("// ============= %s — PIXEL STAGE =============\n%s\n",
                        elem->getName().c_str(), psSource.c_str());
        }

        return 0;
    }
    catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
}
