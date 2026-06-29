// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// mtlx2msl: load a .mtlx, run MaterialXGenMsl, print the generated MSL.
//
// Phase 1 PoC: with stock MaterialXGenMsl, emits a forward-shaded pixel
// shader that includes a hard-coded direct-light loop. The output
// matches what MaterialXView would compile — useful as a sanity check
// that the codegen path is wired end-to-end before swapping in a
// closure-returning surface node for the Metal raytracer (Phase 2,
// mirrors the Vulkan-side `--rt` mode in tools/mtlx2glsl).
//
// Usage:
//     mtlx2msl <input.mtlx> [<libraries-path>]
//
// Default libraries path: $HOME/OpenUSD-install/src/MaterialX-1.39.4
//
// Build:
//     cd tools/mtlx2msl && cmake -B build -S . && cmake --build build
//
// Verified against the chess set's .mtlx files (per-piece materials,
// Standard Surface) and a few MaterialX-shipped sample assets.
//
// Port from Vulkan's tools/mtlx2glsl/main.cpp (commit bbd53c8). The
// only structural differences are:
//   - Generator: GlslShaderGenerator -> MslShaderGenerator
//   - --rt mode dropped — the closure-based RtSurfaceNodeMsl is
//     Phase 2 work, separate effort with its own commit.
//   - Default libraries path defaults to the OpenUSD-install MaterialX
//     source tree (which the chess harness already depends on).

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/File.h>
#include <MaterialXFormat/Util.h>
#include <MaterialXFormat/XmlIo.h>
#include <MaterialXGenMsl/MslShaderGenerator.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/Util.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

namespace mx = MaterialX;

static int usage(const char* argv0)
{
    std::fprintf(stderr,
        "Usage: %s <input.mtlx> [<libraries-path>]\n"
        "\n"
        "Loads a MaterialX document, runs MaterialXGenMsl on every\n"
        "renderable element, and prints each compiled vertex + pixel\n"
        "stage to stdout.\n"
        "\n"
        "Default libraries path: $HOME/OpenUSD-install/src/MaterialX-1.39.4\n"
        "(searchPath should be the directory CONTAINING `libraries/`).\n",
        argv0);
    return 2;
}

int main(int argc, const char** argv)
{
    int argIdx = 1;
    if (argIdx >= argc) return usage(argv[0]);

    const char* inputPath = argv[argIdx++];

    // Default the libraries path to the MaterialX source bundled with
    // OpenUSD-install. Override via the second positional argument or
    // MTLX_LIBS env var when running against a different MaterialX.
    std::string libsRoot;
    if (argIdx < argc) {
        libsRoot = argv[argIdx];
    } else if (const char* env = std::getenv("MTLX_LIBS")) {
        libsRoot = env;
    } else {
        const char* home = std::getenv("HOME");
        libsRoot = std::string(home ? home : "") +
                   "/OpenUSD-install/src/MaterialX-1.39.4";
    }

    try {
        // 1. Load the standard libraries (stdlib + pbrlib + bxdf + targets).
        mx::FileSearchPath searchPath(libsRoot);
        mx::DocumentPtr stdLib = mx::createDocument();
        mx::loadLibraries({ "libraries" }, searchPath, stdLib);

        // 2. Load the input document and attach the data library.
        mx::DocumentPtr doc = mx::createDocument();
        doc->setDataLibrary(stdLib);
        mx::readFromXmlFile(doc, inputPath);

        std::string validateMsg;
        if (!doc->validate(&validateMsg)) {
            std::fprintf(stderr, "MaterialX validate failed:\n%s\n",
                         validateMsg.c_str());
            return 1;
        }

        // 3. MSL shader generator + context + default CMS.
        mx::ShaderGeneratorPtr generator = mx::MslShaderGenerator::create();
        mx::GenContext context(generator);
        context.registerSourceCodeSearchPath(searchPath);

        mx::ColorManagementSystemPtr cms =
            mx::DefaultColorManagementSystem::create(generator->getTarget());
        cms->loadLibrary(stdLib);
        generator->setColorManagementSystem(cms);

        // 4. Renderable elements (surfacematerial, displacement nodes).
        std::vector<mx::TypedElementPtr> renderables =
            mx::findRenderableElements(doc);
        if (renderables.empty()) {
            std::fprintf(stderr, "No renderable elements in %s\n", inputPath);
            return 1;
        }

        // 5. Generate per element, dump MSL to stdout.
        for (auto& elem : renderables) {
            std::fprintf(stderr, "==== %s ====\n", elem->getName().c_str());

            mx::ShaderPtr shader =
                generator->generate(elem->getName(), elem, context);
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
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        return 1;
    }
}
