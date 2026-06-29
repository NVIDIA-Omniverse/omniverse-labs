// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RtSurfaceNodeGlsl — see header for design notes.
//
// Phase 2b structural fixes:
//   * emit closure function via emitFunctionDefinition (file scope)
//   * pull renderable name from GenContext user-data (RtNamingUserData)
//   * coerce vec3 opacity to float via luminance dot
//
// The vertex-stage block is copied verbatim from the upstream
// implementation so position/normal transport into the pixel stage is
// unchanged.

#include "RtSurfaceNodeGlsl.h"

#include <MaterialXGenGlsl/GlslShaderGenerator.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/HwShaderGenerator.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderStage.h>

namespace nusd
{

namespace mx = MaterialX;

mx::ShaderNodeImplPtr RtSurfaceNodeGlsl::create()
{
    return std::make_shared<RtSurfaceNodeGlsl>();
}

void RtSurfaceNodeGlsl::createVariables(const mx::ShaderNode& /*node*/,
                                        mx::GenContext& /*context*/,
                                        mx::Shader& shader) const
{
    // Mirror SurfaceNodeGlsl::createVariables (lines 30-43 of upstream)
    // EXCEPT the trailing addStageLightingUniforms call — see header for
    // rationale. This block is otherwise verbatim from the stock impl.
    mx::ShaderStage& vs = shader.getStage(mx::Stage::VERTEX);
    mx::ShaderStage& ps = shader.getStage(mx::Stage::PIXEL);

    addStageInput(mx::HW::VERTEX_INPUTS, mx::Type::VECTOR3, mx::HW::T_IN_POSITION, vs);
    addStageInput(mx::HW::VERTEX_INPUTS, mx::Type::VECTOR3, mx::HW::T_IN_NORMAL, vs);
    addStageUniform(mx::HW::PRIVATE_UNIFORMS, mx::Type::MATRIX44,
                    mx::HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX, vs);

    addStageConnector(mx::HW::VERTEX_DATA, mx::Type::VECTOR3, mx::HW::T_POSITION_WORLD, vs, ps);
    addStageConnector(mx::HW::VERTEX_DATA, mx::Type::VECTOR3, mx::HW::T_NORMAL_WORLD, vs, ps);

    addStageUniform(mx::HW::PRIVATE_UNIFORMS, mx::Type::VECTOR3, mx::HW::T_VIEW_POSITION, ps);

    // Intentionally NOT calling shadergen.addStageLightingUniforms — the RT
    // path doesn't use the GLSL direct-light loop.
}

// -- helper: pick the function-name suffix for evalSurface_<name> ---------
// Prefer the renderable element's name, supplied via RtNamingUserData by
// main.cpp. Fall back to the ShaderNode's own name if nothing's pushed
// (this keeps the emitter usable in unit tests / harnesses that don't
// know about RtNamingUserData).
static std::string pickFunctionSuffix(const mx::ShaderNode& node, mx::GenContext& context)
{
    auto userData = context.getUserData<RtNamingUserData>(RtNamingUserData::NAME);
    if (userData && !userData->getRenderable().empty())
    {
        return userData->getRenderable();
    }
    return node.getName();
}

// -- helper: coerce an opacity input expression to a GLSL float ----------
// SurfaceClosure::opacity is `float`. The standard_surface NodeGraph
// already pre-converts color3 opacity through a luminance + extract
// chain, so in the vast majority of cases the input port is already
// `float`. But the surface-shader nodedef's opacity port is *typed*
// color3 in MaterialX, and a custom NodeGraph could connect a vec3
// directly. Defensively coerce: vec3 → luminance dot, otherwise emit
// the input expression as-is.
static void emitOpacityAsFloat(const mx::ShaderInput* input,
                               const mx::GlslShaderGenerator& shadergen,
                               mx::GenContext& context,
                               mx::ShaderStage& stage)
{
    const bool isVec3 = input->getType().isFloat3();
    if (isVec3)
    {
        // Rec.709 luminance — same coefficients MaterialX itself uses in
        // the stdlib `<luminance>` node default for color3.
        shadergen.emitString("dot(", stage);
        shadergen.emitInput(input, context, stage);
        shadergen.emitString(", vec3(0.2722287, 0.6740818, 0.0536895))", stage);
    }
    else
    {
        shadergen.emitInput(input, context, stage);
    }
}

void RtSurfaceNodeGlsl::emitFunctionDefinition(const mx::ShaderNode& node,
                                               mx::GenContext& context,
                                               mx::ShaderStage& stage) const
{
    // Closure function is a pixel-stage construct. Vertex stage has
    // nothing to define here.
    DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL)
    {
        const mx::GlslShaderGenerator& shadergen =
            static_cast<const mx::GlslShaderGenerator&>(context.getShaderGenerator());

        // Emit the SurfaceClosure struct definition once per stage. The
        // BSDF / EDF / ClosureData / CLOSURE_TYPE_* identifiers come from
        // the stock GLSL prelude; only the aggregate that combines all
        // four lobes is owned by this emitter. We use a string-search on
        // the accumulated stage source to detect prior emission, which is
        // simpler than a GenUserData flag and works across multiple
        // RtSurfaceNodeGlsl instances driving the same stage.
        if (stage.getSourceCode().find("struct SurfaceClosure") == std::string::npos)
        {
            shadergen.emitComment("Closure aggregate (Phase 2b B-closure)", stage);
            shadergen.emitLine("struct SurfaceClosure", stage, false);
            shadergen.emitScopeBegin(stage);
            shadergen.emitLine("BSDF  reflection", stage);
            shadergen.emitLine("BSDF  transmission", stage);
            shadergen.emitLine("EDF   emission", stage);
            shadergen.emitLine("float opacity", stage);
            shadergen.emitScopeEnd(stage, true);
            shadergen.emitLineBreak(stage);
        }

        const std::string fnName = "evalSurface_" + pickFunctionSuffix(node, context);

        shadergen.emitComment("Closure-based surface shader (Phase 2b B-closure)", stage);
        shadergen.emitLine(
            "SurfaceClosure " + fnName +
                "(int closureType, vec3 L, vec3 V, vec3 N, vec3 P, float occlusion)",
            stage, false);
        shadergen.emitScopeBegin(stage);

        // Build canonical closure data parameterised over closureType so
        // all four canonical closures fan out from one entry.
        shadergen.emitLine(
            "ClosureData closureData = ClosureData(closureType, L, V, N, P, occlusion)", stage);
        shadergen.emitLine("SurfaceClosure result", stage);
        // Match the GLSL aggregate ctors registered in GlslSyntax.cpp:
        //   BSDF(vec3, vec3)  /  EDF == vec3 (so EDF(vec3(0.0)))
        shadergen.emitLine("result.reflection   = BSDF(vec3(0.0), vec3(1.0))", stage);
        shadergen.emitLine("result.transmission = BSDF(vec3(0.0), vec3(1.0))", stage);
        shadergen.emitLine("result.emission     = EDF(vec3(0.0))", stage);
        shadergen.emitLine("result.opacity      = 1.0", stage);
        shadergen.emitLineBreak(stage);

        // ---- BSDF — reflection/indirect or transmission ----
        const mx::ShaderInput* bsdfInput = node.getInput("bsdf");
        if (bsdfInput)
        {
            if (const mx::ShaderNode* bsdf = bsdfInput->getConnectedSibling())
            {
                shadergen.emitComment("Reflection / indirect lobe", stage);
                shadergen.emitLine(
                    "if (closureType == CLOSURE_TYPE_REFLECTION || "
                    "closureType == CLOSURE_TYPE_INDIRECT)",
                    stage, false);
                shadergen.emitScopeBegin(stage);
                if (bsdf->hasClassification(mx::ShaderNode::Classification::BSDF_R))
                {
                    shadergen.emitFunctionCall(*bsdf, context, stage);
                }
                else
                {
                    shadergen.emitLineBegin(stage);
                    shadergen.emitOutput(bsdf->getOutput(), true, true, context, stage);
                    shadergen.emitLineEnd(stage);
                }
                shadergen.emitLine(
                    "result.reflection = " + bsdf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
                shadergen.emitLineBreak(stage);

                shadergen.emitComment("Transmission lobe", stage);
                shadergen.emitLine("if (closureType == CLOSURE_TYPE_TRANSMISSION)",
                                   stage, false);
                shadergen.emitScopeBegin(stage);
                if (bsdf->hasClassification(mx::ShaderNode::Classification::BSDF_T) ||
                    bsdf->hasClassification(mx::ShaderNode::Classification::VDF))
                {
                    shadergen.emitFunctionCall(*bsdf, context, stage);
                }
                else
                {
                    shadergen.emitLineBegin(stage);
                    shadergen.emitOutput(bsdf->getOutput(), true, true, context, stage);
                    shadergen.emitLineEnd(stage);
                }
                shadergen.emitLine(
                    "result.transmission = " + bsdf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
                shadergen.emitLineBreak(stage);
            }
        }

        // ---- EDF — emission ----
        const mx::ShaderInput* edfInput = node.getInput("edf");
        if (edfInput)
        {
            if (const mx::ShaderNode* edf = edfInput->getConnectedSibling())
            {
                shadergen.emitComment("Emission lobe", stage);
                shadergen.emitLine("if (closureType == CLOSURE_TYPE_EMISSION)",
                                   stage, false);
                shadergen.emitScopeBegin(stage);
                if (edf->hasClassification(mx::ShaderNode::Classification::EDF))
                {
                    shadergen.emitFunctionCall(*edf, context, stage);
                }
                else
                {
                    shadergen.emitLineBegin(stage);
                    shadergen.emitOutput(edf->getOutput(), true, true, context, stage);
                    shadergen.emitLineEnd(stage);
                }
                shadergen.emitLine(
                    "result.emission = " + edf->getOutput()->getVariable(), stage);
                shadergen.emitScopeEnd(stage);
                shadergen.emitLineBreak(stage);
            }
        }

        // ---- Opacity (always evaluated; cheap scalar) ----
        const mx::ShaderInput* opacityInput = node.getInput("opacity");
        if (opacityInput)
        {
            shadergen.emitLineBegin(stage);
            shadergen.emitString("result.opacity = ", stage);
            emitOpacityAsFloat(opacityInput, shadergen, context, stage);
            shadergen.emitLineEnd(stage);
        }

        shadergen.emitLine("return result", stage);
        shadergen.emitScopeEnd(stage);
        shadergen.emitLineBreak(stage);
    }
}

void RtSurfaceNodeGlsl::emitFunctionCall(const mx::ShaderNode& node,
                                         mx::GenContext& context,
                                         mx::ShaderStage& stage) const
{
    const mx::GlslShaderGenerator& shadergen =
        static_cast<const mx::GlslShaderGenerator&>(context.getShaderGenerator());

    // ---------------- Vertex stage (verbatim copy of upstream) ----------------
    DEFINE_SHADER_STAGE(stage, mx::Stage::VERTEX)
    {
        mx::VariableBlock& vertexData = stage.getOutputBlock(mx::HW::VERTEX_DATA);
        const mx::string prefix = shadergen.getVertexDataPrefix(vertexData);
        mx::ShaderPort* position = vertexData[mx::HW::T_POSITION_WORLD];
        if (!position->isEmitted())
        {
            position->setEmitted();
            shadergen.emitLine(prefix + position->getVariable() + " = hPositionWorld.xyz", stage);
        }
        mx::ShaderPort* normal = vertexData[mx::HW::T_NORMAL_WORLD];
        if (!normal->isEmitted())
        {
            normal->setEmitted();
            shadergen.emitLine(prefix + normal->getVariable() + " = normalize((" +
                                   mx::HW::T_WORLD_INVERSE_TRANSPOSE_MATRIX +
                                   " * vec4(" + mx::HW::T_IN_NORMAL + ", 0)).xyz)",
                               stage);
        }
        if (context.getOptions().hwAmbientOcclusion)
        {
            mx::ShaderPort* texcoord = vertexData[mx::HW::T_TEXCOORD + "_0"];
            if (!texcoord->isEmitted())
            {
                texcoord->setEmitted();
                shadergen.emitLine(prefix + texcoord->getVariable() + " = " +
                                       mx::HW::T_IN_TEXCOORD + "_0",
                                   stage);
            }
        }
    }

    // ---------------- Pixel stage: stock-style alias only ----------------
    DEFINE_SHADER_STAGE(stage, mx::Stage::PIXEL)
    {
        // The closure-shaped function definition is no longer emitted
        // here (it now lives at file scope, see emitFunctionDefinition).
        //
        // The compound that owns this surface node ends its body with
        //   out1 = shader_constructor_out;
        // so we MUST leave a `surfaceshader shader_constructor_out = ...`
        // alias declaration in scope. That's the entirety of this body.
        const mx::ShaderOutput* output = node.getOutput();
        shadergen.emitLineBegin(stage);
        shadergen.emitOutput(output, true, true, context, stage);
        shadergen.emitLineEnd(stage);
    }
}

} // namespace nusd
