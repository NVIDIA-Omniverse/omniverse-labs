// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RtSurfaceNodeGlsl — closure-based replacement for the stock GLSL Surface
// shader emitter. Phase 2a of the closure-based MaterialX plan ("B-closure").
//
// Where the stock SurfaceNodeGlsl emits a forward-shaded fragment with a
// hard-coded direct-light loop, this class emits a single GLSL function
//
//     SurfaceClosure evalSurface_<name>(int closureType,
//                                       vec3 L, vec3 V, vec3 N, vec3 P,
//                                       float occlusion);
//
// returning the four canonical surface closures (reflection/transmission
// BSDFs, emission EDF, scalar opacity). The caller (the ray tracer's hit
// shader) decides which closure to fill on each call.
//
// Vertex stage emission is left identical to the stock implementation —
// position/normal/texcoord transport into the pixel stage is unchanged.
//
// Phase 2b (this file): the closure function definition is emitted via
// emitFunctionDefinition into the PIXEL stage's FUNCTIONS section (i.e.
// at file scope, ahead of `void main()`). emitFunctionCall is reduced to
// the stock-style output-alias declaration so the compound's
// "Emit final results" loop has a symbol to copy from.

#ifndef NUSD_RT_SURFACE_NODE_GLSL_H
#define NUSD_RT_SURFACE_NODE_GLSL_H

#include <MaterialXGenShader/GenUserData.h>
#include <MaterialXGenGlsl/Nodes/SurfaceNodeGlsl.h>

#include <string>

namespace nusd
{

// GenUserData payload that lets main.cpp tell the closure emitter the
// renderable element's name (e.g. "M_King_B"). Without this the emitter
// falls back to ShaderNode::getName(), which on standard_surface compounds
// is the inner "shader_constructor" alias and therefore gives every
// material the same function name.
class RtNamingUserData : public MaterialX::GenUserData
{
  public:
    static constexpr const char* NAME = "nusd_rt_naming";

    explicit RtNamingUserData(std::string renderable) :
        _renderable(std::move(renderable)) {}

    const std::string& getRenderable() const { return _renderable; }

    static std::shared_ptr<RtNamingUserData> create(std::string renderable)
    {
        return std::make_shared<RtNamingUserData>(std::move(renderable));
    }

  private:
    std::string _renderable;
};

class RtSurfaceNodeGlsl : public MaterialX::SurfaceNodeGlsl
{
  public:
    RtSurfaceNodeGlsl() = default;

    static MaterialX::ShaderNodeImplPtr create();

    // createVariables is overridden to skip addStageLightingUniforms — the
    // RT path has no GLSL light loop, so the lighting uniforms / helper
    // functions (numActiveLightSources, sampleLightSource, etc.) are dead
    // weight. Without this override the boilerplate would emit them and
    // pollute the grep-based validation.
    void createVariables(const MaterialX::ShaderNode& node,
                         MaterialX::GenContext& context,
                         MaterialX::Shader& shader) const override;

    // emitFunctionDefinition writes the SurfaceClosure-returning function
    // at file scope, in the FUNCTIONS section of the PIXEL stage. This is
    // invoked by the parent compound (or by the top-level GlslShaderGenerator)
    // *before* it begins emitting `void main()`, so the resulting GLSL is
    // structurally legal: a free-standing function above main(), not a
    // nested function inside it.
    void emitFunctionDefinition(const MaterialX::ShaderNode& node,
                                MaterialX::GenContext& context,
                                MaterialX::ShaderStage& stage) const override;

    // emitFunctionCall is reduced to the stock-style output-alias
    // declaration only. The compound that owns this surface node ends with
    //   out1 = shader_constructor_out;
    // so we MUST leave a `surfaceshader shader_constructor_out = ...`
    // alias in scope, but we no longer emit the full closure function here
    // (that has been moved to emitFunctionDefinition).
    void emitFunctionCall(const MaterialX::ShaderNode& node,
                          MaterialX::GenContext& context,
                          MaterialX::ShaderStage& stage) const override;
};

} // namespace nusd

#endif // NUSD_RT_SURFACE_NODE_GLSL_H
