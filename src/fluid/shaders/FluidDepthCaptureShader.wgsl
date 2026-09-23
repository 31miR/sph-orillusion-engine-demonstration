#include "Common_vert"
#include "Common_frag"
#include "UnLit_frag"
#include "UnLitMaterialUniform_frag"
#include "FluidParticleData"

@group(1) @binding(0)
var baseMapSampler: sampler;

@group(1) @binding(1)
var baseMap: texture_2d<f32>;

@group(3) @binding(0)
var<storage, read> particles: array<FluidParticle>;

// Captured by SceneCaptureCameraComponent (see FluidParticleField.ts)
// instead of shown on screen — this material never appears in the
// normal view. Its only job is to encode, as an ordinary color value,
// the one number the screen-space renderer actually needs: distance
// from the camera to this particle. Reusing the hardware depth buffer
// directly turned out to require an undocumented internal render pass
// (see git history); this sidesteps that entirely by writing distance
// into the captured color texture instead, which the engine exposes
// through the fully public getCaptureTexture().
//
// View space looks down -Z in this engine's convention (matches
// Common_vert's ORI_MATRIX_V usage elsewhere), so distance in front of
// the camera is the negated Z. Background pixels (no particle drawn)
// keep the capture's clear color, 0 — a real particle can never
// produce exactly 0 since anything at the camera's near plane is
// already clipped, so 0 unambiguously means "no particle here" for
// the smoothing step that reads this back.
fn vert(vertex: VertexAttributes) -> VertexOutput {
    let particle = particles[vertex.index];

    ORI_MATRIX_M = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, 1.0)
    );

    let localPos = vertex.position + particle.position.xyz;
    var worldPos = ORI_MATRIX_M * vec4<f32>(localPos, 1.0);
    var viewPos = ORI_MATRIX_V * worldPos;
    var clipPos = ORI_MATRIX_P * viewPos;

    ORI_VertexOut.varying_ViewPos = viewPos;
    ORI_VertexOut.member = clipPos;
    return ORI_VertexOut;
}

fn frag() {
    let distanceToCamera = -ORI_VertexVarying.viewPosition.z;
    ORI_ShadingInput.BaseColor = vec4<f32>(distanceToCamera, 0.0, 0.0, 1.0);
    UnLit();
}
