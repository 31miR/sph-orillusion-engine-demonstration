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

// Rendered by FluidDepthPass into an isolated target (see
// FluidDepthPass.ts), never shown on screen. Encodes one number per
// pixel: distance from the camera to this particle.
//
// View space looks down +Z here (clip.w = viewPos.z, positive for
// anything visible), so distance is Z itself, unnegated. Background
// pixels keep the clear color, 0 — unambiguous, since a real particle
// can never produce exactly 0.
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
    let distanceToCamera = ORI_VertexVarying.viewPosition.z;
    ORI_ShadingInput.BaseColor = vec4<f32>(distanceToCamera, 0.0, 0.0, 1.0);
    UnLit();
}
