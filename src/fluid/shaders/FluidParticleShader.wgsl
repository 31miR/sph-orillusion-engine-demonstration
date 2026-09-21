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

// Debug visualization constants (milestone step 5: density heatmap).
const DEBUG_REST_DENSITY: f32 = 1000.0;
const DEBUG_OVER_DENSITY_CEILING: f32 = 2.0;

fn densityHeatmapColor(density: f32) -> vec4<f32> {
    let ratio = density / DEBUG_REST_DENSITY;
    let under = vec3<f32>(0.15, 0.35, 1.0);
    let neutral = vec3<f32>(0.92, 0.92, 0.88);
    let over = vec3<f32>(1.0, 0.15, 0.1);
    var color: vec3<f32>;
    if (ratio < 1.0) {
        color = mix(under, neutral, clamp(ratio, 0.0, 1.0));
    } else {
        let t = clamp((ratio - 1.0) / (DEBUG_OVER_DENSITY_CEILING - 1.0), 0.0, 1.0);
        color = mix(neutral, over, t);
    }
    return vec4<f32>(color, 1.0);
}

fn vert(vertex: VertexAttributes) -> VertexOutput {
    let particle = particles[vertex.index];

    // We position particles ourselves from the storage buffer above,
    // instead of using the engine's default per-instance model-matrix
    // lookup (models.matrix[instance_index]) — that array only has one
    // valid entry (this Object3D's real transform), so for the other
    // 511 instances it reads garbage GPU memory. Reset to identity.
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

    ORI_VertexOut.varying_Color = densityHeatmapColor(particle.velocity.w);
    ORI_VertexOut.member = clipPos;
    return ORI_VertexOut;
}

fn frag() {
    let tint = textureSample(baseMap, baseMapSampler, ORI_VertexVarying.fragUV0);
    ORI_ShadingInput.BaseColor = tint * ORI_VertexVarying.vColor;
    UnLit();
}
