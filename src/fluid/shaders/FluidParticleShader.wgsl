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

const DEBUG_HEATMAP_MAX: f32 = 80.0;

fn heatmapColor(t: f32) -> vec4<f32> {
    let clamped = clamp(t, 0.0, 1.0);
    let cold = vec3<f32>(0.1, 0.2, 1.0);
    let mid = vec3<f32>(1.0, 0.9, 0.1);
    let hot = vec3<f32>(1.0, 0.1, 0.1);
    var color: vec3<f32>;
    if (clamped < 0.5) {
        color = mix(cold, mid, clamped * 2.0);
    } else {
        color = mix(mid, hot, (clamped - 0.5) * 2.0);
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

    ORI_VertexOut.varying_Color = heatmapColor(particle.velocity.w / DEBUG_HEATMAP_MAX);
    ORI_VertexOut.member = clipPos;
    return ORI_VertexOut;
}

fn frag() {
    let tint = textureSample(baseMap, baseMapSampler, ORI_VertexVarying.fragUV0);
    ORI_ShadingInput.BaseColor = tint * ORI_VertexVarying.vColor;
    UnLit();
}
