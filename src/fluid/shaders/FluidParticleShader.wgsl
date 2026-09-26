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

// Pressure heatmap: pressure (Eq. 9) can be negative, so this is
// centered at zero (blue = negative/tension, red = positive/
// compression) and mapped on a log scale, since the equation of
// state's 7th power gives it an enormous dynamic range — a linear
// scale would saturate almost instantly. DEBUG_PRESSURE_LOG_CEILING is
// the |pressure| that maps to fully saturated.
const DEBUG_PRESSURE_LOG_CEILING: f32 = 1000000.0;

fn pressureHeatmapColor(pressure: f32) -> vec4<f32> {
    let cold = vec3<f32>(0.15, 0.35, 1.0);
    let neutral = vec3<f32>(0.92, 0.92, 0.88);
    let hot = vec3<f32>(1.0, 0.1, 0.1);

    let t = clamp(log(1.0 + abs(pressure)) / log(1.0 + DEBUG_PRESSURE_LOG_CEILING), 0.0, 1.0);
    var color: vec3<f32>;
    if (pressure < 0.0) {
        color = mix(neutral, cold, t);
    } else {
        color = mix(neutral, hot, t);
    }
    return vec4<f32>(color, 1.0);
}

fn vert(vertex: VertexAttributes) -> VertexOutput {
    let particle = particles[vertex.index];

    // Particles are positioned from the storage buffer above; the
    // engine's default per-instance model-matrix lookup only has one
    // valid entry, so reset to identity instead of using it.
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

    ORI_VertexOut.varying_Color = pressureHeatmapColor(particle.position.w);
    ORI_VertexOut.member = clipPos;
    return ORI_VertexOut;
}

fn frag() {
    let tint = textureSample(baseMap, baseMapSampler, ORI_VertexVarying.fragUV0);
    ORI_ShadingInput.BaseColor = tint * ORI_VertexVarying.vColor;
    UnLit();
}
