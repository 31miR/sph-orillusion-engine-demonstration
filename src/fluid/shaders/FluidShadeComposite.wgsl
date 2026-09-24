// Step 4 of the screen-space fluid renderer: shade the reconstructed
// surface using its normal (basic diffuse + Fresnel reflectivity —
// the "looks like water" effect discussed a few messages back), then
// composite over whatever the scene looked like so far. This is a
// ViewQuad post-processing shader (a full-screen fragment pass, the
// engine's own mechanism for "transform the image and hand it to the
// next stage" — see e.g. TonemapShader/FXAA for the same shape),
// unlike the earlier steps which were bare compute shaders we read
// back manually ourselves.
//
// Light direction and water/sky colors are fixed constants for this
// first version, not yet wired to the scene's real DirectLight or a
// GUI control — expect these to need real tuning once visible.

@group(1) @binding(0)
var baseMapSampler: sampler;

@group(1) @binding(1)
var baseMap: texture_2d<f32>;

// Every other material/ViewQuad shader in this project (and the
// engine's own, e.g. TonemapShader) declares something at group 2 —
// an empty group 2 produced a "can't set empty group! 2" reflection
// warning here, so f0 (Fresnel reflectance at normal incidence) lives
// here instead of as a bare constant, which also makes it a real
// tunable rather than a throwaway fix.
struct MaterialUniform {
    f0: f32,
};

@group(2) @binding(0)
var<uniform> materialUniform: MaterialUniform;

@group(3) @binding(0)
var depthTex: texture_2d<f32>;

@group(3) @binding(1)
var normalTex: texture_2d<f32>;

struct FragmentOutput {
    @location(auto) o_Target: vec4<f32>
};

const LIGHT_DIR_VIEW: vec3<f32> = vec3<f32>(0.3, 0.6, 0.4);
const WATER_COLOR: vec3<f32> = vec3<f32>(0.05, 0.25, 0.35);
const SKY_TINT: vec3<f32> = vec3<f32>(0.6, 0.75, 0.9);

fn shadeSurface(coord: vec2<i32>) -> vec3<f32> {
    let packedNormal = textureLoad(normalTex, coord, 0).xyz;
    let normal = normalize(packedNormal * 2.0 - 1.0);
    let lightDir = normalize(LIGHT_DIR_VIEW);
    let viewDir = vec3<f32>(0.0, 0.0, 1.0);

    let ndotl = max(dot(normal, lightDir), 0.0);
    let diffuse = WATER_COLOR * (0.25 + 0.75 * ndotl);

    let ndotv = max(dot(normal, viewDir), 0.0);
    let f0 = materialUniform.f0;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, 5.0);

    return mix(diffuse, SKY_TINT, fresnel);
}

@fragment
fn main(@location(auto) fragUV: vec2<f32>) -> FragmentOutput {
    // The Y-flip below is specifically because baseMap is a previous
    // post-stage's render-target output (see TonemapShader, which
    // established this same flip for the same reason). depthTex/
    // normalTex need no flip HERE because FluidDepthBlur.wgsl already
    // corrects for the same underlying issue once, on read, right
    // where the raw (render-target-sourced) capture first enters our
    // own compute pipeline — see loadCaptured() there.
    var flippedUV = fragUV;
    flippedUV.y = 1.0 - flippedUV.y;

    let sceneColor = textureSample(baseMap, baseMapSampler, flippedUV);
    let size = textureDimensions(depthTex);
    let coord = vec2<i32>(fragUV * vec2<f32>(size));
    let depth = textureLoad(depthTex, coord, 0).r;

    if (depth <= 0.0) {
        return FragmentOutput(sceneColor);
    }

    return FragmentOutput(vec4<f32>(shadeSurface(coord), 1.0));
}
