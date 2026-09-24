// Step 4 of the screen-space fluid renderer: shade the reconstructed
// surface using its normal (diffuse + Fresnel reflectivity, plus a
// refracted view of the actual scene behind the water instead of a
// flat placeholder color — the "looks like water" effect discussed a
// few messages back), then composite over whatever the scene looked
// like so far. This is a ViewQuad post-processing shader (a full-screen
// fragment pass, the engine's own mechanism for "transform the image
// and hand it to the next stage" — see e.g. TonemapShader/FXAA for the
// same shape), unlike the earlier steps which were bare compute shaders
// we read back manually ourselves.
//
// Light direction is a fixed constant for this first version, not yet
// wired to the scene's real DirectLight — expect this to need real
// tuning once visible. Reflection is still a flat SKY_TINT placeholder
// (no environment map to reflect); only the transmission side is now
// the real scene, refracted.

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
// How far (in UV space) the surface's tilt bends the sampled
// background — a stand-in for tracing a real refraction ray, cheap
// enough for one extra texture sample. Tuned by eye: too large starts
// sampling parts of the screen unrelated to "what's behind this point
// in the water."
const REFRACTION_STRENGTH: f32 = 0.04;
// How much of the water's own color tints the refracted view — at 0
// this would look like clear glass, not colored water.
const WATER_TINT_STRENGTH: f32 = 0.35;

fn shadeSurface(coord: vec2<i32>, flippedUV: vec2<f32>) -> vec3<f32> {
    let packedNormal = textureLoad(normalTex, coord, 0).xyz;
    let normal_needs_flip = normalize(packedNormal * 2.0 - 1.0);
    let normal = normal_needs_flip * vec3<f32>(1.0, -1.0, 1.0);
    let lightDir = normalize(LIGHT_DIR_VIEW);
    let viewDir = vec3<f32>(0.0, 0.0, 1.0);

    let ndotl = max(dot(normal, lightDir), 0.0);
    let ndotv = max(dot(normal, viewDir), 0.0);
    let f0 = materialUniform.f0;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, 5.0);

    // Bend the background sample by the surface's tilt (normal.xy) —
    // the same idea a real refraction ray follows, just without
    // actually tracing one. A flat surface (normal.xy = 0) samples
    // straight through with no distortion. Subtracted, not added: by
    // Snell's law (bending toward the normal on entering the denser
    // medium), a downward-traveling ray bends *opposite* the
    // direction the normal tilts, not the same way — confirmed against
    // the standard vector refraction formula, and against the classic
    // pencil-in-a-glass-of-water photo (the side where the glass's
    // outward normal points away from center is the side the pencil
    // appears displaced toward, matching a ray that bent toward the
    // center to get there).
    let refractedUV = clamp(flippedUV - normal.xy * REFRACTION_STRENGTH, vec2<f32>(0.0), vec2<f32>(1.0));
    // textureSampleLevel, not textureSample: this runs inside
    // shadeSurface(), which main() only calls for pixels where
    // depth > 0 — non-uniform control flow (only some pixels take that
    // branch). textureSample computes implicit derivatives across a
    // 2x2 pixel block for mip selection, which WGSL requires uniform
    // control flow for; textureSampleLevel takes an explicit LOD
    // instead, so it has no such restriction. baseMap has no mipmaps
    // anyway, so LOD 0 is exactly what textureSample would have picked.
    let refracted = textureSampleLevel(baseMap, baseMapSampler, refractedUV, 0.0).rgb;
    let lit = refracted * (0.25 + 0.75 * ndotl);
    let transmission = mix(lit, WATER_COLOR, WATER_TINT_STRENGTH);

    return mix(transmission, SKY_TINT, fresnel);
}

@fragment
fn main(@location(auto) fragUV: vec2<f32>) -> FragmentOutput {
    // The Y-flip below is specifically because baseMap is a previous
    // post-stage's render-target output (see TonemapShader, which
    // established this same flip for the same reason). depthTex/
    // normalTex need no flip HERE because FluidDepthBlur.wgsl already
    // corrects for the same underlying issue once, on read, right
    // where the raw (render-target-sourced) capture first enters our
    // own compute pipeline — see loadDepth() there.
    var flippedUV = fragUV;
    flippedUV.y = 1.0 - flippedUV.y;

    let sceneColor = textureSample(baseMap, baseMapSampler, flippedUV);
    let size = textureDimensions(depthTex);
    let coord = vec2<i32>(fragUV * vec2<f32>(size));
    let depth = textureLoad(depthTex, coord, 0).r;

    if (depth <= 0.0) {
        return FragmentOutput(sceneColor);
    }

    return FragmentOutput(vec4<f32>(shadeSurface(coord, flippedUV), 1.0));
}
