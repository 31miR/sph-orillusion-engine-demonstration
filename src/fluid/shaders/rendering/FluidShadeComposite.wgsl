// Step 4 of the screen-space fluid renderer (van der Laan et al. 2009):
// shade the reconstructed surface (diffuse + Fresnel + refracted
// background) and composite it over the scene. A ViewQuad fragment
// pass, same shape as TonemapShader/FXAA.
//
// Light direction and reflection (SKY_TINT) are still placeholders,
// not wired to the scene's real DirectLight or an environment map.

@group(1) @binding(0)
var baseMapSampler: sampler;

@group(1) @binding(1)
var baseMap: texture_2d<f32>;

// f0 (Fresnel reflectance at normal incidence) lives in a uniform,
// not a bare constant — an empty group 2 triggers a "can't set empty
// group! 2" reflection error, since every ViewQuad shader here is
// expected to declare something there.
struct MaterialUniform {
    f0: f32,
};

@group(2) @binding(0)
var<uniform> materialUniform: MaterialUniform;

@group(3) @binding(0)
var depthTex: texture_2d<f32>;

@group(3) @binding(1)
var normalTex: texture_2d<f32>;

// The real scene's depth (Orillusion's _MainDepthTexture) — used to
// test whether something opaque occludes the fluid at this pixel.
// Fetched informally each frame in FluidShadeCompositePost.ts (not a
// formal b.read: a pass inside the engine's built-in post chain can't
// formally declare graph edges against passes the engine already
// wired up before this project's own code runs).
@group(3) @binding(2)
var sceneDepthTex: texture_depth_2d;

struct ReconstructParams {
    projMatInv: mat4x4<f32>,
};

@group(3) @binding(3)
var<uniform> reconstructParams: ReconstructParams;

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

    // Bend the sampled background by the surface's tilt — a stand-in
    // for tracing a real refraction ray. Subtracted, not added: per
    // Snell's law, a ray bends *opposite* the direction the normal
    // tilts, not the same way.
    let refractedUV = clamp(flippedUV - normal.xy * REFRACTION_STRENGTH, vec2<f32>(0.0), vec2<f32>(1.0));
    // textureSampleLevel, not textureSample: this branch doesn't run
    // for every pixel, and textureSample's implicit mip derivatives
    // require uniform control flow.
    let refracted = textureSampleLevel(baseMap, baseMapSampler, refractedUV, 0.0).rgb;
    let lit = refracted * (0.25 + 0.75 * ndotl);
    let transmission = mix(lit, WATER_COLOR, WATER_TINT_STRENGTH);

    return mix(transmission, SKY_TINT, fresnel);
}

// Recovers linear view-space Z from a real NDC depth sample (same
// convention as FluidDepthCaptureShader.wgsl's "distanceToCamera"),
// so it's directly comparable against the fluid's own distance.
fn linearizeSceneDepth(ndcDepth: f32, flippedUV: vec2<f32>) -> f32 {
    let ndcXY = vec2<f32>(flippedUV.x * 2.0 - 1.0, 1.0 - flippedUV.y * 2.0);
    let clipPos = vec4<f32>(ndcXY, ndcDepth, 1.0);
    let viewPos = reconstructParams.projMatInv * clipPos;
    return viewPos.z / viewPos.w;
}

@fragment
fn main(@location(auto) fragUV: vec2<f32>) -> FragmentOutput {
    // baseMap/sceneDepthTex are raw render targets and need this flip
    // (see TonemapShader); depthTex/normalTex don't — FluidDepthBlur.wgsl
    // already corrects for it once, on read.
    var flippedUV = fragUV;
    flippedUV.y = 1.0 - flippedUV.y;

    let sceneColor = textureSample(baseMap, baseMapSampler, flippedUV);
    let size = textureDimensions(depthTex);
    let coord = vec2<i32>(fragUV * vec2<f32>(size));
    let depth = textureLoad(depthTex, coord, 0).r;

    if (depth <= 0.0) {
        return FragmentOutput(sceneColor);
    }

    let sceneSize = vec2<i32>(textureDimensions(sceneDepthTex));
    let scenePixel = clamp(vec2<i32>(flippedUV * vec2<f32>(sceneSize)), vec2<i32>(0), sceneSize - vec2<i32>(1));
    let sceneDepthNDC = textureLoad(sceneDepthTex, scenePixel, 0);
    let sceneViewZ = linearizeSceneDepth(sceneDepthNDC, flippedUV);
    if (sceneViewZ < depth) {
        // Something opaque occludes the fluid here.
        return FragmentOutput(sceneColor);
    }

    return FragmentOutput(vec4<f32>(shadeSurface(coord, flippedUV), 1.0));
}
