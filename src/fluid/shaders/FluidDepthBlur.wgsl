// Bilateral depth smoothing (single pass, small kernel) — the simpler
// alternative to full curvature-flow PDE smoothing, per van der Laan,
// Green, Sainz 2009 Sec 3.2 (p.93, "An obvious approach is use a
// Gaussian blur or variants such as Bilateral Gaussian filters") and
// STAR report Sec 7.3 (p.15, citing [Gre10] for the Gaussian variant).
// Two Gaussian weights multiplied together: one falls off with pixel
// distance (spatial), one falls off with depth difference (range) so
// the blur doesn't smear across the fluid/background silhouette edge.
//
// This first version is a single full 2D kernel, not yet the
// separable two-pass + multi-iteration form the paper benchmarks
// against curvature flow — that's a later refinement once this basic
// mechanism (reading the depth target, writing a smoothed copy) is
// confirmed working.

@group(0) @binding(0)
var depthTex: texture_2d<f32>;

@group(0) @binding(1)
var outTex: texture_storage_2d<r32float, write>;

const KERNEL_RADIUS: i32 = 4;
const SIGMA_SPACE: f32 = 3.0;
// Depth here is view-space distance-to-camera in world units (see
// FluidDepthCaptureShader.wgsl), not normalized depth — tune once
// visible in-browser.
const SIGMA_RANGE: f32 = 0.1;
// A real particle's distance is always > 0 (anything at the camera
// itself is already clipped) — 0 unambiguously means "no particle",
// since that's what SceneCaptureCameraComponent's clear color leaves
// untouched background pixels holding.
const EMPTY_DEPTH: f32 = 0.0;

@compute @workgroup_size(8, 8)
fn CsMain(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(depthTex);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let centerDepth = textureLoad(depthTex, coord, 0).r;

    // Nothing was drawn here (background) — pass through unchanged,
    // don't let empty-background depth pollute a neighboring pixel's
    // weighted average.
    if (centerDepth <= EMPTY_DEPTH) {
        textureStore(outTex, coord, vec4<f32>(centerDepth, 0.0, 0.0, 0.0));
        return;
    }

    var sumWeight = 0.0;
    var sumDepth = 0.0;
    for (var dy = -KERNEL_RADIUS; dy <= KERNEL_RADIUS; dy = dy + 1) {
        for (var dx = -KERNEL_RADIUS; dx <= KERNEL_RADIUS; dx = dx + 1) {
            let sampleCoord = coord + vec2<i32>(dx, dy);
            if (sampleCoord.x < 0 || sampleCoord.y < 0 || sampleCoord.x >= i32(size.x) || sampleCoord.y >= i32(size.y)) {
                continue;
            }
            let sampleDepth = textureLoad(depthTex, sampleCoord, 0).r;
            if (sampleDepth <= EMPTY_DEPTH) {
                continue;
            }
            let spatialDist2 = f32(dx * dx + dy * dy);
            let rangeDist = sampleDepth - centerDepth;
            let weight = exp(-spatialDist2 / (2.0 * SIGMA_SPACE * SIGMA_SPACE))
                * exp(-(rangeDist * rangeDist) / (2.0 * SIGMA_RANGE * SIGMA_RANGE));
            sumWeight = sumWeight + weight;
            sumDepth = sumDepth + weight * sampleDepth;
        }
    }

    var result = centerDepth;
    if (sumWeight > 0.0) {
        result = sumDepth / sumWeight;
    }
    textureStore(outTex, coord, vec4<f32>(result, 0.0, 0.0, 0.0));
}
