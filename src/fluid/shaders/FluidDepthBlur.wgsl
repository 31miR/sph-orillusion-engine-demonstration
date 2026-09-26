// Bilateral depth smoothing, run several times over ping-ponged
// textures (see FluidDepthSmoothPass.ts) — per van der Laan, Green,
// Sainz 2009 Sec 3.5.1, ~6 iterations of a bilateral Gaussian filter
// reaches similar quality to the full curvature-flow PDE (40-60
// iterations), and isn't equivalent to one pass with a wider kernel.
//
// Two Gaussian weights multiplied together: one falls off with pixel
// distance (spatial), one with depth difference (range), so the blur
// doesn't smear across the fluid/background silhouette edge.

@group(0) @binding(0)
var depthTex: texture_2d<f32>;

@group(0) @binding(1)
var outTex: texture_storage_2d<r32float, write>;

const KERNEL_RADIUS: i32 = 5;
const SIGMA_SPACE: f32 = 3.0;
// Depth is view-space distance-to-camera in world units (see
// FluidDepthCaptureShader.wgsl). Deliberately a fixed constant, not
// derived from particle size: a fixed absolute width becomes
// relatively wider as particles shrink at higher counts, so denser
// configurations end up more smoothed automatically — scaling it down
// with particle size would cancel that out.
const SIGMA_RANGE: f32 = 0.3;
// A real particle's distance is always > 0; 0 is FluidDepthPass's
// clear color, unambiguously "no particle".
const EMPTY_DEPTH: f32 = 0.0;

// Only the first iteration reads FluidDepthPass's raw render-target
// output, which stores rows in the opposite order from a compute
// shader's own pixel-index addressing (same reason
// FluidShadeComposite.wgsl's baseMap needs a Y-flip). Later iterations
// read a texture written by this same shader, already top-down.
fn loadDepth(coord: vec2<i32>, size: vec2<u32>, flipRead: bool) -> f32 {
    var c = coord;
    if (flipRead) {
        c = vec2<i32>(coord.x, i32(size.y) - 1 - coord.y);
    }
    return textureLoad(depthTex, c, 0).r;
}

fn blur(id: vec3<u32>, flipRead: bool) {
    let size = textureDimensions(depthTex);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let centerDepth = loadDepth(coord, size, flipRead);

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
            let sampleDepth = loadDepth(sampleCoord, size, flipRead);
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

@compute @workgroup_size(8, 8)
fn CsMainFirst(@builtin(global_invocation_id) id: vec3<u32>) {
    blur(id, true);
}

@compute @workgroup_size(8, 8)
fn CsMain(@builtin(global_invocation_id) id: vec3<u32>) {
    blur(id, false);
}
