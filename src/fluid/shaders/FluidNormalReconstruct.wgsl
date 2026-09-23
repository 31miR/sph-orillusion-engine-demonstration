// Step 3 of the screen-space fluid renderer: reconstruct a view-space
// surface normal at each pixel from the smoothed distance-to-camera
// texture (FluidDepthSmoothPass's output) — no 3D mesh is ever built,
// just per-pixel math on the depth image (STAR report Sec 7.3;
// van der Laan et al. 2009 describes the same reconstruct-then-shade
// structure). The unprojection formula (linear view-z depth -> view
// position, via the inverse projection matrix) mirrors Orillusion's
// own internal reconstructViewPosFromLinearDepth (used by GTAO/SSR),
// verified by reading it directly — reimplemented against our own
// uniform data rather than depending on the engine's internal
// GlobalUniform bind group, which a standalone ComputeShader (outside
// the normal per-object rendering path) has no way to reach.

struct ReconstructParams {
    projMatInv: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> params: ReconstructParams;

@group(0) @binding(1)
var depthTex: texture_2d<f32>;

@group(0) @binding(2)
var outTex: texture_storage_2d<rgba16float, write>;

fn sampleDepth(coord: vec2<i32>, size: vec2<i32>) -> f32 {
    let c = clamp(coord, vec2<i32>(0), size - vec2<i32>(1));
    return textureLoad(depthTex, c, 0).r;
}

// texUV is a texture-style UV (v=0 at the top row). The engine's own
// reconstruction flips to NDC's v=0-at-bottom convention before
// building the clip-space point — matched here for the same reason.
fn reconstructViewPos(linearDepth: f32, texUV: vec2<f32>) -> vec3<f32> {
    var sampleUV = texUV;
    sampleUV.y = 1.0 - sampleUV.y;
    let clipNear = vec4<f32>((sampleUV * 2.0 - 1.0), 0.0, 1.0);
    let viewNear = params.projMatInv * clipNear;
    let viewRay = viewNear.xyz / viewNear.w;
    return viewRay * (linearDepth / viewRay.z);
}

@compute @workgroup_size(8, 8)
fn CsMain(@builtin(global_invocation_id) id: vec3<u32>) {
    let sizeU = textureDimensions(depthTex);
    let size = vec2<i32>(sizeU);
    if (id.x >= sizeU.x || id.y >= sizeU.y) {
        return;
    }
    let coord = vec2<i32>(id.xy);
    let centerDepth = textureLoad(depthTex, coord, 0).r;

    if (centerDepth <= 0.0) {
        textureStore(outTex, coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    let leftDepth = sampleDepth(coord - vec2<i32>(1, 0), size);
    let rightDepth = sampleDepth(coord + vec2<i32>(1, 0), size);
    let upDepth = sampleDepth(coord - vec2<i32>(0, 1), size);
    let downDepth = sampleDepth(coord + vec2<i32>(0, 1), size);

    // Prefer whichever side of each axis isn't background (or is
    // closer in depth), so a tangent vector never jumps across the
    // fluid's silhouette edge onto empty space.
    var dxCoord = coord + vec2<i32>(1, 0);
    var dxDepth = rightDepth;
    if (rightDepth <= 0.0 || (leftDepth > 0.0 && abs(leftDepth - centerDepth) < abs(rightDepth - centerDepth))) {
        dxCoord = coord - vec2<i32>(1, 0);
        dxDepth = leftDepth;
    }

    var dyCoord = coord + vec2<i32>(0, 1);
    var dyDepth = downDepth;
    if (downDepth <= 0.0 || (upDepth > 0.0 && abs(upDepth - centerDepth) < abs(downDepth - centerDepth))) {
        dyCoord = coord - vec2<i32>(0, 1);
        dyDepth = upDepth;
    }

    if (dxDepth <= 0.0 || dyDepth <= 0.0) {
        // An isolated pixel with no valid neighbor on one axis (e.g. a
        // single-pixel sliver) — face the camera rather than produce a
        // garbage normal from a background neighbor.
        textureStore(outTex, coord, vec4<f32>(0.5, 0.5, 1.0, 1.0));
        return;
    }

    let sizeF = vec2<f32>(size);
    let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / sizeF;
    let uvDx = (vec2<f32>(dxCoord) + vec2<f32>(0.5)) / sizeF;
    let uvDy = (vec2<f32>(dyCoord) + vec2<f32>(0.5)) / sizeF;

    let posCenter = reconstructViewPos(centerDepth, uv);
    let posDx = reconstructViewPos(dxDepth, uvDx);
    let posDy = reconstructViewPos(dyDepth, uvDy);

    let tangentX = posDx - posCenter;
    let tangentY = posDy - posCenter;
    var normal = normalize(cross(tangentX, tangentY));
    // View space looks down -Z; a normal facing back toward the camera
    // should have a positive Z component. Which cross-product operand
    // order that requires can flip per-pixel depending on which side
    // (left/right, up/down) got picked above, so enforce it explicitly
    // instead of relying on a fixed operand order.
    if (dot(normal, vec3<f32>(0.0, 0.0, 1.0)) < 0.0) {
        normal = -normal;
    }

    // Stored in the conventional [0,1]-remapped "normal map" encoding;
    // the shading step unpacks with * 2.0 - 1.0.
    textureStore(outTex, coord, vec4<f32>(normal * 0.5 + 0.5, 1.0));
}
