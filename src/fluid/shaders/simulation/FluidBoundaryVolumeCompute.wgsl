#include "FluidKernel"

struct BoundaryVolumeParams {
    smoothingLength: f32,
};

@group(0) @binding(0)
var<uniform> params: BoundaryVolumeParams;

// xyz = static wall position, w = Akinci volume (written here).
@group(0) @binding(1)
var<storage, read_write> boundaryPositions: array<vec4<f32>>;

// Akinci et al. 2012: V_b = 1 / sum_b' W(x_b - x_b', h) — each boundary
// particle's volume from how densely the *other* boundary particles
// pile onto it. Densely-sampled regions get a smaller volume per
// particle; sparse regions get a bigger one, so the wall's total mass
// contribution stays consistent regardless of exact sampling density.
// Run once at startup — the walls never move.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    let count = arrayLength(&boundaryPositions);
    if (i >= count) {
        return;
    }

    let posI = boundaryPositions[i].xyz;
    var weightSum = 0.0;
    for (var j = 0u; j < count; j = j + 1u) {
        let r = length(posI - boundaryPositions[j].xyz);
        weightSum = weightSum + cubicSplineWeight(r, params.smoothingLength);
    }

    boundaryPositions[i].w = 1.0 / max(weightSum, 1e-6);
}
