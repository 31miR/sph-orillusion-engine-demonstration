#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read> boundaryPositions: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> boundaryCellHead: array<atomic<i32>>;

@group(0) @binding(3)
var<storage, read_write> boundaryNext: array<i32>;

// Same atomic head/next technique as FluidGridBuild.wgsl, sharing the
// fluid grid's cell size and dimensions — but run once at startup, not
// per frame: the walls are static, so this grid never needs rebuilding.
// boundaryCellHead is pre-filled with -1 at buffer creation, so there's
// no separate "clear" dispatch either.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&boundaryPositions)) {
        return;
    }

    let coord = simCellCoord(params, boundaryPositions[i].xyz);
    let cellIndex = simCellIndex(params, coord);

    let previousHead = atomicExchange(&boundaryCellHead[cellIndex], i32(i));
    boundaryNext[i] = previousHead;
}
