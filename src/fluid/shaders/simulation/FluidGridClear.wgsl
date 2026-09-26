#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> cellHead: array<atomic<i32>>;

@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let cellCount = u32(params.gridDimX * params.gridDimY * params.gridDimZ);
    let i = globalId.x;
    if (i >= cellCount) {
        return;
    }
    atomicStore(&cellHead[i], -1);
}
