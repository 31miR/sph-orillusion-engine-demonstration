#include "FluidParticleData"
#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read> particles: array<FluidParticle>;

@group(0) @binding(2)
var<storage, read_write> cellHead: array<atomic<i32>>;

@group(0) @binding(3)
var<storage, read_write> particleNext: array<i32>;

// Standard parallel "atomic head + next pointer" technique: each particle
// atomically swaps itself in as its cell's new head, recording whatever
// was there before as its own "next" — building a singly linked list per
// cell without needing a sort or prefix sum.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let coord = simCellCoord(params, particles[i].position.xyz);
    let cellIndex = simCellIndex(params, coord);

    let previousHead = atomicExchange(&cellHead[cellIndex], i32(i));
    particleNext[i] = previousHead;
}
