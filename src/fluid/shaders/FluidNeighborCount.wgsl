#include "FluidParticleData"
#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<FluidParticle>;

@group(0) @binding(2)
var<storage, read> cellHead: array<i32>;

@group(0) @binding(3)
var<storage, read> particleNext: array<i32>;

// Debug-only pass (milestone step 4): counts *other* particles within
// kernel support and stashes the count in the otherwise-unused
// velocity.w slot, purely so the render shader can visualize it as a
// heatmap and let us sanity-check the grid/neighbor search in isolation,
// before density/pressure/force are built on top of it.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let pos = particles[i].position.xyz;
    let supportRadius = 2.0 * params.smoothingLength;
    let supportRadiusSq = supportRadius * supportRadius;
    let baseCoord = simCellCoord(params, pos);
    let gridDim = vec3<i32>(i32(params.gridDimX), i32(params.gridDimY), i32(params.gridDimZ));

    var count = 0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighborCoord = baseCoord + vec3<i32>(dx, dy, dz);
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= gridDim)) {
                    continue;
                }

                var j = cellHead[simCellIndex(params, neighborCoord)];
                while (j >= 0) {
                    if (u32(j) != i) {
                        let diff = particles[u32(j)].position.xyz - pos;
                        if (dot(diff, diff) < supportRadiusSq) {
                            count = count + 1;
                        }
                    }
                    j = particleNext[u32(j)];
                }
            }
        }
    }

    particles[i].velocity.w = f32(count);
}
