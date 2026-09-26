#include "FluidParticleData"
#include "FluidSimParams"
#include "FluidKernel"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<FluidParticle>;

@group(0) @binding(2)
var<storage, read> cellHead: array<i32>;

@group(0) @binding(3)
var<storage, read> particleNext: array<i32>;

// Density summation, Eq. 3 in the STAR report: rho_i = sum_j m_j * W_ij.
// Unlike the earlier neighbor-count debug pass, this sum DOES include
// j == i — the kernel's self-weight W(0, h) is a real, nonzero term in
// the density estimate, not a degenerate case to skip. Result is stashed
// in velocity.w (same debug slot the neighbor-count pass used), so the
// render shader can visualize it.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let pos = particles[i].position.xyz;
    let baseCoord = simCellCoord(params, pos);
    let gridDim = vec3<i32>(i32(params.gridDimX), i32(params.gridDimY), i32(params.gridDimZ));

    var density = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let neighborCoord = baseCoord + vec3<i32>(dx, dy, dz);
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= gridDim)) {
                    continue;
                }

                var j = cellHead[simCellIndex(params, neighborCoord)];
                while (j >= 0) {
                    let diff = particles[u32(j)].position.xyz - pos;
                    let r = length(diff);
                    density = density + params.particleMass * cubicSplineWeight(r, params.smoothingLength);
                    j = particleNext[u32(j)];
                }
            }
        }
    }

    particles[i].velocity.w = density;
}
