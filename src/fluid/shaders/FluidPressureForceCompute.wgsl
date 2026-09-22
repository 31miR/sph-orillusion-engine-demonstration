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

// Symmetric pressure force (Eq. 6 applied to pressure), the standard
// SPH formulation that guarantees equal-and-opposite forces between
// every pair (Newton's third law) regardless of any density asymmetry:
//
//   a_i = -sum_j m_j * (p_i/rho_i^2 + p_j/rho_j^2) * grad_i W_ij
//
// Expressed as an acceleration (not a raw force) so it adds directly
// into velocity, the same way gravity already does — no extra mass
// division needed at integration time.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let posI = particles[i].position.xyz;
    let densityI = particles[i].velocity.w;
    let pressureI = particles[i].position.w;
    let baseCoord = simCellCoord(params, posI);
    let gridDim = vec3<i32>(i32(params.gridDimX), i32(params.gridDimY), i32(params.gridDimZ));

    var accel = vec3<f32>(0.0, 0.0, 0.0);

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
                        let posJ = particles[u32(j)].position.xyz;
                        let rij = posI - posJ;
                        let r = length(rij);

                        // Guard r ~ 0: direction (rij/r) is undefined
                        // there regardless of dW/dr already being 0 for
                        // a smooth kernel like ours.
                        if (r > 1e-6) {
                            let densityJ = particles[u32(j)].velocity.w;
                            let pressureJ = particles[u32(j)].position.w;
                            let dWdr = cubicSplineDerivative(r, params.smoothingLength);
                            let gradW = (dWdr / r) * rij;
                            let coeff = pressureI / (densityI * densityI) + pressureJ / (densityJ * densityJ);
                            accel = accel - params.particleMass * coeff * gradW;
                        }
                    }
                    j = particleNext[u32(j)];
                }
            }
        }
    }

    particles[i].velocity = vec4<f32>(particles[i].velocity.xyz + accel * params.deltaTime, particles[i].velocity.w);
}
