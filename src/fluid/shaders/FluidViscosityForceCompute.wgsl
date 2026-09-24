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

@group(0) @binding(4)
var<storage, read> maxVelocityBits: array<u32>;

// Viscosity via the Laplacian estimator (Eq. 8, STAR report), applied to
// velocity and scaled by the kinematic viscosity coefficient:
//
//   a_i = nu * 2 * sum_j (m_j/rho_j) * (v_i-v_j) * (x_ij . grad_i W_ij)
//                          / (x_ij . x_ij + 0.01 h^2)
//
// x_ij . grad_i W_ij simplifies to r * dW/dr exactly, since
// grad_i W_ij = (dW/dr) * (x_ij / r) — so no separate dot product is
// needed, just the same scalar derivative the pressure force already
// uses. Unlike that force, no r > 0 guard is needed here: at r = 0 the
// numerator (x_ij . grad_i W_ij = r * dW/dr) is already 0, and the
// +0.01h^2 regularization keeps the denominator safely nonzero — this
// is exactly what that regularization term is for, and it matters for
// us given how often particles end up nearly coincident.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let posI = particles[i].position.xyz;
    let velI = particles[i].velocity.xyz;
    let baseCoord = simCellCoord(params, posI);
    let gridDim = vec3<i32>(i32(params.gridDimX), i32(params.gridDimY), i32(params.gridDimZ));
    let epsSq = 0.01 * params.smoothingLength * params.smoothingLength;

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
                        let velJ = particles[u32(j)].velocity.xyz;
                        let densityJ = particles[u32(j)].velocity.w;
                        let r = length(posI - posJ);
                        let dWdr = cubicSplineDerivative(r, params.smoothingLength);

                        let numerator = r * dWdr;
                        let denom = r * r + epsSq;
                        accel = accel + (params.particleMass / densityJ) * (velI - velJ) * (numerator / denom);
                    }
                    j = particleNext[u32(j)];
                }
            }
        }
    }

    accel = accel * 2.0 * params.viscosity;

    let dt = computeDt(params.deltaTime, maxVelocityBits[0], 2.0 * params.particleRadius);
    particles[i].velocity = vec4<f32>(velI + accel * dt, particles[i].velocity.w);
}
