#include "FluidParticleData"
#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<FluidParticle>;

// Equation of state (Eq. 9, STAR report): p = k * ((rho/rho_0)^7 - 1).
// Only depends on this particle's own density (velocity.w) — no
// neighbor search needed. Left unclamped, including negative values
// (rho < rho_0): a pressure gradient pulling fluid toward an
// under-dense region is physically correct, not an error.
// Result is stashed in position.w, separate from density, since the
// pressure-force step needs both at once.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    let density = particles[i].velocity.w;
    let ratio = density / params.restDensity;
    let pressure = params.stiffness * (pow(ratio, 7.0) - 1.0);

    particles[i].position.w = pressure;
}
