#include "FluidParticleData"
#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<FluidParticle>;

// Equation of state (Eq. 9, STAR report): p = k * ((rho/rho_0)^7 - 1).
// Only depends on this particle's own density (already computed by
// FluidDensityCompute into velocity.w) — no neighbor search needed.
// Left unclamped, including negative values (rho < rho_0, e.g. near a
// free surface/edge): a negative pressure gradient pulling fluid toward
// an under-dense region is physically correct behavior, not an error.
// Whether the eventual pairwise pressure-force formula (Eq. 6, not yet
// implemented) needs to clamp negative pressure to avoid an attractive-force
// instability between close particles ("tensile instability" in the SPH
// literature) is a decision for that step, with the actual force code in
// front of us — not something to bake into the computation here.
// Result is stashed in position.w, kept separate from density
// (velocity.w) since the pressure-force step will need both at once.
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
