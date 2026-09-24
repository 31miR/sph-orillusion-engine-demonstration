#include "FluidParticleData"

@group(0) @binding(0)
var<storage, read> particles: array<FluidParticle>;

@group(0) @binding(1)
var<storage, read_write> maxVelocityBits: array<atomic<u32>>;

// Reduces this frame's particle velocities down to a single "fastest
// particle" speed, written as raw bits rather than a plain float so
// every thread can safely atomicMax it — WGSL has no atomic max for
// floats directly, but since speed (a vector length) is always >= 0,
// comparing its bit pattern as an unsigned integer gives the exact same
// ordering as comparing the floats themselves (an IEEE 754 property).
// Read back by computeDt() in FluidSimParams.wgsl, used by every shader
// that needs a CFL-safe step size (STAR report, p.3): pressure force,
// viscosity force, and integrate.
//
// FluidMaxVelocityClear.wgsl must run before this each frame (see
// FluidSimulator.ts).
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    let speed = length(particles[i].velocity.xyz);
    atomicMax(&maxVelocityBits[0], bitcast<u32>(speed));
}
