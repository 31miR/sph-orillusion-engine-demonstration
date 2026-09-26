#include "FluidParticleData"

@group(0) @binding(0)
var<storage, read> particles: array<FluidParticle>;

@group(0) @binding(1)
var<storage, read_write> maxVelocityBits: array<atomic<u32>>;

// Reduces this frame's velocities to a single "fastest particle"
// speed, written as raw bits so every thread can atomicMax it — WGSL
// has no atomic max for floats, but since speed is always >= 0,
// comparing bit patterns as unsigned ints gives the same ordering
// (IEEE 754). Read back by computeDt() in FluidSimParams.wgsl.
// FluidMaxVelocityClear.wgsl must run before this each frame.
@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }
    let speed = length(particles[i].velocity.xyz);
    atomicMax(&maxVelocityBits[0], bitcast<u32>(speed));
}
