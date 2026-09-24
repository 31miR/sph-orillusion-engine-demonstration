#include "FluidParticleData"
#include "FluidSimParams"

@group(0) @binding(0)
var<uniform> params: SimParams;

@group(0) @binding(1)
var<storage, read_write> particles: array<FluidParticle>;

@group(0) @binding(2)
var<storage, read> maxVelocityBits: array<u32>;

@compute @workgroup_size(64)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    let i = globalId.x;
    if (i >= arrayLength(&particles)) {
        return;
    }

    var particle = particles[i];
    let dt = computeDt(params.deltaTime, maxVelocityBits[0], 2.0 * params.particleRadius);

    // Semi-implicit (symplectic) Euler: update velocity first, then
    // use the *new* velocity to advance position (Algorithm 1 in the
    // STAR report).
    particle.velocity.y = particle.velocity.y - params.gravity * dt;
    particle.position = vec4<f32>(particle.position.xyz + particle.velocity.xyz * dt, particle.position.w);

    // Box boundary: clamp the particle's center so its surface doesn't
    // pass the wall, and reflect+damp the velocity component along that
    // axis (restitution 0 = stick, 1 = perfectly elastic bounce).
    let r = params.particleRadius;
    if (particle.position.x < params.boundsMinX + r) {
        particle.position.x = params.boundsMinX + r;
        particle.velocity.x = -particle.velocity.x * params.restitution;
    } else if (particle.position.x > params.boundsMaxX - r) {
        particle.position.x = params.boundsMaxX - r;
        particle.velocity.x = -particle.velocity.x * params.restitution;
    }

    if (particle.position.y < params.boundsMinY + r) {
        particle.position.y = params.boundsMinY + r;
        particle.velocity.y = -particle.velocity.y * params.restitution;
    } else if (particle.position.y > params.boundsMaxY - r) {
        particle.position.y = params.boundsMaxY - r;
        particle.velocity.y = -particle.velocity.y * params.restitution;
    }

    if (particle.position.z < params.boundsMinZ + r) {
        particle.position.z = params.boundsMinZ + r;
        particle.velocity.z = -particle.velocity.z * params.restitution;
    } else if (particle.position.z > params.boundsMaxZ - r) {
        particle.position.z = params.boundsMaxZ - r;
        particle.velocity.z = -particle.velocity.z * params.restitution;
    }

    particles[i] = particle;
}
