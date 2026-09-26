// Resets last frame's fastest-particle speed to 0 before
// FluidMaxVelocityReduce.wgsl re-measures it, or it could only ever
// grow. Separate file, not a shared entry point: this shader doesn't
// touch the particle buffer, so the bindings don't match.
@group(0) @binding(0)
var<storage, read_write> maxVelocityBits: array<atomic<u32>>;

@compute @workgroup_size(1)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    atomicStore(&maxVelocityBits[0], 0u);
}
