// Resets last frame's fastest-particle speed back to 0 before
// FluidMaxVelocityReduce.wgsl re-measures it this frame — otherwise it
// could only ever grow, never reflect the fluid calming back down. A
// separate file from the reduce shader (not two entry points sharing
// one file, unlike e.g. FluidDepthBlur.wgsl's CsMainFirst/CsMain):
// this shader doesn't touch the particle buffer at all, while the
// reduce shader does, so the two don't share the same bindings the
// way FluidDepthBlur's two entry points do.
@group(0) @binding(0)
var<storage, read_write> maxVelocityBits: array<atomic<u32>>;

@compute @workgroup_size(1)
fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
    atomicStore(&maxVelocityBits[0], 0u);
}
