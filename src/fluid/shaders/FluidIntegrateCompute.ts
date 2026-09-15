export const FluidIntegrateCompute = /* wgsl */ `
    #include "FluidParticleData"

    struct SimParams {
        deltaTime: f32,
        gravity: f32,
    };

    @group(0) @binding(0)
    var<uniform> params: SimParams;

    @group(0) @binding(1)
    var<storage, read_write> particles: array<FluidParticle>;

    @compute @workgroup_size(64)
    fn CsMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let i = globalId.x;
        if (i >= arrayLength(&particles)) {
            return;
        }

        var particle = particles[i];

        // Semi-implicit (symplectic) Euler: update velocity first, then
        // use the *new* velocity to advance position (Algorithm 1 in the
        // STAR report). No boundaries yet — particles fall straight
        // through the floor, which is the expected result for this step.
        particle.velocity.y = particle.velocity.y - params.gravity * params.deltaTime;
        particle.position = vec4<f32>(particle.position.xyz + particle.velocity.xyz * params.deltaTime, particle.position.w);

        particles[i] = particle;
    }
`;
