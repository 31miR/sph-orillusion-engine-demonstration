import { MeshRenderer, Object3D, PassType, SphereGeometry, StorageGPUBuffer } from "@orillusion/core";
import { FluidParticleMaterial } from "./FluidParticleMaterial";

// Must match the `FluidParticle` WGSL struct in shaders/FluidParticleData.ts
// (currently just a vec4<f32> position).
const FLOATS_PER_PARTICLE = 4;

export class FluidParticleField {
    public readonly object3D: Object3D;
    public readonly buffer: StorageGPUBuffer;
    public readonly particleCount: number;

    constructor(particlesPerAxis: number = 8, spacing: number = 0.3, particleRadius: number = 0.12) {
        this.particleCount = particlesPerAxis * particlesPerAxis * particlesPerAxis;

        const data = new Float32Array(this.particleCount * FLOATS_PER_PARTICLE);
        const offset = (particlesPerAxis - 1) * 0.5;
        let i = 0;
        for (let x = 0; x < particlesPerAxis; x++) {
            for (let y = 0; y < particlesPerAxis; y++) {
                for (let z = 0; z < particlesPerAxis; z++) {
                    data[i++] = (x - offset) * spacing;
                    data[i++] = (y - offset) * spacing;
                    data[i++] = (z - offset) * spacing;
                    data[i++] = 0;
                }
            }
        }

        this.buffer = new StorageGPUBuffer(this.particleCount * FLOATS_PER_PARTICLE, 0, data);

        this.object3D = new Object3D();
        this.object3D.name = "FluidParticles";

        const mr = this.object3D.addComponent(MeshRenderer);
        mr.geometry = new SphereGeometry(particleRadius, 8, 6);
        mr.material = new FluidParticleMaterial();
        mr.instanceCount = this.particleCount;

        mr.material.getPass(PassType.COLOR)[0]!.setStorageBuffer("particles", this.buffer);
    }
}
