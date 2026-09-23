import { MeshRenderer, Object3D, PassType, SphereGeometry, StorageGPUBuffer, VisibleLayer } from "@orillusion/core";
import { FluidParticleMaterial } from "./FluidParticleMaterial";
import { FluidDepthCaptureMaterial } from "./FluidDepthCaptureMaterial";

// Must match the `FluidParticle` WGSL struct in shaders/FluidParticleData.ts:
// position (vec4<f32>) + velocity (vec4<f32>).
const FLOATS_PER_PARTICLE = 8;

// Arbitrary, unused-by-the-engine RendererMask bit (built-in values in
// RendererMask.d.ts top out at 1024) — tags only the depth-capture
// echo below so SceneCaptureCameraComponent.captureMask can pick it
// out and ignore everything else (floor, sky, the normal spheres).
export const FLUID_DEPTH_CAPTURE_MASK = 1 << 11;

export class FluidParticleField {
    public readonly object3D: Object3D;
    // Second, screen-invisible instance of the same particles, used
    // only so SceneCaptureCameraComponent can capture their distance
    // from the camera (see FluidDepthCaptureMaterial) without that
    // showing up in the normal on-screen picture.
    public readonly depthCaptureObject3D: Object3D;
    public readonly buffer: StorageGPUBuffer;
    public readonly particleCount: number;
    public readonly particleRadius: number;
    public readonly spacing: number;

    constructor(particlesPerAxis: number = 8, spacing: number = 0.3, particleRadius: number = 0.12) {
        this.particleCount = particlesPerAxis * particlesPerAxis * particlesPerAxis;
        this.particleRadius = particleRadius;
        this.spacing = spacing;

        const data = new Float32Array(this.particleCount * FLOATS_PER_PARTICLE);
        const offset = (particlesPerAxis - 1) * 0.5;
        let i = 0;
        for (let x = 0; x < particlesPerAxis; x++) {
            for (let y = 0; y < particlesPerAxis; y++) {
                for (let z = 0; z < particlesPerAxis; z++) {
                    data[i++] = (x - offset) * spacing;
                    data[i++] = (y - offset) * spacing;
                    data[i++] = (z - offset) * spacing;
                    data[i++] = 0; // position.w (unused)
                    data[i++] = 0; // velocity.x
                    data[i++] = 0; // velocity.y
                    data[i++] = 0; // velocity.z
                    data[i++] = 0; // velocity.w (debug visualization scalar)
                }
            }
        }

        this.buffer = new StorageGPUBuffer(this.particleCount * FLOATS_PER_PARTICLE, 0, data);

        this.object3D = new Object3D();
        this.object3D.name = "FluidParticles";

        const geometry = new SphereGeometry(particleRadius, 8, 6);

        const mr = this.object3D.addComponent(MeshRenderer);
        mr.geometry = geometry;
        mr.material = new FluidParticleMaterial();
        mr.instanceCount = this.particleCount;
        mr.material.getPass(PassType.COLOR)[0]!.setStorageBuffer("particles", this.buffer);

        this.depthCaptureObject3D = new Object3D();
        this.depthCaptureObject3D.name = "FluidParticlesDepthCapture";

        const depthMr = this.depthCaptureObject3D.addComponent(MeshRenderer);
        depthMr.geometry = geometry;
        depthMr.material = new FluidDepthCaptureMaterial();
        depthMr.instanceCount = this.particleCount;
        // VisibleLayer.None hides this from every ordinary camera/pass
        // (visibleLayer is ANDed with layerMask/cullingMask, so 0 never
        // matches anything) — SceneCapturePass filters by rendererMask
        // instead, a wholly separate system, so this stays capturable
        // while being invisible on screen.
        depthMr.visibleLayer = VisibleLayer.None;
        depthMr.rendererMask = FLUID_DEPTH_CAPTURE_MASK;
        depthMr.material.getPass(PassType.COLOR)[0]!.setStorageBuffer("particles", this.buffer);
    }
}
