import { MeshRenderer, Object3D, PassType, SphereGeometry, StorageGPUBuffer, VisibleLayer } from "@orillusion/core";
import { FluidParticleMaterial } from "./FluidParticleMaterial";
import { FluidDepthCaptureMaterial } from "./FluidDepthCaptureMaterial";

// Must match the `FluidParticle` WGSL struct in shaders/FluidParticleData.ts:
// position (vec4<f32>) + velocity (vec4<f32>).
const FLOATS_PER_PARTICLE = 8;

// The depth-capture spheres are drawn larger than the physical
// (collision) particle radius so neighboring particles' depth
// footprints overlap on screen — that overlap is what lets the
// bilateral blur fuse them into one continuous surface instead of a
// cluster of visible balls. Purely a rendering splat size; doesn't
// affect simulation.
const DEPTH_CAPTURE_RADIUS_SCALE = 2.0;

export class FluidParticleField {
    public readonly object3D: Object3D;
    // Second, screen-invisible instance of the same particles, used
    // only so FluidDepthPass can render their distance from the
    // camera (see FluidDepthCaptureMaterial) into an isolated
    // off-screen target without that showing up in the normal
    // on-screen picture. FluidDepthPass draws depthCaptureRenderer
    // directly (see main.ts) rather than going through any layer/mask
    // based scene traversal, so VisibleLayer.None here only affects
    // ordinary passes (ColorPass etc.), not FluidDepthPass itself.
    public readonly depthCaptureObject3D: Object3D;
    public readonly depthCaptureRenderer: MeshRenderer;
    private readonly sphereRenderer: MeshRenderer;
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
        this.sphereRenderer = mr;

        this.depthCaptureObject3D = new Object3D();
        this.depthCaptureObject3D.name = "FluidParticlesDepthCapture";

        const depthGeometry = new SphereGeometry(particleRadius * DEPTH_CAPTURE_RADIUS_SCALE, 8, 6);
        const depthMr = this.depthCaptureObject3D.addComponent(MeshRenderer);
        depthMr.geometry = depthGeometry;
        depthMr.material = new FluidDepthCaptureMaterial();
        depthMr.instanceCount = this.particleCount;
        // Hides it from every ordinary, layer/mask-based pass
        // (ColorPass etc.) — the same mechanism setSpheresVisible(false)
        // below already uses successfully. FluidDepthPass never calls
        // collectLayered()/consults visibleLayer at all, so this has
        // no effect on whether FluidDepthPass draws it.
        depthMr.visibleLayer = VisibleLayer.None;
        depthMr.material.getPass(PassType.COLOR)[0]!.setStorageBuffer("particles", this.buffer);
        this.depthCaptureRenderer = depthMr;
    }

    // The screen-space renderer's shading/composite step paints over
    // the particles' screen-space footprint using the depth capture,
    // not by replacing this mesh — so once it's working, the original
    // spheres need to be hidden or they'd still show through/behind it.
    setSpheresVisible(visible: boolean): void {
        this.sphereRenderer.visibleLayer = visible ? VisibleLayer.Default : VisibleLayer.None;
    }
}
