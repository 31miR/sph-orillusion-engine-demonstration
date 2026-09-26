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
    // only so FluidDepthPass can render their distance from the camera
    // into an isolated target — see FluidDepthCaptureMaterial and the
    // visibleLayer comment below.
    public readonly depthCaptureObject3D: Object3D;
    public readonly depthCaptureRenderer: MeshRenderer;
    private readonly sphereRenderer: MeshRenderer;
    public readonly buffer: StorageGPUBuffer;
    public readonly particleCount: number;
    public readonly particleRadius: number;
    public readonly spacing: number;
    // Kept so reset() can re-upload the original layout.
    private readonly initialData: Float32Array;

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

        this.initialData = data;
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
        // Hides it from ordinary layer/mask-based passes (ColorPass
        // etc.); FluidDepthPass draws it directly and ignores
        // visibleLayer entirely.
        depthMr.visibleLayer = VisibleLayer.None;
        depthMr.material.getPass(PassType.COLOR)[0]!.setStorageBuffer("particles", this.buffer);
        this.depthCaptureRenderer = depthMr;
    }

    // The composite step paints over the particles' footprint using
    // the depth capture, not by replacing this mesh, so the original
    // spheres need hiding once it's active.
    setSpheresVisible(visible: boolean): void {
        this.sphereRenderer.visibleLayer = visible ? VisibleLayer.Default : VisibleLayer.None;
    }

    // Nothing else needs resetting alongside this: FluidSimulator's
    // grid and max-velocity buffer both fully rebuild every frame.
    reset(device: GPUDevice): void {
        // Cast: writeBuffer wants an ArrayBuffer-backed view; a plain
        // Float32Array is typed more loosely than that even though
        // it's always backed by a real ArrayBuffer at runtime.
        device.queue.writeBuffer(this.buffer.buffer, 0, this.initialData as Float32Array<ArrayBuffer>);
    }
}
