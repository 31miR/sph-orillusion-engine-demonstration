import { ComputeShader, ShaderLib, StorageGPUBuffer, Time, UniformGPUBuffer, View3D } from "@orillusion/core";
import { FluidParticleData } from "./shaders/FluidParticleData";
import { FluidIntegrateCompute } from "./shaders/FluidIntegrateCompute";
import type { FluidBounds } from "./FluidBounds";

const WORKGROUP_SIZE = 64;

export interface FluidSimulatorOptions {
    bounds: FluidBounds;
    particleRadius: number;
    // Ceiling on the per-frame timestep (seconds). Real elapsed time is
    // used below this; a slow/stalled frame gets clamped here instead of
    // handing the simulation an unstably large step. Default: 1/30s.
    maxDeltaTime?: number;
    gravity?: number;
    restitution?: number;
}

export class FluidSimulator {
    private readonly computeShader: ComputeShader;
    private readonly params: UniformGPUBuffer;
    private readonly maxDeltaTime: number;

    constructor(particleBuffer: StorageGPUBuffer, particleCount: number, options: FluidSimulatorOptions) {
        const { bounds, particleRadius, maxDeltaTime = 1 / 30, gravity = 9.8, restitution = 0.4 } = options;
        this.maxDeltaTime = maxDeltaTime;

        ShaderLib.register("FluidParticleData", FluidParticleData);
        ShaderLib.register("FluidIntegrateCompute", FluidIntegrateCompute);

        // SimParams: deltaTime, gravity, restitution, particleRadius,
        // boundsMinX/Y/Z, boundsMaxX/Y/Z — 10 plain f32 fields, 40 bytes,
        // no padding concerns since every field is a scalar f32.
        this.params = new UniformGPUBuffer(40);
        // Placeholder — compute() overwrites this every frame with the
        // real, clamped elapsed time before each dispatch.
        this.params.setFloat("deltaTime", 0);
        this.params.setFloat("gravity", gravity);
        this.params.setFloat("restitution", restitution);
        this.params.setFloat("particleRadius", particleRadius);
        this.params.setFloat("boundsMinX", bounds.min.x);
        this.params.setFloat("boundsMinY", bounds.min.y);
        this.params.setFloat("boundsMinZ", bounds.min.z);
        this.params.setFloat("boundsMaxX", bounds.max.x);
        this.params.setFloat("boundsMaxY", bounds.max.y);
        this.params.setFloat("boundsMaxZ", bounds.max.z);
        this.params.apply();

        this.computeShader = new ComputeShader(FluidIntegrateCompute);
        this.computeShader.setUniformBuffer("params", this.params);
        this.computeShader.setStorageBuffer("particles", particleBuffer);
        this.computeShader.workerSizeX = Math.ceil(particleCount / WORKGROUP_SIZE);
        this.computeShader.workerSizeY = 1;
        this.computeShader.workerSizeZ = 1;
    }

    public compute(view: View3D, command: GPUCommandEncoder) {
        // Time.delta is milliseconds (from requestAnimationFrame); our
        // shader math expects seconds. Clamp so a slow/stalled frame
        // can't hand the simulation an unstably large step.
        const dt = Math.min(Time.delta / 1000, this.maxDeltaTime);
        this.params.setFloat("deltaTime", dt);
        this.params.apply();

        view.engine3D.context3D.gpuContext.computeCommand(command, [this.computeShader]);
    }
}
