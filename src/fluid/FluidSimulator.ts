import { ComputeShader, ShaderLib, StorageGPUBuffer, UniformGPUBuffer, View3D } from "@orillusion/core";
import { FluidParticleData } from "./shaders/FluidParticleData";
import { FluidIntegrateCompute } from "./shaders/FluidIntegrateCompute";
import type { FluidBounds } from "./FluidBounds";

const WORKGROUP_SIZE = 64;

export interface FluidSimulatorOptions {
    bounds: FluidBounds;
    particleRadius: number;
    deltaTime?: number;
    gravity?: number;
    restitution?: number;
}

export class FluidSimulator {
    private readonly computeShader: ComputeShader;
    private readonly params: UniformGPUBuffer;

    constructor(particleBuffer: StorageGPUBuffer, particleCount: number, options: FluidSimulatorOptions) {
        const { bounds, particleRadius, deltaTime = 1 / 60, gravity = 9.8, restitution = 0.4 } = options;

        ShaderLib.register("FluidParticleData", FluidParticleData);
        ShaderLib.register("FluidIntegrateCompute", FluidIntegrateCompute);

        // SimParams: deltaTime, gravity, restitution, particleRadius,
        // boundsMinX/Y/Z, boundsMaxX/Y/Z — 10 plain f32 fields, 40 bytes,
        // no padding concerns since every field is a scalar f32.
        this.params = new UniformGPUBuffer(40);
        this.params.setFloat("deltaTime", deltaTime);
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
        view.engine3D.context3D.gpuContext.computeCommand(command, [this.computeShader]);
    }
}
