import { ComputeShader, ShaderLib, StorageGPUBuffer, UniformGPUBuffer, View3D } from "@orillusion/core";
import { FluidParticleData } from "./shaders/FluidParticleData";
import { FluidIntegrateCompute } from "./shaders/FluidIntegrateCompute";

const WORKGROUP_SIZE = 64;

export class FluidSimulator {
    private readonly computeShader: ComputeShader;
    private readonly params: UniformGPUBuffer;

    constructor(particleBuffer: StorageGPUBuffer, particleCount: number, deltaTime: number = 1 / 60, gravity: number = 9.8) {
        ShaderLib.register("FluidParticleData", FluidParticleData);
        ShaderLib.register("FluidIntegrateCompute", FluidIntegrateCompute);

        // SimParams { deltaTime: f32, gravity: f32 } — 8 bytes, no padding
        // needed since both fields are plain f32.
        this.params = new UniformGPUBuffer(8);
        this.params.setFloat("deltaTime", deltaTime);
        this.params.setFloat("gravity", gravity);
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
