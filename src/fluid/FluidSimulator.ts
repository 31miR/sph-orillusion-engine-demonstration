import { ComputeShader, ShaderLib, StorageGPUBuffer, Time, UniformGPUBuffer, View3D } from "@orillusion/core";
import { FluidParticleData } from "./shaders/FluidParticleData";
import { FluidSimParams } from "./shaders/simulation/FluidSimParams";
import { FluidKernel } from "./shaders/simulation/FluidKernel";
import { FluidIntegrateCompute } from "./shaders/simulation/FluidIntegrateCompute";
import { FluidGridClear } from "./shaders/simulation/FluidGridClear";
import { FluidGridBuild } from "./shaders/simulation/FluidGridBuild";
import { FluidDensityCompute } from "./shaders/simulation/FluidDensityCompute";
import { FluidPressureCompute } from "./shaders/simulation/FluidPressureCompute";
import { FluidPressureForceCompute } from "./shaders/simulation/FluidPressureForceCompute";
import { FluidViscosityForceCompute } from "./shaders/simulation/FluidViscosityForceCompute";
import { FluidMaxVelocityClear } from "./shaders/simulation/FluidMaxVelocityClear";
import { FluidMaxVelocityReduce } from "./shaders/simulation/FluidMaxVelocityReduce";
import type { FluidBounds } from "./FluidBounds";

const WORKGROUP_SIZE = 64;

export interface FluidSimulatorOptions {
    bounds: FluidBounds;
    particleRadius: number;
    // SPH smoothing length h. Kernel support is 2h; the neighbor-search
    // grid's cell size is also 2h (STAR report Sec 2.1).
    smoothingLength: number;
    // Target rest density (rho_0). Particle mass is derived from it as
    // h^3 * restDensity (Algorithm 1, STAR report).
    restDensity?: number;
    // Stiffness constant k in the equation of state (Eq. 9). Too large
    // relative to the time step goes unstable (STAR report).
    stiffness?: number;
    // Kinematic viscosity nu (Eq. 8) — tunable, not physically literal;
    // real water's value is too small for SPH stability.
    viscosity?: number;
    maxDeltaTime?: number;
    gravity?: number;
    restitution?: number;
}

function workgroupsFor(count: number): number {
    return Math.ceil(count / WORKGROUP_SIZE);
}

export class FluidSimulator {
    private readonly params: UniformGPUBuffer;
    private maxDeltaTime: number;

    private readonly maxVelocityClearShader: ComputeShader;
    private readonly maxVelocityReduceShader: ComputeShader;
    private readonly gridClearShader: ComputeShader;
    private readonly gridBuildShader: ComputeShader;
    private readonly densityShader: ComputeShader;
    private readonly pressureShader: ComputeShader;
    private readonly pressureForceShader: ComputeShader;
    private readonly viscosityForceShader: ComputeShader;
    private readonly integrateShader: ComputeShader;

    constructor(particleBuffer: StorageGPUBuffer, particleCount: number, options: FluidSimulatorOptions) {
        const {
            bounds,
            particleRadius,
            smoothingLength,
            restDensity = 1000,
            stiffness = 20,
            viscosity = 0.1,
            // Only bounds the very first step(s), before the max-velocity
            // reduction has real data — computeDt() (FluidSimParams.wgsl)
            // handles the live CFL clamp from then on.
            maxDeltaTime = 1 / 10,
            gravity = 9.8,
            restitution = 0.4,
        } = options;
        this.maxDeltaTime = maxDeltaTime;

        const cellSize = 2 * smoothingLength;
        const gridDimX = Math.max(1, Math.ceil((bounds.max.x - bounds.min.x) / cellSize));
        const gridDimY = Math.max(1, Math.ceil((bounds.max.y - bounds.min.y) / cellSize));
        const gridDimZ = Math.max(1, Math.ceil((bounds.max.z - bounds.min.z) / cellSize));
        const cellCount = gridDimX * gridDimY * gridDimZ;
        const particleMass = smoothingLength * smoothingLength * smoothingLength * restDensity;

        ShaderLib.register("FluidParticleData", FluidParticleData);
        ShaderLib.register("FluidSimParams", FluidSimParams);
        ShaderLib.register("FluidKernel", FluidKernel);
        ShaderLib.register("FluidIntegrateCompute", FluidIntegrateCompute);
        ShaderLib.register("FluidGridClear", FluidGridClear);
        ShaderLib.register("FluidGridBuild", FluidGridBuild);
        ShaderLib.register("FluidDensityCompute", FluidDensityCompute);
        ShaderLib.register("FluidPressureCompute", FluidPressureCompute);
        ShaderLib.register("FluidPressureForceCompute", FluidPressureForceCompute);
        ShaderLib.register("FluidViscosityForceCompute", FluidViscosityForceCompute);
        ShaderLib.register("FluidMaxVelocityClear", FluidMaxVelocityClear);
        ShaderLib.register("FluidMaxVelocityReduce", FluidMaxVelocityReduce);

        // SimParams: 19 plain f32 fields, 76 bytes — see FluidSimParams.wgsl.
        this.params = new UniformGPUBuffer(76);
        this.params.setFloat("deltaTime", 0); // overwritten every frame in compute()
        this.params.setFloat("gravity", gravity);
        this.params.setFloat("restitution", restitution);
        this.params.setFloat("particleRadius", particleRadius);
        this.params.setFloat("boundsMinX", bounds.min.x);
        this.params.setFloat("boundsMinY", bounds.min.y);
        this.params.setFloat("boundsMinZ", bounds.min.z);
        this.params.setFloat("boundsMaxX", bounds.max.x);
        this.params.setFloat("boundsMaxY", bounds.max.y);
        this.params.setFloat("boundsMaxZ", bounds.max.z);
        this.params.setFloat("smoothingLength", smoothingLength);
        this.params.setFloat("cellSize", cellSize);
        this.params.setFloat("gridDimX", gridDimX);
        this.params.setFloat("gridDimY", gridDimY);
        this.params.setFloat("gridDimZ", gridDimZ);
        this.params.setFloat("particleMass", particleMass);
        this.params.setFloat("restDensity", restDensity);
        this.params.setFloat("stiffness", stiffness);
        this.params.setFloat("viscosity", viscosity);
        this.params.apply();

        // cellHead[cell] is the last particle written there this frame
        // (or -1); particleNext[particle] chains to whoever occupied the
        // cell before it — a singly linked list per cell, built with one
        // atomic exchange per particle instead of a sort.
        const cellHeadBuffer = new StorageGPUBuffer(cellCount);
        const particleNextBuffer = new StorageGPUBuffer(particleCount);

        // Single u32 slot, read via atomicMax by the reduce shader — see
        // FluidMaxVelocityReduce.wgsl / FluidMaxVelocityClear.wgsl.
        const maxVelocityBuffer = new StorageGPUBuffer(1);

        // Separate files, not shared entry points: the clear shader
        // doesn't touch "particles" at all, and the engine's module-level
        // binding check requires it either way, breaking one or the other.
        this.maxVelocityClearShader = new ComputeShader(FluidMaxVelocityClear);
        this.maxVelocityClearShader.setStorageBuffer("maxVelocityBits", maxVelocityBuffer);
        this.maxVelocityClearShader.workerSizeX = 1;

        this.maxVelocityReduceShader = new ComputeShader(FluidMaxVelocityReduce);
        this.maxVelocityReduceShader.setStorageBuffer("particles", particleBuffer);
        this.maxVelocityReduceShader.setStorageBuffer("maxVelocityBits", maxVelocityBuffer);
        this.maxVelocityReduceShader.workerSizeX = workgroupsFor(particleCount);

        this.gridClearShader = new ComputeShader(FluidGridClear);
        this.gridClearShader.setUniformBuffer("params", this.params);
        this.gridClearShader.setStorageBuffer("cellHead", cellHeadBuffer);
        this.gridClearShader.workerSizeX = workgroupsFor(cellCount);

        this.gridBuildShader = new ComputeShader(FluidGridBuild);
        this.gridBuildShader.setUniformBuffer("params", this.params);
        this.gridBuildShader.setStorageBuffer("particles", particleBuffer);
        this.gridBuildShader.setStorageBuffer("cellHead", cellHeadBuffer);
        this.gridBuildShader.setStorageBuffer("particleNext", particleNextBuffer);
        this.gridBuildShader.workerSizeX = workgroupsFor(particleCount);

        this.densityShader = new ComputeShader(FluidDensityCompute);
        this.densityShader.setUniformBuffer("params", this.params);
        this.densityShader.setStorageBuffer("particles", particleBuffer);
        this.densityShader.setStorageBuffer("cellHead", cellHeadBuffer);
        this.densityShader.setStorageBuffer("particleNext", particleNextBuffer);
        this.densityShader.workerSizeX = workgroupsFor(particleCount);

        this.pressureShader = new ComputeShader(FluidPressureCompute);
        this.pressureShader.setUniformBuffer("params", this.params);
        this.pressureShader.setStorageBuffer("particles", particleBuffer);
        this.pressureShader.workerSizeX = workgroupsFor(particleCount);

        this.pressureForceShader = new ComputeShader(FluidPressureForceCompute);
        this.pressureForceShader.setUniformBuffer("params", this.params);
        this.pressureForceShader.setStorageBuffer("particles", particleBuffer);
        this.pressureForceShader.setStorageBuffer("cellHead", cellHeadBuffer);
        this.pressureForceShader.setStorageBuffer("particleNext", particleNextBuffer);
        this.pressureForceShader.setStorageBuffer("maxVelocityBits", maxVelocityBuffer);
        this.pressureForceShader.workerSizeX = workgroupsFor(particleCount);

        this.viscosityForceShader = new ComputeShader(FluidViscosityForceCompute);
        this.viscosityForceShader.setUniformBuffer("params", this.params);
        this.viscosityForceShader.setStorageBuffer("particles", particleBuffer);
        this.viscosityForceShader.setStorageBuffer("cellHead", cellHeadBuffer);
        this.viscosityForceShader.setStorageBuffer("particleNext", particleNextBuffer);
        this.viscosityForceShader.setStorageBuffer("maxVelocityBits", maxVelocityBuffer);
        this.viscosityForceShader.workerSizeX = workgroupsFor(particleCount);

        this.integrateShader = new ComputeShader(FluidIntegrateCompute);
        this.integrateShader.setUniformBuffer("params", this.params);
        this.integrateShader.setStorageBuffer("particles", particleBuffer);
        this.integrateShader.setStorageBuffer("maxVelocityBits", maxVelocityBuffer);
        this.integrateShader.workerSizeX = workgroupsFor(particleCount);
    }

    // Live-tunable parameters for the debug GUI; compute() re-applies
    // the whole uniform buffer every frame, so no immediate upload here.
    // Note: setRestDensity does not retroactively resize particleMass
    // (fixed at construction) — it only shifts what density the
    // pressure equation of state treats as "correct". Intentional, not
    // a bug.
    public setGravity(value: number): void {
        this.params.setFloat("gravity", value);
    }

    public setRestitution(value: number): void {
        this.params.setFloat("restitution", value);
    }

    public setRestDensity(value: number): void {
        this.params.setFloat("restDensity", value);
    }

    public setStiffness(value: number): void {
        this.params.setFloat("stiffness", value);
    }

    public setViscosity(value: number): void {
        this.params.setFloat("viscosity", value);
    }

    public setMaxDeltaTime(value: number): void {
        this.maxDeltaTime = value;
    }

    public compute(view: View3D, command: GPUCommandEncoder) {
        // Time.delta is milliseconds (from requestAnimationFrame); our
        // shader math expects seconds. Clamp so a slow/stalled frame
        // can't hand the simulation an unstably large step.
        const dt = Math.min(Time.delta / 1000, this.maxDeltaTime);
        this.params.setFloat("deltaTime", dt);
        this.params.apply();

        // Order matters: max-velocity clear/reduce before anything
        // calls computeDt() (a one-step-old estimate, not a bug); grid
        // cleared before built, built before queried; density before
        // pressure; pressure before pressure-force (needs every
        // particle's value, not just its own); both forces before
        // integrate. Matches Algorithm 1's order in the STAR report.
        view.engine3D.context3D.gpuContext.computeCommand(command, [
            this.maxVelocityClearShader,
            this.maxVelocityReduceShader,
            this.gridClearShader,
            this.gridBuildShader,
            this.densityShader,
            this.pressureShader,
            this.pressureForceShader,
            this.viscosityForceShader,
            this.integrateShader,
        ]);
    }
}
