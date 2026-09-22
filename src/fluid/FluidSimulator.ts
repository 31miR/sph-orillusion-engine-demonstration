import { ComputeShader, ShaderLib, StorageGPUBuffer, Time, UniformGPUBuffer, View3D } from "@orillusion/core";
import { FluidParticleData } from "./shaders/FluidParticleData";
import { FluidSimParams } from "./shaders/FluidSimParams";
import { FluidKernel } from "./shaders/FluidKernel";
import { FluidIntegrateCompute } from "./shaders/FluidIntegrateCompute";
import { FluidGridClear } from "./shaders/FluidGridClear";
import { FluidGridBuild } from "./shaders/FluidGridBuild";
import { FluidDensityCompute } from "./shaders/FluidDensityCompute";
import { FluidPressureCompute } from "./shaders/FluidPressureCompute";
import { FluidPressureForceCompute } from "./shaders/FluidPressureForceCompute";
import { FluidViscosityForceCompute } from "./shaders/FluidViscosityForceCompute";
import type { FluidBounds } from "./FluidBounds";

const WORKGROUP_SIZE = 64;

export interface FluidSimulatorOptions {
    bounds: FluidBounds;
    particleRadius: number;
    // SPH smoothing length h. Kernel support is 2h; the neighbor-search
    // grid's cell size is also 2h (Sec 2.1 of the STAR report). Defaults
    // to matching the particle field's initial spacing.
    smoothingLength: number;
    // Target rest density (rho_0). Particle mass is derived from it as
    // h^3 * restDensity (Algorithm 1, STAR report) rather than set
    // independently. Defaults to water, 1000 kg/m^3.
    restDensity?: number;
    // Stiffness constant k in the equation of state (Eq. 9). Now
    // load-bearing: it directly drives the pressure force. Explicit SPH
    // with a stiff EOS can go unstable if k is too large relative to the
    // time step (the STAR report notes this tradeoff directly) — this
    // default is a conservative starting point, expect it (and possibly
    // maxDeltaTime) to need real tuning once tested.
    stiffness?: number;
    // Kinematic viscosity nu (Eq. 8). Larger values damp jitter/noise
    // between neighboring particles more aggressively; the STAR report
    // notes real water's physical value is far too small to be useful
    // for SPH stability, so larger user-tuned values are standard —
    // treat as tunable, not physically literal.
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
    private readonly maxDeltaTime: number;

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
            maxDeltaTime = 1 / 30,
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

        // Grid buffers: cellHead[cell] is the index of the last particle
        // written there this frame (or -1); particleNext[particle] chains
        // back to whatever occupied the cell before it — a singly linked
        // list per cell, built with one atomic exchange per particle
        // instead of a sort. Neither needs CPU-supplied initial data since
        // gridClear/gridBuild fully overwrite both every frame.
        const cellHeadBuffer = new StorageGPUBuffer(cellCount);
        const particleNextBuffer = new StorageGPUBuffer(particleCount);

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
        this.pressureForceShader.workerSizeX = workgroupsFor(particleCount);

        this.viscosityForceShader = new ComputeShader(FluidViscosityForceCompute);
        this.viscosityForceShader.setUniformBuffer("params", this.params);
        this.viscosityForceShader.setStorageBuffer("particles", particleBuffer);
        this.viscosityForceShader.setStorageBuffer("cellHead", cellHeadBuffer);
        this.viscosityForceShader.setStorageBuffer("particleNext", particleNextBuffer);
        this.viscosityForceShader.workerSizeX = workgroupsFor(particleCount);

        this.integrateShader = new ComputeShader(FluidIntegrateCompute);
        this.integrateShader.setUniformBuffer("params", this.params);
        this.integrateShader.setStorageBuffer("particles", particleBuffer);
        this.integrateShader.workerSizeX = workgroupsFor(particleCount);
    }

    public compute(view: View3D, command: GPUCommandEncoder) {
        // Time.delta is milliseconds (from requestAnimationFrame); our
        // shader math expects seconds. Clamp so a slow/stalled frame
        // can't hand the simulation an unstably large step.
        const dt = Math.min(Time.delta / 1000, this.maxDeltaTime);
        this.params.setFloat("deltaTime", dt);
        this.params.apply();

        // Order matters throughout: grid cleared before built, built
        // before queried, density before pressure (Eq. 9 needs it), and
        // pressure before the pressure-force pass (which needs every
        // particle's pressure, not just its own). Viscosity force can
        // run either side of pressure force — both just accumulate
        // independently into velocity — as long as both finish before
        // integrate. Matches Algorithm 1's order in the STAR report:
        // neighbors, density, pressure, forces, integrate.
        view.engine3D.context3D.gpuContext.computeCommand(command, [
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
