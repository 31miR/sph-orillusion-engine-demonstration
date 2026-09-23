import { ComputeShader, RenderGraphPass, RenderTexture, UniformGPUBuffer } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";
import { FluidNormalReconstruct } from "./shaders/FluidNormalReconstruct";
import { FluidDepthSmoothPass } from "./FluidDepthSmoothPass";

const WORKGROUP_SIZE = 8;

function workgroupsFor(pixels: number): number {
    return Math.ceil(pixels / WORKGROUP_SIZE);
}

// Step 3 of the screen-space fluid renderer: reconstruct a view-space
// surface normal per pixel from the smoothed depth (see
// FluidNormalReconstruct.wgsl for the math + citations).
export class FluidNormalReconstructPass extends RenderGraphPass {
    readonly name = "FluidNormalReconstructPass";

    private normalShader!: ComputeShader;
    private outputTexture!: RenderTexture;
    private params!: UniformGPUBuffer;
    private smoothPass: FluidDepthSmoothPass;

    constructor(smoothPass: FluidDepthSmoothPass) {
        super();
        this.smoothPass = smoothPass;
    }

    setup(b: RenderGraphBuilder): void {
        b.dependsOn("FluidDepthSmoothPass");

        const ctx = b.context3D;
        this.outputTexture = new RenderTexture(
            window.innerWidth,
            window.innerHeight,
            "rgba16float",
            false,
            GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
            1,
            0,
            false,
            true,
            ctx,
        );

        // Just the camera's inverse projection matrix (64 bytes) — see
        // FluidNormalReconstruct.wgsl for why this is our own small
        // uniform buffer instead of the engine's internal
        // GlobalUniform bind group.
        this.params = new UniformGPUBuffer(64);

        this.normalShader = new ComputeShader(FluidNormalReconstruct);
        this.normalShader.setUniformBuffer("params", this.params);
        this.normalShader.setStorageTexture("outTex", this.outputTexture);
    }

    get reconstructedNormalTexture(): RenderTexture {
        return this.outputTexture;
    }

    execute(ctx: RenderGraphPassContext): void {
        const view = ctx.view;
        const depthTexture = this.smoothPass.smoothedDepthTexture;

        // setMatrix, not setFloat32Array — the field is declared
        // `mat4x4<f32>` in the shader, and there's a dedicated method
        // for exactly that on GPUBufferBase; setFloat32Array doesn't
        // reliably map onto a matrix-typed struct field.
        this.params.setMatrix("projMatInv", view.camera.projectionMatrixInv);
        this.params.apply();

        this.normalShader.setSamplerTexture("depthTex", depthTexture);
        this.normalShader.workerSizeX = workgroupsFor(depthTexture.width);
        this.normalShader.workerSizeY = workgroupsFor(depthTexture.height);

        const device = view.engine3D.context3D.device;
        const command = device.createCommandEncoder();
        view.engine3D.context3D.gpuContext.computeCommand(command, [this.normalShader]);
        device.queue.submit([command.finish()]);
    }
}
