import { ComputeShader, RenderGraphPass, RenderTexture } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";
import { FluidDepthBlur } from "./shaders/FluidDepthBlur";
import { FluidDepthPass } from "./FluidDepthPass";

const WORKGROUP_SIZE = 8;

function workgroupsFor(pixels: number): number {
    return Math.ceil(pixels / WORKGROUP_SIZE);
}

// Bilateral-blur iterations to chain, ping-ponging between two
// textures — van der Laan et al. 2009 Sec 3.5.1 reports ~6 iterations
// reaching similar quality to full curvature flow.
const BLUR_ITERATIONS = 6;

// Step 2 of the screen-space fluid renderer: smooth FluidDepthPass's
// captured distances into one continuous surface (see
// FluidDepthBlur.wgsl for the bilateral-blur math).
//
// `b.dependsOn("FluidDepthPass")`, not `b.read`: FluidDepthPass
// doesn't publish its texture as a named graph resource (it's read via
// capturedDistanceTexture instead), so dependsOn is the only way to
// order against it.
export class FluidDepthSmoothPass extends RenderGraphPass {
    readonly name = "FluidDepthSmoothPass";

    private blurShaders: ComputeShader[] = [];
    private pingTextures!: [RenderTexture, RenderTexture];
    private finalTexture!: RenderTexture;
    private depthPass: FluidDepthPass;

    constructor(depthPass: FluidDepthPass) {
        super();
        this.depthPass = depthPass;
    }

    // Consumers should read this from their own execute() (which
    // always runs after every pass's setup()), not from their setup() —
    // setup() order across passes isn't guaranteed within one compile.
    get smoothedDepthTexture(): RenderTexture {
        return this.finalTexture;
    }

    private createPingTexture(ctx: any): RenderTexture {
        // RTResourceMap.createRTTexture's usage flags don't include
        // STORAGE_BINDING, so a compute shader can't write into it —
        // constructing RenderTexture directly exposes the usage flags.
        //
        // Sized from ctx.presentationSize, not window.innerWidth/Height:
        // the real canvas resolution is clientWidth * devicePixelRatio,
        // which only matches window.innerWidth when devicePixelRatio is
        // 1 (false on HiDPI displays).
        const presentationWidth = ctx.presentationSize[0]!;
        const presentationHeight = ctx.presentationSize[1]!;
        const texture = new RenderTexture(
            presentationWidth,
            presentationHeight,
            "r32float",
            false,
            GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
            1,
            0,
            false,
            true,
            ctx,
        );
        // r32float doesn't support linear filtering in WebGPU, so it
        // must be bound as "unfilterable-float" — RenderTexture
        // defaults to "float", which WebGPU rejects for this format
        // (same fix the engine's own HiZPass applies to r32float).
        texture.textureBindingLayout.sampleType = "unfilterable-float";
        return texture;
    }

    setup(b: RenderGraphBuilder): void {
        b.dependsOn("FluidDepthPass");

        const ctx = b.context3D;
        this.pingTextures = [this.createPingTexture(ctx), this.createPingTexture(ctx)];

        this.blurShaders = [];
        for (let i = 0; i < BLUR_ITERATIONS; i++) {
            const shader = new ComputeShader(FluidDepthBlur);
            shader.entryPoint = i === 0 ? "CsMainFirst" : "CsMain";
            const outputTexture = this.pingTextures[i % 2]!;
            shader.setStorageTexture("outTex", outputTexture);
            this.blurShaders.push(shader);
        }
        this.finalTexture = this.pingTextures[(BLUR_ITERATIONS - 1) % 2]!;
    }

    execute(ctx: RenderGraphPassContext): void {
        const view = ctx.view;
        const captureTexture = this.depthPass.capturedDistanceTexture;

        for (let i = 0; i < this.blurShaders.length; i++) {
            const shader = this.blurShaders[i]!;
            const inputTexture = i === 0 ? captureTexture : this.pingTextures[(i - 1) % 2]!;
            shader.setSamplerTexture("depthTex", inputTexture);
            shader.workerSizeX = workgroupsFor(inputTexture.width);
            shader.workerSizeY = workgroupsFor(inputTexture.height);
        }

        const device = view.engine3D.context3D.device;
        const command = device.createCommandEncoder();
        view.engine3D.context3D.gpuContext.computeCommand(command, this.blurShaders);
        device.queue.submit([command.finish()]);
    }
}
