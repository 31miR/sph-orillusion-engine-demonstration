import { ComputeShader, RenderGraphPass, RenderTexture } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";
import { FluidDepthBlur } from "./shaders/FluidDepthBlur";
import { FluidDepthPass } from "./FluidDepthPass";

const WORKGROUP_SIZE = 8;

function workgroupsFor(pixels: number): number {
    return Math.ceil(pixels / WORKGROUP_SIZE);
}

// Number of bilateral-blur passes to chain, ping-ponging between two
// textures — see FluidDepthBlur.wgsl's header comment (van der Laan et
// al. 2009 Sec 3.5.1): ~6 iterations of a bilateral Gaussian filter is
// reported to reach similar quality to full curvature flow.
const BLUR_ITERATIONS = 6;

// Step 2 of the screen-space fluid renderer: smooth FluidDepthPass's
// captured distances into one continuous surface (see
// FluidDepthBlur.wgsl for the bilateral-blur math + citations).
//
// `b.dependsOn("FluidDepthPass")` is what guarantees this runs after
// the capture for the current frame — FluidDepthPass doesn't publish
// its color texture as a named graph-pool resource, so there's no
// `b.read(...)` edge to hook into. `dependsOn` is the render graph's
// documented mechanism for exactly this situation: an upstream pass
// whose output is consumed through a side channel (here,
// capturedDistanceTexture) rather than a tracked resource.
export class FluidDepthSmoothPass extends RenderGraphPass {
    readonly name = "FluidDepthSmoothPass";

    // One ComputeShader per iteration, each with its own fixed
    // input/output texture binding set once in setup() (rather than
    // rebinding a single shared instance's textures every frame) —
    // matches how the engine's own multi-pass compute chains (e.g.
    // Bloom's per-mip blurComputes) are built.
    private blurShaders: ComputeShader[] = [];
    private pingTextures!: [RenderTexture, RenderTexture];
    private finalTexture!: RenderTexture;
    private depthPass: FluidDepthPass;

    constructor(depthPass: FluidDepthPass) {
        super();
        this.depthPass = depthPass;
    }

    // this.finalTexture is set in setup(), but setup() across
    // different passes can run in any order within the same compile —
    // consumers (FluidNormalReconstructPass) should read this from
    // their own execute(), which always runs after every pass's
    // setup() has completed, not from their own setup().
    get smoothedDepthTexture(): RenderTexture {
        return this.finalTexture;
    }

    private createPingTexture(ctx: any): RenderTexture {
        // RTResourceMap.createRTTexture (used originally here) builds a
        // texture meant for ordinary render-target use — its usage
        // flags include TextureBinding/RenderAttachment but not
        // StorageBinding, so a compute shader can never write into it
        // (confirmed by the browser's own WebGPU validation error).
        // Constructing RenderTexture directly exposes the usage flags
        // so we can ask for STORAGE_BINDING explicitly.
        const texture = new RenderTexture(
            window.innerWidth,
            window.innerHeight,
            "r32float",
            false,
            GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
            1,
            0,
            false,
            true,
            ctx,
        );
        // r32float (and other 32-bit float formats) can only ever be
        // bound as "unfilterable-float" in WebGPU — they don't support
        // linear filtering. RenderTexture's own default assumes the
        // filterable "float" sample type, which WebGPU then rejects at
        // bind-group creation for this format. This isn't a guess: the
        // engine's own internal r32float consumers (HiZPass and
        // others) hit the identical problem and fix it exactly this
        // way — see the _patchMipLevels doc comment in
        // orillusion.es.max.js for the engine's own account of it.
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
