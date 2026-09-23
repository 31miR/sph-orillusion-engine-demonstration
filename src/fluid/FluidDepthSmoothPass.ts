import { ComputeShader, RenderGraphPass, RenderTexture, SceneCaptureCameraComponent } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";
import { FluidDepthBlur } from "./shaders/FluidDepthBlur";

const WORKGROUP_SIZE = 8;

function workgroupsFor(pixels: number): number {
    return Math.ceil(pixels / WORKGROUP_SIZE);
}

// Step 2 of the screen-space fluid renderer: smooth the depth-capture
// component's captured distances into one continuous surface (see
// FluidDepthBlur.wgsl for the bilateral-blur math + citations).
//
// `b.dependsOn("SceneCapturePass")` is what guarantees this runs after
// the capture for the current frame — SceneCaptureCameraComponent
// doesn't publish a named graph-pool resource (each capture owns its
// own render target privately), so there's no `b.read(...)` edge to
// hook into. `dependsOn` is the render graph's documented mechanism
// for exactly this situation: an upstream pass whose output is
// consumed through a side channel (here, getCaptureTexture()) rather
// than a tracked resource.
export class FluidDepthSmoothPass extends RenderGraphPass {
    readonly name = "FluidDepthSmoothPass";

    private blurShader!: ComputeShader;
    private outputTexture!: RenderTexture;
    private captureComponent: SceneCaptureCameraComponent;

    constructor(captureComponent: SceneCaptureCameraComponent) {
        super();
        this.captureComponent = captureComponent;
    }

    setup(b: RenderGraphBuilder): void {
        b.dependsOn("SceneCapturePass");

        const ctx = b.context3D;
        // RTResourceMap.createRTTexture (used originally here) builds a
        // texture meant for ordinary render-target use — its usage
        // flags include TextureBinding/RenderAttachment but not
        // StorageBinding, so a compute shader can never write into it
        // (confirmed by the browser's own WebGPU validation error).
        // Constructing RenderTexture directly exposes the usage flags
        // so we can ask for STORAGE_BINDING explicitly.
        this.outputTexture = new RenderTexture(
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

        this.blurShader = new ComputeShader(FluidDepthBlur);
        this.blurShader.setStorageTexture("outTex", this.outputTexture);
    }

    execute(ctx: RenderGraphPassContext): void {
        const view = ctx.view;
        const captureTexture = this.captureComponent.getCaptureTexture();
        if (!captureTexture) {
            return;
        }

        this.blurShader.setSamplerTexture("depthTex", captureTexture);
        this.blurShader.workerSizeX = workgroupsFor(captureTexture.width);
        this.blurShader.workerSizeY = workgroupsFor(captureTexture.height);

        const device = view.engine3D.context3D.device;
        const command = device.createCommandEncoder();
        view.engine3D.context3D.gpuContext.computeCommand(command, [this.blurShader]);
        device.queue.submit([command.finish()]);
    }
}
