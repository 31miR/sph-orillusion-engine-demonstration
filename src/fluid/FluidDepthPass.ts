import { MeshRenderer, PassType, RenderGraphPass, RenderGraphRenderTarget, RenderTexture } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";

export const FLUID_DEPTH_RT = "FluidDepthRT";

// Step 1 of the screen-space fluid renderer (van der Laan, Green,
// Sainz 2009, "Screen Space Fluid Rendering with Curvature Flow";
// STAR report Sec 7.3). Renders only the depth-capture echo (a
// specific MeshRenderer, see main.ts) into an isolated target using
// the main camera directly — not through collectLayered()/layer-mask
// scene traversal, and not a second camera — so the fluid-only depth
// stays uncontaminated by floor/skybox depth before later passes
// smooth it.
//
// Two color attachments: this material's PassType.COLOR pipeline
// writes two targets (rgba16float + the engine's rgba32float GBuffer
// slot), confirmed via the WebGPU validation error when only one was
// declared. Slot 0 is the one FluidDepthCaptureShader.wgsl actually
// writes distance-to-camera into (storeOp "store", exposed below);
// slot 1 is discarded.
export class FluidDepthPass extends RenderGraphPass {
    readonly name = "FluidDepthPass";

    private target!: RenderGraphRenderTarget;
    private readonly node: MeshRenderer;

    constructor(node: MeshRenderer) {
        super();
        this.node = node;
    }

    setup(b: RenderGraphBuilder): void {
        this.target = b.createRenderTarget(FLUID_DEPTH_RT, {
            label: FLUID_DEPTH_RT,
            colors: [
                { name: "FluidDepthCapturedColor", format: "rgba16float", storeOp: "store" },
                { name: "FluidDepthDiscardColor1", format: "rgba32float", storeOp: "discard" },
            ],
            depth: { format: "depth32float", depthClearValue: 1.0 },
        });
    }

    get capturedDistanceTexture(): RenderTexture {
        return this.target.colorTextures[0]!;
    }

    // Matches the engine's own PreDepthPass._drawOpaque line for line —
    // deviating from it (missing bindCamera, unconditional nodeUpdate)
    // produced validation errors.
    execute(ctx: RenderGraphPassContext): void {
        const view = ctx.view;
        const camera = view.camera;
        const opened = this.target.beginPass(ctx, {});
        const { encoder, passState } = opened;
        passState.camera3D = camera;
        // Every material's shader reads camera matrices from bind
        // group 0; the normal color pass binds this once before
        // drawing, so this pass must do the same explicitly.
        view.engine3D.context3D.gpuContext.bindCamera(encoder, camera);

        if (!this.node.preInit(PassType.COLOR)) {
            this.node.nodeUpdate(view, PassType.COLOR, passState, undefined as any);
        }
        this.node.renderPass2(view, PassType.COLOR, passState, undefined as any, encoder);

        this.target.endPass(ctx, opened);
    }
}
