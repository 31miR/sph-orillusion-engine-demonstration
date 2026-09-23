import { MeshRenderer, PassType, RenderGraphPass, RenderGraphRenderTarget, RenderTexture } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";

export const FLUID_DEPTH_RT = "FluidDepthRT";

// Renders ONLY the depth-capture echo (a specific MeshRenderer, handed
// in directly — see main.ts) into an isolated off-screen target,
// using the MAIN camera. Draws that one node directly instead of
// going through collectLayered()/layerMask/cullingMask-based scene
// traversal — every bug hit building this pass (echo leaking onto the
// main screen, echo vanishing from the capture, needing a second
// camera, that camera needing its own frustum/transform/scene
// attachment...) came from that generic layer-based machinery having
// more moving parts than expected, not from actually drawing the
// particle. We already know exactly which one object we want drawn;
// there's no reason to ask the engine's generic node-collection system
// to rediscover that.
//
// Not a second camera either, not SceneCaptureCameraComponent — this
// pass uses view.camera directly, the exact same camera object
// already proven correct for every other draw in this project.
//
// This is step 1 of the screen-space fluid renderer (van der Laan,
// Green, Sainz 2009, "Screen Space Fluid Rendering with Curvature
// Flow" [vdLGS09]; surveyed in the STAR report Sec 7.3, "Screen Space
// Approaches"): later passes smooth this depth with curvature flow
// and reconstruct a surface from it, which only works if the depth is
// fluid-only — mixed in with floor/skybox depth, smoothing would
// bleed across that boundary.
//
// Reuses PassType.COLOR (the same pass FluidDepthCaptureMaterial is
// registered under) rather than a dedicated depth-only PassType — see
// the two color attachments below.
//
// The two color attachments aren't a guess: WebGPU requires a
// pipeline and the render pass it's used in to match color-target
// count/format/order exactly, and this material's PassType.COLOR
// pipeline turns out to write TWO targets (rgba16float + rgba32float
// — the engine's GBuffer layout, the second likely packed
// normal/material data other built-in passes read back later).
// Confirmed from the browser's own WebGPU validation error, which
// reports a pipeline's actual compiled attachment state directly.
// Slot 0 (rgba16float) is what FluidDepthCaptureShader.wgsl actually
// writes distance-to-camera into, so — unlike a pass that only needs
// correct depth-testing side effects — this one keeps it (storeOp
// "store") instead of discarding it, and exposes it via
// capturedDistanceTexture below.
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

    // Per-node draw sequence below matches the engine's own
    // PreDepthPass._drawOpaque (dist/orillusion.es.max.js) line for
    // line — verified against the reference implementation rather
    // than guessed, after two earlier validation errors (missing
    // bindCamera, unconditional nodeUpdate) turned up from deviating
    // from it.
    execute(ctx: RenderGraphPassContext): void {
        const view = ctx.view;
        const camera = view.camera;
        const opened = this.target.beginPass(ctx, {});
        const { encoder, passState } = opened;
        passState.camera3D = camera;
        // Every material's shader reads camera matrices from bind
        // group 0 (the "camera" group). The normal color pass binds
        // this once before drawing; this pass has to do the same
        // explicitly, or every draw call fails with "No bind group
        // set at group index 0".
        view.engine3D.context3D.gpuContext.bindCamera(encoder, camera);

        // preInit() returning true means this node's pipeline for this
        // passType is already set up — nodeUpdate() only needs to run
        // when it wasn't.
        if (!this.node.preInit(PassType.COLOR)) {
            this.node.nodeUpdate(view, PassType.COLOR, passState, undefined as any);
        }
        this.node.renderPass2(view, PassType.COLOR, passState, undefined as any, encoder);

        this.target.endPass(ctx, opened);
    }
}
