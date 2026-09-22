import { PassType, RenderGraphPass, RenderGraphRenderTarget } from "@orillusion/core";
import type { RenderGraphBuilder, RenderGraphPassContext } from "@orillusion/core";
import { FLUID_LAYER } from "./FluidLayers";

// Renders ONLY the fluid particle instances (see FluidLayers.ts) into
// an isolated off-screen depth target, separate from the main scene's
// GBuffer depth. This is step 1 of the screen-space fluid renderer
// (van der Laan, Green, Sainz 2009, "Screen Space Fluid Rendering with
// Curvature Flow" [vdLGS09]; surveyed in the STAR report Sec 7.3,
// "Screen Space Approaches"): later passes smooth this depth with
// curvature flow and reconstruct a surface from it, which only works
// if the depth is fluid-only — mixed in with floor/skybox depth,
// smoothing would bleed across that boundary.
//
// Reuses PassType.COLOR (the same pass FluidParticleMaterial already
// renders through in the main scene) rather than a dedicated
// depth-only PassType — this pass's render target still has a (unread,
// discarded) color attachment alongside the depth attachment we
// actually care about, purely so the already-working color pipeline
// can be reused unmodified. A true depth-only pipeline (skipping the
// fragment shader entirely) would be a later optimization, not a
// correctness requirement, since all we read afterward is the depth
// texture.
export class FluidDepthPass extends RenderGraphPass {
    readonly name = "FluidDepthPass";
    layerMask = FLUID_LAYER;

    private target!: RenderGraphRenderTarget;

    setup(b: RenderGraphBuilder): void {
        this.target = b.createRenderTarget("FluidDepthRT", {
            label: "FluidDepthRT",
            colors: [{ name: "FluidDepthDiscardColor", format: "rgba8unorm", storeOp: "discard" }],
            depth: { format: "depth32float", depthClearValue: 1.0 },
        });
    }

    execute(ctx: RenderGraphPassContext): void {
        const view = ctx.view;
        const opened = this.target.beginPass(ctx, {});
        const { opaque } = this.collectLayered(view);
        for (const node of opaque) {
            node.preInit(PassType.COLOR);
            node.nodeUpdate(view, PassType.COLOR, opened.passState);
            node.renderPass2(view, PassType.COLOR, opened.passState, undefined as any, opened.encoder);
        }
        this.target.endPass(ctx, opened);
    }
}
