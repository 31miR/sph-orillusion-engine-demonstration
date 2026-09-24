import { Engine3D, PostBase, RenderTexture, ShaderLib, View3D, ViewQuad } from "@orillusion/core";
import { FluidShadeComposite } from "./shaders/FluidShadeComposite";
import { FluidDepthSmoothPass } from "./FluidDepthSmoothPass";
import { FluidNormalReconstructPass } from "./FluidNormalReconstructPass";

// Step 4 of the screen-space fluid renderer: shade the reconstructed
// surface and composite it over the scene (see FluidShadeComposite.wgsl
// for the actual math + citations).
//
// Unlike FluidDepthSmoothPass/FluidNormalReconstructPass (bare compute
// shaders we read back ourselves), this needs to be a real, engine-
// recognized post-processing step — createViewQuad + a fragment
// shader is the established mechanism other built-in effects
// (TonemapPost, FXAAPost) use for "transform the image and hand it to
// the next stage"; a compute shader writing to an arbitrary texture,
// as the earlier two steps do, is never picked up as the chain's
// current image, so nothing downstream (including the final present)
// would ever see it.
export class FluidShadeCompositePost extends PostBase {
    private postQuad!: ViewQuad;
    private renderTexture!: RenderTexture;
    // PostProcessingComponent.addPost() only ever calls `new C()` with
    // no arguments, so these can't be constructor parameters like
    // every other pass in this feature — set via configure() right
    // after addPost() returns the instance instead.
    private smoothPass!: FluidDepthSmoothPass;
    private normalPass!: FluidNormalReconstructPass;

    constructor() {
        super();
        ShaderLib.register("FluidShadeCompositeShader", FluidShadeComposite);
    }

    configure(smoothPass: FluidDepthSmoothPass, normalPass: FluidNormalReconstructPass): void {
        this.smoothPass = smoothPass;
        this.normalPass = normalPass;
    }

    protected createResource(view: View3D): void {
        // NOTE: this runs synchronously inside addPost() itself, before
        // main.ts gets the returned instance back to call configure()
        // on — so this.smoothPass/normalPass aren't set yet here.
        this.renderTexture = this.createRTTexture("FluidShadeComposite", window.innerWidth, window.innerHeight, "rgba16float");
        this.postQuad = this.createViewQuad("fluidShadeComposite", "FluidShadeCompositeShader", this.renderTexture);

        // Bind SOME valid texture immediately, same reason QuadShader's
        // own constructor binds a placeholder blackTexture to baseMap
        // before the real upstream image exists: the pipeline/bind
        // group gets built from whatever is bound right now, and
        // leaving these bindings empty at that moment is what produced
        // the WebGPU bind-group validation errors here — render()
        // rebinds them to the real textures every frame once
        // configure() has run.
        const placeholder = Engine3D.resFor(this._boundCtx!).whiteTexture;
        this.postQuad.quadShader.setTexture("depthTex", placeholder);
        this.postQuad.quadShader.setTexture("normalTex", placeholder);

        // Fresnel reflectance at normal incidence — see
        // FluidShadeComposite.wgsl's materialUniform comment.
        this.postQuad.quadShader.setUniform("f0", 0.02);
    }

    onResize(): void {
        this.renderTexture.resize(window.innerWidth, window.innerHeight);
    }

    private updateInputTextures(): void {
        this.postQuad.quadShader.setTexture("depthTex", this.smoothPass.smoothedDepthTexture);
        this.postQuad.quadShader.setTexture("normalTex", this.normalPass.reconstructedNormalTexture);
    }

    render(view: View3D, command: GPUCommandEncoder): void {
        this.updateInputTextures();
        this.rtViewQuad.forEach((viewQuad) => {
            const lastTexture = this._boundCtx!.gpuContext.lastRenderPassState.getLastRenderTexture(this._boundCtx!);
            viewQuad.renderToViewQuad(view, viewQuad, command, lastTexture);
        });
    }
}
