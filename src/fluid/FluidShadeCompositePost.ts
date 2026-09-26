import { Engine3D, MAIN_DEPTH_TEXTURE, PostBase, RenderTexture, ShaderLib, UniformGPUBuffer, View3D, ViewQuad } from "@orillusion/core";
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
    // Camera's inverse projection matrix, for linearizeSceneDepth() in
    // FluidShadeComposite.wgsl.
    private reconstructParams!: UniformGPUBuffer;

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
        //
        // Sized from presentationSize, not window.innerWidth/Height —
        // see the matching comment in FluidDepthSmoothPass.ts for why
        // those two differ on any HiDPI display (e.g. a MacBook).
        const [presentationWidth, presentationHeight] = this._boundCtx!.presentationSize;
        this.renderTexture = this.createRTTexture("FluidShadeComposite", presentationWidth!, presentationHeight!, "rgba16float");
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

        // sceneDepthTex is texture_depth_2d in the shader, which needs
        // a real depth-format placeholder — whiteTexture (an ordinary
        // color texture) would mismatch the "depth" sampleType the
        // bind group layout is reflected with, and error the same way
        // leaving it unbound would. Replaced with the real
        // _MainDepthTexture every frame in updateInputTextures().
        const depthPlaceholder = new RenderTexture(1, 1, "depth32float", false, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT, 1, 0, false, false, this._boundCtx!);
        depthPlaceholder.textureBindingLayout.sampleType = "depth";
        this.postQuad.quadShader.setTexture("sceneDepthTex", depthPlaceholder);

        // Fresnel reflectance at normal incidence — see
        // FluidShadeComposite.wgsl's materialUniform comment.
        this.postQuad.quadShader.setUniform("f0", 0.02);

        this.reconstructParams = new UniformGPUBuffer(64);
        this.postQuad.quadShader.setUniformBuffer("reconstructParams", this.reconstructParams);
    }

    onResize(): void {
        const [presentationWidth, presentationHeight] = this._boundCtx!.presentationSize;
        this.renderTexture.resize(presentationWidth!, presentationHeight!);
    }

    private updateInputTextures(view: View3D): void {
        this.postQuad.quadShader.setTexture("depthTex", this.smoothPass.smoothedDepthTexture);
        this.postQuad.quadShader.setTexture("normalTex", this.normalPass.reconstructedNormalTexture);

        // Fetched informally through the graph's resource pool rather
        // than a formal RenderGraphPass b.read — see
        // FluidShadeComposite.wgsl's sceneDepthTex comment for why a
        // formal graph edge can't work for a pass living inside the
        // engine's built-in PostPass chain. Safe in practice because
        // this pass, like every post effect, runs at the tail of the
        // frame — well after PreDepthPass has already populated it.
        const sceneDepthTex = view.renderGraph!.pool.get<RenderTexture>(MAIN_DEPTH_TEXTURE);
        this.postQuad.quadShader.setTexture("sceneDepthTex", sceneDepthTex);

        this.reconstructParams.setMatrix("projMatInv", view.camera.projectionMatrixInv);
        this.reconstructParams.apply();
    }

    render(view: View3D, command: GPUCommandEncoder): void {
        this.updateInputTextures(view);
        this.rtViewQuad.forEach((viewQuad) => {
            const lastTexture = this._boundCtx!.gpuContext.lastRenderPassState.getLastRenderTexture(this._boundCtx!);
            viewQuad.renderToViewQuad(view, viewQuad, command, lastTexture);
        });
    }
}
