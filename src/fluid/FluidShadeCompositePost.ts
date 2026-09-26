import { Engine3D, MAIN_DEPTH_TEXTURE, PostBase, RenderTexture, ShaderLib, UniformGPUBuffer, View3D, ViewQuad } from "@orillusion/core";
import { FluidShadeComposite } from "./shaders/FluidShadeComposite";
import { FluidDepthSmoothPass } from "./FluidDepthSmoothPass";
import { FluidNormalReconstructPass } from "./FluidNormalReconstructPass";

// Step 4 of the screen-space fluid renderer (see FluidShadeComposite.wgsl
// for the math). A PostBase/ViewQuad step, not a plain RenderGraphPass —
// only PostBase output is picked up by the engine's present chain.
export class FluidShadeCompositePost extends PostBase {
    private postQuad!: ViewQuad;
    private renderTexture!: RenderTexture;
    // addPost() only calls `new C()`, so these are wired in via
    // configure() after the fact instead of the constructor.
    private smoothPass!: FluidDepthSmoothPass;
    private normalPass!: FluidNormalReconstructPass;
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
        // Runs before configure(), so smoothPass/normalPass aren't set yet.
        const [presentationWidth, presentationHeight] = this._boundCtx!.presentationSize;
        this.renderTexture = this.createRTTexture("FluidShadeComposite", presentationWidth!, presentationHeight!, "rgba16float");
        this.postQuad = this.createViewQuad("fluidShadeComposite", "FluidShadeCompositeShader", this.renderTexture);

        // Every binding needs a valid resource before the pipeline's
        // first build, or it errors — render() rebinds the real
        // textures every frame once configure() has run.
        const placeholder = Engine3D.resFor(this._boundCtx!).whiteTexture;
        this.postQuad.quadShader.setTexture("depthTex", placeholder);
        this.postQuad.quadShader.setTexture("normalTex", placeholder);

        // sceneDepthTex is texture_depth_2d, so it needs a depth-format
        // placeholder (whiteTexture would mismatch the reflected
        // sampleType) — replaced with the real _MainDepthTexture every
        // frame in updateInputTextures().
        const depthPlaceholder = new RenderTexture(1, 1, "depth32float", false, GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT, 1, 0, false, false, this._boundCtx!);
        depthPlaceholder.textureBindingLayout.sampleType = "depth";
        this.postQuad.quadShader.setTexture("sceneDepthTex", depthPlaceholder);

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

        // Fetched informally through the graph's pool, not a formal
        // b.read — see FluidShadeComposite.wgsl's sceneDepthTex comment.
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
