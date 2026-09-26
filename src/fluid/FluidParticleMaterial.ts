import { Color, Context3D, Engine3D, Material, PassType, RenderShaderPass, Shader, ShaderLib, Vector4 } from "@orillusion/core";
import { FluidParticleData } from "./shaders/FluidParticleData";
import { FluidParticleShader } from "./shaders/rendering/FluidParticleShader";

export class FluidParticleMaterial extends Material {
    constructor(ctx?: Context3D) {
        super();

        ShaderLib.register("FluidParticleData", FluidParticleData);
        ShaderLib.register("FluidParticleShader", FluidParticleShader);

        const pass = new RenderShaderPass("FluidParticleShader", "FluidParticleShader");
        pass.passType = PassType.COLOR;
        pass.setShaderEntry("VertMain", "FragMain");
        pass.setUniformVector4("transformUV1", new Vector4(0, 0, 1, 1));
        pass.setUniformVector4("transformUV2", new Vector4(0, 0, 1, 1));
        pass.setUniformColor("baseColor", new Color(0.2, 0.55, 1.0, 1.0));
        pass.setUniformFloat("alphaCutoff", 0.5);
        pass.setTexture("baseMap", Engine3D.resFor(ctx).whiteTexture);

        const shaderState = pass.shaderState;
        shaderState.acceptShadow = false;
        shaderState.receiveEnv = false;
        shaderState.acceptGI = false;
        shaderState.useLight = false;

        const shader = new Shader();
        shader.addRenderPass(pass);
        this.shader = shader;
    }
}
