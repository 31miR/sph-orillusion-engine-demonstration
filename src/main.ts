import { Engine3D, Scene3D, Camera3D, Object3D, View3D, DirectLight, Color, HoverCameraController, AtmosphericComponent, MeshRenderer, PlaneGeometry, LitMaterial, PostProcessingComponent } from "@orillusion/core";
import { Stats } from "@orillusion/stats";
import * as dat from "dat.gui";
import { FluidParticleField } from "./fluid/FluidParticleField";
import { FluidSimulator } from "./fluid/FluidSimulator";
import { FluidSimulationComponent } from "./fluid/FluidSimulationComponent";
import { DEFAULT_FLUID_BOUNDS } from "./fluid/FluidBounds";
import { FluidDepthPass } from "./fluid/FluidDepthPass";
import { FluidDepthSmoothPass } from "./fluid/FluidDepthSmoothPass";
import { FluidNormalReconstructPass } from "./fluid/FluidNormalReconstructPass";
import { FluidShadeCompositePost } from "./fluid/FluidShadeCompositePost";


const FLUID_BLOCK_SPAN = 2.85;
// Same particleRadius:spacing ratio established when first tuning the
// depth-capture splat overlap — kept constant so denser configurations
// don't also change the particles' relative size.
const RADIUS_TO_SPACING_RATIO = 0.4;
// Performance cap, not a correctness one.
const MAX_PARTICLES_PER_AXIS = 100;

interface StartConfig {
    particlesPerAxis: number;
}

function askStartConfig(): Promise<StartConfig> {
    return new Promise((resolve) => {
        const config: StartConfig = { particlesPerAxis: 18 };
        const gui = new dat.GUI({ name: "Setup" });
        gui.add(config, "particlesPerAxis", 4, MAX_PARTICLES_PER_AXIS, 1).name("Particles per axis");
        gui.add({ start: () => { gui.destroy(); resolve(config); } }, "start").name("Start");
    });
}

async function init() {
    const startConfig = await askStartConfig();

    const engine = await Engine3D.init({
        canvasConfig: {
            canvas: document.getElementById('canvas') as HTMLCanvasElement
        },
        setting: {
            render: {
                debug: true
            }
        }
    });

    const scene = new Scene3D();
    scene.addComponent(AtmosphericComponent);

    const stats = scene.addComponent(Stats);
    stats.container.style.transform = "scale(1.5)";
    stats.container.style.transformOrigin = "top left";

    const cameraObj = new Object3D();
    const camera = cameraObj.addComponent(Camera3D);
    camera.perspective(60, window.innerWidth / window.innerHeight, 1, 5000);

    const controller = cameraObj.addComponent(HoverCameraController);
    controller.setCamera(45, -30, 6);
    scene.addChild(cameraObj);

    const lightObj = new Object3D();
    const light = lightObj.addComponent(DirectLight);
    light.lightColor = new Color(1.0, 1.0, 1.0, 1.0);
    light.intensity = 15;
    scene.addChild(lightObj);

    const bounds = DEFAULT_FLUID_BOUNDS;

    const floorObj = new Object3D();
    const floorRenderer = floorObj.addComponent(MeshRenderer);
    const floorWidth = bounds.max.x - bounds.min.x;
    const floorDepth = bounds.max.z - bounds.min.z;
    floorRenderer.geometry = new PlaneGeometry(floorWidth, floorDepth, 1, 1);
    floorRenderer.material = new LitMaterial();
    floorObj.y = bounds.min.y;
    scene.addChild(floorObj);

    // spacing/particleRadius derived from the chosen particle count so
    // the block's overall footprint stays at FLUID_BLOCK_SPAN
    // regardless of resolution — see the constant's own comment above.
    const spacing = FLUID_BLOCK_SPAN / Math.max(1, startConfig.particlesPerAxis - 1);
    const particleRadius = spacing * RADIUS_TO_SPACING_RATIO;
    const fluidParticles = new FluidParticleField(startConfig.particlesPerAxis, spacing, particleRadius);
    scene.addChild(fluidParticles.object3D);
    scene.addChild(fluidParticles.depthCaptureObject3D);

    // Explicit values, not FluidSimulator's internal defaults, so the
    // GUI below can't silently drift out of sync with the constructor.
    const tunables = {
        gravity: 9.8,
        restitution: 0.4,
        restDensity: 1000,
        stiffness: 20,
        viscosity: 0.1,
        maxDeltaTime: 1 / 10,
    };

    const simulator = new FluidSimulator(fluidParticles.buffer, fluidParticles.particleCount, {
        bounds,
        particleRadius: fluidParticles.particleRadius,
        smoothingLength: fluidParticles.spacing,
        ...tunables,
    });
    fluidParticles.object3D.addComponent(FluidSimulationComponent, simulator);

    // Structural params (particle count, spacing, bounds, smoothingLength)
    // are baked into buffer/grid sizes at construction and aren't
    // live-tunable here — only the plain uniform values below are.
    const gui = new dat.GUI();
    const fluidFolder = gui.addFolder("Fluid");
    fluidFolder.add(tunables, "gravity", 0, 30).onChange((v: number) => simulator.setGravity(v));
    fluidFolder.add(tunables, "restitution", 0, 1).onChange((v: number) => simulator.setRestitution(v));
    fluidFolder.add(tunables, "restDensity", 100, 5000).onChange((v: number) => simulator.setRestDensity(v));
    fluidFolder.add(tunables, "stiffness", 0, 200).onChange((v: number) => simulator.setStiffness(v));
    fluidFolder.add(tunables, "viscosity", 0, 2).onChange((v: number) => simulator.setViscosity(v));
    fluidFolder.add(tunables, "maxDeltaTime", 1 / 200, 1 / 5).onChange((v: number) => simulator.setMaxDeltaTime(v));
    fluidFolder.add({ reset: () => fluidParticles.reset(engine.context3D.device) }, "reset").name("Reset particles");
    fluidFolder.open();

    const view = new View3D();
    view.scene = scene;
    view.camera = camera;

    engine.startRenderView(view);

    // Screen-space fluid renderer: capture depth (FluidDepthPass),
    // smooth it (FluidDepthSmoothPass), reconstruct normals
    // (FluidNormalReconstructPass), then shade + composite
    // (FluidShadeCompositePost).
    const depthPass = view.renderGraph!.add(FluidDepthPass, fluidParticles.depthCaptureRenderer);
    const smoothPass = view.renderGraph!.add(FluidDepthSmoothPass, depthPass);
    const normalPass = view.renderGraph!.add(FluidNormalReconstructPass, smoothPass);

    // addPost() only constructs with no arguments, so upstream passes
    // are wired in afterward via configure().
    const postProcessing = scene.addComponent(PostProcessingComponent);
    const shadeComposite = postProcessing.addPost(FluidShadeCompositePost);
    shadeComposite.configure(smoothPass, normalPass);

    // The composite paints over the particles using the depth capture;
    // the original spheres would otherwise show through.
    fluidParticles.setSpheresVisible(false);
}

init();
