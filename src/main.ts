import { Engine3D, Scene3D, Camera3D, Object3D, View3D, DirectLight, Color, HoverCameraController, AtmosphericComponent, MeshRenderer, BoxGeometry, LitMaterial, PostProcessingComponent } from "@orillusion/core";
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

// Blur radius (in pixels) the depth smoothing was tuned at, at full
// resolution; scaled by the chosen resolutions so its on-screen
// footprint stays the same.
const FULL_RES_BLUR_RADIUS = 12;

const WALL_THICKNESS = 0.2;

interface StartConfig {
    particlesPerAxis: number;
    resolutionScale: number;
    fluidRenderScale: number;
}

function askStartConfig(): Promise<StartConfig> {
    return new Promise((resolve) => {
        const config: StartConfig = { particlesPerAxis: 18, resolutionScale: 1, fluidRenderScale: 0.5 };
        const gui = new dat.GUI({ name: "Setup" });
        gui.add(config, "particlesPerAxis", 4, MAX_PARTICLES_PER_AXIS, 1).name("Particles per axis");
        gui.add(config, "resolutionScale", 0.25, 1, 0.05).name("Resolution scale");
        gui.add(config, "fluidRenderScale", 0.25, 1, 0.05).name("Fluid render scale");
        gui.add({ start: () => { gui.destroy(); resolve(config); } }, "start").name("Start");
    });
}

async function init() {
    const startConfig = await askStartConfig();

    const engine = await Engine3D.init({
        canvasConfig: {
            canvas: document.getElementById('canvas') as HTMLCanvasElement,
            // Renders below native resolution; the browser stretches the
            // canvas back to the window size.
            devicePixelRatio: (window.devicePixelRatio || 1) * startConfig.resolutionScale,
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

    // Slabs sit entirely outside the bounds, so their inner faces lie
    // exactly on the simulation's walls. Extents are chosen so no two
    // slabs overlap (overlapping coplanar faces would z-fight).
    const { min, max } = bounds;
    const addSlab = (x0: number, x1: number, y0: number, y1: number, z0: number, z1: number) => {
        const obj = new Object3D();
        const renderer = obj.addComponent(MeshRenderer);
        renderer.geometry = new BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
        renderer.material = new LitMaterial();
        obj.x = (x0 + x1) / 2;
        obj.y = (y0 + y1) / 2;
        obj.z = (z0 + z1) / 2;
        scene.addChild(obj);
    };

    addSlab(min.x, max.x, min.y - WALL_THICKNESS, min.y, min.z, max.z);
    // Only the two walls on the far side of the initial camera
    // (setCamera above starts it on the +x/+z side), so the fluid
    // stays visible while the container still reads as one.
    addSlab(min.x - WALL_THICKNESS, min.x, min.y - WALL_THICKNESS, max.y, min.z, max.z);
    addSlab(min.x - WALL_THICKNESS, max.x, min.y - WALL_THICKNESS, max.y, min.z - WALL_THICKNESS, min.z);

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
    const blurRadius = Math.max(1, Math.round(FULL_RES_BLUR_RADIUS * startConfig.resolutionScale * startConfig.fluidRenderScale));
    const depthPass = view.renderGraph!.add(FluidDepthPass, fluidParticles.depthCaptureRenderer, startConfig.fluidRenderScale);
    const smoothPass = view.renderGraph!.add(FluidDepthSmoothPass, depthPass, blurRadius);
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
