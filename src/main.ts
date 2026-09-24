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

async function init() {
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

    // 32^3=32768 particles (up from 20^3=8000) — spacing/radius shrunk
    // proportionally (same radius:spacing ratio as before, 0.4) so the
    // block covers roughly the same ~2.85-unit span with finer,
    // smaller particles instead of a bigger volume.
    const fluidParticles = new FluidParticleField(32, 0.09, 0.036);
    scene.addChild(fluidParticles.object3D);
    scene.addChild(fluidParticles.depthCaptureObject3D);

    // Explicit values (not FluidSimulator's internal defaults) so the GUI
    // below is seeded with exactly what's actually running, not a value
    // that could silently drift out of sync with the constructor's own
    // defaults.
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
    // Range widened back out now that computeDt() (FluidSimParams.wgsl)
    // carries the actual per-step CFL safety — this only bounds the
    // first step(s), before that dynamic measurement has real data. See
    // FluidSimulator.ts's maxDeltaTime comment.
    fluidFolder.add(tunables, "maxDeltaTime", 1 / 200, 1 / 5).onChange((v: number) => simulator.setMaxDeltaTime(v));
    fluidFolder.open();

    const view = new View3D();
    view.scene = scene;
    view.camera = camera;

    engine.startRenderView(view);

    // Excludes the depth-capture echo from the normal on-screen
    // picture. depthCaptureRenderer already has visibleLayer set to
    // VisibleLayer.None (see FluidParticleField), which keeps every
    // ordinary, layer/mask-based pass (ColorPass etc.) from drawing
    // it. FluidDepthPass draws that same renderer directly — bypassing
    // layer/mask-based scene traversal entirely — so it's unaffected
    // by that setting.

    // Screen-space fluid renderer: render the depth-capture echo into
    // an isolated off-screen target using the main camera directly
    // (FluidDepthPass), smooth it (FluidDepthSmoothPass), reconstruct
    // a surface normal per pixel (FluidNormalReconstructPass), then
    // shade + composite over the scene (FluidShadeCompositePost).
    const depthPass = view.renderGraph!.add(FluidDepthPass, fluidParticles.depthCaptureRenderer);
    const smoothPass = view.renderGraph!.add(FluidDepthSmoothPass, depthPass);
    const normalPass = view.renderGraph!.add(FluidNormalReconstructPass, smoothPass);

    // addPost() only ever constructs with no arguments, so the two
    // upstream passes are wired in afterward via configure().
    const postProcessing = scene.addComponent(PostProcessingComponent);
    const shadeComposite = postProcessing.addPost(FluidShadeCompositePost);
    shadeComposite.configure(smoothPass, normalPass);

    // The composite paints over the particles' screen-space footprint
    // using the depth capture; the original spheres would otherwise
    // still be drawn underneath/showing through.
    fluidParticles.setSpheresVisible(false);
}

init();
