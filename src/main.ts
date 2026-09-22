import { Engine3D, Scene3D, Camera3D, Object3D, View3D, DirectLight, Color, HoverCameraController, AtmosphericComponent, MeshRenderer, PlaneGeometry, LitMaterial } from "@orillusion/core";
import { Stats } from "@orillusion/stats";
import * as dat from "dat.gui";
import { FluidParticleField } from "./fluid/FluidParticleField";
import { FluidSimulator } from "./fluid/FluidSimulator";
import { FluidSimulationComponent } from "./fluid/FluidSimulationComponent";
import { DEFAULT_FLUID_BOUNDS } from "./fluid/FluidBounds";

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
    const floorWidth = (bounds.max.x - bounds.min.x) * 1.5;
    const floorDepth = (bounds.max.z - bounds.min.z) * 1.5;
    floorRenderer.geometry = new PlaneGeometry(floorWidth, floorDepth, 1, 1);
    floorRenderer.material = new LitMaterial();
    floorObj.y = bounds.min.y;
    scene.addChild(floorObj);

    // Scaled up from the original 8^3=512 toy grid to 20^3=8000 particles,
    // spacing/radius shrunk proportionally so the block still fits the
    // box with room to fall and move (previously spanned 2.1 units in a
    // +/-2 box; this spans ~2.85).
    const fluidParticles = new FluidParticleField(20, 0.15, 0.06);
    scene.addChild(fluidParticles.object3D);

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
        maxDeltaTime: 1 / 30,
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
    fluidFolder.add(tunables, "maxDeltaTime", 1 / 240, 1 / 10).onChange((v: number) => simulator.setMaxDeltaTime(v));
    fluidFolder.open();

    const view = new View3D();
    view.scene = scene;
    view.camera = camera;

    engine.startRenderView(view);
}

init();
