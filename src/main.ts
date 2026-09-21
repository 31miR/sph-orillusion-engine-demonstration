import { Engine3D, Scene3D, Camera3D, Object3D, View3D, DirectLight, Color, HoverCameraController, AtmosphericComponent, MeshRenderer, PlaneGeometry, LitMaterial } from "@orillusion/core";
import { Stats } from "@orillusion/stats";
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

    const fluidParticles = new FluidParticleField();
    scene.addChild(fluidParticles.object3D);

    const simulator = new FluidSimulator(fluidParticles.buffer, fluidParticles.particleCount, {
        bounds,
        particleRadius: fluidParticles.particleRadius,
        smoothingLength: fluidParticles.spacing,
    });
    fluidParticles.object3D.addComponent(FluidSimulationComponent, simulator);

    const view = new View3D();
    view.scene = scene;
    view.camera = camera;

    engine.startRenderView(view);
}

init();
