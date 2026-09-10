import { Engine3D, Scene3D, Camera3D, BoxGeometry, MeshRenderer, UnLitMaterial, Object3D, View3D, DirectLight, Color, HoverCameraController } from "@orillusion/core";

async function init() {
    // 1. Initialize the engine with config settings
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

    // 2. Create Scene
    const scene = new Scene3D();

    // 3. Create Camera
    const cameraObj = new Object3D();
    const camera = cameraObj.addComponent(Camera3D);
    camera.perspective(60, window.innerWidth / window.innerHeight, 1, 5000);

    // Add camera controller (orbit/drag)
    const controller = cameraObj.addComponent(HoverCameraController);
    controller.setCamera(0, 0, 15);
    scene.addChild(cameraObj);

    // 4. Create a light source
    const lightObj = new Object3D();
    const light = lightObj.addComponent(DirectLight);
    light.lightColor = new Color(1.0, 1.0, 1.0, 1.0);
    light.intensity = 15;
    scene.addChild(lightObj);

    // 5. Add Geometry / Cube (placeholder until the SPH particle renderer replaces it)
    const obj = new Object3D();
    obj.name = "Cube";

    const meshRenderer = obj.addComponent(MeshRenderer);
    meshRenderer.geometry = new BoxGeometry(1, 1, 1);
    meshRenderer.material = new UnLitMaterial();

    scene.addChild(obj);

    // 6. Create View3D and start rendering
    const view = new View3D();
    view.scene = scene;
    view.camera = camera;

    engine.startRenderView(view);
}

init();
