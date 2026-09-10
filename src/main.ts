import { Engine3D, Scene3D, Camera3D, Object3D, View3D, DirectLight, Color, HoverCameraController, AtmosphericComponent } from "@orillusion/core";

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

    const cameraObj = new Object3D();
    const camera = cameraObj.addComponent(Camera3D);
    camera.perspective(60, window.innerWidth / window.innerHeight, 1, 5000);

    const controller = cameraObj.addComponent(HoverCameraController);
    controller.setCamera(0, 0, 15);
    scene.addChild(cameraObj);

    const lightObj = new Object3D();
    const light = lightObj.addComponent(DirectLight);
    light.lightColor = new Color(1.0, 1.0, 1.0, 1.0);
    light.intensity = 15;
    scene.addChild(lightObj);

    const view = new View3D();
    view.scene = scene;
    view.camera = camera;

    engine.startRenderView(view);
}

init();
