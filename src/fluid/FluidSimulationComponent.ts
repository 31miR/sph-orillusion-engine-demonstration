import { ComponentBase, View3D } from "@orillusion/core";
import { FluidSimulator } from "./FluidSimulator";

// Thin adapter onto the engine's component lifecycle — simulation
// logic lives in FluidSimulator, kept independent of this API surface.
export class FluidSimulationComponent extends ComponentBase {
    private simulator!: FluidSimulator;

    public override init(param: FluidSimulator) {
        this.simulator = param;
    }

    public override onCompute(view: View3D, command: GPUCommandEncoder) {
        this.simulator.compute(view, command);
    }
}
