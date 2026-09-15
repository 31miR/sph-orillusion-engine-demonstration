import { ComponentBase, View3D } from "@orillusion/core";
import { FluidSimulator } from "./FluidSimulator";

// Thin adapter onto the engine's component lifecycle — the actual
// simulation logic lives in FluidSimulator, kept independent of
// Orillusion's API surface. addComponent(FluidSimulationComponent, sim)
// threads `sim` through here via init(param).
export class FluidSimulationComponent extends ComponentBase {
    private simulator!: FluidSimulator;

    public override init(param: FluidSimulator) {
        this.simulator = param;
    }

    public override onCompute(view: View3D, command: GPUCommandEncoder) {
        this.simulator.compute(view, command);
    }
}
