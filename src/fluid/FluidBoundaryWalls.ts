import { StorageGPUBuffer } from "@orillusion/core";
import type { FluidBounds } from "./FluidBounds";

// Samples the tank's 6 inner walls with static Akinci-style boundary
// particles (Akinci et al. 2012), at roughly the fluid's own particle
// spacing — the STAR report's Sec. 4 documents this as the standard
// fix for particle deficiency at a solid boundary (an under-sampled
// support domain near a wall under-reports density there).
export class FluidBoundaryWalls {
    // vec4 per particle: xyz = static world position, w = Akinci
    // volume (filled in by FluidBoundaryVolumeCompute.wgsl).
    public readonly positionBuffer: StorageGPUBuffer;
    public readonly count: number;

    constructor(bounds: FluidBounds, spacing: number) {
        const points: number[] = [];
        const stepX = Math.max(1, Math.round((bounds.max.x - bounds.min.x) / spacing));
        const stepY = Math.max(1, Math.round((bounds.max.y - bounds.min.y) / spacing));
        const stepZ = Math.max(1, Math.round((bounds.max.z - bounds.min.z) / spacing));

        const addFace = (fixedAxis: "x" | "y" | "z", fixedValue: number, aSteps: number, bSteps: number, aMin: number, aMax: number, bMin: number, bMax: number) => {
            for (let ai = 0; ai <= aSteps; ai++) {
                for (let bi = 0; bi <= bSteps; bi++) {
                    const a = aMin + (aMax - aMin) * (ai / aSteps);
                    const b = bMin + (bMax - bMin) * (bi / bSteps);
                    if (fixedAxis === "x") {
                        points.push(fixedValue, a, b, 1);
                    } else if (fixedAxis === "y") {
                        points.push(a, fixedValue, b, 1);
                    } else {
                        points.push(a, b, fixedValue, 1);
                    }
                }
            }
        };

        addFace("x", bounds.min.x, stepY, stepZ, bounds.min.y, bounds.max.y, bounds.min.z, bounds.max.z);
        addFace("x", bounds.max.x, stepY, stepZ, bounds.min.y, bounds.max.y, bounds.min.z, bounds.max.z);
        addFace("y", bounds.min.y, stepX, stepZ, bounds.min.x, bounds.max.x, bounds.min.z, bounds.max.z);
        addFace("y", bounds.max.y, stepX, stepZ, bounds.min.x, bounds.max.x, bounds.min.z, bounds.max.z);
        addFace("z", bounds.min.z, stepX, stepY, bounds.min.x, bounds.max.x, bounds.min.y, bounds.max.y);
        addFace("z", bounds.max.z, stepX, stepY, bounds.min.x, bounds.max.x, bounds.min.y, bounds.max.y);

        this.count = points.length / 4;
        this.positionBuffer = new StorageGPUBuffer(points.length, 0, new Float32Array(points));
    }
}
