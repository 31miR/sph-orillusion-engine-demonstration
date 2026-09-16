export interface FluidBounds {
    min: { x: number; y: number; z: number };
    max: { x: number; y: number; z: number };
}

// Single source of truth for the box boundary, shared between the
// compute shader's collision response and the visual floor in main.ts
// so the two can never silently drift apart.
export const DEFAULT_FLUID_BOUNDS: FluidBounds = {
    min: { x: -2, y: -2, z: -2 },
    max: { x: 2, y: 2, z: 2 },
};
