// Dedicated VisibleLayer bit for fluid particle instances, so custom
// render passes (e.g. FluidDepthPass) can isolate them from the rest
// of the scene (floor/skybox/etc, all left on the default bit 0).
// Bit 0 is reserved by the engine for "unassigned/legacy" — custom
// layers must use bits 1..31.
export const FLUID_LAYER = 1 << 1;
