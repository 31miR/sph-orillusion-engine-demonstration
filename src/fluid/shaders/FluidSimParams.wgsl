struct SimParams {
    deltaTime: f32,
    gravity: f32,
    restitution: f32,
    particleRadius: f32,
    boundsMinX: f32,
    boundsMinY: f32,
    boundsMinZ: f32,
    boundsMaxX: f32,
    boundsMaxY: f32,
    boundsMaxZ: f32,
    // Kernel support radius is 2 * smoothingLength (cubic spline, per the
    // STAR report). Cell size equals the kernel support, matching the
    // paper's stated optimal choice for uniform-grid neighbor search.
    smoothingLength: f32,
    cellSize: f32,
    gridDimX: f32,
    gridDimY: f32,
    gridDimZ: f32,
    // particleMass = smoothingLength^3 * restDensity (Algorithm 1, STAR
    // report). restDensity is the fluid's target rest density (rho_0);
    // density summation (Eq. 3) should read close to this for a properly
    // spaced interior particle.
    particleMass: f32,
    restDensity: f32,
    // Stiffness constant k in the Tait-style equation of state (Eq. 9):
    // p = k * ((rho/rho_0)^7 - 1). Larger k reduces compressibility but
    // demands a smaller time step (drives the pressure force — tune with
    // care, see FluidPressureForceCompute.wgsl).
    stiffness: f32,
    // Kinematic viscosity nu (Eq. 8's Laplacian, scaled). Real water is
    // ~1e-6 m^2/s, but the STAR report notes larger user-defined values
    // are typically preferred for SPH stability — treat as tunable.
    viscosity: f32,
};

// CFL condition (STAR report, p.3, right after Algorithm 1): a stable
// step satisfies dt <= lambda * particleDiameter / v_max, lambda ~= 0.4
// per [Mon92]. Every shader that multiplies an acceleration/velocity by
// dt (pressure force, viscosity force, integrate) calls this instead of
// reading params.deltaTime directly, so the step size actually taken
// adapts to how fast the fluid currently is — smaller during a fast
// splash, allowed back up to the requested size once it calms down —
// rather than always assuming a fixed worst case.
const CFL_LAMBDA: f32 = 0.4;

// maxVelocityBits comes from FluidMaxVelocityCompute.wgsl's atomicMax
// reduction over last step's velocities (raw bits — see that file for
// why), not this step's: the reduction can only see the previous
// step's finished velocities, since this step's forces haven't been
// computed yet when it runs. A one-step-old estimate is a fine
// approximation (velocity doesn't change instantly), not a bug.
fn computeDt(requestedDt: f32, maxVelocityBits: u32, particleDiameter: f32) -> f32 {
    let maxVelocity = bitcast<f32>(maxVelocityBits);
    let safeDt = CFL_LAMBDA * particleDiameter / max(maxVelocity, 1e-5);
    return min(requestedDt, safeDt);
}

fn simCellCoord(params: SimParams, pos: vec3<f32>) -> vec3<i32> {
    let gridDim = vec3<i32>(i32(params.gridDimX), i32(params.gridDimY), i32(params.gridDimZ));
    let boundsMin = vec3<f32>(params.boundsMinX, params.boundsMinY, params.boundsMinZ);
    let coord = vec3<i32>(floor((pos - boundsMin) / params.cellSize));
    return clamp(coord, vec3<i32>(0), gridDim - vec3<i32>(1));
}

fn simCellIndex(params: SimParams, coord: vec3<i32>) -> u32 {
    let gridDim = vec3<i32>(i32(params.gridDimX), i32(params.gridDimY), i32(params.gridDimZ));
    return u32(coord.x + coord.y * gridDim.x + coord.z * gridDim.x * gridDim.y);
}
