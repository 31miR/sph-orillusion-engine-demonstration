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
    // demands a smaller time step; not yet load-bearing until pressure
    // actually drives a force (a later step), so treat as a placeholder
    // pending real tuning.
    stiffness: f32,
};

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
