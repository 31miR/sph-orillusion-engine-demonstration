const PI: f32 = 3.14159265358979;

// Cubic spline kernel (Monaghan 1992), Eq. 4-5 in the STAR report.
// q = r/h; support is q < 2, i.e. r < 2h. sigma is the 3D normalization
// constant (3 / (2 * pi * h^3)) that makes the kernel integrate to 1
// over all space.
fn cubicSplineWeight(r: f32, h: f32) -> f32 {
    let q = r / h;
    if (q >= 2.0) {
        return 0.0;
    }

    let sigma = 3.0 / (2.0 * PI * h * h * h);
    var f: f32;
    if (q < 1.0) {
        f = (2.0 / 3.0) - q * q + 0.5 * q * q * q;
    } else {
        let t = 2.0 - q;
        f = (1.0 / 6.0) * t * t * t;
    }
    return sigma * f;
}
