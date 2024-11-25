use crate::consts::{MACHEP, MAXLOG, MINLOG};
use crate::{beta, betaln};

const MAXGAM: f64 = 34.844_256_272_771_76;

pub fn betainc(aa: f64, bb: f64, xx: f64) -> f64 {
    // domain errors
    if aa < 0.0 || bb < 0.0 || !(0.0..=1.0).contains(&xx) {
        return f64::NAN;
    }
    if xx == 0.0 {
        return 0.0;
    }
    if xx == 1.0 {
        return 1.0;
    }

    let mut flag = false;
    let t = if bb * xx <= 1.0 && xx <= 0.95 {
        pseries(aa, bb, xx)
    } else {
        let mut a = aa;
        let mut b = bb;
        let mut x = xx;
        let mut xc = 1.0 - xx;
        // reverse a and b if x is greater than the mean
        if xx > aa / (aa + bb) {
            flag = true;
            a = bb;
            b = aa;
            x = xc;
            xc = xx;
        }

        if flag && b * x <= 1.0 && x <= 0.95 {
            pseries(a, b, x)
        } else {
            // choose expansion for better convergence
            let y = x * (a + b - 2.0) - (a - 1.0);
            let w = if y < 0.0 { incbcf(a, b, x) } else { incbd(a, b, x) / xc };
            // multiply w by the factor
            //  a      b  _            _     _
            // x  (1-x)  | (a+b) / (a | (a) | (b) ) .
            let y = a * x.ln();
            let t = b * xc.ln();
            if a + b < MAXGAM && y.abs() < MAXLOG && t.abs() < MAXLOG {
                xc.powf(b) * x.powf(a) / a * w * (1.0 / beta(a, b))
            } else {
                // resort to logarithms
                let y = y + t - betaln(a, b) + (w / a).ln();
                if y < MINLOG {
                    0.0
                } else {
                    y.exp()
                }
            }
        }
    };

    if flag {
        if t <= MACHEP {
            1.0 - MACHEP
        } else {
            1.0 - t
        }
    } else {
        t
    }
}

const BIG: f64 = 4.503599627370496e15;
const BIGINV: f64 = 2.220_446_049_250_313e-16;

// continued fraction expansion #1 for incomplete beta integral
fn incbcf(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = a + b;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = b - 1.0;
    let mut k7 = k4;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;

    for _ in 0..300 {
        let xk = -(x * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        let xk = (x * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }

        let mut t = 1.0;
        if r != 0.0 {
            t = ((ans - r) / r).abs();
            ans = r;
        }

        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 += 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 -= 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

// continued fraction expansion #2 for incomplete beta integral
fn incbd(a: f64, b: f64, x: f64) -> f64 {
    let mut k1 = a;
    let mut k2 = b - 1.0;
    let mut k3 = a;
    let mut k4 = a + 1.0;
    let mut k5 = 1.0;
    let mut k6 = a + b;
    let mut k7 = a + 1.0;
    let mut k8 = a + 2.0;

    let mut pkm2 = 0.0;
    let mut qkm2 = 1.0;
    let mut pkm1 = 1.0;
    let mut qkm1 = 1.0;
    let z = x / (1.0 - x);
    let mut ans = 1.0;
    let mut r = 1.0;
    let thresh = 3.0 * MACHEP;

    for _ in 0..300 {
        let xk = -(z * k1 * k2) / (k3 * k4);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        let xk = (z * k5 * k6) / (k7 * k8);
        let pk = pkm1 + pkm2 * xk;
        let qk = qkm1 + qkm2 * xk;
        pkm2 = pkm1;
        pkm1 = pk;
        qkm2 = qkm1;
        qkm1 = qk;

        if qk != 0.0 {
            r = pk / qk;
        }

        let mut t = 1.0;
        if r != 0.0 {
            t = ((ans - r) / r).abs();
            ans = r;
        }

        if t < thresh {
            break;
        }

        k1 += 1.0;
        k2 -= 1.0;
        k3 += 2.0;
        k4 += 2.0;
        k5 += 1.0;
        k6 += 1.0;
        k7 += 2.0;
        k8 += 2.0;

        if qk.abs() + pk.abs() > BIG {
            pkm2 *= BIGINV;
            pkm1 *= BIGINV;
            qkm2 *= BIGINV;
            qkm1 *= BIGINV;
        }
        if qk.abs() < BIGINV || pk.abs() < BIGINV {
            pkm2 *= BIG;
            pkm1 *= BIG;
            qkm2 *= BIG;
            qkm1 *= BIG;
        }
    }
    ans
}

// power series for incomplete beta integral.
// use when b * x is small and x not too close to 1.
fn pseries(a: f64, b: f64, x: f64) -> f64 {
    let ai = 1.0 / a;
    let mut u = (1.0 - b) * x;
    let mut v = u / (a + 1.0);
    let t1 = v;
    let mut t = u;
    let mut n = 2.0;
    let mut s = 0.0;
    let z = MACHEP * ai;

    while v.abs() > z {
        u = (n - b) * x / n;
        t *= u;
        v = t / (a + n);
        s += v;
        n += 1.0;
    }
    s += t1;
    s += ai;

    let u = a * x.ln();

    if a + b < MAXGAM && u.abs() < MAXLOG {
        let t = 1.0 / beta(a, b);
        s * t * x.powf(a)
    } else {
        let t = -betaln(a, b) + u + s.ln();
        if t < MINLOG {
            0.0
        } else {
            t.exp()
        }
    }
}
