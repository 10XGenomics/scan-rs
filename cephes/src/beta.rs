use crate::consts::MAXLOG;
use crate::gamma;
use crate::gamma::gammaln_sign;

const ASYMP_FACTOR: f64 = 1e6;
const MAXGAM: f64 = 171.624_376_956_302_7;

pub fn beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 && a == a.floor() {
        let ai = a as i32;
        if a == ai as f64 {
            return beta_negint(ai, b);
        } else {
            return std::f64::INFINITY;
        }
    }

    if b <= 0.0 && b == b.floor() {
        let bi = b as i32;
        if a == bi as f64 {
            return beta_negint(bi, a);
        } else {
            return std::f64::INFINITY;
        }
    }

    let (a, b) = if a.abs() < b.abs() { (b, a) } else { (a, b) };

    let mut sign = 1i32;
    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        // avoid loss of precision in lgam(a + b) - lgam(a)
        let y = betaln_asymp(a, b, &mut sign);
        return sign as f64 * y.exp();
    }

    let y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let mut sgngam = 1i32;
        let y = gammaln_sign(y, &mut sgngam);
        sign *= sgngam;
        let y = gammaln_sign(b, &mut sgngam) - y;
        sign *= sgngam;
        let y = gammaln_sign(a, &mut sgngam) + y;
        sign *= sgngam;
        if y > MAXLOG {
            return sign as f64 * std::f64::INFINITY;
        }
        return sign as f64 * y.exp();
    }

    let y = gamma(y);
    if y == 0.0 {
        return sign as f64 * std::f64::INFINITY;
    }

    let a = gamma(a);
    let b = gamma(b);
    if (a.abs() - y.abs()).abs() > (b.abs() - y.abs()).abs() {
        let y = b / y;
        y * a
    } else {
        let y = a / y;
        y * b
    }
}

fn beta_negint(a: i32, b: f64) -> f64 {
    let bi = b as i32;
    if b == bi as f64 && 1.0 - a as f64 - b > 0.0 {
        let sign = if bi % 2 == 0 { 1i32 } else { -1 };
        sign as f64 * beta(1.0 - a as f64 - b, b)
    } else {
        std::f64::INFINITY
    }
}

pub fn betaln(a: f64, b: f64) -> f64 {
    if a <= 0.0 && a == a.floor() {
        let ai = a as i32;
        if a == ai as f64 {
            return betaln_negint(ai, b);
        } else {
            return std::f64::INFINITY;
        }
    }

    if b <= 0.0 && b == b.floor() {
        let bi = b as i32;
        if b == bi as f64 {
            return betaln_negint(bi, a);
        } else {
            return std::f64::INFINITY;
        }
    }

    let (a, b) = if a.abs() < b.abs() { (b, a) } else { (a, b) };

    if a.abs() > ASYMP_FACTOR * b.abs() && a > ASYMP_FACTOR {
        let mut sign = 1i32;
        return betaln_asymp(a, b, &mut sign);
    }

    let y = a + b;
    if y.abs() > MAXGAM || a.abs() > MAXGAM || b.abs() > MAXGAM {
        let mut sgngam = 1i32;
        let y = gammaln_sign(y, &mut sgngam);
        let y = gammaln_sign(b, &mut sgngam) - y;
        let y = gammaln_sign(a, &mut sgngam) + y;
        return y;
    }

    let y = gamma(y);
    if y == 0.0 {
        return std::f64::INFINITY;
    }

    let a = gamma(a);
    let b = gamma(b);
    let y = if (a.abs() - y.abs()).abs() > (b.abs() - y.abs()).abs() {
        let y = b / y;
        y * a
    } else {
        let y = a / y;
        y * b
    };

    y.abs().ln()
}

fn betaln_asymp(a: f64, b: f64, sign: &mut i32) -> f64 {
    let mut r = gammaln_sign(b, sign);
    r -= b * a.ln();

    r += b * (1.0 - b) / (2.0 * a);
    r += b * (1.0 - b) * (1.0 - 2.0 * b) / (12.0 * a * a);
    r -= b * b * (1.0 - b) * (1.0 - b) / (12.0 * a * a * a);

    r
}

fn betaln_negint(a: i32, b: f64) -> f64 {
    let bi = b as i32;
    if b == bi as f64 && 1.0 - a as f64 - b > 0.0 {
        betaln(1.0 - a as f64 - b, b)
    } else {
        std::f64::INFINITY
    }
}
