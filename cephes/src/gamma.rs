use crate::polevl::{p1evl, polevl};
use std::f64::consts::PI;

const STIR: &[f64] = &[
    7.873_113_957_930_937E-4,
    -2.295_499_616_133_781_3E-4,
    -2.681_326_178_057_812_4E-3,
    3.472_222_216_054_586_6E-3,
    8.333_333_333_334_822E-2,
];
const MAXGAM: f64 = 171.624_376_956_302_7;
const MAXSTIR: f64 = 143.01608;
const SQTPI: f64 = 2.506_628_274_631_000_7;

pub fn stirf(x: f64) -> f64 {
    if x >= MAXGAM {
        return std::f64::INFINITY;
    }
    let w = 1.0 / x;
    let w = 1.0 + w * polevl(w, STIR);
    let y = x.exp();
    let y = if x > MAXSTIR {
        // avoid overflow in powf
        let v = x.powf(0.5 * x - 0.25);
        v * (v / y)
    } else {
        x.powf(x - 0.5) / y
    };
    SQTPI * y * w
}

const P: &[f64] = &[
    1.601_195_224_767_518_5E-4,
    1.191_351_470_065_863_8E-3,
    1.042_137_975_617_615_8E-2,
    4.763_678_004_571_372E-2,
    2.074_482_276_484_359_8E-1,
    4.942_148_268_014_971E-1,
    1.0,
];
const Q: &[f64] = &[
    -2.315_818_733_241_201_4E-5,
    5.396_055_804_933_034E-4,
    -4.456_419_138_517_973E-3,
    1.181_397_852_220_604_3E-2,
    3.582_363_986_054_986_5E-2,
    -2.345_917_957_182_433_5E-1,
    7.143_049_170_302_73E-2,
    1.0,
];

#[inline(always)]
fn small(x: f64, z: f64) -> f64 {
    if x == 0.0 {
        std::f64::NAN
    } else {
        z / ((1.0 + 0.5772156649015329 * x) * x)
    }
}

pub fn gamma(x: f64) -> f64 {
    if x.is_nan() || x == std::f64::INFINITY {
        return x;
    } else if x == std::f64::NEG_INFINITY {
        return std::f64::NAN;
    }
    let q = x.abs();

    if q > 33.0 {
        let mut sgngam = 1i32;
        let z = if x < 0.0 {
            let p = q.floor();
            if p == q {
                return std::f64::NAN;
            }
            let i = p as i32;
            if i & 1 == 0 {
                sgngam = -1;
            }
            let mut z = q - p;
            if z > 0.5 {
                z = q - (p + 1.0);
            }
            let z = q * (PI * z).sin();
            if z == 0.0 {
                return sgngam as f64 * std::f64::INFINITY;
            }
            let z = z.abs();
            PI / (z * stirf(q))
        } else {
            stirf(x)
        };
        return sgngam as f64 * z;
    }

    let mut x = x;
    let mut z = 1.0;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }

    while x < 0.0 {
        if x > -1e-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }

    while x < 2.0 {
        if x < 1e-9 {
            return small(x, z);
        }
        z /= x;
        x += 1.0;
    }

    if x == 2.0 {
        return z;
    }

    x -= 2.0;
    let p = polevl(x, P);
    let q = polevl(x, Q);
    z * p / q
}

const A: &[f64] = &[
    8.116_141_674_705_085E-4,
    -5.950_619_042_843_014E-4,
    7.936_503_404_577_169E-4,
    -2.777_777_777_300_997E-3,
    8.333_333_333_333_319E-2,
];
const B: &[f64] = &[
    -1.378_251_525_691_208_6E3,
    -3.880_163_151_346_378_4E4,
    -3.316_129_927_388_712E5,
    -1.162_370_974_927_623E6,
    -1.721_737_008_208_396_6E6,
    -8.535_556_642_457_654E5,
];
const C: &[f64] = &[
    -3.518_157_014_365_234_5E2,
    -1.706_421_066_518_811_5E4,
    -2.205_285_905_538_544_5E5,
    -1.139_334_443_679_825_2E6,
    -2.532_523_071_775_829_4E6,
    -2.018_891_414_335_327_7E6,
];
const LOGPI: f64 = 1.144_729_885_849_400_2;
const LS2PI: f64 = 0.918_938_533_204_672_8;
const MAXLGM: f64 = 2.556348e305;

pub(crate) fn gammaln_sign(x: f64, sgngam: &mut i32) -> f64 {
    *sgngam = 1;
    if x.is_nan() || x.is_infinite() {
        return x;
    }
    if x < -34.0 {
        let q = -x;
        let w = gammaln_sign(q, sgngam);
        let p = q.floor();
        if p == q {
            return std::f64::INFINITY;
        }
        let i = p as i32;
        if (i & 1) == 0 {
            *sgngam = -1;
        } else {
            *sgngam = 1;
        }
        let mut z = q - p;
        if z > 0.5 {
            z = (p + 1.0) - q;
        };
        let z = q * (PI * z).sin();
        if z == 0.0 {
            return std::f64::INFINITY;
        }
        return LOGPI - z.ln() - w;
    }
    if x < 13.0 {
        let mut z = 1.0;
        let mut p = 0.0;
        let mut u = x;
        while u >= 3.0 {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while u < 2.0 {
            if u == 0.0 {
                return std::f64::INFINITY;
            }
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if z < 0.0 {
            *sgngam = -1;
            z = -z;
        } else {
            *sgngam = 1;
        }
        if u == 2.0 {
            return z.ln();
        }
        p -= 2.0;
        let x = x + p;
        let p = x * polevl(x, B) / p1evl(x, C);
        return z.ln() + p;
    }
    if x > MAXLGM {
        return (*sgngam) as f64 * std::f64::INFINITY;
    }
    let q = (x - 0.5) * x.ln() - x + LS2PI;
    if x > 1e8 {
        return q;
    }
    let p = 1.0 / (x * x);
    if x > 1000.0 {
        q + ((7.936_507_936_507_937e-4 * p - 2.777_777_777_777_778e-3) * p + 0.083_333_333_333_333_33) / x
    } else {
        q + polevl(p, A) / x
    }
}

pub fn gammaln(x: f64) -> f64 {
    let mut sgngam = 1i32;
    gammaln_sign(x, &mut sgngam)
}
