#![cfg_attr(rustfmt, rustfmt_skip)]

use crate::gammaln;
use approx::assert_abs_diff_eq;

const EPS: f64 = 1e-9;

#[test]
fn test_gammaln() {
    // this corpus was generated using cargo-fuzzcheck, in an effort to exercise many codepaths
    assert_abs_diff_eq!(gammaln(-29629400.14014041), -480123034.718047, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(4.429573899359102), 2.3565442400806806, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-2.0739033318514786e-12), 26.901588616730617, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(7.382743961174512), 7.307081078840936, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-1.7657446926169909), 1.0551362684952932, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-7.07024205371492), -6.00313426343079, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-3.539042595484911), -1.355879595849078, epsilon = EPS);
    assert!(gammaln(std::f64::NAN).is_nan());
    assert_abs_diff_eq!(gammaln(5.398338681547479), 3.7950652248147976, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-474103171.8672542), -8997025330.33021, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-452.10876678681046), -2313.78732269773, epsilon = EPS);
    assert!(gammaln(1.557_376_483_595_712e306) == std::f64::INFINITY);
    assert_abs_diff_eq!(gammaln(6.413979178980679), 5.5089525209185615, epsilon = EPS);
    assert!(gammaln(-2.042_751_265_974_212e154) == std::f64::INFINITY);
    assert_abs_diff_eq!(gammaln(-1.2178952235844431e-154), 354.40098017882525, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-28.265098832087116), -67.3329902136446, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(12.835526832822119), 19.57284234629955, epsilon = EPS);
    assert_abs_diff_eq!(gammaln(-32.980521918390856), -81.0469841647776, epsilon = EPS);
}
