#![cfg_attr(rustfmt, rustfmt_skip)]

use crate::gamma;
use approx::assert_abs_diff_eq;

const EPS: f64 = 1e-9;

#[test]
fn test_gamma() {
    // this corpus was generated using cargo-fuzzcheck, in an effort to exercise many codepaths
    assert_abs_diff_eq!(gamma(4.429573899359102), 10.554414834148393, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-113.08773949558939), 3.415561605492608e-184, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-2.546862844206968), -0.9071705039516926, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-162.80146510546749), -7.381071768511042e-291, epsilon = EPS);
    assert!(gamma(2.147_500_404_792_697_2e154) == f64::INFINITY);
    assert_abs_diff_eq!(gamma(7.382743961174512), 1490.8192481946378, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-1.7657446926169909), 2.8723665403002996, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-7.07024205371492), 0.0024709952767988973, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-3.539042595484911), 0.25772050483856823, epsilon = EPS);
    assert!(gamma(f64::NAN).is_nan());
    assert_abs_diff_eq!(gamma(14.773056397209785), 47599110938.18181, epsilon = EPS);
    assert_abs_diff_eq!(gamma(5.398338681547479), 44.48113758487423, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-474103171.8672542), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-452.10876678681046), -0.0, epsilon = EPS);
    assert_abs_diff_eq!(gamma(6.413979178980679), 246.8923769733807, epsilon = EPS);
    // TODO: scipy says this is +inf, but we compute NaN here, either seems fine
    assert!(gamma(-2.042_751_265_974_212e154).is_nan());
    assert_abs_diff_eq!(gamma(-1.2178952235844431e-154), -8.210_886_951_808_994e153, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-28.265098832087116), -5.723397377422474e-30, epsilon = EPS);
    assert_abs_diff_eq!(gamma(118.08777001498584), 6.032_331_992_455_862e192, epsilon = EPS);
    assert_abs_diff_eq!(gamma(29.515106461936398), 1.720_131_496_003_831_4e30, epsilon = EPS);
    assert_abs_diff_eq!(gamma(1.293644352290662e-154), 7.730_099_839_490_625e153, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-14.148052582290601), -5.403384873362962e-11, epsilon = EPS);
    assert_abs_diff_eq!(gamma(158.8013125166455), 6.77446211509801e+279, epsilon = EPS);
    assert_abs_diff_eq!(gamma(-32.980521918390856), -6.334932654461348e-36, epsilon = EPS);
}
