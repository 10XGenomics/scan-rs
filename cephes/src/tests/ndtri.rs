#![cfg_attr(rustfmt, rustfmt_skip)]

use crate::ndtri;
use approx::assert_abs_diff_eq;

const EPS: f64 = 1e-9;

#[test]
fn test_ndtri() {
    assert_abs_diff_eq!(ndtri(0.999), 3.090232306167813, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.99), 2.3263478740408408, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.9), 1.2815515655446004, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.8), 0.8416212335729143, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.7), 0.5244005127080407, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.6), 0.2533471031357997, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.5), 0.0, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.4), -0.2533471031357997, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.3), -0.5244005127080409, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.2), -0.8416212335729142, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.1), -1.2815515655446004, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.01), -2.3263478740408408, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.001), -3.090232306167813, epsilon = EPS);
    // this corpus was generated using cargo-fuzzcheck, in an effort to exercise many codepaths
    assert_abs_diff_eq!(ndtri(0.8682156981488498), 1.1179962295510544, epsilon = EPS);
    assert!(ndtri(-1.7657446926169909).is_nan());
    assert!(ndtri(std::f64::NAN).is_nan());
    assert_abs_diff_eq!(ndtri(2.2586453105439303e-12), -6.919973210677117, epsilon = EPS);
    assert_abs_diff_eq!(ndtri(0.1551595169687376), -1.0145528304620457, epsilon = EPS);
}
