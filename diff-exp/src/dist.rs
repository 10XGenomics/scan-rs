use crate::gamma::ln_gamma;
use cephes::{betainc, betaincinv as beta_ppf};
// use cached::proc_macro::cached;
use std::cmp::Ordering;

/// Log(PMF) of negative binomial distribution with mean mu and dispersion phi,
/// conveniently parameterized.
/// Args:
///  k  - NB random variable
///  u  - mean
///  phi  - dispersion
/// Returns:
///  The log of the pmf at k.
#[inline]
pub fn negative_binomial_log_pmf(k: f64, mu: f64, phi: f64) -> f64 {
    let r = 1.0 / phi;
    ln_gamma(r + k) - (ln_gamma(r) + ln_gamma(k + 1.0)) + k * (mu / (r + mu)).ln() + r * (r / (r + mu)).ln()
}

/// adjusted_pvalue_bh
#[inline]
pub fn adjusted_pvalue_bh(pvalue: &[(usize, f64)]) -> Vec<(usize, f64)> {
    // sort pvalue and conserve the original indexes, NaNs to the front
    let mut arr = pvalue.to_vec();
    arr.sort_by(|&(_, a), &(_, b)| match a.partial_cmp(&b) {
        Some(o) => o.reverse(),
        None => {
            if a.is_nan() && b.is_nan() {
                Ordering::Equal
            } else if a.is_nan() {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }
    });

    // compute q = np.minimum(1, np.minimum.accumulate(scale * p[descending])
    let len = arr.len() as f64;
    let mut min = std::f64::MAX;
    for (idx, (_, ref mut val)) in arr.iter_mut().enumerate() {
        *val *= len / (len - idx as f64);
        if *val < min {
            min = *val
        }
        *val = min.min(1.0);
    }

    arr
}

/// nb_exact_test
/// compute p-value; the probability that a random pair of counts under the null hypothesis is more extreme
/// than the observed pair of counts.
#[inline]
pub fn nb_exact_test(x_a: u64, x_b: u64, size_factor_a: f64, size_factor_b: f64, mu: f64, phi: f64) -> f64 {
    if x_a + x_b == 0u64 {
        return 1f64;
    }

    if phi == 0f64 {
        return 1f64;
    }

    if size_factor_a == 0f64 || size_factor_b == 0f64 {
        return 1f64;
    }

    let log_p_all = log_prob_all(x_a + x_b, size_factor_a, size_factor_b, mu, 1f64 / phi);
    let log_p_obs = log_p_all[x_a as usize];

    // to avoid several iterations or copies of this, we're going to fold two log_sum_exp together:
    // - first collect the maximum value
    // - then compute the log_sum_exp
    let mut max_all = std::f64::NEG_INFINITY;
    let mut max_ext = std::f64::NEG_INFINITY;
    for &x in &log_p_all {
        if x <= log_p_obs {
            max_ext = max_ext.max(x);
        }
        max_all = max_all.max(x);
    }

    let mut sum_all = 0.0;
    let mut sum_ext = 0.0;
    for &x in &log_p_all {
        if x <= log_p_obs {
            sum_ext += (x - max_ext).exp();
        }
        sum_all += (x - max_all).exp();
    }
    sum_all = sum_all.ln() + max_all;
    sum_ext = sum_ext.ln() + max_ext;

    (sum_ext - sum_all).exp()
}

fn beta_cdf(a: f64, b: f64, x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        betainc(a, b, x)
    }
}

/// Compute p-value for a pairwise exact test using a fast beta approximation
/// to the conditional joint distribution of (x_a, x_b).
/// Robinson MD and Smyth GK (2008). Small-sample estimation of negative binomial dispersion,
///     with applications to SAGE data. Biostatistics, 9, 321-332
///  "It is based a method-of-moments gamma approximation to the negative binomial distribution."
///     - Personal communication w/ author
#[inline]
pub fn nb_asymptotic_test(
    count_a: u64,
    count_b: u64,
    size_factor_a: f64,
    size_factor_b: f64,
    mu: f64,
    phi: f64,
) -> f64 {
    let alpha = size_factor_a * mu / (1f64 + phi * mu);
    let beta = (size_factor_b / size_factor_a) * alpha;

    let x_a = count_a as f64;
    let x_b = count_b as f64;

    let median = beta_ppf(alpha, beta, 0.5);

    if (x_a + 0.5f64) / (x_a + x_b) < median {
        2f64 * beta_cdf(alpha, beta, (x_a + 0.5f64) / (x_a + x_b))
    } else {
        2f64 * beta_cdf(beta, alpha, (x_b + 0.5f64) / (x_a + x_b))
    }
}

/// log_prob_all
#[inline]
fn log_prob_all(count: u64, sa: f64, sb: f64, mu: f64, r: f64) -> Vec<f64> {
    let mut total = Vec::<f64>::with_capacity(count as usize + 1);
    let x = count as f64;
    let mut j: f64 = x;

    // additional term
    let add_total = x * (mu / (r + mu)).ln() + (sa + sb) * (r / (r + mu)).ln() - ln_gamma(sa * r) - ln_gamma(sb * r);

    for idx in 0..=count {
        let a_x = idx as f64;
        let t = ln_gamma(sa * r + a_x) + ln_gamma(sb * r + j) - (ln_gamma(a_x + 1f64) + ln_gamma(j + 1f64));
        total.push(t + add_total);
        j -= 1f64;
    }
    total
}

#[cfg(test)]
mod test {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_adjusted_pvalue_bh() {
        let data = [
            ("Blue_fish", 0.34f64),
            ("Bread", 0.594f64),
            ("Butter", 0.212f64),
            ("Carbohydrates", 0.384f64),
            ("Cereals_and_pasta", 0.074f64),
            ("Dairy_products", 0.94f64),
            ("Eggs", 0.275f64),
            ("Fats", 0.696f64),
            ("Fruit", 0.269f64),
            ("Legumes", 0.341f64),
            ("Nuts", 0.06f64),
            ("Olive_oil", 0.008f64),
            ("Potatoes", 0.569f64),
            ("Processed_meat", 0.986f64),
            ("Proteins", 0.042f64),
            ("Red_meat", 0.251f64),
            ("Semi-skimmed_milk", 0.942f64),
            ("Skimmed_milk", 0.222f64),
            ("Sweets", 0.762f64),
            ("Total_calories", 0.001f64),
            ("Total_meat", 0.975f64),
            ("Vegetables", 0.216f64),
            ("White_fish", 0.205f64),
            ("White_meat", 0.041f64),
            ("Whole_milk", 0.039f64),
        ];
        let pv_value: Vec<(usize, f64)> = data.iter().enumerate().map(|x| (x.0, (x.1).1)).collect();

        let expected: Vec<(usize, f64)> = vec![
            0.5328125f64,
            0.781578947368421,
            0.49107142857142866,
            0.5647058823529413,
            0.2642857142857143,
            0.986,
            0.49107142857142866,
            0.8699999999999999,
            0.49107142857142866,
            0.5328125,
            0.25,
            0.1,
            0.781578947368421,
            0.986,
            0.21000000000000002,
            0.49107142857142866,
            0.986,
            0.49107142857142866,
            0.9071428571428571,
            0.025,
            0.986,
            0.49107142857142866,
            0.49107142857142866,
            0.21000000000000002,
            0.21000000000000002,
        ]
        .iter()
        .enumerate()
        .map(|(idx, &val)| (idx, val))
        .collect();

        let mut adjusted_pvs = adjusted_pvalue_bh(&pv_value);
        adjusted_pvs.sort_by_key(|&(i, _)| i);
        assert_eq!(expected, adjusted_pvs);
    }

    #[test]
    fn test_log_prob_all() {
        let sa = 2f64;
        let sb = 3f64;
        let mu = 3f64;
        let phi = 2f64;
        let x_a = 4u64;
        let x_b = 6u64;
        let count = x_a + x_b;
        let r = 1f64 / phi;

        let res = log_prob_all(count, sa, sb, mu, r);

        let expected = vec![
            -9.962687402422226f64,
            -10.011477566591564,
            -10.065544787861924,
            -10.126169409678283,
            -10.195162281165276,
            -10.27520498883885,
            -10.370515168643138,
            -10.488298204299513,
            -10.642448884126784,
            -10.865592435440986,
            -11.271057543549151,
        ];

        for (&e, &r) in expected.iter().zip(&res) {
            assert_approx_eq!(e, r, 1e-5);
        }
    }

    #[test]
    fn test_nb_exact_test() {
        let size_factor_a = 885.743_286_299_499_5_f64;
        let size_factor_b = 2023.055530268548f64;
        let x_a = 6u64;
        let x_b = 3u64;
        let mu = 0.0029272959469517066f64;
        let phi = 27.024221110009037f64;

        let res = nb_exact_test(x_a, x_b, size_factor_a, size_factor_b, mu, phi);

        assert_approx_eq!(0.03254f64, res, 0.00001f64);
    }
    #[test]
    fn test_nb_asymptotic_test() {
        let size_factor_a = 885.743_286_299_499_5_f64;
        let size_factor_b = 2023.055530268548f64;
        let cond_a = 1792u64;
        let cond_b = 1436u64;
        let mu = 1.0159265507499822f64;
        let phi = 29.483072138841884f64;

        let res = nb_asymptotic_test(cond_a, cond_b, size_factor_a, size_factor_b, mu, phi);

        assert_approx_eq!(7.2549e-07, res, 1e-11f64);
    }
}
