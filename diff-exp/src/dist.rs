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
    let mut min = f64::MAX;
    for (idx, (_, val)) in arr.iter_mut().enumerate() {
        *val *= len / (len - idx as f64);
        if *val < min {
            min = *val;
        }
        *val = min.min(1.0);
    }

    arr
}

/// Backend selection for the negative-binomial pairwise exact test.
///
/// `LogSpace` is the original, always-valid log-sum-exp implementation
/// ([`nb_exact_test`]). `Ratio` selects the transcendental-free, mode-anchored
/// ratio recurrence ([`nb_exact_test_ratio`]), which is algebraically identical
/// but deletes the per-term log/exp/gamma work; it falls back to `LogSpace` when
/// the observed term underflows. `LogSpace` is the default and is retained
/// permanently as both the fallback and the differential test oracle.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NbExactBackend {
    /// Log-space log-sum-exp implementation (default).
    #[default]
    LogSpace,
    /// Mode-anchored rational ratio recurrence.
    Ratio,
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
    let mut max_all = f64::NEG_INFINITY;
    let mut max_ext = f64::NEG_INFINITY;
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

/// Ratio of consecutive unnormalized terms `T(k+1)/T(k)` of the conditional NB
/// distribution summed by the exact test. Purely rational — no log/exp/gamma:
///
/// `T(k+1)/T(k) = (sa_r + k)(n - k) / ((k + 1)(sb_r + n - k - 1))`,
///
/// where `sa_r = size_factor_a / phi` and `sb_r = size_factor_b / phi`. Used to
/// sweep the term array in [`nb_exact_test_ratio`] without transcendental calls.
#[inline]
fn nb_exact_ratio_step(k: f64, n: f64, sa_r: f64, sb_r: f64) -> f64 {
    (sa_r + k) * (n - k) / ((k + 1.0) * (sb_r + n - k - 1.0))
}

/// nb_exact_test_ratio
/// Mode-anchored, transcendental-free variant of [`nb_exact_test`] that computes
/// the same conditional exact-test p-value via the multiplicative ratio
/// recurrence (`nb_exact_ratio_step`) rather than a log-sum-exp over per-term
/// gamma values. The sweep is anchored at the first downward step (`U[anchor] =
/// 1`); the normalizing constant cancels in `sum_ext / sum_all`, so the choice of
/// anchor affects only numerical range, not the result.
///
/// For the usual unimodal distribution the anchor is the mode (the sequence
/// maximum), so every other term is `<= 1` and nothing can overflow. In the
/// U-shaped regime (both `sa_r = size_factor_a/phi` and `sb_r = size_factor_b/phi`
/// below 1) the distribution is bimodal with maxima at the boundaries `k = 0` and
/// `k = N`; the anchor then lands on the `k = 0` boundary, which is only a *local*
/// maximum. Terms can therefore exceed 1 while sweeping up toward `k = N`, but the
/// growth is bounded by `U[N]/U[0] ~ N^(sa_r - sb_r)` with `|sa_r - sb_r| < 1` —
/// sub-linear in `N`. For realistic O(1) size factors (and `size_factor == 0`
/// guarded below) this stays many orders of magnitude below the `f64` overflow
/// threshold even at `N ~ 800k`, so no term overflows in practice. Far-tail terms
/// merely underflow toward 0.
///
/// When the observed term `U[x_a]` underflows to `0.0` (or is non-finite) the
/// rational partition would return a spurious exact `0.0`, so this falls back to
/// the validated log-space [`nb_exact_test`] (which anchors on the observed
/// term and stays accurate down to ~1e-308). The degenerate-input guards
/// (`x_a + x_b == 0`, `phi == 0`, `size_factor == 0`) return exactly `1.0`,
/// bit-identical to [`nb_exact_test`].
#[inline]
pub fn nb_exact_test_ratio(x_a: u64, x_b: u64, size_factor_a: f64, size_factor_b: f64, mu: f64, phi: f64) -> f64 {
    if x_a + x_b == 0u64 {
        return 1f64;
    }

    if phi == 0f64 {
        return 1f64;
    }

    if size_factor_a == 0f64 || size_factor_b == 0f64 {
        return 1f64;
    }

    let n = x_a + x_b;
    let nn = n as f64;
    let r = 1f64 / phi;
    let sa_r = size_factor_a * r;
    let sb_r = size_factor_b * r;

    // Anchor at the first k whose forward ratio drops below 1 (else n). For a
    // unimodal distribution this is the mode (global max); in the U-shaped regime
    // it is the k=0 boundary (a local max) — see the doc comment for why the
    // anchor choice is correct either way and stays within f64 range.
    let mut mode = n;
    for k in 0..n {
        if nb_exact_ratio_step(k as f64, nn, sa_r, sb_r) < 1.0 {
            mode = k;
            break;
        }
    }

    // Build the unnormalized terms anchored at U[mode] = 1, sweeping out in both
    // directions. The large normalizing constant cancels in sum_ext / sum_all.
    let mut u = vec![0f64; n as usize + 1];
    u[mode as usize] = 1f64;
    for k in mode..n {
        u[(k + 1) as usize] = u[k as usize] * nb_exact_ratio_step(k as f64, nn, sa_r, sb_r);
    }
    for k in (0..mode).rev() {
        u[k as usize] = u[(k + 1) as usize] / nb_exact_ratio_step(k as f64, nn, sa_r, sb_r);
    }

    let u_obs = u[x_a as usize];

    // Deep-tail fallback: a 0.0/non-finite observed term would make the
    // rational partition return a spurious 0.0, so defer to the log-space method.
    if u_obs == 0f64 || !u_obs.is_finite() {
        return nb_exact_test(x_a, x_b, size_factor_a, size_factor_b, mu, phi);
    }

    let mut sum_all = 0f64;
    let mut sum_ext = 0f64;
    for &v in &u {
        sum_all += v;
        if v <= u_obs {
            sum_ext += v;
        }
    }

    sum_ext / sum_all
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
    let mut j: f64 = x - 1f64;

    // additional term
    let add_total = x * (mu / (r + mu)).ln() + (sa + sb) * (r / (r + mu)).ln() - ln_gamma(sa * r) - ln_gamma(sb * r);

    // This function suffers from a pathology where for large count we calculate A LOT of ln_gamma functions
    // see https://github.com/10XDev/scan-rs/pull/254
    // However, we are walking in integer loops here so can use the relationship that
    // ln_gamma(x+1) = ln_gamma(x) + ln(x) to avoid doing expensive lngamma calculations recurrently.
    // as well as the fact that we're calculating ln_gamma for 0..count and count..0, so can just
    // calculate the value for (count+1) once and add it at the start/end of the array

    // Initial computation inside loop before this change was an easier to read version:
    // Note with this code the current `j` was equal to `j+1` but I changed it above.
    //  for idx in 0..=count {
    //     let a_x = idx as f64;
    //     let t = ln_gamma(sa * r + a_x) + ln_gamma(sb * r + j) - (ln_gamma(a_x + 1f64) + ln_gamma(j + 1f64));
    //     total.push(t + add_total);
    //     j -= 1f64;
    //  }

    // --- Precompute initial values for idx = 0, j = count ---
    let mut ln_a = ln_gamma(sa * r); // Γ(sa*r)
    let mut ln_b = ln_gamma(sb * r + x); // Γ(sb*r + count)
    for idx in 0..=count {
        //let t = ln_gamma(sa * r + a_x) + ln_gamma(sb * r + j - 1) + add_total;
        let t = ln_a + ln_b + add_total;
        total.push(t);
        let idxf = idx as f64;

        // prepare for next iteration
        // Note technically we don't need to do this when idx==count but I wanted to avoid an if statement here
        // as (untested) it might slow pipelining due to branching that could be slower than just taking 2 extra logs
        //at the end of the loop

        //if idx < count {
        ln_a += (sa * r + idxf).ln(); // ln Γ(sa*r + idx + 1)
        ln_b -= (sb * r + j).ln(); // stepping downward
        j -= 1f64;
        //}
    }
    // Now subtract the terms for lngamma(a_x+1) and lngamma(count - a_x + 1), doing the start..end and end..start iteration with one pass
    let mut ln_index = 0.0; // ln Γ(1) = 0
    for idx in 0..=count {
        total[idx as usize] -= ln_index;
        total[(count - idx) as usize] -= ln_index;
        ln_index += (idx as f64 + 1.0).ln();
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

        let expected: Vec<(usize, f64)> = [
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

        let expected = [
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

    // Tier 1 of the ratio-vs-log-space equivalence suite (see the module
    // overview in `tests/ratio_backend.rs` for the full hierarchy). It lives
    // here, not there, because it is a white-box test of the *private*
    // `nb_exact_ratio_step` / `log_prob_all` that an integration test cannot see.
    //
    // Isolates the ratio recurrence step from summation / partition logic by
    // comparing `nb_exact_ratio_step` against the exact ratio of consecutive
    // log-space terms produced by `log_prob_all`.
    //
    // Identity under test:
    //   nb_exact_ratio_step(k, N, sa*r, sb*r) == exp(lp[k+1] - lp[k])
    // where lp = log_prob_all(N, sa, sb, mu, r), r = 1/phi. The recurrence is
    // independent of mu (mu cancels), so mu is held at an arbitrary value.
    #[test]
    fn test_nb_exact_ratio_step_matches_logspace() {
        let sas = [0.6f64, 1.2, 2.0, 3.0];
        let sbs = [0.6f64, 1.2, 2.0, 3.0];
        let phis = [0.05f64, 0.3, 1.0, 2.0];
        let ns = [10u64, 50, 200];
        let mu = 5.0f64; // arbitrary; recurrence does not depend on mu.

        for &sa in &sas {
            for &sb in &sbs {
                for &phi in &phis {
                    for &n in &ns {
                        let r = 1.0 / phi;
                        let lp = log_prob_all(n, sa, sb, mu, r);
                        assert_eq!(
                            lp.len(),
                            (n + 1) as usize,
                            "log_prob_all should return N+1 terms (sa={sa}, sb={sb}, phi={phi}, N={n})"
                        );
                        for k in 0..n {
                            let step = nb_exact_ratio_step(k as f64, n as f64, sa * r, sb * r);
                            let expected = (lp[(k + 1) as usize] - lp[k as usize]).exp();
                            // relative tolerance ~1e-12 with a small absolute floor.
                            let tol = 1e-10 + 1e-9 * expected.abs();
                            assert!(
                                (step - expected).abs() <= tol,
                                "ratio step mismatch: sa={sa}, sb={sb}, phi={phi}, N={n}, k={k}: \
                                 step={step}, expected={expected}, diff={}, tol={tol}",
                                (step - expected).abs()
                            );
                        }
                    }
                }
            }
        }
    }
}
