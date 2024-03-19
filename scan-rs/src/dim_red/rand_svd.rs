#![allow(non_snake_case)]

use super::{DataMat, Pca, PcaResult};
use anyhow::{format_err, Error};
use ndarray::linalg::Dot;
use ndarray::{s, Array2, ArrayView2};
use ndarray_linalg::svddc::JobSvd;
use ndarray_linalg::{SVDDCInto, QR};
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use snoop::CancelProgress;

/// Settings for Randomized PCA
pub struct RandSvd {
    /// Multiple of the requested k to use in randomized projections
    pub l_multiplier: f64,

    /// Number of power iteration to perform
    pub n_iter: usize,
}

impl RandSvd {
    /// Create a new RandSvd with default settings.
    pub fn new() -> RandSvd {
        RandSvd {
            l_multiplier: 10.0,
            n_iter: 2,
        }
    }
}

impl Default for RandSvd {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pca<T, f64> for RandSvd
where
    T: DataMat + for<'a> Dot<ArrayView2<'a, f64>, Output = Array2<f64>> + Dot<Array2<f64>, Output = Array2<f64>>,
    for<'a> ArrayView2<'a, f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<Array2<f64>, Output = Array2<f64>>,
{
    fn run_pca_cancellable(&self, array: &T, k: usize, _snoop: impl CancelProgress) -> Result<PcaResult, Error> {
        // not bothering to do progress/cancellation for this one.
        let l = std::cmp::max(k + 4, ((k as f64) * self.l_multiplier) as usize);
        let (u, s, vt) = svd_rand(array, k, l, self.n_iter, 0)?;
        Ok((u, s, vt.reversed_axes()))
    }
}

/// Perform an SVD of matrix `A`, making a rank `k` approximation. Use `l` projection dimensions and `n_iter` power iterations.
#[inline(never)]
pub fn svd_rand<T>(
    A: &T,
    k: usize, // svd rank
    l: usize,
    n_iter: usize, // power iterations
    seed: u64,
) -> Result<PcaResult, Error>
where
    T: DataMat + for<'a> Dot<ArrayView2<'a, f64>, Output = Array2<f64>> + Dot<Array2<f64>, Output = Array2<f64>>,
    for<'a> ArrayView2<'a, f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<Array2<f64>, Output = Array2<f64>>,
{
    let m = A.shape()[0];
    let n = A.shape()[1];

    if m < 2 || n < 2 {
        return Err(format_err!("The input matrix must be at least 2x2."));
    }

    if k > std::cmp::min(m, n) {
        return Err(format_err!("invalid k"));
    }

    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let unif = Uniform::new(-1.0, 1.0);

    // FIXME: Additional cases to handle
    // - fall through to straight svd when l/k is within ~25% of m or n.
    // - handle case of n > m

    if m >= n {
        let omega = Array2::random_using((n, l), unif, &mut rng);
        let mut Q: Array2<f64> = A.dot(&omega).qr()?.0;

        for _ in 0..n_iter {
            Q = Q.t().dot(A).reversed_axes().qr()?.0;
            Q = A.dot(&Q).qr()?.0;
        }

        let (U, sigma, Va) = {
            let B = Q.t().dot(A);
            let svd = B.svddc_into(JobSvd::Some)?;
            (
                svd.0.unwrap().slice(s![.., ..k]).to_owned(),
                svd.1.slice(s![..k]).to_owned(),
                svd.2.unwrap().slice(s![..k, ..]).to_owned(),
            )
        };

        let U = Q.dot(&U);
        Ok((U, sigma, Va))
    } else {
        // n > m
        let omega = Array2::random_using((l, m), unif, &mut rng);
        let mut Q = omega.dot(A).reversed_axes().qr()?.0;

        for _ in 0..n_iter {
            Q = A.dot(&Q).qr()?.0;
            Q = Q.t().dot(A).reversed_axes().qr()?.0;
        }

        let (U, sigma, Va) = {
            let B = A.dot(&Q);
            let svd = B.svddc_into(JobSvd::Some)?;
            (
                svd.0.unwrap().slice(s![.., ..k]).to_owned(),
                svd.1.slice(s![..k]).to_owned(),
                svd.2.unwrap().slice(s![..k, ..]).to_owned(),
            )
        };

        let Va = Va.dot(&Q.t());
        Ok((U, sigma, Va))
    }
}
