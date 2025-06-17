#![allow(non_snake_case)]

use super::{DataMat, Pca, PcaResult};
use anyhow::{format_err, Error};
use ndarray::linalg::Dot;
use ndarray::{s, Array2, ArrayView2};
use ndarray_linalg::svddc::JobSvd;
use ndarray_linalg::{SVDDCInto, QR};
use rand::distr::{Distribution, Uniform};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use snoop::CancelProgress;

/// Based on the "Randomized Block Krylov Methods for Stronger and Faster Approximate
/// Singular Value Decomposition", by Cameron and Christopher Musco, NIPS 2015
/// <https://papers.nips.cc/paper/5735-randomized-block-krylov-methods-for-stronger-and-faster-approximate-singular-value-decomposition.pdf>

/// Settings for Block Krylov SVD
pub struct BkSvd {
    /// Multiple of the requested k to use as block size in randomized projections,
    /// must be >= 1.0
    pub k_multiplier: f64,

    /// Number of power iteration to perform
    pub n_iter: usize,
}

impl BkSvd {
    /// Create a new BkSvd with default settings.
    pub fn new() -> BkSvd {
        BkSvd {
            k_multiplier: 2.0,
            n_iter: 5,
        }
    }
}

impl Default for BkSvd {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pca<T, f64> for BkSvd
where
    T: DataMat + for<'a> Dot<ArrayView2<'a, f64>, Output = Array2<f64>> + Dot<Array2<f64>, Output = Array2<f64>>,
    for<'a> ArrayView2<'a, f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<Array2<f64>, Output = Array2<f64>>,
{
    fn run_pca_cancellable(&self, array: &T, k: usize, snoop: impl CancelProgress) -> Result<PcaResult, Error> {
        let bsize = (k as f64 * self.k_multiplier).ceil() as usize;
        let (u, s, vt) = svd_bk(array, k, bsize, self.n_iter, 0, snoop)?;
        Ok((u, s, vt.reversed_axes()))
    }
}

/// Perform an SVD of matrix `A`, making a rank `k` approximation. Use `l` projection dimensions and `n_iter` power iterations.
#[inline(never)]
pub fn svd_bk<T>(
    A: &T,
    k: usize,      // svd rank
    b: usize,      // size of blocks, must be >= k
    n_iter: usize, // number of blocks
    seed: u64,
    mut snoop: impl CancelProgress,
) -> Result<PcaResult, Error>
where
    T: DataMat + for<'a> Dot<ArrayView2<'a, f64>, Output = Array2<f64>> + Dot<Array2<f64>, Output = Array2<f64>>,
    for<'a> ArrayView2<'a, f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<Array2<f64>, Output = Array2<f64>>,
{
    let [m, n] = A.shape();

    if m < 2 || n < 2 {
        return Err(format_err!("The input matrix must be at least 2x2."));
    }

    if k > std::cmp::min(m, n) {
        return Err(format_err!("invalid k"));
    }

    let b = std::cmp::min(std::cmp::min(m, n), b);

    let mut rng = SmallRng::seed_from_u64(seed);
    let unif = Uniform::new(-1.0, 1.0).unwrap();
    // FIXME: Additional cases to handle
    // - fall through to straight svd when l/k is within ~25% of m or n.
    // - handle case of n > m

    if m >= n {
        let mut B = Array2::from_shape_simple_fn((n, b), || unif.sample(&mut rng));
        let mut K = Array2::<f64>::zeros((n, b * n_iter));

        for i in 0..n_iter {
            B = A.dot(&B).reversed_axes().dot(A).reversed_axes().qr()?.0;
            K.slice_mut(s![.., i * b..(i + 1) * b]).assign(&B);
            snoop.set_progress_check(i as f64 / n_iter as f64 * 0.8)?;
        }
        let Q = K.qr()?.0;
        snoop.set_progress_check(0.82)?;

        let (U, sigma, Va) = {
            let T = A.dot(&Q);
            snoop.set_progress_check(0.93)?;

            let svd = T.svddc_into(JobSvd::Some)?;
            (
                svd.0.unwrap().slice(s![.., ..k]).to_owned(),
                svd.1.slice(s![..k]).to_owned(),
                svd.2.unwrap().slice(s![..k, ..]).to_owned(),
            )
        };

        let Va = Va.dot(&Q.t());
        snoop.set_progress_check(1.0)?;
        Ok((U, sigma, Va))
    } else {
        // n > m
        let mut B = Array2::from_shape_simple_fn((b, m), || unif.sample(&mut rng));
        let mut K = Array2::<f64>::zeros((b * n_iter, m));

        for i in 0..n_iter {
            let T = B.dot(A).reversed_axes();
            B = A.dot(&T).qr()?.0.reversed_axes();
            K.slice_mut(s![i * b..(i + 1) * b, ..]).assign(&B);
            snoop.set_progress_check(i as f64 / n_iter as f64 * 0.8)?;
        }
        let Q = K.t().qr()?.0;
        snoop.set_progress_check(0.82)?;

        let (U, sigma, Va) = {
            let T = Q.t().dot(A);
            snoop.set_progress_check(0.93)?;

            let svd = T.svddc_into(JobSvd::Some)?;
            (
                svd.0.unwrap().slice(s![.., ..k]).to_owned(),
                svd.1.slice(s![..k]).to_owned(),
                svd.2.unwrap().slice(s![..k, ..]).to_owned(),
            )
        };

        let U = Q.dot(&U);
        snoop.set_progress_check(1.0)?;
        Ok((U, sigma, Va))
    }
}
