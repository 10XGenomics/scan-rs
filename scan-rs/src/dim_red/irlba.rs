#![allow(non_snake_case)]

use super::{DataMat, Pca, PcaResult};
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::{s, Array1, Array2, LinalgScalar};
use ndarray_linalg::SVD;
use ndarray_rand::RandomExt;
use num_traits::Float;
use rand_distr::Normal;
use snoop::CancelProgress;
use std::cmp::{max, min};
use std::ops::Mul;

fn norm<T: LinalgScalar + Mul + Float>(x: &ArrayView1<T>) -> T {
    x.fold(T::zero(), |sum, v| sum + (*v) * (*v)).sqrt()
}

/// Orthogonalize a vector or matrix Y against the columns of the matrix X.
/// This function requires that the column dimension of Y is less than X and
/// that Y and X have the same number of rows.
fn orthog<T: LinalgScalar>(y: &ArrayView1<T>, x: &ArrayView2<T>) -> Array1<T> {
    let dot_y = &x.t().dot(y);
    y - &x.dot(dot_y)
}

/// utility function used to check linear dependencies during computation:
fn invcheck<T: Float>(x: T) -> T {
    let eps2 = (T::one() + T::one()) * T::epsilon();

    if x > eps2 {
        T::one() / x
    } else {
        T::zero()
    }
}

/// Struct for storing IRLBA parameters
pub struct Irlba {
    /// IRLBA convergence tolerance
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl Irlba {
    /// New IRLBA with default settings
    pub fn new() -> Irlba {
        Irlba {
            tol: 0.0001,
            max_iter: 50,
        }
    }
}

impl Default for Irlba {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Pca<T, f64> for Irlba
where
    T: DataMat + for<'a> Dot<ArrayView1<'a, f64>, Output = Array1<f64>> + Dot<Array1<f64>, Output = Array1<f64>>,
    for<'a> ArrayView1<'a, f64>: Dot<T, Output = Array1<f64>>,
{
    fn run_pca_cancellable(&self, array: &T, k: usize, snoop: impl CancelProgress) -> Result<PcaResult, anyhow::Error> {
        irlba(array, k, self.tol, self.max_iter, snoop)
    }
}

#[allow(non_snake_case)]
/// Implementation of the IRLBA algorithm. Perform the SVD of matrix `A`, retaining `nu` singlular dimenstion
/// Try to acheive tolerance `tol`, stop after at most `maxit` iterations.
pub fn irlba<T>(
    A: &T,
    nu: usize,
    tol: f64,
    maxit: usize,
    mut snoop: impl CancelProgress,
) -> Result<PcaResult, anyhow::Error>
where
    T: DataMat + for<'a> Dot<ArrayView1<'a, f64>, Output = Array1<f64>> + Dot<Array1<f64>, Output = Array1<f64>>,
    for<'a> ArrayView1<'a, f64>: Dot<T, Output = Array1<f64>>,
{
    let m = A.shape()[0];
    let n = A.shape()[1];

    if m < 2 || n < 2 {
        panic!("The input matrix must be at least 2x2.");
    }

    if nu > std::cmp::min(m, n) {
        panic!("invalid k");
    }

    let m_b = min(nu + 20, min(3 * nu, n));
    let mut mprod = 0;
    let mut it = 0;
    let mut j = 0;
    let mut k = nu;
    let mut smax = std::f64::MIN;

    let mut V: Array2<f64> = Array2::zeros((n, m_b));
    let mut W: Array2<f64> = Array2::zeros((m, m_b));
    let mut F: Array1<f64> = Array1::zeros(n);
    let mut B: Array2<f64> = Array2::zeros((m_b, m_b));
    let mut u: Array2<f64> = Array2::zeros((1, 1));
    let mut sigma: Array1<f64> = Array1::zeros(nu);
    let mut vt: Array2<f64> = Array2::zeros((1, 1));

    // random initial vector
    {
        let mut Vs = V.slice_mut(s![.., 0]);

        use rand::SeedableRng;
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(0);
        let rnorm = Normal::new(0.0f64, 1.0f64).unwrap();

        let mut rand = Array1::random_using(n, rnorm, &mut rng);
        rand *= 1.0f64 / norm(&rand.view());
        Vs.assign(&rand);
    }

    while it < maxit {
        if it > 0 {
            j = k;
        }

        W.column_mut(j).assign(&A.dot(&V.column(j)));
        mprod += 1;

        if it > 0 {
            let nc = orthog(&W.column(j), &W.slice(s![.., 0..j]));
            W.column_mut(k).assign(&nc);
        }

        let mut s = norm(&W.column(j));
        let mut sinv = invcheck(s);
        W.column_mut(j).mapv_inplace(|x| x * sinv);

        let mut fnorm = 0.0;

        // Lanczos process
        while j < m_b {
            //F = A.ldot(&W.column(j));
            F = W.column(j).dot(A);
            mprod += 1;

            F -= &(&V.column(j) * s);
            F = orthog(&F.view(), &V.slice(s![.., 0..j + 1]));
            fnorm = norm(&F.view());
            let finv = invcheck(fnorm);
            F *= finv;

            if j == m_b - 1 {
                B[(j, j)] = s
            } else {
                V.column_mut(j + 1).assign(&F);
                B[(j, j)] = s;
                B[(j, j + 1)] = fnorm;
                W.column_mut(j + 1).assign(&A.dot(&V.column(j + 1)));
                mprod += 1;

                let mut new_w_col = A.dot(&V.column(j + 1));
                new_w_col -= &(&W.column(j) * fnorm);
                new_w_col = orthog(&new_w_col.view(), &W.slice(s![.., 0..j + 1]));
                s = norm(&new_w_col.view());
                sinv = invcheck(s);

                W.column_mut(j + 1).assign(&(&new_w_col * sinv));
            }

            j += 1;
        }

        let svd = SVD::svd(&B, true, true).expect("svd error");
        u = svd.0.unwrap();
        sigma = svd.1;
        vt = svd.2.unwrap();

        let resid = fnorm * &u.slice(s![m_b - 1, ..]);
        smax = if sigma[0] > smax { sigma[0] } else { smax };

        let mut num_converged = 0;
        for i in 0..nu {
            if resid[i] < tol * smax {
                num_converged += 1;
            }
        }

        if num_converged < nu {
            k = max(num_converged + nu, k);
            k = min(k, m_b - 3);
        } else {
            break;
        }

        // Update Ritz vectors
        let v_update = V.slice(s![.., 0..m_b]).dot(&vt.t().slice(s![.., 0..k]));
        V.slice_mut(s![.., 0..k]).assign(&v_update);
        V.column_mut(k).assign(&F);

        B = Array2::zeros((m_b, m_b));
        for l in 0..k {
            B[(l, l)] = sigma[l];
        }

        B.slice_mut(s![0..k, k]).assign(&resid.slice(s![0..k]));

        // right update
        let upd = W.slice(s![.., 0..m_b]).dot(&u.slice(s![.., 0..k]));
        W.slice_mut(s![.., 0..k]).assign(&upd);

        it += 1;
        snoop.set_progress_check(it as f64 / maxit as f64)?;
    }

    let U = W.slice(s![.., 0..m_b]).dot(&u.slice(s![.., 0..nu]));
    let V = V.slice(s![.., 0..m_b]).dot(&vt.t().slice(s![.., 0..nu]));

    println!("number of matrix products: {mprod}");
    let sigma_out = sigma.slice(s![0..nu]).to_owned();
    Ok((U, sigma_out, V))
}
