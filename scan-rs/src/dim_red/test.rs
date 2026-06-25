use super::bk_svd::BkSvd;
use super::irlba::Irlba;
use super::rand_svd::RandSvd;
use super::*;
use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use ndarray::linalg::Dot;
use ndarray::{s, Array, Array2, ArrayView1};
use ndarray_linalg::SVD;
use ndarray_npy::NpzReader;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Gamma, Normal, Poisson, Uniform};
use sprs::io::read_matrix_market_from_bufread;
use sprs::{CsMat, TriMat};
use std::f64;
use std::fs::File;
use std::io::BufReader;

fn seeded_rng() -> SmallRng {
    SmallRng::seed_from_u64(0)
}

#[test]
fn test_mtx_to_csr() -> Result<()> {
    let mtx = "test/pbmc_1k.mtx.gz";
    let mut reader = BufReader::new(GzDecoder::new(File::open(mtx).context(mtx)?));
    let trimat: TriMat<i32> = read_matrix_market_from_bufread(&mut reader)?;
    let _: CsMat<i32> = trimat.to_csr();
    Ok(())
}

struct TestSvd<T>
where
    T: DataMat
        + for<'a> Dot<ArrayView2<'a, f64>, Output = Array2<f64>>
        + Dot<Array2<f64>, Output = Array2<f64>>
        + for<'a> Dot<ArrayView1<'a, f64>, Output = Array1<f64>>
        + Dot<Array1<f64>, Output = Array1<f64>>,
    for<'a> ArrayView2<'a, f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<Array2<f64>, Output = Array2<f64>>,
{
    phantom: std::marker::PhantomData<T>,
}

impl<T> TestSvd<T>
where
    T: DataMat
        + for<'a> Dot<ArrayView2<'a, f64>, Output = Array2<f64>>
        + Dot<Array2<f64>, Output = Array2<f64>>
        + for<'a> Dot<ArrayView1<'a, f64>, Output = Array1<f64>>
        + Dot<Array1<f64>, Output = Array1<f64>>,
    for<'a> ArrayView2<'a, f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<T, Output = Array2<f64>>,
    Array2<f64>: Dot<Array2<f64>, Output = Array2<f64>>,
{
    fn cmp_svd(
        a: &T,
        nu: usize,
        pca: &impl Pca<T, f64>,
        true_svd: &(Array2<f64>, Array1<f64>, Array2<f64>),
    ) -> (f64, f64, f64, f64) where {
        let now = std::time::Instant::now();
        let (u, s, v) = pca.run_pca(a, nu).unwrap();
        let t = now.elapsed();
        let t = t.as_nanos() as f64 / 1e9;

        let av = a.dot(&v);
        let us = &u * &s;

        let diff = &av - &us;
        let frob_err = frobenius(&diff.view());
        println!("||Av - Us||_frob = {frob_err}");

        //Compare estimated values with np.linalg.svd:
        let (_u_gt, s_gt, v_gt) = true_svd;
        let s_gt = s_gt.slice(s![0..nu]);
        let err = ((&s - &s_gt) / s_gt).mapv(f64::abs);

        let s_err = err.fold(-1.0f64, |mm, &v| v.max(mm));
        println!("Sgt: {s_gt}");
        println!("||S_tsvd - S_svd||_inf = {s_err}");

        let av = av.mapv(f64::abs);
        let av_gt = a.dot(&v_gt.slice(s![0..nu, ..]).t()).mapv(f64::abs);
        // println!("av_gt[:5] = {:?}", av_gt.slice(s![0..5, ..]));
        // println!("av[:5] = {:?}", av.slice(s![0..5, ..]));
        let proj_err = ((&av - &av_gt) / &av_gt)
            .mapv(f64::abs)
            .fold(-1.0f64, |mm, &v| v.max(mm));
        println!("||AV_tsvd - AV_svd||_inf = {proj_err}");

        (frob_err, s_err, proj_err, t)
    }

    fn test_svd(a: &T, nu: usize, pca: &impl Pca<T, f64>) {
        let n = a.shape()[1];
        let dense = a.dot(&Array2::<f64>::eye(n));

        println!("performing true SVD");
        let true_svd = dense.svd(true, true).unwrap();
        let true_svd = (true_svd.0.unwrap(), true_svd.1, true_svd.2.unwrap());

        let (frob_err, s_err, proj_err, _) = Self::cmp_svd(a, nu, pca, &true_svd);

        assert!(frob_err < 0.001);
        assert!(s_err < 0.001);
        assert!(proj_err < 0.001);
    }
}

fn fast_test_battery() -> Vec<Array2<f64>> {
    let tests = vec![
        simple_deterministic_ex(100, 1000),
        simple_deterministic_ex(1000, 100),
        complex_ex(100, 1000, 20),
        complex_ex(1000, 100, 20),
        gene_exp_sim_real_ex(100, 1000, 20),
        gene_exp_sim_real_ex(1000, 100, 20),
    ];

    tests
}

fn run_tests(battery: Vec<Array2<f64>>, nu: usize, pca: impl Pca<Array2<f64>, f64>) {
    for t in battery {
        TestSvd::test_svd(&t, nu, &pca);
    }
}

#[test]
fn irlba_fast_test() {
    let irlba = Irlba {
        tol: 0.00001,
        max_iter: 300,
    };
    run_tests(fast_test_battery(), 10, irlba);
}

#[test]
fn rsvd_fast_test() {
    let rsvd = RandSvd::new();
    run_tests(fast_test_battery(), 10, rsvd);
}

#[test]
fn rsvd_pbmc4k_test() {
    let a = pbmc4k_tiny();
    let rsvd = RandSvd::new();
    TestSvd::test_svd(&a, 10, &rsvd);
}

#[test]
fn bksvd_fast_test() {
    let svd = BkSvd::new();
    run_tests(fast_test_battery(), 10, svd);
}

#[test]
fn bksvd_pbmc4k_test() {
    let a = pbmc4k_tiny();
    let svd = BkSvd::new();
    TestSvd::test_svd(&a, 10, &svd);
}

// deterministic matrix (useful for comparing w/ python)
fn simple_deterministic_ex(m: usize, n: usize) -> Array2<f64> {
    let mut v = Vec::new();
    for x in 0..(m * n) {
        let val = x % 7 + x % 4 + x % 50 + x % 47 + x % 12;
        v.push(val as f64);
    }

    Array::from_shape_vec((m, n), v).unwrap()
}

// random matrix with `fix_cols` columns set to be random linear combinations of
// other columns.
fn complex_ex(m: usize, n: usize, fix_cols: usize) -> Array2<f64> {
    let mut rng = seeded_rng();
    let r = Normal::new(0.0f64, 1.0f64).unwrap();
    let mut a: Array2<f64> = Array2::from_shape_simple_fn((m, n), || r.sample(&mut rng));

    // Make some columns into linear combinations of other columns.
    for i in 0..fix_cols {
        let mix = Array1::from_shape_simple_fn(n, || r.sample(&mut rng));
        let new_col = a.dot(&mix);
        a.column_mut(i).assign(&new_col);
    }

    a
}

fn pbmc4k_tiny() -> Array2<f64> {
    let f = File::open("test/pbmc4k_tiny.npz").unwrap();
    let mut rdr = NpzReader::new(f).unwrap();
    rdr.by_index(0).unwrap()
}

// simulate a gene expression matrix with `nc` clusters
fn gene_exp_sim_real_ex(m: usize, n: usize, nc: usize) -> Array2<f64> {
    let mut rng = seeded_rng();

    let mut clusters = Vec::new();
    for _ in 0..nc {
        let r = Normal::new(0.0f64, 10.0f64).unwrap();
        let a = Array1::from_shape_simple_fn(n, || r.sample(&mut rng));
        clusters.push(a);
    }

    let mut a: Array2<f64> = Array2::<f64>::zeros((m, n));
    let jitter = Normal::new(0.0f64, 1.0f64).unwrap();

    let c = Uniform::new(0, clusters.len()).unwrap();
    for i in 0..m {
        let cluster_id = c.sample(&mut rng);
        let row = &clusters[cluster_id];
        let new_row = row + &Array1::from_shape_simple_fn(n, || jitter.sample(&mut rng));
        a.row_mut(i).assign(&new_row);
    }

    a
}

/// simulate a gene expression matrix with `nc` clusters
fn gene_exp_sim_sprs_ex(m: usize, n: usize, nc: usize) -> TriMat<f64> {
    let mut rng = seeded_rng();

    let r = Gamma::new(0.4, 2.0).unwrap();
    let clusters: Vec<Array1<f64>> = std::iter::repeat_with(|| Array1::from_shape_simple_fn(n, || r.sample(&mut rng)))
        .take(nc)
        .collect();

    let mut mat = TriMat::new((m, n));

    let c = Uniform::new(0, clusters.len()).unwrap();
    for i in 0..m {
        let cluster_id = c.sample(&mut rng);
        let row = &clusters[cluster_id];

        for j in 0..n {
            let dist = Poisson::new(row[j]).unwrap();
            let v = dist.sample(&mut rng);

            if v > 0.0 {
                mat.add_triplet(i, j, v);
            }
        }
    }

    mat
}

#[test]
fn test_sparse() {
    let sprs = gene_exp_sim_sprs_ex(1000, 200, 6).to_csc();
    let dense = sprs.to_dense();

    let rsvd = RandSvd {
        l_multiplier: 11.0,
        n_iter: 11,
    };

    TestSvd::test_svd(&sprs, 10, &rsvd);
    TestSvd::test_svd(&dense, 10, &rsvd);

    let bksvd = BkSvd {
        k_multiplier: 2.0,
        n_iter: 10,
    };

    TestSvd::test_svd(&sprs, 10, &bksvd);
    TestSvd::test_svd(&dense, 10, &bksvd);
}
