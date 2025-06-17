// Various methods in here a unused but can be used for
// manual exploration
#![allow(dead_code)]

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
use serde::{Deserialize, Serialize};
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
        //simple_rand_ex(1000, 100),
        //simple_rand_ex(100, 1000),
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

// Guassian random matrix -- this matrix has no structure, so
// the randomized SVD requires very high number of columns to be accurate.
// Therefore we leave if off the test battery.
fn simple_rand_ex(m: usize, n: usize) -> Array2<f64> {
    let r = Normal::new(0.0f64, 1.0f64).unwrap();
    Array2::from_shape_simple_fn((m, n), || r.sample(&mut seeded_rng()))
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

pub(crate) fn cr_test_zeroed_matrix(
    nfeatures: usize,
    nbarcodes: usize,
    proportion_empty: f64,
    zeroed_features: usize,
) -> (Array2<u32>, Array2<u32>, Vec<usize>) {
    let mut rng = seeded_rng();
    let r = Uniform::new(0.0, 1.0).unwrap();
    let mat = Array2::from_shape_simple_fn((nfeatures, nbarcodes), || r.sample(&mut rng));

    let mat = mat.map(|x| {
        if *x > proportion_empty {
            (*x - proportion_empty) * 10000.0
        } else {
            0.0
        }
    });

    let orig_mat = mat.map(|x| (*x * 1000.0).round() as u32);

    // sample without replacement
    let mut zeroed_features_mat = orig_mat.clone();
    let zeroed_feature_rows = rand::seq::index::sample(&mut rng, nfeatures, zeroed_features).into_vec();

    for r in zeroed_feature_rows.iter() {
        let mut row = zeroed_features_mat.row_mut(*r);
        row.fill(0)
    }

    (orig_mat, zeroed_features_mat, zeroed_feature_rows)
}

/*
        # Now for a matrix with far more structure as the irlba algorithm isn't stable enough to
        # handle complete randomness, we'll make 5 populations each with a similar signature of features
        mat = np.zeros((self.nfeatures, self.nbarcodes))
        rnd_order = np.random.choice(100, 100, replace=False)
        pop_sizes = [30, 20] + ([10] * 5)
        i = 0
        sigma = 20
        for pop in pop_sizes:
            nfeats = self.nfeatures / 10
            feats = np.random.choice(self.nfeatures, nfeats, replace=False)
            means = 20000 * np.random.randn(nfeats)
            bcs = rnd_order[i:(i + pop)]
            i += pop
            for col in bcs:
                cnts = means + sigma * np.random.randn(nfeats)
                for m, row in zip(cnts, feats):
                    mat[row, col] = max(m, 0)
        self.structured_matrix = convert_to_countmat(mat)
        self.test_dir = tempfile.mkdtemp()
*/

fn cr_test_structured_mat(nfeatures: usize, nbarcodes: usize, _proportion_empty: f64) -> Array2<f64> {
    let mut mat = Array2::zeros((nfeatures, nbarcodes));

    let mut rng = seeded_rng();
    let rnd_order = rand::seq::index::sample(&mut rng, 100, 100).into_vec();

    let pop_sizes = vec![30, 20, 10, 10, 10, 10, 10];

    let mut i = 0;
    let sigma = 20.0;

    for pop in pop_sizes {
        let pop_nfeats = nfeatures / 10;
        let feats = rand::seq::index::sample(&mut rng, pop_nfeats, nfeatures).into_vec();

        let dist = Normal::new(0.0f64, 1.0f64).unwrap();
        let means = 20000.0 * Array1::from_shape_simple_fn(pop_nfeats, || dist.sample(&mut rng));

        let bcs = &rnd_order[i..i + pop];
        i += pop;

        for col in bcs.iter() {
            let noise = sigma * Array1::from_shape_simple_fn(pop_nfeats, || dist.sample(&mut rng));
            let cnts = &means + &noise;
            for (m, row) in cnts.iter().zip(feats.iter()) {
                mat[(*row, *col)] = if *m < 0.0 { 0.0 } else { *m };
            }
        }
    }

    mat
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

// simulate a gene expression matrix with `nc` clusters
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
                mat.add_triplet(i, j, v)
            }
        }
    }

    mat
}

// #[test]
// fn test_sparse() {
//     let sprs = gene_exp_sim_sprs_ex(1000, 200, 6).to_csc();
//     let dense = sprs.to_dense();

//     let rsvd = RandSvd {
//         l_multiplier: 11.0,
//         n_iter: 11,
//     };

//     TestSvd::test_svd(&sprs, 10, &rsvd);
//     TestSvd::test_svd(&dense, 10, &rsvd);

//     let bksvd = BkSvd {
//         k_multiplier: 2.0,
//         n_iter: 10,
//     };

//     TestSvd::test_svd(&sprs, 10, &bksvd);
//     TestSvd::test_svd(&dense, 10, &bksvd);
// }

#[derive(Serialize, Deserialize)]
struct SvdTestPt {
    dataset: String,
    method: String,
    tol: f64,
    k_multiplier: f64,
    n_iter: usize,
    time: f64,
    frob_err: f64,
    s_err: f64,
    proj_err: f64,
}

type SvdRes = (Array2<f64>, Array1<f64>, Array2<f64>);

/// various datasets for testing SVD
fn datasets() -> Vec<(&'static str, Array2<f64>, SvdRes)> {
    let datasets = vec![
        ("gene_exp_sim_real_ex", gene_exp_sim_real_ex(2000, 10000, 50)),
        ("complex_ex1", complex_ex(2000, 10000, 50)),
        ("complex_ex2", complex_ex(1000, 5000, 50)),
        ("simple_rand_ex", simple_rand_ex(1000, 150)),
        ("simple_deterministic_ex", simple_deterministic_ex(1000, 1000)),
        ("pbmc4k_tiny", pbmc4k_tiny()),
    ];

    let mut res = vec![];

    for (name, a) in datasets {
        let true_svd = a.svd(true, true).unwrap();
        let true_svd = (true_svd.0.unwrap(), true_svd.1, true_svd.2.unwrap());

        res.push((name, a, true_svd))
    }

    res
}

/// Run this test to get a CSV that compares runtime and accuracy of
/// different methods, as a function of their parameters.
#[allow(dead_code)]
//#[test]
fn method_comparison_tool() {
    println!("prepping datasets");
    let datasets = datasets();

    let mut res = Vec::new();

    println!("testing svd_bk");
    svd_bk_acc_curves(&datasets, &mut res);

    println!("testing rand_svd");
    rand_svd_acc_curves(&datasets, &mut res);

    println!("testing irlba");
    irlba_acc_curves(&datasets, &mut res);

    let mut r = csv::WriterBuilder::new().from_path("svd_results.csv").unwrap();

    for pt in res {
        r.serialize(&pt).unwrap();
    }
}

/// Sweep the number of power iterations and the excess random columns in the randomized SVD implementation
/// to understand how to select the best parameters.
#[allow(dead_code)]
fn rand_svd_acc_curves(datasets: &[(&'static str, Array2<f64>, SvdRes)], res: &mut Vec<SvdTestPt>) {
    for (name, a, true_svd) in datasets {
        for &l_multiplier in &[1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.4, 7.5] {
            for n_iter in 2..7 {
                let rsvd = RandSvd { l_multiplier, n_iter };
                let (frob_err, s_err, proj_err, time) = TestSvd::cmp_svd(a, 20, &rsvd, true_svd);

                println!(
                    "lmult: {l_multiplier}, n_iter: {n_iter}, frob_err: {frob_err}, s_err: {s_err}, proj_err: {proj_err}"
                );

                let pt = SvdTestPt {
                    dataset: name.to_string(),
                    method: "rand_svd".to_string(),
                    tol: 0.0,
                    k_multiplier: l_multiplier,
                    n_iter,
                    time,
                    frob_err,
                    s_err,
                    proj_err,
                };

                res.push(pt);
            }
        }
    }
}

/// Sweep the number of power iterations and the excess random columns in the randomized SVD implementation
/// to understand how to select the best parameters.
// #[test]
#[allow(dead_code)]
fn svd_bk_acc_curves(datasets: &[(&'static str, Array2<f64>, SvdRes)], res: &mut Vec<SvdTestPt>) {
    for (name, a, true_svd) in datasets {
        for &k_multiplier in &[1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0] {
            for n_iter in 2..7 {
                let rsvd = BkSvd { k_multiplier, n_iter };
                let (frob_err, s_err, proj_err, time) = TestSvd::cmp_svd(a, 20, &rsvd, true_svd);

                println!(
                    "lmult: {k_multiplier}, n_iter: {n_iter}, frob_err: {frob_err}, s_err: {s_err}, proj_err: {proj_err}"
                );

                let pt = SvdTestPt {
                    dataset: name.to_string(),
                    method: "svd_bk".to_string(),
                    tol: 0.0,
                    k_multiplier,
                    n_iter,
                    time,
                    frob_err,
                    s_err,
                    proj_err,
                };

                res.push(pt);
            }
        }
    }
}

/// Sweep the number of power iterations and the excess random columns in the randomized SVD implementation
/// to understand how to select the best parameters.
#[allow(dead_code)]
fn irlba_acc_curves(datasets: &[(&'static str, Array2<f64>, SvdRes)], res: &mut Vec<SvdTestPt>) {
    for (name, a, true_svd) in datasets {
        for tol in &[0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003] {
            let irlba = Irlba {
                tol: *tol,
                max_iter: 400,
            };
            let (frob_err, s_err, proj_err, time) = TestSvd::cmp_svd(a, 20, &irlba, true_svd);
            println!(
                "tol: {}, max_iter: {}, frob_err: {}, s_err: {}, proj_err: {}",
                tol, 200, frob_err, s_err, proj_err,
            );

            let pt = SvdTestPt {
                dataset: name.to_string(),
                method: "irlba".to_string(),
                tol: *tol,
                k_multiplier: 0.0,
                n_iter: 0,
                time,
                frob_err,
                s_err,
                proj_err,
            };

            res.push(pt);
        }
    }
}
