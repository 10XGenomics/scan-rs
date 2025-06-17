#![deny(warnings)]

use ndarray::{s, Array2, ArrayView2};
use ndarray_linalg::svd::SVD;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use scan_rs::dim_red::irlba::irlba;
use snoop::NoOpSnoop;
use std::f64;

fn test_irlba(a: Array2<f64>, nu: usize) {
    println!("||A||_F = {}", frobenius(&a.view()));

    let (u, s, v) = irlba(&a, nu, 0.00001, 100, NoOpSnoop).unwrap();
    println!("u: {:?}, s:{:?}, v:{:?}", u.shape(), s.shape(), v.shape());

    let av = a.dot(&v);
    let us = &u * &s;
    println!("av: {:?}, us: {:?}", av.shape(), us.shape());

    let diff = &av - &us;
    println!("TSVD: ||AV - US||_F = {}", frobenius(&diff.view()));

    //Compare estimated values with np.linalg.svd:
    let (_, _s_gt, _) = a.svd(true, true).unwrap();
    let s_gt = _s_gt.slice(s![0..nu]);
    let err = (&s - &s_gt).mapv(f64::abs);

    println!("Estmated/accurate singular values:");
    println!("{s}");
    println!("{s_gt}");

    println!(
        "||S_tsvd - S_svd||_inf = {}",
        err.fold(-1.0, |mm, v| if *v > mm { *v } else { mm })
    );
}

fn frobenius(a: &ArrayView2<f64>) -> f64 {
    let mut acc = 0.0;
    for v in a {
        acc += v * v;
    }

    acc.sqrt()
}

fn main() {
    let m = 10;
    let n = 10;
    let nu = 8;

    let mut rng = SmallRng::seed_from_u64(0);
    let r = Normal::new(0.0f64, 1.0f64).unwrap();
    let mut a: Array2<f64> = Array2::from_shape_simple_fn((m, n), || r.sample(&mut rng));

    let new_col = &a.column(0) + (4.0 * f64::EPSILON);
    a.column_mut(0).assign(&new_col);

    test_irlba(a, nu);
}
