use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use rand::prelude::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sqz::{gen_rand, prod};

fn criterion_benchmark(c: &mut Criterion) {
    let rng = &mut Pcg64Mcg::seed_from_u64(42);

    let rows = 1000;
    let cols = 10000;
    let range = 50;
    let cols2 = 16;

    let csr = gen_rand::random_adaptive_mat(rng, rows, cols, range, Some(sprs::CSR));
    let m2 = gen_rand::random_dense_mat(rng, cols, cols2);

    c.bench_function("csr-mul 1k", move |b| {
        b.iter(|| {
            let mut test: Array2<u32> = Array2::zeros((rows, cols2));
            prod::mat_densemat_mult(&csr, &m2.view(), test.view_mut());
        })
    });

    let csc = gen_rand::random_adaptive_mat(rng, rows, cols, range, Some(sprs::CSC));
    let m2 = gen_rand::random_dense_mat(rng, cols, cols2);

    c.bench_function("csc-mul 1k", move |b| {
        b.iter(|| {
            let mut test: Array2<u32> = Array2::zeros((rows, cols2));
            prod::mat_densemat_mult(&csc, &m2.view(), test.view_mut());
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark,
}

criterion_main!(benches);
