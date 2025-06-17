use crate::{AdaptiveMat, AdaptiveVec};
use ndarray::Array2;
use num_traits::{One, Zero};
use rand::distr::uniform::SampleUniform;
use rand::prelude::Rng;

/// Generate a random dense matrix with values in the range `[0, 100)`.
pub fn random_dense_mat(rng: &mut impl Rng, rows: usize, cols: usize) -> Array2<u32> {
    Array2::from_shape_fn((rows, cols), |_| rng.random_range(0..100))
}

/// Generate a vector of random numbers of length `size`, with values in the range `[0, bound)`
pub fn gen_vec_bounded<R: Rng + ?Sized, T: Zero + One + SampleUniform + Copy + PartialOrd>(
    rng: &mut R,
    size: usize,
    bound: T,
) -> Vec<T> {
    std::iter::repeat(())
        .map(|_| rng.random_range(T::zero() + T::one()..bound))
        .take(size)
        .collect()
}

/// Generate a random sparse, adaptively compressed matrix of length `vec_len`, with values
/// in the range `[0, range)`
pub fn random_adaptive_vec(rng: &mut impl Rng, vec_len: usize, range: u32) -> AdaptiveVec {
    let nnz: usize = if vec_len == 0 { 0 } else { rng.random_range(0..vec_len) };

    let mut indexes: Vec<u32> = gen_vec_bounded(rng, nnz, vec_len as u32);
    indexes.sort_unstable();
    indexes.dedup();

    let values = gen_vec_bounded(rng, indexes.len(), range);
    AdaptiveVec::new(vec_len, &values, &indexes)
}

/// Generate a random sparse, adaptively compressed matrix of size `(rows, cols)`, with values in the range `[0, range)`
pub fn random_adaptive_mat(
    rng: &mut impl Rng,
    rows: usize,
    cols: usize,
    range: u32,
    order: Option<sprs::CompressedStorage>,
) -> AdaptiveMat {
    let storage = order.unwrap_or(if rng.random_bool(0.5) { sprs::CSR } else { sprs::CSC });

    let mut data = Vec::with_capacity(if storage == sprs::CSR { rows } else { cols });

    if storage == sprs::CSR {
        for _ in 0..rows {
            let v = random_adaptive_vec(rng, cols, range);
            data.push(v);
        }
    } else {
        for _ in 0..cols {
            let v = random_adaptive_vec(rng, rows, range);
            data.push(v);
        }
    }
    AdaptiveMat::new(rows, cols, storage, data)
}
