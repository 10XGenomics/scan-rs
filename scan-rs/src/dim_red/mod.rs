#![allow(non_snake_case)]

//! Notes on generic matrix algos
//! For dimensionaltiy reduction algorithms, we want to write the method to be generic
//! over the sparse matrix type. This lets us write a DR method that works on dense or
//! sparse matrices, or on sparse matrices with special compression schemes, or special structure
//! (e.g. sparse + rank-1) matrices, without complicating the PCA code.
//! So we need to declare a set of constraints on the 'raw data' matrix that let randomized PCA
//! or IRLBA perform the needed operations on the raw matrix. For PCA and IRLBA we need to be able
//! to multiply the data by a dense matrix on the right or left. That is, for data matrix `A` and
//! query matrix `b` we need to be able to compute `A * b` or `b * A`. (Equivalently, if we could
//! transpose A we could get the same thing - this path proved harded to get working).
//! The trait orphan rules and the definition of `std::ops::Mul` mean that it's hard to implement
//! the `b * A` case, so we introduce the `LDot` trait which is implemented for
//! `A`, so that `A.ldot(b)` implements the mathematical operation `b * A`.
//! For some background see: <https://www.jstuber.net/2019/04/17/scalar-multiplication-in-rust/>
//
//! Other things I tried that didn't work.
//! - add a transpose method to `trait DataMat`: this doesn't work because you need to track of
//!   the lifetime of the transposed view, and you can't handle that generically without
//!   'Generic Associated Types'
//! - make `trait DataMat: Dot<Array2<f64>, Output=Array2<f64>> + ...` and list out all the other
//!   required impl, combining them into the DataMat trait. This doesn't work because all the
//!   super-traits end up getting 'collapsed' into a single DataMat trait, and so only 1 `dot`
//!   method survives, so you don't get the overloading behavior that lets you multiply by
//!   `Array2<f64>` or `ArrayView2<f64>` inside you algorithm. So you need to declare all the
//!   constraints directly on the method that implements your algo.

use anyhow::Error;
use ndarray::{Array1, Array2, ArrayView2};
use snoop::{CancelProgress, NoOpSnoop};
use sprs::{CsMatBase, SpIndex};
use std::ops::Deref;

/// IRLBA svd method
pub mod irlba;

/// Randomized SVD method
pub mod rand_svd;

/// Block Krylov SVD method
pub mod bk_svd;

#[cfg(test)]
pub(crate) mod test;

type PcaResult = (Array2<f64>, Array1<f64>, Array2<f64>);

/// Trait for getting the dimensions of a matrix
pub trait DataMat {
    /// Get the shape of the matrxix
    fn shape(&self) -> [usize; 2];
}

impl DataMat for ArrayView2<'_, f64> {
    fn shape(&self) -> [usize; 2] {
        [self.shape()[0], self.shape()[1]]
    }
}

impl DataMat for Array2<f64> {
    fn shape(&self) -> [usize; 2] {
        [self.shape()[0], self.shape()[1]]
    }
}

impl<N, I, IptrStorage, IndStorage, DataStorage> DataMat for CsMatBase<N, I, IptrStorage, IndStorage, DataStorage>
where
    I: SpIndex,
    IptrStorage: Deref<Target = [I]>,
    IndStorage: Deref<Target = [I]>,
    DataStorage: Deref<Target = [N]>,
{
    fn shape(&self) -> [usize; 2] {
        [self.rows(), self.cols()]
    }
}

impl<N, D, M> DataMat for sqz::AdaptiveMat<N, D, M>
where
    N: sqz::AdaptiveMatNum,
    M: sqz::MatrixMap<u32, N>,
    D: Deref<Target = [sqz::AdaptiveVec]>,
{
    fn shape(&self) -> [usize; 2] {
        [self.rows(), self.cols()]
    }
}

impl<D, M> DataMat for sqz::low_rank_offset::LowRankOffset<D, M>
where
    M: sqz::MatrixMap<u32, f64>,
    D: Deref<Target = [sqz::AdaptiveVec]>,
{
    fn shape(&self) -> [usize; 2] {
        [self.inner_sparse().rows(), self.inner_sparse().cols()]
    }
}

/// Perform a SVD of a `matrix`, retaining `k` principal components.
/// This trait always performs the pure SVD of the matrix. Special cases of SVD
/// such as PCA can be achieved by the appropriate shifts and scaling of `matrix`
pub trait Pca<T, N> {
    /// Compute a rank `k` PCA for `matrix`, with cancellation/progress token `c`
    fn run_pca_cancellable(&self, matrix: &T, k: usize, c: impl CancelProgress) -> Result<PcaResult, Error>;

    /// Compute a rank `k` PCA for `matrix`, without cancellation/progress tracking
    fn run_pca(&self, array: &T, k: usize) -> Result<PcaResult, Error> {
        self.run_pca_cancellable(array, k, NoOpSnoop)
    }
}

/// Mean-squared value of cells in `a`
pub fn frobenius(a: &ArrayView2<f64>) -> f64 {
    let mut acc = 0.0;
    for v in a {
        acc += v * v;
    }

    let sz = (a.shape()[0] * a.shape()[1]) as f64;
    acc.sqrt() / sz
}
