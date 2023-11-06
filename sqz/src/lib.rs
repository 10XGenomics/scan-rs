//! # Sqz: Memory-efficient data structures for sparse count matrices.
//!
//! Typical sparse matrix memory formats require a 4-byte index and a 4-byte value for
//! each non-zero values.
//! For a typical 10x Single-Cell RNA-seq count matrix (~13% non-zero fraction)
//! a normal sparse 4-byte index + 4-byte count layout consumes ~32kb/cell.
//! With Sqz, the gene-major (CSR) layout takes ~4kb/cell and the cell-major (CSC)
//! layout takes ~8kb/cell.
//!
//! Sqz implements a variety of ideas from the literature:
//! - Adaptive selection of sparse or dense layouts,
//! - Block compression of sparse index vectors.
//! - Use small bit-width types to hold values, with fallback to wider types for outliers

#![deny(missing_docs)]
#![deny(warnings)]

#[allow(unused_extern_crates)]
#[cfg(feature = "blas")]
extern crate blas_src;

#[macro_use]
/// Compressed sparse vectors
pub mod vec;

/// Compressed sparse matrices, comprised of sparse vectors
pub mod mat;

/// Common transformations implemented as type mat::MatrixMap
pub mod matrix_map;

/// Matrices that can be represented as the sum of a sparse-matrix
/// and a low rank matrix.
pub mod low_rank_offset;

/// Methods for multiplying sparse matrices defined in this crate by dense arrays.
pub mod prod;

/// Methods for generating random sparse vectors and matrices. Useful for testing and benchmarking
pub mod gen_rand;

pub use low_rank_offset::LowRankOffset;
pub use mat::{AdaptiveMat, AdaptiveMatNum, AdaptiveMatOwned, AdaptiveMatView};
pub use matrix_map::{MatrixMap, ScaleAxis, TransposeMap};
pub use vec::{AbstractVec, AdaptiveVec};
