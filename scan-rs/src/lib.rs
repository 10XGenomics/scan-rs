//! # scan-rs: Single Cell Analysis in Rust

#![deny(missing_docs)]
#![deny(warnings)]

#[allow(unused_extern_crates)]
extern crate blas_src;

/// Dimensionality reduction methods
pub mod dim_red;

/// Hierarchical clustering methods
pub mod linkage;

/// Merge clustering using differential expression
pub mod merge_clusters;

/// MTX loading routine
pub mod mtx;

/// Count matrix normalization methods
pub mod normalization;

//pub mod sseq;

/// Nearest-neighbor graphs
pub mod nn;

/// differential expression
extern crate diff_exp;

pub mod stats;
