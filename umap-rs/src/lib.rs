#![deny(warnings)]

#[allow(unused_extern_crates)]
extern crate blas_src;

pub mod curve_fit;
pub mod dist;
pub mod embedding;
pub mod func_1d;
pub mod fuzzy;
pub mod knn;
pub mod optimize;
pub mod optimize_original;

pub mod rand_test;
pub mod test_data;

pub mod tester;
pub mod umap;
pub mod utils;
