/// differential expression algo
#[allow(clippy::module_inception)]
pub mod diff_exp;
/// distribution extension
pub mod dist;
/// gamma(ln) function
pub mod gamma;
/// statistics extension
pub mod stat;
/// utils for testing and integration with xena
pub mod utils;

pub use crate::diff_exp::{
    compute_sseq_params, sseq_differential_expression, sseq_differential_expression_with_cancellation,
};
