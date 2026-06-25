/// differential expression algo
pub mod diff_exp;
/// distribution extension
pub mod dist;
/// gamma(ln) function
pub mod gamma;
/// statistics extension
pub mod stat;
/// utils for testing and integration
pub mod utils;

pub use crate::diff_exp::{
    compute_sseq_params, sseq_de_from_sums, sseq_de_from_sums_with_cancellation, sseq_differential_expression,
    sseq_differential_expression_backend, sseq_differential_expression_with_cancellation,
    sseq_differential_expression_with_cancellation_backend, sseq_params_from_moments,
};
pub use crate::dist::NbExactBackend;
