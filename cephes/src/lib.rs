#![deny(warnings)]

mod beta;
mod betainc;
mod betaincinv;
mod consts;
mod gamma;
mod ndtri;
mod polevl;

#[cfg(test)]
mod tests;

pub use beta::{beta, betaln};
pub use betainc::betainc;
pub use betaincinv::betaincinv;
pub use gamma::{gamma, gammaln};
pub use ndtri::ndtri;
