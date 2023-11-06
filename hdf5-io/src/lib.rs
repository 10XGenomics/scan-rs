#![deny(warnings)]

/// io for analysis results (diffExp, clusterings) for the analysis h5 file from xena
pub mod analysis;
/// io for matrix group in filtered and analysis h5 file from xena
pub mod matrix;
#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
