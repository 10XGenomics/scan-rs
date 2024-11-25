use anyhow::{bail, format_err, Context, Error};
use flate2::bufread::MultiGzDecoder;
use sprs::{CsMatI, TriMat};
use sqz::AdaptiveMat;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Load an AdaptiveMat from gzipped MTX format
pub fn load_mtx(path: impl AsRef<Path>) -> Result<AdaptiveMat, Error> {
    let path = path.as_ref();
    let file = BufReader::new(File::open(path).with_context(|| path.display().to_string())?);
    let mut gz = BufReader::new(MultiGzDecoder::new(file));
    let mut line = String::new();
    let mut mat: Option<TriMat<u32>> = None;

    while let Ok(sz) = gz.read_line(&mut line) {
        if sz == 0 {
            break;
        }
        if line.starts_with('%') {
            line.clear();
            continue;
        }
        let mut data = line.split_whitespace();
        if mat.is_none() {
            let nrow = data.next().ok_or_else(|| format_err!("no NROW"))?.parse::<usize>()?;
            let ncol = data.next().ok_or_else(|| format_err!("no NCOL"))?.parse::<usize>()?;
            let nnz = data.next().ok_or_else(|| format_err!("no NNZ"))?.parse::<usize>()?;
            mat = Some(TriMat::with_capacity((nrow, ncol), nnz));
        } else {
            let row = data
                .next()
                .ok_or_else(|| format_err!("missing ROW"))?
                .parse::<usize>()?
                - 1;
            let col = data
                .next()
                .ok_or_else(|| format_err!("missing COL"))?
                .parse::<usize>()?
                - 1;
            let val = data.next().ok_or_else(|| format_err!("missing VAL"))?.parse::<u32>()?;
            mat.as_mut().unwrap().add_triplet(row, col, val);
        }
        line.clear();
    }

    let Some(matrix) = mat else { bail!("no matrix found") };
    let csc_matrix: CsMatI<u32, usize, u32> = matrix.to_csr();
    Ok(AdaptiveMat::from_csmat(&csc_matrix))
}
