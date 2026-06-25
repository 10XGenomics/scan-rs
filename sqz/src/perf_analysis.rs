//! Performance analysis of sqz
use crate::AdaptiveMat;
use anyhow::{bail, format_err, Context, Error};
use flate2::bufread::MultiGzDecoder;
use sprs::{CsMatI, TriMat};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

#[ignore]
#[test]
/// Spit out a CSV showing how each column of a sparse matrix is compressed,
/// and the size consumed. Let us figure out if we're actually achieving
/// the compression we should, and where we could improve.
fn compression_stats() {
    // get an example matrix from /mnt/showroom & put it in this location
    let m = load_mtx("../sample_filtered_feature_bc_matrix/matrix.mtx.gz").unwrap();

    let mut f = std::io::BufWriter::new(File::create("compression-stats.csv").unwrap());

    writeln!(&mut f, "i,nnz,sz,storage_type").unwrap();

    for (idx, v) in m.data.iter().enumerate() {
        let nnz = v.nnz();
        let sz = v.mem_size();
        let storage_type = v.storage_type();

        writeln!(&mut f, "{idx},{nnz},{sz},{storage_type}").unwrap();
    }
}

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
        if let Some(mat) = mat.as_mut() {
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
            mat.add_triplet(row, col, val);
        } else {
            let nrow = data.next().ok_or_else(|| format_err!("no NROW"))?.parse::<usize>()?;
            let ncol = data.next().ok_or_else(|| format_err!("no NCOL"))?.parse::<usize>()?;
            let nnz = data.next().ok_or_else(|| format_err!("no NNZ"))?.parse::<usize>()?;
            mat = Some(TriMat::with_capacity((nrow, ncol), nnz));
        }
        line.clear();
    }

    let Some(matrix) = mat else { bail!("no matrix found") };
    let csc_matrix: CsMatI<u32, usize, u32> = matrix.to_csr();
    Ok(AdaptiveMat::from_csmat(&csc_matrix))
}
