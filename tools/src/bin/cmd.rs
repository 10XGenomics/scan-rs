// Command line utility for running scan-rs functions

use anyhow::{Context, Error};
use clap::{value_parser, Arg, Command};
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::prelude::*;
use scan_rs::dim_red::bk_svd::BkSvd;
use scan_rs::dim_red::Pca;
use scan_rs::mtx::load_mtx;
use scan_rs::normalization::{binom_deviance_resid, binom_pearson_resid, normalize, Normalization};
use std::fs::{create_dir, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

pub fn main() -> Result<(), Error> {
    let matches = Command::new("scan-rs-cmd")
        .arg(
            Arg::new("INPUT")
                .help("mtx file to use")
                .required(true)
                .index(1)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("OUT_DIR")
                .help("Output directory")
                .short('o')
                .long("out_dir")
                .default_value(".")
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("NORMALIZATION")
                .help("Normalization method to use")
                .short('n')
                .long("norm")
                .default_value("cellranger")
                .value_parser([
                    "cellranger",
                    "cellranger8",
                    "seuratlog",
                    "binomialdeviance",
                    "binomialpearson",
                ]),
        )
        .arg(
            Arg::new("NUM_PCS")
                .help("Number of PCA dimensions to use")
                .short('d')
                .long("num_pcs")
                .default_value("10")
                .value_parser(value_parser!(usize)),
        )
        .get_matches();

    let mtx_filename: &PathBuf = matches.get_one("INPUT").unwrap();
    let out_dir: &PathBuf = matches.get_one("OUT_DIR").unwrap();
    let normalization: Normalization = matches.get_one::<String>("NORMALIZATION").unwrap().parse()?;
    let num_pcs: usize = *matches.get_one("NUM_PCS").unwrap();
    let matrix = load_mtx(mtx_filename)?;

    if !out_dir.exists() {
        create_dir(out_dir).with_context(|| out_dir.display().to_string())?;
    }

    let (u, d, v) = match normalization {
        Normalization::CellRanger | Normalization::CellRanger8 | Normalization::SeuratLog => {
            let norm_mat = normalize(matrix.view(), normalization);
            BkSvd::new().run_pca(&norm_mat, num_pcs).unwrap()
        }
        Normalization::BinomialDeviance => {
            let norm_mat = binom_deviance_resid(matrix.view());
            BkSvd::new().run_pca(&norm_mat, num_pcs).unwrap()
        }
        Normalization::BinomialPearson => {
            let norm_mat = binom_pearson_resid(matrix.view());
            BkSvd::new().run_pca(&norm_mat, num_pcs).unwrap()
        }
        _ => unimplemented!("Size factor yet to be implemented"),
    };

    array_to_csv(u, out_dir.join("svd_u.csv.gz"))?;
    let d2 = d.into_shape_with_order((num_pcs, 1))?;
    array_to_csv(d2, out_dir.join("svd_d.csv.gz"))?;
    array_to_csv(v, out_dir.join("svd_v.csv.gz"))?;

    Ok(())
}

pub fn array_to_csv(array: Array2<f64>, path: impl AsRef<Path>) -> Result<(), Error> {
    let mut writer = BufWriter::new(GzEncoder::new(File::create(path)?, Compression::default()));
    let num_cols = array.shape()[1];
    for row in array.axis_iter(Axis(0)) {
        for (i, entry) in row.iter().enumerate() {
            write!(writer, "{}", *entry)?;
            if i + 1 < num_cols {
                write!(writer, ",")?;
            }
        }
        writeln!(writer)?;
    }
    Ok(())
}
