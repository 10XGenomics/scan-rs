//! Differential expression NB2 GLM
//!
use std::sync::Arc;

use crate::dist;
use crate::stat::Statistics;
use anyhow::Error;
use ndarray::{arr1, Array1, Axis, Zip};
use noisy_float::types::{n64, N64};
use snoop::{CancelProgress, NoOpSnoop};
use sqz::{AdaptiveMat, MatrixMap, TransposeMap};

/// Default threshold for big count (p_value computation)
pub static BIG_COUNT_DEFAULT: u64 = 900;
/// Default Zeta quintile value
pub static ZETA_QUINTILE_DEFAULT: f64 = 0.995f64;

///sseq_params struct holding global parameters for the sSeq differential expression method
#[derive(Debug, PartialEq)]
#[cfg_attr(any(feature = "serde", test), derive(serde::Deserialize))]
pub struct SSeqParams {
    /// num_cells
    pub num_cells: u32,
    /// number of genes
    pub num_genes: u32,
    /// size_factors
    pub size_factors: Vec<f64>,
    /// gene_means
    pub gene_means: Vec<f64>,
    /// gene variances
    pub gene_variances: Vec<f64>,
    /// use genes
    pub use_genes: Vec<bool>,
    /// gene moment dispersion
    pub gene_moment_phi: Vec<f64>,
    /// zeta hat
    pub zeta_hat: f64,
    /// delta
    pub delta: f64,
    /// gene dispersion
    pub gene_phi: Vec<f64>,
}

/// Result of Differential expression
#[derive(Debug)]
pub struct DiffExpResult {
    /// genes tested
    pub genes_tested: Array1<bool>,
    /// sums experiments in (cond_a)
    pub sums_in: Array1<u64>,
    /// sums experiments out (cond_b)
    pub sums_out: Array1<u64>,
    /// common means
    pub common_mean: Array1<f64>,
    /// common dispersion
    pub common_dispersion: Array1<f64>,
    /// normalized experiments in (cond_a)
    pub normalized_mean_in: Array1<f64>,
    /// normalized experiments out (cond_a)
    pub normalized_mean_out: Array1<f64>,
    /// p_values
    pub p_values: Array1<f64>,
    /// adjusted p_values (bh)
    pub adjusted_p_values: Array1<f64>,
    /// Introduce a pseudo-count into log2(fold_change)
    pub log2_fold_change: Array1<f64>,
}

pub fn sseq_differential_expression(
    mat: &AdaptiveMat,
    cond_a: &[usize],
    cond_b: &[usize],
    sseq_params: &SSeqParams,
    big_count: Option<u64>,
) -> DiffExpResult {
    let snoop = NoOpSnoop {};
    sseq_differential_expression_with_cancellation(snoop, mat, cond_a, cond_b, sseq_params, big_count).unwrap()
}

///sseq_differential_expression
/// Run sSeq pairwise differential expression test
pub fn sseq_differential_expression_with_cancellation(
    mut snoop: impl CancelProgress,
    mat: &AdaptiveMat,
    cond_a: &[usize],
    cond_b: &[usize],
    sseq_params: &SSeqParams,
    big_count: Option<u64>,
) -> Result<DiffExpResult, Error> {
    let big_count = big_count.unwrap_or(BIG_COUNT_DEFAULT);

    snoop.set_progress_check(0.0)?;

    let size_factor_a = cond_a.iter().fold(0., |acc, &idx| acc + sseq_params.size_factors[idx]);
    let size_factor_b = cond_b.iter().fold(0., |acc, &idx| acc + sseq_params.size_factors[idx]);
    snoop.set_progress_check(0.1)?;

    let s = snoop.get_subsnoop(0.5);
    let (feature_sums_a, feature_sums_b) = mat.sum_rows_dual_with_cancellation::<u64, _>(s, cond_a, cond_b)?;

    snoop.set_progress_check(0.6)?;

    // compute p_value
    let mut p_values = Array1::zeros(feature_sums_a.dim());

    Zip::from(&mut p_values)
        .and(&feature_sums_a)
        .and(&feature_sums_b)
        .and(&sseq_params.gene_means)
        .and(&sseq_params.gene_phi)
        .and(&sseq_params.use_genes)
        .par_for_each(|pv, &feature_sum_a, &feature_sum_b, &gene_mean, &gene_phi, &use_gene| {
            *pv = if use_gene && feature_sum_a > big_count && feature_sum_b > big_count {
                dist::nb_asymptotic_test(
                    feature_sum_a,
                    feature_sum_b,
                    size_factor_a,
                    size_factor_b,
                    gene_mean,
                    gene_phi,
                )
            } else {
                dist::nb_exact_test(
                    feature_sum_a,
                    feature_sum_b,
                    size_factor_a,
                    size_factor_b,
                    gene_mean,
                    gene_phi,
                )
            };
        });

    snoop.set_progress_check(0.75)?;

    // Adjust p-values for multiple testing correction
    // Only adjust the features that were actually tested
    let adj_p_values_in = p_values
        .iter()
        .enumerate()
        .filter(|(i, _)| sseq_params.use_genes[*i])
        .map(|(i, &val)| (i, val))
        .collect::<Vec<(usize, f64)>>();

    let adj_p_values = dist::adjusted_pvalue_bh(&adj_p_values_in);
    let mut p_values_bh = p_values.clone();
    adj_p_values.iter().for_each(|&v| p_values_bh[v.0] = v.1);

    snoop.set_progress_check(0.9)?;

    let log2_fold_change = feature_sums_a
        .iter()
        .zip(feature_sums_b.iter())
        .map(|(&feature_sum_a, &feature_sum_b)| {
            ((1 + feature_sum_a) as f64 / (1.0 + size_factor_a)).log2()
                - ((1 + feature_sum_b) as f64 / (1.0 + size_factor_b)).log2()
        })
        .collect::<Array1<f64>>();

    snoop.set_progress_check(0.95)?;

    let normalized_mean_in = if size_factor_a == 0.0 {
        Array1::zeros(feature_sums_a.dim())
    } else {
        feature_sums_a.map(|&feature_sum_a| feature_sum_a as f64 / size_factor_a)
    };
    let normalized_mean_out = if size_factor_b == 0.0 {
        Array1::zeros(feature_sums_b.dim())
    } else {
        feature_sums_b.map(|&feature_sum_b| feature_sum_b as f64 / size_factor_b)
    };

    snoop.set_progress_check(1.0)?;

    let res = DiffExpResult {
        genes_tested: arr1(&sseq_params.use_genes),
        sums_in: feature_sums_a,
        sums_out: feature_sums_b,
        common_mean: arr1(&sseq_params.gene_means),
        common_dispersion: arr1(&sseq_params.gene_phi),
        normalized_mean_in,
        normalized_mean_out,
        p_values,
        adjusted_p_values: p_values_bh,
        log2_fold_change,
    };

    Ok(res)
}

/// size_factor
fn size_factors(mat: &AdaptiveMat, cell_indices: Option<&[usize]>, umi_counts: Option<&[f64]>) -> Array1<f64> {
    let counts_per_cell = match umi_counts {
        None => match cell_indices {
            Some(cells) => mat.sum_cols::<f64>(cells),
            None => mat.sum_axis::<f64>(Axis(0)),
        },
        Some(u) => arr1(u),
    };
    let median = counts_per_cell.iter().cloned().collect::<Vec<_>>().median();

    match cell_indices {
        Some(cells) => {
            let mut arr = Array1::zeros((mat.cols(),));
            for (i, &cell) in cells.iter().enumerate() {
                arr[cell] = counts_per_cell[i] / median;
            }
            arr
        }
        None => counts_per_cell.mapv(|v| v / median),
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct SizeNormalized {
    size_factors: Arc<Array1<N64>>,
}

impl SizeNormalized {
    fn new(size_factors: &Array1<f64>) -> Self {
        let size_factors = Arc::new(size_factors.mapv(|v| if v.is_nan() { n64(0.0) } else { n64(v) }));
        SizeNormalized { size_factors }
    }
}

impl MatrixMap<u32, f64> for SizeNormalized {
    type T = TransposeMap<u32, f64, SizeNormalized>;

    #[inline(always)]
    fn map(&self, v: u32, _: usize, c: usize) -> f64 {
        v as f64 / self.size_factors[c].raw()
    }

    fn t(&self) -> Self::T {
        TransposeMap::new(self.clone())
    }
}

/// compute_sseq_params
pub fn compute_sseq_params(
    mat: &AdaptiveMat,
    zeta_quintile: Option<f64>,
    cell_indices: Option<&[usize]>,
    umi_counts: Option<&[f64]>,
) -> SSeqParams {
    let cell_count = cell_indices.map(<[usize]>::len).unwrap_or_else(|| mat.cols()) as f64;
    let gene_count = mat.rows() as f64;
    let size_factors = size_factors(mat, cell_indices, umi_counts);
    let norm_mat = mat.view().compose_map(SizeNormalized::new(&size_factors));
    let (mean_g, var_g) = match cell_indices {
        Some(cells) => norm_mat.mean_var_rows(cells),
        None => norm_mat.mean_var_axis(Axis(1)),
    };

    // Method of moments estimate of feature-wise dispersion (phi)
    // Only use features with non-zero variance in the following estimation

    let use_g = var_g.map(|&v| v > 0f64);
    let sum_size_factors = size_factors
        .iter()
        .filter(|&&v| v != 0f64)
        .fold(0f64, |acc, &v| acc + 1.0 / v);
    let mut phi_mm_g_used = Vec::<f64>::new();
    let phi_mm_g = use_g
        .iter()
        .enumerate()
        .map(|(i, &cond)| {
            let mut res: f64 = 0f64;
            if cond {
                res = 0f64.max(
                    (cell_count * var_g[i] - mean_g[i] * sum_size_factors) / (mean_g[i] * mean_g[i] * sum_size_factors),
                );

                phi_mm_g_used.push(res);
            }
            res
        })
        .collect::<Vec<f64>>();

    let (zeta_hat, delta) = if !phi_mm_g_used.is_empty() {
        //Use a high quintile of the MoM dispersion as our shrinkage target
        //# per the rule of thumb libray_tyein Yu, et al.
        let zeta_hat = phi_mm_g_used.percentile(100.0 * zeta_quintile.unwrap_or(ZETA_QUINTILE_DEFAULT));
        //Compute delta, the optimal shrinkage towards zeta_hat
        // This defines a linear function that shrinks the MoM dispersion estimates
        let mean_phi_mm_g = phi_mm_g_used.mean();
        let delta = (phi_mm_g_used
            .iter()
            .fold(0., |acc, &x| acc + (x - mean_phi_mm_g) * (x - mean_phi_mm_g))
            / (gene_count - 1f64))
            / (phi_mm_g_used
                .iter()
                .fold(0., |acc, &x| acc + (x - zeta_hat) * (x - zeta_hat))
                / (gene_count - 2f64));
        (zeta_hat, delta)
    } else {
        // variance of all genes was 0
        (0.0, 0.0)
    };

    // Compute the shrunken dispersion estimates
    // Interpolate between the MoM estimates and zeta_hat by delta
    let mut phi_g = vec![std::f64::NAN; gene_count as usize];

    let cond = phi_mm_g_used.iter().any(|x| x > &0f64);
    for (i, &v) in var_g.iter().enumerate() {
        if cond && v > 0f64 {
            phi_g[i] = (1f64 - delta) * phi_mm_g[i] + delta * zeta_hat;
        } else {
            phi_g[i] = 0f64;
        }
    }

    SSeqParams {
        size_factors: size_factors.to_vec(),
        num_cells: cell_count as u32,
        num_genes: gene_count as u32,
        gene_means: mean_g.to_vec(),
        gene_variances: var_g.to_vec(),
        use_genes: use_g.to_vec(),
        gene_moment_phi: phi_mm_g,
        zeta_hat,
        delta,
        gene_phi: phi_g,
    }
}

///
/// test mod
#[cfg(test)]
pub mod test {
    use super::*;
    #[cfg(hdf5)]
    use crate::diff_exp::utils;
    #[cfg(hdf5)]
    use log::info;
    use ndarray::array;
    use rand::prelude::SeedableRng;
    use rand_pcg::Pcg64Mcg;
    #[cfg(hdf5)]
    use rayon::prelude::*;
    use sqz::gen_rand::random_adaptive_mat;
    #[cfg(hdf5)]
    use std::path::Path;

    /// Test size factors
    #[test]
    fn test_size_factors() {
        let cols: usize = 10;
        let rows: usize = 7;
        let range = 10u32;
        let mut rng = Pcg64Mcg::seed_from_u64(42);
        let mat = random_adaptive_mat(&mut rng, rows, cols, range, None);
        let res_factor = size_factors(&mat, None, None);
        assert_eq!(
            array![
                0.631578947368421f64,
                2.0,
                2.0,
                0.8421052631578947,
                1.263157894736842,
                1.1578947368421053,
                1.368421052631579,
                0.0,
                0.0,
                0.42105263157894735
            ],
            res_factor
        );
    }

    #[test]
    #[cfg(hdf5)]
    fn test_compare_sseq_differential_expression() {
        let use_umi = false;
        let min_feature_sum = Some(f64::MIN_POSITIVE);
        let library_type = utils::GENE_EXPRESSION_LIBRARY_TYPE;
        struct TestInput<'a> {
            input_file: String,
            input_type: String,
            clustering_key: String,
            sseq_params_json: String,
            zeta_quintile: Option<f64>,
            feature_selection: Option<&'a [u32]>,
            barcode_selection: Option<&'a [usize]>,
            ground_truth_file: String,
            output_file: Option<String>,
        }
        let tests = maplit::hashmap! {
            "h5" => TestInput {
               input_file: "./test/analysis.h5".to_owned(),
               input_type: "h5".to_owned(),
               clustering_key: "_kmeans_3_clusters".to_owned(),
               zeta_quintile: None,
               output_file: "./test/analysis_diff_exp.csv".to_owned(),
            },
        };

        tests.par_iter().for_each(|(name, input)| {
            info!("testing {} ....", name);
            utils::test::test_sseq_differential_expression(
                &input.input_type,
                &Path::new(&input.input_file),
                &Path::new(&input.input_file),
                library_type,
                &input.clustering_key,
                &Path::new(&input.sseq_params_json),
                use_umi,
                min_feature_sum,
                input.zeta_quintile,
                input.feature_selection,
                input.barcode_selection,
                &Path::new(&input.ground_truth_file),
                input.output_file.as_ref().map(|f| Path::new(f)),
            )
            .unwrap();
        });
    }

    // TODO: restore this test
    // #[test]
    // fn test_compare_sseq_diffexp_submatrix() -> Result<(), Error> {
    //     let use_umi = false;
    //     let min_feature_sum = Some(f64::MIN_POSITIVE);
    //     let library_type = utils::GENE_EXPRESSION_LIBRARY_TYPE;
    //     let matrix_path = "./test/test2.cloupe";
    //     let barcode_selection = Some(
    //         &[
    //             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 25, 26, 27, 28, 29, 30, 32, 33,
    //             34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 63,
    //             64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    //             92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    //         ][..],
    //     );
    //     let test_cluster = 0;

    //     let (mat, sseq_params, _) = utils::init_matrix(
    //         "loupe",
    //         &Path::new(&matrix_path),
    //         library_type,
    //         use_umi,
    //         min_feature_sum,
    //         None,
    //         None,
    //         barcode_selection,
    //     )?;

    //     utils::test::compare_sseq_params(
    //         &Path::new("./test/c1+2.sseq_params.json.gz"),
    //         &sseq_params,
    //         barcode_selection,
    //     );

    //     let clusters = {
    //         let mut reader = CLoupeReader::new(&matrix_path)?;
    //         let section = reader.get_immutable_section()?;
    //         let cluster = section.clusterings.iter().find(|&c| c.name == "Graph-based");
    //         read_clustering_assignments(&reader, cluster.unwrap())?
    //     };

    //     let (cond_a, cond_b) = {
    //         let mut cond_a = vec![];
    //         let mut cond_b = vec![];
    //         for (i, &cluster) in clusters.iter().enumerate() {
    //             if cluster == test_cluster {
    //                 cond_a.push(i);
    //             } else if cluster == test_cluster + 1 {
    //                 cond_b.push(i);
    //             }
    //         }
    //         (cond_a, cond_b)
    //     };

    //     let de_result = sseq_differential_expression(&mat.matrix, &cond_a, &cond_b, &sseq_params, None);
    //     let xena_map = utils::test::XenaResult::read_xena_result("./test/c1+2.diffexp.csv.gz", test_cluster);

    //     utils::test::CompareResult::diff_result(None, xena_map, &de_result, &mat);
    //     Ok(())
    // }
}
