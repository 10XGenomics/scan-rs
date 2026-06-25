//! Utility functions for differential expression.
#![cfg(feature = "hdf5")]

use crate::compute_sseq_params;
use crate::diff_exp::{DiffExpResult, SSeqParams};
use anyhow::{Context, Result};
use hdf5::{Dataset, Extents, Group};
use ndarray::Array;
use scan_types::matrix::AdaptiveFeatureBarcodeMatrix;
use std::fs::File;
use std::path::{Path, PathBuf};

/// ANTIBODY_LIBRARY_TYPE = "Antibody Capture"
pub static ANTIBODY_LIBRARY_TYPE: &str = "Antibody Capture";
/// GENE_EXPRESSION_LIBRARY_TYPE = "Gene Expression
pub static GENE_EXPRESSION_LIBRARY_TYPE: &str = "Gene Expression";
/// clustering group
pub static ANALYSIS_H5_CLUSTERING_GROUP: &str = "clustering";

macro_rules! parse_path {
    ($path:expr, $msg:literal) => {
        $path.to_str().with_context(|| format!("{}: {}", $msg, $path.display()))
    };
}

/// Retrieve count matrix and compute global params sseq_params
#[expect(clippy::too_many_arguments)]
pub fn init_matrix(
    input_type: &str,
    matrix_path: &Path,
    library_type: &str,
    use_umi: bool,
    min_row_sum: Option<u32>,
    zeta_quintile: Option<f64>,
    _row_indices: Option<&[u32]>,
    col_indices: Option<&[usize]>,
) -> Result<(AdaptiveFeatureBarcodeMatrix, SSeqParams, Vec<usize>)> {
    let matrix_file = parse_path!(matrix_path, "matrix_path")?;

    let (mat, umi, cols_idx) = match input_type {
        "h5" => {
            let (csmat, _) = hdf5_io::matrix::read_adaptive_csr_matrix(matrix_file, Some(library_type), min_row_sum)?;

            let matrix = hdf5_io::matrix::get_matrix(matrix_file)?;
            let umi = if use_umi {
                Some(
                    hdf5_io::matrix::read_umi_counts_from_matrix(&matrix)?
                        .iter()
                        .map(|&v| v as f64)
                        .collect::<Vec<f64>>(),
                )
            } else {
                None
            };

            (csmat, umi, Vec::<usize>::new())
        }
        _ => unreachable!(),
    };

    let umi = umi.unwrap_or_default();
    let umi = if umi.is_empty() { None } else { Some(umi.as_slice()) };

    let sseq_params = compute_sseq_params(&mat.matrix, zeta_quintile, col_indices, umi);

    Ok((mat, sseq_params, cols_idx))
}

/// load clustering keys from analysis_h5
pub fn get_clustering_keys(analysis_h5: &Path) -> Result<Vec<String>> {
    let analysis_file = parse_path!(analysis_h5, "Analysis_h5 is none")?;
    hdf5_io::analysis::get_clustering_keys(analysis_file)
}

/// Retrieve clustering info and generate assignment vector
#[expect(clippy::type_complexity)]
pub fn initial_cluster_assignments(
    input_type: &str,
    clustering_path: &Path,
    clustering_key: &str,
) -> Result<Vec<(i16, Vec<usize>, Vec<usize>)>> {
    let clustering_file = parse_path!(clustering_path, "clustering_path")?;

    let (clusters_count, assignments) = match input_type {
        "h5" => {
            let (clusters_count, v) = hdf5_io::analysis::get_clustering(clustering_file, clustering_key)?;
            (clusters_count, v.into_iter().map(|v| v - 1).collect::<Vec<i16>>())
        }
        _ => unreachable!(),
    };

    let conditions = (0..clusters_count)
        .map(|cluster| {
            let mut cond_a = Vec::<usize>::new();
            let mut cond_b = Vec::<usize>::new();
            for (idx, &val) in assignments.iter().enumerate() {
                if val == (cluster as i16) {
                    cond_a.push(idx);
                } else {
                    cond_b.push(idx);
                }
            }
            (cluster as i16, cond_a, cond_b)
        })
        .collect::<Vec<(i16, Vec<usize>, Vec<usize>)>>();

    Ok(conditions)
}

///  IO for differential expression Result
pub struct ResultIo<'a> {
    mat: &'a AdaptiveFeatureBarcodeMatrix,
    analysis_csv: &'a Path,
    file_h5: hdf5::File,
    group_all_diff_exp: Group,
}

impl<'a> ResultIo<'a> {
    /// create ResultIo to process Io
    pub fn new(
        mat: &'a AdaptiveFeatureBarcodeMatrix,
        analysis_h5: &'a Path,
        analysis_csv: &'a Path,
    ) -> Result<ResultIo<'a>> {
        let file_h5 = hdf5::File::create(analysis_h5)?;
        let group_all_diff_exp = file_h5.create_group("all_differential_expression")?;

        Ok(ResultIo {
            mat,
            analysis_csv,
            file_h5,
            group_all_diff_exp,
        })
    }

    /// header
    fn header(res: &[DiffExpResult]) -> Vec<String> {
        std::iter::chain(
            ["Feature ID".to_string(), "Feature Name".to_string()],
            (1..=res.len()).flat_map(|i| {
                [
                    format!("Cluster {i} Mean Counts"),
                    format!("Cluster {i} Log2 fold change"),
                    format!("Cluster {i} Adjusted p value"),
                ]
            }),
        )
        .collect()
    }

    fn get_output_file(&self, clustering_key: &str) -> Result<PathBuf> {
        let dir = self.analysis_csv.join("diffexp").join(&clustering_key[1..]);
        std::fs::create_dir_all(dir.as_path())?;
        Ok(dir.as_path().join("differential_expression.csv"))
    }

    /// dump differential expression results to csv file
    pub fn dump_csv_results(&self, res: &[DiffExpResult], clustering_key: &str) -> Result<()> {
        let out_file = self.get_output_file(clustering_key)?;
        let output = File::create(out_file)?;

        let mut wtr = csv::WriterBuilder::new().from_writer(output);

        let features_count = res[0].common_mean.len();

        wtr.write_record(Self::header(res))?;
        for feature in 0..features_count {
            let mut record = Vec::<String>::new();
            record.push(self.mat.feature_ids[feature].clone());
            record.push(self.mat.feature_names[feature].clone());
            for cluster in res {
                record.push(cluster.normalized_mean_in[feature].to_string());
                record.push(cluster.log2_fold_change[feature].to_string());
                record.push(cluster.adjusted_p_values[feature].to_string());
            }
            wtr.write_record(record)?;
        }
        Ok(())
    }

    /// create dataset
    fn get_ds(&self, res: &[DiffExpResult], group: &Group, name: &str) -> Result<Dataset> {
        Ok(group
            .new_dataset::<f64>()
            .shuffle()
            .deflate(1)
            .chunk((1 << 16, 3 * res.len()))
            .shape(Extents::new((self.mat.feature_ids.len(), 3 * res.len())).resizable())
            .create(name)?)
    }

    /// write dataset in group
    pub fn dump_h5_results(&self, res: &[DiffExpResult], dataset_name: &str) -> Result<()> {
        let mut data = Vec::<f64>::new();
        let shape = (self.mat.feature_ids.len(), 3 * res.len());
        for feature in 0..self.mat.feature_ids.len() {
            for cluster in res {
                data.push(cluster.normalized_mean_in[feature]);
                data.push(cluster.log2_fold_change[feature]);
                data.push(cluster.adjusted_p_values[feature]);
            }
        }
        let a = Array::from_shape_vec(shape, data)?;
        let group = self.group_all_diff_exp.create_group(dataset_name)?;
        self.get_ds(res, &group, "data")?.as_writer().write(&a)?;
        Ok(())
    }

    /// write feature indices in group
    pub fn dump_h5_feature_indices(&self) -> Result<()> {
        let ds = self
            .file_h5
            .new_dataset::<u32>()
            .shuffle()
            .deflate(1)
            .chunk((1 << 16,))
            .shape(Extents::new(self.mat.feature_ids.len()).resizable())
            .create("diffexp_feature_indices")?;
        let data: Vec<u32> = (0..(self.mat.feature_ids.len() as u32)).collect();
        ds.write(&data)?;
        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::sseq_differential_expression;
    use approx::assert_relative_eq;
    use flate2::read::GzDecoder;
    use scan_types::matrix::AdaptiveFeatureBarcodeMatrix;
    use serde::Serialize;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;

    pub fn compare_sseq_params(sseq_params_json: &Path, sseq_params: &SSeqParams, cell_indices: Option<&[usize]>) {
        let reader = GzDecoder::new(File::open(sseq_params_json).unwrap());
        let expected_sseq_params: SSeqParams = serde_json::from_reader(reader).unwrap();
        assert_eq!(expected_sseq_params.num_cells, sseq_params.num_cells);
        assert_eq!(expected_sseq_params.num_genes, sseq_params.num_genes);
        let size_factors = match cell_indices {
            Some(cells) => {
                let mut size_factors = Vec::with_capacity(cells.len());
                for &cell in cells {
                    size_factors.push(sseq_params.size_factors[cell]);
                }
                size_factors
            }
            None => sseq_params.size_factors.clone(),
        };
        assert_relative_eq!(
            &expected_sseq_params.size_factors[..],
            &size_factors[..],
            epsilon = 1e-10
        );
        assert_relative_eq!(
            &expected_sseq_params.gene_means[..],
            &sseq_params.gene_means[..],
            epsilon = 1e-7
        );
        assert_relative_eq!(
            &expected_sseq_params.gene_variances[..],
            &sseq_params.gene_variances[..],
            epsilon = 1e-7
        );
        assert_eq!(expected_sseq_params.use_genes, sseq_params.use_genes);
        assert_relative_eq!(
            &expected_sseq_params.gene_moment_phi[..],
            &sseq_params.gene_moment_phi[..],
            epsilon = 1e-7
        );
        assert_relative_eq!(expected_sseq_params.zeta_hat, sseq_params.zeta_hat, epsilon = 1e-7);
        assert_relative_eq!(expected_sseq_params.delta, sseq_params.delta, epsilon = 1e-7);
        assert_relative_eq!(
            &expected_sseq_params.gene_phi[..],
            &sseq_params.gene_phi[..],
            epsilon = 1e-7
        );
    }

    /// test and compare diff exp computation
    #[expect(clippy::too_many_arguments)]
    pub fn test_sseq_differential_expression(
        input_type: &str,
        matrix_path: &Path,
        clustering_path: &Path,
        library_type: &str,
        clustering_key: &str,
        sseq_params_json: &Path,
        use_umi: bool,
        min_feature_sum: Option<u32>,
        zeta_quintile: Option<f64>,
        feature_selection: Option<&[u32]>,
        barcode_selection: Option<&[usize]>,
        ground_truth_file: &Path,
        output_file: Option<&Path>,
    ) -> Result<Vec<(i16, DiffExpResult)>> {
        let (mat, sseq_params, _) = init_matrix(
            input_type,
            matrix_path,
            library_type,
            use_umi,
            min_feature_sum,
            zeta_quintile,
            feature_selection,
            barcode_selection,
        )?;

        compare_sseq_params(sseq_params_json, &sseq_params, barcode_selection);

        let conditions = initial_cluster_assignments(input_type, clustering_path, clustering_key)?;

        let res = conditions
            .iter()
            .take(1)
            .map(|(cluster, cond_a, cond_b)| {
                let res = sseq_differential_expression(&mat.matrix, cond_a, cond_b, &sseq_params, None);
                (*cluster, res)
            })
            .collect::<Vec<(i16, DiffExpResult)>>();

        for ires in &res {
            compare_exp(
                parse_path!(matrix_path, "matrix path is none")?,
                input_type,
                clustering_key,
                ires,
                &mat,
                ground_truth_file,
                output_file,
            );
        }

        Ok(res)
    }

    /// compare diff_exp
    fn compare_exp(
        input_file: &str,
        input_type: &str,
        clustering_key: &str,
        (cluster, res): &(i16, DiffExpResult),
        mat: &AdaptiveFeatureBarcodeMatrix,
        ground_truth_file: &Path,
        output_file: Option<&Path>,
    ) {
        let deg_map = match input_type {
            "loupe" => DegResult::read_deg_result(ground_truth_file, *cluster),
            "h5" => DegResult::read_deg_result_h5(input_file, clustering_key),
            _ => unimplemented!(),
        };
        CompareResult::diff_result(output_file, deg_map, res, mat);
    }

    /// Differential Result Record
    #[derive(Debug)]
    pub struct DegResult {
        id: String,
        name: String,
        mean: f64,
        log2_fold_change: f64,
        adj_p_value: f64,
        tested: bool,
    }

    impl DegResult {
        pub fn read_deg_result(path: impl AsRef<Path>, cluster: i16) -> HashMap<String, DegResult> {
            let input = File::open(&path).unwrap_or_else(|_| panic!("can't open file {}", path.as_ref().display()));
            let input = GzDecoder::new(input);
            let mut reader = csv::Reader::from_reader(input);

            let m: HashMap<String, DegResult> = reader
                .records()
                .filter_map(|record| match record {
                    Ok(r) => {
                        let c = r[2]
                            .parse::<i16>()
                            .unwrap_or_else(|_| panic!("not a cluster: {}", &r[2]))
                            - 1;
                        if c != cluster {
                            return None;
                        }

                        Some((
                            r[0].to_owned(),
                            DegResult {
                                id: r[0].to_owned(),
                                name: r[1].to_owned(),
                                mean: r[3].parse::<f64>().unwrap(),
                                log2_fold_change: r[4].parse::<f64>().unwrap(),
                                adj_p_value: r[5].parse::<f64>().unwrap(),
                                tested: true,
                            },
                        ))
                    }
                    Err(r) => {
                        panic!("{r:?}");
                    }
                })
                .collect();
            m
        }

        fn read_deg_result_h5(analysis_h5: &str, clustering_key: &str) -> HashMap<String, DegResult> {
            let diff_exp_res = hdf5_io::analysis::get_differential_expression(analysis_h5, clustering_key).unwrap();
            let matrix = hdf5_io::matrix::get_matrix(analysis_h5).unwrap();
            let features_ids = hdf5_io::matrix::read_features_between("id", 0, None, &matrix).unwrap();
            let features_name = hdf5_io::matrix::read_features_between("name", 0, None, &matrix).unwrap();
            let m: HashMap<String, DegResult> = diff_exp_res
                .iter()
                .enumerate()
                .map(|(idx, record)| {
                    (
                        features_ids[idx].clone(),
                        DegResult {
                            id: features_ids[0].clone(),
                            name: features_name[idx].clone(),
                            mean: record[0],
                            log2_fold_change: record[1],
                            adj_p_value: record[2],
                            tested: true,
                        },
                    )
                })
                .collect();
            m
        }
    }

    /// DiffResult
    #[derive(Serialize)]
    pub struct CompareResult {
        id: String,
        name_1: String,
        name_2: String,
        mean_1: f64,
        mean_2: f64,
        log2_1: f64,
        log2_2: f64,
        adj_p_1: f64,
        adj_p_2: f64,
        tested_1: bool,
        tested_2: bool,
        mean_diff: f64,
        log2_diff: f64,
        p_val_diff: f64,
    }

    impl CompareResult {
        pub fn diff_result(
            out_file: Option<&Path>,
            deg_map: HashMap<String, DegResult>,
            res: &DiffExpResult,
            mat: &AdaptiveFeatureBarcodeMatrix,
        ) {
            let mut wtr = out_file.map(|f| {
                let output = File::create(f).unwrap_or_else(|_| panic!("can t create file {} ", f.display()));
                csv::Writer::from_writer(output)
            });
            for (idx, &adj_p_value) in res.adjusted_p_values.iter().enumerate() {
                let key = &mat.feature_ids[idx];
                let deg_r = deg_map.get(key).with_context(|| key.clone()).unwrap();
                let name = mat.feature_names[idx].clone();
                let mean = res.normalized_mean_in[idx];
                let log2_fold_change = res.log2_fold_change.to_vec()[idx];
                let tested = res.genes_tested.to_vec()[idx];
                let cmp_rec = CompareResult {
                    id: deg_r.id.clone(),
                    name_1: deg_r.name.clone(),
                    name_2: name.clone(),
                    mean_1: deg_r.mean,
                    mean_2: mean,
                    log2_1: deg_r.log2_fold_change,
                    log2_2: log2_fold_change,
                    adj_p_1: deg_r.adj_p_value,
                    adj_p_2: adj_p_value,
                    tested_1: deg_r.tested,
                    tested_2: tested,
                    mean_diff: deg_r.mean - mean,
                    log2_diff: deg_r.log2_fold_change - log2_fold_change,
                    p_val_diff: deg_r.adj_p_value - adj_p_value,
                };
                if let Some(wtr) = wtr.as_mut() {
                    wtr.serialize(cmp_rec).expect("failed to write");
                }

                assert_eq!(name, deg_r.name);
                assert_relative_eq!(deg_r.mean, mean, epsilon = 1e-7);
                assert_relative_eq!(deg_r.log2_fold_change, log2_fold_change, epsilon = 5e-3);
                assert_relative_eq!(deg_r.adj_p_value, adj_p_value, epsilon = 5e-3);
            }

            if let Some(wtr) = wtr.as_mut() {
                wtr.flush().expect("Can't flush");
            }
        }
    }

    #[test]
    fn test_result_io_new() {
        struct TestInput<'a> {
            input_file: String,
            input_type: String,
            clustering_key: String,
            zeta_quintile: Option<f64>,
            feature_selection: Option<&'a [u32]>,
            barcode_selection: Option<&'a [usize]>,
        }
        let use_umi = true;
        let min_feature_sum = None;
        let library_type = GENE_EXPRESSION_LIBRARY_TYPE;

        let tests = maplit::hashmap! {
            "loupe" => TestInput {
                input_file: "./test/analysis.h5".to_owned(),
                input_type: "h5".to_owned(),
                clustering_key: "_kmeans_2_clusters".to_owned(),
                zeta_quintile: None,
                feature_selection: None,
                barcode_selection: None,
            },
        };
        let analysis_h5 = Path::new("ana.h5");
        let analysis_csv = Path::new("ana.csv");
        for (name, input) in tests {
            println!("testing {name} ....");
            let (mat, sseq_params, _use_genes) = init_matrix(
                &input.input_type,
                Path::new(&input.input_file),
                library_type,
                use_umi,
                min_feature_sum,
                input.zeta_quintile,
                input.feature_selection,
                input.barcode_selection,
            )
            .unwrap();
            let result_io = ResultIo::new(&mat, analysis_h5, analysis_csv).unwrap();

            let conditions =
                initial_cluster_assignments(&input.input_type, Path::new(&input.input_file), &input.clustering_key)
                    .unwrap();

            let res = conditions
                .iter()
                .map(|(_assignments, cond_a, cond_b)| {
                    sseq_differential_expression(&mat.matrix, cond_a, cond_b, &sseq_params, None)
                })
                .collect::<Vec<DiffExpResult>>();
            result_io.dump_h5_results(&res, &input.clustering_key).unwrap();
            result_io.dump_csv_results(&res, &input.clustering_key).unwrap();
        }
    }
}
