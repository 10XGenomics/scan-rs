#[allow(unused_imports)]
use crate::diff_exp;
#[cfg(hdf5)]
use anyhow::{format_err, Error};
#[cfg(hdf5)]
use hdf5_rs::Group;
#[cfg(hdf5)]
use scan_types::matrix::AdaptiveFeatureBarcodeMatrix;
#[cfg(hdf5)]
use std::path::Path;

/// ANTIBODY_LIBRARY_TYPE = "Antibody Capture"
pub static ANTIBODY_LIBRARY_TYPE: &str = "Antibody Capture";
/// GENE_EXPRESSION_LIBRARY_TYPE = "Gene Expression
pub static GENE_EXPRESSION_LIBRARY_TYPE: &str = "Gene Expression";
/// clustering group
pub static ANALYSIS_H5_CLUSTERING_GROUP: &str = "clustering";

#[cfg(hdf5)]
macro_rules! parse_path {
    ($path:expr, $msg:literal) => {
        match $path.to_str() {
            Some(file) => Ok(file),
            None => Err(format_err!("{}", $msg)),
        };
    };
}

/// Retrieve count matrix and compute global params sseq_params
#[cfg(hdf5)]
pub fn init_matrix(
    input_type: &str,
    matrix_path: &Path,
    _library_type: &str,
    use_umi: bool,
    min_row_sum: Option<f64>,
    zeta_quintile: Option<f64>,
    row_indices: Option<&[u32]>,
    col_indices: Option<&[usize]>,
) -> Result<(AdaptiveFeatureBarcodeMatrix, diff_exp::SSeqParams, Vec<usize>), Error> {
    let matrix_file = parse_path!(matrix_path, "matrix_path")?;

    let (mat, umi, cols_idx) = match input_type {
        #[cfg(hdf5)]
        "h5" => {
            let (csmat, _) = hdf5_io::matrix::read_adaptive_csr_matrix(matrix_file, Some(_library_type), min_row_sum)?;

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
    let umi = if umi.len() == 0 { None } else { Some(umi.as_slice()) };

    let sseq_params = diff_exp::compute_sseq_params(&mat.matrix, zeta_quintile, col_indices, umi);

    Ok((mat, sseq_params, cols_idx))
}

/// load clustering keys from analysis_h5
#[cfg(hdf5)]
pub fn get_clustering_keys(analysis_h5: &Path) -> Result<Vec<String>, Error> {
    let analysis_file = parse_path!(analysis_h5, "Analysis_h5 is none")?;
    hdf5_io::analysis::get_clustering_keys(analysis_file)
}

/// Retrieve clustering info and generate assignment vector
#[cfg(hdf5)]
pub fn initial_cluster_assignments(
    input_type: &str,
    clustering_path: &Path,
    clustering_key: &str,
) -> Result<Vec<(i16, Vec<usize>, Vec<usize>)>, Error> {
    let clustering_file = parse_path!(clustering_path, "clustering_path")?;

    let (clusters_count, assignments) = match input_type {
        #[cfg(hdf5)]
        "h5" => {
            let (clusters_count, v) = hdf5_io::analysis::get_clustering(clustering_file, clustering_key)?;
            (clusters_count, v.into_iter().map(|v| v - 1).collect::<Vec<i16>>())
        }
        _ => unreachable!(),
    };

    let conditions = (0..clusters_count)
        .into_iter()
        .map(|cluster| {
            let mut cond_a = Vec::<usize>::new();
            let mut cond_b = Vec::<usize>::new();
            for (idx, &val) in assignments.iter().enumerate() {
                if val == (cluster as i16) {
                    cond_a.push(idx as usize);
                } else {
                    cond_b.push(idx as usize);
                }
            }
            (cluster as i16, cond_a, cond_b)
        })
        .collect::<Vec<(i16, Vec<usize>, Vec<usize>)>>();

    Ok(conditions)
}

///  IO for differential expression Result
#[cfg(hdf5)]
pub struct ResultIo<'a> {
    mat: &'a AdaptiveFeatureBarcodeMatrix,
    analysis_csv: &'a Path,
    file_h5: hdf5_rs::File,
    group_diff_exp: Group,
    group_all_diff_exp: Group,
}

#[cfg(hdf5)]
impl<'a> ResultIo<'a> {
    /// create ResultIo to process Io
    pub fn new(
        mat: &'a AdaptiveFeatureBarcodeMatrix,
        analysis_h5: &'a Path,
        analysis_csv: &'a Path,
    ) -> Result<ResultIo<'a>, Error> {
        let file_h5 = hdf5_rs::File::create(analysis_h5)?;
        let group_diff_exp = file_h5.create_group("differential_expression")?;
        let group_all_diff_exp = file_h5.create_group("all_differential_expression")?;

        Ok(ResultIo {
            mat,
            analysis_csv,
            file_h5,
            group_diff_exp,
            group_all_diff_exp,
        })
    }

    /// header
    fn header(&self, res: &Vec<diff_exp::DiffExpResult>) -> Vec<String> {
        let mut h = vec!["Feature ID".to_owned(), "Feature Name".to_owned()];
        (1..(res.len() + 1)).into_iter().for_each(|i| {
            h.push(format!("Cluster {} Mean Counts", i));
            h.push(format!("Cluster {} Log2 fold change", i));
            h.push(format!("Cluster {} Adjusted p value", i));
        });
        h
    }

    fn get_output_file(&self, clustering_key: &str) -> Result<PathBuf, Error> {
        let dir = self.analysis_csv.join("diffexp").join(&clustering_key[1..]);
        fs::create_dir_all(dir.as_path())?;
        Ok(dir.as_path().join("differential_expression.csv"))
    }

    /// dump differential expression results to csv file
    pub fn dump_csv_results(&self, res: &Vec<diff_exp::DiffExpResult>, clustering_key: &str) -> Result<(), Error> {
        let out_file = self.get_output_file(clustering_key)?;
        let output = File::create(out_file)?;

        let mut wtr = csv::WriterBuilder::new().from_writer(output);

        let features_count = res[0].common_mean.len();

        wtr.write_record(self.header(res))?;
        for feature in 0..features_count {
            let mut record = Vec::<String>::new();
            record.push(self.mat.feature_ids[feature].to_string());
            record.push(self.mat.feature_names[feature].to_string());
            res.iter().for_each(|cluster| {
                record.push(cluster.normalized_mean_in[feature].to_string());
                record.push(cluster.log2_fold_change[feature].to_string());
                record.push(cluster.adjusted_p_values[feature].to_string());
            });
            wtr.write_record(record)?;
        }
        Ok(())
    }

    /// create dataset
    fn get_ds(&self, res: &Vec<diff_exp::DiffExpResult>, group: &Group, name: &str) -> Result<hdf5_rs::Dataset, Error> {
        let ds = group
            .new_dataset::<f64>()
            .resizable(true)
            .shuffle(true)
            .gzip(1)
            .chunk((1 << 16, 3 * res.len()))
            .create(name, (self.mat.feature_ids.len(), 3 * res.len()))?;
        Ok(ds)
    }

    /// write dataset in group
    pub fn dump_h5_results(&self, res: &Vec<diff_exp::DiffExpResult>, dataset_name: &str) -> Result<(), Error> {
        let mut data = Vec::<f64>::new();
        let shape = (self.mat.feature_ids.len(), 3 * res.len());
        for feature in 0..self.mat.feature_ids.len() {
            res.iter().for_each(|cluster| {
                data.push(cluster.normalized_mean_in[feature]);
                data.push(cluster.log2_fold_change[feature]);
                data.push(cluster.adjusted_p_values[feature]);
            });
        }
        let a = Array::from_shape_vec(shape, data)?;
        let group = self.group_all_diff_exp.create_group(dataset_name)?;
        self.get_ds(res, &group, "data")?.as_writer().write(&a)?;
        Ok(())
    }

    /// write feature indices in group
    pub fn dump_h5_feature_indices(&self) -> Result<(), Error> {
        let ds = self
            .file_h5
            .new_dataset::<u32>()
            .resizable(true)
            .shuffle(true)
            .gzip(1)
            .chunk((1 << 16,))
            .create("diffexp_feature_indices", (self.mat.feature_ids.len(),))?;
        let data: Vec<u32> = (0..(self.mat.feature_ids.len() as u32)).collect();
        ds.write(&data)?;
        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use crate::diff_exp::SSeqParams;
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
    #[cfg(hdf5)]
    pub fn test_sseq_differential_expression(
        input_type: &str,
        matrix_path: &Path,
        clustering_path: &Path,
        library_type: &str,
        clustering_key: &str,
        sseq_params_json: &Path,
        use_umi: bool,
        min_feature_sum: Option<f64>,
        zeta_quintile: Option<f64>,
        feature_selection: Option<&[u32]>,
        barcode_selection: Option<&[usize]>,
        ground_truth_file: &Path,
        output_file: Option<&Path>,
    ) -> Result<Vec<(i16, diff_exp::DiffExpResult)>, Error> {
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
                let res = diff_exp::sseq_differential_expression(&mat.matrix, &cond_a, &cond_b, &sseq_params, None);
                (*cluster, res)
            })
            .collect::<Vec<(i16, diff_exp::DiffExpResult)>>();

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

    /// compare diff_exp against xena or cloupe file
    #[allow(dead_code)]
    fn compare_exp(
        _input_file: &str,
        input_type: &str,
        _clustering_key: &str,
        (cluster, res): &(i16, diff_exp::DiffExpResult),
        mat: &AdaptiveFeatureBarcodeMatrix,
        ground_truth_file: &Path,
        output_file: Option<&Path>,
    ) {
        //let analysis_path = "./test/analysis.h5";
        let xena_map = match input_type {
            "loupe" => XenaResult::read_xena_result(ground_truth_file, *cluster),
            #[cfg(hdf5)]
            "h5" => XenaResult::read_xena_result_h5(_input_file, _clustering_key),
            _ => unimplemented!("Not yet !"),
        };
        CompareResult::diff_result(output_file, xena_map, res, mat);
    }

    /// Xena Differential Result Record
    #[derive(Debug)]
    pub struct XenaResult {
        id: String,
        name: String,
        mean: f64,
        log2_fold_change: f64,
        adj_p_value: f64,
        tested: bool,
    }

    impl XenaResult {
        pub fn read_xena_result(path: impl AsRef<Path>, cluster: i16) -> HashMap<String, XenaResult> {
            let input = File::open(&path).unwrap_or_else(|_| panic!("can't open file {}", path.as_ref().display()));
            let input = GzDecoder::new(input);
            let mut reader = csv::Reader::from_reader(input);

            let m: HashMap<String, XenaResult> = reader
                .records()
                .flat_map(|record| match record {
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
                            XenaResult {
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

        #[cfg(hdf5)]
        fn read_xena_result_h5(analysis_h5: &str, clustering_key: &str) -> HashMap<String, XenaResult> {
            let diff_exp_res = hdf5_io::analysis::get_differential_expression(analysis_h5, clustering_key).unwrap();
            let matrix = hdf5_io::matrix::get_matrix(analysis_h5).unwrap();
            let features_ids = hdf5_io::matrix::read_features_between("id", 0, None, &matrix).unwrap();
            let features_name = hdf5_io::matrix::read_features_between("name", 0, None, &matrix).unwrap();
            let m: HashMap<String, XenaResult> = diff_exp_res
                .iter()
                .enumerate()
                .map(|(idx, record)| {
                    (
                        features_ids[idx].clone(),
                        XenaResult {
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
        name_xena: String,
        name_rust: String,
        mean_xena: f64,
        mean_rust: f64,
        log2_xena: f64,
        log2_rust: f64,
        adj_p_xena: f64,
        adj_p_rust: f64,
        tested_xena: bool,
        tested_rust: bool,
        mean_diff: f64,
        log2_diff: f64,
        p_val_diff: f64,
    }

    impl CompareResult {
        pub fn diff_result(
            out_file: Option<&Path>,
            xena_map: HashMap<String, XenaResult>,
            res: &diff_exp::DiffExpResult,
            mat: &AdaptiveFeatureBarcodeMatrix,
        ) {
            let mut wtr = out_file.map(|f| {
                let output = File::create(f).unwrap_or_else(|_| panic!("can t create file {} ", f.display()));
                csv::Writer::from_writer(output)
            });
            res.adjusted_p_values
                .to_vec()
                .iter()
                .enumerate()
                .for_each(|(idx, &adj_p_value)| {
                    let key = mat.feature_ids[idx].clone();
                    match xena_map.get(&key) {
                        Some(xena_r) => {
                            let name = mat.feature_names[idx].clone();
                            let mean = res.normalized_mean_in[idx];
                            let log2_fold_change = res.log2_fold_change.to_vec()[idx];
                            let tested = res.genes_tested.to_vec()[idx];
                            let cmp_rec = CompareResult {
                                id: xena_r.id.clone(),
                                name_xena: xena_r.name.clone(),
                                name_rust: name.clone(),
                                mean_xena: xena_r.mean,
                                mean_rust: mean,
                                log2_xena: xena_r.log2_fold_change,
                                log2_rust: log2_fold_change,
                                adj_p_xena: xena_r.adj_p_value,
                                adj_p_rust: adj_p_value,
                                tested_xena: xena_r.tested,
                                tested_rust: tested,
                                mean_diff: xena_r.mean - mean,
                                log2_diff: xena_r.log2_fold_change - log2_fold_change,
                                p_val_diff: xena_r.adj_p_value - adj_p_value,
                            };
                            if let Some(wtr) = wtr.as_mut() {
                                wtr.serialize(cmp_rec).expect("failed to write");
                            }

                            assert_eq!(name, xena_r.name);
                            // println!("name: {}, {}, {}", name, idx, res.genes_tested[idx]);
                            assert_relative_eq!(xena_r.mean, mean, epsilon = 1e-7);
                            assert_relative_eq!(xena_r.log2_fold_change, log2_fold_change, epsilon = 5e-3);
                            assert_relative_eq!(xena_r.adj_p_value, adj_p_value, epsilon = 5e-3);
                        }
                        None => panic!("can't map gene {key}"),
                    }
                });

            if let Some(wtr) = wtr.as_mut() {
                wtr.flush().expect("Can't flush");
            }
        }
    }

    #[cfg(hdf5)]
    #[test]
    fn test_result_io_new() {
        struct TestInput<'a> {
            input_file: String,
            input_type: String,
            clustering_key: String,
            zeta_quintile: Option<f64>,
            output_file: Option<String>,
            feature_selection: Option<&'a [u32]>,
            barcode_selection: Option<&'a [u32]>,
        }
        let use_umi = true;
        let min_feature_sum = None;
        let col_indices: Option<&[u32]> = None;
        let library_type = GENE_EXPRESSION_LIBRARY_TYPE;

        let tests = maplit::hashmap! {
            "loupe" => TestInput {
                input_file: "./test/analysis.h5".to_owned(),
                input_type: "h5".to_owned(),
                clustering_key: "_kmeans_2_clusters".to_owned(),
                zeta_quintile: None,
                output_file: None,
                feature_selection: None,
                barcode_selection: None,
            },
        };
        let analysis_h5 = Path::new("ana.h5");
        let analysis_csv = Path::new("ana.csv");
        tests.par_iter().for_each(|(name, input)| {
            info!("testing {} ....", name);
            let (mat, sseq_params, use_genes) = init_matrix(
                &input.input_type,
                &Path::new(&input.input_file),
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
                initial_cluster_assignments(&input.input_type, &Path::new(&input.input_file), &input.clustering_key)
                    .unwrap();

            let res = conditions
                .iter()
                .map(|(cond_a, cond_b)| {
                    diff_exp::sseq_differential_expression(&mat.matrix, &cond_a, &cond_b, &sseq_params, None)
                })
                .collect::<Vec<diff_exp::DiffExpResult>>();
            result_io.dump_h5_results(&res, &input.clustering_key).unwrap();
            result_io.dump_csv_results(&res, &input.clustering_key).unwrap();
        });
    }
}
