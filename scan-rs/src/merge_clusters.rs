use crate::diff_exp::{compute_sseq_params, sseq_differential_expression};
use crate::linkage::{linkage, Complete};
use crate::stats::median_array_rows_mut;
use log::info;
use ndarray::Array2;
use noisy_float::types::{n64, N64};
use scan_types::matrix::AdaptiveFeatureBarcodeMatrix as FBM;
use std::collections::{BTreeMap, HashMap, HashSet};

const ADJUSTED_P_VALUE_THRESHOLD: f64 = 0.05;

fn bincount(values: &[i16]) -> BTreeMap<i16, usize> {
    let mut res = BTreeMap::default();
    for &v in values {
        *res.entry(v).or_insert(0) += 1;
    }
    res
}

fn medioids(pca: &Array2<f64>, labels: &[i16], bins: &BTreeMap<i16, usize>) -> Array2<f64> {
    let (_, n) = pca.dim();
    let mut res = Array2::<f64>::zeros((bins.len(), n));
    for (i, (&label, &sz)) in bins.iter().enumerate() {
        let mut buf = Array2::<N64>::zeros((n, sz));
        let mut j = 0;
        for (k, &l) in labels.iter().enumerate() {
            if l == label {
                for c in 0..n {
                    buf[[c, j]] = n64(pca[[k, c]]);
                }
                j += 1;
            }
        }
        let medioid = median_array_rows_mut(&mut buf).expect("Mean calculation/Quantile Error");
        for c in 0..n {
            res[[i, c]] = medioid[c].raw();
        }
    }
    res
}

/// Relabel a clustering from greatest cluster size to least
pub fn relabel_by_size(mut labels: Vec<i16>) -> Vec<i16> {
    let mut hist = bincount(&labels).into_iter().collect::<Vec<_>>();
    hist.sort_by(|(_, x), (_, y)| y.cmp(x));
    let map = hist
        .into_iter()
        .enumerate()
        .map(|(i, j)| (j.0, i as i16))
        .collect::<HashMap<_, _>>();
    for x in labels.iter_mut() {
        *x = map[x];
    }
    labels
}

/// Iteratively merge clusters whose medioids are adjacent by complete linkage and have no
/// differentially expressed genes between them
pub fn merge_clusters(fbm: &FBM, pca: &Array2<f64>, mut labels: Vec<i16>) -> Vec<i16> {
    let m = labels.len();
    if m == 0 {
        return vec![];
    }
    let mut seen_pairs: HashSet<(Vec<usize>, Vec<usize>)> = HashSet::default();
    loop {
        let bins = bincount(labels.as_slice());
        let centers = medioids(pca, labels.as_slice(), &bins);
        let z = linkage(&centers, &Complete {});
        let max_label = labels.iter().fold(labels[0], |acc, &x| acc.max(x)) as f64;

        let mut any_merged = false;
        for i in 0..z.shape()[0] {
            if z[[i, 0]] <= max_label && z[[i, 1]] <= max_label {
                let leaf0 = z[[i, 0]] as i16;
                let leaf1 = z[[i, 1]] as i16;

                let mut group0 = vec![];
                let mut group1 = vec![];
                for (i, &l) in labels.iter().enumerate() {
                    if l == leaf0 {
                        group0.push(i);
                    }
                    if l == leaf1 {
                        group1.push(i);
                    }
                }
                if !seen_pairs.insert((group0.clone(), group1.clone())) {
                    continue;
                }
                let cell_indices = {
                    let mut cells = group0.clone();
                    cells.extend(group1.iter().cloned());
                    cells.sort_unstable();
                    cells
                };

                info!(
                    "computing DE params on {} x {} matrix",
                    fbm.matrix.rows(),
                    cell_indices.len()
                );
                let params = compute_sseq_params(&fbm.matrix, None, Some(&cell_indices), None);
                info!("computing DE on {} vs {} cells", group0.len(), group1.len());
                let de_result = sseq_differential_expression(&fbm.matrix, &group0, &group1, &params, None);

                let n_de_genes = de_result
                    .adjusted_p_values
                    .iter()
                    .filter(|&&pv| pv < ADJUSTED_P_VALUE_THRESHOLD)
                    .count();
                if n_de_genes == 0 {
                    info!(
                        "found {} DE genes, merging clusters {} and {}.",
                        n_de_genes,
                        leaf0 + 1,
                        leaf1 + 1
                    );
                    for l in labels.iter_mut() {
                        use std::cmp::Ordering;
                        match (*l).cmp(&leaf1) {
                            Ordering::Equal => *l = leaf0,
                            Ordering::Greater => *l -= 1,
                            Ordering::Less => {}
                        };
                    }
                    any_merged = true;
                    break;
                }
            }
        }

        if !any_merged {
            break;
        }
    }

    relabel_by_size(labels)
}

#[cfg(test)]
mod test {
    // TODO(lhepler): restore this using an npz file?)
    // use super::*;
    // use anyhow::Error;
    // use ndarray_npy::NpzReader;
    // use std::fs::File;

    // #[test]
    // fn test_merge_clusters() -> Result<(), Error> {
    //     let (fbm, gt_labels) = {
    //         let mut reader = CLoupeReader::new("./test/test2.cloupe")?;
    //         let section = reader.get_immutable_section()?;
    //         let (fbm, _) = read_adaptive_csr_matrix(
    //             &reader,
    //             &section.matrices[0],
    //             Some("Gene Expression"),
    //             Some(f64::MIN_POSITIVE),
    //             None,
    //             None,
    //         )?;
    //         let cluster = section.clusterings.iter().find(|&c| c.name == "Graph-based");
    //         let labels = read_clustering_assignments(&reader, cluster.unwrap())?;
    //         (fbm, labels)
    //     };
    //     let pca = {
    //         let f = File::open("./test/test2.pca.npz").unwrap();
    //         let mut rdr = NpzReader::new(f).unwrap();
    //         rdr.by_index(0).unwrap()
    //     };
    //     let initial_labels = vec![
    //         5i16, 3, 2, 1, 1, 2, 2, 0, 3, 0, 0, 2, 0, 3, 3, 2, 0, 5, 4, 2, 0, 5, 0, 5, 5, 1, 2, 0, 4, 0, 0, 5, 4, 0, 1,
    //         2, 1, 4, 0, 0, 3, 4, 5, 0, 1, 4, 1, 1, 5, 2, 4, 0, 0, 0, 1, 1, 1, 4, 3, 0, 5, 3, 5, 1, 4, 3, 4, 0, 0, 3, 0,
    //         1, 3, 2, 1, 3, 0, 2, 3, 5, 4, 1, 2, 1, 3, 3, 2, 1, 2, 2, 5, 3, 3, 3, 4, 2, 4, 2, 1, 4, 1, 0,
    //     ];
    //     let labels = merge_clusters(&fbm, &pca, initial_labels);
    //     assert_eq!(gt_labels, labels);
    //     Ok(())
    // }
}
