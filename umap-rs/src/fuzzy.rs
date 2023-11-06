use ndarray::{Array2, ArrayView1};
use sprs;

type Q = crate::utils::Q;
type P = usize;

const BANDWIDTH: Q = 1.0;
const NITER: usize = 64;
const SMOOTH_K_TOLERANCE: Q = 1e-5;
const MIN_K_DIST_SCALE: Q = 1e-3;

/// Given a set of data X, a neighborhood size, and a measure of distance compute the fuzzy simplicial set(here represented as a fuzzy graph in the form of a sparse matrix) associated
/// to the data. This is done by locally approximating geodesic distance at each point, creating a fuzzy simplicial set for each such point, and then combining all the local fuzzy
/// simplicial sets into a global one via a fuzzy union.

///Construct the local fuzzy simplicial sets of each point represented by its distances
///to its `n_neighbors` nearest neighbors, stored in `knn_indices` and `knn_distances`, normalizing the distances
///on the manifolds, and converting the metric space to a simplicial set.
///`n_points` indicates the total number of points of the original data, while `knn_indices` contains
///indices of some subset of those points (ie some subset of 1:`n_points`). If `knn_indices` represents
/// neighbors of the elements of some set with itself, then `knn_distances` should have `n_points` number of
/// columns. Otherwise, these two values may be nonequivalent.
/// If `apply_fuzzy_combine` is true, use intersections and unions to combine
///fuzzy sets of neighbors (default true).

pub fn fuzzy_simplicial_set(
    knn_indices: &Array2<P>,
    knn_distances: &Array2<Q>,
    local_connectivity: Q,
    set_operation_ratio: Q,
    apply_fuzzy_combine: bool,
    n_iter: Option<usize>, // op
    bandwidth: Option<Q>,  //optional
) -> sprs::CsMat<Q> {
    let (n_points, _) = knn_indices.dim();
    let n_iter = n_iter.unwrap_or(NITER);
    let bandwidth = bandwidth.unwrap_or(BANDWIDTH);

    let (sigmas, rhos) = smooth_knn_distances(knn_distances, local_connectivity, n_iter, bandwidth);

    let (rows, cols, values) = compute_membership_strengths(knn_indices, knn_distances, &sigmas, &rhos);

    let result = sprs::TriMat::from_triplets((n_points, n_points), rows, cols, values).to_csr();
    if !apply_fuzzy_combine {
        return result;
    }

    let transpose = result.transpose_view().to_csr();

    let prod_mat = sprs::binop::mul_mat_same_storage(&result, &transpose);
    let sum_mat = &result + &transpose;
    let lhs = &(&sum_mat - &prod_mat) * set_operation_ratio;
    let rhs = &prod_mat * (1.0 - set_operation_ratio);
    let ret = &lhs + &rhs;

    log::info!(">>>>>>non_zero rows counts {}", ret.nnz());
    ret
}

/// Compute the smooth_knn_distances
fn smooth_knn_distances(
    knn_distances: &Array2<Q>,
    local_connectivity: Q,
    n_iter: usize, // op
    bandwidth: Q,  //optional
) -> (Vec<Q>, Vec<Q>) {
    let (n_cells, k) = knn_distances.dim();
    let mut rho = vec![0.0; n_cells];
    let mut result = vec![0.0; n_cells];

    let scale = MIN_K_DIST_SCALE;

    for i in 0..n_cells {
        let non_zero_dist = knn_distances
            .row(i)
            .into_iter()
            .cloned()
            .filter(|&d| d > 0.0)
            .collect::<Vec<Q>>();
        if non_zero_dist.len() >= local_connectivity as usize {
            let index = local_connectivity.floor();
            let interpolation = local_connectivity - index;
            if index > 0.0 {
                let index = index as usize;
                rho[i] = non_zero_dist[index - 1];
                if interpolation > SMOOTH_K_TOLERANCE {
                    rho[i] += interpolation * (non_zero_dist[index] - non_zero_dist[index - 1]);
                }
            } else {
                rho[i] = interpolation * non_zero_dist[0];
            }
        } else if !non_zero_dist.is_empty() {
            rho[i] = non_zero_dist.iter().fold(Q::MIN, |a, &b| a.max(b));
        }

        result[i] = smooth_knn_dist(knn_distances.row(i), rho[i], k, bandwidth, n_iter);

        if rho[i] > 0.0 {
            let (_, n) = knn_distances.dim();
            let mean_ith_distances = knn_distances.row(i).fold(0.0, |acc, &v| acc + v) / n as Q;
            if result[i] < scale * mean_ith_distances {
                result[i] = scale * mean_ith_distances;
            }
        } else {
            let (m, n) = knn_distances.dim();
            let mean_distances = knn_distances.fold(0.0, |acc, &v| acc + v) / (m as Q * n as Q);
            if result[i] < scale * mean_distances {
                result[i] = scale * mean_distances;
            }
        }
    }
    (result, rho)
}

/// calculate sigma for an individual point
fn smooth_knn_dist(distances: ArrayView1<Q>, rho: Q, k: usize, bandwidth: Q, n_iter: usize) -> Q {
    let target = (k as Q).log2() * bandwidth;
    let mut lo = 0.0;
    let mut mid = 1.0;
    let mut hi = Q::MAX;

    for _ in 0..n_iter {
        let psum = distances
            .iter()
            .fold(0.0_f64, |acc, &v| acc + (-(v.max(-rho).max(0.0) / mid)).exp());

        if (psum - target).abs() < SMOOTH_K_TOLERANCE {
            break;
        }
        let two = 2.0;
        if psum > target {
            hi = mid;
            mid = lo + (hi - lo) / two;
        } else {
            lo = mid;
            if hi == Q::MAX {
                mid *= two;
            } else {
                mid = lo + (hi - lo) / two
            }
        }
    }
    mid
}

/// Compute compute_membership_strengths
fn compute_membership_strengths(
    knn_indices: &Array2<P>,
    knn_distances: &Array2<Q>,
    sigmas: &[Q],
    rhos: &[Q],
) -> (Vec<usize>, Vec<usize>, Vec<Q>) {
    let n_samples = knn_indices.shape()[0];
    assert!(!knn_indices.is_empty());
    let n_neighbors = knn_indices.shape()[1];
    let mut rows: Vec<usize> = vec![0; n_samples * n_neighbors];
    let mut cols: Vec<usize> = vec![0; n_samples * n_neighbors];
    let mut values: Vec<Q> = vec![0.0; n_samples * n_neighbors];
    let mut k = 0;
    for i in 0..n_samples {
        for j in 0..n_neighbors {
            if knn_indices[[i, j]] == P::MAX {
                continue;
            }
            let val = if j == knn_indices[[i, j]] {
                0.0
            } else if knn_distances[[i, j]] - rhos[i] <= 0.0 || sigmas[i] == 0.0 {
                1.0
            } else {
                (-((knn_distances[[i, j]] - rhos[i]) / sigmas[i])).exp()
            };
            cols[k] = i;
            rows[k] = knn_indices[[i, j]];
            values[k] = val;
            k += 1;
        }
    }

    (rows, cols, values)
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use ndarray::{arr1, arr2};

    use super::*;

    #[test]
    fn fuzzy_simplicial_set_test() {
        let knns = arr2(&[[1, 2], [0, 2], [1, 0]]);
        let dists = arr2(&[[1.5, 0.5], [0.5, 2.], [1.5, 2.]]);
        let umap_graph = fuzzy_simplicial_set(&knns, &dists, 1.0, 1., true, Some(NITER), Some(1.0));
        //let t = umap_graph.clone().transpose_into();
        //assert_eq!(umap_graph, t);
        assert_eq!(umap_graph.shape(), (3, 3))
    }
    #[test]
    fn smooth_knn_dist_test() {
        let dists = arr1(&[0., 1., 2., 3., 4., 5.]);
        let rho = 1.0;
        let k = 6;
        let bandwidth = 1.;
        let niter = 64;
        let sigma = smooth_knn_dist(dists.view(), rho, k, bandwidth, niter);
        println!("sigma {sigma}");
        let psum = dists
            .iter()
            .fold(0.0_f64, |acc, &v| acc + (-(v.max(-rho).max(0.0) / sigma)).exp());
        let ret = psum - (k as Q).log2() * bandwidth;
        println!("psum {psum}  lg2 {}", (k as Q).log2() * bandwidth);
        println!("ret {ret} K_TOLERANCE {SMOOTH_K_TOLERANCE}");
        assert!(ret <= SMOOTH_K_TOLERANCE);
    }
    #[test]

    fn smooth_knn_distances_test() {
        let local_connectivity = 1.0;
        let bandwidth = 1.;
        let niter = 64;
        let knn_distances = arr2(&[
            [0., 0., 0.0],
            [1., 2., 3.],
            [2., 4., 5.],
            [3., 4., 5.],
            [4., 6., 6.],
            [5., 6., 10.],
        ]);

        let (_, rhos) = smooth_knn_distances(&knn_distances, local_connectivity, niter, bandwidth);

        assert_eq!(rhos, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let knn_distances = arr2(&[[0., 0., 0.0], [0., 1., 2.], [0., 2., 3.]]);

        let (_, rhos) = smooth_knn_distances(&knn_distances, local_connectivity, niter, bandwidth);
        assert_eq!(rhos, vec![0., 1., 2.]);

        let (_, rhos) = smooth_knn_distances(&knn_distances, 1.5, niter, bandwidth);
        assert_eq!(rhos, vec![0., 1.5, 2.5]);
    }

    #[test]
    fn compute_membership_strengths_test() {
        let knns = arr2(&[[0, 1, 2], [1, 0, 1]]);
        let dists = arr2(&[[0., 0., 0.], [2., 2., 3.]]);
        let rhos = vec![2., 1., 4.];
        let sigmas = vec![1., 1., 1.];
        let true_rows = vec![0, 1, 2, 1, 0, 1];
        let true_cols = vec![0, 0, 0, 1, 1, 1];
        let true_vals = vec![
            0.0,
            0.0,
            0.0,
            0.36787944117144233,
            0.36787944117144233,
            0.1353352832366127,
        ];
        let (rows, cols, vals) = compute_membership_strengths(&knns, &dists, &sigmas, &rhos);
        assert_eq!(rows, true_rows);
        assert_eq!(cols, true_cols);
        assert_eq!(vals, true_vals);
    }
}
