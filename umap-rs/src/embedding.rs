use log::{info, warn};
use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::Array2;
use ndarray_linalg::lobpcg::LobpcgResult;
use ndarray_linalg::{self as na};
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use sprs::{CompressedStorage, CsMat};
use std::time::Instant;

type Q = crate::utils::Q;

// initializing and optimizing embeddings
pub fn initialize_embedding(graph: &CsMat<Q>, embedding_dim: usize, random: &mut impl Rng) -> Array2<Q> {
    //Todo: create a parameters later
    let max_cols: usize = 10000;
    if graph.cols() < max_cols {
        spectral_layout(graph, embedding_dim)
    } else {
        let mut embedding = vec![0.0_f64; graph.rows() * embedding_dim];
        crate::utils::uniform(&mut embedding, 10.0, random);
        Array2::from_shape_vec((graph.rows(), embedding_dim), embedding).unwrap()
    }
}

/// Initialize a fuzzy simplicial set embedding, using a specified initialization method and then minimizing the fuzzy set cross entropy between the 1-skeletons of the high and low
/// dimensional fuzzy simplicial sets.
pub fn initialize_simplicial_set_embedding(
    graph: &mut CsMat<Q>,
    n_epochs: Q,
    random: &mut impl Rng,
) -> (Vec<usize>, Vec<usize>, Vec<Q>) {
    let mut weights = Vec::<Q>::new();
    let mut head = Vec::<usize>::new();
    let mut tail = Vec::<usize>::new();

    let graph_max = graph.iter().fold(0.0_f64, |acc, (&value, _)| acc.max(value));

    graph.map_inplace(|&value| if value < graph_max / n_epochs { 0.0 } else { value });

    // Get graph data in ordered way...
    graph.iter().for_each(|(&value, (row, col))| {
        if value != 0.0 {
            weights.push(value);
            tail.push(row);
            head.push(col);
        }
    });

    shuffle_together(&mut head, &mut tail, &mut weights, random);

    (head, tail, make_epochs_per_sample(&weights, n_epochs))
}

fn shuffle_together<T, T2, T3>(list: &mut [T], other: &mut [T2], weights: &mut [T3], random: &mut impl Rng)
where
    T: Copy,
    T2: Copy,
    T3: Copy,
{
    let mut n = list.len();
    if other.len() != n {
        panic!("Error incompatible");
    }
    while n > 1 {
        n -= 1;
        let k = random.gen_range(0..n + 1);
        list.swap(k, n);

        other.swap(k, n);

        weights.swap(k, n);
    }
}

fn make_epochs_per_sample(weights: &[Q], n_epochs: Q) -> Vec<Q> {
    let mut result = vec![-1.0; weights.len()];
    let max = weights.iter().fold(Q::MIN, |a, &b| a.max(b));
    weights.iter().enumerate().for_each(|(i, &w)| {
        let n = (w / max) * (n_epochs);
        if n > 0.0 {
            result[i] = n_epochs / n;
        }
    });
    result
}

fn spectral_layout(graph: &CsMat<Q>, embedding_dim: usize) -> Array2<Q> {
    let tick = Instant::now();
    let dim = graph.cols();
    let k = embedding_dim + 1;

    let i = CsMat::<Q>::eye(dim);
    let d =
        graph
            .degrees()
            .into_iter()
            .enumerate()
            .fold(CsMat::empty(CompressedStorage::CSR, dim), |mut d, (i, deg)| {
                d.insert(i, i, 1.0 / (deg as Q).sqrt());
                d
            });
    let l = &i - &(&(&d * graph) * &d);

    let initial = Pcg64Mcg::seed_from_u64(423)
        .sample_iter(Standard)
        .take(dim * k)
        .collect::<Vec<Q>>();

    // approximation
    let x = Array2::from_shape_vec((dim, k), initial).unwrap();
    let max_iter = 20;
    let tol = 1e-8;
    let order = na::TruncatedOrder::Smallest;

    info!(
        "initialization for {} samples took {:.3}s",
        dim,
        tick.elapsed().as_millis() as f64 / 1000.0
    );

    let result = na::lobpcg::lobpcg(|y| l.dot(&y), x, |_| {}, None, tol, max_iter, order);
    match result {
        LobpcgResult::NoResult(err) => panic!("Did not converge: {err:?}"),
        LobpcgResult::Ok(values, vecs, r_norms) | LobpcgResult::Err(values, vecs, r_norms, _) => {
            // check convergence
            for (i, norm) in r_norms.into_iter().enumerate() {
                if norm > 1e-5 {
                    warn!("The {}th eigenvalue estimation did not converge!", i);
                    warn!("Too large deviation of residual norm: {} > 1e-5", norm);
                }
            }
            let mut sorted = values
                .iter()
                .enumerate()
                .map(|(i, &val)| (i, val))
                .collect::<Vec<(usize, Q)>>();

            sorted.sort_by(|&a, &b| -> std::cmp::Ordering { (a.1).partial_cmp(&(b.1)).unwrap() });
            let indices = sorted[1..k].iter().map(|(i, _)| *i).collect::<Vec<usize>>();
            vecs.select(Axis(1), &indices).as_standard_layout().into_owned()
        }
    }
}
