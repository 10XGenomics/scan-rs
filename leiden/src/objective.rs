use crate::{Clustering, Network};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

/// The 'Constant Pott's Model' objective function of a network clustering.
/// Related to the 'modularity' -- see paper for details.
pub fn cpm(resolution: f64, graph: &Network, clustering: &impl Clustering) -> f64 {
    let mut quality = 0.0f64;
    let mut total_edge_weight = 0.0f64;

    for e in graph.graph.edge_references() {
        let c1 = clustering.get(e.source().index() as usize);
        let c2 = clustering.get(e.target().index() as usize);

        if c1 == c2 {
            quality += 2.0 * *e.weight() as f64;
        }

        total_edge_weight += *e.weight() as f64;
    }

    let mut cluster_weights = vec![0.0; clustering.num_clusters()];

    for i in 0..graph.nodes() {
        cluster_weights[clustering.get(i)] += graph.weight(i);
    }

    for cluster_weight in cluster_weights {
        quality -= cluster_weight * cluster_weight * resolution / (2.0 * total_edge_weight);
    }

    // Note we are dropping the factor of 2 mention in the paper here.
    // The results presented in the paper only make sense if you double-count each edge and divide by 2,
    // or single-count each edge and don't divide by 2.

    quality / (2.0 * total_edge_weight)
}

/// The 'Constant Pott's Model' objective function of a network clustering.
/// Related to the 'modularity' -- see paper for details.
/// Computed using parallelization.
pub fn par_cpm<C: Clustering + Sync>(resolution: f64, graph: &Network, clustering: &C) -> f64 {
    // Create a number of chunks that is large relative to typical thread-counts
    // To allow rayon to balance the uneven chunk loads induced by the "node ordering" constraint.
    let chunk_size = std::cmp::max(1, ((graph.nodes() as f64) / 64.0) as usize);
    let qual_weight_chunks = (0..graph.nodes())
        .collect::<Vec<usize>>()
        .par_chunks(chunk_size)
        .map(|nodes| {
            let mut quality = 0f64;
            let mut total_edge_weight = 0f64;

            for i in nodes {
                let c_i = clustering.get(*i);
                for edge in graph.graph.edges((*i as u32).into()) {
                    let j = edge.target().index() as usize;
                    // Enforce ordering of node indices to avoid processing edges twice.
                    if j < *i {
                        let edge_weight = *edge.weight() as f64;
                        total_edge_weight += edge_weight;
                        let c_j = clustering.get(j);
                        quality += if c_i == c_j { 2.0 * edge_weight } else { 0.0 };
                    }
                }
            }
            (quality, total_edge_weight)
        })
        .collect::<Vec<(f64, f64)>>();

    // Reduce serially to ensure deterministic order of adds
    let (mut quality, total_edge_weight) = qual_weight_chunks
        .into_iter()
        .fold((0f64, 0f64), |a, b| (a.0 + b.0, a.1 + b.1));

    let mut cluster_weights = vec![0.0; clustering.num_clusters()];

    for i in 0..graph.nodes() {
        cluster_weights[clustering.get(i)] += graph.weight(i);
    }

    for cluster_weight in cluster_weights {
        quality -= cluster_weight * cluster_weight * resolution / (2.0 * total_edge_weight);
    }

    // Note we are dropping the factor of 2 mention in the paper here.
    // The results presented in the paper only make sense if you double-count each edge and divide by 2,
    // or single-count each edge and don't divide by 2.

    quality / (2.0 * total_edge_weight)
}
