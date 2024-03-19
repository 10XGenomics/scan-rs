//! Parallelized louvain clustering.
use crate::{Clustering, Network, ZeroVec};
use fxhash::FxHasher;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
#[derive(Default)]
pub(crate) struct ParallelLocalMoving {
    resolution: f64,
    // We store these here to reuse heap space between iterations.
    cluster_weights: Vec<f64>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
}

impl ParallelLocalMoving {
    pub fn new(resolution: f64) -> Self {
        ParallelLocalMoving {
            resolution,
            ..ParallelLocalMoving::default()
        }
    }

    /// Run a louvain iteration, but update all the nodes simultaneously using only information
    /// from the previous iteration.
    pub fn iterate<C: Clustering + Clone + Send + Sync + Default>(&mut self, n: &Network, c: &mut C) -> bool {
        let total_edge_weight = n.get_total_edge_weight_par();

        self.cluster_weights.zero_len(n.nodes());
        self.nodes_per_cluster.zero_len(n.nodes());

        // Serially compute statistics. This is very fast.
        for i in 0..n.nodes() {
            self.cluster_weights[c.get(i)] += n.weight(i);
            self.nodes_per_cluster[c.get(i)] += 1;
        }

        let mut initial_num_unused_clusters = 0;
        self.unused_clusters.zero_len(n.nodes());

        // make a list of unused cluster ids.
        for i in (0..n.nodes()).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[initial_num_unused_clusters] = i;
                initial_num_unused_clusters += 1;
            }
        }

        // Doesn't need to be permuted, because processing order is irrelevant.
        // (There are no serial dependencies among individual node updates).
        let node_order = (0..n.nodes()).collect::<Vec<_>>();

        let chunk_size = ((n.nodes() as f64) / (rayon::current_num_threads() as f64)) as usize;
        let chunk_size = std::cmp::max(256, chunk_size);

        let mut updates = vec![0usize; n.nodes()];
        node_order
            .par_chunks(chunk_size)
            .zip(updates.par_chunks_mut(chunk_size))
            .for_each(|(nodes, updates)| {
                let mut neighboring_clusters = vec![0usize; n.nodes()];
                let mut edge_weight_per_cluster = vec![0f64; n.nodes()];

                // Process a chunk of nodes
                for (node_idx, update) in nodes.iter().zip(updates.iter_mut()) {
                    let j = *node_idx;
                    let current_cluster = c.get(j);

                    // Compute the new cluster stats after removing the currently selected node from its current cluster.
                    let curr_cluster_nodes = self.nodes_per_cluster[current_cluster] - 1;
                    let curr_cluster_unused = curr_cluster_nodes == 0;

                    /*
                     * Identify the neighboring clusters of the currently selected
                     * node, that is, the clusters with which the currently selected
                     * node is connected. An empty cluster is also included in the set
                     * of neighboring clusters. In this way, it is always possible that
                     * the currently selected node will be moved to an empty cluster.
                     */
                    if curr_cluster_unused {
                        neighboring_clusters[0] = current_cluster;
                    } else {
                        neighboring_clusters[0] = self.unused_clusters[initial_num_unused_clusters - 1];
                    }
                    let mut num_neighboring_clusters = 1;

                    for (target, edge_weight) in n.neighbors(j) {
                        //for edge in n.graph.edges((j as u32).into()) {
                        //   let neighbor_cluster = c.get(edge.target().index() as usize);
                        let neighbor_cluster = c.get(target);

                        if edge_weight_per_cluster[neighbor_cluster] == 0.0 {
                            // Encountered a new neighboring cluster; add it to the list of neighboring clusters.
                            neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                            num_neighboring_clusters += 1;
                        }
                        edge_weight_per_cluster[neighbor_cluster] += edge_weight;
                    }

                    /*
                     * For each neighboring cluster of the currently selected node,
                     * calculate the increment of the quality function obtained by
                     * moving the currently selected node to the neighboring cluster.
                     * Determine the neighboring cluster for which the increment of the
                     * quality function is largest. The currently selected node will be
                     * moved to this optimal cluster.
                     */
                    let mut best_cluster = 0;
                    let mut max_qv_increment = f64::NEG_INFINITY;

                    for &l in &neighboring_clusters[..num_neighboring_clusters] {
                        let cluster_weight = if l == current_cluster {
                            // Use the most up-to-date information (including the removal of this node)
                            self.cluster_weights[l] - n.weight(j)
                        } else {
                            // Use the previous iteration's information.
                            self.cluster_weights[l]
                        };
                        let qv_increment = edge_weight_per_cluster[l]
                            - n.weight(j) * cluster_weight * self.resolution / (2.0 * total_edge_weight);

                        if qv_increment > max_qv_increment {
                            best_cluster = l;
                            max_qv_increment = qv_increment;
                        } else if qv_increment == max_qv_increment && l != current_cluster {
                            // Generalized minimum label heursitic.
                            // When there are multiple optimal clusters, choose the cluster
                            // with the smallest label (by some arbitrary ordering of cluster labels).
                            let mut h = FxHasher::default();
                            l.hash(&mut h);
                            let l_hash = h.finish();
                            let mut h = FxHasher::default();
                            best_cluster.hash(&mut h);
                            let best_cluster_hash = h.finish();

                            if l_hash < best_cluster_hash {
                                best_cluster = l;
                            }
                        }
                        // Erase any updated values in `edge_weight_per_cluster` to give the next node a blank slate.
                        edge_weight_per_cluster[l] = 0.0;
                    }

                    *update = best_cluster;
                }
            });

        let mut changed = false;
        for (i, new_cluster_label) in updates.into_iter().enumerate() {
            changed |= c.get(i) != new_cluster_label;
            c.set(i, new_cluster_label);
        }
        if changed {
            c.remove_empty_clusters();
        }

        changed
    }
}
