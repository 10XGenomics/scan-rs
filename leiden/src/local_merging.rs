use crate::{Clustering, Network, SimpleClustering, ZeroVec};
use rand::seq::SliceRandom;
use rand::Rng;

pub struct LocalMerging {
    randomness: f64,
    resolution: f64,
    cluster_weights: Vec<f64>,
    non_singleton_clusters: Vec<bool>,
    external_edge_weight_per_cluster: Vec<f64>,
    edge_weight_per_cluster: Vec<f64>,
    neighboring_clusters: Vec<usize>,
    cum_transformed_qv_incr_per_cluster: Vec<f64>,
    node_order: Vec<usize>,
}

impl LocalMerging {
    pub fn new(randomness: f64, resolution: f64) -> Self {
        LocalMerging {
            randomness,
            resolution,
            cluster_weights: Vec::new(),
            non_singleton_clusters: Vec::new(),
            external_edge_weight_per_cluster: Vec::new(),
            edge_weight_per_cluster: Vec::new(),
            neighboring_clusters: Vec::new(),
            cum_transformed_qv_incr_per_cluster: Vec::new(),
            node_order: Vec::new(),
        }
    }

    pub fn run(&mut self, n: &Network, rng: &mut impl Rng) -> SimpleClustering {
        let mut c = SimpleClustering::init_same_cluster(n.nodes());

        if n.nodes() == 1 {
            return c;
        }

        let mut update = false;

        let total_node_weight = n.get_total_node_weight();
        self.cluster_weights.clear();
        for i in 0..n.nodes() {
            self.cluster_weights.push(n.weight(i));
        }

        n.get_total_edge_weight_per_node(&mut self.external_edge_weight_per_cluster);

        // generate random permutation of the nodes
        self.node_order.clear();
        self.node_order.extend(0..n.nodes());
        self.node_order.shuffle(rng);

        self.non_singleton_clusters.zero_len(n.nodes());
        self.edge_weight_per_cluster.zero_len(n.nodes());
        self.neighboring_clusters.zero_len(n.nodes());

        for i in 0..n.nodes() {
            let j = self.node_order[i];

            /*
             * Only nodes belonging to singleton clusters can be moved to a
             * different cluster. This guarantees that clusters will never be
             * split up. Additionally, only nodes that are well connected with
             * the rest of the network are considered for moving.
             */
            let thresh = self.cluster_weights[j] * (total_node_weight - self.cluster_weights[j]) * self.resolution;
            if !self.non_singleton_clusters[j] && self.external_edge_weight_per_cluster[j] >= thresh {
                /*
                 * Remove the currently selected node from its current cluster.
                 * This causes the cluster to be empty.
                 */
                self.cluster_weights[j] = 0.0;
                self.external_edge_weight_per_cluster[j] = 0.0;

                /*
                 * Identify the neighboring clusters of the currently selected
                 * node, that is, the clusters with which the currently
                 * selected node is connected. The old cluster of the currently
                 * selected node is also included in the set of neighboring
                 * clusters. In this way, it is always possible that the
                 * currently selected node will be moved back to its old
                 * cluster.
                 */
                self.neighboring_clusters[0] = j;
                let mut num_neighboring_clusters = 1;

                for (neighbor, edge_weight) in n.neighbors(j) {
                    let neighbor_cluster = c.get(neighbor);
                    if self.edge_weight_per_cluster[neighbor_cluster] == 0.0 {
                        self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                        num_neighboring_clusters += 1;
                    }
                    self.edge_weight_per_cluster[neighbor_cluster] += edge_weight;
                }

                /*
                 * For each neighboring cluster of the currently selected node,
                 * determine whether the neighboring cluster is well connected
                 * with the rest of the network. For each neighboring cluster
                 * that is well connected, calculate the increment of the
                 * quality function obtained by moving the currently selected
                 * node to the neighboring cluster. For each neighboring
                 * cluster for which the increment is non-negative, calculate a
                 * transformed increment that will determine the probability
                 * with which the currently selected node is moved to the
                 * neighboring cluster.
                 */
                let mut best_cluster = j;
                let mut max_qv_increment = 0.0;
                let mut total_transformed_qv_increment = 0.0;
                self.cum_transformed_qv_incr_per_cluster.clear();
                for k in 0..num_neighboring_clusters {
                    // l is the neighboring cluster
                    let l = self.neighboring_clusters[k];

                    let thresh =
                        self.cluster_weights[l] * (total_node_weight - self.cluster_weights[l]) * self.resolution;
                    if self.external_edge_weight_per_cluster[l] >= thresh {
                        let qv_increment =
                            self.edge_weight_per_cluster[l] - n.weight(j) * self.cluster_weights[l] * self.resolution;

                        if qv_increment > max_qv_increment {
                            best_cluster = l;
                            max_qv_increment = qv_increment;
                        }

                        if qv_increment >= 0.0 {
                            total_transformed_qv_increment += (qv_increment / self.randomness).exp();
                        }
                    }

                    self.cum_transformed_qv_incr_per_cluster
                        .push(total_transformed_qv_increment);
                    self.edge_weight_per_cluster[l] = 0.0;
                }

                /*
                 * Determine the neighboring cluster to which the currently
                 * selected node will be moved.
                 */

                let mut chosen_cluster = best_cluster;

                if total_transformed_qv_increment < f64::INFINITY {
                    let r = total_transformed_qv_increment * rng.random::<f64>();
                    let mut min_idx = -1;
                    let mut max_idx = (num_neighboring_clusters + 1) as isize;

                    while min_idx < max_idx - 1 {
                        let mid_idx = (min_idx + max_idx) / 2;
                        if self.cum_transformed_qv_incr_per_cluster[mid_idx as usize] >= r {
                            max_idx = mid_idx;
                        } else {
                            min_idx = mid_idx;
                        }
                    }

                    chosen_cluster = self.neighboring_clusters[max_idx as usize];
                }

                /*
                 * Move the currently selected node to its new cluster and
                 * update the clustering statistics.
                 */
                self.cluster_weights[chosen_cluster] += n.weight(j);

                for (neighbor, edge_weight) in n.neighbors(j) {
                    if c.get(neighbor) == chosen_cluster {
                        self.external_edge_weight_per_cluster[chosen_cluster] -= edge_weight;
                    } else {
                        self.external_edge_weight_per_cluster[chosen_cluster] += edge_weight;
                    }
                }

                if chosen_cluster != j {
                    c.set(j, chosen_cluster);
                    self.non_singleton_clusters[chosen_cluster] = true;
                    update = true;
                }
            }
        }

        if update {
            c.remove_empty_clusters();
        }

        c
    }
}
