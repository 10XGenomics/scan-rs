use crate::{Clustering, Network, ZeroVec};
use rand::{seq::SliceRandom, Rng};

#[derive(Default)]
pub(crate) struct StandardLocalMoving {
    resolution: f64,
    cluster_weights: Vec<f64>,
    nodes_per_cluster: Vec<usize>,
    unused_clusters: Vec<usize>,
    node_order: Vec<usize>,
    edge_weight_per_cluster: Vec<f64>,
    neighboring_clusters: Vec<usize>,
}

impl StandardLocalMoving {
    pub fn new(resolution: f64) -> Self {
        StandardLocalMoving {
            resolution,
            ..StandardLocalMoving::default()
        }
    }

    pub fn iterate(&mut self, n: &Network, c: &mut impl Clustering, rng: &mut impl Rng) -> bool {
        let mut update = false;

        let total_edge_weight = n.get_total_edge_weight();

        self.cluster_weights.zero_len(n.nodes());
        self.nodes_per_cluster.zero_len(n.nodes());

        for i in 0..n.nodes() {
            self.cluster_weights[c.get(i)] += n.weight(i);
            self.nodes_per_cluster[c.get(i)] += 1;
        }

        let mut num_unused_clusters = 0;
        self.unused_clusters.zero_len(n.nodes());

        // make a list of unused cluster ids.
        for i in (0..n.nodes()).rev() {
            if self.nodes_per_cluster[i] == 0 {
                self.unused_clusters[num_unused_clusters] = i;
                num_unused_clusters += 1;
            }
        }

        // generate random permutation of the nodes
        self.node_order.clear();
        self.node_order.extend(0..n.nodes());
        self.node_order.shuffle(rng);

        /*
         * Iterate over the node_oder array in a cyclical manner. When the end
         * of the array has been reached, start again from the beginning. The
         * queue of nodes that still need to be visited is given by
         * nodeOrder[i], ..., nodeOrder[i + nUnstableNodes - 1]. Continue
         * iterating until the queue is empty.
         */
        self.edge_weight_per_cluster.zero_len(n.nodes());
        self.neighboring_clusters.zero_len(n.nodes());

        let mut num_unstable_nodes = n.nodes();
        let mut i = 0;

        loop {
            let j = self.node_order[i];
            let current_cluster = c.get(j);

            // Remove the currently selected node from its current cluster.
            self.cluster_weights[current_cluster] -= n.weight(j);
            self.nodes_per_cluster[current_cluster] -= 1;
            if self.nodes_per_cluster[current_cluster] == 0 {
                self.unused_clusters[num_unused_clusters] = current_cluster;
                num_unused_clusters += 1;
            }

            /*
             * Identify the neighboring clusters of the currently selected
             * node, that is, the clusters with which the currently selected
             * node is connected. An empty cluster is also included in the set
             * of neighboring clusters. In this way, it is always possible that
             * the currently selected node will be moved to an empty cluster.
             */
            self.neighboring_clusters[0] = self.unused_clusters[num_unused_clusters - 1];
            let mut num_neighboring_clusters = 1;
            for (target, edge_weight) in n.neighbors(j) {
                let neighbor_cluster = c.get(target);

                if self.edge_weight_per_cluster[neighbor_cluster] == 0.0 {
                    self.neighboring_clusters[num_neighboring_clusters] = neighbor_cluster;
                    num_neighboring_clusters += 1;
                }
                self.edge_weight_per_cluster[neighbor_cluster] += edge_weight;
            }

            /*
             * For each neighboring cluster of the currently selected node,
             * calculate the increment of the quality function obtained by
             * moving the currently selected node to the neighboring cluster.
             * Determine the neighboring cluster for which the increment of the
             * quality function is largest. The currently selected node will be
             * moved to this optimal cluster. In order to guarantee convergence
             * of the algorithm, if the old cluster of the currently selected
             * node is optimal but there are also other optimal clusters, the
             * currently selected node will be moved back to its old cluster.
             */
            let mut best_cluster = current_cluster;
            let mut max_qv_increment = self.edge_weight_per_cluster[current_cluster]
                - n.weight(j) * self.cluster_weights[current_cluster] * self.resolution / (2.0 * total_edge_weight);

            for &l in &self.neighboring_clusters[..num_neighboring_clusters] {
                let qv_increment = self.edge_weight_per_cluster[l]
                    - n.weight(j) * self.cluster_weights[l] * self.resolution / (2.0 * total_edge_weight);
                if qv_increment > max_qv_increment {
                    best_cluster = l;
                    max_qv_increment = qv_increment;
                } else if qv_increment == max_qv_increment && l < best_cluster {
                    best_cluster = l;
                }
                self.edge_weight_per_cluster[l] = 0.0;
            }

            /*
             * Move the currently selected node to its new cluster. Update the
             * clustering statistics.
             */
            self.cluster_weights[best_cluster] += n.weight(j);
            self.nodes_per_cluster[best_cluster] += 1;

            if best_cluster == self.unused_clusters[num_unused_clusters - 1] {
                num_unused_clusters -= 1;
            }

            /*
             * Mark the currently selected node as stable and remove it from
             * the queue.
             */
            num_unstable_nodes -= 1;

            /*
             * If the new cluster of the currently selected node is different
             * from the old cluster, some further updating of the clustering
             * statistics is performed. Also, the neighbors of the currently
             * selected node that do not belong to the new cluster are marked
             * as unstable and are added to the queue.
             */
            if best_cluster != current_cluster {
                c.set(j, best_cluster);

                // disabling this gives us parity with upstream louvain,
                // num_unstable_nodes = n.nodes();
                update = true;
            }

            i = (i + 1) % n.nodes();

            if num_unstable_nodes == 0 {
                break;
            }
        }

        if update {
            c.remove_empty_clusters();
        }

        update
    }
}
