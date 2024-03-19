use crate::standard_local_moving::StandardLocalMoving;
use crate::{Clustering, Graph, Network};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::HashSet;

/// Perform the Louvain clustering algorithm
pub struct Louvain {
    rng: ChaCha20Rng,
    local_moving: StandardLocalMoving,
}

/// Default resolution for Louvain
pub const DEFAULT_RESOLUTION: f64 = 1.0;

impl Louvain {
    /// Initialize the Leiden algorithm with the given resolution and randomness parameters.
    /// An optional random seed can be supplied, otherwise a seed of 0 will be used.
    pub fn new(resolution: f64, seed: Option<usize>) -> Louvain {
        let seed = seed.unwrap_or_default() as u64;

        Louvain {
            rng: ChaCha20Rng::seed_from_u64(seed),
            local_moving: StandardLocalMoving::new(resolution),
        }
    }

    /// Iterate the Louvain algorithm a single level
    pub fn iterate_one_level<C: Clustering>(&mut self, n: &Network, c: &mut C) -> bool {
        self.local_moving.iterate(n, c, &mut self.rng)
    }

    /// Iterate the Louvain algorithm one step. Returns true if cluster labels were updated, otherwise returns false.
    pub fn iterate<C: Clustering>(&mut self, n: &Network, c: &mut C) -> bool {
        // Update the clustering by moving individual nodes between clusters.
        let mut update = self.local_moving.iterate(n, c, &mut self.rng);

        if c.num_clusters() == n.nodes() {
            return update;
        }

        // Create an aggregate network based on the refined clustering of
        // the non-aggregate network.
        let reduced_n = n.create_reduced_network(c);

        // Create one-cluster-per-node clustering
        let mut reduced_clusters = C::init_different_clusters(reduced_n.nodes());

        update |= self.iterate(&reduced_n, &mut reduced_clusters);

        c.merge_clusters(&reduced_clusters);

        update
    }

    /// Build a Louvain-compatible network from a list of adjacencies
    pub fn build_network<I: Iterator<Item = (u32, u32)>>(n_nodes: usize, n_edges: usize, adjacency: I) -> Network {
        let mut graph = Graph::with_capacity(n_nodes, n_edges);
        let mut node_indices = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            node_indices.push(graph.add_node(1.0));
        }
        let mut seen = vec![HashSet::<u32>::new(); n_nodes];
        let mut node_weights = vec![0.0; n_nodes];
        for (i, j) in adjacency {
            let (i, j) = if i < j { (i, j) } else { (j, i) };
            let i_ = i as usize;
            let j_ = j as usize;
            if seen[i_].insert(j) {
                graph.add_edge(i.into(), j.into(), 1.0);
                // weights are just degree here
                node_weights[i_] += 1.0;
                node_weights[j_] += 1.0;
            }
        }
        // reweight nodes based on # of edges
        for &i in &node_indices {
            *graph.node_weight_mut(i).unwrap() = node_weights[i.index() as usize];
        }
        Network::new_from_graph(graph)
    }
}
